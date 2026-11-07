//! Python boundary for native terms and their store.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::rule::map_rule_error;
use lofi_chart::program::{DrawError, Scope};
use lofi_chart::term::{
    AtomId, BindingSite, Identity, OperatorId, OperatorSpec, PayloadKind, SignatureSpec,
    Term as CoreTerm, TermError as CoreError, TermStore as CoreStore,
};
use num_bigint::BigUint;
use pyo3::basic::CompareOp;
use pyo3::exceptions::{PyException, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyInt, PyTuple, PyWeakrefReference};

const ROOT_MAX_NODES: usize = 1_000_000;

fn operator_index(identifier: OperatorId) -> usize {
    usize::try_from(identifier).unwrap_or(usize::MAX)
}

/// The base exception raised by the term layer.
#[pyclass(extends=PyException, subclass, module = "lofi._native", name = "TermError")]
#[derive(Debug)]
pub struct PyTermError;

#[pymethods]
impl PyTermError {
    #[new]
    #[pyo3(signature = (*args, **kwargs))]
    const fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> Self {
        let _ = (args, kwargs);
        Self
    }
}

macro_rules! integer_triplet_error {
    ($rust:ident, $python:literal, $first:ident, $message:expr) => {
        #[doc = concat!("Native ", $python, ".")]
        #[pyclass(extends=PyTermError, module = "lofi._native", name = $python)]
        #[derive(Debug)]
        pub struct $rust {
            #[pyo3(get)]
            $first: String,
            #[pyo3(get)]
            expected: usize,
            #[pyo3(get)]
            received: usize,
        }

        #[pymethods]
        impl $rust {
            #[new]
            #[pyo3(signature = (*, $first, expected, received))]
            fn new($first: String, expected: usize, received: usize) -> PyClassInitializer<Self> {
                PyClassInitializer::from(PyTermError).add_subclass(Self {
                    $first,
                    expected,
                    received,
                })
            }

            #[pyo3(signature = (*, $first, expected, received))]
            fn __init__(&mut self, $first: String, expected: usize, received: usize) {
                self.$first = $first;
                self.expected = expected;
                self.received = received;
            }

            fn __str__(&self) -> String {
                ($message)(self)
            }
        }
    };
}

integer_triplet_error!(
    PyArityError,
    "ArityError",
    operator,
    |held: &PyArityError| format!(
        "{:?} takes {} arguments, and was given {}",
        held.operator, held.expected, held.received
    )
);

integer_triplet_error!(
    PyBindingAtomCountError,
    "BindingAtomCountError",
    operator,
    |held: &PyBindingAtomCountError| format!(
        "operator {:?} binds {} variables, and was given {} atoms",
        held.operator, held.expected, held.received
    )
);

/// An argument from outside a store lineage.
#[pyclass(extends=PyTermError, module = "lofi._native", name = "ForeignTermError")]
#[derive(Debug)]
pub struct PyForeignTermError {
    #[pyo3(get)]
    operator: String,
    #[pyo3(get)]
    position: usize,
    #[pyo3(get)]
    expected: u32,
    #[pyo3(get)]
    received: u32,
}

#[pymethods]
impl PyForeignTermError {
    #[new]
    #[pyo3(signature = (*, operator, position, expected, received))]
    fn new(
        operator: String,
        position: usize,
        expected: u32,
        received: u32,
    ) -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyTermError).add_subclass(Self {
            operator,
            position,
            expected,
            received,
        })
    }

    #[pyo3(signature = (*, operator, position, expected, received))]
    fn __init__(&mut self, operator: String, position: usize, expected: u32, received: u32) {
        self.operator = operator;
        self.position = position;
        self.expected = expected;
        self.received = received;
    }

    fn __str__(&self) -> String {
        format!(
            "argument {} of {:?} was interned in generation {}, which is not generation {} or an ancestor of it",
            self.position, self.operator, self.received, self.expected
        )
    }
}

/// An argument with an incompatible sort.
#[pyclass(extends=PyTermError, module = "lofi._native", name = "SortError")]
#[derive(Debug)]
pub struct PySortError {
    #[pyo3(get)]
    operator: String,
    #[pyo3(get)]
    position: usize,
    #[pyo3(get)]
    expected: String,
    #[pyo3(get)]
    received: String,
}

#[pymethods]
impl PySortError {
    #[new]
    #[pyo3(signature = (*, operator, position, expected, received))]
    fn new(
        operator: String,
        position: usize,
        expected: String,
        received: String,
    ) -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyTermError).add_subclass(Self {
            operator,
            position,
            expected,
            received,
        })
    }

    #[pyo3(signature = (*, operator, position, expected, received))]
    fn __init__(&mut self, operator: String, position: usize, expected: String, received: String) {
        self.operator = operator;
        self.position = position;
        self.expected = expected;
        self.received = received;
    }

    fn __str__(&self) -> String {
        format!(
            "argument {} of {:?} must have sort {:?}, and has sort {:?}",
            self.position, self.operator, self.expected, self.received
        )
    }
}

/// An undeclared operator.
#[pyclass(extends=PyTermError, module = "lofi._native", name = "UnknownOperatorError")]
#[derive(Debug)]
pub struct PyUnknownOperatorError {
    #[pyo3(get)]
    operator: Py<PyAny>,
    #[pyo3(get)]
    signature: String,
}

#[pymethods]
impl PyUnknownOperatorError {
    #[new]
    #[pyo3(signature = (*, operator, signature))]
    fn new(operator: Py<PyAny>, signature: String) -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyTermError).add_subclass(Self {
            operator,
            signature,
        })
    }

    #[pyo3(signature = (*, operator, signature))]
    fn __init__(&mut self, operator: Py<PyAny>, signature: String) {
        self.operator = operator;
        self.signature = signature;
    }

    fn __str__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "signature {:?} declares no operator {}",
            self.signature,
            self.operator.bind(py).repr()?
        ))
    }
}

/// An invalid term payload.
#[pyclass(extends=PyTermError, module = "lofi._native", name = "PayloadError")]
#[derive(Debug)]
pub struct PyPayloadError {
    #[pyo3(get)]
    operator: String,
    #[pyo3(get)]
    payload: Py<PyAny>,
    #[pyo3(get)]
    reason: String,
}

#[pymethods]
impl PyPayloadError {
    #[new]
    #[pyo3(signature = (*, operator, payload, reason))]
    fn new(operator: String, payload: Py<PyAny>, reason: String) -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyTermError).add_subclass(Self {
            operator,
            payload,
            reason,
        })
    }

    #[pyo3(signature = (*, operator, payload, reason))]
    fn __init__(&mut self, operator: String, payload: Py<PyAny>, reason: String) {
        self.operator = operator;
        self.payload = payload;
        self.reason = reason;
    }

    fn __str__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "operator {:?} refuses the payload {}: {}",
            self.operator,
            self.payload.bind(py).repr()?,
            self.reason
        ))
    }
}

/// A region that exhausted its node cap.
#[pyclass(extends=PyTermError, module = "lofi._native", name = "StoreFullError")]
#[derive(Debug)]
pub struct PyStoreFullError {
    #[pyo3(get)]
    region: String,
    #[pyo3(get)]
    capacity: usize,
    #[pyo3(get)]
    size: usize,
}

#[pymethods]
impl PyStoreFullError {
    #[new]
    #[pyo3(signature = (*, region, capacity, size))]
    fn new(region: String, capacity: usize, size: usize) -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyTermError).add_subclass(Self {
            region,
            capacity,
            size,
        })
    }

    #[pyo3(signature = (*, region, capacity, size))]
    fn __init__(&mut self, region: String, capacity: usize, size: usize) {
        self.region = region;
        self.capacity = capacity;
        self.size = size;
    }

    fn __str__(&self) -> String {
        format!(
            "region {:?} would hold {} nodes and may hold at most {}",
            self.region, self.size, self.capacity
        )
    }
}

/// An operation incompatible with a region's state.
#[pyclass(extends=PyTermError, module = "lofi._native", name = "RegionStateError")]
#[derive(Debug)]
pub struct PyRegionStateError {
    #[pyo3(get)]
    region: String,
    #[pyo3(get)]
    operation: String,
    #[pyo3(get)]
    state: String,
}

#[pymethods]
impl PyRegionStateError {
    #[new]
    #[pyo3(signature = (*, region, operation, state))]
    fn new(region: String, operation: String, state: String) -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyTermError).add_subclass(Self {
            region,
            operation,
            state,
        })
    }

    #[pyo3(signature = (*, region, operation, state))]
    fn __init__(&mut self, region: String, operation: String, state: String) {
        self.region = region;
        self.operation = operation;
        self.state = state;
    }

    fn __str__(&self) -> String {
        format!(
            "region {:?} cannot {} while {}",
            self.region, self.operation, self.state
        )
    }
}

/// A replacement with open indices that would be captured.
#[pyclass(extends=PyTermError, module = "lofi._native", name = "NotLocallyClosedError")]
#[derive(Debug)]
pub struct PyNotLocallyClosedError {
    #[pyo3(get)]
    max_index: u32,
    #[pyo3(get)]
    depth: u32,
}

#[pymethods]
impl PyNotLocallyClosedError {
    #[new]
    #[pyo3(signature = (*, max_index, depth))]
    fn new(max_index: u32, depth: u32) -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyTermError).add_subclass(Self { max_index, depth })
    }

    #[pyo3(signature = (*, max_index, depth))]
    const fn __init__(&mut self, max_index: u32, depth: u32) {
        self.max_index = max_index;
        self.depth = depth;
    }

    fn __str__(&self) -> String {
        format!(
            "a term needing {} enclosing binders is not closed under binder depth {}, so substituting it would capture",
            self.max_index, self.depth
        )
    }
}

/// A strict signature's refusal of a vacuous binder.
#[pyclass(extends=PyTermError, module = "lofi._native", name = "VacuousBindingError")]
#[derive(Debug)]
pub struct PyVacuousBindingError {
    #[pyo3(get)]
    operator: String,
    #[pyo3(get)]
    variable: String,
    #[pyo3(get)]
    signature: String,
}

#[pymethods]
impl PyVacuousBindingError {
    #[new]
    #[pyo3(signature = (*, operator, variable, signature))]
    fn new(operator: String, variable: String, signature: String) -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyTermError).add_subclass(Self {
            operator,
            variable,
            signature,
        })
    }

    #[pyo3(signature = (*, operator, variable, signature))]
    fn __init__(&mut self, operator: String, variable: String, signature: String) {
        self.operator = operator;
        self.variable = variable;
        self.signature = signature;
    }

    fn __str__(&self) -> String {
        format!(
            "binder {:?} never uses its variable {:?}, and signature {:?} rejects vacuous binding",
            self.operator, self.variable, self.signature
        )
    }
}

macro_rules! named_fields_error {
    ($rust:ident, $python:literal, ($($field:ident),+), $message:expr) => {
        #[doc = concat!("Native ", $python, ".")]
        #[pyclass(extends=PyTermError, module = "lofi._native", name = $python)]
        #[derive(Debug)]
        pub struct $rust {
            $(#[pyo3(get)] $field: String),+
        }

        #[pymethods]
        impl $rust {
            #[new]
            #[pyo3(signature = (*, $($field),+))]
            fn new($($field: String),+) -> PyClassInitializer<Self> {
                PyClassInitializer::from(PyTermError)
                    .add_subclass(Self { $($field),+ })
            }

            #[pyo3(signature = (*, $($field),+))]
            fn __init__(&mut self, $($field: String),+) {
                $(self.$field = $field;)+
            }

            fn __str__(&self) -> String {
                ($message)(self)
            }
        }
    };
}

named_fields_error!(
    PyBindingShapeError,
    "BindingShapeError",
    (operator, reason),
    |held: &PyBindingShapeError| format!(
        "a binding site of operator {:?} {}",
        held.operator, held.reason
    )
);

named_fields_error!(
    PyUnknownConstantError,
    "UnknownConstantError",
    (sort, name),
    |held: &PyUnknownConstantError| format!(
        "the vocabulary of sort {:?} has no constant {:?}",
        held.sort, held.name
    )
);

named_fields_error!(
    PyDuplicateNameError,
    "DuplicateNameError",
    (name, kind, signature),
    |held: &PyDuplicateNameError| format!(
        "signature {:?} declares the {} {:?} twice",
        held.signature, held.kind, held.name
    )
);

named_fields_error!(
    PyUndeclaredSortError,
    "UndeclaredSortError",
    (sort, operator, signature),
    |held: &PyUndeclaredSortError| format!(
        "operator {:?} of signature {:?} names the sort {:?}, which the signature does not declare",
        held.operator, held.signature, held.sort
    )
);

named_fields_error!(
    PyReservedDeclarationError,
    "ReservedDeclarationError",
    (operator, reason, signature),
    |held: &PyReservedDeclarationError| format!(
        "operator {:?} of signature {:?} {}, which only the kernel derives",
        held.operator, held.signature, held.reason
    )
);

named_fields_error!(
    PyErasureError,
    "ErasureError",
    (operator, target, requirement, signature),
    |held: &PyErasureError| format!(
        "operator {:?} of signature {:?} erases to {:?}, and that erasure is not {}",
        held.operator, held.signature, held.target, held.requirement
    )
);

/// An encoding wider than its declared layout.
#[pyclass(extends=PyTermError, module = "lofi._native", name = "EncodingOverflowError")]
#[derive(Debug)]
pub struct PyEncodingOverflowError {
    #[pyo3(get)]
    layout: String,
    #[pyo3(get)]
    capacity: usize,
    #[pyo3(get)]
    size: usize,
}

#[pymethods]
impl PyEncodingOverflowError {
    #[new]
    #[pyo3(signature = (*, layout, capacity, size))]
    fn new(layout: String, capacity: usize, size: usize) -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyTermError).add_subclass(Self {
            layout,
            capacity,
            size,
        })
    }

    #[pyo3(signature = (*, layout, capacity, size))]
    fn __init__(&mut self, layout: String, capacity: usize, size: usize) {
        self.layout = layout;
        self.capacity = capacity;
        self.size = size;
    }

    fn __str__(&self) -> String {
        format!(
            "layout {:?} holds {} slots, and the term needs {}",
            self.layout, self.capacity, self.size
        )
    }
}

/// A malformed term encoding.
#[pyclass(extends=PyTermError, module = "lofi._native", name = "EncodingShapeError")]
#[derive(Debug)]
pub struct PyEncodingShapeError {
    #[pyo3(get)]
    layout: String,
    #[pyo3(get)]
    reason: String,
    #[pyo3(get)]
    slot: usize,
}

#[pymethods]
impl PyEncodingShapeError {
    #[new]
    #[pyo3(signature = (*, layout, reason, slot))]
    fn new(layout: String, reason: String, slot: usize) -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyTermError).add_subclass(Self {
            layout,
            reason,
            slot,
        })
    }

    #[pyo3(signature = (*, layout, reason, slot))]
    fn __init__(&mut self, layout: String, reason: String, slot: usize) {
        self.layout = layout;
        self.reason = reason;
        self.slot = slot;
    }

    fn __str__(&self) -> String {
        let detail = if self.reason == "exhausted" {
            "ran out of symbols with an argument still owed"
        } else {
            "read a complete term and found another symbol"
        };
        format!("layout {:?} {detail} at slot {}", self.layout, self.slot)
    }
}

/// A rewrite that exhausted its step budget.
#[pyclass(extends=PyTermError, module = "lofi._native", name = "NonTerminatingError")]
#[derive(Debug)]
pub struct PyNonTerminatingError {
    #[pyo3(get)]
    operation: String,
    #[pyo3(get)]
    budget: usize,
}

#[pymethods]
impl PyNonTerminatingError {
    #[new]
    #[pyo3(signature = (*, operation, budget))]
    fn new(operation: String, budget: usize) -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyTermError).add_subclass(Self { operation, budget })
    }

    #[pyo3(signature = (*, operation, budget))]
    fn __init__(&mut self, operation: String, budget: usize) {
        self.operation = operation;
        self.budget = budget;
    }

    fn __str__(&self) -> String {
        format!(
            "{} did not terminate within {} steps",
            self.operation, self.budget
        )
    }
}

macro_rules! initialized_error {
    ($py:expr, $initializer:expr) => {
        match Py::new($py, $initializer) {
            Ok(value) => PyErr::from_value(value.bind($py).as_any().clone()),
            Err(error) => error,
        }
    };
}

fn sort_error(
    py: Python<'_>,
    registry: &Registry,
    operator: String,
    position: usize,
    expected: u16,
    received: u16,
) -> PyErr {
    initialized_error!(
        py,
        PyClassInitializer::from(PyTermError).add_subclass(PySortError {
            operator,
            position,
            expected: registry.sorts[usize::from(expected)].clone(),
            received: registry.sorts[usize::from(received)].clone(),
        })
    )
}

fn core_payload_error(
    py: Python<'_>,
    operator: String,
    payload: Option<u64>,
    reason: String,
) -> PyErr {
    let payload = payload.map_or_else(
        || py.None(),
        |value| {
            value
                .into_pyobject(py)
                .expect("integer conversion")
                .unbind()
                .into_any()
        },
    );
    initialized_error!(
        py,
        PyClassInitializer::from(PyTermError).add_subclass(PyPayloadError {
            operator,
            payload,
            reason,
        })
    )
}

fn atom_error(index: AtomId, sort: Option<u16>, registry: &Registry) -> PyErr {
    PyErr::new::<PyValueError, _>(sort.map_or_else(
        || format!("no atom has index {index}"),
        |sort| {
            format!(
                "atom {index} is not a free variable of sort {:?}",
                registry.sorts[usize::from(sort)]
            )
        },
    ))
}

fn missing_derived_error(py: Python<'_>, registry: &Registry, kind: &str, sort: u16) -> PyErr {
    initialized_error!(
        py,
        PyClassInitializer::from(PyTermError).add_subclass(PyUnknownOperatorError {
            operator: format!("#{kind}[{}]", registry.sorts[usize::from(sort)])
                .into_pyobject(py)
                .expect("string conversion")
                .unbind()
                .into_any(),
            signature: registry.signature_name.clone(),
        })
    )
}

fn unknown_core_operator_error(py: Python<'_>, registry: &Registry, operator: OperatorId) -> PyErr {
    initialized_error!(
        py,
        PyClassInitializer::from(PyTermError).add_subclass(PyUnknownOperatorError {
            operator: u64::from(operator)
                .into_pyobject(py)
                .expect("integer conversion")
                .unbind()
                .into_any(),
            signature: registry.signature_name.clone(),
        })
    )
}

pub(crate) fn error(core: CoreError, registry: &Registry, py: Python<'_>) -> PyErr {
    match core {
        CoreError::UnknownOperator { operator } => {
            unknown_core_operator_error(py, registry, operator)
        }
        CoreError::Arity {
            operator,
            expected,
            received,
        } => initialized_error!(
            py,
            PyClassInitializer::from(PyTermError).add_subclass(PyArityError {
                operator,
                expected,
                received,
            })
        ),
        CoreError::Sort {
            operator,
            position,
            expected,
            received,
        } => sort_error(py, registry, operator, position, expected, received),
        CoreError::Payload {
            operator,
            payload,
            reason,
        } => core_payload_error(py, operator, payload, reason),
        CoreError::Foreign {
            operator,
            position,
            expected,
            received,
        } => initialized_error!(
            py,
            PyClassInitializer::from(PyTermError).add_subclass(PyForeignTermError {
                operator,
                position,
                expected,
                received,
            })
        ),
        CoreError::Full {
            region,
            capacity,
            size,
        } => initialized_error!(
            py,
            PyClassInitializer::from(PyTermError).add_subclass(PyStoreFullError {
                region,
                capacity,
                size,
            })
        ),
        CoreError::Region {
            region,
            operation,
            state,
        } => initialized_error!(
            py,
            PyClassInitializer::from(PyTermError).add_subclass(PyRegionStateError {
                region,
                operation,
                state,
            })
        ),
        CoreError::Classification { reason } => PyValueError::new_err(reason),
        CoreError::Atom { index, sort } => atom_error(index, sort, registry),
        CoreError::MissingDerived { kind, sort } => missing_derived_error(py, registry, kind, sort),
        CoreError::NotLocallyClosed { max_index, depth } => initialized_error!(
            py,
            PyClassInitializer::from(PyTermError)
                .add_subclass(PyNotLocallyClosedError { max_index, depth })
        ),
        CoreError::BindingAtoms {
            operator,
            expected,
            received,
        } => initialized_error!(
            py,
            PyClassInitializer::from(PyTermError).add_subclass(PyBindingAtomCountError {
                operator,
                expected,
                received,
            })
        ),
        CoreError::VacuousBinding { operator, variable } => initialized_error!(
            py,
            PyClassInitializer::from(PyTermError).add_subclass(PyVacuousBindingError {
                operator,
                variable,
                signature: registry.signature_name.clone(),
            })
        ),
    }
}

/// Map a refusal at the scoped drawing boundary to its Python exception.
pub(crate) fn draw_error(error: DrawError<Registry>) -> PyErr {
    match error {
        DrawError::Invalid(message) => PyValueError::new_err(message),
        DrawError::Internal(message) => PyRuntimeError::new_err(message),
        DrawError::Chart(error) => PyValueError::new_err(error.to_string()),
        DrawError::Rule { error, scope } => Python::attach(|py| map_rule_error(error, &scope, py)),
        DrawError::Host(error) => match error.downcast::<PyErr>() {
            Ok(error) => *error,
            Err(error) => PyRuntimeError::new_err(error.to_string()),
        },
    }
}

#[derive(Debug)]
pub(crate) struct Registry {
    signature: Py<PyAny>,
    signature_name: String,
    sorts: Vec<String>,
    sort_ids: HashMap<String, u16>,
    operators: Vec<Py<PyAny>>,
    operator_ids: HashMap<String, OperatorId>,
    payload_kinds: Vec<PayloadKind>,
    vocabularies: Vec<Option<Vec<String>>>,
    stores: Mutex<HashMap<u32, CoreStore>>,
    terms: Mutex<HashMap<u64, Py<PyWeakrefReference>>>,
}

impl Scope for Registry {
    fn store_for(&self, term: &CoreTerm) -> Option<CoreStore> {
        Self::store_for(self, term)
    }
}

impl Registry {
    pub(crate) fn store_for(&self, term: &CoreTerm) -> Option<CoreStore> {
        self.stores
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&term.generation())
            .cloned()
    }

    pub(crate) fn operator_id(&self, name: &str) -> Option<OperatorId> {
        self.operator_ids.get(name).copied()
    }

    pub(crate) fn sort_id(&self, name: &str) -> Option<u16> {
        self.sort_ids.get(name).copied()
    }

    fn prune_dead_terms(&self, py: Python<'_>) -> PyResult<()> {
        let mut terms = self
            .terms
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut dead = Vec::new();
        for (serial, weak) in terms.iter() {
            if weak.bind(py).upgrade_as::<PyTerm>()?.is_none() {
                dead.push(*serial);
            }
        }
        for serial in dead {
            terms.remove(&serial);
        }
        drop(terms);
        Ok(())
    }

    fn display(&self, term: &CoreTerm) -> String {
        let operator = operator_index(term.operator());
        let head = match (self.payload_kinds[operator], term.identity()) {
            (PayloadKind::Constant, Identity::Integer(identifier)) => self.vocabularies
                [usize::from(term.sort())]
            .as_ref()
            .and_then(|names| {
                usize::try_from(*identifier)
                    .ok()
                    .and_then(|index| names.get(index))
            })
            .cloned()
            .unwrap_or_else(|| format!("{}({identifier})", term.operator_name())),
            (_, Identity::None) => term.operator_name().to_owned(),
            (_, Identity::Integer(value)) => format!("{}({value})", term.operator_name()),
            (_, Identity::Name(value)) => format!("{}({value})", term.operator_name()),
        };
        if term.arguments().is_empty() {
            return head;
        }
        let written = term
            .arguments()
            .iter()
            .map(|argument| self.display(argument))
            .collect::<Vec<_>>()
            .join(" ");
        format!("({head} {written})")
    }
}

// Registry identity must be shared by every wrapper. This helper cannot use
// `Registry::wrap` until a registry is already in an Arc.
pub(crate) fn wrap(
    registry: &Arc<Registry>,
    py: Python<'_>,
    term: CoreTerm,
) -> PyResult<Py<PyTerm>> {
    let serial = term.serial();
    if let Some(known) = registry
        .terms
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(&serial)
        && let Some(term) = known.bind(py).upgrade_as::<PyTerm>()?
    {
        return Ok(term.clone().unbind());
    }
    let wrapped = Py::new(
        py,
        PyTerm {
            term,
            registry: registry.clone(),
            arguments: OnceLock::new(),
        },
    )?;
    let weak = PyWeakrefReference::new(wrapped.bind(py).as_any())?.unbind();
    registry
        .terms
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert(serial, weak);
    Ok(wrapped)
}

/// One immutable interned term.
#[pyclass(frozen, weakref, module = "lofi._native", name = "Term")]
#[derive(Debug)]
pub struct PyTerm {
    term: CoreTerm,
    registry: Arc<Registry>,
    arguments: OnceLock<Py<PyTuple>>,
}

impl PyTerm {
    pub(crate) fn core_term(&self) -> CoreTerm {
        self.term.clone()
    }

    pub(crate) fn registry(&self) -> Arc<Registry> {
        self.registry.clone()
    }
}

#[pymethods]
impl PyTerm {
    #[new]
    fn new() -> PyResult<Self> {
        Err(PyTypeError::new_err(
            "Term is interned by TermStore.op and made nowhere else",
        ))
    }

    #[classattr]
    const GENERATION_SHIFT: u32 = lofi_chart::term::GENERATION_SHIFT;

    #[classattr]
    const SHALLOW: usize = 256;

    #[getter]
    fn op(&self, py: Python<'_>) -> Py<PyAny> {
        self.registry.operators[operator_index(self.term.operator())].clone_ref(py)
    }

    #[getter]
    fn args(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        if let Some(arguments) = self.arguments.get() {
            return Ok(arguments.clone_ref(py));
        }
        let arguments = self
            .term
            .arguments()
            .iter()
            .cloned()
            .map(|term| wrap(&self.registry, py, term))
            .collect::<PyResult<Vec<_>>>()?;
        let arguments = PyTuple::new(py, arguments)?.unbind();
        if let Err(discarded) = self.arguments.set(arguments) {
            drop(discarded);
        }
        self.arguments
            .get()
            .map(|arguments| arguments.clone_ref(py))
            .ok_or_else(|| PyRuntimeError::new_err("term arguments were not cached"))
    }

    #[getter]
    fn payload(&self) -> Option<u64> {
        self.term.payload()
    }

    #[getter]
    fn identity(&self) -> Option<Py<PyAny>> {
        Python::attach(|py| match self.term.identity() {
            Identity::None => None,
            Identity::Integer(value) => Some(
                value
                    .into_pyobject(py)
                    .expect("integer conversion")
                    .unbind()
                    .into_any(),
            ),
            Identity::Name(value) => Some(
                value
                    .into_pyobject(py)
                    .expect("string conversion")
                    .unbind()
                    .into_any(),
            ),
        })
    }

    #[getter]
    fn core(&self, py: Python<'_>) -> PyResult<Py<Self>> {
        wrap(&self.registry, py, self.term.core())
    }

    #[getter]
    fn size(&self) -> usize {
        self.term.size()
    }

    #[getter]
    fn maxidx(&self) -> u32 {
        self.term.max_index()
    }

    #[getter]
    fn unbound(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let values = self
            .term
            .unbound()
            .iter()
            .map(|(sort, bound)| (self.registry.sorts[usize::from(*sort)].clone(), *bound))
            .collect::<Vec<_>>();
        Ok(PyTuple::new(py, values)?.unbind())
    }

    fn maxidx_of(&self, sort: &str) -> u32 {
        self.registry
            .sort_ids
            .get(sort)
            .map_or(0, |held| self.term.max_index_of(*held))
    }

    #[getter]
    fn free(&self) -> BigUint {
        self.term.free().clone()
    }

    #[getter]
    fn serial(&self) -> u64 {
        self.term.serial()
    }

    #[getter]
    fn is_closed(&self) -> bool {
        self.term.is_closed()
    }

    #[getter]
    fn generation(&self) -> u32 {
        self.term.generation()
    }

    fn key(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let mut built: HashMap<u64, Py<PyAny>> = HashMap::new();
        let mut pending = vec![(self.term.clone(), false)];
        while let Some((term, expanded)) = pending.pop() {
            if built.contains_key(&term.serial()) {
                continue;
            }
            if expanded {
                let identity: Py<PyAny> = match term.identity() {
                    Identity::None => py.None(),
                    Identity::Integer(value) => {
                        value.into_pyobject(py)?.clone().unbind().into_any()
                    }
                    Identity::Name(value) => value.into_pyobject(py)?.clone().unbind().into_any(),
                };
                let arguments = term
                    .arguments()
                    .iter()
                    .map(|argument| {
                        built
                            .get(&argument.serial())
                            .ok_or_else(|| PyRuntimeError::new_err("child key was not built"))
                            .map(|key| key.clone_ref(py))
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                let operator = term
                    .operator_name()
                    .into_pyobject(py)?
                    .clone()
                    .unbind()
                    .into_any();
                let argument_tuple = PyTuple::new(py, arguments)?.unbind().into_any();
                let key = PyTuple::new(py, vec![operator, identity, argument_tuple])?
                    .unbind()
                    .into_any();
                built.insert(term.serial(), key);
            } else {
                pending.push((term.clone(), true));
                pending.extend(
                    term.arguments()
                        .iter()
                        .rev()
                        .filter(|argument| !built.contains_key(&argument.serial()))
                        .cloned()
                        .map(|argument| (argument, false)),
                );
            }
        }
        built
            .remove(&self.term.serial())
            .ok_or_else(|| PyRuntimeError::new_err("root key was not built"))
    }

    fn __hash__(&self) -> isize {
        let mask = u64::try_from(isize::MAX).unwrap_or(u64::MAX);
        isize::try_from(self.term.structural_hash() & mask).unwrap_or(isize::MAX)
    }

    fn __richcmp__(
        &self,
        other: &Bound<'_, PyAny>,
        op: CompareOp,
        py: Python<'_>,
    ) -> PyResult<Py<PyAny>> {
        if let Ok(other) = other.extract::<PyRef<'_, Self>>() {
            let equal = self.term == other.term;
            return Ok(match op {
                CompareOp::Eq => equal,
                CompareOp::Ne => !equal,
                _ => return Ok(py.NotImplemented()),
            }
            .into_pyobject(py)?
            .to_owned()
            .unbind()
            .into_any());
        }
        Ok(py.NotImplemented())
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let head = match self.term.identity() {
            Identity::None => self.term.operator_name().to_owned(),
            Identity::Integer(value) => format!("{}({value})", self.term.operator_name()),
            Identity::Name(value) => format!("{}({value})", self.term.operator_name()),
        };
        if self.term.arguments().is_empty() {
            return Ok(head);
        }
        let written = self
            .term
            .arguments()
            .iter()
            .cloned()
            .map(|term| {
                wrap(&self.registry, py, term)?
                    .bind(py)
                    .repr()?
                    .extract::<String>()
            })
            .collect::<PyResult<Vec<_>>>()?
            .join(" ");
        Ok(format!("({head} {written})"))
    }

    fn __str__(&self) -> String {
        self.registry.display(&self.term)
    }
}

fn payload_kind(value: &str) -> PyResult<PayloadKind> {
    match value {
        "none" => Ok(PayloadKind::None),
        "constant" => Ok(PayloadKind::Constant),
        "index" => Ok(PayloadKind::Index),
        "free" => Ok(PayloadKind::Free),
        _ => Err(PyValueError::new_err(format!(
            "unknown payload sort {value:?}"
        ))),
    }
}

fn lower_operator(
    signature: &Bound<'_, PyAny>,
    operator: &Bound<'_, PyAny>,
    sort_ids: &HashMap<String, u16>,
    operator_ids: &HashMap<String, OperatorId>,
) -> PyResult<(OperatorSpec, Option<Vec<String>>)> {
    let op_name: String = operator.getattr("name")?.extract()?;
    let sort_name: String = operator.getattr("sort")?.extract()?;
    let sort = sort_ids[&sort_name];
    let argument_names: Vec<String> = operator.getattr("arg_sorts")?.extract()?;
    let arguments: Vec<_> = argument_names.iter().map(|held| sort_ids[held]).collect();
    let payload_name: String = operator.getattr("payload_sort")?.extract()?;
    let payload = payload_kind(&payload_name)?;
    let erased_name: String = signature
        .call_method1("erase", (operator,))?
        .getattr("name")?
        .extract()?;
    let erased = operator_ids[&erased_name];
    let mut binding = Vec::new();
    for site in operator.getattr("binding")?.try_iter()? {
        let site = site?;
        let site_sort: String = site.getattr("sort")?.extract()?;
        binding.push(BindingSite {
            arguments: site.getattr("arguments")?.extract()?,
            sort: sort_ids[&site_sort],
        });
    }
    let mut shifts = Vec::new();
    for position in 0..arguments.len() {
        let mut row = Vec::new();
        for (held_sort, sort_id) in sort_ids {
            let amount: u32 = operator
                .call_method1("shift", (position, held_sort))?
                .extract()?;
            if amount != 0 {
                row.push((*sort_id, amount));
            }
        }
        shifts.push(row);
    }
    let vocabulary = if payload == PayloadKind::Constant {
        let vocabulary = signature.call_method1("vocabulary", (&sort_name,))?;
        Some(vocabulary.getattr("names")?.extract()?)
    } else {
        None
    };
    Ok((
        OperatorSpec {
            name: op_name,
            sort,
            arguments,
            payload,
            erased,
            binding,
            shifts,
        },
        vocabulary,
    ))
}

fn lower_constant_cores(
    signature: &Bound<'_, PyAny>,
    sorts: &[String],
    sort_ids: &HashMap<String, u16>,
    vocabularies: &[Option<Vec<String>>],
) -> PyResult<Vec<Option<Vec<usize>>>> {
    let declared = signature
        .getattr("constant_cores")?
        .call_method0("items")?
        .try_iter()?
        .map(|item| {
            let item = item?;
            let sort = item.get_item(0)?.extract::<String>()?;
            let row = item
                .get_item(1)?
                .call_method0("items")?
                .try_iter()?
                .map(|entry| {
                    let entry = entry?;
                    Ok((
                        entry.get_item(0)?.extract::<String>()?,
                        entry.get_item(1)?.extract::<String>()?,
                    ))
                })
                .collect::<PyResult<HashMap<_, _>>>()?;
            Ok((sort, row))
        })
        .collect::<PyResult<HashMap<_, _>>>()?;
    for sort in declared.keys() {
        if !sort_ids.contains_key(sort) {
            return Err(PyValueError::new_err(format!(
                "constant cores name unknown sort {sort:?}"
            )));
        }
    }
    sorts
        .iter()
        .zip(vocabularies)
        .map(|(sort, vocabulary)| {
            let Some(names) = vocabulary else {
                if declared.get(sort).is_some_and(|row| !row.is_empty()) {
                    return Err(PyValueError::new_err(format!(
                        "sort {sort:?} has constant cores but no vocabulary"
                    )));
                }
                return Ok(None);
            };
            let mut row = (0..names.len()).collect::<Vec<_>>();
            for (source, target) in declared.get(sort).into_iter().flatten() {
                let source_id = names
                    .iter()
                    .position(|name| name == source)
                    .ok_or_else(|| {
                        PyValueError::new_err(format!(
                            "constant core source {source:?} is outside vocabulary {sort:?}"
                        ))
                    })?;
                let target_id = names
                    .iter()
                    .position(|name| name == target)
                    .ok_or_else(|| {
                        PyValueError::new_err(format!(
                            "constant core target {target:?} is outside vocabulary {sort:?}"
                        ))
                    })?;
                if source_id == target_id {
                    return Err(PyValueError::new_err(format!(
                        "constant {source:?} cannot name itself as a distinct core"
                    )));
                }
                row[source_id] = target_id;
            }
            if let Some((source, target)) = row
                .iter()
                .copied()
                .enumerate()
                .find(|(_, target)| row[*target] != *target)
            {
                return Err(PyValueError::new_err(format!(
                    "constant core {source} targets mapped constant {target} in vocabulary {sort:?}"
                )));
            }
            Ok(Some(row))
        })
        .collect()
}

fn lower_signature(
    py: Python<'_>,
    signature: &Bound<'_, PyAny>,
) -> PyResult<(SignatureSpec, Arc<Registry>)> {
    let name: String = signature.getattr("name")?.extract()?;
    let sorts: Vec<String> = signature.getattr("sorts")?.extract()?;
    let sort_ids: HashMap<_, _> = sorts
        .iter()
        .enumerate()
        .map(|(index, sort)| {
            Ok((
                sort.clone(),
                u16::try_from(index).map_err(|_| PyValueError::new_err("too many sorts"))?,
            ))
        })
        .collect::<PyResult<_>>()?;
    let declared = signature.getattr("operators")?;
    let mut operators = Vec::new();
    for operator in declared.try_iter()? {
        operators.push(operator?.unbind());
    }
    let mut operator_ids = HashMap::new();
    for (index, operator) in operators.iter().enumerate() {
        let op_name: String = operator.bind(py).getattr("name")?.extract()?;
        operator_ids.insert(
            op_name,
            u32::try_from(index).map_err(|_| PyValueError::new_err("too many operators"))?,
        );
    }
    let mut specs = Vec::with_capacity(operators.len());
    let mut vocabularies = vec![None; sorts.len()];
    let mut constants = vec![None; sorts.len()];
    let mut bound = vec![None; sorts.len()];
    let mut free = vec![None; sorts.len()];
    let mut holes = vec![None; sorts.len()];
    for (index, operator) in operators.iter().enumerate() {
        let operator = operator.bind(py);
        let (spec, vocabulary) = lower_operator(signature, operator, &sort_ids, &operator_ids)?;
        let operator_id =
            u32::try_from(index).map_err(|_| PyValueError::new_err("too many operators"))?;
        let sort_index = usize::from(spec.sort);
        match spec.payload {
            PayloadKind::Constant => {
                constants[sort_index] = Some(operator_id);
                vocabularies[sort_index] = vocabulary;
            }
            PayloadKind::Index => bound[sort_index] = Some(operator_id),
            PayloadKind::Free => free[sort_index] = Some(operator_id),
            PayloadKind::None if spec.name.starts_with("#hole[") => {
                holes[sort_index] = Some(operator_id);
            }
            PayloadKind::None => {}
        }
        specs.push(spec);
    }
    let allows_vacuous_binding = signature.getattr("allows_vacuous_binding")?.extract()?;
    let constant_cores = lower_constant_cores(signature, &sorts, &sort_ids, &vocabularies)?;
    let payload_kinds = specs.iter().map(|operator| operator.payload).collect();
    let registry = Arc::new(Registry {
        signature: signature.clone().unbind(),
        signature_name: name.clone(),
        sorts: sorts.clone(),
        sort_ids,
        operators,
        operator_ids,
        payload_kinds,
        vocabularies: vocabularies.clone(),
        stores: Mutex::new(HashMap::new()),
        terms: Mutex::new(HashMap::new()),
    });
    Ok((
        SignatureSpec {
            name,
            sorts,
            operators: specs,
            vocabularies: vocabularies
                .iter()
                .map(|held| held.as_ref().map(Vec::len))
                .collect(),
            constant_cores,
            constants,
            bound,
            free,
            holes,
            allows_vacuous_binding,
        },
        registry,
    ))
}

fn operator_id(
    py: Python<'_>,
    registry: &Registry,
    operator: &Bound<'_, PyAny>,
) -> PyResult<OperatorId> {
    let name: String = operator.getattr("name")?.extract().map_err(|_| {
        initialized_error!(
            py,
            PyClassInitializer::from(PyTermError).add_subclass(PyUnknownOperatorError {
                operator: operator.clone().unbind(),
                signature: registry.signature_name.clone(),
            })
        )
    })?;
    let Some(identifier) = registry.operator_ids.get(&name).copied() else {
        return Err(initialized_error!(
            py,
            PyClassInitializer::from(PyTermError).add_subclass(PyUnknownOperatorError {
                operator: operator.clone().unbind(),
                signature: registry.signature_name.clone(),
            })
        ));
    };
    if !operator.eq(registry.operators[operator_index(identifier)].bind(py))? {
        return Err(initialized_error!(
            py,
            PyClassInitializer::from(PyTermError).add_subclass(PyUnknownOperatorError {
                operator: operator.clone().unbind(),
                signature: registry.signature_name.clone(),
            })
        ));
    }
    Ok(identifier)
}

fn nonnegative_u64(name: &str, value: &Bound<'_, PyAny>) -> PyResult<u64> {
    if value.is_instance_of::<PyBool>() || !value.is_instance_of::<PyInt>() {
        return Err(PyTypeError::new_err(format!(
            "{name} is not an integer: {}",
            value.repr()?
        )));
    }
    if value.lt(0_i8)? {
        return Err(PyValueError::new_err(format!(
            "{name} is negative: {}",
            value.repr()?
        )));
    }
    value.extract()
}

fn nonnegative_usize(name: &str, value: &Bound<'_, PyAny>) -> PyResult<usize> {
    let held = nonnegative_u64(name, value)?;
    usize::try_from(held).map_err(|_| PyValueError::new_err(format!("{name} is too large: {held}")))
}

fn positive_usize(name: &str, value: &Bound<'_, PyAny>) -> PyResult<usize> {
    let held = nonnegative_usize(name, value)?;
    if held == 0 {
        return Err(PyValueError::new_err(format!(
            "{name} is not positive: {}",
            value.repr()?
        )));
    }
    Ok(held)
}

fn max_nodes(value: &Bound<'_, PyAny>) -> PyResult<usize> {
    positive_usize("max_nodes", value)
}

fn nonnegative_u32(name: &str, value: &Bound<'_, PyAny>) -> PyResult<u32> {
    let held = nonnegative_u64(name, value)?;
    u32::try_from(held).map_err(|_| PyValueError::new_err(format!("{name} is too large: {held}")))
}

fn depth(value: &Bound<'_, PyAny>) -> PyResult<u32> {
    nonnegative_u32("depth", value)
}

fn atom_id(name: &str, value: &Bound<'_, PyAny>) -> PyResult<AtomId> {
    let held = nonnegative_u64(name, value)?;
    AtomId::try_from(held)
        .map_err(|_| PyValueError::new_err(format!("{name} is too large: {held}")))
}

fn payload_error(
    py: Python<'_>,
    operator: &str,
    payload: Option<&Bound<'_, PyAny>>,
    reason: &str,
) -> PyErr {
    initialized_error!(
        py,
        PyClassInitializer::from(PyTermError).add_subclass(PyPayloadError {
            operator: operator.to_owned(),
            payload: payload.map_or_else(|| py.None(), |held| held.clone().unbind()),
            reason: reason.to_owned(),
        })
    )
}

fn payload_value(
    py: Python<'_>,
    op: &OperatorSpec,
    payload: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<u64>> {
    if op.payload == PayloadKind::None {
        return if payload.is_none() {
            Ok(None)
        } else {
            Err(payload_error(py, &op.name, payload, "it takes no payload"))
        };
    }
    let Some(payload) = payload else {
        return Err(payload_error(
            py,
            &op.name,
            None,
            "it takes an integer payload",
        ));
    };
    if payload.is_instance_of::<PyBool>() || !payload.is_instance_of::<PyInt>() {
        return Err(payload_error(
            py,
            &op.name,
            Some(payload),
            "it takes an integer payload",
        ));
    }
    if payload.lt(0_i8)? {
        return Err(payload_error(
            py,
            &op.name,
            Some(payload),
            "a payload is never negative",
        ));
    }
    let value = payload.extract::<u64>().map_err(|_| {
        payload_error(
            py,
            &op.name,
            Some(payload),
            "the integer payload exceeds the native index range",
        )
    })?;
    Ok(Some(value))
}

/// One native term store or child region.
#[pyclass(module = "lofi._native", name = "TermStore")]
#[derive(Debug)]
pub struct PyTermStore {
    store: CoreStore,
    registry: Arc<Registry>,
}

#[pymethods]
impl PyTermStore {
    #[new]
    #[pyo3(signature = (signature, *, max_nodes=ROOT_MAX_NODES))]
    fn new(
        py: Python<'_>,
        signature: &Bound<'_, PyAny>,
        #[pyo3(from_py_with = max_nodes)] max_nodes: usize,
    ) -> PyResult<Self> {
        if max_nodes > (1_usize << lofi_chart::term::GENERATION_SHIFT) {
            return Err(PyValueError::new_err(format!(
                "max_nodes exceeds the local serial space: {max_nodes}"
            )));
        }
        let (spec, registry) = lower_signature(py, signature)?;
        let store = CoreStore::new(spec, max_nodes).map_err(|held| error(held, &registry, py))?;
        registry
            .stores
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(store.generation(), store.clone());
        Ok(Self { store, registry })
    }

    #[getter]
    fn signature(&self, py: Python<'_>) -> Py<PyAny> {
        self.registry.signature.clone_ref(py)
    }

    #[getter]
    fn name(&self) -> String {
        self.store.name()
    }

    #[getter]
    fn max_nodes(&self) -> usize {
        self.store.max_nodes()
    }

    #[getter]
    fn local_nodes(&self) -> usize {
        self.store.local_nodes()
    }

    #[getter]
    fn active(&self) -> bool {
        self.store.active()
    }

    #[getter]
    fn generation(&self) -> u32 {
        self.store.generation()
    }

    #[getter]
    fn hits(&self) -> usize {
        self.store.hits()
    }

    #[getter]
    fn misses(&self) -> usize {
        self.store.misses()
    }

    #[pyo3(signature = (op, /, *args, payload=None))]
    fn op(
        &self,
        py: Python<'_>,
        op: &Bound<'_, PyAny>,
        args: &Bound<'_, PyTuple>,
        payload: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyTerm>> {
        let operator = operator_id(py, &self.registry, op)?;
        let payload = payload_value(
            py,
            &self.store.signature().operators[operator_index(operator)],
            payload,
        )?;
        let arguments = args
            .iter()
            .map(|argument| {
                argument
                    .extract::<PyRef<'_, PyTerm>>()
                    .map_err(PyErr::from)
                    .map(|held| held.term.clone())
            })
            .collect::<PyResult<Vec<_>>>()?;
        let term = self
            .store
            .intern(operator, arguments, payload)
            .map_err(|held| error(held, &self.registry, py))?;
        wrap(&self.registry, py, term)
    }

    #[pyo3(name = "const", signature = (sort, name, /))]
    fn constant(&self, py: Python<'_>, sort: &str, name: &str) -> PyResult<Py<PyTerm>> {
        let sort_id = self.sort_id(sort)?;
        if self.store.signature().constants[usize::from(sort_id)].is_none() {
            return Err(error(
                CoreError::MissingDerived {
                    kind: "constant",
                    sort: sort_id,
                },
                &self.registry,
                py,
            ));
        }
        let identifier = self.registry.vocabularies[usize::from(sort_id)]
            .as_ref()
            .and_then(|values| values.iter().position(|held| held == name))
            .ok_or_else(|| {
                initialized_error!(
                    py,
                    PyClassInitializer::from(PyTermError).add_subclass(PyUnknownConstantError {
                        sort: sort.to_owned(),
                        name: name.to_owned(),
                    })
                )
            })?;
        let term = self
            .store
            .constant(
                sort_id,
                u64::try_from(identifier)
                    .map_err(|_| PyValueError::new_err("constant identifier is too large"))?,
            )
            .map_err(|held| error(held, &self.registry, py))?;
        wrap(&self.registry, py, term)
    }

    #[pyo3(signature = (sort, index, /))]
    fn bvar(&self, py: Python<'_>, sort: &str, index: &Bound<'_, PyAny>) -> PyResult<Py<PyTerm>> {
        let index = nonnegative_u64("index", index)?;
        let term = self
            .store
            .bound(self.sort_id(sort)?, index)
            .map_err(|held| error(held, &self.registry, py))?;
        wrap(&self.registry, py, term)
    }

    #[pyo3(signature = (sort, name, /))]
    fn fvar(&self, py: Python<'_>, sort: &str, name: &str) -> PyResult<Py<PyTerm>> {
        let term = self
            .store
            .free(self.sort_id(sort)?, name)
            .map_err(|held| error(held, &self.registry, py))?;
        wrap(&self.registry, py, term)
    }

    #[pyo3(signature = (sort, /))]
    fn hole(&self, py: Python<'_>, sort: &str) -> PyResult<Py<PyTerm>> {
        let term = self
            .store
            .hole(self.sort_id(sort)?)
            .map_err(|held| error(held, &self.registry, py))?;
        wrap(&self.registry, py, term)
    }

    #[pyo3(signature = (binder, variable, /, *args))]
    fn bind(
        &self,
        py: Python<'_>,
        binder: &str,
        variable: &str,
        args: &Bound<'_, PyTuple>,
    ) -> PyResult<Py<PyTerm>> {
        let operator = *self.registry.operator_ids.get(binder).ok_or_else(|| {
            initialized_error!(
                py,
                PyClassInitializer::from(PyTermError).add_subclass(PyUnknownOperatorError {
                    operator: binder
                        .to_owned()
                        .into_pyobject(py)
                        .expect("string conversion")
                        .unbind()
                        .into_any(),
                    signature: self.registry.signature_name.clone(),
                })
            )
        })?;
        let arguments = args
            .iter()
            .map(|argument| {
                argument
                    .extract::<PyRef<'_, PyTerm>>()
                    .map_err(PyErr::from)
                    .map(|held| held.term.clone())
            })
            .collect::<PyResult<Vec<_>>>()?;
        let term = self
            .store
            .bind(operator, variable, arguments)
            .map_err(|held| error(held, &self.registry, py))?;
        wrap(&self.registry, py, term)
    }

    #[pyo3(signature = (sort, name, /))]
    fn atom(&self, py: Python<'_>, sort: &str, name: &str) -> PyResult<AtomId> {
        if name.is_empty() {
            return Err(PyValueError::new_err("name is empty"));
        }
        self.store
            .atom(self.sort_id(sort)?, name)
            .map_err(|held| error(held, &self.registry, py))
    }

    #[pyo3(signature = (index, /))]
    fn atom_name(&self, py: Python<'_>, index: &Bound<'_, PyAny>) -> PyResult<String> {
        let index = atom_id("index", index)?;
        self.store
            .atom_value(index)
            .map(|(_, name)| name)
            .map_err(|held| error(held, &self.registry, py))
    }

    #[pyo3(signature = (index, /))]
    fn atom_sort(&self, py: Python<'_>, index: &Bound<'_, PyAny>) -> PyResult<String> {
        let index = atom_id("index", index)?;
        self.store
            .atom_value(index)
            .map(|(sort, _)| self.registry.sorts[usize::from(sort)].clone())
            .map_err(|held| error(held, &self.registry, py))
    }

    #[pyo3(signature = (term, /))]
    fn payload_name(&self, py: Python<'_>, term: PyRef<'_, PyTerm>) -> PyResult<String> {
        let rendered = term.__repr__(py)?;
        let result = match term.term.identity() {
            Identity::Name(name) => Ok(name.clone()),
            Identity::Integer(identifier) => self.registry.vocabularies
                [usize::from(term.term.sort())]
            .as_ref()
            .and_then(|values| {
                usize::try_from(*identifier)
                    .ok()
                    .and_then(|index| values.get(index))
            })
            .cloned()
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "{rendered} is neither a constant nor a free variable"
                ))
            }),
            Identity::None => Err(PyValueError::new_err(format!(
                "{rendered} is neither a constant nor a free variable"
            ))),
        };
        drop(term);
        result
    }

    #[pyo3(signature = (sort, avoid, count, /))]
    fn fresh(
        &self,
        py: Python<'_>,
        sort: &str,
        avoid: &Bound<'_, PyAny>,
        count: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyTuple>> {
        if avoid.is_instance_of::<PyBool>() || !avoid.is_instance_of::<PyInt>() {
            return Err(PyTypeError::new_err(format!(
                "avoid is not an integer: {}",
                avoid.repr()?
            )));
        }
        if avoid.lt(0_i8)? {
            return Err(PyValueError::new_err(format!(
                "avoid is negative: {}",
                avoid.repr()?
            )));
        }
        let avoid = avoid.extract::<BigUint>()?;
        let count = nonnegative_usize("count", count)?;
        let atoms = self
            .store
            .fresh(self.sort_id(sort)?, &avoid, count)
            .map_err(|held| error(held, &self.registry, py))?;
        Ok(PyTuple::new(py, atoms)?.unbind())
    }

    #[pyo3(signature = (term, /))]
    fn contains(&self, py: Python<'_>, term: PyRef<'_, PyTerm>) -> PyResult<bool> {
        let result = self
            .store
            .contains(&term.term)
            .map_err(|held| error(held, &self.registry, py));
        drop(term);
        result
    }

    #[pyo3(signature = (sort, classifiers, /))]
    fn classify(&self, py: Python<'_>, sort: &str, classifiers: &Bound<'_, PyAny>) -> PyResult<()> {
        let sort_id = self.sort_id(sort)?;
        let expected_sort: Option<String> = self
            .registry
            .signature
            .bind(py)
            .getattr("classifiers")?
            .get_item(sort)?
            .extract()
            .ok();
        let expected_sort = expected_sort.ok_or_else(|| {
            PyValueError::new_err(format!("sort {sort:?} has no declared classifier sort"))
        })?;
        let held = classifiers
            .try_iter()?
            .map(|value| {
                let value = value?;
                if value.is_none() {
                    return Ok(None);
                }
                let term = value.extract::<PyRef<'_, PyTerm>>()?;
                let received = &self.registry.sorts[usize::from(term.term.sort())];
                if received != &expected_sort {
                    return Err(initialized_error!(
                        py,
                        PyClassInitializer::from(PyTermError).add_subclass(PySortError {
                            operator: format!("#const[{sort}]"),
                            position: 0,
                            expected: expected_sort.clone(),
                            received: received.clone(),
                        })
                    ));
                }
                Ok(Some(term.term.clone()))
            })
            .collect::<PyResult<Vec<_>>>()?;
        let expected = self.registry.vocabularies[usize::from(sort_id)]
            .as_ref()
            .map_or(0, Vec::len);
        if held.len() != expected {
            return Err(PyValueError::new_err(format!(
                "sort {sort:?} has {expected} constants and was given {} classifiers",
                held.len()
            )));
        }
        self.store
            .classify(sort_id, held)
            .map_err(|held| error(held, &self.registry, py))
    }

    #[pyo3(signature = (constant, /))]
    fn classifier(
        &self,
        py: Python<'_>,
        constant: PyRef<'_, PyTerm>,
    ) -> PyResult<Option<Py<PyTerm>>> {
        let core = constant.term.clone();
        let rendered = constant.__repr__(py)?;
        drop(constant);
        let op = self.store.signature().operators[operator_index(core.operator())].clone();
        if op.payload != PayloadKind::Constant {
            return Err(PyValueError::new_err(format!(
                "{rendered} is not a constant"
            )));
        }
        let sort_name = &self.registry.sorts[usize::from(op.sort)];
        if self
            .registry
            .signature
            .bind(py)
            .getattr("classifiers")?
            .get_item(sort_name)
            .is_err()
        {
            return Err(PyValueError::new_err(format!(
                "sort {sort_name:?} has no declared classifier sort"
            )));
        }
        self.store
            .classifier(
                op.sort,
                usize::try_from(core.payload().unwrap_or(0)).unwrap_or(usize::MAX),
            )
            .map(|term| wrap(&self.registry, py, term))
            .transpose()
    }

    #[pyo3(signature = (sort, tag, /))]
    fn view(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        sort: String,
        tag: &Bound<'_, PyAny>,
    ) -> PyResult<PySortView> {
        let _ = tag;
        slf.sort_id(&sort)?;
        Ok(PySortView {
            store: slf.into_pyobject(py)?.unbind(),
            sort,
        })
    }

    #[pyo3(signature = (name, /, *, max_nodes))]
    fn region(
        &self,
        py: Python<'_>,
        name: String,
        #[pyo3(from_py_with = max_nodes)] max_nodes: usize,
    ) -> PyResult<PyRegion> {
        if name.is_empty() {
            return Err(PyValueError::new_err("name is empty"));
        }
        if max_nodes > (1_usize << lofi_chart::term::GENERATION_SHIFT) {
            return Err(PyValueError::new_err(format!(
                "max_nodes exceeds the local serial space: {max_nodes}"
            )));
        }
        let child = self
            .store
            .child(name, max_nodes)
            .map_err(|held| error(held, &self.registry, py))?;
        self.registry
            .stores
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(child.generation(), child.clone());
        Ok(PyRegion {
            child: Py::new(
                py,
                Self {
                    store: child,
                    registry: self.registry.clone(),
                },
            )?,
        })
    }

    #[pyo3(signature = (term, /))]
    fn promote(&self, py: Python<'_>, term: PyRef<'_, PyTerm>) -> PyResult<Py<PyTerm>> {
        let promoted = self
            .store
            .promote(&term.term)
            .map_err(|held| error(held, &self.registry, py))?;
        drop(term);
        wrap(&self.registry, py, promoted)
    }

    fn release(&self, py: Python<'_>) -> PyResult<()> {
        self.store
            .release()
            .map_err(|held| error(held, &self.registry, py))?;
        self.registry.prune_dead_terms(py)
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let name = self
            .registry
            .signature_name
            .as_str()
            .into_pyobject(py)?
            .repr()?;
        Ok(format!(
            "TermStore({name}, generation {}, {} nodes)",
            self.store.generation(),
            self.store.local_nodes()
        ))
    }
}

/// A statically typed view over one sort of a native store.
#[pyclass(frozen, module = "lofi._native", name = "SortView")]
#[derive(Debug)]
pub struct PySortView {
    store: Py<PyTermStore>,
    sort: String,
}

#[pymethods]
impl PySortView {
    #[new]
    fn new() -> PyResult<Self> {
        Err(PyTypeError::new_err(
            "SortView is created by TermStore.view and made nowhere else",
        ))
    }

    #[getter]
    fn store(&self, py: Python<'_>) -> Py<PyTermStore> {
        self.store.clone_ref(py)
    }

    #[getter]
    fn sort(&self) -> &str {
        &self.sort
    }

    #[pyo3(signature = (operator, /, *args, payload=None))]
    fn op(
        &self,
        py: Python<'_>,
        operator: &Bound<'_, PyAny>,
        args: &Bound<'_, PyTuple>,
        payload: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyTerm>> {
        let store = self.store.borrow(py);
        let resolved = if let Ok(name) = operator.extract::<String>() {
            let identifier = store
                .registry
                .operator_ids
                .get(&name)
                .ok_or_else(|| PyValueError::new_err(format!("unknown operator {name:?}")))?;
            store.registry.operators[operator_index(*identifier)]
                .bind(py)
                .clone()
        } else {
            operator.clone()
        };
        let held_sort: String = resolved.getattr("sort")?.extract()?;
        if held_sort != self.sort {
            let name: String = resolved.getattr("name")?.extract()?;
            return Err(PyValueError::new_err(format!(
                "operator {name:?} builds terms of sort {held_sort:?}, not {:?}",
                self.sort
            )));
        }
        store.op(py, &resolved, args, payload)
    }

    #[pyo3(name = "const", signature = (name, /))]
    fn constant(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyTerm>> {
        self.store.borrow(py).constant(py, &self.sort, name)
    }

    #[pyo3(signature = (index, /))]
    fn bvar(&self, py: Python<'_>, index: &Bound<'_, PyAny>) -> PyResult<Py<PyTerm>> {
        self.store.borrow(py).bvar(py, &self.sort, index)
    }

    #[pyo3(signature = (name, /))]
    fn fvar(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyTerm>> {
        self.store.borrow(py).fvar(py, &self.sort, name)
    }

    fn hole(&self, py: Python<'_>) -> PyResult<Py<PyTerm>> {
        self.store.borrow(py).hole(py, &self.sort)
    }

    #[pyo3(signature = (binder, variable, /, *args))]
    fn bind(
        &self,
        py: Python<'_>,
        binder: &str,
        variable: &str,
        args: &Bound<'_, PyTuple>,
    ) -> PyResult<Py<PyTerm>> {
        let store = self.store.borrow(py);
        let operator = store
            .registry
            .operator_ids
            .get(binder)
            .and_then(|identifier| store.registry.operators.get(operator_index(*identifier)))
            .ok_or_else(|| PyValueError::new_err(format!("unknown binder {binder:?}")))?;
        let held_sort: String = operator.bind(py).getattr("sort")?.extract()?;
        if held_sort != self.sort {
            return Err(PyValueError::new_err(format!(
                "binder {binder:?} builds terms of sort {held_sort:?}, not {:?}",
                self.sort
            )));
        }
        store.bind(py, binder, variable, args)
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let name = self
            .store
            .borrow(py)
            .registry
            .signature_name
            .as_str()
            .into_pyobject(py)?
            .repr()?;
        let sort = self.sort.as_str().into_pyobject(py)?.repr()?;
        Ok(format!("SortView({name}, {sort})"))
    }
}

impl PyTermStore {
    pub(crate) fn from_core(
        py: Python<'_>,
        signature: &Bound<'_, PyAny>,
        store: CoreStore,
    ) -> PyResult<Self> {
        let (spec, registry) = lower_signature(py, signature)?;
        if store.signature() != &spec {
            return Err(PyValueError::new_err(
                "the materialized signature does not match the linked native store",
            ));
        }
        registry
            .stores
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(store.generation(), store.clone());
        Ok(Self { store, registry })
    }

    pub(crate) fn core_store(&self) -> CoreStore {
        self.store.clone()
    }

    pub(crate) fn registry(&self) -> Arc<Registry> {
        self.registry.clone()
    }

    fn sort_id(&self, sort: &str) -> PyResult<u16> {
        self.registry.sort_ids.get(sort).copied().ok_or_else(|| {
            PyValueError::new_err(format!(
                "signature {:?} declares no sort {sort:?}",
                self.registry.signature_name
            ))
        })
    }

    fn instantiate_term(
        &self,
        py: Python<'_>,
        body: &CoreTerm,
        value: &CoreTerm,
        depth: u32,
    ) -> PyResult<Py<PyTerm>> {
        let term = self
            .store
            .instantiate(body, value, depth)
            .map_err(|held| error(held, &self.registry, py))?;
        wrap(&self.registry, py, term)
    }

    fn abstract_term(
        &self,
        py: Python<'_>,
        body: &CoreTerm,
        atom: AtomId,
        depth: u32,
    ) -> PyResult<Py<PyTerm>> {
        let term = self
            .store
            .abstract_atom(body, atom, depth)
            .map_err(|held| error(held, &self.registry, py))?;
        wrap(&self.registry, py, term)
    }

    fn substitute_term(
        &self,
        py: Python<'_>,
        body: &CoreTerm,
        atom: AtomId,
        value: &CoreTerm,
    ) -> PyResult<Py<PyTerm>> {
        let term = self
            .store
            .substitute(body, atom, value)
            .map_err(|held| error(held, &self.registry, py))?;
        wrap(&self.registry, py, term)
    }

    fn open_binding(
        &self,
        py: Python<'_>,
        binder: &CoreTerm,
    ) -> PyResult<(Py<PyTerm>, Py<PyTuple>)> {
        let (term, atoms) = self
            .store
            .open_binder(binder)
            .map_err(|held| error(held, &self.registry, py))?;
        Ok((
            wrap(&self.registry, py, term)?,
            PyTuple::new(py, atoms)?.unbind(),
        ))
    }

    fn close_binding(
        &self,
        py: Python<'_>,
        binder: &CoreTerm,
        atoms: &[AtomId],
    ) -> PyResult<Py<PyTerm>> {
        let term = self
            .store
            .close_binder(binder, atoms)
            .map_err(|held| error(held, &self.registry, py))?;
        wrap(&self.registry, py, term)
    }
}

/// The context manager returned by `TermStore.region`.
#[pyclass(module = "lofi._native", name = "TermRegion")]
#[derive(Debug)]
pub struct PyRegion {
    child: Py<PyTermStore>,
}

#[pymethods]
impl PyRegion {
    fn __enter__(&self, py: Python<'_>) -> Py<PyTermStore> {
        self.child.clone_ref(py)
    }

    fn __exit__(
        &self,
        py: Python<'_>,
        _kind: &Bound<'_, PyAny>,
        _value: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        self.child.borrow(py).release(py)?;
        Ok(false)
    }
}

/// Instantiate one bound variable through the native store.
/// # Errors
/// Returns a Python exception when the store refuses the substitution.
#[pyfunction]
#[pyo3(signature = (store, body, value, depth=0))]
pub fn instantiate(
    py: Python<'_>,
    store: PyRef<'_, PyTermStore>,
    body: PyRef<'_, PyTerm>,
    value: PyRef<'_, PyTerm>,
    #[pyo3(from_py_with = depth)] depth: u32,
) -> PyResult<Py<PyTerm>> {
    let body_term = body.term.clone();
    let value_term = value.term.clone();
    drop(body);
    drop(value);
    let result = store.instantiate_term(py, &body_term, &value_term, depth);
    drop(store);
    result
}

/// Abstract one free variable through the native store.
/// # Errors
/// Returns a Python exception when the atom is invalid or the store refuses the abstraction.
#[pyfunction(name = "abstract")]
#[pyo3(signature = (store, body, atom, depth=0))]
pub fn abstract_(
    py: Python<'_>,
    store: PyRef<'_, PyTermStore>,
    body: PyRef<'_, PyTerm>,
    atom: &Bound<'_, PyAny>,
    #[pyo3(from_py_with = depth)] depth: u32,
) -> PyResult<Py<PyTerm>> {
    let atom = atom_id("atom", atom)?;
    let body_term = body.term.clone();
    drop(body);
    let result = store.abstract_term(py, &body_term, atom, depth);
    drop(store);
    result
}

/// Substitute one free variable through the native store.
/// # Errors
/// Returns a Python exception when the atom is invalid or the store refuses the substitution.
#[pyfunction]
pub fn substitute(
    py: Python<'_>,
    store: PyRef<'_, PyTermStore>,
    body: PyRef<'_, PyTerm>,
    atom: &Bound<'_, PyAny>,
    value: PyRef<'_, PyTerm>,
) -> PyResult<Py<PyTerm>> {
    let atom = atom_id("atom", atom)?;
    let body_term = body.term.clone();
    let value_term = value.term.clone();
    drop(body);
    drop(value);
    let result = store.substitute_term(py, &body_term, atom, &value_term);
    drop(store);
    result
}

/// Open every binding site through the native store.
/// # Errors
/// Returns a Python exception when the binder is invalid or the store refuses the rewrite.
#[pyfunction]
pub fn open_binder(
    py: Python<'_>,
    store: PyRef<'_, PyTermStore>,
    binder: PyRef<'_, PyTerm>,
) -> PyResult<(Py<PyTerm>, Py<PyTuple>)> {
    let binder_term = binder.term.clone();
    drop(binder);
    let result = store.open_binding(py, &binder_term);
    drop(store);
    result
}

/// Close every binding site through the native store.
/// # Errors
/// Returns a Python exception when an atom is invalid or the store refuses the rewrite.
#[pyfunction]
pub fn close_binder(
    py: Python<'_>,
    store: PyRef<'_, PyTermStore>,
    binder: PyRef<'_, PyTerm>,
    atoms: &Bound<'_, PyTuple>,
) -> PyResult<Py<PyTerm>> {
    let atoms = atoms
        .iter()
        .map(|atom| atom_id("atom", &atom))
        .collect::<PyResult<Vec<_>>>()?;
    let binder_term = binder.term.clone();
    drop(binder);
    let result = store.close_binding(py, &binder_term, &atoms);
    drop(store);
    result
}
