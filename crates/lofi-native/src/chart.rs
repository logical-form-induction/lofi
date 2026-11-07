//! Python boundary for dense-id chart drawing.

use std::collections::{HashMap, HashSet};
use std::convert::Infallible;
use std::fmt;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::{Mutex, MutexGuard};

use lofi_chart::chart::{
    CellExecution, ChartError, ForestEdge, ForestKey as CoreForestKey,
    ForestTable as CoreForestTable, InsideTable as CoreInsideTable, ItemId,
    MissBatch as CoreMissBatch, NormalDecision, ProductAlternative, Span, Step, SweepError,
    TableError, UnaryClosure,
};
use lofi_chart::drawing::{
    DrawingHost, DrawingProgram, DrawingResolver, DrawingRule, EdgeScoring, ForestHost,
    ReachabilityInputs, ResidualRules, represent as represent_scoped,
};
use lofi_chart::program::{DrawError, HostError};
use lofi_chart::reachability::ReachabilityRule as CoreReachabilityRule;
use lofi_chart::semiring::{
    Boolean as CoreBoolean, Count, Counting as CoreCounting, Expectation as CoreExpectation,
    ExpectedWeight, KBest as CoreKBest, LogInside as CoreLogInside, Semiring,
    Viterbi as CoreViterbi,
};
use num_bigint::BigUint;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::rule::{PyRepresentativeKernel, PyRuleProgram, lower_object_pattern};
use crate::semiring::{
    PyBoolean, PyCounting, PyExpectation, PyKBest, PyLogInside, PyViterbi, PythonCarrier,
};
use crate::term::{PyTerm, Registry, draw_error, wrap};
use lofi_chart::rule::{
    RepresentativeKernel as CoreRepresentativeKernel, RepresentativeTable, represent_ground_pattern,
};
use lofi_chart::term::{Term as CoreTerm, TermStore as CoreStore};

type PyClosure = (Vec<((u32, u8), u16, (u32, u8))>, Vec<(u32, u8)>, bool);
type PyForestKey = (Span, (u32, u8));
type PyForestEdge = (u32, Vec<PyForestKey>, u32, Py<PyAny>, Option<usize>, usize);
type PyForestProduct = (
    u32,
    Vec<Vec<PyForestKey>>,
    u32,
    Py<PyAny>,
    Option<usize>,
    usize,
);
type PyDecisionRow = (Vec<u8>, String, u8);
type PyRuleRow = (usize, usize, bool, Vec<PyDecisionRow>);

fn chart_error(error: &ChartError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

/// Declare that an edge score is invariant across spans and splits.
#[pyclass(frozen, eq, hash, module = "lofi._native")]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct SpanFree;

#[pymethods]
impl SpanFree {
    #[new]
    const fn new() -> Self {
        Self
    }

    fn __repr__(&self) -> String {
        format!("{self:?}()")
    }
}

/// Declare that every rule edge has the carrier's multiplicative identity.
#[pyclass(frozen, eq, hash, module = "lofi._native")]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct UnitEdges;

#[pymethods]
impl UnitEdges {
    #[new]
    const fn new() -> Self {
        Self
    }

    fn __repr__(&self) -> String {
        format!("{self:?}()")
    }
}

/// Declare that an edge score may depend on its span or split.
#[pyclass(frozen, eq, hash, module = "lofi._native")]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct SpanDependent;

#[pymethods]
impl SpanDependent {
    #[new]
    const fn new() -> Self {
        Self
    }

    fn __repr__(&self) -> String {
        format!("{self:?}()")
    }
}

fn scoring_mode(lowering: &Bound<'_, PyAny>) -> PyResult<EdgeScoring> {
    if lowering.is_instance_of::<SpanDependent>() {
        Ok(EdgeScoring::Dependent)
    } else if lowering.is_instance_of::<SpanFree>() {
        Ok(EdgeScoring::SpanFree)
    } else if lowering.is_instance_of::<UnitEdges>() {
        Ok(EdgeScoring::UnitEdges)
    } else {
        Err(PyValueError::new_err(
            "a scorer lowering is SpanDependent, SpanFree, or UnitEdges",
        ))
    }
}

/// One native-to-Python batch of unresolved rule applications.
#[pyclass(module = "lofi._native", name = "MissBatch")]
#[derive(Clone, Debug)]
pub struct PyMissBatch {
    inner: CoreMissBatch,
}

#[pymethods]
impl PyMissBatch {
    #[getter]
    fn inputs(&self) -> Vec<u32> {
        self.inner.inputs().to_vec()
    }

    #[getter]
    fn tags(&self) -> Vec<u8> {
        self.inner.tags().to_vec()
    }

    #[getter]
    fn candidates(&self) -> Vec<u16> {
        self.inner.candidates().to_vec()
    }

    #[getter]
    const fn span(&self) -> (usize, usize) {
        self.inner.span()
    }

    /// Add one resolved and admitted step to this batch.
    fn push(&mut self, rule: u16, output: u32, tag: u8) -> PyResult<()> {
        self.inner
            .push(rule, output, tag)
            .map_err(|error| chart_error(&error))
    }

    fn __repr__(&self) -> String {
        format!(
            "MissBatch(inputs={}, candidates={}, results={})",
            self.inner.inputs().len(),
            self.inner.candidates().len(),
            self.inner.results().len()
        )
    }
}

/// The native rule-drawing program shared by both chart retention modes.
#[pyclass(frozen, module = "lofi._native", name = "ChartProgram")]
pub struct PyChartProgram {
    core: DrawingProgram<Registry>,
}

fn represent(
    kernel: &CoreRepresentativeKernel,
    table: &mut RepresentativeTable,
    term: &CoreTerm,
) -> PyResult<CoreTerm> {
    represent_scoped::<Registry>(kernel, table, term).map_err(draw_error)
}

/// The Python resolver a chart sweep consults for weights, residual rules and kept items.
struct PythonHost<'py, D> {
    py: Python<'py>,
    resolver: Bound<'py, PyAny>,
    decoder: D,
}

fn host_error(error: PyErr) -> HostError {
    Box::new(error)
}

impl<D> ResidualRules for PythonHost<'_, D> {
    fn resolve(&mut self, batch: CoreMissBatch) -> Result<Vec<Step>, HostError> {
        let batch = Bound::new(self.py, PyMissBatch { inner: batch }).map_err(host_error)?;
        self.resolver
            .call_method1("resolve", (&batch,))
            .map_err(host_error)?;
        let results = batch.borrow().inner.results().to_vec();
        Ok(results)
    }
}

impl<W, D: WeightDecoder<W>> DrawingHost<W> for PythonHost<'_, D> {
    fn admit_rows(
        &mut self,
        rule: u16,
        tags: &[u8],
        input_pairs: &[[u32; 2]],
        output: u32,
    ) -> Result<Vec<bool>, HostError> {
        self.resolver
            .call_method1("admit_rows", (rule, tags, input_pairs, output))
            .and_then(|admitted| admitted.extract())
            .map_err(host_error)
    }

    fn weight(
        &mut self,
        rule: u16,
        inputs: &[u32],
        output: u32,
        span: Span,
        split: Option<usize>,
    ) -> Result<W, HostError> {
        self.resolver
            .call_method1("weight", (rule, inputs, output, span, split))
            .and_then(|object| self.decoder.decode(&object))
            .map_err(host_error)
    }

    fn finish_width(&mut self) -> Result<(), HostError> {
        self.py.check_signals().map_err(host_error)
    }
}

impl<W, D: WeightDecoder<W> + WeightEncoder<W>> ForestHost<W> for PythonHost<'_, D> {
    fn root_keep(
        &mut self,
        span: Span,
        items: &[(ItemId, W)],
        closure: &UnaryClosure,
    ) -> Result<Option<Vec<ItemId>>, HostError> {
        let weighted = weighted_items_object(self.py, items, &self.decoder).map_err(host_error)?;
        self.resolver
            .call_method1(
                "root_keep",
                (span, weighted, closure_tuple(closure.clone())),
            )
            .and_then(|result| result.extract::<Option<Vec<(u32, u8)>>>())
            .map(|items| items.map(dense_items))
            .map_err(host_error)
    }

    fn prune_keep(
        &mut self,
        span: Span,
        items: &[(ItemId, W)],
    ) -> Result<Option<Vec<ItemId>>, HostError> {
        let weighted = weighted_items_object(self.py, items, &self.decoder).map_err(host_error)?;
        self.resolver
            .call_method1("prune_keep", (span, weighted))
            .and_then(|result| result.extract::<Option<Vec<(u32, u8)>>>())
            .map(|items| items.map(dense_items))
            .map_err(host_error)
    }
}

fn reachability_terms(
    value: &Bound<'_, PyAny>,
    field: &str,
    registry: &Arc<Registry>,
    representatives: &CoreRepresentativeKernel,
    table: &mut RepresentativeTable,
) -> PyResult<Option<Vec<CoreTerm>>> {
    if value.is_none() {
        return Ok(None);
    }
    value
        .try_iter()?
        .map(|term| {
            let term = term?.extract::<PyRef<'_, PyTerm>>()?;
            if !Arc::ptr_eq(registry, &term.registry()) {
                return Err(PyValueError::new_err(format!(
                    "a rule's {field} category belongs to another term signature"
                )));
            }
            represent(representatives, table, &term.core_term())
        })
        .collect::<PyResult<Vec<_>>>()
        .map(Some)
}

fn lower_reachability_rules(
    rules: &Bound<'_, PyAny>,
    registry: &Arc<Registry>,
    store: &CoreStore,
    representatives: &CoreRepresentativeKernel,
    table: &mut RepresentativeTable,
) -> PyResult<Vec<CoreReachabilityRule>> {
    rules
        .try_iter()?
        .map(|row| {
            let row = row?;
            let arity: usize = row.get_item(0)?.extract()?;
            let schemata = row
                .get_item(1)?
                .try_iter()?
                .map(|schema| {
                    let schema = schema?;
                    let premises = schema
                        .get_item(0)?
                        .try_iter()?
                        .map(|pattern| {
                            let pattern = lower_object_pattern(&pattern?, registry)?;
                            represent_ground_pattern(&pattern, store, representatives, table)
                                .map_err(|error| PyValueError::new_err(error.to_string()))
                        })
                        .collect::<PyResult<Vec<_>>>()?;
                    let conclusion = lower_object_pattern(&schema.get_item(1)?, registry)?;
                    let conclusion =
                        represent_ground_pattern(&conclusion, store, representatives, table)
                            .map_err(|error| PyValueError::new_err(error.to_string()))?;
                    Ok((premises, conclusion))
                })
                .collect::<PyResult<Vec<_>>>()?;
            Ok(CoreReachabilityRule {
                arity,
                schemata,
                introduces: reachability_terms(
                    &row.get_item(2)?,
                    "introduced",
                    registry,
                    representatives,
                    table,
                )?,
                consumes: reachability_terms(
                    &row.get_item(3)?,
                    "consumed",
                    registry,
                    representatives,
                    table,
                )?,
            })
        })
        .collect()
}
/// Native inside weights and their single width-ordered sweep.
#[pyclass(frozen, module = "lofi._native", name = "InsideTable")]
pub struct PyInsideTable {
    inner: Mutex<InsideStorage>,
}

/// Native packed-forest storage and its single width-ordered sweep.
#[pyclass(frozen, module = "lofi._native", name = "ForestTable")]
pub struct PyForestTable {
    inner: Mutex<ForestStorage>,
}

impl PyForestTable {
    fn inner(&self) -> PyResult<MutexGuard<'_, ForestStorage>> {
        self.inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("the forest table lock is poisoned"))
    }
}

impl PyInsideTable {
    fn inner(&self) -> PyResult<MutexGuard<'_, InsideStorage>> {
        self.inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("the inside table lock is poisoned"))
    }
}

#[pymethods]
impl PyInsideTable {
    #[new]
    fn new(
        tokens: usize,
        semiring: &Bound<'_, PyAny>,
        lowering: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let inner = if lowering.is_instance_of::<PyCounting>() {
            InsideStorage::Counting(CoreInsideTable::new(tokens))
        } else if lowering.is_instance_of::<PyViterbi>() {
            InsideStorage::Viterbi(CoreInsideTable::new(tokens))
        } else if lowering.is_instance_of::<PyLogInside>() {
            InsideStorage::LogInside(CoreInsideTable::new(tokens))
        } else if lowering.is_instance_of::<PyBoolean>() {
            InsideStorage::Boolean(CoreInsideTable::new(tokens))
        } else if lowering.is_instance_of::<PyKBest>() {
            let k = lowering.getattr("k")?.extract::<usize>()?;
            let k = NonZeroUsize::new(k)
                .ok_or_else(|| PyValueError::new_err("a k-best width is at least one, not 0"))?;
            InsideStorage::KBest {
                table: CoreInsideTable::new(tokens),
                semiring: CoreKBest::new(k),
            }
        } else if lowering.is_instance_of::<PyExpectation>() {
            let dim = lowering.getattr("dim")?.extract::<usize>()?;
            let dim = NonZeroUsize::new(dim).ok_or_else(|| {
                PyValueError::new_err("an expectation has at least one coordinate, not 0")
            })?;
            InsideStorage::Expectation {
                table: CoreInsideTable::new(tokens),
                semiring: CoreExpectation::new(dim),
            }
        } else if lowering.is_instance_of::<PythonCarrier>() {
            InsideStorage::Python {
                table: CoreInsideTable::new(tokens),
                semiring: PythonSemiring::new(semiring)?,
            }
        } else {
            return Err(PyValueError::new_err(
                "a semiring lowering must select a native carrier or PythonCarrier",
            ));
        };
        Ok(Self {
            inner: Mutex::new(inner),
        })
    }

    #[getter]
    fn tokens(&self) -> PyResult<usize> {
        Ok(self.inner()?.tokens())
    }

    #[getter]
    fn truncated(&self) -> PyResult<bool> {
        Ok(self.inner()?.truncated())
    }

    #[getter]
    fn num_categories(&self) -> PyResult<usize> {
        Ok(self.inner()?.num_categories())
    }

    fn mark_truncated(&self) -> PyResult<()> {
        self.inner()?.mark_truncated();
        Ok(())
    }

    fn add(&self, span: Span, item: (u32, u8), weight: &Bound<'_, PyAny>) -> PyResult<()> {
        let item = ItemId {
            category: item.0,
            tag: item.1,
        };
        let mut inner = self.inner()?;

        match &mut *inner {
            InsideStorage::Counting(table) => {
                add_native(table, &CoreCounting, span, item, weight, &CountDecoder)
            }
            InsideStorage::Viterbi(table) => {
                add_native(table, &CoreViterbi, span, item, weight, &FloatDecoder)
            }
            InsideStorage::LogInside(table) => {
                add_native(table, &CoreLogInside, span, item, weight, &FloatDecoder)
            }
            InsideStorage::Boolean(table) => {
                add_native(table, &CoreBoolean, span, item, weight, &BooleanDecoder)
            }
            InsideStorage::KBest { table, semiring } => {
                add_native(table, semiring, span, item, weight, &KBestDecoder)
            }
            InsideStorage::Expectation { table, semiring } => add_native(
                table,
                semiring,
                span,
                item,
                weight,
                &ExpectationDecoder {
                    dim: semiring.dim().get(),
                },
            ),
            InsideStorage::Python { table, semiring } => table
                .add(span, item, PythonDecoder.decode(weight)?, semiring)
                .map_err(python_table_error),
        }
    }

    fn restrict(&self, span: Span, keep: Vec<(u32, u8)>) -> PyResult<()> {
        let keep: Vec<ItemId> = keep
            .into_iter()
            .map(|(category, tag)| ItemId { category, tag })
            .collect();
        self.inner()?
            .restrict(span, &keep)
            .map_err(|error| table_error(&error))
    }

    fn items_at(&self, span: Span) -> PyResult<Vec<(u32, u8)>> {
        Ok(self
            .inner()?
            .items_at(span)
            .map_err(|error| table_error(&error))?
            .into_iter()
            .map(|item| (item.category, item.tag))
            .collect())
    }

    fn tagged_at(&self, py: Python<'_>, span: Span) -> PyResult<Vec<(u32, u8, Py<PyAny>)>> {
        let inner = self.inner()?;

        match &*inner {
            InsideStorage::Counting(table) => tagged_objects(py, table, span, &CountDecoder),
            InsideStorage::Viterbi(table) | InsideStorage::LogInside(table) => {
                tagged_objects(py, table, span, &FloatDecoder)
            }
            InsideStorage::Boolean(table) => tagged_objects(py, table, span, &BooleanDecoder),
            InsideStorage::KBest { table, .. } => tagged_objects(py, table, span, &KBestDecoder),
            InsideStorage::Expectation { table, semiring } => tagged_objects(
                py,
                table,
                span,
                &ExpectationDecoder {
                    dim: semiring.dim().get(),
                },
            ),
            InsideStorage::Python { table, .. } => tagged_objects(py, table, span, &PythonDecoder),
        }
    }

    fn categories_at(&self, py: Python<'_>, span: Span) -> PyResult<Vec<(u32, Py<PyAny>)>> {
        let inner = self.inner()?;

        match &*inner {
            InsideStorage::Counting(table) => {
                categories_native(py, table, &CoreCounting, span, &CountDecoder)
            }
            InsideStorage::Viterbi(table) => {
                categories_native(py, table, &CoreViterbi, span, &FloatDecoder)
            }
            InsideStorage::LogInside(table) => {
                categories_native(py, table, &CoreLogInside, span, &FloatDecoder)
            }
            InsideStorage::Boolean(table) => {
                categories_native(py, table, &CoreBoolean, span, &BooleanDecoder)
            }
            InsideStorage::KBest { table, semiring } => {
                categories_native(py, table, semiring, span, &KBestDecoder)
            }
            InsideStorage::Expectation { table, semiring } => categories_native(
                py,
                table,
                semiring,
                span,
                &ExpectationDecoder {
                    dim: semiring.dim().get(),
                },
            ),
            InsideStorage::Python { table, semiring } => table
                .categories_at(span, semiring)
                .map_err(python_table_error)?
                .into_iter()
                .map(|(category, weight)| Ok((category, PythonDecoder.encode(py, &weight)?)))
                .collect(),
        }
    }

    fn tagged(&self, py: Python<'_>, span: Span, item: (u32, u8)) -> PyResult<Py<PyAny>> {
        let item = ItemId {
            category: item.0,
            tag: item.1,
        };
        let inner = self.inner()?;

        match &*inner {
            InsideStorage::Counting(table) => {
                tagged_native(py, table, &CoreCounting, span, item, &CountDecoder)
            }
            InsideStorage::Viterbi(table) => {
                tagged_native(py, table, &CoreViterbi, span, item, &FloatDecoder)
            }
            InsideStorage::LogInside(table) => {
                tagged_native(py, table, &CoreLogInside, span, item, &FloatDecoder)
            }
            InsideStorage::Boolean(table) => {
                tagged_native(py, table, &CoreBoolean, span, item, &BooleanDecoder)
            }
            InsideStorage::KBest { table, semiring } => {
                tagged_native(py, table, semiring, span, item, &KBestDecoder)
            }
            InsideStorage::Expectation { table, semiring } => tagged_native(
                py,
                table,
                semiring,
                span,
                item,
                &ExpectationDecoder {
                    dim: semiring.dim().get(),
                },
            ),
            InsideStorage::Python { table, semiring } => {
                let weight = table
                    .tagged(span, item)
                    .map_err(|error| table_error(&error))?
                    .unwrap_or_else(|| semiring.zero());
                PythonDecoder.encode(py, &weight)
            }
        }
    }

    fn inside(&self, py: Python<'_>, span: Span, category: u32) -> PyResult<Py<PyAny>> {
        let inner = self.inner()?;

        match &*inner {
            InsideStorage::Counting(table) => {
                inside_native(py, table, &CoreCounting, span, category, &CountDecoder)
            }
            InsideStorage::Viterbi(table) => {
                inside_native(py, table, &CoreViterbi, span, category, &FloatDecoder)
            }
            InsideStorage::LogInside(table) => {
                inside_native(py, table, &CoreLogInside, span, category, &FloatDecoder)
            }
            InsideStorage::Boolean(table) => {
                inside_native(py, table, &CoreBoolean, span, category, &BooleanDecoder)
            }
            InsideStorage::KBest { table, semiring } => {
                inside_native(py, table, semiring, span, category, &KBestDecoder)
            }
            InsideStorage::Expectation { table, semiring } => inside_native(
                py,
                table,
                semiring,
                span,
                category,
                &ExpectationDecoder {
                    dim: semiring.dim().get(),
                },
            ),
            InsideStorage::Python { table, semiring } => {
                let weight = table
                    .inside(span, category, semiring)
                    .map_err(python_table_error)?;
                PythonDecoder.encode(py, &weight)
            }
        }
    }

    #[pyo3(signature = (active_arities, unary_depth, threads, program, resolver, *, scoring))]
    fn sweep(
        &self,
        active_arities: Vec<usize>,
        unary_depth: usize,
        threads: usize,
        program: &Bound<'_, PyChartProgram>,
        resolver: Py<PyAny>,
        scoring: &Bound<'_, PyAny>,
    ) -> PyResult<PyClosure> {
        Python::attach(|py| {
            let active_arities = active_arities.into_boxed_slice();
            let program = program.borrow();
            let resolver = resolver.into_bound(py);
            let scoring = scoring_mode(scoring)?;
            let threads = NonZeroUsize::new(threads).ok_or_else(|| {
                PyValueError::new_err("a chart thread count is at least one, not 0")
            })?;
            let boundary = SweepBoundary {
                py,
                active_arities: &active_arities,
                unary_depth,
                threads,
                program: &program,
                resolver,
                scoring,
            };
            let mut inner = self.inner()?;
            let closure = match &mut *inner {
                InsideStorage::Counting(table) => {
                    run_native_sweep(table, &CoreCounting, boundary, CountDecoder)?
                }
                InsideStorage::Viterbi(table) => {
                    run_native_sweep(table, &CoreViterbi, boundary, FloatDecoder)?
                }
                InsideStorage::LogInside(table) => {
                    run_native_sweep(table, &CoreLogInside, boundary, FloatDecoder)?
                }
                InsideStorage::Boolean(table) => {
                    run_native_sweep(table, &CoreBoolean, boundary, BooleanDecoder)?
                }
                InsideStorage::KBest { table, semiring } => {
                    run_native_sweep(table, semiring, boundary, KBestDecoder)?
                }
                InsideStorage::Expectation { table, semiring } => {
                    let dim = semiring.dim().get();
                    run_native_sweep(table, semiring, boundary, ExpectationDecoder { dim })?
                }
                InsideStorage::Python { table, semiring } => {
                    let mut resolver = DrawingResolver::new(
                        &boundary.program.core,
                        PythonHost {
                            py: boundary.py,
                            resolver: boundary.resolver,
                            decoder: PythonDecoder,
                        },
                        boundary.scoring,
                        semiring,
                    );
                    table
                        .sweep(
                            boundary.active_arities,
                            boundary.unary_depth,
                            CellExecution::Attached,
                            semiring,
                            &mut resolver,
                        )
                        .map_err(python_sweep_error)?
                }
            };
            drop(inner);
            Ok(closure_tuple(closure))
        })
    }

    fn __repr__(&self) -> PyResult<String> {
        let inner = self.inner()?;
        let representation = format!(
            "InsideTable(tokens={}, categories={})",
            inner.tokens(),
            inner.num_categories()
        );
        drop(inner);
        Ok(representation)
    }
}

#[derive(Clone)]
struct PythonWeight(Arc<Py<PyAny>>);

impl fmt::Debug for PythonWeight {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("PythonWeight(..)")
    }
}

struct PythonSemiring {
    inner: Py<PyAny>,
    zero: PythonWeight,
    one: PythonWeight,
}

impl PythonSemiring {
    fn new(semiring: &Bound<'_, PyAny>) -> PyResult<Self> {
        let zero = PythonWeight(Arc::new(semiring.getattr("zero")?.unbind()));
        let one = PythonWeight(Arc::new(semiring.getattr("one")?.unbind()));
        Ok(Self {
            inner: semiring.clone().unbind(),
            zero,
            one,
        })
    }

    fn clone_weight(weight: &PythonWeight) -> PythonWeight {
        Python::attach(|py| PythonWeight(Arc::new(weight.0.as_ref().clone_ref(py))))
    }
}

impl Semiring for PythonSemiring {
    type Weight = PythonWeight;
    type Error = PyErr;

    fn zero(&self) -> Self::Weight {
        Self::clone_weight(&self.zero)
    }

    fn one(&self) -> Self::Weight {
        Self::clone_weight(&self.one)
    }

    fn plus_assign(
        &self,
        accumulator: &mut Self::Weight,
        weight: &Self::Weight,
    ) -> Result<(), Self::Error> {
        Python::attach(|py| {
            let summed = self.inner.bind(py).call_method1(
                "plus",
                (accumulator.0.as_ref().bind(py), weight.0.as_ref().bind(py)),
            )?;
            *accumulator = PythonWeight(Arc::new(summed.unbind()));
            Ok(())
        })
    }

    fn times(
        &self,
        left: &Self::Weight,
        right: &Self::Weight,
    ) -> Result<Self::Weight, Self::Error> {
        Python::attach(|py| {
            let product = self.inner.bind(py).call_method1(
                "times",
                (left.0.as_ref().bind(py), right.0.as_ref().bind(py)),
            )?;
            Ok(PythonWeight(Arc::new(product.unbind())))
        })
    }
}

enum InsideStorage {
    Counting(CoreInsideTable<Count>),
    Viterbi(CoreInsideTable<f64>),
    LogInside(CoreInsideTable<f64>),
    Boolean(CoreInsideTable<bool>),
    KBest {
        table: CoreInsideTable<Box<[f64]>>,
        semiring: CoreKBest,
    },
    Expectation {
        table: CoreInsideTable<ExpectedWeight>,
        semiring: CoreExpectation,
    },
    Python {
        table: CoreInsideTable<PythonWeight>,
        semiring: PythonSemiring,
    },
}

impl InsideStorage {
    const fn tokens(&self) -> usize {
        match self {
            Self::Counting(table) => table.tokens(),
            Self::Viterbi(table) | Self::LogInside(table) => table.tokens(),
            Self::Boolean(table) => table.tokens(),
            Self::KBest { table, .. } => table.tokens(),
            Self::Expectation { table, .. } => table.tokens(),
            Self::Python { table, .. } => table.tokens(),
        }
    }

    const fn truncated(&self) -> bool {
        match self {
            Self::Counting(table) => table.truncated(),
            Self::Viterbi(table) | Self::LogInside(table) => table.truncated(),
            Self::Boolean(table) => table.truncated(),
            Self::KBest { table, .. } => table.truncated(),
            Self::Expectation { table, .. } => table.truncated(),
            Self::Python { table, .. } => table.truncated(),
        }
    }

    fn num_categories(&self) -> usize {
        match self {
            Self::Counting(table) => table.num_categories(),
            Self::Viterbi(table) | Self::LogInside(table) => table.num_categories(),
            Self::Boolean(table) => table.num_categories(),
            Self::KBest { table, .. } => table.num_categories(),
            Self::Expectation { table, .. } => table.num_categories(),
            Self::Python { table, .. } => table.num_categories(),
        }
    }

    const fn mark_truncated(&mut self) {
        match self {
            Self::Counting(table) => table.mark_truncated(),
            Self::Viterbi(table) | Self::LogInside(table) => table.mark_truncated(),
            Self::Boolean(table) => table.mark_truncated(),
            Self::KBest { table, .. } => table.mark_truncated(),
            Self::Expectation { table, .. } => table.mark_truncated(),
            Self::Python { table, .. } => table.mark_truncated(),
        }
    }

    fn items_at(&self, span: Span) -> Result<Vec<ItemId>, TableError> {
        match self {
            Self::Counting(table) => table.items_at(span),
            Self::Viterbi(table) | Self::LogInside(table) => table.items_at(span),
            Self::Boolean(table) => table.items_at(span),
            Self::KBest { table, .. } => table.items_at(span),
            Self::Expectation { table, .. } => table.items_at(span),
            Self::Python { table, .. } => table.items_at(span),
        }
    }

    fn restrict(&mut self, span: Span, keep: &[ItemId]) -> Result<(), TableError> {
        match self {
            Self::Counting(table) => table.restrict(span, keep),
            Self::Viterbi(table) | Self::LogInside(table) => table.restrict(span, keep),
            Self::Boolean(table) => table.restrict(span, keep),
            Self::KBest { table, .. } => table.restrict(span, keep),
            Self::Expectation { table, .. } => table.restrict(span, keep),
            Self::Python { table, .. } => table.restrict(span, keep),
        }
    }
}

enum ForestStorage {
    Counting(CoreForestTable<u32, Count>),
    Viterbi(CoreForestTable<u32, f64>),
    LogInside(CoreForestTable<u32, f64>),
    Boolean(CoreForestTable<u32, bool>),
    KBest {
        table: CoreForestTable<u32, Box<[f64]>>,
        semiring: CoreKBest,
    },
    Expectation {
        table: CoreForestTable<u32, ExpectedWeight>,
        semiring: CoreExpectation,
    },
    Python {
        table: CoreForestTable<u32, PythonWeight>,
        semiring: PythonSemiring,
    },
}

impl ForestStorage {
    const fn tokens(&self) -> usize {
        match self {
            Self::Counting(table) => table.tokens(),
            Self::Viterbi(table) | Self::LogInside(table) => table.tokens(),
            Self::Boolean(table) => table.tokens(),
            Self::KBest { table, .. } => table.tokens(),
            Self::Expectation { table, .. } => table.tokens(),
            Self::Python { table, .. } => table.tokens(),
        }
    }

    fn num_items(&self) -> usize {
        match self {
            Self::Counting(table) => table.num_items(),
            Self::Viterbi(table) | Self::LogInside(table) => table.num_items(),
            Self::Boolean(table) => table.num_items(),
            Self::KBest { table, .. } => table.num_items(),
            Self::Expectation { table, .. } => table.num_items(),
            Self::Python { table, .. } => table.num_items(),
        }
    }

    const fn num_edges(&self) -> usize {
        match self {
            Self::Counting(table) => table.num_edges(),
            Self::Viterbi(table) | Self::LogInside(table) => table.num_edges(),
            Self::Boolean(table) => table.num_edges(),
            Self::KBest { table, .. } => table.num_edges(),
            Self::Expectation { table, .. } => table.num_edges(),
            Self::Python { table, .. } => table.num_edges(),
        }
    }

    const fn truncated(&self) -> bool {
        match self {
            Self::Counting(table) => table.truncated(),
            Self::Viterbi(table) | Self::LogInside(table) => table.truncated(),
            Self::Boolean(table) => table.truncated(),
            Self::KBest { table, .. } => table.truncated(),
            Self::Expectation { table, .. } => table.truncated(),
            Self::Python { table, .. } => table.truncated(),
        }
    }

    const fn mark_truncated(&mut self) {
        match self {
            Self::Counting(table) => table.mark_truncated(),
            Self::Viterbi(table) | Self::LogInside(table) => table.mark_truncated(),
            Self::Boolean(table) => table.mark_truncated(),
            Self::KBest { table, .. } => table.mark_truncated(),
            Self::Expectation { table, .. } => table.mark_truncated(),
            Self::Python { table, .. } => table.mark_truncated(),
        }
    }

    fn items_at(&self, span: Span) -> Result<Vec<CoreForestKey>, TableError> {
        match self {
            Self::Counting(table) => table.items_at(span),
            Self::Viterbi(table) | Self::LogInside(table) => table.items_at(span),
            Self::Boolean(table) => table.items_at(span),
            Self::KBest { table, .. } => table.items_at(span),
            Self::Expectation { table, .. } => table.items_at(span),
            Self::Python { table, .. } => table.items_at(span),
        }
    }

    fn contains(&self, key: CoreForestKey) -> bool {
        match self {
            Self::Counting(table) => table.contains(key),
            Self::Viterbi(table) | Self::LogInside(table) => table.contains(key),
            Self::Boolean(table) => table.contains(key),
            Self::KBest { table, .. } => table.contains(key),
            Self::Expectation { table, .. } => table.contains(key),
            Self::Python { table, .. } => table.contains(key),
        }
    }
}

trait WeightDecoder<W> {
    fn decode(&self, object: &Bound<'_, PyAny>) -> PyResult<W>;
}

trait WeightEncoder<W> {
    fn encode(&self, py: Python<'_>, weight: &W) -> PyResult<Py<PyAny>>;
}

struct CountDecoder;

impl WeightDecoder<Count> for CountDecoder {
    fn decode(&self, object: &Bound<'_, PyAny>) -> PyResult<Count> {
        object.extract::<BigUint>().map(Count::from)
    }
}

impl WeightEncoder<Count> for CountDecoder {
    fn encode(&self, py: Python<'_>, weight: &Count) -> PyResult<Py<PyAny>> {
        weight.clone().into_big_uint().into_py_any(py)
    }
}

struct FloatDecoder;

impl WeightDecoder<f64> for FloatDecoder {
    fn decode(&self, object: &Bound<'_, PyAny>) -> PyResult<f64> {
        object.extract()
    }
}

impl WeightEncoder<f64> for FloatDecoder {
    fn encode(&self, py: Python<'_>, weight: &f64) -> PyResult<Py<PyAny>> {
        (*weight).into_py_any(py)
    }
}

struct BooleanDecoder;

impl WeightDecoder<bool> for BooleanDecoder {
    fn decode(&self, object: &Bound<'_, PyAny>) -> PyResult<bool> {
        object.extract()
    }
}

impl WeightEncoder<bool> for BooleanDecoder {
    fn encode(&self, py: Python<'_>, weight: &bool) -> PyResult<Py<PyAny>> {
        (*weight).into_py_any(py)
    }
}

struct KBestDecoder;

impl WeightDecoder<Box<[f64]>> for KBestDecoder {
    fn decode(&self, object: &Bound<'_, PyAny>) -> PyResult<Box<[f64]>> {
        Ok(object.extract::<Vec<f64>>()?.into_boxed_slice())
    }
}

impl WeightEncoder<Box<[f64]>> for KBestDecoder {
    fn encode(&self, py: Python<'_>, weight: &Box<[f64]>) -> PyResult<Py<PyAny>> {
        float_tuple(py, weight)
    }
}

struct ExpectationDecoder {
    dim: usize,
}

impl WeightDecoder<ExpectedWeight> for ExpectationDecoder {
    fn decode(&self, object: &Bound<'_, PyAny>) -> PyResult<ExpectedWeight> {
        let (probability, coordinates) = object.extract::<(f64, Vec<f64>)>()?;

        if coordinates.len() != self.dim {
            return Err(PyValueError::new_err(format!(
                "an expectation weight has {} coordinates, not {}",
                coordinates.len(),
                self.dim
            )));
        }
        Ok((probability, coordinates.into_boxed_slice()))
    }
}

impl WeightEncoder<ExpectedWeight> for ExpectationDecoder {
    fn encode(&self, py: Python<'_>, weight: &ExpectedWeight) -> PyResult<Py<PyAny>> {
        expected_object(py, weight)
    }
}

struct PythonDecoder;

impl WeightDecoder<PythonWeight> for PythonDecoder {
    fn decode(&self, object: &Bound<'_, PyAny>) -> PyResult<PythonWeight> {
        Ok(PythonWeight(Arc::new(object.clone().unbind())))
    }
}

impl WeightEncoder<PythonWeight> for PythonDecoder {
    fn encode(&self, py: Python<'_>, weight: &PythonWeight) -> PyResult<Py<PyAny>> {
        Ok(python_object(py, weight))
    }
}

fn table_error(error: &TableError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn native_table_error(error: SweepError<TableError, Infallible>) -> PyErr {
    match error {
        SweepError::Resolver(error) | SweepError::Table(error) => table_error(&error),
        SweepError::Semiring(error) => match error {},
    }
}

fn python_table_error(error: SweepError<TableError, PyErr>) -> PyErr {
    match error {
        SweepError::Resolver(error) | SweepError::Table(error) => table_error(&error),
        SweepError::Semiring(error) => error,
    }
}

fn drawn_sweep_error(error: SweepError<DrawError<Registry>, Infallible>) -> PyErr {
    match error {
        SweepError::Resolver(error) => draw_error(error),
        SweepError::Semiring(error) => match error {},
        SweepError::Table(error) => table_error(&error),
    }
}

fn python_sweep_error(error: SweepError<DrawError<Registry>, PyErr>) -> PyErr {
    match error {
        SweepError::Resolver(error) => draw_error(error),
        SweepError::Semiring(error) => error,
        SweepError::Table(error) => table_error(&error),
    }
}

fn closure_tuple(closure: UnaryClosure) -> PyClosure {
    let edges = closure
        .edges
        .into_iter()
        .map(|(child, step)| {
            (
                (child.category, child.tag),
                step.rule,
                (step.output, step.tag),
            )
        })
        .collect();
    let concluded = closure
        .concluded
        .into_iter()
        .map(|item| (item.category, item.tag))
        .collect();
    (edges, concluded, closure.truncated)
}

const fn dense_item(item: (u32, u8)) -> ItemId {
    ItemId {
        category: item.0,
        tag: item.1,
    }
}

fn dense_items(items: Vec<(u32, u8)>) -> Vec<ItemId> {
    items.into_iter().map(dense_item).collect()
}

const fn forest_key(key: PyForestKey) -> CoreForestKey {
    CoreForestKey {
        span: key.0,
        item: dense_item(key.1),
    }
}

const fn forest_key_tuple(key: CoreForestKey) -> PyForestKey {
    (key.span, (key.item.category, key.item.tag))
}

fn weighted_items_object<W, D>(
    py: Python<'_>,
    items: &[(ItemId, W)],
    encoder: &D,
) -> PyResult<Vec<(u32, u8, Py<PyAny>)>>
where
    D: WeightEncoder<W>,
{
    items
        .iter()
        .map(|(item, weight)| Ok((item.category, item.tag, encoder.encode(py, weight)?)))
        .collect()
}

fn float_tuple(py: Python<'_>, values: &[f64]) -> PyResult<Py<PyAny>> {
    Ok(PyTuple::new(py, values.iter().copied())?
        .into_any()
        .unbind())
}

fn expected_object(py: Python<'_>, weight: &ExpectedWeight) -> PyResult<Py<PyAny>> {
    let coordinates = PyTuple::new(py, weight.1.iter().copied())?;
    (weight.0, coordinates).into_py_any(py)
}

fn python_object(py: Python<'_>, weight: &PythonWeight) -> Py<PyAny> {
    weight.0.as_ref().clone_ref(py)
}

struct SweepBoundary<'a, 'py> {
    py: Python<'py>,
    active_arities: &'a [usize],
    unary_depth: usize,
    threads: NonZeroUsize,
    program: &'a PyChartProgram,
    resolver: Bound<'py, PyAny>,
    scoring: EdgeScoring,
}

fn run_native_sweep<W, S, D>(
    table: &mut CoreInsideTable<W>,
    semiring: &S,
    boundary: SweepBoundary<'_, '_>,
    decoder: D,
) -> PyResult<UnaryClosure>
where
    W: Clone + Send + Sync,
    S: Semiring<Weight = W, Error = Infallible> + Sync,
    D: WeightDecoder<W>,
{
    let mut resolver = DrawingResolver::new(
        &boundary.program.core,
        PythonHost {
            py: boundary.py,
            resolver: boundary.resolver,
            decoder,
        },
        boundary.scoring,
        semiring,
    );
    table
        .sweep(
            boundary.active_arities,
            boundary.unary_depth,
            CellExecution::Parallel(boundary.threads),
            semiring,
            &mut resolver,
        )
        .map_err(drawn_sweep_error)
}

fn add_native<W, S, D>(
    table: &mut CoreInsideTable<W>,
    semiring: &S,
    span: Span,
    item: ItemId,
    weight: &Bound<'_, PyAny>,
    decoder: &D,
) -> PyResult<()>
where
    W: Clone,
    S: Semiring<Weight = W, Error = Infallible>,
    D: WeightDecoder<W>,
{
    table
        .add(span, item, decoder.decode(weight)?, semiring)
        .map_err(native_table_error)
}

fn tagged_objects<W, D>(
    py: Python<'_>,
    table: &CoreInsideTable<W>,
    span: Span,
    encoder: &D,
) -> PyResult<Vec<(u32, u8, Py<PyAny>)>>
where
    W: Clone,
    D: WeightEncoder<W>,
{
    table
        .tagged_at(span)
        .map_err(|error| table_error(&error))?
        .into_iter()
        .map(|(item, weight)| Ok((item.category, item.tag, encoder.encode(py, &weight)?)))
        .collect()
}

fn categories_native<W, S, D>(
    py: Python<'_>,
    table: &CoreInsideTable<W>,
    semiring: &S,
    span: Span,
    encoder: &D,
) -> PyResult<Vec<(u32, Py<PyAny>)>>
where
    W: Clone,
    S: Semiring<Weight = W, Error = Infallible>,
    D: WeightEncoder<W>,
{
    table
        .categories_at(span, semiring)
        .map_err(native_table_error)?
        .into_iter()
        .map(|(category, weight)| Ok((category, encoder.encode(py, &weight)?)))
        .collect()
}

fn tagged_native<W, S, D>(
    py: Python<'_>,
    table: &CoreInsideTable<W>,
    semiring: &S,
    span: Span,
    item: ItemId,
    encoder: &D,
) -> PyResult<Py<PyAny>>
where
    W: Clone,
    S: Semiring<Weight = W, Error = Infallible>,
    D: WeightEncoder<W>,
{
    let weight = table
        .tagged(span, item)
        .map_err(|error| table_error(&error))?
        .unwrap_or_else(|| semiring.zero());
    encoder.encode(py, &weight)
}

fn inside_native<W, S, D>(
    py: Python<'_>,
    table: &CoreInsideTable<W>,
    semiring: &S,
    span: Span,
    category: u32,
    encoder: &D,
) -> PyResult<Py<PyAny>>
where
    W: Clone,
    S: Semiring<Weight = W, Error = Infallible>,
    D: WeightEncoder<W>,
{
    let weight = table
        .inside(span, category, semiring)
        .map_err(native_table_error)?;
    encoder.encode(py, &weight)
}

struct ForestEdgeInput<'py> {
    label: u32,
    children: Vec<CoreForestKey>,
    output: u32,
    weight: Bound<'py, PyAny>,
    split: Option<usize>,
    order: usize,
}

struct ForestProductInput<'py> {
    label: u32,
    axes: Vec<Vec<CoreForestKey>>,
    output: u32,
    weight: Bound<'py, PyAny>,
    split: Option<usize>,
    order: usize,
}

fn add_forest_native<W, S, D>(
    table: &mut CoreForestTable<u32, W>,
    semiring: &S,
    key: CoreForestKey,
    edge: ForestEdgeInput<'_>,
    decoder: &D,
) -> PyResult<()>
where
    W: Clone,
    S: Semiring<Weight = W, Error = Infallible>,
    D: WeightDecoder<W>,
{
    table
        .add(
            key,
            ForestEdge {
                label: edge.label,
                children: edge.children,
                output: edge.output,
                weight: decoder.decode(&edge.weight)?,
                split: edge.split,
                order: edge.order,
            },
            semiring,
        )
        .map_err(native_table_error)
}

fn add_forest_product_native<W, S, D>(
    table: &mut CoreForestTable<u32, W>,
    semiring: &S,
    key: CoreForestKey,
    product: ForestProductInput<'_>,
    decoder: &D,
) -> PyResult<()>
where
    W: Clone,
    S: Semiring<Weight = W, Error = Infallible>,
    D: WeightDecoder<W>,
{
    table
        .add_product(
            key,
            ProductAlternative {
                label: product.label,
                axes: product.axes,
                output: product.output,
                weight: decoder.decode(&product.weight)?,
                split: product.split,
                order: product.order,
            },
            semiring,
        )
        .map_err(native_table_error)
}

fn forest_edges_object<W, D>(
    py: Python<'_>,
    table: &CoreForestTable<u32, W>,
    key: CoreForestKey,
    encoder: &D,
) -> PyResult<Vec<PyForestEdge>>
where
    W: Clone,
    D: WeightEncoder<W>,
{
    if !table.contains(key) {
        return Ok(Vec::new());
    }

    table
        .edges(key)
        .map_err(|error| table_error(&error))?
        .into_iter()
        .map(|edge| {
            Ok((
                edge.label,
                edge.children.into_iter().map(forest_key_tuple).collect(),
                edge.output,
                encoder.encode(py, &edge.weight)?,
                edge.split,
                edge.order,
            ))
        })
        .collect()
}

fn forest_products_object<W, D>(
    py: Python<'_>,
    table: &CoreForestTable<u32, W>,
    key: CoreForestKey,
    encoder: &D,
) -> PyResult<Vec<PyForestProduct>>
where
    W: Clone,
    D: WeightEncoder<W>,
{
    if !table.contains(key) {
        return Ok(Vec::new());
    }

    table
        .products(key)
        .map_err(|error| table_error(&error))?
        .into_iter()
        .map(|product| {
            Ok((
                product.label,
                product
                    .axes
                    .into_iter()
                    .map(|axis| axis.into_iter().map(forest_key_tuple).collect())
                    .collect(),
                product.output,
                encoder.encode(py, &product.weight)?,
                product.split,
                product.order,
            ))
        })
        .collect()
}

fn forest_inside_object<W, S, D>(
    py: Python<'_>,
    table: &CoreForestTable<u32, W>,
    semiring: &S,
    key: CoreForestKey,
    encoder: &D,
) -> PyResult<Py<PyAny>>
where
    W: Clone,
    S: Semiring<Weight = W>,
    D: WeightEncoder<W>,
{
    encoder.encode(py, &table.inside(key, semiring))
}

fn forest_categories_native<W, S, D>(
    py: Python<'_>,
    table: &CoreForestTable<u32, W>,
    semiring: &S,
    span: Span,
    encoder: &D,
) -> PyResult<Vec<(u32, Py<PyAny>)>>
where
    W: Clone,
    S: Semiring<Weight = W, Error = Infallible>,
    D: WeightEncoder<W>,
{
    table
        .categories_at(span, semiring)
        .map_err(native_table_error)?
        .into_iter()
        .map(|(category, weight)| Ok((category, encoder.encode(py, &weight)?)))
        .collect()
}

fn restrict_forest_native<W, S>(
    table: &mut CoreForestTable<u32, W>,
    semiring: &S,
    span: Span,
    keep: &[CoreForestKey],
    truncates: bool,
) -> PyResult<()>
where
    W: Clone,
    S: Semiring<Weight = W, Error = Infallible>,
{
    table
        .restrict(span, keep, semiring, truncates)
        .map_err(native_table_error)
}

fn recompute_forest_native<W, S>(
    table: &mut CoreForestTable<u32, W>,
    semiring: &S,
    keys: &[CoreForestKey],
) -> PyResult<()>
where
    W: Clone,
    S: Semiring<Weight = W, Error = Infallible>,
{
    table
        .recompute_inside(keys, semiring)
        .map_err(|error| match error {})
}

fn run_native_forest_sweep<W, S, D>(
    table: &mut CoreForestTable<u32, W>,
    semiring: &S,
    boundary: SweepBoundary<'_, '_>,
    decoder: D,
) -> PyResult<()>
where
    W: Clone + Send + Sync,
    S: Semiring<Weight = W, Error = Infallible> + Sync,
    D: WeightDecoder<W> + WeightEncoder<W>,
{
    let mut resolver = DrawingResolver::new(
        &boundary.program.core,
        PythonHost {
            py: boundary.py,
            resolver: boundary.resolver,
            decoder,
        },
        boundary.scoring,
        semiring,
    );
    table
        .sweep(
            boundary.active_arities,
            boundary.unary_depth,
            CellExecution::Parallel(boundary.threads),
            semiring,
            &mut resolver,
        )
        .map_err(drawn_sweep_error)
}

#[pymethods]
impl PyForestTable {
    #[new]
    fn new(
        tokens: usize,
        semiring: &Bound<'_, PyAny>,
        lowering: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let inner = if lowering.is_instance_of::<PyCounting>() {
            ForestStorage::Counting(CoreForestTable::new(tokens))
        } else if lowering.is_instance_of::<PyViterbi>() {
            ForestStorage::Viterbi(CoreForestTable::new(tokens))
        } else if lowering.is_instance_of::<PyLogInside>() {
            ForestStorage::LogInside(CoreForestTable::new(tokens))
        } else if lowering.is_instance_of::<PyBoolean>() {
            ForestStorage::Boolean(CoreForestTable::new(tokens))
        } else if lowering.is_instance_of::<PyKBest>() {
            let k = lowering.getattr("k")?.extract::<usize>()?;
            let k = NonZeroUsize::new(k)
                .ok_or_else(|| PyValueError::new_err("a k-best width is at least one, not 0"))?;
            ForestStorage::KBest {
                table: CoreForestTable::new(tokens),
                semiring: CoreKBest::new(k),
            }
        } else if lowering.is_instance_of::<PyExpectation>() {
            let dim = lowering.getattr("dim")?.extract::<usize>()?;
            let dim = NonZeroUsize::new(dim).ok_or_else(|| {
                PyValueError::new_err("an expectation has at least one coordinate, not 0")
            })?;
            ForestStorage::Expectation {
                table: CoreForestTable::new(tokens),
                semiring: CoreExpectation::new(dim),
            }
        } else if lowering.is_instance_of::<PythonCarrier>() {
            ForestStorage::Python {
                table: CoreForestTable::new(tokens),
                semiring: PythonSemiring::new(semiring)?,
            }
        } else {
            return Err(PyValueError::new_err(
                "a semiring lowering must select a native carrier or PythonCarrier",
            ));
        };
        Ok(Self {
            inner: Mutex::new(inner),
        })
    }

    #[getter]
    fn tokens(&self) -> PyResult<usize> {
        Ok(self.inner()?.tokens())
    }

    #[getter]
    fn num_items(&self) -> PyResult<usize> {
        Ok(self.inner()?.num_items())
    }

    #[getter]
    fn num_edges(&self) -> PyResult<usize> {
        Ok(self.inner()?.num_edges())
    }

    #[getter]
    fn truncated(&self) -> PyResult<bool> {
        Ok(self.inner()?.truncated())
    }

    fn mark_truncated(&self) -> PyResult<()> {
        self.inner()?.mark_truncated();
        Ok(())
    }

    fn add(&self, py: Python<'_>, key: PyForestKey, edge: PyForestEdge) -> PyResult<()> {
        let (label, children, output, weight, split, order) = edge;
        let key = forest_key(key);
        let children = children.into_iter().map(forest_key).collect();
        let edge = ForestEdgeInput {
            label,
            children,
            output,
            weight: weight.into_bound(py),
            split,
            order,
        };
        let mut inner = self.inner()?;

        match &mut *inner {
            ForestStorage::Counting(table) => {
                add_forest_native(table, &CoreCounting, key, edge, &CountDecoder)
            }
            ForestStorage::Viterbi(table) => {
                add_forest_native(table, &CoreViterbi, key, edge, &FloatDecoder)
            }
            ForestStorage::LogInside(table) => {
                add_forest_native(table, &CoreLogInside, key, edge, &FloatDecoder)
            }
            ForestStorage::Boolean(table) => {
                add_forest_native(table, &CoreBoolean, key, edge, &BooleanDecoder)
            }
            ForestStorage::KBest { table, semiring } => {
                add_forest_native(table, semiring, key, edge, &KBestDecoder)
            }
            ForestStorage::Expectation { table, semiring } => add_forest_native(
                table,
                semiring,
                key,
                edge,
                &ExpectationDecoder {
                    dim: semiring.dim().get(),
                },
            ),
            ForestStorage::Python { table, semiring } => table
                .add(
                    key,
                    ForestEdge {
                        label: edge.label,
                        children: edge.children,
                        output: edge.output,
                        weight: PythonDecoder.decode(&edge.weight)?,
                        split: edge.split,
                        order: edge.order,
                    },
                    semiring,
                )
                .map_err(python_table_error),
        }
    }

    fn add_product(
        &self,
        py: Python<'_>,
        key: PyForestKey,
        product: PyForestProduct,
    ) -> PyResult<()> {
        let key = forest_key(key);
        let (label, axes, output, weight, split, order) = product;
        let product = ForestProductInput {
            label,
            axes: axes
                .into_iter()
                .map(|axis| axis.into_iter().map(forest_key).collect())
                .collect(),
            output,
            weight: weight.into_bound(py),
            split,
            order,
        };
        let mut inner = self.inner()?;

        match &mut *inner {
            ForestStorage::Counting(table) => {
                add_forest_product_native(table, &CoreCounting, key, product, &CountDecoder)
            }
            ForestStorage::Viterbi(table) => {
                add_forest_product_native(table, &CoreViterbi, key, product, &FloatDecoder)
            }
            ForestStorage::LogInside(table) => {
                add_forest_product_native(table, &CoreLogInside, key, product, &FloatDecoder)
            }
            ForestStorage::Boolean(table) => {
                add_forest_product_native(table, &CoreBoolean, key, product, &BooleanDecoder)
            }
            ForestStorage::KBest { table, semiring } => {
                add_forest_product_native(table, semiring, key, product, &KBestDecoder)
            }
            ForestStorage::Expectation { table, semiring } => add_forest_product_native(
                table,
                semiring,
                key,
                product,
                &ExpectationDecoder {
                    dim: semiring.dim().get(),
                },
            ),
            ForestStorage::Python { table, semiring } => table
                .add_product(
                    key,
                    ProductAlternative {
                        label: product.label,
                        axes: product.axes,
                        output: product.output,
                        weight: PythonDecoder.decode(&product.weight)?,
                        split: product.split,
                        order: product.order,
                    },
                    semiring,
                )
                .map_err(python_table_error),
        }
    }

    fn items_at(&self, span: Span) -> PyResult<Vec<(u32, u8)>> {
        self.inner()?
            .items_at(span)
            .map_err(|error| table_error(&error))
            .map(|items| {
                items
                    .into_iter()
                    .map(|key| (key.item.category, key.item.tag))
                    .collect()
            })
    }

    fn contains(&self, key: PyForestKey) -> PyResult<bool> {
        Ok(self.inner()?.contains(forest_key(key)))
    }

    fn edges(&self, py: Python<'_>, key: PyForestKey) -> PyResult<Vec<PyForestEdge>> {
        let key = forest_key(key);
        let inner = self.inner()?;

        match &*inner {
            ForestStorage::Counting(table) => forest_edges_object(py, table, key, &CountDecoder),
            ForestStorage::Viterbi(table) | ForestStorage::LogInside(table) => {
                forest_edges_object(py, table, key, &FloatDecoder)
            }
            ForestStorage::Boolean(table) => forest_edges_object(py, table, key, &BooleanDecoder),
            ForestStorage::KBest { table, .. } => {
                forest_edges_object(py, table, key, &KBestDecoder)
            }
            ForestStorage::Expectation { table, semiring } => forest_edges_object(
                py,
                table,
                key,
                &ExpectationDecoder {
                    dim: semiring.dim().get(),
                },
            ),
            ForestStorage::Python { table, .. } => {
                forest_edges_object(py, table, key, &PythonDecoder)
            }
        }
    }

    fn products(&self, py: Python<'_>, key: PyForestKey) -> PyResult<Vec<PyForestProduct>> {
        let key = forest_key(key);
        let inner = self.inner()?;

        match &*inner {
            ForestStorage::Counting(table) => forest_products_object(py, table, key, &CountDecoder),
            ForestStorage::Viterbi(table) | ForestStorage::LogInside(table) => {
                forest_products_object(py, table, key, &FloatDecoder)
            }
            ForestStorage::Boolean(table) => {
                forest_products_object(py, table, key, &BooleanDecoder)
            }
            ForestStorage::KBest { table, .. } => {
                forest_products_object(py, table, key, &KBestDecoder)
            }
            ForestStorage::Expectation { table, semiring } => forest_products_object(
                py,
                table,
                key,
                &ExpectationDecoder {
                    dim: semiring.dim().get(),
                },
            ),
            ForestStorage::Python { table, .. } => {
                forest_products_object(py, table, key, &PythonDecoder)
            }
        }
    }

    fn inside(&self, py: Python<'_>, key: PyForestKey) -> PyResult<Py<PyAny>> {
        let key = forest_key(key);
        let inner = self.inner()?;

        match &*inner {
            ForestStorage::Counting(table) => {
                forest_inside_object(py, table, &CoreCounting, key, &CountDecoder)
            }
            ForestStorage::Viterbi(table) => {
                forest_inside_object(py, table, &CoreViterbi, key, &FloatDecoder)
            }
            ForestStorage::LogInside(table) => {
                forest_inside_object(py, table, &CoreLogInside, key, &FloatDecoder)
            }
            ForestStorage::Boolean(table) => {
                forest_inside_object(py, table, &CoreBoolean, key, &BooleanDecoder)
            }
            ForestStorage::KBest { table, semiring } => {
                forest_inside_object(py, table, semiring, key, &KBestDecoder)
            }
            ForestStorage::Expectation { table, semiring } => forest_inside_object(
                py,
                table,
                semiring,
                key,
                &ExpectationDecoder {
                    dim: semiring.dim().get(),
                },
            ),
            ForestStorage::Python { table, semiring } => {
                forest_inside_object(py, table, semiring, key, &PythonDecoder)
            }
        }
    }

    fn categories_at(&self, py: Python<'_>, span: Span) -> PyResult<Vec<(u32, Py<PyAny>)>> {
        let inner = self.inner()?;

        match &*inner {
            ForestStorage::Counting(table) => {
                forest_categories_native(py, table, &CoreCounting, span, &CountDecoder)
            }
            ForestStorage::Viterbi(table) => {
                forest_categories_native(py, table, &CoreViterbi, span, &FloatDecoder)
            }
            ForestStorage::LogInside(table) => {
                forest_categories_native(py, table, &CoreLogInside, span, &FloatDecoder)
            }
            ForestStorage::Boolean(table) => {
                forest_categories_native(py, table, &CoreBoolean, span, &BooleanDecoder)
            }
            ForestStorage::KBest { table, semiring } => {
                forest_categories_native(py, table, semiring, span, &KBestDecoder)
            }
            ForestStorage::Expectation { table, semiring } => forest_categories_native(
                py,
                table,
                semiring,
                span,
                &ExpectationDecoder {
                    dim: semiring.dim().get(),
                },
            ),
            ForestStorage::Python { table, semiring } => table
                .categories_at(span, semiring)
                .map_err(python_table_error)?
                .into_iter()
                .map(|(category, weight)| Ok((category, PythonDecoder.encode(py, &weight)?)))
                .collect(),
        }
    }

    #[pyo3(signature = (span, keep, *, truncates=true))]
    fn restrict(&self, span: Span, keep: Vec<PyForestKey>, truncates: bool) -> PyResult<()> {
        let keep: Vec<CoreForestKey> = keep.into_iter().map(forest_key).collect();
        let mut inner = self.inner()?;

        match &mut *inner {
            ForestStorage::Counting(table) => {
                restrict_forest_native(table, &CoreCounting, span, &keep, truncates)
            }
            ForestStorage::Viterbi(table) => {
                restrict_forest_native(table, &CoreViterbi, span, &keep, truncates)
            }
            ForestStorage::LogInside(table) => {
                restrict_forest_native(table, &CoreLogInside, span, &keep, truncates)
            }
            ForestStorage::Boolean(table) => {
                restrict_forest_native(table, &CoreBoolean, span, &keep, truncates)
            }
            ForestStorage::KBest { table, semiring } => {
                restrict_forest_native(table, semiring, span, &keep, truncates)
            }
            ForestStorage::Expectation { table, semiring } => {
                restrict_forest_native(table, semiring, span, &keep, truncates)
            }
            ForestStorage::Python { table, semiring } => table
                .restrict(span, &keep, semiring, truncates)
                .map_err(python_table_error),
        }
    }

    fn recompute_inside(&self, keys: Vec<PyForestKey>) -> PyResult<()> {
        let keys: Vec<CoreForestKey> = keys.into_iter().map(forest_key).collect();
        let mut inner = self.inner()?;

        match &mut *inner {
            ForestStorage::Counting(table) => recompute_forest_native(table, &CoreCounting, &keys),
            ForestStorage::Viterbi(table) => recompute_forest_native(table, &CoreViterbi, &keys),
            ForestStorage::LogInside(table) => {
                recompute_forest_native(table, &CoreLogInside, &keys)
            }
            ForestStorage::Boolean(table) => recompute_forest_native(table, &CoreBoolean, &keys),
            ForestStorage::KBest { table, semiring } => {
                recompute_forest_native(table, semiring, &keys)
            }
            ForestStorage::Expectation { table, semiring } => {
                recompute_forest_native(table, semiring, &keys)
            }
            ForestStorage::Python { table, semiring } => table.recompute_inside(&keys, semiring),
        }
    }

    #[pyo3(signature = (active_arities, unary_depth, threads, program, resolver, *, scoring))]
    fn sweep(
        &self,
        active_arities: Vec<usize>,
        unary_depth: usize,
        threads: usize,
        program: &Bound<'_, PyChartProgram>,
        resolver: Py<PyAny>,
        scoring: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        Python::attach(|py| {
            let active_arities = active_arities.into_boxed_slice();
            let program = program.borrow();
            let resolver = resolver.into_bound(py);
            let scoring = scoring_mode(scoring)?;
            let threads = NonZeroUsize::new(threads).ok_or_else(|| {
                PyValueError::new_err("a chart thread count is at least one, not 0")
            })?;
            let boundary = SweepBoundary {
                py,
                active_arities: &active_arities,
                unary_depth,
                threads,
                program: &program,
                resolver,
                scoring,
            };
            let mut inner = self.inner()?;

            match &mut *inner {
                ForestStorage::Counting(table) => {
                    run_native_forest_sweep(table, &CoreCounting, boundary, CountDecoder)
                }
                ForestStorage::Viterbi(table) => {
                    run_native_forest_sweep(table, &CoreViterbi, boundary, FloatDecoder)
                }
                ForestStorage::LogInside(table) => {
                    run_native_forest_sweep(table, &CoreLogInside, boundary, FloatDecoder)
                }
                ForestStorage::Boolean(table) => {
                    run_native_forest_sweep(table, &CoreBoolean, boundary, BooleanDecoder)
                }
                ForestStorage::KBest { table, semiring } => {
                    run_native_forest_sweep(table, semiring, boundary, KBestDecoder)
                }
                ForestStorage::Expectation { table, semiring } => {
                    let dim = semiring.dim().get();
                    run_native_forest_sweep(table, semiring, boundary, ExpectationDecoder { dim })
                }
                ForestStorage::Python { table, semiring } => {
                    let mut resolver = DrawingResolver::new(
                        &boundary.program.core,
                        PythonHost {
                            py: boundary.py,
                            resolver: boundary.resolver,
                            decoder: PythonDecoder,
                        },
                        boundary.scoring,
                        semiring,
                    );
                    table
                        .sweep(
                            boundary.active_arities,
                            boundary.unary_depth,
                            CellExecution::Attached,
                            semiring,
                            &mut resolver,
                        )
                        .map_err(python_sweep_error)
                }
            }
        })
    }

    fn __repr__(&self) -> PyResult<String> {
        let inner = self.inner()?;
        let representation = format!(
            "ForestTable(tokens={}, items={}, edges={})",
            inner.tokens(),
            inner.num_items(),
            inner.num_edges()
        );
        drop(inner);
        Ok(representation)
    }
}

#[pymethods]
impl PyChartProgram {
    #[new]
    #[pyo3(signature = (
        rules,
        tags,
        rule_program,
        *,
        normalizes,
        category_measure=Vec::new(),
        category_bounds=(None, None)
    ))]
    fn new(
        rules: Vec<PyRuleRow>,
        tags: usize,
        rule_program: &Bound<'_, PyRuleProgram>,
        normalizes: bool,
        category_measure: Vec<(String, usize)>,
        category_bounds: (Option<usize>, Option<usize>),
    ) -> PyResult<Self> {
        let rules = rules
            .into_iter()
            .map(|(arity, order, complete, row)| {
                let decisions = row
                    .into_iter()
                    .map(|(children, decision, output)| {
                        let decision = match decision.as_str() {
                            "admit" => NormalDecision::Admit(output),
                            "refuse" => NormalDecision::Refuse,
                            "premises" => NormalDecision::Premises(output),
                            _ => {
                                return Err(PyValueError::new_err(format!(
                                    "unknown normal-form decision {decision:?}"
                                )));
                            }
                        };
                        Ok((children, decision))
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                Ok(DrawingRule {
                    arity,
                    order,
                    complete,
                    decisions,
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        let program = rule_program.borrow().core();
        let core = DrawingProgram::new(
            rules,
            tags,
            program,
            normalizes,
            category_measure,
            category_bounds,
        )
        .map_err(draw_error)?;
        Ok(Self { core })
    }

    /// Install the exact per-span category demands for one root-directed parse.
    ///
    /// `plan` is the sentence length in tokens, the unary closure depth per
    /// cell, and the most binding-and-candidate pairs one lexical refinement
    /// enumerates.
    fn configure_reachability(
        &self,
        owner: PyRef<'_, PyTerm>,
        representatives: PyRef<'_, PyRepresentativeKernel>,
        plan: (usize, usize, usize),
        roots: &Bound<'_, PyAny>,
        lexical: &Bound<'_, PyAny>,
        rules: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let (tokens, unary_depth, refinement_pairs) = plan;
        let owner_core = owner.core_term();
        let default_sort = owner_core.sort();
        let registry = owner.registry();
        drop(owner);
        let store = registry.store_for(&owner_core).ok_or_else(|| {
            PyValueError::new_err("the reachability owner has no live term store")
        })?;
        if let Some(representative_registry) = representatives.registry()
            && !Arc::ptr_eq(&registry, &representative_registry)
        {
            return Err(PyValueError::new_err(
                "a reachability representative kernel belongs to another term signature",
            ));
        }
        let representative_core = representatives.inner_core();
        drop(representatives);
        let mut representative_table = RepresentativeTable::default();

        let roots = roots
            .try_iter()?
            .map(|value| {
                let term = value?.extract::<PyRef<'_, PyTerm>>()?;
                if !Arc::ptr_eq(&registry, &term.registry()) {
                    return Err(PyValueError::new_err(
                        "a root reachability category belongs to another term signature",
                    ));
                }
                represent(
                    &representative_core,
                    &mut representative_table,
                    &term.core_term(),
                )
            })
            .collect::<PyResult<Vec<_>>>()?;

        let mut lexical_rows = HashMap::new();
        for row in lexical.try_iter()? {
            let row = row?;
            let span: Span = row.get_item(0)?.extract()?;
            let mut terms = Vec::new();
            let mut seen = HashSet::new();
            for value in row.get_item(1)?.try_iter()? {
                let term = value?.extract::<PyRef<'_, PyTerm>>()?;
                if !Arc::ptr_eq(&registry, &term.registry()) {
                    return Err(PyValueError::new_err(
                        "a lexical reachability category belongs to another term signature",
                    ));
                }
                let representative = represent(
                    &representative_core,
                    &mut representative_table,
                    &term.core_term(),
                )?;
                if seen.insert(representative.clone()) {
                    terms.push(representative);
                }
            }
            if lexical_rows.insert(span, terms).is_some() {
                return Err(PyValueError::new_err(format!(
                    "reachability repeats lexical span {span:?}"
                )));
            }
        }

        let reachability_rules = lower_reachability_rules(
            rules,
            &registry,
            &store,
            &representative_core,
            &mut representative_table,
        )?;
        self.core
            .configure_reachability(ReachabilityInputs {
                kernel: representative_core,
                table: representative_table,
                scope: registry,
                tokens,
                unary_depth,
                refinement_pairs,
                roots,
                lexical: lexical_rows,
                rules: reachability_rules,
                default_sort,
            })
            .map_err(draw_error)
    }

    /// Return whether one registered category remains root-reachable at a span.
    fn reachable(&self, span: Span, category: u32) -> PyResult<bool> {
        self.core.reachable(span, category).map_err(draw_error)
    }

    /// Return the dense identifier owned by one exact native category.
    fn identify(&self, term: PyRef<'_, PyTerm>) -> PyResult<Option<u32>> {
        let core = term.core_term();
        let registry = term.registry();
        drop(term);
        self.core.identify(core, registry).map_err(draw_error)
    }

    /// Return how many distinct exact categories the declared bounds refused.
    #[getter]
    fn refused_categories(&self) -> PyResult<usize> {
        self.core.refused_categories().map_err(draw_error)
    }

    /// Return one registered category's dense identifier when present.
    fn lookup(&self, term: PyRef<'_, PyTerm>) -> PyResult<Option<u32>> {
        let core = term.core_term();
        drop(term);
        self.core.lookup(&core).map_err(draw_error)
    }

    /// Return the exact category at one dense identifier.
    fn term(&self, py: Python<'_>, identifier: u32) -> PyResult<Py<PyTerm>> {
        let (term, registry) = self.core.term(identifier).map_err(draw_error)?;
        wrap(&registry, py, term)
    }

    /// Return candidate rule identifiers of one arity.
    fn rules_of_arity(&self, arity: usize) -> PyResult<Vec<u16>> {
        self.core.rules_of_arity(arity).map_err(draw_error)
    }

    /// Return memoized normalized outputs for one complete rule application.
    fn cached_outputs(&self, rule: u16, inputs: Vec<u32>) -> PyResult<Option<Vec<u32>>> {
        let inputs = inputs.into_boxed_slice();
        self.core.cached_outputs(rule, &inputs).map_err(draw_error)
    }

    /// Memoize normalized outputs for one complete rule application.
    fn remember_outputs(&self, rule: u16, inputs: Vec<u32>, outputs: Vec<u32>) -> PyResult<()> {
        self.core
            .remember_outputs(rule, inputs, outputs)
            .map_err(draw_error)
    }

    /// Enumerate indexed binary pairs in Cartesian order.
    fn pairs(&self, left: Vec<u32>, right: Vec<u32>) -> PyResult<Vec<(usize, usize, Vec<u16>)>> {
        let left = left.into_boxed_slice();
        let right = right.into_boxed_slice();
        Ok(self
            .core
            .pairs(&left, &right)
            .map_err(draw_error)?
            .into_iter()
            .map(|pair| (pair.left, pair.right, pair.candidates))
            .collect())
    }

    /// Draw one exact premise tuple, resolving a cache miss in Python.
    fn steps(
        &self,
        py: Python<'_>,
        inputs: Vec<u32>,
        tags: Vec<u8>,
        span: (usize, usize),
        candidates: Vec<u16>,
        resolver: Py<PyAny>,
    ) -> PyResult<Vec<(u16, u32, u8)>> {
        let candidates = candidates.into_boxed_slice();
        let mut host = PythonHost {
            py,
            resolver: resolver.into_bound(py),
            decoder: PythonDecoder,
        };
        Ok(self
            .core
            .steps(inputs, tags, span, &candidates, &mut host)
            .map_err(draw_error)?
            .into_iter()
            .map(|step| (step.rule, step.output, step.tag))
            .collect())
    }

    /// Close unary rules over one cell up to the supplied depth.
    fn closure(
        &self,
        py: Python<'_>,
        arrival: Vec<(u32, u8)>,
        span: (usize, usize),
        depth: usize,
        resolver: Py<PyAny>,
    ) -> PyResult<PyClosure> {
        let items: Vec<ItemId> = arrival
            .into_iter()
            .map(|(category, tag)| ItemId { category, tag })
            .collect();
        let mut host = PythonHost {
            py,
            resolver: resolver.into_bound(py),
            decoder: PythonDecoder,
        };
        let closure = self
            .core
            .closure(&items, span, depth, &mut host)
            .map_err(draw_error)?;
        let edges = closure
            .edges
            .into_iter()
            .map(|(child, step)| {
                (
                    (child.category, child.tag),
                    step.rule,
                    (step.output, step.tag),
                )
            })
            .collect();
        let concluded = closure
            .concluded
            .into_iter()
            .map(|item| (item.category, item.tag))
            .collect();

        Ok((edges, concluded, closure.truncated))
    }

    fn __repr__(&self) -> PyResult<String> {
        let (rules, tags) = self.core.counts().map_err(draw_error)?;
        Ok(format!("ChartProgram(rules={rules}, tags={tags})"))
    }
}
