//! Python boundary for native term-rule kernels.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, MutexGuard};

use lofi_chart::program::{RuleBoundary, ScopedRuleProgram, TemplateCache, application_store};
use lofi_chart::rule::{
    AbsorptionRule, AbsorptionSide, CompositionRestriction, CompositionRule, CompositionWitness,
    CoordinationPhase, CoordinationPhaseRule, CoreRelation, FunctorSpec, OperatorPattern, Pattern,
    PatternClause, PatternRule, PayloadPattern, PremiseRestriction, ProgramRule as CoreProgramRule,
    RaisingRule, RepresentativeKernel as CoreRepresentativeKernel, RepresentativeTable,
    RestrictedRule, RuleError, RuleJoinBinding, RuleJoinKey, RuleKernel as CoreRuleKernel,
    RuleProgram as CoreRuleProgram, SubstitutionRule,
};
use lofi_chart::term::{OperatorId, Term as CoreTerm, TermStore as CoreStore};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyInt, PyTuple};

use crate::term::{PyTerm, PyTermStore, Registry, draw_error, error as term_error, wrap};

fn mutex<'a, T>(value: &'a Mutex<T>, name: &str) -> PyResult<MutexGuard<'a, T>> {
    value
        .lock()
        .map_err(|_| PyRuntimeError::new_err(format!("{name} lock was poisoned")))
}

fn lower_restriction(value: &str) -> PyResult<CompositionRestriction> {
    match value {
        "unrestricted" => Ok(CompositionRestriction::Unrestricted),
        "kuhlmann" => Ok(CompositionRestriction::Kuhlmann),
        "steedman" => Ok(CompositionRestriction::Steedman),
        _ => Err(PyValueError::new_err(format!(
            "unknown composition restriction {value:?}"
        ))),
    }
}

fn lower_side(value: &str) -> PyResult<AbsorptionSide> {
    match value {
        "left" => Ok(AbsorptionSide::Left),
        "right" => Ok(AbsorptionSide::Right),
        _ => Err(PyValueError::new_err(format!(
            "unknown absorption side {value:?}"
        ))),
    }
}

fn lower_phase(value: &str) -> PyResult<CoordinationPhase> {
    match value {
        "mark" => Ok(CoordinationPhase::Mark),
        "merge" => Ok(CoordinationPhase::Merge),
        _ => Err(PyValueError::new_err(format!(
            "unknown coordination phase {value:?}"
        ))),
    }
}

fn lower_core_relation(value: &str) -> PyResult<CoreRelation> {
    match value {
        "same" => Ok(CoreRelation::Same),
        "different" => Ok(CoreRelation::Different),
        _ => Err(PyValueError::new_err(format!(
            "unknown coordination core relation {value:?}"
        ))),
    }
}

fn lower_compiled_pattern(value: &Bound<'_, PyAny>, registry: &Registry) -> PyResult<Pattern> {
    if let Ok(arguments) = value.getattr("args") {
        let operator = value.getattr("op")?;
        let operator = if let Ok(name) = operator.extract::<String>() {
            OperatorPattern::Fixed(registry.operator_id(&name))
        } else {
            OperatorPattern::Variable(operator.getattr("name")?.extract()?)
        };
        let arguments = arguments
            .try_iter()?
            .map(|argument| lower_compiled_pattern(&argument?, registry))
            .collect::<PyResult<Vec<_>>>()?;
        let payload = value.getattr("payload")?;
        let payload = if payload.is_none() {
            PayloadPattern::None
        } else if payload.is_instance_of::<PyInt>() && !payload.is_instance_of::<PyBool>() {
            PayloadPattern::Integer(payload.str()?.to_str()?.to_owned())
        } else {
            PayloadPattern::Variable(payload.getattr("name")?.extract()?)
        };
        return Ok(Pattern::Node {
            operator,
            arguments,
            payload,
        });
    }
    if let Ok(core) = value.getattr("core") {
        let names: Vec<String> = value.getattr("ops")?.extract()?;
        let operators = names
            .into_iter()
            .map(|operator| {
                registry.operator_id(&operator).ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "unknown compiled pattern spine operator {operator:?}"
                    ))
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        return Ok(Pattern::Spine {
            name: value.getattr("name")?.extract()?,
            core: Box::new(lower_compiled_pattern(&core, registry)?),
            operators,
            slot: value.getattr("slot")?.extract()?,
            min_order: value.getattr("min_order")?.extract()?,
            max_order: value.getattr("max_order")?.extract()?,
        });
    }

    let sort_name: String = value.getattr("sort")?.extract()?;
    let sort = registry.sort_id(&sort_name).ok_or_else(|| {
        PyValueError::new_err(format!("unknown compiled pattern sort {sort_name:?}"))
    })?;
    Ok(Pattern::Term {
        name: value.getattr("name")?.extract()?,
        sort,
    })
}

pub(crate) fn lower_object_pattern(
    value: &Bound<'_, PyAny>,
    registry: &Registry,
) -> PyResult<Pattern> {
    if let Ok(arguments) = value.getattr("args") {
        let operator = value.getattr("op")?;
        let operator = if let Ok(name) = operator.extract::<String>() {
            OperatorPattern::Fixed(registry.operator_id(&name))
        } else {
            OperatorPattern::Variable(operator.getattr("name")?.extract()?)
        };
        let arguments = arguments
            .try_iter()?
            .map(|argument| lower_object_pattern(&argument?, registry))
            .collect::<PyResult<Vec<_>>>()?;
        let payload = value.getattr("payload")?;
        let payload = if payload.is_none() {
            PayloadPattern::None
        } else if payload.is_instance_of::<PyInt>() && !payload.is_instance_of::<PyBool>() {
            PayloadPattern::Integer(payload.str()?.to_str()?.to_owned())
        } else {
            PayloadPattern::Variable(payload.getattr("name")?.extract()?)
        };
        return Ok(Pattern::Node {
            operator,
            arguments,
            payload,
        });
    }
    if let Ok(core) = value.getattr("core") {
        let names: Vec<String> = value.getattr("ops")?.extract()?;
        return Ok(Pattern::Spine {
            name: value.getattr("name")?.extract()?,
            core: Box::new(lower_object_pattern(&core, registry)?),
            operators: names
                .into_iter()
                .filter_map(|operator| registry.operator_id(&operator))
                .collect(),
            slot: value.getattr("slot")?.extract()?,
            min_order: value.getattr("min_order")?.extract()?,
            max_order: value.getattr("max_order")?.extract()?,
        });
    }

    let sort_name: String = value.getattr("sort")?.extract()?;
    let sort = registry.sort_id(&sort_name).ok_or_else(|| {
        PyValueError::new_err(format!(
            "reachability metavariable sort {sort_name:?} is not a sort of the signature"
        ))
    })?;
    Ok(Pattern::Term {
        name: value.getattr("name")?.extract()?,
        sort,
    })
}

pub(crate) fn map_rule_error(error: RuleError, registry: &Registry, py: Python<'_>) -> PyErr {
    match error {
        RuleError::Term(error) => term_error(error, registry, py),
        other => PyValueError::new_err(other.to_string()),
    }
}

fn join_binding_object(
    py: Python<'_>,
    registry: &Arc<Registry>,
    binding: &RuleJoinBinding,
) -> PyResult<Py<PyAny>> {
    Ok(match binding {
        RuleJoinBinding::Term(term) => ("term", wrap(registry, py, term.clone())?)
            .into_pyobject(py)?
            .into_any()
            .unbind(),
        RuleJoinBinding::Operator(operator) => ("operator", *operator)
            .into_pyobject(py)?
            .into_any()
            .unbind(),
        RuleJoinBinding::Integer(value) => {
            ("integer", *value).into_pyobject(py)?.into_any().unbind()
        }
        RuleJoinBinding::Spine(frames) => {
            let frames = frames
                .iter()
                .map(|(term, slot)| Ok((wrap(registry, py, term.clone())?, *slot)))
                .collect::<PyResult<Vec<_>>>()?;
            ("spine", PyTuple::new(py, frames)?)
                .into_pyobject(py)?
                .into_any()
                .unbind()
        }
    })
}

fn join_bindings_object<'py>(
    py: Python<'py>,
    registry: &Arc<Registry>,
    bindings: &[RuleJoinBinding],
) -> PyResult<Bound<'py, PyTuple>> {
    PyTuple::new(
        py,
        bindings
            .iter()
            .map(|binding| join_binding_object(py, registry, binding))
            .collect::<PyResult<Vec<_>>>()?,
    )
}

fn join_key_object(
    py: Python<'_>,
    registry: &Arc<Registry>,
    key: &RuleJoinKey,
) -> PyResult<Py<PyAny>> {
    Ok(match key {
        RuleJoinKey::Direct => ("direct",).into_pyobject(py)?.into_any().unbind(),
        RuleJoinKey::Composition {
            orientation,
            order,
            representative,
        } => (
            "composition",
            *orientation,
            *order,
            wrap(registry, py, representative.clone())?,
        )
            .into_pyobject(py)?
            .into_any()
            .unbind(),
        RuleJoinKey::Substitution {
            orientation,
            order,
            frames,
            representative,
        } => {
            let frames = frames
                .iter()
                .map(|(slash, argument)| Ok((slash.clone(), wrap(registry, py, argument.clone())?)))
                .collect::<PyResult<Vec<_>>>()?;
            (
                "substitution",
                *orientation,
                *order,
                PyTuple::new(py, frames)?,
                wrap(registry, py, representative.clone())?,
            )
                .into_pyobject(py)?
                .into_any()
                .unbind()
        }
        RuleJoinKey::Pattern { clause, bindings } => (
            "pattern",
            *clause,
            join_bindings_object(py, registry, bindings)?,
        )
            .into_pyobject(py)?
            .into_any()
            .unbind(),
        RuleJoinKey::Absorption {
            clause,
            bindings,
            absorbing,
        } => (
            "absorption",
            *clause,
            join_bindings_object(py, registry, bindings)?,
            *absorbing,
        )
            .into_pyobject(py)?
            .into_any()
            .unbind(),
        RuleJoinKey::CoordinationPhase {
            clause,
            bindings,
            pair,
        } => (
            "coordination-phase",
            *clause,
            join_bindings_object(py, registry, bindings)?,
            *pair,
        )
            .into_pyobject(py)?
            .into_any()
            .unbind(),
    })
}

fn pattern_owner(owner: &Bound<'_, PyAny>) -> PyResult<(CoreStore, Arc<Registry>)> {
    if let Ok(store) = owner.extract::<PyRef<'_, PyTermStore>>() {
        return Ok((store.core_store(), store.registry()));
    }
    let term = owner.extract::<PyRef<'_, PyTerm>>()?;
    let registry = term.registry();
    let store = registry.store_for(&term.core_term()).ok_or_else(|| {
        PyValueError::new_err("the pattern owner's native term store is unavailable")
    })?;
    Ok((store, registry))
}

fn lower_terms(values: &Bound<'_, PyAny>, registry: &Arc<Registry>) -> PyResult<Vec<CoreTerm>> {
    values
        .try_iter()?
        .map(|value| {
            let term = value?.extract::<PyRef<'_, PyTerm>>()?;
            if !Arc::ptr_eq(registry, &term.registry()) {
                return Err(PyValueError::new_err(
                    "native rule constants belong to different term signatures",
                ));
            }
            Ok(term.core_term())
        })
        .collect()
}

/// One compiled native implementation of a declared term rule.
#[pyclass(frozen, module = "lofi._native", name = "RuleKernel")]
#[derive(Debug)]
pub struct PyRuleKernel {
    inner: CoreRuleKernel,
    matching: Py<PyAny>,
    representatives: CoreRepresentativeKernel,
    representative_registry: Option<Arc<Registry>>,
    store: Option<CoreStore>,
    registry: Option<Arc<Registry>>,
    constants: Vec<CoreTerm>,
    cache: Mutex<MatchingCache>,
    reachability_templates: TemplateCache,
}

/// One compiled native representative function for term-rule matching.
#[pyclass(frozen, module = "lofi._native", name = "RepresentativeKernel")]
#[derive(Clone, Debug)]
pub struct PyRepresentativeKernel {
    inner: CoreRepresentativeKernel,
    registry: Option<Arc<Registry>>,
    cache: Arc<Mutex<MatchingCache>>,
}

/// One ordered inventory of native rule kernels.
#[pyclass(frozen, module = "lofi._native", name = "RuleProgram")]
#[derive(Debug)]
pub struct PyRuleProgram {
    inner: Arc<ScopedRuleProgram<Registry>>,
}

#[derive(Debug, Default)]
struct MatchingCache {
    representatives: RepresentativeTable,
}

/// Principal slot, container slot, principal constructor and carried spine.
type WitnessRow = (usize, usize, String, Vec<String>);

fn register_tree(
    representatives: &CoreRepresentativeKernel,
    root: &CoreTerm,
    cache: &mut MatchingCache,
) {
    let mut pending = vec![root.clone()];
    let mut visited = HashSet::new();
    while let Some(term) = pending.pop() {
        if !visited.insert(term.clone()) || cache.representatives.contains(&term) {
            continue;
        }
        cache.representatives.register_with(representatives, &term);
        pending.extend(term.arguments().iter().cloned());
    }
}

/// A rule restriction whose categories are lowered into one term signature.
#[derive(Clone, Debug)]
pub(crate) struct BoundRestriction {
    restriction: PremiseRestriction,
    store: CoreStore,
    registry: Arc<Registry>,
}

impl BoundRestriction {
    /// Lower the restricted categories and the functor descents of one rule.
    pub(crate) fn lower(
        primary: usize,
        targets: &Bound<'_, PyAny>,
        secondary: Option<(usize, &Bound<'_, PyAny>)>,
        descents: Vec<(String, Vec<usize>)>,
    ) -> PyResult<Self> {
        let owner = if targets.len()? > 0 {
            targets.get_item(0)?
        } else if let Some((_, categories)) = secondary
            && categories.len()? > 0
        {
            categories.get_item(0)?
        } else {
            return Err(PyValueError::new_err(
                "a rule restriction names no primary target and no secondary category",
            ));
        };
        let (store, registry) = pattern_owner(&owner)?;
        let lowered = lower_terms(targets, &registry)?;
        let secondary = secondary
            .map(|(slot, categories)| Ok::<_, PyErr>((slot, lower_terms(categories, &registry)?)))
            .transpose()?;
        Ok(Self {
            restriction: PremiseRestriction {
                primary,
                targets: (!lowered.is_empty()).then_some(lowered),
                secondary,
                descents,
            },
            store,
            registry,
        })
    }
}

impl PyRuleKernel {
    /// Restrict this kernel's premises to the bound restriction's categories.
    pub(crate) fn restricted(mut self, bound: &BoundRestriction) -> PyResult<Self> {
        if let Some(registry) = &self.registry
            && !Arc::ptr_eq(registry, &bound.registry)
        {
            return Err(PyValueError::new_err(
                "a rule restriction belongs to a different term signature than its rule",
            ));
        }
        self.store.get_or_insert_with(|| bound.store.clone());
        self.registry
            .get_or_insert_with(|| Arc::clone(&bound.registry));
        self.constants
            .extend(bound.restriction.targets.iter().flatten().cloned());
        self.constants.extend(
            bound
                .restriction
                .secondary
                .iter()
                .flat_map(|(_, categories)| categories.iter().cloned()),
        );
        let rule = std::mem::replace(&mut self.inner, CoreRuleKernel::Empty);
        self.inner = CoreRuleKernel::Restricted(Box::new(RestrictedRule {
            rule,
            restriction: bound.restriction.clone(),
        }));
        Ok(self)
    }

    pub(crate) fn compiled_empty(py: Python<'_>, matching: Py<PyAny>) -> PyResult<Self> {
        let (representatives, representative_registry) = Self::representatives(py, &matching)?;
        Ok(Self {
            inner: CoreRuleKernel::Empty,
            matching,
            representatives,
            representative_registry,
            store: None,
            registry: None,
            constants: Vec::new(),
            cache: Mutex::new(MatchingCache::default()),
            reachability_templates: Arc::new(Mutex::new(Vec::new())),
        })
    }

    pub(crate) fn compiled_composition(
        py: Python<'_>,
        matching: Py<PyAny>,
        functors: (Vec<String>, String, String, Vec<usize>, Vec<usize>),
        declaration: (Option<usize>, String, bool, String),
    ) -> PyResult<Self> {
        let (representatives, representative_registry) = Self::representatives(py, &matching)?;
        let (operators, forward, backward, result_path, argument_path) = functors;
        let (order, slash, crossed, restriction) = declaration;
        Ok(Self {
            inner: CoreRuleKernel::Composition(CompositionRule {
                functors: FunctorSpec {
                    operators,
                    forward,
                    backward,
                    result_path,
                    argument_path,
                },
                order,
                slash,
                crossed,
                restriction: lower_restriction(&restriction)?,
            }),
            matching,
            representatives,
            representative_registry,
            store: None,
            registry: None,
            constants: Vec::new(),
            cache: Mutex::new(MatchingCache::default()),
            reachability_templates: Arc::new(Mutex::new(Vec::new())),
        })
    }

    pub(crate) fn compiled_substitution(
        py: Python<'_>,
        matching: Py<PyAny>,
        functors: (Vec<String>, String, String, Vec<usize>, Vec<usize>),
        declaration: (Option<usize>, String, bool),
    ) -> PyResult<Self> {
        let (representatives, representative_registry) = Self::representatives(py, &matching)?;
        let (operators, forward, backward, result_path, argument_path) = functors;
        let (order, slash, crossed) = declaration;
        Ok(Self {
            inner: CoreRuleKernel::Substitution(SubstitutionRule {
                functors: FunctorSpec {
                    operators,
                    forward,
                    backward,
                    result_path,
                    argument_path,
                },
                order,
                slash,
                crossed,
            }),
            matching,
            representatives,
            representative_registry,
            store: None,
            registry: None,
            constants: Vec::new(),
            cache: Mutex::new(MatchingCache::default()),
            reachability_templates: Arc::new(Mutex::new(Vec::new())),
        })
    }

    pub(crate) fn compiled_raising(
        py: Python<'_>,
        matching: Py<PyAny>,
        functors: (Vec<String>, String, String, Vec<usize>, Vec<usize>),
        slash: String,
        targets: Option<&Bound<'_, PyAny>>,
        arguments: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let (representatives, representative_registry) = Self::representatives(py, &matching)?;
        let (store, registry) = match targets.or(arguments) {
            None => (None, None),
            Some(terms) => {
                let (store, registry) = pattern_owner(&terms.get_item(0)?)?;
                (Some(store), Some(registry))
            }
        };
        let lower = |terms: Option<&Bound<'_, PyAny>>| -> PyResult<Option<Vec<CoreTerm>>> {
            match (terms, &registry) {
                (Some(terms), Some(registry)) => Ok(Some(lower_terms(terms, registry)?)),
                _ => Ok(None),
            }
        };
        let targets = lower(targets)?;
        let arguments = lower(arguments)?;
        let constants = targets
            .iter()
            .chain(&arguments)
            .flatten()
            .cloned()
            .collect::<Vec<_>>();
        let (operators, forward, backward, result_path, argument_path) = functors;
        Ok(Self {
            inner: CoreRuleKernel::Raising(RaisingRule {
                functors: FunctorSpec {
                    operators,
                    forward,
                    backward,
                    result_path,
                    argument_path,
                },
                slash,
                targets,
                arguments,
            }),
            matching,
            representatives,
            representative_registry,
            store,
            registry,
            constants,
            cache: Mutex::new(MatchingCache::default()),
            reachability_templates: Arc::new(Mutex::new(Vec::new())),
        })
    }

    pub(crate) fn compiled_pattern(
        py: Python<'_>,
        premises: &Bound<'_, PyAny>,
        conclusions: &Bound<'_, PyAny>,
        matching: Py<PyAny>,
        owner: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let (representatives, representative_registry) = Self::representatives(py, &matching)?;
        let (store, registry) = pattern_owner(owner)?;
        let premises = premises
            .try_iter()?
            .map(|premise| lower_compiled_pattern(&premise?, &registry))
            .collect::<PyResult<Vec<_>>>()?;
        let clauses = conclusions
            .try_iter()?
            .map(|conclusion| {
                Ok(PatternClause {
                    premises: premises.clone(),
                    conclusion: lower_compiled_pattern(&conclusion?, &registry)?,
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            inner: CoreRuleKernel::Pattern(PatternRule { clauses }),
            matching,
            representatives,
            representative_registry,
            store: Some(store),
            registry: Some(registry),
            constants: Vec::new(),
            cache: Mutex::new(MatchingCache::default()),
            reachability_templates: Arc::new(Mutex::new(Vec::new())),
        })
    }

    pub(crate) fn compiled_absorption(
        py: Python<'_>,
        pattern: &Bound<'_, Self>,
        absorbing: &Bound<'_, PyAny>,
        side: &str,
    ) -> PyResult<Self> {
        let pattern = pattern.borrow();
        let CoreRuleKernel::Pattern(pattern_rule) = &pattern.inner else {
            return Err(PyValueError::new_err(
                "a native absorption rule requires a pattern kernel",
            ));
        };
        let registry = pattern.registry.clone().ok_or_else(|| {
            PyValueError::new_err("a native absorption pattern has no term registry")
        })?;
        let absorbing = lower_terms(absorbing, &registry)?;
        let constants = pattern
            .constants
            .iter()
            .cloned()
            .chain(absorbing.iter().cloned())
            .collect();
        Ok(Self {
            inner: CoreRuleKernel::Absorption(AbsorptionRule {
                pattern: pattern_rule.clone(),
                absorbing,
                side: lower_side(side)?,
            }),
            matching: pattern.matching.clone_ref(py),
            representatives: pattern.representatives.clone(),
            representative_registry: pattern.representative_registry.clone(),
            store: pattern.store.clone(),
            registry: Some(registry),
            constants,
            cache: Mutex::new(MatchingCache::default()),
            reachability_templates: Arc::clone(&pattern.reachability_templates),
        })
    }

    pub(crate) fn compiled_coordination_phase(
        py: Python<'_>,
        pattern: &Bound<'_, Self>,
        pairs: &Bound<'_, PyAny>,
        targets: &Bound<'_, PyAny>,
        phase: &str,
        core_relation: &str,
        marking: Option<&str>,
    ) -> PyResult<Self> {
        let pattern = pattern.borrow();
        let CoreRuleKernel::Pattern(pattern_rule) = &pattern.inner else {
            return Err(PyValueError::new_err(
                "a native coordination phase requires a pattern kernel",
            ));
        };
        let marking = marking
            .map(|name| {
                let store = pattern.store.as_ref().ok_or_else(|| {
                    PyValueError::new_err("a native coordination pattern has no term store")
                })?;
                store
                    .signature()
                    .operators
                    .iter()
                    .position(|operator| operator.name == name)
                    .and_then(|position| OperatorId::try_from(position).ok())
                    .ok_or_else(|| {
                        PyValueError::new_err(format!(
                            "the coordination marking constructor {name:?} is not declared"
                        ))
                    })
            })
            .transpose()?;
        let registry = pattern.registry.clone().ok_or_else(|| {
            PyValueError::new_err("a native coordination pattern has no term registry")
        })?;
        let pairs = pairs
            .try_iter()?
            .map(|pair| {
                let pair = pair?;
                let left = pair.get_item(0)?.extract::<PyRef<'_, PyTerm>>()?;
                let right = pair.get_item(1)?.extract::<PyRef<'_, PyTerm>>()?;
                if !Arc::ptr_eq(&registry, &left.registry())
                    || !Arc::ptr_eq(&registry, &right.registry())
                {
                    return Err(PyValueError::new_err(
                        "native rule constants belong to different term signatures",
                    ));
                }
                Ok((left.core_term(), right.core_term()))
            })
            .collect::<PyResult<Vec<_>>>()?;
        let targets = if targets.is_none() {
            None
        } else {
            Some(lower_terms(targets, &registry)?)
        };
        let constants = pattern
            .constants
            .iter()
            .cloned()
            .chain(
                pairs
                    .iter()
                    .flat_map(|(left, right)| [left.clone(), right.clone()]),
            )
            .chain(targets.iter().flatten().cloned())
            .collect();
        Ok(Self {
            inner: CoreRuleKernel::CoordinationPhase(CoordinationPhaseRule {
                pattern: pattern_rule.clone(),
                pairs,
                marking,
                targets,
                phase: lower_phase(phase)?,
                core_relation: lower_core_relation(core_relation)?,
            }),
            matching: pattern.matching.clone_ref(py),
            representatives: pattern.representatives.clone(),
            representative_registry: pattern.representative_registry.clone(),
            store: pattern.store.clone(),
            registry: Some(registry),
            constants,
            cache: Mutex::new(MatchingCache::default()),
            reachability_templates: Arc::clone(&pattern.reachability_templates),
        })
    }

    fn representatives(
        py: Python<'_>,
        matching: &Py<PyAny>,
    ) -> PyResult<(CoreRepresentativeKernel, Option<Arc<Registry>>)> {
        let lowering = matching.bind(py).getattr("representatives")?;
        let lowering = lowering.extract::<PyRef<'_, PyRepresentativeKernel>>()?;
        Ok((lowering.inner.clone(), lowering.registry.clone()))
    }

    fn register_tree(&self, root: &CoreTerm, cache: &mut MatchingCache) {
        register_tree(&self.representatives, root, cache);
    }

    fn store_and_registry(
        &self,
        inputs: &[(CoreTerm, Arc<Registry>)],
    ) -> PyResult<(CoreStore, Arc<Registry>)> {
        application_store(
            &RuleBoundary {
                representative_scope: self.representative_registry.clone(),
                store: self.store.clone(),
                scope: self.registry.clone(),
            },
            inputs,
        )
        .map_err(draw_error)
    }

    fn lower_inputs(values: &Bound<'_, PyAny>) -> PyResult<Vec<(CoreTerm, Arc<Registry>)>> {
        values
            .try_iter()?
            .map(|value| {
                let value = value?;
                let term = value.extract::<PyRef<'_, PyTerm>>()?;
                Ok((term.core_term(), term.registry()))
            })
            .collect()
    }

    fn boundary_of(
        &self,
        terms: &[(CoreTerm, Arc<Registry>)],
    ) -> PyResult<(CoreStore, Arc<Registry>)> {
        let (store, registry) = self.store_and_registry(terms)?;
        if terms
            .iter()
            .any(|(_, held_registry)| !Arc::ptr_eq(&registry, held_registry))
        {
            return Err(PyValueError::new_err(
                "native rule terms belong to different term signatures",
            ));
        }
        Ok((store, registry))
    }

    fn composition(&self) -> PyResult<&CompositionRule> {
        match self.inner.unrestricted() {
            CoreRuleKernel::Composition(rule) => Ok(rule),
            _ => Err(PyValueError::new_err(
                "only a composition kernel has composition witnesses",
            )),
        }
    }

    fn witness_rows(
        &self,
        py: Python<'_>,
        inputs: &Bound<'_, PyAny>,
        output: &PyTerm,
    ) -> PyResult<Vec<CompositionWitness>> {
        let rule = self.composition()?;
        let mut terms = vec![(output.core_term(), output.registry())];
        terms.extend(Self::lower_inputs(inputs)?);
        let (store, registry) = self.boundary_of(&terms)?;
        let premises = terms[1..]
            .iter()
            .map(|(term, _)| term.clone())
            .collect::<Vec<_>>();
        let mut cache = mutex(&self.cache, "rule matching cache")?;
        rule.witnesses(
            &store,
            &self.representatives,
            &mut cache.representatives,
            &premises,
            &terms[0].0,
        )
        .map_err(|error| map_rule_error(error, &registry, py))
    }

    pub(crate) fn apply_terms(
        &self,
        py: Python<'_>,
        inputs: &[(CoreTerm, Arc<Registry>)],
    ) -> PyResult<Vec<Py<PyTerm>>> {
        let (store, registry) = self.store_and_registry(inputs)?;
        for (_, held_registry) in inputs {
            if !Arc::ptr_eq(&registry, held_registry) {
                return Err(PyValueError::new_err(
                    "native rule premises belong to different term signatures",
                ));
            }
        }
        let terms = inputs
            .iter()
            .map(|(term, _)| term.clone())
            .collect::<Vec<_>>();
        let outputs = {
            let mut cache = mutex(&self.cache, "rule matching cache")?;
            for term in terms.iter().chain(&self.constants) {
                self.register_tree(term, &mut cache);
            }
            self.inner
                .apply(&store, &cache.representatives, &terms)
                .map_err(|error| map_rule_error(error, &registry, py))?
        };
        outputs
            .into_iter()
            .map(|term| wrap(&registry, py, term))
            .collect()
    }
}

impl PyRuleProgram {
    /// Return the scoped rule program this inventory compiles to.
    pub(crate) fn core(&self) -> Arc<ScopedRuleProgram<Registry>> {
        Arc::clone(&self.inner)
    }
}

#[pymethods]
impl PyRuleProgram {
    #[new]
    fn new(py: Python<'_>, rules: Vec<Py<PyRuleKernel>>) -> PyResult<Self> {
        let reachability_templates = rules.first().map_or_else(
            || Arc::new(Mutex::new(Vec::new())),
            |rule| Arc::clone(&rule.bind(py).borrow().reachability_templates),
        );
        let mut compiled = Vec::with_capacity(rules.len());
        let mut boundaries = Vec::with_capacity(rules.len());
        for rule in rules {
            let rule = rule.bind(py).borrow();
            compiled.push(CoreProgramRule {
                kernel: rule.inner.clone(),
                matching: rule.matching.as_ptr().addr(),
                representatives: rule.representatives.clone(),
                constants: rule.constants.clone(),
            });
            boundaries.push(RuleBoundary {
                representative_scope: rule.representative_registry.clone(),
                store: rule.store.clone(),
                scope: rule.registry.clone(),
            });
        }
        let inner = ScopedRuleProgram::new(
            CoreRuleProgram::new(compiled),
            boundaries,
            reachability_templates,
        )
        .map_err(draw_error)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    fn apply(
        &self,
        py: Python<'_>,
        candidates: Vec<u16>,
        inputs: &Bound<'_, PyTuple>,
    ) -> PyResult<Vec<(u16, Vec<Py<PyTerm>>)>> {
        let candidates = candidates.into_boxed_slice();
        let terms = inputs
            .iter()
            .map(|input| {
                let input = input.extract::<PyRef<'_, PyTerm>>()?;
                Ok((input.core_term(), input.registry()))
            })
            .collect::<PyResult<Vec<_>>>()?;
        self.inner
            .apply(&candidates, &terms)
            .map_err(draw_error)?
            .into_iter()
            .map(|(position, outputs, registry)| {
                let wrapped = outputs
                    .into_iter()
                    .map(|term| wrap(&registry, py, term))
                    .collect::<PyResult<Vec<_>>>()?;
                Ok((position, wrapped))
            })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!("RuleProgram(rules={})", self.inner.len())
    }
}

impl PyRepresentativeKernel {
    pub(crate) fn inner_core(&self) -> CoreRepresentativeKernel {
        self.inner.clone()
    }

    pub(crate) fn registry(&self) -> Option<Arc<Registry>> {
        self.registry.clone()
    }
}

#[pymethods]
impl PyRepresentativeKernel {
    /// Compile exact term identity.
    #[staticmethod]
    fn identity() -> Self {
        Self {
            inner: CoreRepresentativeKernel::Identity,
            registry: None,
            cache: Arc::new(Mutex::new(MatchingCache::default())),
        }
    }

    /// Compile the precomputed feature-erasure core.
    #[staticmethod]
    fn core() -> Self {
        Self {
            inner: CoreRepresentativeKernel::Core,
            registry: None,
            cache: Arc::new(Mutex::new(MatchingCache::default())),
        }
    }

    /// Intersect two compiled representative relations.
    #[staticmethod]
    fn product(left: PyRef<'_, Self>, right: PyRef<'_, Self>) -> PyResult<Self> {
        let registry = match (&left.registry, &right.registry) {
            (Some(left), Some(right)) if !Arc::ptr_eq(left, right) => {
                return Err(PyValueError::new_err(
                    "product representatives belong to different term signatures",
                ));
            }
            (Some(registry), _) | (_, Some(registry)) => Some(registry.clone()),
            (None, None) => None,
        };
        let inner = CoreRepresentativeKernel::Product {
            left: Box::new(left.inner.clone()),
            right: Box::new(right.inner.clone()),
        };
        drop(left);
        drop(right);
        Ok(Self {
            inner,
            registry,
            cache: Arc::new(Mutex::new(MatchingCache::default())),
        })
    }

    /// Compile mutual structural free-variable subsumption.
    #[staticmethod]
    fn subsumption() -> Self {
        Self {
            inner: CoreRepresentativeKernel::Subsumption,
            registry: None,
            cache: Arc::new(Mutex::new(MatchingCache::default())),
        }
    }

    /// Override a finite exact domain before applying this base function.
    fn refine(&self, pairs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mut registry = self.registry.clone();
        let overrides = pairs
            .try_iter()?
            .map(|pair| {
                let pair = pair?;
                let exact = pair.get_item(0)?.extract::<PyRef<'_, PyTerm>>()?;
                let representative = pair.get_item(1)?.extract::<PyRef<'_, PyTerm>>()?;
                let exact_registry = exact.registry();
                let representative_registry = representative.registry();
                if !Arc::ptr_eq(&exact_registry, &representative_registry) {
                    return Err(PyValueError::new_err(
                        "a representative override crosses term signatures",
                    ));
                }
                if let Some(held) = &registry {
                    if !Arc::ptr_eq(held, &exact_registry) {
                        return Err(PyValueError::new_err(
                            "representative overrides belong to different term signatures",
                        ));
                    }
                } else {
                    registry = Some(exact_registry);
                }
                Ok((exact.core_term(), representative.core_term()))
            })
            .collect::<PyResult<HashMap<_, _>>>()?;
        Ok(Self {
            inner: CoreRepresentativeKernel::Refined {
                base: Box::new(self.inner.clone()),
                overrides,
            },
            registry,
            cache: Arc::new(Mutex::new(MatchingCache::default())),
        })
    }

    /// Return one exact term's representative.
    fn key(&self, py: Python<'_>, term: Py<PyTerm>) -> PyResult<Py<PyTerm>> {
        let term = term.into_bound(py);
        let term = term.borrow();
        let registry = term.registry();
        if let Some(held) = &self.registry
            && !Arc::ptr_eq(held, &registry)
        {
            return Err(PyValueError::new_err(
                "a representative kernel belongs to another term signature",
            ));
        }
        let core = term.core_term();
        let representative = {
            let mut cache = mutex(&self.cache, "representative cache")?;
            register_tree(&self.inner, &core, &mut cache);
            cache
                .representatives
                .representative(&core)
                .map_err(|error| map_rule_error(error, &registry, py))?
        };
        wrap(&registry, py, representative)
    }

    fn __repr__(&self) -> String {
        let kind = match &self.inner {
            CoreRepresentativeKernel::Identity => "identity",
            CoreRepresentativeKernel::Core => "core",
            CoreRepresentativeKernel::Refined { .. } => "refined",
            CoreRepresentativeKernel::Product { .. } => "product",
            CoreRepresentativeKernel::Subsumption => "subsumption",
        };
        format!("RepresentativeKernel(kind={kind:?})")
    }
}

#[pymethods]
impl PyRuleKernel {
    /// Return the exact join keys one premise offers in one slot.
    fn join_keys(
        &self,
        py: Python<'_>,
        term: &Bound<'_, PyTerm>,
        slot: usize,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let term = term.get();
        let terms = [(term.core_term(), term.registry())];
        let (_, registry) = self.boundary_of(&terms)?;
        let keys = {
            let mut cache = mutex(&self.cache, "rule matching cache")?;
            self.register_tree(&terms[0].0, &mut cache);
            self.inner
                .join_keys(&cache.representatives, &terms[0].0, slot)
                .map_err(|error| map_rule_error(error, &registry, py))?
        };
        keys.iter()
            .map(|key| join_key_object(py, &registry, key))
            .collect()
    }

    /// Return each witness of one composition step.
    ///
    /// A witness is the principal slot, the container slot, the principal
    /// constructor and the carried spine's constructors, outermost first.
    fn witnesses(
        &self,
        py: Python<'_>,
        inputs: &Bound<'_, PyAny>,
        output: &Bound<'_, PyTerm>,
    ) -> PyResult<Vec<WitnessRow>> {
        let slash = self.composition()?.slash.clone();
        Ok(self
            .witness_rows(py, inputs, output.get())?
            .into_iter()
            .map(|witness| {
                (
                    witness.principal_slot,
                    witness.container_slot,
                    slash.clone(),
                    witness.spine,
                )
            })
            .collect())
    }

    /// Return the composition order one step used, if the rule draws it.
    fn applied_order(
        &self,
        py: Python<'_>,
        inputs: &Bound<'_, PyAny>,
        output: &Bound<'_, PyTerm>,
    ) -> PyResult<Option<usize>> {
        let rule = self.composition()?;
        let output = output.get();
        let mut terms = vec![(output.core_term(), output.registry())];
        terms.extend(Self::lower_inputs(inputs)?);
        let (store, registry) = self.boundary_of(&terms)?;
        let premises = terms[1..]
            .iter()
            .map(|(term, _)| term.clone())
            .collect::<Vec<_>>();
        let mut cache = mutex(&self.cache, "rule matching cache")?;
        rule.applied_order(
            &store,
            &self.representatives,
            &mut cache.representatives,
            &premises,
            &terms[0].0,
        )
        .map_err(|error| map_rule_error(error, &registry, py))
    }

    /// Decide whether the rule relates the premises to the conclusion.
    fn admits(
        &self,
        py: Python<'_>,
        inputs: &Bound<'_, PyAny>,
        output: &Bound<'_, PyTerm>,
    ) -> PyResult<bool> {
        let output = output.get();
        let mut terms = vec![(output.core_term(), output.registry())];
        terms.extend(Self::lower_inputs(inputs)?);
        let (store, registry) = self.boundary_of(&terms)?;
        let premises = terms[1..]
            .iter()
            .map(|(term, _)| term.clone())
            .collect::<Vec<_>>();
        let mut cache = mutex(&self.cache, "rule matching cache")?;
        self.inner
            .admits(
                &store,
                &self.representatives,
                &mut cache.representatives,
                &premises,
                &terms[0].0,
            )
            .map_err(|error| map_rule_error(error, &registry, py))
    }

    /// Recover the fillers of the one missing premise, given as ``None``.
    fn residuate(
        &self,
        py: Python<'_>,
        output: &Bound<'_, PyTerm>,
        known: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<Py<PyTerm>>> {
        let output = output.get();
        let mut terms = vec![(output.core_term(), output.registry())];
        let mut slots = Vec::new();
        for value in known.try_iter()? {
            let value = value?;
            if value.is_none() {
                slots.push(None);
            } else {
                let term = value.extract::<PyRef<'_, PyTerm>>()?;
                slots.push(Some(term.core_term()));
                terms.push((term.core_term(), term.registry()));
            }
        }
        let (store, registry) = self.boundary_of(&terms)?;
        let residuals = {
            let mut cache = mutex(&self.cache, "rule matching cache")?;
            self.inner
                .residuate(
                    &store,
                    &self.representatives,
                    &mut cache.representatives,
                    &terms[0].0,
                    &slots,
                )
                .map_err(|error| map_rule_error(error, &registry, py))?
        };
        residuals
            .into_iter()
            .map(|term| wrap(&registry, py, term))
            .collect()
    }

    fn apply(&self, py: Python<'_>, inputs: &Bound<'_, PyTuple>) -> PyResult<Vec<Py<PyTerm>>> {
        let terms = inputs
            .iter()
            .map(|input| {
                let input = input.extract::<PyRef<'_, PyTerm>>()?;
                Ok((input.core_term(), input.registry()))
            })
            .collect::<PyResult<Vec<_>>>()?;
        self.apply_terms(py, &terms)
    }

    fn __repr__(&self) -> String {
        let kind = match &self.inner {
            CoreRuleKernel::Empty => "empty",
            CoreRuleKernel::Composition(_) => "composition",
            CoreRuleKernel::Substitution(_) => "substitution",
            CoreRuleKernel::Raising(_) => "raising",
            CoreRuleKernel::Pattern(_) => "pattern",
            CoreRuleKernel::Absorption(_) => "absorption",
            CoreRuleKernel::CoordinationPhase(_) => "coordination-phase",
            CoreRuleKernel::Restricted(_) => "restricted",
        };
        format!("RuleKernel({kind:?})")
    }
}

#[cfg(test)]
mod tests {
    use super::lower_restriction;
    use lofi_chart::rule::CompositionRestriction;

    #[test]
    fn composition_restrictions_are_exhaustive() {
        assert!(matches!(
            lower_restriction("unrestricted"),
            Ok(CompositionRestriction::Unrestricted)
        ));
        assert!(matches!(
            lower_restriction("kuhlmann"),
            Ok(CompositionRestriction::Kuhlmann)
        ));
        assert!(matches!(
            lower_restriction("steedman"),
            Ok(CompositionRestriction::Steedman)
        ));
        assert!(lower_restriction("unknown").is_err());
    }
}
