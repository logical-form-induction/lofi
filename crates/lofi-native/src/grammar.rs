//! Python conversion boundary for native grammar compilation.

use lofi_grammar::{
    DeclarationEnvelope, GrammarCompiler, LinkedCompositionDirection, LinkedGrammar,
    LinkedNormalFormRole, LinkedOperatorPattern, LinkedPattern, LinkedPatternRecipe,
    LinkedPayloadKind, LinkedPayloadPattern, LinkedRule, LinkedRuleKind, LinkedRulePlan,
    RulePlanSource,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes};

use crate::rule::{BoundRestriction, PyRuleKernel};
use crate::term::PyTermStore;

type FunctorKernelRow = (Vec<String>, String, String, Vec<usize>, Vec<usize>);
/// Primary targets, secondary categories, and the primary and secondary slots.
type RestrictionRow = (Vec<String>, Vec<String>, usize, Option<usize>);

fn functor_kernel_row(functors: &Bound<'_, PyAny>) -> PyResult<FunctorKernelRow> {
    let operators = functors
        .getattr("slashes")?
        .try_iter()?
        .map(|operator| operator?.extract())
        .collect::<PyResult<Vec<_>>>()?;
    Ok((
        operators,
        functors.getattr("forward_slash")?.extract()?,
        functors.getattr("backward_slash")?.extract()?,
        functors.getattr("result_path")?.extract()?,
        functors.getattr("argument_path")?.extract()?,
    ))
}

type BindingRow = (Vec<usize>, u16, Option<usize>);
type OperatorRow = (
    String,
    u32,
    u16,
    Vec<u16>,
    String,
    String,
    u32,
    Vec<BindingRow>,
);
type VocabularyRow = (u16, Vec<String>, Vec<usize>);
type FeatureRow = (String, u32, Vec<String>);
type TheoryConstructorRow = (
    String,
    String,
    Vec<String>,
    Vec<String>,
    usize,
    Vec<String>,
    String,
);

/// One Python-facing native-validated term pattern.
#[pyclass(frozen, module = "lofi._native", name = "_CompiledPattern")]
pub struct PyCompiledPattern {
    pattern: LinkedPattern,
}

impl PyCompiledPattern {
    const fn new(pattern: LinkedPattern) -> Self {
        Self { pattern }
    }

    fn wrap(py: Python<'_>, pattern: &LinkedPattern) -> PyResult<Py<Self>> {
        Py::new(py, Self::new(pattern.clone()))
    }
}

#[pymethods]
impl PyCompiledPattern {
    #[new]
    fn refuse_public_construction() -> PyResult<Self> {
        Err(PyTypeError::new_err(
            "compiled patterns are created by the grammar compiler",
        ))
    }

    #[getter]
    fn kind(&self) -> &'static str {
        match &self.pattern {
            LinkedPattern::Term { .. } => "term",
            LinkedPattern::Constant { .. } => "constant",
            LinkedPattern::Node { .. } => "node",
            LinkedPattern::Spine { .. } => "spine",
        }
    }

    #[getter]
    fn name(&self) -> Option<&str> {
        match &self.pattern {
            LinkedPattern::Term { name, .. }
            | LinkedPattern::Constant { name, .. }
            | LinkedPattern::Spine { name, .. } => Some(name),
            LinkedPattern::Node { .. } => None,
        }
    }

    #[getter]
    fn sort(&self) -> Option<&str> {
        match &self.pattern {
            LinkedPattern::Term { sort, .. } | LinkedPattern::Constant { sort, .. } => {
                sort.as_deref()
            }
            LinkedPattern::Node { .. } | LinkedPattern::Spine { .. } => None,
        }
    }

    #[getter]
    const fn operator_kind(&self) -> Option<&'static str> {
        match &self.pattern {
            LinkedPattern::Node {
                operator: LinkedOperatorPattern::Fixed(_),
                ..
            } => Some("fixed"),
            LinkedPattern::Node {
                operator: LinkedOperatorPattern::Variable(_),
                ..
            } => Some("variable"),
            _ => None,
        }
    }

    #[getter]
    fn operator_name(&self) -> Option<&str> {
        match &self.pattern {
            LinkedPattern::Node {
                operator: LinkedOperatorPattern::Fixed(name),
                ..
            }
            | LinkedPattern::Node {
                operator: LinkedOperatorPattern::Variable(name),
                ..
            } => Some(name),
            _ => None,
        }
    }

    #[getter]
    fn arguments(&self, py: Python<'_>) -> PyResult<Vec<Py<Self>>> {
        match &self.pattern {
            LinkedPattern::Node { arguments, .. } => arguments
                .iter()
                .map(|argument| Self::wrap(py, argument))
                .collect(),
            _ => Ok(Vec::new()),
        }
    }

    #[getter]
    const fn payload_kind(&self) -> Option<&'static str> {
        match &self.pattern {
            LinkedPattern::Node {
                payload: Some(LinkedPayloadPattern::Fixed(_)),
                ..
            } => Some("fixed"),
            LinkedPattern::Node {
                payload: Some(LinkedPayloadPattern::Variable(_)),
                ..
            } => Some("variable"),
            _ => None,
        }
    }

    #[getter]
    const fn payload_value(&self) -> Option<i64> {
        match &self.pattern {
            LinkedPattern::Node {
                payload: Some(LinkedPayloadPattern::Fixed(value)),
                ..
            } => Some(*value),
            _ => None,
        }
    }

    #[getter]
    fn payload_name(&self) -> Option<&str> {
        match &self.pattern {
            LinkedPattern::Node {
                payload: Some(LinkedPayloadPattern::Variable(name)),
                ..
            } => Some(name),
            _ => None,
        }
    }

    #[getter]
    fn core(&self, py: Python<'_>) -> PyResult<Option<Py<Self>>> {
        match &self.pattern {
            LinkedPattern::Spine { core, .. } => Self::wrap(py, core).map(Some),
            _ => Ok(None),
        }
    }

    #[getter]
    fn operators(&self) -> Vec<String> {
        match &self.pattern {
            LinkedPattern::Spine { operators, .. } => operators.clone(),
            _ => Vec::new(),
        }
    }

    #[getter]
    fn slot(&self) -> Option<u64> {
        match &self.pattern {
            LinkedPattern::Spine { slot, .. } => Some(*slot),
            _ => None,
        }
    }

    #[getter]
    fn min_order(&self) -> Option<u64> {
        match &self.pattern {
            LinkedPattern::Spine { min_order, .. } => Some(*min_order),
            _ => None,
        }
    }

    #[getter]
    fn max_order(&self) -> Option<u64> {
        match &self.pattern {
            LinkedPattern::Spine { max_order, .. } => *max_order,
            _ => None,
        }
    }
}

/// One Python-facing structural normal-form role.
#[pyclass(frozen, module = "lofi._native", name = "_CompiledNormalFormRole")]
pub struct PyCompiledNormalFormRole {
    role: LinkedNormalFormRole,
}

#[pymethods]
impl PyCompiledNormalFormRole {
    #[new]
    fn new() -> PyResult<Self> {
        Err(PyTypeError::new_err(
            "compiled normal-form roles are created by the grammar compiler",
        ))
    }

    #[getter]
    const fn kind(&self) -> &'static str {
        self.role.kind()
    }

    #[getter]
    const fn direction(&self) -> Option<&'static str> {
        self.role.direction()
    }

    #[getter]
    const fn principal_slot(&self) -> Option<u64> {
        self.role.principal_slot()
    }

    #[getter]
    const fn order(&self) -> Option<u64> {
        self.role.order()
    }

    #[getter]
    const fn crossed(&self) -> bool {
        self.role.crossed()
    }
}

/// One Python-facing row in a native linked-rule plan.
#[pyclass(frozen, module = "lofi._native", name = "_CompiledRule")]
#[derive(Clone)]
pub struct PyCompiledRule {
    identifier: u32,
    tag: String,
    source: &'static str,
    source_index: usize,
    member_index: Option<usize>,
    kind: &'static str,
    view: Option<String>,
    constructor: Option<String>,
    direction: Option<&'static str>,
    order: Option<u64>,
    crossed: bool,
    targets: Option<Vec<String>>,
    label: Option<String>,
    coercion_member: Option<u64>,
    source_category: Option<String>,
    target_category: Option<String>,
    recipe_name: Option<String>,
    head_policy: Option<&'static str>,
    head_slot: Option<u64>,
    premises: Option<Vec<LinkedPattern>>,
    conclusion: Option<LinkedPattern>,
    normal_form_role: Option<LinkedNormalFormRole>,
    source_view: Option<String>,
    target_view: Option<String>,
    source_core: Option<String>,
    argument: Option<String>,
    absorbing: Option<Vec<String>>,
    side: Option<&'static str>,
    phase: Option<&'static str>,
    core_relation: Option<&'static str>,
    conjunction_feature: Option<String>,
    matching_recipe: Option<String>,
    restriction: Option<RestrictionRow>,
    bound_restriction: Option<BoundRestriction>,
    raising_arguments: Option<Vec<String>>,
    introduces: Option<Vec<String>>,
    consumes: Option<Vec<String>>,
    direct_schemata: bool,
}

impl PyCompiledRule {
    /// Restrict one built kernel to this rule's bound restriction, if declared.
    fn restrict(&self, kernel: PyRuleKernel) -> PyResult<PyRuleKernel> {
        match (&self.restriction, &self.bound_restriction) {
            (None, _) => Ok(kernel),
            (Some(_), Some(bound)) => kernel.restricted(bound),
            (Some(_), None) => Err(PyValueError::new_err(format!(
                "compiled rule {:?} declares a restriction whose categories were never bound",
                self.tag
            ))),
        }
    }

    fn adopt_pattern_recipe(&mut self, recipe: &LinkedPatternRecipe) {
        self.recipe_name = Some(recipe.name().to_owned());
        self.head_policy = Some(recipe.head_policy());
        self.head_slot = recipe.head_slot();
        self.premises = Some(recipe.premises().to_vec());
        self.conclusion = Some(recipe.conclusion().clone());
        self.targets = recipe.targets().map(<[String]>::to_vec);
        self.normal_form_role = Some(recipe.normal_form_role().clone());
        self.direct_schemata = recipe.has_direct_schemata();
    }

    fn from_linked(rule: &LinkedRule, linked: &LinkedGrammar) -> PyResult<Self> {
        let location = rule.location();
        let source = match location.source() {
            RulePlanSource::Explicit => "explicit",
            RulePlanSource::Combinator => "combinator",
            RulePlanSource::Coercion => "coercion",
        };
        let mut entry = Self {
            identifier: rule.identifier(),
            tag: rule.tag().to_owned(),
            source,
            source_index: location.index(),
            member_index: location.member(),
            kind: "explicit",
            view: None,
            constructor: None,
            direction: None,
            order: None,
            crossed: false,
            targets: None,
            label: None,
            coercion_member: None,
            source_category: None,
            target_category: None,
            recipe_name: None,
            head_policy: None,
            head_slot: None,
            premises: None,
            conclusion: None,
            normal_form_role: None,
            source_view: None,
            target_view: None,
            source_core: None,
            argument: None,
            absorbing: None,
            side: None,
            phase: None,
            core_relation: None,
            conjunction_feature: None,
            matching_recipe: rule.matching().map(str::to_owned),
            restriction: rule.restriction().map(|restriction| {
                (
                    restriction.primary_targets().to_vec(),
                    restriction.secondary_patterns().to_vec(),
                    restriction.primary_slot(),
                    restriction.secondary_slot(),
                )
            }),
            bound_restriction: None,
            raising_arguments: None,
            introduces: rule.introduces().map(<[String]>::to_vec),
            consumes: rule.consumes().map(<[String]>::to_vec),
            direct_schemata: false,
        };
        match rule.kind() {
            LinkedRuleKind::Application { functor } => {
                entry.kind = "application";
                let (view, constructor) = functor_row(linked, *functor)?;
                entry.view = Some(view);
                entry.constructor = Some(constructor);
            }
            LinkedRuleKind::Composition {
                functor,
                direction,
                order,
                crossed,
            } => {
                entry.kind = "composition";
                let (view, constructor) = functor_row(linked, *functor)?;
                entry.view = Some(view);
                entry.constructor = Some(constructor);
                entry.direction = Some(match direction {
                    LinkedCompositionDirection::Steedman => "steedman",
                    LinkedCompositionDirection::Kuhlmann => "kuhlmann",
                    LinkedCompositionDirection::Unrestricted => "unrestricted",
                });
                entry.order = *order;
                entry.crossed = *crossed;
            }
            LinkedRuleKind::Substitution {
                functor,
                order,
                crossed,
            } => {
                entry.kind = "substitution";
                let (view, constructor) = functor_row(linked, *functor)?;
                entry.view = Some(view);
                entry.constructor = Some(constructor);
                entry.order = *order;
                entry.crossed = *crossed;
            }
            LinkedRuleKind::TypeRaising {
                functor,
                targets,
                arguments,
            } => {
                entry.kind = "type-raising";
                let (view, constructor) = functor_row(linked, *functor)?;
                entry.view = Some(view);
                entry.constructor = Some(constructor);
                entry.targets.clone_from(targets);
                entry.raising_arguments.clone_from(arguments);
            }
            LinkedRuleKind::ClausalAdjunct { recipe } => {
                entry.kind = "clausal-adjunct";
                entry.recipe_name = Some(recipe.name().to_owned());
                entry.head_policy = Some("pass-through");
                entry.head_slot = Some(0);
                entry.source_view = Some(recipe.source_view().to_owned());
                entry.target_view = Some(recipe.target_view().to_owned());
                entry.source_core = Some(recipe.source_core().to_owned());
                entry.argument = Some(recipe.argument().to_owned());
            }
            LinkedRuleKind::Pattern { recipe } => {
                entry.kind = "pattern";
                entry.adopt_pattern_recipe(recipe);
            }
            LinkedRuleKind::Absorption { recipe } => {
                entry.kind = "absorption";
                entry.adopt_pattern_recipe(recipe.pattern());
                entry.absorbing = Some(recipe.absorbing().to_vec());
                entry.side = Some(recipe.side());
            }
            LinkedRuleKind::Coordination { recipe } => {
                entry.kind = "coordination";
                entry.adopt_pattern_recipe(recipe.pattern());
                entry.phase = Some(recipe.phase());
                entry.core_relation = Some(recipe.core_relation());
                entry.conjunction_feature = Some(recipe.conjunction_feature().to_owned());
            }
            LinkedRuleKind::Extension { recipe } => {
                entry.kind = "extension";
                entry.adopt_pattern_recipe(recipe);
            }
            LinkedRuleKind::Coercion {
                label,
                member,
                source,
                target,
            } => {
                entry.kind = "coercion";
                entry.direct_schemata = true;
                entry.label = Some(label.clone());
                entry.coercion_member = Some(*member);
                entry.source_category = Some(source.clone());
                entry.target_category = Some(target.clone());
            }
        }
        Ok(entry)
    }
}

fn functor_row(linked: &LinkedGrammar, identifier: u32) -> PyResult<(String, String)> {
    let functor = usize::try_from(identifier)
        .ok()
        .and_then(|index| linked.ir().category_syntax().functors().get(index))
        .ok_or_else(|| PyValueError::new_err("a linked rule lost its dense functor reference"))?;
    let constructor = linked
        .ir()
        .operators()
        .iter()
        .find(|operator| operator.operator() == functor.constructor())
        .ok_or_else(|| PyValueError::new_err("a linked functor lost its constructor operator"))?;
    Ok((functor.name().to_owned(), constructor.name().to_owned()))
}

#[pymethods]
impl PyCompiledRule {
    #[new]
    fn new() -> PyResult<Self> {
        Err(PyTypeError::new_err(
            "compiled rules are created by the grammar compiler",
        ))
    }

    #[getter]
    const fn identifier(&self) -> u32 {
        self.identifier
    }

    #[getter]
    fn tag(&self) -> &str {
        &self.tag
    }

    #[getter]
    const fn source(&self) -> &'static str {
        self.source
    }

    #[getter]
    const fn source_index(&self) -> usize {
        self.source_index
    }

    #[getter]
    const fn member_index(&self) -> Option<usize> {
        self.member_index
    }

    #[getter]
    const fn kind(&self) -> &'static str {
        self.kind
    }

    #[getter]
    fn view(&self) -> Option<&str> {
        self.view.as_deref()
    }

    #[getter]
    const fn direction(&self) -> Option<&'static str> {
        self.direction
    }

    #[getter]
    const fn order(&self) -> Option<u64> {
        self.order
    }

    #[getter]
    const fn crossed(&self) -> bool {
        self.crossed
    }

    #[getter]
    fn targets(&self) -> Option<Vec<String>> {
        self.targets.clone()
    }

    #[getter]
    fn raising_arguments(&self) -> Option<Vec<String>> {
        self.raising_arguments.clone()
    }

    #[getter]
    fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    #[getter]
    const fn coercion_member(&self) -> Option<u64> {
        self.coercion_member
    }

    #[getter]
    fn source_category(&self) -> Option<&str> {
        self.source_category.as_deref()
    }

    #[getter]
    fn target_category(&self) -> Option<&str> {
        self.target_category.as_deref()
    }

    #[getter]
    fn name(&self) -> Option<&str> {
        self.recipe_name.as_deref()
    }

    #[getter]
    const fn head_policy(&self) -> Option<&'static str> {
        self.head_policy
    }

    #[getter]
    const fn head_slot(&self) -> Option<u64> {
        self.head_slot
    }

    #[getter]
    fn premises(&self, py: Python<'_>) -> PyResult<Vec<Py<PyCompiledPattern>>> {
        self.premises.as_deref().map_or_else(
            || Ok(Vec::new()),
            |premises| {
                premises
                    .iter()
                    .map(|premise| PyCompiledPattern::wrap(py, premise))
                    .collect()
            },
        )
    }

    #[getter]
    fn conclusion(&self, py: Python<'_>) -> PyResult<Option<Py<PyCompiledPattern>>> {
        self.conclusion
            .as_ref()
            .map(|conclusion| PyCompiledPattern::wrap(py, conclusion))
            .transpose()
    }

    #[getter]
    fn normal_form_role(&self, py: Python<'_>) -> PyResult<Option<Py<PyCompiledNormalFormRole>>> {
        self.normal_form_role
            .as_ref()
            .map(|role| Py::new(py, PyCompiledNormalFormRole { role: role.clone() }))
            .transpose()
    }

    #[getter]
    fn source_view(&self) -> Option<&str> {
        self.source_view.as_deref()
    }

    #[getter]
    fn target_view(&self) -> Option<&str> {
        self.target_view.as_deref()
    }

    #[getter]
    fn source_core(&self) -> Option<&str> {
        self.source_core.as_deref()
    }

    #[getter]
    fn argument(&self) -> Option<&str> {
        self.argument.as_deref()
    }

    #[getter]
    fn absorbing(&self) -> Option<Vec<String>> {
        self.absorbing.clone()
    }

    #[getter]
    const fn side(&self) -> Option<&'static str> {
        self.side
    }

    #[getter]
    const fn phase(&self) -> Option<&'static str> {
        self.phase
    }

    #[getter]
    const fn core_relation(&self) -> Option<&'static str> {
        self.core_relation
    }

    #[getter]
    fn conjunction_feature(&self) -> Option<&str> {
        self.conjunction_feature.as_deref()
    }

    #[getter]
    fn matching_recipe(&self) -> Option<&str> {
        self.matching_recipe.as_deref()
    }

    #[getter]
    fn restriction(&self) -> Option<(Vec<String>, Vec<String>)> {
        self.restriction
            .as_ref()
            .map(|(targets, patterns, _, _)| (targets.clone(), patterns.clone()))
    }

    /// Return this rule with its restriction's categories lowered for enforcement.
    ///
    /// `targets` and `secondary` are the terms the restriction's primary
    /// targets and secondary patterns name, in declaration order, and
    /// `descents` pairs each functor constructor with its result path, along
    /// which a premise's target is found. Every kernel the returned rule
    /// builds admits only premises the restriction admits.
    fn restricted(
        &self,
        targets: &Bound<'_, PyAny>,
        secondary: &Bound<'_, PyAny>,
        descents: Vec<(String, Vec<usize>)>,
    ) -> PyResult<Self> {
        let Some((_, patterns, primary, secondary_slot)) = &self.restriction else {
            return Err(PyValueError::new_err(format!(
                "compiled rule {:?} declares no restriction",
                self.tag
            )));
        };
        if targets.len()? != self.restriction.as_ref().map_or(0, |row| row.0.len())
            || secondary.len()? != patterns.len()
        {
            return Err(PyValueError::new_err(format!(
                "compiled rule {:?} needs one term per restricted category",
                self.tag
            )));
        }
        let mut bound = self.clone();
        bound.bound_restriction = Some(BoundRestriction::lower(
            *primary,
            targets,
            secondary_slot.map(|slot| (slot, secondary)),
            descents,
        )?);
        Ok(bound)
    }

    #[getter]
    fn introduces(&self) -> Option<Vec<String>> {
        self.introduces.clone()
    }

    #[getter]
    fn consumes(&self) -> Option<Vec<String>> {
        self.consumes.clone()
    }

    #[getter]
    const fn direct_schemata(&self) -> bool {
        self.direct_schemata
    }

    fn empty_kernel(&self, py: Python<'_>, matching: Py<PyAny>) -> PyResult<Py<PyRuleKernel>> {
        if self.tag != "clausal-adjunct" {
            return Err(PyValueError::new_err(format!(
                "compiled rule {:?} is not an empty-kernel recipe",
                self.tag
            )));
        }
        Py::new(
            py,
            self.restrict(PyRuleKernel::compiled_empty(py, matching)?)?,
        )
    }

    fn composition_kernel(
        &self,
        py: Python<'_>,
        matching: Py<PyAny>,
        functors: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyRuleKernel>> {
        let (order, direction) = match self.kind {
            "application" => (Some(0), "unrestricted"),
            "composition" => (
                self.order
                    .map(|order| {
                        usize::try_from(order)
                            .map_err(|_| PyValueError::new_err("composition order exceeds usize"))
                    })
                    .transpose()?,
                self.direction.ok_or_else(|| {
                    PyValueError::new_err("a compiled composition lost its direction")
                })?,
            ),
            _ => {
                return Err(PyValueError::new_err(format!(
                    "compiled rule {:?} is not composition",
                    self.tag
                )));
            }
        };
        let slash = self.constructor.clone().ok_or_else(|| {
            PyValueError::new_err("a compiled composition lost its functor constructor")
        })?;
        Py::new(
            py,
            self.restrict(PyRuleKernel::compiled_composition(
                py,
                matching,
                functor_kernel_row(functors)?,
                (order, slash, self.crossed, direction.to_owned()),
            )?)?,
        )
    }

    fn substitution_kernel(
        &self,
        py: Python<'_>,
        matching: Py<PyAny>,
        functors: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyRuleKernel>> {
        if self.kind != "substitution" {
            return Err(PyValueError::new_err(format!(
                "compiled rule {:?} is not substitution",
                self.tag
            )));
        }
        let slash = self.constructor.clone().ok_or_else(|| {
            PyValueError::new_err("a compiled substitution lost its functor constructor")
        })?;
        let order = self
            .order
            .map(|order| {
                usize::try_from(order)
                    .map_err(|_| PyValueError::new_err("substitution order exceeds usize"))
            })
            .transpose()?;
        Py::new(
            py,
            self.restrict(PyRuleKernel::compiled_substitution(
                py,
                matching,
                functor_kernel_row(functors)?,
                (order, slash, self.crossed),
            )?)?,
        )
    }

    #[pyo3(signature = (matching, functors, targets, arguments))]
    fn raising_kernel(
        &self,
        py: Python<'_>,
        matching: Py<PyAny>,
        functors: &Bound<'_, PyAny>,
        targets: Option<&Bound<'_, PyAny>>,
        arguments: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyRuleKernel>> {
        if self.kind != "type-raising" {
            return Err(PyValueError::new_err(format!(
                "compiled rule {:?} is not type raising",
                self.tag
            )));
        }
        if targets.is_some() != self.targets.is_some() {
            return Err(PyValueError::new_err(format!(
                "compiled rule {:?} declares {} targets",
                self.tag,
                if self.targets.is_some() {
                    "finite"
                } else {
                    "open"
                }
            )));
        }
        if arguments.is_some() != self.raising_arguments.is_some() {
            return Err(PyValueError::new_err(format!(
                "compiled rule {:?} declares {} arguments",
                self.tag,
                if self.raising_arguments.is_some() {
                    "finite"
                } else {
                    "open"
                }
            )));
        }
        let slash = self.constructor.clone().ok_or_else(|| {
            PyValueError::new_err("compiled type raising lost its functor constructor")
        })?;
        Py::new(
            py,
            self.restrict(PyRuleKernel::compiled_raising(
                py,
                matching,
                functor_kernel_row(functors)?,
                slash,
                targets,
                arguments,
            )?)?,
        )
    }

    fn pattern_kernel(
        &self,
        py: Python<'_>,
        premises: &Bound<'_, PyAny>,
        conclusions: &Bound<'_, PyAny>,
        matching: Py<PyAny>,
        owner: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyRuleKernel>> {
        if self.source != "explicit" && self.kind != "coercion" {
            return Err(PyValueError::new_err(format!(
                "compiled rule {:?} is not a pattern recipe",
                self.tag
            )));
        }
        Py::new(
            py,
            self.restrict(PyRuleKernel::compiled_pattern(
                py,
                premises,
                conclusions,
                matching,
                owner,
            )?)?,
        )
    }

    fn absorption_kernel(
        &self,
        py: Python<'_>,
        pattern: &Bound<'_, PyRuleKernel>,
        absorbing: &Bound<'_, PyAny>,
        side: &str,
    ) -> PyResult<Py<PyRuleKernel>> {
        if self.tag != "absorption" {
            return Err(PyValueError::new_err(format!(
                "compiled rule {:?} is not absorption",
                self.tag
            )));
        }
        Py::new(
            py,
            self.restrict(PyRuleKernel::compiled_absorption(
                py, pattern, absorbing, side,
            )?)?,
        )
    }

    fn coordination_phase_kernel(
        &self,
        py: Python<'_>,
        pattern: &Bound<'_, PyRuleKernel>,
        pairs: &Bound<'_, PyAny>,
        targets: &Bound<'_, PyAny>,
        phase: &str,
        core_relation: &str,
        marking: Option<&str>,
    ) -> PyResult<Py<PyRuleKernel>> {
        if self.tag != "coordination" {
            return Err(PyValueError::new_err(format!(
                "compiled rule {:?} is not coordination",
                self.tag
            )));
        }
        Py::new(
            py,
            self.restrict(PyRuleKernel::compiled_coordination_phase(
                py,
                pattern,
                pairs,
                targets,
                phase,
                core_relation,
                marking,
            )?)?,
        )
    }
}

/// The linked signature, adopted core store, and deterministic native rule plan.
#[pyclass(frozen, module = "lofi._native", name = "_CompiledGrammar")]
pub struct PyCompiledGrammar {
    linked: LinkedGrammar,
    rules: LinkedRulePlan,
}

#[pymethods]
impl PyCompiledGrammar {
    #[new]
    fn new() -> PyResult<Self> {
        Err(PyTypeError::new_err(
            "compiled grammars are created by the grammar compiler",
        ))
    }

    #[getter]
    fn name(&self) -> &str {
        self.linked.ir().name()
    }

    #[getter]
    fn signature_fingerprint(&self) -> &str {
        self.linked.ir().fingerprint()
    }

    #[getter]
    fn rule_fingerprint(&self) -> &str {
        self.rules.fingerprint()
    }

    #[getter]
    fn sorts(&self) -> Vec<String> {
        self.linked.ir().sorts().to_vec()
    }

    #[getter]
    fn operators(&self) -> Vec<OperatorRow> {
        self.linked
            .ir()
            .operators()
            .iter()
            .map(|operator| {
                (
                    operator.name().to_owned(),
                    operator.operator(),
                    operator.sort(),
                    operator.arguments().to_vec(),
                    operator.family().to_owned(),
                    match operator.payload() {
                        LinkedPayloadKind::None => "none",
                        LinkedPayloadKind::Constant => "constant",
                        LinkedPayloadKind::Index => "index",
                        LinkedPayloadKind::Free => "free",
                    }
                    .to_owned(),
                    operator.erased(),
                    operator
                        .binding()
                        .iter()
                        .map(|site| (site.arguments().to_vec(), site.sort(), site.annotation()))
                        .collect(),
                )
            })
            .collect()
    }

    #[getter]
    fn declarations(&self) -> Vec<String> {
        self.linked
            .ir()
            .declarations()
            .iter()
            .map(|declaration| declaration.name().to_owned())
            .collect()
    }

    #[getter]
    fn vocabularies(&self) -> Vec<VocabularyRow> {
        self.linked
            .ir()
            .vocabularies()
            .iter()
            .map(|vocabulary| {
                (
                    vocabulary.sort(),
                    vocabulary.names().to_vec(),
                    vocabulary.cores().to_vec(),
                )
            })
            .collect()
    }

    #[getter]
    fn open_sorts(&self) -> Vec<u16> {
        self.linked.ir().open_sorts().to_vec()
    }

    #[getter]
    fn classifiers(&self) -> Vec<(u16, u16)> {
        self.linked
            .ir()
            .classifiers()
            .iter()
            .map(|classifier| (classifier.sort(), classifier.classifier()))
            .collect()
    }

    #[getter]
    fn features(&self) -> Vec<FeatureRow> {
        self.linked
            .ir()
            .features()
            .iter()
            .map(|features| {
                (
                    features.name().to_owned(),
                    features.operator(),
                    features.values().to_vec(),
                )
            })
            .collect()
    }

    #[getter]
    const fn has_holes(&self) -> bool {
        self.linked.ir().has_holes()
    }

    #[getter]
    const fn allows_vacuous_binding(&self) -> bool {
        self.linked.ir().allows_vacuous_binding()
    }

    #[getter]
    fn constant_family(&self) -> &str {
        self.linked.ir().constant_family()
    }

    #[getter]
    fn variable_family(&self) -> &str {
        self.linked.ir().variable_family()
    }

    #[getter]
    fn functor_names(&self) -> Vec<String> {
        self.linked
            .ir()
            .category_syntax()
            .functors()
            .iter()
            .map(|functor| functor.name().to_owned())
            .collect()
    }

    #[getter]
    fn plugin_tags(&self) -> Vec<String> {
        self.rules.plugin_tags().to_vec()
    }

    #[getter]
    fn theory_constructors(&self) -> Vec<TheoryConstructorRow> {
        self.rules
            .theory()
            .constructors()
            .iter()
            .map(|constructor| {
                (
                    constructor.rule().to_owned(),
                    constructor.constructor().to_owned(),
                    constructor.inferred().to_vec(),
                    constructor.declared().to_vec(),
                    constructor.premises(),
                    constructor.premise_indices().to_vec(),
                    constructor.conclusion_index().to_owned(),
                )
            })
            .collect()
    }

    #[getter]
    fn elaborated_rules(&self) -> Vec<String> {
        self.rules.theory().elaborated().to_vec()
    }

    #[getter]
    fn runtime_categories(&self) -> Vec<String> {
        self.rules.runtime_categories()
    }

    #[getter]
    fn rules(&self, py: Python<'_>) -> PyResult<Vec<Py<PyCompiledRule>>> {
        self.rules
            .rules()
            .iter()
            .map(|rule| Py::new(py, PyCompiledRule::from_linked(rule, &self.linked)?))
            .collect()
    }

    fn term_store(&self, py: Python<'_>, signature: &Bound<'_, PyAny>) -> PyResult<PyTermStore> {
        let fingerprint: String = signature.getattr("digest")?.extract()?;
        if fingerprint != self.linked.ir().fingerprint() {
            return Err(PyValueError::new_err(format!(
                "materialized signature fingerprint {fingerprint:?} does not match native fingerprint {:?}",
                self.linked.ir().fingerprint()
            )));
        }
        PyTermStore::from_core(py, signature, self.linked.store().clone())
    }
}

/// Prepare, link, and plan one canonical language and grammar declaration pair.
/// # Errors
/// Returns a Python value error when either declaration cannot be decoded,
/// linked, or expanded into one dense rule plan.
#[pyfunction(name = "_compile_grammar")]
pub fn compile_grammar(
    language: &Bound<'_, PyBytes>,
    grammar: &Bound<'_, PyBytes>,
) -> PyResult<PyCompiledGrammar> {
    let compiled = GrammarCompiler
        .compile(
            DeclarationEnvelope::new(language.as_bytes()),
            DeclarationEnvelope::new(grammar.as_bytes()),
        )
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let (linked, rules) = compiled.into_parts();
    Ok(PyCompiledGrammar { linked, rules })
}
