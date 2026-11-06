//! Deterministic metadata for rules expanded from one grammar declaration.

use crate::{RulePlanLocation, RulePlanSource};

/// One operator position in a compiled term pattern.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LinkedOperatorPattern {
    /// One fixed linked operator name.
    Fixed(String),
    /// One operator variable name.
    Variable(String),
}

/// One payload position in a compiled term pattern.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LinkedPayloadPattern {
    /// One fixed integer payload.
    Fixed(i64),
    /// One payload variable name.
    Variable(String),
}

/// One native-validated runtime term pattern.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LinkedPattern {
    /// A sorted term variable.
    Term {
        /// Variable name.
        name: String,
        /// Explicit sort, or the grammar category sort.
        sort: Option<String>,
    },
    /// A vocabulary constant.
    Constant {
        /// Constant name.
        name: String,
        /// Explicit sort, or the grammar category sort.
        sort: Option<String>,
    },
    /// An operator-headed pattern.
    Node {
        /// Fixed or variable operator position.
        operator: LinkedOperatorPattern,
        /// Child patterns in operator-argument order.
        arguments: Vec<Self>,
        /// Optional fixed or variable payload position.
        payload: Option<LinkedPayloadPattern>,
    },
    /// A bounded or unbounded functor-spine pattern.
    Spine {
        /// Shared spine-variable name.
        name: String,
        /// Pattern at the foot of the spine.
        core: Box<Self>,
        /// Operators admitted at each frame.
        operators: Vec<String>,
        /// Child slot that continues the spine.
        slot: u64,
        /// Minimum frame count.
        min_order: u64,
        /// Maximum frame count, or no finite upper bound.
        max_order: Option<u64>,
    },
}

/// One structural normal-form role carried by a compiled pattern rule.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LinkedNormalFormRole {
    pub(crate) kind: &'static str,
    pub(crate) direction: Option<&'static str>,
    pub(crate) principal_slot: Option<u64>,
    pub(crate) order: Option<u64>,
    pub(crate) crossed: bool,
}

impl LinkedNormalFormRole {
    /// Return the structural role kind.
    #[must_use]
    pub const fn kind(&self) -> &'static str {
        self.kind
    }

    /// Return the directed role orientation, if any.
    #[must_use]
    pub const fn direction(&self) -> Option<&'static str> {
        self.direction
    }

    /// Return the primary premise slot, if any.
    #[must_use]
    pub const fn principal_slot(&self) -> Option<u64> {
        self.principal_slot
    }

    /// Return the fixed composition order, if any.
    #[must_use]
    pub const fn order(&self) -> Option<u64> {
        self.order
    }

    /// Return whether this role is crossed.
    #[must_use]
    pub const fn crossed(&self) -> bool {
        self.crossed
    }
}

/// Native-validated fields shared by compiled pattern relations.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LinkedPatternRecipe {
    pub(crate) name: String,
    pub(crate) premises: Vec<LinkedPattern>,
    pub(crate) conclusion: LinkedPattern,
    pub(crate) head_policy: &'static str,
    pub(crate) head_slot: Option<u64>,
    pub(crate) targets: Option<Vec<String>>,
    pub(crate) normal_form_role: LinkedNormalFormRole,
}

impl LinkedPattern {
    /// Return whether the pattern is built directly, with fixed operators and
    /// payloads and no functor spine at any depth.
    #[must_use]
    pub fn is_direct(&self) -> bool {
        match self {
            Self::Term { .. } | Self::Constant { .. } => true,
            Self::Node {
                operator,
                arguments,
                payload,
            } => {
                matches!(operator, LinkedOperatorPattern::Fixed(_))
                    && !matches!(payload, Some(LinkedPayloadPattern::Variable(_)))
                    && arguments.iter().all(Self::is_direct)
            }
            Self::Spine { .. } => false,
        }
    }
}

impl LinkedPatternRecipe {
    /// Return whether the recipe declares no finite targets and only direct
    /// premise and conclusion patterns, so its schemata are its own patterns.
    #[must_use]
    pub fn has_direct_schemata(&self) -> bool {
        self.targets.is_none()
            && self.premises.iter().all(LinkedPattern::is_direct)
            && self.conclusion.is_direct()
    }

    /// Return the rule name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Return premise patterns in physical order.
    #[must_use]
    pub fn premises(&self) -> &[LinkedPattern] {
        &self.premises
    }

    /// Return the conclusion pattern.
    #[must_use]
    pub const fn conclusion(&self) -> &LinkedPattern {
        &self.conclusion
    }

    /// Return the semantic head policy.
    #[must_use]
    pub const fn head_policy(&self) -> &'static str {
        self.head_policy
    }

    /// Return the physical premise selected as the head, if any.
    #[must_use]
    pub const fn head_slot(&self) -> Option<u64> {
        self.head_slot
    }

    /// Return finite unbound-result targets, if any.
    #[must_use]
    pub fn targets(&self) -> Option<&[String]> {
        self.targets.as_deref()
    }

    /// Return the structural normal-form role.
    #[must_use]
    pub const fn normal_form_role(&self) -> &LinkedNormalFormRole {
        &self.normal_form_role
    }
}

/// One native-validated clausal-adjunct recipe.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LinkedClausalAdjunctRecipe {
    pub(crate) name: String,
    pub(crate) source_view: String,
    pub(crate) target_view: String,
    pub(crate) source_core: String,
    pub(crate) argument: String,
}

impl LinkedClausalAdjunctRecipe {
    /// Return the runtime rule name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Return the source functor view.
    #[must_use]
    pub fn source_view(&self) -> &str {
        &self.source_view
    }

    /// Return the target functor view.
    #[must_use]
    pub fn target_view(&self) -> &str {
        &self.target_view
    }

    /// Return the source category core.
    #[must_use]
    pub fn source_core(&self) -> &str {
        &self.source_core
    }

    /// Return the distinguished argument category.
    #[must_use]
    pub fn argument(&self) -> &str {
        &self.argument
    }
}

/// One native-validated absorption recipe.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LinkedAbsorptionRecipe {
    pub(crate) pattern: LinkedPatternRecipe,
    pub(crate) absorbing: Vec<String>,
    pub(crate) side: &'static str,
}

impl LinkedAbsorptionRecipe {
    /// Return the compiled pattern relation.
    #[must_use]
    pub const fn pattern(&self) -> &LinkedPatternRecipe {
        &self.pattern
    }

    /// Return absorbing category texts.
    #[must_use]
    pub fn absorbing(&self) -> &[String] {
        &self.absorbing
    }

    /// Return the absorbing premise side.
    #[must_use]
    pub const fn side(&self) -> &'static str {
        self.side
    }
}

/// One native-validated coordination-phase recipe.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LinkedCoordinationRecipe {
    pub(crate) pattern: LinkedPatternRecipe,
    pub(crate) phase: &'static str,
    pub(crate) core_relation: &'static str,
    pub(crate) conjunction_feature: String,
}

impl LinkedCoordinationRecipe {
    /// Return the compiled pattern relation.
    #[must_use]
    pub const fn pattern(&self) -> &LinkedPatternRecipe {
        &self.pattern
    }

    /// Return the coordination phase.
    #[must_use]
    pub const fn phase(&self) -> &'static str {
        self.phase
    }

    /// Return the core-identity requirement.
    #[must_use]
    pub const fn core_relation(&self) -> &'static str {
        self.core_relation
    }

    /// Return the conjunction feature name.
    #[must_use]
    pub fn conjunction_feature(&self) -> &str {
        &self.conjunction_feature
    }
}

/// The direction restriction of one planned generalized-composition rule.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LinkedCompositionDirection {
    /// Use Steedman's directional restriction.
    Steedman,
    /// Use Kuhlmann's directional restriction.
    Kuhlmann,
    /// Admit every direction supported by the functor view.
    Unrestricted,
}

/// Expansion-specific metadata carried by one planned rule.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LinkedRuleKind {
    /// Generalized application over one dense functor view.
    Application {
        /// The dense functor-view index.
        functor: u32,
    },
    /// Positive-order generalized composition over one dense functor view.
    Composition {
        /// The dense functor-view index.
        functor: u32,
        /// The admitted composition direction.
        direction: LinkedCompositionDirection,
        /// The exact order, or no finite upper bound.
        order: Option<u64>,
        /// Whether the deepest secondary slash is crossed.
        crossed: bool,
    },
    /// Generalized substitution over one dense functor view.
    Substitution {
        /// The dense functor-view index.
        functor: u32,
        /// The exact order, or no finite upper bound.
        order: Option<u64>,
        /// Whether the deepest secondary slash is crossed.
        crossed: bool,
    },
    /// Type raising over one dense functor view.
    TypeRaising {
        /// The dense functor-view index.
        functor: u32,
        /// Finite target category texts, or an unrestricted target family.
        targets: Option<Vec<String>>,
        /// Category texts that may raise, or every category.
        arguments: Option<Vec<String>>,
    },
    /// One explicit clausal-adjunct relation.
    ClausalAdjunct {
        /// Native-validated runtime recipe.
        recipe: LinkedClausalAdjunctRecipe,
    },
    /// One explicit general pattern relation.
    Pattern {
        /// Native-validated runtime recipe.
        recipe: LinkedPatternRecipe,
    },
    /// One explicit finite absorption relation.
    Absorption {
        /// Native-validated runtime recipe.
        recipe: LinkedAbsorptionRecipe,
    },
    /// One explicit feature-sensitive coordination phase.
    Coordination {
        /// Native-validated runtime recipe.
        recipe: LinkedCoordinationRecipe,
    },
    /// One installed extension relation with a native runtime contract.
    Extension {
        /// Native-validated pattern runtime recipe.
        recipe: LinkedPatternRecipe,
    },
    /// One ground member expanded from a coercion set.
    Coercion {
        /// The shared coercion-set label.
        label: String,
        /// The member position inside the coercion set.
        member: u64,
        /// The source category text.
        source: String,
        /// The target category text.
        target: String,
    },
}

/// One rule in final declaration and expansion order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LinkedRule {
    identifier: u32,
    tag: String,
    location: RulePlanLocation,
    kind: LinkedRuleKind,
    matching: Option<String>,
    restriction: Option<LinkedRuleRestriction>,
    introduces: Option<Vec<String>>,
    consumes: Option<Vec<String>>,
}

/// The premise categories one restricted relation admits.
///
/// The primary premise's target must be one of the primary targets, and the
/// secondary premise one of the secondary categories; an empty list leaves
/// its premise unrestricted.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LinkedRuleRestriction {
    primary_targets: Vec<String>,
    secondary_patterns: Vec<String>,
    primary_slot: usize,
    secondary_slot: Option<usize>,
}

impl LinkedRuleRestriction {
    pub(crate) const fn new(
        primary_targets: Vec<String>,
        secondary_patterns: Vec<String>,
        primary_slot: usize,
        secondary_slot: Option<usize>,
    ) -> Self {
        Self {
            primary_targets,
            secondary_patterns,
            primary_slot,
            secondary_slot,
        }
    }

    /// Return the finite primary-category targets.
    #[must_use]
    pub fn primary_targets(&self) -> &[String] {
        &self.primary_targets
    }

    /// Return the finite secondary-category patterns.
    #[must_use]
    pub fn secondary_patterns(&self) -> &[String] {
        &self.secondary_patterns
    }

    /// Return the premise slot whose target is restricted.
    #[must_use]
    pub const fn primary_slot(&self) -> usize {
        self.primary_slot
    }

    /// Return the premise slot the secondary categories restrict, if any.
    #[must_use]
    pub const fn secondary_slot(&self) -> Option<usize> {
        self.secondary_slot
    }
}

/// One deterministic rule schema in the indexed derivation theory.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LinkedTheoryConstructor {
    rule: String,
    constructor: String,
    inferred: Vec<String>,
    declared: Vec<String>,
    premises: usize,
    premise_indices: Vec<String>,
    conclusion_index: String,
}

impl LinkedTheoryConstructor {
    pub(crate) const fn new(
        rule: String,
        constructor: String,
        inferred: Vec<String>,
        declared: Vec<String>,
        premises: usize,
        premise_indices: Vec<String>,
        conclusion_index: String,
    ) -> Self {
        Self {
            rule,
            constructor,
            inferred,
            declared,
            premises,
            premise_indices,
            conclusion_index,
        }
    }

    /// Return the runtime rule name.
    #[must_use]
    pub fn rule(&self) -> &str {
        &self.rule
    }

    /// Return the stable theory-constructor name.
    #[must_use]
    pub fn constructor(&self) -> &str {
        &self.constructor
    }

    /// Return premise-determined index variables in encounter order.
    #[must_use]
    pub fn inferred(&self) -> &[String] {
        &self.inferred
    }

    /// Return explicit result index variables in encounter order.
    #[must_use]
    pub fn declared(&self) -> &[String] {
        &self.declared
    }

    /// Return the physical premise count.
    #[must_use]
    pub const fn premises(&self) -> usize {
        self.premises
    }

    /// Return complete canonical premise-index expressions.
    #[must_use]
    pub fn premise_indices(&self) -> &[String] {
        &self.premise_indices
    }

    /// Return the complete canonical conclusion-index expression.
    #[must_use]
    pub fn conclusion_index(&self) -> &str {
        &self.conclusion_index
    }
}

/// The finite constructor records and elaborated rules of one grammar theory.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LinkedTheory {
    constructors: Vec<LinkedTheoryConstructor>,
    elaborated: Vec<String>,
}

impl LinkedTheory {
    pub(crate) const fn new(
        constructors: Vec<LinkedTheoryConstructor>,
        elaborated: Vec<String>,
    ) -> Self {
        Self {
            constructors,
            elaborated,
        }
    }

    /// Return deterministic constructors in dense rule and schema order.
    #[must_use]
    pub fn constructors(&self) -> &[LinkedTheoryConstructor] {
        &self.constructors
    }

    /// Return rules whose individual steps carry elaborated proof witnesses.
    #[must_use]
    pub fn elaborated(&self) -> &[String] {
        &self.elaborated
    }
}

impl LinkedRule {
    pub(crate) fn new(
        identifier: u32,
        tag: String,
        location: RulePlanLocation,
        kind: LinkedRuleKind,
        matching: Option<String>,
        restriction: Option<LinkedRuleRestriction>,
        markers: (Option<Vec<String>>, Option<Vec<String>>),
    ) -> Self {
        let (introduces, consumes) = markers;
        Self {
            identifier,
            tag,
            location,
            kind,
            matching,
            restriction,
            introduces,
            consumes,
        }
    }

    /// Return the zero-based dense rule identifier.
    #[must_use]
    pub const fn identifier(&self) -> u32 {
        self.identifier
    }

    /// Return the installed rule-plugin tag.
    #[must_use]
    pub fn tag(&self) -> &str {
        &self.tag
    }

    /// Return the declaration row that produced this rule.
    #[must_use]
    pub const fn location(&self) -> RulePlanLocation {
        self.location
    }

    /// Return expansion-specific metadata.
    #[must_use]
    pub const fn kind(&self) -> &LinkedRuleKind {
        &self.kind
    }

    /// Return the canonical matching recipe, or the grammar matching.
    #[must_use]
    pub fn matching(&self) -> Option<&str> {
        self.matching.as_deref()
    }

    /// Return the compiled restriction, when declared.
    #[must_use]
    pub const fn restriction(&self) -> Option<&LinkedRuleRestriction> {
        self.restriction.as_ref()
    }

    /// Return categories introduced for reachability, when declared.
    #[must_use]
    pub fn introduces(&self) -> Option<&[String]> {
        self.introduces.as_deref()
    }

    /// Return categories consumed for reachability, when declared.
    #[must_use]
    pub fn consumes(&self) -> Option<&[String]> {
        self.consumes.as_deref()
    }
}

/// Every rule plugin and expansion selected by one grammar declaration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LinkedRulePlan {
    signature_fingerprint: String,
    fingerprint: String,
    plugin_tags: Vec<String>,
    rules: Vec<LinkedRule>,
    theory: LinkedTheory,
}

impl LinkedRulePlan {
    pub(crate) const fn new(
        signature_fingerprint: String,
        fingerprint: String,
        plugin_tags: Vec<String>,
        rules: Vec<LinkedRule>,
        theory: LinkedTheory,
    ) -> Self {
        Self {
            signature_fingerprint,
            fingerprint,
            plugin_tags,
            rules,
            theory,
        }
    }

    /// Return the linked-signature fingerprint this plan was checked against.
    #[must_use]
    pub fn signature_fingerprint(&self) -> &str {
        &self.signature_fingerprint
    }

    /// Return the deterministic fingerprint of the expanded rule metadata.
    #[must_use]
    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    /// Return distinct plugin tags in first-use order.
    #[must_use]
    pub fn plugin_tags(&self) -> &[String] {
        &self.plugin_tags
    }

    /// Return every rule in dense identifier order.
    #[must_use]
    pub fn rules(&self) -> &[LinkedRule] {
        &self.rules
    }

    /// Return the checked indexed derivation theory.
    #[must_use]
    pub const fn theory(&self) -> &LinkedTheory {
        &self.theory
    }

    /// Return every category a rule names outside its premises, once each in
    /// first-use order: finite targets, coercion endpoints, absorbed and
    /// clausal-adjunct atoms, and introduced or consumed categories.
    #[must_use]
    pub fn runtime_categories(&self) -> Vec<String> {
        let mut categories = Vec::new();
        for rule in &self.rules {
            match rule.kind() {
                LinkedRuleKind::Application { .. }
                | LinkedRuleKind::Composition { .. }
                | LinkedRuleKind::Substitution { .. } => {}
                LinkedRuleKind::TypeRaising {
                    targets, arguments, ..
                } => {
                    extend_unique(
                        &mut categories,
                        targets.iter().chain(arguments).flatten().cloned(),
                    );
                }
                LinkedRuleKind::ClausalAdjunct { recipe } => {
                    extend_unique(
                        &mut categories,
                        [
                            recipe.source_core().to_owned(),
                            recipe.argument().to_owned(),
                        ],
                    );
                }
                LinkedRuleKind::Pattern { recipe } | LinkedRuleKind::Extension { recipe } => {
                    extend_unique(
                        &mut categories,
                        recipe.targets().into_iter().flatten().cloned(),
                    );
                }
                LinkedRuleKind::Absorption { recipe } => {
                    extend_unique(
                        &mut categories,
                        recipe.pattern().targets().into_iter().flatten().cloned(),
                    );
                    extend_unique(&mut categories, recipe.absorbing().iter().cloned());
                }
                LinkedRuleKind::Coordination { recipe } => {
                    extend_unique(
                        &mut categories,
                        recipe.pattern().targets().into_iter().flatten().cloned(),
                    );
                }
                LinkedRuleKind::Coercion { source, target, .. } => {
                    extend_unique(&mut categories, [source.clone(), target.clone()]);
                }
            }
            if let Some(restriction) = rule.restriction() {
                extend_unique(
                    &mut categories,
                    restriction
                        .primary_targets()
                        .iter()
                        .chain(restriction.secondary_patterns())
                        .cloned(),
                );
            }
            extend_unique(
                &mut categories,
                rule.introduces().into_iter().flatten().cloned(),
            );
            extend_unique(
                &mut categories,
                rule.consumes().into_iter().flatten().cloned(),
            );
        }
        categories
    }

    /// Return the number of explicit source rows.
    #[must_use]
    pub fn explicit_rules(&self) -> usize {
        self.rules
            .iter()
            .filter(|rule| rule.location().source() == RulePlanSource::Explicit)
            .count()
    }
}

fn extend_unique(target: &mut Vec<String>, values: impl IntoIterator<Item = String>) {
    for value in values {
        if !target.contains(&value) {
            target.push(value);
        }
    }
}
