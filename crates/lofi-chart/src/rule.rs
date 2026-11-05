//! Native kernels for declared term rules.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt;
use std::sync::{Arc, Mutex};

use crate::term::{OperatorId, SortId, Term, TermError, TermStore};
use rayon::ThreadPool;
use rayon::prelude::*;

mod inverse;

pub use inverse::CompositionWitness;

/// One declared restriction on a composition spine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompositionRestriction {
    /// Only the deepest frame is constrained by harmonicity or crossing.
    Unrestricted,
    /// Every frame must use the principal slash.
    Kuhlmann,
    /// Every frame must use the deepest frame's slash.
    Steedman,
}

/// The binary-functor structure one rule reads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FunctorSpec {
    /// Operators recognized as functor constructors.
    pub operators: Vec<String>,
    /// Constructor taking its argument on the right.
    pub forward: String,
    /// Constructor taking its argument on the left.
    pub backward: String,
    /// Path from one functor to its result.
    pub result_path: Vec<usize>,
    /// Path from one functor to its argument.
    pub argument_path: Vec<usize>,
}

impl FunctorSpec {
    fn is_functor(&self, term: &Term) -> bool {
        self.operators
            .iter()
            .any(|operator| operator == term.operator_name())
    }

    fn opposite(&self, operator: &str) -> Option<&str> {
        if operator == self.forward {
            Some(&self.backward)
        } else if operator == self.backward {
            Some(&self.forward)
        } else {
            None
        }
    }

    fn orientations(&self, slash: &str) -> Vec<(usize, usize)> {
        let mut orientations = Vec::with_capacity(2);
        if slash == self.forward {
            orientations.push((0, 1));
        }
        if slash == self.backward {
            orientations.push((1, 0));
        }
        orientations
    }

    fn make(
        &self,
        store: &TermStore,
        slash: &str,
        result: Term,
        argument: Term,
    ) -> Result<Term, RuleError> {
        let arguments = if self.result_path == [0] && self.argument_path == [1] {
            vec![result, argument]
        } else if self.result_path == [1] && self.argument_path == [0] {
            vec![argument, result]
        } else {
            return Err(RuleError::InvalidFunctor(format!(
                "result path {:?} and argument path {:?} do not name the two children",
                self.result_path, self.argument_path
            )));
        };
        let operator = store
            .signature()
            .operators
            .iter()
            .position(|operator| operator.name == slash)
            .and_then(|position| OperatorId::try_from(position).ok())
            .ok_or_else(|| RuleError::InvalidFunctor(format!("unknown operator {slash:?}")))?;
        Ok(store.intern(operator, arguments, None)?)
    }
}

/// One generalized-composition declaration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompositionRule {
    /// Binary-functor structure.
    pub functors: FunctorSpec,
    /// Fixed order, or every positive finite order.
    pub order: Option<usize>,
    /// Principal functor constructor.
    pub slash: String,
    /// Whether the deepest frame is crossed.
    pub crossed: bool,
    /// Direction restriction over the container spine.
    pub restriction: CompositionRestriction,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum OutputFrame {
    Hole,
    Exact(Term),
    Node {
        operator: OperatorId,
        payload: Option<u64>,
        arguments: Vec<OutputFrame>,
    },
}

impl OutputFrame {
    fn from_term(term: &Term, path: &[usize]) -> Result<Self, RuleError> {
        let Some((slot, tail)) = path.split_first() else {
            return Ok(Self::Hole);
        };
        if *slot >= term.arguments().len() {
            return Err(RuleError::InvalidFunctor(format!(
                "path {path:?} leaves {:?}",
                term.key()
            )));
        }
        let arguments = term
            .arguments()
            .iter()
            .enumerate()
            .map(|(position, argument)| {
                if position == *slot {
                    Self::from_term(argument, tail)
                } else {
                    Ok(Self::Exact(argument.clone()))
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self::Node {
            operator: term.operator(),
            payload: term.payload(),
            arguments,
        })
    }

    fn fill(&self, store: &TermStore, result: &Term) -> Result<Term, RuleError> {
        match self {
            Self::Hole => Ok(result.clone()),
            Self::Exact(term) => Ok(term.clone()),
            Self::Node {
                operator,
                payload,
                arguments,
            } => Ok(store.intern(
                *operator,
                arguments
                    .iter()
                    .map(|argument| argument.fill(store, result))
                    .collect::<Result<Vec<_>, _>>()?,
                *payload,
            )?),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum OutputFactorKind {
    Result(Term),
    Frames(Vec<OutputFrame>),
}

/// One exact, independently groupable contribution to a binary rule output.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RuleOutputFactor(OutputFactorKind);

/// Combine complementary rule-output factors without revisiting exact premises.
///
/// # Errors
///
/// Returns a typed term error when rebuilding the exact conclusion fails.
pub fn combine_output_factors(
    store: &TermStore,
    left: &RuleOutputFactor,
    right: &RuleOutputFactor,
) -> Result<Option<Term>, RuleError> {
    let (result, frames) = match (&left.0, &right.0) {
        (OutputFactorKind::Result(result), OutputFactorKind::Frames(frames))
        | (OutputFactorKind::Frames(frames), OutputFactorKind::Result(result)) => {
            (result.clone(), frames)
        }
        _ => return Ok(None),
    };
    frames
        .iter()
        .rev()
        .try_fold(result, |built, frame| frame.fill(store, &built))
        .map(Some)
}

/// One generalized-substitution declaration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SubstitutionRule {
    /// Binary-functor structure.
    pub functors: FunctorSpec,
    /// Fixed positive order, or every positive finite order.
    pub order: Option<usize>,
    /// Principal functor constructor.
    pub slash: String,
    /// Whether the deepest shared frame is crossed.
    pub crossed: bool,
}

/// One finite type-raising declaration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RaisingRule {
    /// Binary-functor structure.
    pub functors: FunctorSpec,
    /// Outer functor constructor.
    pub slash: String,
    /// Exact declared targets in declaration order, or `None` for every
    /// target, which forward application cannot enumerate.
    pub targets: Option<Vec<Term>>,
    /// Categories that may raise, compared under the rule's matching, or
    /// `None` for every category.
    pub arguments: Option<Vec<Term>>,
}

/// An operator occurrence in one term pattern.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum OperatorPattern {
    /// One fixed operator, or a name absent from the signature.
    Fixed(Option<OperatorId>),
    /// One operator variable shared by name.
    Variable(String),
}

/// A payload occurrence in one term pattern.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum PayloadPattern {
    /// No payload.
    None,
    /// One fixed integer payload.
    Integer(String),
    /// One payload variable shared by name.
    Variable(String),
}

/// One declarative term pattern.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Pattern {
    /// A term variable of one sort.
    Term {
        /// Shared variable name.
        name: String,
        /// Required result sort.
        sort: SortId,
    },
    /// An operator node.
    Node {
        /// Fixed or variable operator.
        operator: OperatorPattern,
        /// Argument patterns.
        arguments: Vec<Self>,
        /// Fixed or variable payload.
        payload: PayloadPattern,
    },
    /// A bounded one-hole context.
    Spine {
        /// Shared context name.
        name: String,
        /// Pattern below the context.
        core: Box<Self>,
        /// Operators admitted in each frame.
        operators: Vec<OperatorId>,
        /// Child slot followed through each frame.
        slot: usize,
        /// Minimum frame count.
        min_order: usize,
        /// Maximum frame count, or no finite maximum.
        max_order: Option<usize>,
    },
}

/// One oriented clause of a pattern-rule declaration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PatternClause {
    /// Premise patterns after ground parts have been represented.
    pub premises: Vec<Pattern>,
    /// Conclusion pattern used to construct exact outputs.
    pub conclusion: Pattern,
}

/// One finite family of pattern-rule clauses.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PatternRule {
    /// Clauses in declaration order.
    pub clauses: Vec<PatternClause>,
}

/// Which premise a finite absorption rule removes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AbsorptionSide {
    /// Remove the left premise.
    Left,
    /// Remove the right premise.
    Right,
}

/// One guarded pattern relation that absorbs exactly one finite category.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AbsorptionRule {
    /// Underlying declarative relation.
    pub pattern: PatternRule,
    /// Finite absorbing categories.
    pub absorbing: Vec<Term>,
    /// Premise side removed by the rule.
    pub side: AbsorptionSide,
}

/// One phase of feature-sensitive finite coordination.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoordinationPhase {
    /// Turn an unmarked category into its marked counterpart.
    Mark,
    /// Merge an unmarked category with its marked counterpart.
    Merge,
}

/// Required relation between the second premise and an unmarked conclusion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoreRelation {
    /// The two categories share one matching representative.
    Same,
    /// The two categories have different matching representatives.
    Different,
}

/// One finite coordination phase wrapped around a declarative pattern rule.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoordinationPhaseRule {
    /// Underlying declarative relation.
    pub pattern: PatternRule,
    /// Exact unmarked/marked pairs.
    pub pairs: Vec<(Term, Term)>,
    /// Unary constructor whose application to any category marks it, if declared.
    pub marking: Option<OperatorId>,
    /// Optional finite forward targets.
    pub targets: Option<Vec<Term>>,
    /// Marking or merging phase.
    pub phase: CoordinationPhase,
    /// Matching relation required of a mark phase.
    pub core_relation: CoreRelation,
}

/// One native term-rule family.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RuleKernel {
    /// A relation with no forward-enumerable clauses.
    Empty,
    /// Generalized application or composition.
    Composition(CompositionRule),
    /// Generalized substitution.
    Substitution(SubstitutionRule),
    /// Finite type raising.
    Raising(RaisingRule),
    /// A declarative pattern relation.
    Pattern(PatternRule),
    /// A guarded finite absorption relation.
    Absorption(AbsorptionRule),
    /// One finite feature-sensitive coordination phase.
    CoordinationPhase(CoordinationPhaseRule),
    /// A rule whose instances admit only declared premise categories.
    Restricted(Box<RestrictedRule>),
}

/// Categories a restricted rule instance admits in its premises.
///
/// The target of a category is the category reached by following the
/// declared result path through every functor constructor; an atom is its
/// own target. Both lists are compared under the rule's matching.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PremiseRestriction {
    /// The primary premise slot.
    pub primary: usize,
    /// Targets the primary premise may have, or `None` for every target.
    pub targets: Option<Vec<Term>>,
    /// The secondary premise slot and the categories it may hold, if restricted.
    pub secondary: Option<(usize, Vec<Term>)>,
    /// Each functor constructor with the path from one of its terms to its result.
    pub descents: Vec<(String, Vec<usize>)>,
}

/// One rule kernel restricted to declared premise categories.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RestrictedRule {
    /// The unrestricted relation.
    pub rule: RuleKernel,
    /// The premise categories its instances admit.
    pub restriction: PremiseRestriction,
}

impl PremiseRestriction {
    fn target(&self, term: &Term) -> Result<Term, RuleError> {
        let mut current = term.clone();
        while let Some((_, path)) = self
            .descents
            .iter()
            .find(|(operator, _)| operator == current.operator_name())
        {
            current = at_path(&current, path)?;
        }
        Ok(current)
    }

    /// Whether one term may fill one premise slot.
    ///
    /// # Errors
    ///
    /// Returns a typed error when a declared result path is invalid or a
    /// compared category has no registered representative.
    pub fn admits_slot(
        &self,
        representatives: &RepresentativeTable,
        term: &Term,
        slot: usize,
    ) -> Result<bool, RuleError> {
        if slot == self.primary
            && let Some(targets) = &self.targets
        {
            let target = self.target(term)?;
            if !any_equivalent(representatives, &target, targets)? {
                return Ok(false);
            }
        }
        if let Some((secondary, categories)) = &self.secondary
            && slot == *secondary
            && !any_equivalent(representatives, term, categories)?
        {
            return Ok(false);
        }
        Ok(true)
    }

    fn admits_all(
        &self,
        representatives: &RepresentativeTable,
        inputs: &[Term],
    ) -> Result<bool, RuleError> {
        for (slot, term) in inputs.iter().enumerate() {
            if !self.admits_slot(representatives, term, slot)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn constants(&self) -> impl Iterator<Item = &Term> {
        self.targets
            .iter()
            .flatten()
            .chain(self.secondary.iter().flat_map(|(_, categories)| categories))
    }
}

fn any_equivalent(
    representatives: &RepresentativeTable,
    term: &Term,
    categories: &[Term],
) -> Result<bool, RuleError> {
    for category in categories {
        if representatives.equivalent(term, category)? {
            return Ok(true);
        }
    }
    Ok(false)
}

/// One exact native key shared by the two premises of a joined rule.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum RuleJoinKey {
    /// A binary rule whose premises require no shared join binding.
    Direct,
    /// Shared category under one composition orientation.
    Composition {
        /// Orientation in the rule's declared order.
        orientation: u8,
        /// Fixed order of a bounded rule; unbounded rules share one fragment.
        order: Option<usize>,
        /// Matching representative shared by principal and container.
        representative: Term,
    },
    /// Shared category at one substitution spine depth.
    Substitution {
        /// Orientation in the rule's declared order.
        orientation: u8,
        /// Positive substitution order.
        order: usize,
        /// Parallel outer frames, represented by slash and argument category.
        frames: Vec<(String, Term)>,
        /// Matching representative shared by functor and container.
        representative: Term,
    },
    /// Shared bindings of one declarative pattern clause.
    Pattern {
        /// Clause position in declaration order.
        clause: usize,
        /// Values bound in more than one premise, ordered by declared name.
        bindings: Vec<RuleJoinBinding>,
    },
    /// One finite absorbing alternative under a declarative pattern clause.
    Absorption {
        /// Clause position in declaration order.
        clause: usize,
        /// Values bound in more than one premise, ordered by declared name.
        bindings: Vec<RuleJoinBinding>,
        /// Declared absorbing-category position.
        absorbing: usize,
    },
    /// One finite merge alternative under a declarative coordination clause.
    CoordinationPhase {
        /// Clause position in declaration order.
        clause: usize,
        /// Values bound in more than one premise, ordered by declared name.
        bindings: Vec<RuleJoinBinding>,
        /// Declared unmarked/marked pair position.
        pair: usize,
    },
}

/// One value shared between premise patterns in a declarative join.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum RuleJoinBinding {
    /// One represented term variable.
    Term(Term),
    /// One operator-valued payload variable.
    Operator(OperatorId),
    /// One integer-valued payload variable.
    Integer(u64),
    /// One represented context spine and its followed child slots.
    Spine(Vec<(Term, usize)>),
}

/// A native rule refusal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RuleError {
    /// A declared normal-form table is inconsistent with the reachability rules.
    InvalidNormalForm(String),
    /// A rule collection cannot be represented by the chart's rule identifier.
    TooManyRules {
        /// Supplied number of rules.
        rules: usize,
        /// Greatest representable number of rules.
        maximum: usize,
    },
    /// A rule identifier lies outside its compiled program.
    UnknownRule {
        /// Supplied rule identifier.
        rule: u16,
        /// Number of rules in the program.
        rules: usize,
    },
    /// No matching representative was registered for a visited term.
    MissingRepresentative(Term),
    /// A declared term path leaves a node.
    InvalidPath {
        /// Operator at the node where the path failed.
        operator: String,
        /// Unavailable child position.
        position: usize,
    },
    /// A functor declaration names no physical orientation.
    InvalidSlash(String),
    /// A functor declaration cannot construct terms.
    InvalidFunctor(String),
    /// Candidate indexing omitted a join-key row required by a declared rule.
    MissingJoinMetadata {
        /// Rule position in program order.
        rule: u16,
        /// Premise slot within the rule.
        slot: u8,
    },
    /// Witnesses of one composition step carry spines of different lengths.
    InconsistentOrder(Vec<usize>),
    /// A kernel was asked for a direction it does not declare.
    UnsupportedInverse(&'static str),
    /// Term construction was refused by its native store.
    Term(TermError),
}

impl fmt::Display for RuleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidNormalForm(reason) => {
                write!(
                    formatter,
                    "invalid normal-form reachability table: {reason}"
                )
            }
            Self::TooManyRules { rules, maximum } => write!(
                formatter,
                "a reachability program has {rules} rules, but at most {maximum} are representable"
            ),
            Self::UnknownRule { rule, rules } => {
                write!(
                    formatter,
                    "rule id {rule} is outside a program of {rules} rules"
                )
            }
            Self::MissingRepresentative(term) => write!(
                formatter,
                "no matching representative was registered for {term:?}"
            ),
            Self::InvalidPath { operator, position } => write!(
                formatter,
                "operator {operator:?} has no child at position {position}"
            ),
            Self::InvalidSlash(operator) => {
                write!(
                    formatter,
                    "operator id {operator} is not a distinguished slash"
                )
            }
            Self::InvalidFunctor(reason) => write!(formatter, "invalid functor view: {reason}"),
            Self::MissingJoinMetadata { rule, slot } => write!(
                formatter,
                "candidate indexing omitted join metadata for rule {rule} slot {slot}"
            ),
            Self::InconsistentOrder(orders) => write!(
                formatter,
                "composition witnesses disagree on the applied order: {orders:?}"
            ),
            Self::UnsupportedInverse(direction) => {
                write!(formatter, "the rule kernel declares no {direction}")
            }
            Self::Term(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for RuleError {}

impl From<TermError> for RuleError {
    fn from(error: TermError) -> Self {
        Self::Term(error)
    }
}

/// One native rule-matching representative function.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RepresentativeKernel {
    /// Keep each exact term as its own representative.
    Identity,
    /// Use each term's precomputed feature-erasure core.
    Core,
    /// Override finitely many exact terms before applying a base function.
    Refined {
        /// Representative function outside the finite override domain.
        base: Box<Self>,
        /// Exact input-to-representative overrides.
        overrides: HashMap<Term, Term>,
    },
    /// Intersect two independently declared equivalence relations.
    Product {
        /// First component relation.
        left: Box<Self>,
        /// Second component relation.
        right: Box<Self>,
    },
    /// Identify terms up to consistent renaming of free variables by sort.
    Subsumption,
}

impl RepresentativeKernel {
    fn direct_representative(&self, term: &Term) -> Option<Term> {
        match self {
            Self::Identity => Some(term.clone()),
            Self::Core => Some(term.core()),
            Self::Refined { base, overrides } => overrides
                .get(term)
                .cloned()
                .or_else(|| base.direct_representative(term)),
            Self::Product { .. } | Self::Subsumption => None,
        }
    }

    fn class_key(&self, term: &Term) -> RepresentativeKey {
        match self {
            Self::Identity => RepresentativeKey::Term(term.clone()),
            Self::Core => RepresentativeKey::Term(term.core()),
            Self::Refined { base, overrides } => overrides.get(term).map_or_else(
                || base.class_key(term),
                |representative| RepresentativeKey::Term(representative.clone()),
            ),
            Self::Product { left, right } => RepresentativeKey::Product(
                Box::new(left.class_key(term)),
                Box::new(right.class_key(term)),
            ),
            Self::Subsumption => RepresentativeKey::Subsumption(subsumption_key(term)),
        }
    }

    fn candidate_key(&self, term: &Term) -> Term {
        match self {
            Self::Core => term.core(),
            Self::Identity | Self::Refined { .. } | Self::Product { .. } | Self::Subsumption => {
                term.clone()
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum RepresentativeKey {
    Term(Term),
    Product(Box<Self>, Box<Self>),
    Subsumption(SubsumptionKey),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum SubsumptionKey {
    Free {
        sort: SortId,
        class: usize,
    },
    Node {
        operator: OperatorId,
        payload: Option<u64>,
        arguments: Vec<Self>,
    },
}

fn subsumption_key(root: &Term) -> SubsumptionKey {
    fn visit(
        term: &Term,
        variables: &mut HashMap<(SortId, u64), usize>,
        next: &mut HashMap<SortId, usize>,
    ) -> SubsumptionKey {
        if term.operator_name().starts_with("#fvar[") {
            let payload = term.payload().unwrap_or(u64::MAX);
            let held = variables.get(&(term.sort(), payload)).copied();
            let class = held.unwrap_or_else(|| {
                let class = *next.entry(term.sort()).or_default();
                next.insert(term.sort(), class + 1);
                variables.insert((term.sort(), payload), class);
                class
            });
            return SubsumptionKey::Free {
                sort: term.sort(),
                class,
            };
        }
        SubsumptionKey::Node {
            operator: term.operator(),
            payload: term.payload(),
            arguments: term
                .arguments()
                .iter()
                .map(|argument| visit(argument, variables, next))
                .collect(),
        }
    }

    visit(root, &mut HashMap::new(), &mut HashMap::new())
}

/// Matching representatives computed once outside the repeated rule loop.
#[derive(Clone, Debug, Default)]
pub struct RepresentativeTable {
    representatives: HashMap<u64, Term>,
    classes: HashMap<RepresentativeKey, Term>,
}

/// One immutable rule entry in a parallel native program.
#[derive(Clone, Debug)]
pub struct ProgramRule {
    /// Pure rule kernel.
    pub kernel: RuleKernel,
    /// Opaque identity of the rule's declared matching relation.
    pub matching: usize,
    /// Compiled representative function.
    pub representatives: RepresentativeKernel,
    /// Finite constants visited by the rule outside its premises.
    pub constants: Vec<Term>,
}

/// One exact application submitted to a native rule program.
#[derive(Clone, Debug)]
pub struct RuleApplication {
    /// Rule identifier in program order.
    pub rule: u16,
    /// Opaque term-signature scope shared by matching tables.
    pub scope: usize,
    /// Store used to construct exact conclusions.
    pub store: TermStore,
    /// Exact premises in rule order.
    pub inputs: Vec<Term>,
}

/// One binary output-computation fragment applied to exact premises.
///
/// The join key owns the schema or orientation binding. Exact chart
/// multiplicity remains outside this value, so every pair that shares the
/// fragment reuses one declared computation.
#[derive(Clone, Debug)]
pub struct RuleFragmentApplication {
    /// Rule identifier in program order.
    pub rule: u16,
    /// Opaque term-signature scope shared by matching tables.
    pub scope: usize,
    /// Store used to construct exact conclusions.
    pub store: TermStore,
    /// Exact premises in rule order.
    pub inputs: Vec<Term>,
    /// Exact schema or orientation binding shared by the premises.
    pub fragment: RuleJoinKey,
}

/// One exact category submitted to native candidate indexing.
#[derive(Clone, Debug)]
pub struct RuleCategoryQuery {
    /// Exact category term.
    pub term: Term,
    /// Opaque term-signature scope shared by matching tables.
    pub scope: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct GroundPatternKey {
    operator: OperatorId,
    arguments: Vec<Self>,
    payload: Option<u64>,
}

enum GroundPattern {
    Dynamic,
    Impossible,
    Key(GroundPatternKey),
}

fn fixed_payload(value: &str) -> Option<u64> {
    value.parse().ok()
}

impl GroundPatternKey {
    fn matches(&self, term: &Term) -> bool {
        self.operator == term.operator()
            && self.payload == term.payload()
            && self.arguments.len() == term.arguments().len()
            && self
                .arguments
                .iter()
                .zip(term.arguments())
                .all(|(pattern, argument)| pattern.matches(argument))
    }

    fn from_pattern(pattern: &Pattern) -> GroundPattern {
        let Pattern::Node {
            operator,
            arguments,
            payload,
        } = pattern
        else {
            return GroundPattern::Dynamic;
        };
        let OperatorPattern::Fixed(operator) = operator else {
            return GroundPattern::Dynamic;
        };
        let Some(operator) = operator else {
            return GroundPattern::Impossible;
        };
        let payload = match payload {
            PayloadPattern::None => None,
            PayloadPattern::Integer(value) => {
                let Some(value) = fixed_payload(value) else {
                    return GroundPattern::Impossible;
                };
                Some(value)
            }
            PayloadPattern::Variable(_) => return GroundPattern::Dynamic,
        };
        let mut children = Vec::with_capacity(arguments.len());
        for argument in arguments {
            match Self::from_pattern(argument) {
                GroundPattern::Key(child) => children.push(child),
                GroundPattern::Dynamic => return GroundPattern::Dynamic,
                GroundPattern::Impossible => return GroundPattern::Impossible,
            }
        }
        GroundPattern::Key(Self {
            operator: *operator,
            arguments: children,
            payload,
        })
    }
}

/// Candidate-index metadata for one exact category.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuleCategoryMeta {
    /// Unary rules whose pattern admits the category.
    pub unary: Vec<u16>,
    /// Unjoined binary rule slots whose pattern admits the category.
    pub viable: Vec<(u16, u8)>,
    /// Joined binary rule slots and their exact shared representatives.
    pub joins: Vec<(u16, u8, Arc<[RuleJoinKey]>)>,
    /// Order-zero composition outputs determined by one principal category.
    pub factors: Vec<(u16, u8, RuleJoinKey, Term)>,
    /// Positive-order composition and substitution contributions grouped before pairing.
    pub output_factors: Vec<(u16, u8, RuleJoinKey, RuleOutputFactor)>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct JoinCacheKey {
    rule: u16,
    slot: u8,
    scope: usize,
    category: Term,
}

type GroundUnaryBucket = Vec<(GroundPatternKey, Vec<u16>)>;
type GroundUnaryIndex = HashMap<usize, HashMap<(OperatorId, Option<u64>), GroundUnaryBucket>>;

/// One compiled native rule inventory and its matching tables.
#[derive(Debug)]
pub struct RuleProgram {
    rules: Vec<ProgramRule>,
    tables: Mutex<HashMap<(usize, usize), Arc<RepresentativeTable>>>,
    ground_unary: GroundUnaryIndex,
    indexed_unary: Vec<bool>,
    join_cache: Mutex<HashMap<JoinCacheKey, Arc<[RuleJoinKey]>>>,
}

impl RepresentativeTable {
    /// Whether this exact term already has a declared representative.
    #[must_use]
    pub fn contains(&self, term: &Term) -> bool {
        self.representatives.contains_key(&term.serial())
    }

    /// Register one exact term's declared representative.
    pub fn register(&mut self, term: &Term, representative: Term) {
        self.representatives.insert(term.serial(), representative);
    }

    /// Register one exact term under a compiled equivalence relation.
    pub fn register_with(&mut self, representatives: &RepresentativeKernel, term: &Term) {
        let key = representatives.class_key(term);
        let representative = representatives
            .direct_representative(term)
            .unwrap_or_else(|| {
                self.classes
                    .entry(key.clone())
                    .or_insert_with(|| term.clone())
                    .clone()
            });
        self.classes
            .entry(key)
            .or_insert_with(|| representative.clone());
        self.representatives.insert(term.serial(), representative);
    }

    /// Return the registered representative for one exact term.
    ///
    /// # Errors
    ///
    /// Returns [`RuleError::MissingRepresentative`] when the term has not been
    /// registered under this table's compiled relation.
    pub fn representative(&self, term: &Term) -> Result<Term, RuleError> {
        self.key(term).cloned()
    }

    fn key(&self, term: &Term) -> Result<&Term, RuleError> {
        self.representatives
            .get(&term.serial())
            .ok_or_else(|| RuleError::MissingRepresentative(term.clone()))
    }

    fn equivalent(&self, left: &Term, right: &Term) -> Result<bool, RuleError> {
        Ok(self.key(left)?.is_identical(self.key(right)?))
    }

    fn equivalent_constructed(&self, left: &Term, right: &Term) -> bool {
        let left = self.representatives.get(&left.serial()).unwrap_or(left);
        let right = self.representatives.get(&right.serial()).unwrap_or(right);
        left.is_identical(right)
    }
}

fn register_tree(
    representatives: &RepresentativeKernel,
    root: &Term,
    table: &mut RepresentativeTable,
) {
    if table.contains(root) {
        return;
    }
    let mut pending = vec![root.clone()];
    while let Some(term) = pending.pop() {
        if table.contains(&term) {
            continue;
        }
        table.register_with(representatives, &term);
        pending.extend(term.arguments().iter().cloned());
    }
}

impl RuleProgram {
    /// Compile one immutable rule inventory with empty matching tables.
    #[must_use]
    pub fn new(mut rules: Vec<ProgramRule>) -> Self {
        let mut matchings = Vec::<RepresentativeKernel>::new();
        for rule in &mut rules {
            if let Some(position) = matchings
                .iter()
                .position(|representatives| representatives == &rule.representatives)
            {
                rule.matching = position;
            } else {
                rule.matching = matchings.len();
                matchings.push(rule.representatives.clone());
            }
        }
        let mut ground_unary = GroundUnaryIndex::new();
        let mut indexed_unary = vec![false; rules.len()];
        for (position, rule) in rules.iter().enumerate() {
            let Some(keys) = rule.kernel.ground_unary_keys() else {
                continue;
            };
            indexed_unary[position] = true;
            let Ok(position) = u16::try_from(position) else {
                continue;
            };
            for key in keys {
                let bucket = ground_unary
                    .entry(rule.matching)
                    .or_default()
                    .entry((key.operator, key.payload))
                    .or_default();
                if let Some((_, positions)) = bucket.iter_mut().find(|(held, _)| held == &key) {
                    positions.push(position);
                } else {
                    bucket.push((key, vec![position]));
                }
            }
        }
        Self {
            rules,
            tables: Mutex::new(HashMap::new()),
            ground_unary,
            indexed_unary,
            join_cache: Mutex::new(HashMap::new()),
        }
    }

    /// Return the number of rules in program order.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.rules.len()
    }

    /// Return whether the program contains no rules.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// Return whether one rule exposes exact binary join keys.
    #[must_use]
    pub fn uses_join_keys(&self, rule: u16) -> Option<bool> {
        self.rules
            .get(usize::from(rule))
            .map(|entry| entry.kernel.uses_join_keys())
    }

    fn category_meta_from(
        &self,
        shapes: &[(usize, bool)],
        scanned: &[usize],
        query: &RuleCategoryQuery,
        tables: &HashMap<(usize, usize), Arc<RepresentativeTable>>,
        join_keys: &HashMap<JoinCacheKey, Result<Arc<[RuleJoinKey]>, RuleError>>,
    ) -> Result<RuleCategoryMeta, RuleError> {
        let mut unary = Vec::new();
        let mut viable = Vec::new();
        let mut joins = Vec::new();
        let mut factors = Vec::new();
        let mut output_factors = Vec::new();

        for position in scanned.iter().copied() {
            let Some((rule, (arity, complete))) =
                self.rules.get(position).zip(shapes.get(position))
            else {
                continue;
            };
            if !matches!(*arity, 1 | 2) {
                continue;
            }
            let position = u16::try_from(position).map_err(|_| RuleError::UnknownRule {
                rule: u16::MAX,
                rules: self.rules.len(),
            })?;
            if !*complete {
                if *arity == 1 {
                    unary.push(position);
                } else {
                    viable.extend([(position, 0), (position, 1)]);
                }
                continue;
            }

            let table = tables
                .get(&(rule.matching, query.scope))
                .cloned()
                .unwrap_or_default();
            if *arity == 1 {
                if rule.kernel.slot_viable(&table, &query.term, 0)? {
                    unary.push(position);
                }
                continue;
            }
            if rule.kernel.uses_join_keys() {
                for slot in 0_u8..2 {
                    if !rule
                        .kernel
                        .join_slot_viable(&table, &query.term, usize::from(slot))?
                    {
                        continue;
                    }
                    let key = JoinCacheKey {
                        rule: position,
                        slot,
                        scope: query.scope,
                        category: rule.representatives.candidate_key(&query.term),
                    };
                    let keys =
                        join_keys
                            .get(&key)
                            .cloned()
                            .ok_or(RuleError::MissingJoinMetadata {
                                rule: position,
                                slot,
                            })??;
                    joins.push((position, slot, keys.clone()));
                    for key in keys.iter() {
                        for factor in rule.kernel.output_factors(
                            &table,
                            &query.term,
                            usize::from(slot),
                            key,
                        )? {
                            output_factors.push((position, slot, key.clone(), factor));
                        }
                    }
                    factors.extend(
                        rule.kernel
                            .join_factors(&table, &query.term, usize::from(slot))?
                            .into_iter()
                            .map(|(key, output)| (position, slot, key, output)),
                    );
                }
            } else {
                for slot in 0_u8..2 {
                    if rule
                        .kernel
                        .slot_viable(&table, &query.term, usize::from(slot))?
                    {
                        viable.push((position, slot));
                    }
                }
            }
        }

        unary.extend(self.ground_unary_candidates(shapes, query, tables)?);
        unary.sort_unstable();

        Ok(RuleCategoryMeta {
            unary,
            viable,
            joins,
            factors,
            output_factors,
        })
    }

    fn ground_unary_candidates(
        &self,
        shapes: &[(usize, bool)],
        query: &RuleCategoryQuery,
        tables: &HashMap<(usize, usize), Arc<RepresentativeTable>>,
    ) -> Result<Vec<u16>, RuleError> {
        let mut candidates = Vec::new();
        for (matching, index) in &self.ground_unary {
            let table = tables
                .get(&(*matching, query.scope))
                .cloned()
                .unwrap_or_default();
            let represented = table.key(&query.term)?;
            if let Some(bucket) = index.get(&(represented.operator(), represented.payload())) {
                for (_, positions) in bucket.iter().filter(|(key, _)| key.matches(represented)) {
                    candidates.extend(positions.iter().copied().filter(|position| {
                        shapes
                            .get(usize::from(*position))
                            .is_some_and(|(arity, complete)| *arity == 1 && *complete)
                    }));
                }
            }
        }
        Ok(candidates)
    }

    /// Compile candidate-index metadata for exact categories in submission order.
    ///
    /// Matching tables are updated once before a supplied worker pool evaluates
    /// independent categories in parallel.
    #[must_use]
    pub fn category_meta_batch(
        &self,
        shapes: &[(usize, bool)],
        queries: &[RuleCategoryQuery],
        workers: Option<&ThreadPool>,
    ) -> Vec<Result<RuleCategoryMeta, RuleError>> {
        let tables = self.matching_tables(shapes, queries);
        let scanned = shapes
            .iter()
            .enumerate()
            .filter_map(|(position, (arity, complete))| {
                (!(*arity == 1
                    && *complete
                    && self.indexed_unary.get(position).copied().unwrap_or(false)))
                .then_some(position)
            })
            .collect::<Vec<_>>();
        let requested = self.requested_joins(shapes, queries, &scanned, &tables);
        let mut join_keys = HashMap::with_capacity(requested.len());
        let missing = {
            let cache = self
                .join_cache
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            requested
                .into_iter()
                .filter_map(|(key, term)| {
                    if let Some(keys) = cache.get(&key) {
                        join_keys.insert(key, Ok(keys.clone()));
                        None
                    } else {
                        Some((key, term))
                    }
                })
                .collect::<Vec<_>>()
        };
        let compile_join = |(key, term): &(JoinCacheKey, Term)| {
            let result = self
                .rules
                .get(usize::from(key.rule))
                .ok_or(RuleError::UnknownRule {
                    rule: key.rule,
                    rules: self.rules.len(),
                })
                .and_then(|rule| {
                    let table = tables
                        .get(&(rule.matching, key.scope))
                        .cloned()
                        .unwrap_or_default();
                    rule.kernel
                        .join_keys(&table, term, usize::from(key.slot))
                        .map(Arc::from)
                });
            (key.clone(), result)
        };
        let compiled = workers.map_or_else(
            || missing.iter().map(compile_join).collect::<Vec<_>>(),
            |pool| pool.install(|| missing.par_iter().map(compile_join).collect::<Vec<_>>()),
        );
        {
            let mut cache = self
                .join_cache
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            for (key, result) in compiled {
                match result {
                    Ok(keys) => {
                        let held = cache.entry(key.clone()).or_insert(keys).clone();
                        join_keys.insert(key, Ok(held));
                    }
                    Err(error) => {
                        join_keys.insert(key, Err(error));
                    }
                }
            }
            drop(cache);
        }
        let compile = |query: &RuleCategoryQuery| {
            self.category_meta_from(shapes, &scanned, query, &tables, &join_keys)
        };

        workers.map_or_else(
            || queries.iter().map(compile).collect(),
            |pool| pool.install(|| queries.par_iter().map(compile).collect()),
        )
    }

    fn matching_tables(
        &self,
        shapes: &[(usize, bool)],
        queries: &[RuleCategoryQuery],
    ) -> HashMap<(usize, usize), Arc<RepresentativeTable>> {
        let mut tables = self
            .tables
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let scopes: BTreeSet<usize> = queries.iter().map(|query| query.scope).collect();
        let mut matchings = HashMap::new();
        for rule in self.rules.iter().take(shapes.len()) {
            matchings
                .entry(rule.matching)
                .or_insert(&rule.representatives);
            for scope in &scopes {
                let table = tables.entry((rule.matching, *scope)).or_default();
                for constant in &rule.constants {
                    register_tree(&rule.representatives, constant, Arc::make_mut(table));
                }
            }
        }
        for query in queries {
            for (matching, representatives) in &matchings {
                let table = tables.entry((*matching, query.scope)).or_default();
                register_tree(representatives, &query.term, Arc::make_mut(table));
            }
        }
        let snapshots = tables.clone();
        drop(tables);
        snapshots
    }

    fn requested_joins(
        &self,
        shapes: &[(usize, bool)],
        queries: &[RuleCategoryQuery],
        scanned: &[usize],
        tables: &HashMap<(usize, usize), Arc<RepresentativeTable>>,
    ) -> Vec<(JoinCacheKey, Term)> {
        let mut seen = HashSet::new();
        let mut requested = Vec::new();
        for query in queries {
            for position in scanned.iter().copied() {
                let Some((rule, (2, true))) = self.rules.get(position).zip(shapes.get(position))
                else {
                    continue;
                };
                if !rule.kernel.uses_join_keys() {
                    continue;
                }
                let Ok(rule_id) = u16::try_from(position) else {
                    continue;
                };
                let table = tables
                    .get(&(rule.matching, query.scope))
                    .cloned()
                    .unwrap_or_default();
                for slot in 0_u8..2 {
                    if matches!(
                        rule.kernel
                            .join_slot_viable(&table, &query.term, usize::from(slot)),
                        Ok(false)
                    ) {
                        continue;
                    }
                    let key = JoinCacheKey {
                        rule: rule_id,
                        slot,
                        scope: query.scope,
                        category: rule.representatives.candidate_key(&query.term),
                    };
                    if seen.insert(key.clone()) {
                        requested.push((key, query.term.clone()));
                    }
                }
            }
        }
        requested
    }

    /// Apply independent exact rule applications in submission order.
    ///
    /// A supplied worker pool runs the pure applications in parallel. Matching
    /// tables are snapshotted before worker execution, so no application holds
    /// the shared table lock.
    #[must_use]
    pub fn apply_batch(
        &self,
        applications: &[RuleApplication],
        workers: Option<&ThreadPool>,
    ) -> Vec<Result<(u16, Vec<Term>), RuleError>> {
        let snapshots = {
            let mut tables = self
                .tables
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            for application in applications {
                let Some(rule) = self.rules.get(usize::from(application.rule)) else {
                    continue;
                };
                let table = tables
                    .entry((rule.matching, application.scope))
                    .or_default();
                let mutable = Arc::make_mut(table);
                for term in application.inputs.iter().chain(&rule.constants) {
                    register_tree(&rule.representatives, term, mutable);
                }
            }
            let snapshots = applications
                .iter()
                .map(|application| {
                    let rule = self.rules.get(usize::from(application.rule))?;
                    tables
                        .get(&(rule.matching, application.scope))
                        .map(Arc::clone)
                })
                .collect::<Vec<_>>();
            drop(tables);
            snapshots
        };

        let apply =
            |(application, table): (&RuleApplication, &Option<Arc<RepresentativeTable>>)| {
                let Some((rule, table)) = self
                    .rules
                    .get(usize::from(application.rule))
                    .zip(table.as_deref())
                else {
                    return Err(RuleError::UnknownRule {
                        rule: application.rule,
                        rules: self.rules.len(),
                    });
                };
                rule.kernel
                    .apply(&application.store, table, &application.inputs)
                    .map(|outputs| (application.rule, outputs))
            };

        workers.map_or_else(
            || applications.iter().zip(&snapshots).map(apply).collect(),
            |pool| pool.install(|| applications.par_iter().zip(&snapshots).map(apply).collect()),
        )
    }

    /// Apply independent binary output-computation fragments in submission order.
    ///
    /// A supplied worker pool runs the exact materializations in parallel. The
    /// fragment fixes the rule schema before execution, so application does not
    /// rediscover a witness already established by candidate indexing.
    #[must_use]
    pub fn apply_fragment_batch(
        &self,
        applications: &[RuleFragmentApplication],
        workers: Option<&ThreadPool>,
    ) -> Vec<Result<(u16, Vec<Term>), RuleError>> {
        let snapshots = {
            let mut tables = self
                .tables
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            for application in applications {
                let Some(rule) = self.rules.get(usize::from(application.rule)) else {
                    continue;
                };
                let table = tables
                    .entry((rule.matching, application.scope))
                    .or_default();
                let mutable = Arc::make_mut(table);
                for term in application.inputs.iter().chain(&rule.constants) {
                    register_tree(&rule.representatives, term, mutable);
                }
            }
            let snapshots = applications
                .iter()
                .map(|application| {
                    let rule = self.rules.get(usize::from(application.rule))?;
                    tables
                        .get(&(rule.matching, application.scope))
                        .map(Arc::clone)
                })
                .collect::<Vec<_>>();
            drop(tables);
            snapshots
        };

        let apply = |(application, table): (
            &RuleFragmentApplication,
            &Option<Arc<RepresentativeTable>>,
        )| {
            let Some((rule, table)) = self
                .rules
                .get(usize::from(application.rule))
                .zip(table.as_deref())
            else {
                return Err(RuleError::UnknownRule {
                    rule: application.rule,
                    rules: self.rules.len(),
                });
            };
            rule.kernel
                .apply_fragment(
                    &application.store,
                    table,
                    &application.inputs,
                    &application.fragment,
                )
                .map(|outputs| (application.rule, outputs))
        };

        workers.map_or_else(
            || applications.iter().zip(&snapshots).map(apply).collect(),
            |pool| pool.install(|| applications.par_iter().zip(&snapshots).map(apply).collect()),
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Frame {
    term: Term,
    slot: usize,
}

fn walk_frames(
    functors: &FunctorSpec,
    term: &Term,
    order: usize,
) -> Result<Option<(Vec<Frame>, Term)>, RuleError> {
    let mut current = term.clone();
    let mut frames = Vec::with_capacity(order);
    for _ in 0..order {
        if !functors.is_functor(&current) {
            return Ok(None);
        }
        frames.push(Frame {
            term: current.clone(),
            slot: *functors.result_path.first().unwrap_or(&0),
        });
        current = at_path(&current, &functors.result_path)?;
    }
    Ok(Some((frames, current)))
}

fn at_path(term: &Term, path: &[usize]) -> Result<Term, RuleError> {
    let mut current = term.clone();
    for position in path {
        current =
            current
                .arguments()
                .get(*position)
                .cloned()
                .ok_or_else(|| RuleError::InvalidPath {
                    operator: current.operator_name().to_owned(),
                    position: *position,
                })?;
    }
    Ok(current)
}

fn replace_path(
    store: &TermStore,
    term: &Term,
    path: &[usize],
    replacement: Term,
) -> Result<Term, RuleError> {
    let Some((&position, below)) = path.split_first() else {
        return Ok(replacement);
    };
    let child = term
        .arguments()
        .get(position)
        .ok_or_else(|| RuleError::InvalidPath {
            operator: term.operator_name().to_owned(),
            position,
        })?;
    let replaced = replace_path(store, child, below, replacement)?;
    let mut arguments = term.arguments().to_vec();
    arguments[position] = replaced;
    Ok(store.intern(term.operator(), arguments, term.payload())?)
}

fn rebuild(
    store: &TermStore,
    frames: &[Frame],
    result_path: &[usize],
    result: Term,
) -> Result<Term, RuleError> {
    frames.iter().rev().try_fold(result, |built, frame| {
        replace_path(store, &frame.term, result_path, built)
    })
}

fn push_unique(outputs: &mut Vec<Term>, output: Term) {
    if !outputs.iter().any(|known| known.is_identical(&output)) {
        outputs.push(output);
    }
}

fn push_representative_unique(
    outputs: &mut Vec<Term>,
    output: Term,
    representatives: &RepresentativeTable,
) -> Result<(), RuleError> {
    for known in outputs.iter() {
        if representatives.equivalent(known, &output)? {
            return Ok(());
        }
    }
    outputs.push(output);
    Ok(())
}

impl CompositionRule {
    fn admitted_order(&self, order: usize) -> bool {
        self.order.map_or(order > 0, |declared| order == declared)
    }

    fn licensed(&self, frames: &[Frame]) -> bool {
        let Some(deepest) = frames.last().map(|frame| frame.term.operator_name()) else {
            return true;
        };
        let required = if self.crossed {
            let Some(opposite) = self.functors.opposite(&self.slash) else {
                return false;
            };
            opposite
        } else {
            &self.slash
        };
        let deepest_ok = match self.restriction {
            CompositionRestriction::Unrestricted => self
                .functors
                .operators
                .iter()
                .any(|operator| operator == deepest),
            CompositionRestriction::Kuhlmann | CompositionRestriction::Steedman => {
                deepest == required
            }
        };
        if !deepest_ok {
            return false;
        }
        frames[..frames.len() - 1]
            .iter()
            .all(|frame| match self.restriction {
                CompositionRestriction::Unrestricted | CompositionRestriction::Kuhlmann => self
                    .functors
                    .operators
                    .iter()
                    .any(|operator| operator == frame.term.operator_name()),
                CompositionRestriction::Steedman => frame.term.operator_name() == deepest,
            })
    }

    fn apply(
        &self,
        store: &TermStore,
        representatives: &RepresentativeTable,
        inputs: &[Term],
    ) -> Result<Vec<Term>, RuleError> {
        if inputs.len() != 2 {
            return Ok(Vec::new());
        }
        let orientations = self.functors.orientations(&self.slash);
        if orientations.is_empty() {
            return Err(RuleError::InvalidSlash(self.slash.clone()));
        }
        let mut outputs = Vec::new();
        for (principal_slot, container_slot) in orientations {
            let principal = &inputs[principal_slot];
            if !self.functors.is_functor(principal) || principal.operator_name() != self.slash {
                continue;
            }
            let argument = at_path(principal, &self.functors.argument_path)?;
            let result = at_path(principal, &self.functors.result_path)?;
            let mut current = inputs[container_slot].clone();
            let mut frames = Vec::new();
            loop {
                if self.admitted_order(frames.len())
                    && self.licensed(&frames)
                    && representatives.equivalent(&argument, &current)?
                {
                    push_unique(
                        &mut outputs,
                        rebuild(store, &frames, &self.functors.result_path, result.clone())?,
                    );
                }
                if self.order == Some(frames.len()) || !self.functors.is_functor(&current) {
                    break;
                }
                frames.push(Frame {
                    term: current.clone(),
                    slot: *self.functors.result_path.first().unwrap_or(&0),
                });
                current = at_path(&current, &self.functors.result_path)?;
            }
        }
        Ok(outputs)
    }

    fn apply_fragment(
        &self,
        store: &TermStore,
        inputs: &[Term],
        orientation: u8,
        order: usize,
    ) -> Result<Vec<Term>, RuleError> {
        let Some((principal_slot, container_slot)) = self
            .functors
            .orientations(&self.slash)
            .get(usize::from(orientation))
            .copied()
        else {
            return Ok(Vec::new());
        };
        let Some(principal) = inputs.get(principal_slot) else {
            return Ok(Vec::new());
        };
        let Some(container) = inputs.get(container_slot) else {
            return Ok(Vec::new());
        };
        let result = at_path(principal, &self.functors.result_path)?;
        let Some((frames, _)) = walk_frames(&self.functors, container, order)? else {
            return Ok(Vec::new());
        };
        Ok(vec![rebuild(
            store,
            &frames,
            &self.functors.result_path,
            result,
        )?])
    }

    fn join_keys(
        &self,
        representatives: &RepresentativeTable,
        term: &Term,
        slot: usize,
    ) -> Result<Vec<RuleJoinKey>, RuleError> {
        let mut keys = Vec::new();
        for (at, (principal_slot, container_slot)) in self
            .functors
            .orientations(&self.slash)
            .into_iter()
            .enumerate()
        {
            let orientation = u8::try_from(at).unwrap_or(u8::MAX);
            if slot == principal_slot
                && self.functors.is_functor(term)
                && term.operator_name() == self.slash
            {
                let argument = at_path(term, &self.functors.argument_path)?;
                let key = RuleJoinKey::Composition {
                    orientation,
                    order: self.order,
                    representative: representatives.key(&argument)?.clone(),
                };
                if !keys.contains(&key) {
                    keys.push(key);
                }
            }
            if slot != container_slot {
                continue;
            }
            let mut current = term.clone();
            let mut frames = Vec::new();
            loop {
                if self.admitted_order(frames.len()) && self.licensed(&frames) {
                    let key = RuleJoinKey::Composition {
                        orientation,
                        order: self.order,
                        representative: representatives.key(&current)?.clone(),
                    };
                    if !keys.contains(&key) {
                        keys.push(key);
                    }
                }
                if self.order == Some(frames.len()) || !self.functors.is_functor(&current) {
                    break;
                }
                frames.push(Frame {
                    term: current.clone(),
                    slot: *self.functors.result_path.first().unwrap_or(&0),
                });
                current = at_path(&current, &self.functors.result_path)?;
            }
        }
        Ok(keys)
    }
}

impl SubstitutionRule {
    fn walk(&self, term: &Term, order: usize) -> Result<Option<(Vec<Frame>, Term)>, RuleError> {
        walk_frames(&self.functors, term, order)
    }

    fn orders(&self, left: &Term, right: &Term) -> Result<Vec<usize>, RuleError> {
        if let Some(order) = self.order {
            return Ok(vec![order]);
        }
        let mut orders = Vec::new();
        let mut left = left.clone();
        let mut right = right.clone();
        while self.functors.is_functor(&left) && self.functors.is_functor(&right) {
            orders.push(orders.len() + 1);
            left = at_path(&left, &self.functors.result_path)?;
            right = at_path(&right, &self.functors.result_path)?;
        }
        Ok(orders)
    }

    fn apply(
        &self,
        store: &TermStore,
        representatives: &RepresentativeTable,
        inputs: &[Term],
    ) -> Result<Vec<Term>, RuleError> {
        if inputs.len() != 2 {
            return Ok(Vec::new());
        }
        let orientations = self.functors.orientations(&self.slash);
        if orientations.is_empty() {
            return Err(RuleError::InvalidSlash(self.slash.clone()));
        }
        let required = if self.crossed {
            self.functors
                .opposite(&self.slash)
                .ok_or_else(|| RuleError::InvalidSlash(self.slash.clone()))?
        } else {
            &self.slash
        };
        let mut outputs = Vec::new();
        for (functor_slot, container_slot) in orientations {
            let functor = &inputs[functor_slot];
            let container = &inputs[container_slot];
            for order in self.orders(functor, container)? {
                let Some((functor_frames, functor_foot)) = self.walk(functor, order)? else {
                    continue;
                };
                let Some((container_frames, container_foot)) = self.walk(container, order)? else {
                    continue;
                };
                let mut parallel = true;
                for (left, right) in functor_frames.iter().zip(&container_frames) {
                    let left_argument = at_path(&left.term, &self.functors.argument_path)?;
                    let right_argument = at_path(&right.term, &self.functors.argument_path)?;
                    if left.term.operator_name() != right.term.operator_name()
                        || !representatives.equivalent(&left_argument, &right_argument)?
                    {
                        parallel = false;
                        break;
                    }
                }
                if !parallel
                    || container_frames
                        .last()
                        .is_none_or(|frame| frame.term.operator_name() != required)
                    || !self.functors.is_functor(&functor_foot)
                    || functor_foot.operator_name() != self.slash
                {
                    continue;
                }
                let result = at_path(&functor_foot, &self.functors.result_path)?;
                let argument = at_path(&functor_foot, &self.functors.argument_path)?;
                if representatives.equivalent(&argument, &container_foot)? {
                    push_unique(
                        &mut outputs,
                        rebuild(store, &container_frames, &self.functors.result_path, result)?,
                    );
                }
            }
        }
        Ok(outputs)
    }

    fn apply_fragment(
        &self,
        store: &TermStore,
        inputs: &[Term],
        orientation: u8,
        order: usize,
    ) -> Result<Vec<Term>, RuleError> {
        let Some((functor_slot, container_slot)) = self
            .functors
            .orientations(&self.slash)
            .get(usize::from(orientation))
            .copied()
        else {
            return Ok(Vec::new());
        };
        let Some(functor) = inputs.get(functor_slot) else {
            return Ok(Vec::new());
        };
        let Some(container) = inputs.get(container_slot) else {
            return Ok(Vec::new());
        };
        let Some((_, foot)) = self.walk(functor, order)? else {
            return Ok(Vec::new());
        };
        let Some((frames, _)) = self.walk(container, order)? else {
            return Ok(Vec::new());
        };
        let result = at_path(&foot, &self.functors.result_path)?;
        Ok(vec![rebuild(
            store,
            &frames,
            &self.functors.result_path,
            result,
        )?])
    }

    fn join_keys(
        &self,
        representatives: &RepresentativeTable,
        term: &Term,
        slot: usize,
    ) -> Result<Vec<RuleJoinKey>, RuleError> {
        let mut keys = Vec::new();
        for (at, (functor_slot, container_slot)) in self
            .functors
            .orientations(&self.slash)
            .into_iter()
            .enumerate()
        {
            let orientation = u8::try_from(at).unwrap_or(u8::MAX);
            let mut current = term.clone();
            let mut order = 0;
            while self.functors.is_functor(&current)
                && self.order.is_none_or(|declared| order < declared)
            {
                order += 1;
                current = at_path(&current, &self.functors.result_path)?;
                if self.order.is_some_and(|declared| order != declared) {
                    continue;
                }
                let Some((frames, foot)) = self.walk(term, order)? else {
                    continue;
                };
                let parallel = frames
                    .iter()
                    .map(|frame| {
                        let argument = at_path(&frame.term, &self.functors.argument_path)?;
                        Ok((
                            frame.term.operator_name().to_owned(),
                            representatives.key(&argument)?.clone(),
                        ))
                    })
                    .collect::<Result<Vec<_>, RuleError>>()?;
                if slot == container_slot {
                    let required = if self.crossed {
                        self.functors
                            .opposite(&self.slash)
                            .ok_or_else(|| RuleError::InvalidSlash(self.slash.clone()))?
                    } else {
                        &self.slash
                    };
                    if frames
                        .last()
                        .is_none_or(|frame| frame.term.operator_name() != required)
                    {
                        continue;
                    }
                    let key = RuleJoinKey::Substitution {
                        orientation,
                        order,
                        frames: parallel.clone(),
                        representative: representatives.key(&foot)?.clone(),
                    };
                    if !keys.contains(&key) {
                        keys.push(key);
                    }
                }
                if slot == functor_slot
                    && self.functors.is_functor(&foot)
                    && foot.operator_name() == self.slash
                {
                    let argument = at_path(&foot, &self.functors.argument_path)?;
                    let key = RuleJoinKey::Substitution {
                        orientation,
                        order,
                        frames: parallel,
                        representative: representatives.key(&argument)?.clone(),
                    };
                    if !keys.contains(&key) {
                        keys.push(key);
                    }
                }
            }
        }
        Ok(keys)
    }
}

impl RaisingRule {
    /// Whether one premise is a declared raisable category.
    fn raises(
        &self,
        representatives: &RepresentativeTable,
        input: &Term,
    ) -> Result<bool, RuleError> {
        let Some(arguments) = &self.arguments else {
            return Ok(true);
        };
        for argument in arguments {
            if representatives.equivalent(input, argument)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn apply(
        &self,
        store: &TermStore,
        representatives: &RepresentativeTable,
        inputs: &[Term],
    ) -> Result<Vec<Term>, RuleError> {
        if inputs.len() != 1 {
            return Ok(Vec::new());
        }
        let Some(targets) = &self.targets else {
            return Ok(Vec::new());
        };
        if !self.raises(representatives, &inputs[0])? {
            return Ok(Vec::new());
        }
        let opposite = self
            .functors
            .opposite(&self.slash)
            .ok_or_else(|| RuleError::InvalidSlash(self.slash.clone()))?;
        targets
            .iter()
            .map(|target| {
                let inner =
                    self.functors
                        .make(store, opposite, target.clone(), inputs[0].clone())?;
                self.functors
                    .make(store, &self.slash, target.clone(), inner)
            })
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum BoundPayload {
    Operator(OperatorId),
    Integer(u64),
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct BindingMap<T>(Vec<(String, T)>);

impl<T> Default for BindingMap<T> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl<T> BindingMap<T> {
    fn get(&self, name: &str) -> Option<&T> {
        self.0
            .iter()
            .find_map(|(held, value)| (held == name).then_some(value))
    }

    fn insert(&mut self, name: String, value: T) {
        if let Some((_, held)) = self.0.iter_mut().find(|(held, _)| held == &name) {
            *held = value;
        } else {
            self.0.push((name, value));
        }
    }

    #[cfg(test)]
    fn iter(&self) -> impl Iterator<Item = (&String, &T)> {
        self.0.iter().map(|(name, value)| (name, value))
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Bindings {
    terms: BindingMap<Term>,
    payloads: BindingMap<BoundPayload>,
    spines: BindingMap<Vec<Frame>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum BindingName {
    Term(String),
    Payload(String),
    Spine(String),
}

fn binding_names(pattern: &Pattern, names: &mut BTreeSet<BindingName>) {
    match pattern {
        Pattern::Term { name, .. } => {
            names.insert(BindingName::Term(name.clone()));
        }
        Pattern::Node {
            operator,
            arguments,
            payload,
        } => {
            if let OperatorPattern::Variable(name) = operator {
                names.insert(BindingName::Payload(name.clone()));
            }
            if let PayloadPattern::Variable(name) = payload {
                names.insert(BindingName::Payload(name.clone()));
            }
            for argument in arguments {
                binding_names(argument, names);
            }
        }
        Pattern::Spine { name, core, .. } => {
            names.insert(BindingName::Spine(name.clone()));
            binding_names(core, names);
        }
    }
}

fn shared_binding_names(clause: &PatternClause) -> Vec<BindingName> {
    let mut counts = HashMap::new();
    for premise in &clause.premises {
        let mut names = BTreeSet::new();
        binding_names(premise, &mut names);
        for name in names {
            *counts.entry(name).or_insert(0_usize) += 1;
        }
    }
    let mut shared = counts
        .into_iter()
        .filter_map(|(name, count)| (count > 1).then_some(name))
        .collect::<Vec<_>>();
    shared.sort();
    shared
}

fn join_bindings(bindings: &Bindings, names: &[BindingName]) -> Option<Vec<RuleJoinBinding>> {
    names
        .iter()
        .map(|name| match name {
            BindingName::Term(name) => bindings.terms.get(name).cloned().map(RuleJoinBinding::Term),
            BindingName::Payload(name) => bindings.payloads.get(name).map(|value| match value {
                BoundPayload::Operator(operator) => RuleJoinBinding::Operator(*operator),
                BoundPayload::Integer(value) => RuleJoinBinding::Integer(*value),
            }),
            BindingName::Spine(name) => bindings.spines.get(name).map(|frames| {
                RuleJoinBinding::Spine(
                    frames
                        .iter()
                        .map(|frame| (frame.term.clone(), frame.slot))
                        .collect(),
                )
            }),
        })
        .collect()
}

fn bind_term(bindings: &mut Bindings, name: &str, term: &Term) -> bool {
    if let Some(known) = bindings.terms.get(name) {
        return known.is_identical(term);
    }
    bindings.terms.insert(name.to_owned(), term.clone());
    true
}

fn bind_term_core(bindings: &mut Bindings, name: &str, term: &Term) -> bool {
    let core = term.core();
    if let Some(known) = bindings.terms.get(name) {
        return known.is_identical(&core);
    }
    bindings.terms.insert(name.to_owned(), core);
    true
}

fn bind_payload(bindings: &mut Bindings, name: &str, value: BoundPayload) -> bool {
    if let Some(known) = bindings.payloads.get(name) {
        return known == &value;
    }
    bindings.payloads.insert(name.to_owned(), value);
    true
}

fn bind_spine(bindings: &mut Bindings, name: &str, frames: &[Frame]) -> bool {
    if let Some(known) = bindings.spines.get(name) {
        return known == frames;
    }
    bindings.spines.insert(name.to_owned(), frames.to_vec());
    true
}

fn spine_candidates(
    term: &Term,
    operators: &[OperatorId],
    slot: usize,
    max_order: Option<usize>,
) -> (Vec<Frame>, Vec<Term>) {
    let mut frames = Vec::new();
    let mut remainders = vec![term.clone()];
    let mut current = term.clone();
    while max_order.is_none_or(|maximum| frames.len() < maximum)
        && operators.contains(&current.operator())
        && slot < current.arguments().len()
    {
        frames.push(Frame {
            term: current.clone(),
            slot,
        });
        current = current.arguments()[slot].clone();
        remainders.push(current.clone());
    }
    (frames, remainders)
}

fn match_one(pattern: &Pattern, term: &Term, bindings: &mut Bindings) -> bool {
    match pattern {
        Pattern::Term { name, sort } => term.sort() == *sort && bind_term(bindings, name, term),
        Pattern::Node {
            operator,
            arguments,
            payload,
        } => {
            let operator_matches = match operator {
                OperatorPattern::Fixed(expected) => expected == &Some(term.operator()),
                OperatorPattern::Variable(name) => {
                    bind_payload(bindings, name, BoundPayload::Operator(term.operator()))
                }
            };
            if !operator_matches || arguments.len() != term.arguments().len() {
                return false;
            }
            let payload_matches = match payload {
                PayloadPattern::None => term.payload().is_none(),
                PayloadPattern::Integer(expected) => {
                    fixed_payload(expected).is_some_and(|expected| term.payload() == Some(expected))
                }
                PayloadPattern::Variable(name) => term.payload().is_some_and(|value| {
                    bind_payload(bindings, name, BoundPayload::Integer(value))
                }),
            };
            payload_matches
                && arguments
                    .iter()
                    .zip(term.arguments())
                    .all(|(child, value)| match_one(child, value, bindings))
        }
        Pattern::Spine {
            name,
            core,
            operators,
            slot,
            min_order,
            max_order,
        } => {
            let (frames, remainders) = spine_candidates(term, operators, *slot, *max_order);
            for order in (*min_order..=frames.len()).rev() {
                let mut candidate = bindings.clone();
                if match_one(core, &remainders[order], &mut candidate)
                    && bind_spine(&mut candidate, name, &frames[..order])
                {
                    *bindings = candidate;
                    return true;
                }
            }
            false
        }
    }
}

fn visit_pattern_matches(
    pattern: &Pattern,
    term: &Term,
    mut bindings: Bindings,
    binds_cores: bool,
    emit: &mut dyn FnMut(Bindings),
) {
    if !binds_cores && !has_spine(pattern) {
        if match_one(pattern, term, &mut bindings) {
            emit(bindings);
        }
        return;
    }
    match pattern {
        Pattern::Term { name, sort } => {
            let bound = if binds_cores {
                bind_term_core(&mut bindings, name, term)
            } else {
                bind_term(&mut bindings, name, term)
            };
            if term.sort() == *sort && bound {
                emit(bindings);
            }
        }
        Pattern::Node {
            operator,
            arguments,
            payload,
        } => {
            let core = term.core();
            let candidates = if binds_cores && !core.is_identical(term) {
                vec![term.clone(), core]
            } else {
                vec![term.clone()]
            };
            for candidate in candidates {
                if arguments.len() != candidate.arguments().len() {
                    continue;
                }
                let mut candidate_bindings = bindings.clone();
                let operator_matches = match operator {
                    OperatorPattern::Fixed(expected) => expected == &Some(candidate.operator()),
                    OperatorPattern::Variable(name) => bind_payload(
                        &mut candidate_bindings,
                        name,
                        BoundPayload::Operator(candidate.operator()),
                    ),
                };
                let payload_matches = match payload {
                    PayloadPattern::None => candidate.payload().is_none(),
                    PayloadPattern::Integer(expected) => fixed_payload(expected)
                        .is_some_and(|expected| candidate.payload() == Some(expected)),
                    PayloadPattern::Variable(name) => candidate.payload().is_some_and(|value| {
                        bind_payload(&mut candidate_bindings, name, BoundPayload::Integer(value))
                    }),
                };
                if operator_matches && payload_matches {
                    visit_argument_matches(
                        arguments,
                        candidate.arguments(),
                        0,
                        candidate_bindings,
                        binds_cores,
                        emit,
                    );
                }
            }
        }
        Pattern::Spine {
            name,
            core,
            operators,
            slot,
            min_order,
            max_order,
        } => {
            let (frames, remainders) = spine_candidates(term, operators, *slot, *max_order);
            for order in *min_order..=frames.len() {
                let mut matched = |mut candidate: Bindings| {
                    if bind_spine(&mut candidate, name, &frames[..order]) {
                        emit(candidate);
                    }
                };
                visit_pattern_matches(
                    core,
                    &remainders[order],
                    bindings.clone(),
                    binds_cores,
                    &mut matched,
                );
            }
        }
    }
}

fn visit_argument_matches(
    patterns: &[Pattern],
    terms: &[Term],
    at: usize,
    bindings: Bindings,
    binds_cores: bool,
    emit: &mut dyn FnMut(Bindings),
) {
    let Some(pattern) = patterns.get(at) else {
        emit(bindings);
        return;
    };
    let term = &terms[at];
    let mut matched = |bindings| {
        visit_argument_matches(patterns, terms, at + 1, bindings, binds_cores, emit);
    };
    visit_pattern_matches(pattern, term, bindings, binds_cores, &mut matched);
}

fn visit_clause_matches(
    patterns: &[Pattern],
    terms: &[Term],
    at: usize,
    bindings: Bindings,
    binds_cores: bool,
    emit: &mut dyn FnMut(Bindings),
) {
    let Some(pattern) = patterns.get(at) else {
        emit(bindings);
        return;
    };
    let term = &terms[at];
    let mut matched = |bindings| {
        visit_clause_matches(patterns, terms, at + 1, bindings, binds_cores, emit);
    };
    visit_pattern_matches(pattern, term, bindings, binds_cores, &mut matched);
}

/// One exact, partially bound reachability-schema environment.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct ReachabilityBindings(Bindings);

/// The binding names needed by a remaining reachability computation.
#[derive(Debug)]
pub(crate) struct ReachabilityProjection(Vec<BindingName>);

/// One exact environment projection used by a reachability join.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum ReachabilityProjectionKey {
    Empty,
    One(Option<RuleJoinBinding>),
    Two(Option<RuleJoinBinding>, Option<RuleJoinBinding>),
    Three(
        Option<RuleJoinBinding>,
        Option<RuleJoinBinding>,
        Option<RuleJoinBinding>,
    ),
    Many(Box<[Option<RuleJoinBinding>]>),
}

pub(crate) fn reachability_extend_into(
    pattern: &Pattern,
    term: &Term,
    bindings: &ReachabilityBindings,
    matched: &mut Vec<ReachabilityBindings>,
) {
    matched.clear();
    visit_pattern_matches(pattern, term, bindings.0.clone(), false, &mut |bindings| {
        matched.push(ReachabilityBindings(bindings));
    });
}

/// Collect the binding names referenced by a set of remaining patterns.
#[cfg(test)]
pub(crate) fn reachability_projection<'a>(
    patterns: impl IntoIterator<Item = &'a Pattern>,
) -> ReachabilityProjection {
    let mut names = BTreeSet::new();
    for pattern in patterns {
        binding_names(pattern, &mut names);
    }
    ReachabilityProjection(names.into_iter().collect())
}

/// Collect conclusion bindings and bindings shared by multiple premises.
#[cfg(test)]
pub(crate) fn reachability_separator_projection(
    premises: &[Pattern],
    conclusion: &Pattern,
) -> ReachabilityProjection {
    let mut retained = BTreeSet::new();
    binding_names(conclusion, &mut retained);
    let mut counts = HashMap::new();
    for premise in premises {
        let mut names = BTreeSet::new();
        binding_names(premise, &mut names);
        for name in names {
            *counts.entry(name).or_insert(0_usize) += 1;
        }
    }
    retained.extend(
        counts
            .into_iter()
            .filter_map(|(name, count)| (count > 1).then_some(name)),
    );
    ReachabilityProjection(retained.into_iter().collect())
}

/// Combine two binding projections in canonical name order.
#[cfg(test)]
pub(crate) fn reachability_union_projection(
    left: &ReachabilityProjection,
    right: &ReachabilityProjection,
) -> ReachabilityProjection {
    ReachabilityProjection(
        left.0
            .iter()
            .chain(&right.0)
            .cloned()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect(),
    )
}

/// Collect binding names shared by processed premises and one next premise.
pub(crate) fn reachability_shared_projection<'a>(
    processed: impl IntoIterator<Item = &'a Pattern>,
    next: &Pattern,
) -> ReachabilityProjection {
    let mut left = BTreeSet::new();
    for pattern in processed {
        binding_names(pattern, &mut left);
    }
    let mut right = BTreeSet::new();
    binding_names(next, &mut right);
    ReachabilityProjection(left.intersection(&right).cloned().collect())
}

/// Return one environment's values for a declared binding projection.
pub(crate) fn reachability_projection_key(
    bindings: &ReachabilityBindings,
    projection: &ReachabilityProjection,
) -> ReachabilityProjectionKey {
    let value = |name: &BindingName| match name {
        BindingName::Term(name) => bindings
            .0
            .terms
            .get(name)
            .cloned()
            .map(RuleJoinBinding::Term),
        BindingName::Payload(name) => bindings.0.payloads.get(name).map(|value| match value {
            BoundPayload::Operator(operator) => RuleJoinBinding::Operator(*operator),
            BoundPayload::Integer(value) => RuleJoinBinding::Integer(*value),
        }),
        BindingName::Spine(name) => bindings.0.spines.get(name).map(|frames| {
            RuleJoinBinding::Spine(
                frames
                    .iter()
                    .map(|frame| (frame.term.clone(), frame.slot))
                    .collect(),
            )
        }),
    };
    match projection.0.as_slice() {
        [] => ReachabilityProjectionKey::Empty,
        [one] => ReachabilityProjectionKey::One(value(one)),
        [one, two] => ReachabilityProjectionKey::Two(value(one), value(two)),
        [one, two, three] => ReachabilityProjectionKey::Three(value(one), value(two), value(three)),
        names => ReachabilityProjectionKey::Many(
            names
                .iter()
                .map(value)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        ),
    }
}

/// Merge two compatible environments directly into one retained projection.
#[cfg(test)]
pub(crate) fn reachability_merge_project(
    left: &ReachabilityBindings,
    right: &ReachabilityBindings,
    projection: &ReachabilityProjection,
) -> Option<(ReachabilityBindings, ReachabilityProjectionKey)> {
    let terms_match = left.0.terms.iter().all(|(name, term)| {
        right
            .0
            .terms
            .get(name)
            .is_none_or(|held| held.is_identical(term))
    });
    let payloads_match = left
        .0
        .payloads
        .iter()
        .all(|(name, value)| right.0.payloads.get(name).is_none_or(|held| held == value));
    let spines_match = left
        .0
        .spines
        .iter()
        .all(|(name, frames)| right.0.spines.get(name).is_none_or(|held| held == frames));
    if !(terms_match && payloads_match && spines_match) {
        return None;
    }

    let mut projected = Bindings::default();
    for name in &projection.0 {
        match name {
            BindingName::Term(name) => {
                if let Some(term) = left.0.terms.get(name).or_else(|| right.0.terms.get(name)) {
                    projected.terms.insert(name.clone(), term.clone());
                }
            }
            BindingName::Payload(name) => {
                if let Some(value) = left
                    .0
                    .payloads
                    .get(name)
                    .or_else(|| right.0.payloads.get(name))
                {
                    projected.payloads.insert(name.clone(), value.clone());
                }
            }
            BindingName::Spine(name) => {
                if let Some(frames) = left.0.spines.get(name).or_else(|| right.0.spines.get(name)) {
                    projected.spines.insert(name.clone(), frames.clone());
                }
            }
        }
    }
    let projected = ReachabilityBindings(projected);
    let key = reachability_projection_key(&projected, projection);
    Some((projected, key))
}

#[cfg(test)]
pub(crate) fn reachability_build(
    pattern: &Pattern,
    store: &TermStore,
    bindings: &ReachabilityBindings,
) -> Result<Option<Term>, RuleError> {
    build_pattern(pattern, store, &bindings.0)
}

/// Return whether an exact premise tuple satisfies one joint reachability row.
#[must_use]
pub fn reachability_patterns_match(patterns: &[Pattern], terms: &[Term]) -> bool {
    if patterns.len() != terms.len() {
        return false;
    }
    let mut matched = false;
    visit_clause_matches(patterns, terms, 0, Bindings::default(), false, &mut |_| {
        matched = true;
    });
    matched
}

/// Return whether one reachability schema jointly admits premises and conclusion.
#[must_use]
pub(crate) fn reachability_application_matches(
    premises: &[Pattern],
    conclusion: &Pattern,
    inputs: &[Term],
    output: &Term,
) -> bool {
    if premises.len() != inputs.len() {
        return false;
    }
    let mut matched = false;
    visit_clause_matches(
        premises,
        inputs,
        0,
        Bindings::default(),
        true,
        &mut |bindings| {
            visit_pattern_matches(conclusion, output, bindings, true, &mut |_| {
                matched = true;
            });
        },
    );
    matched
}

fn has_spine(pattern: &Pattern) -> bool {
    match pattern {
        Pattern::Term { .. } => false,
        Pattern::Node { arguments, .. } => arguments.iter().any(has_spine),
        Pattern::Spine { .. } => true,
    }
}

/// Apply reachability schemata to one exact premise tuple.
///
/// # Errors
///
/// Returns a typed rule or term error when a matched conclusion cannot be
/// constructed in the supplied term store.
pub fn reachability_conclusions(
    schemata: &[(Vec<Pattern>, Pattern)],
    store: &TermStore,
    terms: &[Term],
) -> Result<Vec<Term>, RuleError> {
    let mut outputs = Vec::new();
    reachability_conclusions_into(schemata, store, terms, &mut outputs)?;
    Ok(outputs)
}

pub(crate) fn reachability_conclusions_into(
    schemata: &[(Vec<Pattern>, Pattern)],
    store: &TermStore,
    terms: &[Term],
    outputs: &mut Vec<Term>,
) -> Result<(), RuleError> {
    outputs.clear();
    for (premises, conclusion) in schemata {
        if premises.len() != terms.len() {
            continue;
        }
        let mut failure = None;
        visit_clause_matches(
            premises,
            terms,
            0,
            Bindings::default(),
            false,
            &mut |bindings| match build_pattern(conclusion, store, &bindings) {
                Ok(Some(output)) => {
                    if !outputs.iter().any(|held: &Term| held.is_identical(&output)) {
                        outputs.push(output);
                    }
                }
                Ok(None) => {}
                Err(error) => failure = Some(error),
            },
        );
        if let Some(error) = failure {
            return Err(error);
        }
    }
    Ok(())
}

fn root_matches(pattern: &Pattern, term: &Term, bindings: &Bindings) -> Vec<Bindings> {
    let Pattern::Spine {
        name,
        core,
        operators,
        slot,
        min_order,
        max_order,
    } = pattern
    else {
        let mut candidate = bindings.clone();
        return match_one(pattern, term, &mut candidate)
            .then_some(candidate)
            .into_iter()
            .collect();
    };
    let mut frames = Vec::new();
    let mut remainders = vec![term.clone()];
    let mut current = term.clone();
    while max_order.is_none_or(|maximum| frames.len() < maximum)
        && operators.contains(&current.operator())
        && *slot < current.arguments().len()
    {
        frames.push(Frame {
            term: current.clone(),
            slot: *slot,
        });
        current = current.arguments()[*slot].clone();
        remainders.push(current.clone());
    }
    (*min_order..=frames.len())
        .filter_map(|order| {
            let mut candidate = bindings.clone();
            (match_one(core, &remainders[order], &mut candidate)
                && bind_spine(&mut candidate, name, &frames[..order]))
            .then_some(candidate)
        })
        .collect()
}

fn root_may_match(pattern: &Pattern, term: &Term) -> bool {
    match pattern {
        Pattern::Term { sort, .. } => term.sort() == *sort,
        Pattern::Node {
            operator,
            arguments,
            payload,
        } => {
            let operator_matches = match operator {
                OperatorPattern::Fixed(expected) => expected == &Some(term.operator()),
                OperatorPattern::Variable(_) => true,
            };
            let payload_matches = match payload {
                PayloadPattern::None => term.payload().is_none(),
                PayloadPattern::Integer(expected) => {
                    fixed_payload(expected).is_some_and(|expected| term.payload() == Some(expected))
                }
                PayloadPattern::Variable(_) => term.payload().is_some(),
            };
            operator_matches && payload_matches && arguments.len() == term.arguments().len()
        }
        Pattern::Spine {
            core,
            operators,
            slot,
            min_order,
            max_order,
            ..
        } => {
            let frame = max_order != &Some(0)
                && operators.contains(&term.operator())
                && *slot < term.arguments().len();
            frame || (*min_order == 0 && root_may_match(core, term))
        }
    }
}

fn build_pattern(
    pattern: &Pattern,
    store: &TermStore,
    bindings: &Bindings,
) -> Result<Option<Term>, RuleError> {
    match pattern {
        Pattern::Term { name, .. } => Ok(bindings.terms.get(name).cloned()),
        Pattern::Node {
            operator,
            arguments,
            payload,
        } => {
            let operator = match operator {
                OperatorPattern::Fixed(Some(operator)) => *operator,
                OperatorPattern::Fixed(None) => return Ok(None),
                OperatorPattern::Variable(name) => match bindings.payloads.get(name) {
                    Some(BoundPayload::Operator(operator)) => *operator,
                    _ => return Ok(None),
                },
            };
            let payload = match payload {
                PayloadPattern::None => None,
                PayloadPattern::Integer(value) => {
                    let Some(value) = fixed_payload(value) else {
                        return Ok(None);
                    };
                    Some(value)
                }
                PayloadPattern::Variable(name) => match bindings.payloads.get(name) {
                    Some(BoundPayload::Integer(value)) => Some(*value),
                    _ => return Ok(None),
                },
            };
            let mut built = Vec::with_capacity(arguments.len());
            for argument in arguments {
                let Some(argument) = build_pattern(argument, store, bindings)? else {
                    return Ok(None);
                };
                built.push(argument);
            }
            Ok(Some(store.intern(operator, built, payload)?))
        }
        Pattern::Spine {
            name,
            core,
            operators,
            slot,
            min_order,
            max_order,
        } => {
            let Some(frames) = bindings.spines.get(name) else {
                return Ok(None);
            };
            if frames.len() < *min_order || max_order.is_some_and(|maximum| frames.len() > maximum)
            {
                return Ok(None);
            }
            let Some(mut built) = build_pattern(core, store, bindings)? else {
                return Ok(None);
            };
            for frame in frames.iter().rev() {
                if frame.slot != *slot || !operators.contains(&frame.term.operator()) {
                    return Ok(None);
                }
                let mut arguments = frame.term.arguments().to_vec();
                arguments[frame.slot] = built;
                built = store.intern(frame.term.operator(), arguments, frame.term.payload())?;
            }
            Ok(Some(built))
        }
    }
}

fn fixed_pattern(term: &Term) -> Pattern {
    Pattern::Node {
        operator: OperatorPattern::Fixed(Some(term.operator())),
        arguments: term.arguments().iter().map(fixed_pattern).collect(),
        payload: term.payload().map_or(PayloadPattern::None, |value| {
            PayloadPattern::Integer(value.to_string())
        }),
    }
}

/// Represent every maximal ground subpattern under one compiled relation.
///
/// A reachability plan compares categories after applying its root matching
/// relation. Rule schemata may have been declared under another relation, so
/// their fixed portions must enter the same representative coordinate before
/// the plan propagates them.
///
/// # Errors
///
/// Returns a typed rule or term error when a ground subpattern cannot be
/// constructed in the supplied store.
pub fn represent_ground_pattern(
    pattern: &Pattern,
    store: &TermStore,
    representatives: &RepresentativeKernel,
    table: &mut RepresentativeTable,
) -> Result<Pattern, RuleError> {
    if let Some(term) = build_pattern(pattern, store, &Bindings::default())? {
        register_tree(representatives, &term, table);
        return table.representative(&term).map(|term| fixed_pattern(&term));
    }
    match pattern {
        Pattern::Term { .. } => Ok(pattern.clone()),
        Pattern::Node {
            operator,
            arguments,
            payload,
        } => Ok(Pattern::Node {
            operator: operator.clone(),
            arguments: arguments
                .iter()
                .map(|argument| represent_ground_pattern(argument, store, representatives, table))
                .collect::<Result<Vec<_>, _>>()?,
            payload: payload.clone(),
        }),
        Pattern::Spine {
            name,
            core,
            operators,
            slot,
            min_order,
            max_order,
        } => Ok(Pattern::Spine {
            name: name.clone(),
            core: Box::new(represent_ground_pattern(
                core,
                store,
                representatives,
                table,
            )?),
            operators: operators.clone(),
            slot: *slot,
            min_order: *min_order,
            max_order: *max_order,
        }),
    }
}

impl PatternRule {
    fn ground_unary_keys(&self) -> Option<Vec<GroundPatternKey>> {
        let mut keys = Vec::new();
        for clause in &self.clauses {
            let [premise] = clause.premises.as_slice() else {
                return None;
            };
            match GroundPatternKey::from_pattern(premise) {
                GroundPattern::Dynamic => return None,
                GroundPattern::Key(key) if !keys.contains(&key) => keys.push(key),
                GroundPattern::Impossible | GroundPattern::Key(_) => {}
            }
        }
        Some(keys)
    }

    fn slot_viable(
        &self,
        representatives: &RepresentativeTable,
        term: &Term,
        slot: usize,
    ) -> Result<bool, RuleError> {
        let represented = representatives.key(term)?;
        Ok(self.clauses.iter().any(|clause| {
            clause.premises.get(slot).is_some_and(|pattern| {
                ground_matches(pattern, represented).unwrap_or_else(|| {
                    !root_matches(pattern, represented, &Bindings::default()).is_empty()
                })
            })
        }))
    }

    fn join_keys(
        &self,
        representatives: &RepresentativeTable,
        term: &Term,
        slot: usize,
    ) -> Result<Vec<RuleJoinKey>, RuleError> {
        let represented = representatives.key(term)?;
        let mut keys = Vec::new();
        for (clause_at, clause) in self.clauses.iter().enumerate() {
            let Some(pattern) = clause.premises.get(slot) else {
                continue;
            };
            let shared = shared_binding_names(clause);
            for bindings in root_matches(pattern, represented, &Bindings::default()) {
                let Some(bindings) = join_bindings(&bindings, &shared) else {
                    continue;
                };
                let key = RuleJoinKey::Pattern {
                    clause: clause_at,
                    bindings,
                };
                if !keys.contains(&key) {
                    keys.push(key);
                }
            }
        }
        Ok(keys)
    }

    fn apply_clause(
        clause: &PatternClause,
        store: &TermStore,
        representatives: &RepresentativeTable,
        inputs: &[Term],
    ) -> Result<Vec<Term>, RuleError> {
        if inputs.len() != clause.premises.len() {
            return Ok(Vec::new());
        }
        let mut environments = vec![Bindings::default()];
        for (pattern, term) in clause.premises.iter().zip(inputs) {
            let represented = representatives.key(term)?;
            environments = environments
                .iter()
                .flat_map(|bindings| root_matches(pattern, represented, bindings))
                .collect();
            if environments.is_empty() {
                break;
            }
        }
        let mut outputs = Vec::new();
        for bindings in environments {
            if let Some(output) = build_pattern(&clause.conclusion, store, &bindings)? {
                push_unique(&mut outputs, output);
            }
        }
        Ok(outputs)
    }

    fn apply(
        &self,
        store: &TermStore,
        representatives: &RepresentativeTable,
        inputs: &[Term],
    ) -> Result<Vec<Term>, RuleError> {
        let mut outputs = Vec::new();
        for clause in &self.clauses {
            for output in Self::apply_clause(clause, store, representatives, inputs)? {
                push_unique(&mut outputs, output);
            }
        }
        Ok(outputs)
    }

    fn admits(
        &self,
        _store: &TermStore,
        representatives: &RepresentativeTable,
        inputs: &[Term],
        output: &Term,
    ) -> Result<bool, RuleError> {
        for clause in &self.clauses {
            if inputs.len() != clause.premises.len() {
                continue;
            }
            let mut environments = vec![Bindings::default()];
            for (pattern, term) in clause.premises.iter().zip(inputs) {
                let represented = representatives.key(term)?;
                environments = environments
                    .iter()
                    .flat_map(|bindings| root_matches(pattern, represented, bindings))
                    .collect();
                if environments.is_empty() {
                    break;
                }
            }
            let represented_output = representatives.key(output)?;
            if environments.iter().any(|bindings| {
                !root_matches(&clause.conclusion, represented_output, bindings).is_empty()
            }) {
                return Ok(true);
            }
        }
        Ok(false)
    }
}

fn ground_matches(pattern: &Pattern, term: &Term) -> Option<bool> {
    let Pattern::Node {
        operator,
        arguments,
        payload,
    } = pattern
    else {
        return None;
    };
    let OperatorPattern::Fixed(expected) = operator else {
        return None;
    };
    if expected != &Some(term.operator()) || arguments.len() != term.arguments().len() {
        return Some(false);
    }
    let payload_matches = match payload {
        PayloadPattern::None => term.payload().is_none(),
        PayloadPattern::Integer(expected) => {
            fixed_payload(expected).is_some_and(|expected| term.payload() == Some(expected))
        }
        PayloadPattern::Variable(_) => return None,
    };
    if !payload_matches {
        return Some(false);
    }
    for (child, value) in arguments.iter().zip(term.arguments()) {
        if !ground_matches(child, value)? {
            return Some(false);
        }
    }
    Some(true)
}

impl AbsorptionRule {
    fn is_absorbing(
        &self,
        representatives: &RepresentativeTable,
        term: &Term,
    ) -> Result<bool, RuleError> {
        self.absorbing.iter().try_fold(false, |found, candidate| {
            Ok(found || representatives.equivalent(term, candidate)?)
        })
    }

    fn join_keys(
        &self,
        representatives: &RepresentativeTable,
        term: &Term,
        slot: usize,
    ) -> Result<Vec<RuleJoinKey>, RuleError> {
        let absorbing_at = match self.side {
            AbsorptionSide::Left => 0,
            AbsorptionSide::Right => 1,
        };
        let absorbing: Vec<usize> = if slot == absorbing_at {
            self.absorbing
                .iter()
                .enumerate()
                .map(|(position, candidate)| {
                    representatives
                        .equivalent(term, candidate)
                        .map(|matches| matches.then_some(position))
                })
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .flatten()
                .collect()
        } else {
            if self.is_absorbing(representatives, term)? {
                return Ok(Vec::new());
            }
            (0..self.absorbing.len()).collect()
        };
        let mut keys = Vec::new();
        for pattern in self.pattern.join_keys(representatives, term, slot)? {
            let RuleJoinKey::Pattern { clause, bindings } = pattern else {
                continue;
            };
            for position in &absorbing {
                keys.push(RuleJoinKey::Absorption {
                    clause,
                    bindings: bindings.clone(),
                    absorbing: *position,
                });
            }
        }
        Ok(keys)
    }

    fn apply(
        &self,
        store: &TermStore,
        representatives: &RepresentativeTable,
        inputs: &[Term],
    ) -> Result<Vec<Term>, RuleError> {
        if inputs.len() != 2 {
            return Ok(Vec::new());
        }
        let (absorbing_at, retained_at) = match self.side {
            AbsorptionSide::Left => (0, 1),
            AbsorptionSide::Right => (1, 0),
        };
        let retained = &inputs[retained_at];
        if !self.is_absorbing(representatives, &inputs[absorbing_at])?
            || self.is_absorbing(representatives, retained)?
            || !self
                .pattern
                .admits(store, representatives, inputs, retained)?
        {
            return Ok(Vec::new());
        }
        Ok(vec![retained.clone()])
    }
}

impl CoordinationPhaseRule {
    fn join_keys(
        &self,
        representatives: &RepresentativeTable,
        term: &Term,
        slot: usize,
    ) -> Result<Vec<RuleJoinKey>, RuleError> {
        if self.phase == CoordinationPhase::Mark {
            return self.pattern.join_keys(representatives, term, slot);
        }
        let mut keys = Vec::new();
        for (pair, (plain, marked)) in self.pairs.iter().enumerate() {
            let (matches, pattern_term) = if slot == 0 {
                (
                    representatives.equivalent(term, plain)?
                        && self.target_admitted(representatives, plain)?,
                    term,
                )
            } else {
                (representatives.equivalent(term, marked)?, plain)
            };
            if !matches {
                continue;
            }
            if self.marks_by_construction(plain, marked) {
                continue;
            }
            self.push_merge_keys(representatives, pattern_term, slot, pair, &mut keys)?;
        }
        if let Some(marking) = self.marking {
            let base = if slot == 0 {
                self.target_admitted(representatives, term)?.then_some(term)
            } else if term.operator() == marking {
                term.arguments().first()
            } else {
                None
            };
            if let Some(base) = base {
                self.push_merge_keys(representatives, base, slot, self.pairs.len(), &mut keys)?;
            }
        }
        Ok(keys)
    }

    fn push_merge_keys(
        &self,
        representatives: &RepresentativeTable,
        pattern_term: &Term,
        slot: usize,
        pair: usize,
        keys: &mut Vec<RuleJoinKey>,
    ) -> Result<(), RuleError> {
        for pattern in self
            .pattern
            .join_keys(representatives, pattern_term, slot)?
        {
            let RuleJoinKey::Pattern { clause, bindings } = pattern else {
                continue;
            };
            keys.push(RuleJoinKey::CoordinationPhase {
                clause,
                bindings,
                pair,
            });
        }
        Ok(())
    }

    fn marks_by_construction(&self, plain: &Term, marked: &Term) -> bool {
        self.marking.is_some_and(|marking| {
            marked.operator() == marking
                && marked.arguments().len() == 1
                && marked.arguments()[0].is_identical(plain)
        })
    }

    fn core_matches(
        &self,
        representatives: &RepresentativeTable,
        source: &Term,
        target: &Term,
    ) -> Result<bool, RuleError> {
        let equivalent = representatives.equivalent(source, target)?;
        Ok(match self.core_relation {
            CoreRelation::Same => equivalent,
            CoreRelation::Different => !equivalent,
        })
    }

    fn target_admitted(
        &self,
        representatives: &RepresentativeTable,
        target: &Term,
    ) -> Result<bool, RuleError> {
        let Some(targets) = &self.targets else {
            return Ok(true);
        };
        targets.iter().try_fold(false, |found, declared| {
            Ok(found || representatives.equivalent(target, declared)?)
        })
    }

    fn marks(
        &self,
        store: &TermStore,
        representatives: &RepresentativeTable,
        unmarked: &Term,
        outputs: &mut Vec<Term>,
    ) -> Result<(), RuleError> {
        for (plain, marked) in &self.pairs {
            if representatives.equivalent_constructed(unmarked, plain) {
                push_representative_unique(outputs, marked.clone(), representatives)?;
            }
        }
        if let Some(marking) = self.marking {
            let marked = store.intern(marking, vec![unmarked.clone()], None)?;
            if !outputs
                .iter()
                .any(|known| representatives.equivalent_constructed(known, &marked))
            {
                outputs.push(marked);
            }
        }
        Ok(())
    }

    fn mark(
        &self,
        store: &TermStore,
        representatives: &RepresentativeTable,
        inputs: &[Term],
    ) -> Result<Vec<Term>, RuleError> {
        if inputs.len() != 2 {
            return Ok(Vec::new());
        }
        let bases = if let Some(targets) = &self.targets {
            let mut bases = Vec::new();
            for target in targets {
                if self
                    .pattern
                    .admits(store, representatives, inputs, target)?
                    && self.core_matches(representatives, &inputs[1], target)?
                {
                    bases.push(target.clone());
                }
            }
            bases
        } else if self.core_relation == CoreRelation::Same {
            self.pattern.apply(store, representatives, inputs)?
        } else {
            Vec::new()
        };
        let mut outputs = Vec::new();
        for base in bases {
            self.marks(store, representatives, &base, &mut outputs)?;
        }
        Ok(outputs)
    }

    fn merge(
        &self,
        store: &TermStore,
        representatives: &RepresentativeTable,
        inputs: &[Term],
    ) -> Result<Vec<Term>, RuleError> {
        if inputs.len() != 2 {
            return Ok(Vec::new());
        }
        let (left, right) = (&inputs[0], &inputs[1]);
        for (plain, marked) in &self.pairs {
            if representatives.equivalent(right, marked)?
                && representatives.equivalent(left, plain)?
                && self.target_admitted(representatives, plain)?
                && self.pattern.admits(
                    store,
                    representatives,
                    &[left.clone(), plain.clone()],
                    left,
                )?
            {
                return Ok(vec![left.clone()]);
            }
        }
        if let Some(marking) = self.marking
            && right.operator() == marking
            && let [base] = right.arguments()
            && representatives.equivalent_constructed(left, base)
            && self.target_admitted(representatives, base)?
            && self
                .pattern
                .admits(store, representatives, &[left.clone(), base.clone()], left)?
        {
            return Ok(vec![left.clone()]);
        }
        Ok(Vec::new())
    }

    fn apply(
        &self,
        store: &TermStore,
        representatives: &RepresentativeTable,
        inputs: &[Term],
    ) -> Result<Vec<Term>, RuleError> {
        match self.phase {
            CoordinationPhase::Mark => self.mark(store, representatives, inputs),
            CoordinationPhase::Merge => self.merge(store, representatives, inputs),
        }
    }
}

impl RuleKernel {
    fn ground_unary_keys(&self) -> Option<Vec<GroundPatternKey>> {
        match self {
            Self::Pattern(rule) => rule.ground_unary_keys(),
            _ => None,
        }
    }

    /// Return the premise restriction of a restricted rule, if any.
    #[must_use]
    pub fn premise_restriction(&self) -> Option<&PremiseRestriction> {
        match self {
            Self::Restricted(rule) => Some(&rule.restriction),
            _ => None,
        }
    }

    /// Return the unrestricted relation of this rule.
    #[must_use]
    pub fn unrestricted(&self) -> &Self {
        match self {
            Self::Restricted(rule) => &rule.rule,
            _ => self,
        }
    }

    /// Whether this rule supplies exact join keys for its binary premises.
    #[must_use]
    pub fn uses_join_keys(&self) -> bool {
        match self {
            Self::Restricted(rule) => rule.rule.uses_join_keys(),
            _ => matches!(
                self,
                Self::Composition(_)
                    | Self::Substitution(_)
                    | Self::Pattern(_)
                    | Self::Absorption(_)
                    | Self::CoordinationPhase(_)
            ),
        }
    }

    fn join_slot_viable(
        &self,
        representatives: &RepresentativeTable,
        term: &Term,
        slot: usize,
    ) -> Result<bool, RuleError> {
        let viable =
            match self {
                Self::Composition(rule) => rule.functors.orientations(&rule.slash).into_iter().any(
                    |(principal, container)| {
                        (slot == principal
                            && rule.functors.is_functor(term)
                            && term.operator_name() == rule.slash)
                            || (slot == container
                                && (rule.order == Some(0) || rule.functors.is_functor(term)))
                    },
                ),
                Self::Substitution(rule) => {
                    rule.functors.orientations(&rule.slash).into_iter().any(
                        |(functor, container)| {
                            (slot == functor || slot == container) && rule.functors.is_functor(term)
                        },
                    )
                }
                Self::Pattern(rule) => {
                    let represented = representatives.key(term)?;
                    rule.clauses.iter().any(|clause| {
                        clause
                            .premises
                            .get(slot)
                            .is_some_and(|pattern| root_may_match(pattern, represented))
                    })
                }
                Self::Absorption(rule) => {
                    let represented = representatives.key(term)?;
                    rule.pattern.clauses.iter().any(|clause| {
                        clause
                            .premises
                            .get(slot)
                            .is_some_and(|pattern| root_may_match(pattern, represented))
                    })
                }
                Self::CoordinationPhase(rule) => {
                    let represented = representatives.key(term)?;
                    rule.pattern.clauses.iter().any(|clause| {
                        clause
                            .premises
                            .get(slot)
                            .is_some_and(|pattern| root_may_match(pattern, represented))
                    })
                }
                Self::Empty | Self::Raising(_) => false,
                Self::Restricted(rule) => {
                    rule.rule.join_slot_viable(representatives, term, slot)?
                        && rule.restriction.admits_slot(representatives, term, slot)?
                }
            };
        Ok(viable)
    }

    /// Report whether one exact term can fill one premise slot.
    ///
    /// This check is conservative: `true` may still fail once bindings shared
    /// across the other premises are known, while `false` proves that no rule
    /// instance can use the term in this slot.
    ///
    /// # Errors
    ///
    /// Returns a typed error when the term has no registered matching
    /// representative.
    pub fn slot_viable(
        &self,
        representatives: &RepresentativeTable,
        term: &Term,
        slot: usize,
    ) -> Result<bool, RuleError> {
        match self {
            Self::Raising(rule) => Ok(slot == 0 && rule.raises(representatives, term)?),
            Self::Empty | Self::Composition(_) | Self::Substitution(_) => Ok(true),
            Self::Pattern(rule) => rule.slot_viable(representatives, term, slot),
            Self::Absorption(rule) => rule.pattern.slot_viable(representatives, term, slot),
            Self::CoordinationPhase(rule) => rule.pattern.slot_viable(representatives, term, slot),
            Self::Restricted(rule) => Ok(rule.rule.slot_viable(representatives, term, slot)?
                && rule.restriction.admits_slot(representatives, term, slot)?),
        }
    }

    /// Return exact native join keys for one premise slot.
    ///
    /// # Errors
    ///
    /// Returns a typed error when a declared term path or matching
    /// representative is unavailable.
    pub fn join_keys(
        &self,
        representatives: &RepresentativeTable,
        term: &Term,
        slot: usize,
    ) -> Result<Vec<RuleJoinKey>, RuleError> {
        match self {
            Self::Composition(rule) => rule.join_keys(representatives, term, slot),
            Self::Substitution(rule) => rule.join_keys(representatives, term, slot),
            Self::Pattern(rule) => rule.join_keys(representatives, term, slot),
            Self::Absorption(rule) => rule.join_keys(representatives, term, slot),
            Self::CoordinationPhase(rule) => rule.join_keys(representatives, term, slot),
            Self::Restricted(rule) => {
                if rule.restriction.admits_slot(representatives, term, slot)? {
                    rule.rule.join_keys(representatives, term, slot)
                } else {
                    Ok(Vec::new())
                }
            }
            _ => Ok(Vec::new()),
        }
    }

    /// Return join keys whose conclusion is determined by this premise alone.
    ///
    /// # Errors
    ///
    /// Returns a typed error when a declared term path or matching
    /// representative is unavailable.
    pub fn join_factors(
        &self,
        representatives: &RepresentativeTable,
        term: &Term,
        slot: usize,
    ) -> Result<Vec<(RuleJoinKey, Term)>, RuleError> {
        if let Self::Restricted(rule) = self {
            if !rule.restriction.admits_slot(representatives, term, slot)? {
                return Ok(Vec::new());
            }
            return rule.rule.join_factors(representatives, term, slot);
        }
        let Self::Composition(rule) = self else {
            return Ok(Vec::new());
        };
        if rule.order != Some(0) {
            return Ok(Vec::new());
        }

        let mut factors = Vec::new();
        for (at, (principal, _)) in rule
            .functors
            .orientations(&rule.slash)
            .into_iter()
            .enumerate()
        {
            if slot != principal
                || !rule.functors.is_functor(term)
                || term.operator_name() != rule.slash
            {
                continue;
            }
            let argument = at_path(term, &rule.functors.argument_path)?;
            let output = at_path(term, &rule.functors.result_path)?;
            let key = RuleJoinKey::Composition {
                orientation: u8::try_from(at).unwrap_or(u8::MAX),
                order: rule.order,
                representative: representatives.key(&argument)?.clone(),
            };
            if !factors
                .iter()
                .any(|held| held == &(key.clone(), output.clone()))
            {
                factors.push((key, output));
            }
        }
        Ok(factors)
    }

    /// Return independently groupable exact output contributions for one join key.
    ///
    /// Positive-order composition and substitution expose complementary result
    /// and frame factors. Other rule forms retain exact-pair application.
    ///
    /// # Errors
    ///
    /// Returns a typed error when a declared term path is unavailable.
    pub fn output_factors(
        &self,
        representatives: &RepresentativeTable,
        term: &Term,
        slot: usize,
        key: &RuleJoinKey,
    ) -> Result<Vec<RuleOutputFactor>, RuleError> {
        if let Self::Restricted(rule) = self {
            if !rule.restriction.admits_slot(representatives, term, slot)? {
                return Ok(Vec::new());
            }
            return rule.rule.output_factors(representatives, term, slot, key);
        }
        let frames_factor =
            |functors: &FunctorSpec, frames: Vec<Frame>| -> Result<RuleOutputFactor, RuleError> {
                Ok(RuleOutputFactor(OutputFactorKind::Frames(
                    frames
                        .into_iter()
                        .map(|frame| OutputFrame::from_term(&frame.term, &functors.result_path))
                        .collect::<Result<Vec<_>, _>>()?,
                )))
            };
        match (self, key) {
            (
                Self::Composition(rule),
                RuleJoinKey::Composition {
                    orientation,
                    order: Some(order),
                    ..
                },
            ) if *order > 0 && rule.order == Some(*order) => {
                let Some((principal, container)) = rule
                    .functors
                    .orientations(&rule.slash)
                    .get(usize::from(*orientation))
                    .copied()
                else {
                    return Ok(Vec::new());
                };
                if slot == principal {
                    Ok(vec![RuleOutputFactor(OutputFactorKind::Result(at_path(
                        term,
                        &rule.functors.result_path,
                    )?))])
                } else if slot == container {
                    let Some((frames, _)) = walk_frames(&rule.functors, term, *order)? else {
                        return Ok(Vec::new());
                    };
                    Ok(vec![frames_factor(&rule.functors, frames)?])
                } else {
                    Ok(Vec::new())
                }
            }
            (
                Self::Composition(rule),
                RuleJoinKey::Composition {
                    orientation,
                    order: None,
                    representative,
                },
            ) if rule.order.is_none() => {
                let Some((principal, container)) = rule
                    .functors
                    .orientations(&rule.slash)
                    .get(usize::from(*orientation))
                    .copied()
                else {
                    return Ok(Vec::new());
                };
                if slot == principal {
                    return Ok(vec![RuleOutputFactor(OutputFactorKind::Result(at_path(
                        term,
                        &rule.functors.result_path,
                    )?))]);
                }
                if slot != container {
                    return Ok(Vec::new());
                }
                let mut current = term.clone();
                let mut frames = Vec::new();
                let mut factors = Vec::new();
                loop {
                    if rule.admitted_order(frames.len())
                        && rule.licensed(&frames)
                        && representatives.key(&current)? == representative
                    {
                        let factor = frames_factor(&rule.functors, frames.clone())?;
                        if !factors.contains(&factor) {
                            factors.push(factor);
                        }
                    }
                    if !rule.functors.is_functor(&current) {
                        break;
                    }
                    frames.push(Frame {
                        term: current.clone(),
                        slot: *rule.functors.result_path.first().unwrap_or(&0),
                    });
                    current = at_path(&current, &rule.functors.result_path)?;
                }
                Ok(factors)
            }
            (
                Self::Substitution(rule),
                RuleJoinKey::Substitution {
                    orientation, order, ..
                },
            ) if rule.order.is_none_or(|declared| declared == *order) => {
                let Some((functor, container)) = rule
                    .functors
                    .orientations(&rule.slash)
                    .get(usize::from(*orientation))
                    .copied()
                else {
                    return Ok(Vec::new());
                };
                if slot == functor {
                    let Some((_, foot)) = walk_frames(&rule.functors, term, *order)? else {
                        return Ok(Vec::new());
                    };
                    Ok(vec![RuleOutputFactor(OutputFactorKind::Result(at_path(
                        &foot,
                        &rule.functors.result_path,
                    )?))])
                } else if slot == container {
                    let Some((frames, _)) = walk_frames(&rule.functors, term, *order)? else {
                        return Ok(Vec::new());
                    };
                    Ok(vec![frames_factor(&rule.functors, frames)?])
                } else {
                    Ok(Vec::new())
                }
            }
            _ => Ok(Vec::new()),
        }
    }

    /// Apply this rule to exact premise terms.
    ///
    /// # Errors
    ///
    /// Returns a typed error when a matching representative, term path, or
    /// native term construction is invalid.
    pub fn apply(
        &self,
        store: &TermStore,
        representatives: &RepresentativeTable,
        inputs: &[Term],
    ) -> Result<Vec<Term>, RuleError> {
        match self {
            Self::Empty => Ok(Vec::new()),
            Self::Composition(rule) => rule.apply(store, representatives, inputs),
            Self::Substitution(rule) => rule.apply(store, representatives, inputs),
            Self::Raising(rule) => rule.apply(store, representatives, inputs),
            Self::Pattern(rule) => rule.apply(store, representatives, inputs),
            Self::Absorption(rule) => rule.apply(store, representatives, inputs),
            Self::CoordinationPhase(rule) => rule.apply(store, representatives, inputs),
            Self::Restricted(rule) => {
                if rule.restriction.admits_all(representatives, inputs)? {
                    rule.rule.apply(store, representatives, inputs)
                } else {
                    Ok(Vec::new())
                }
            }
        }
    }

    fn apply_fragment(
        &self,
        store: &TermStore,
        representatives: &RepresentativeTable,
        inputs: &[Term],
        fragment: &RuleJoinKey,
    ) -> Result<Vec<Term>, RuleError> {
        if let Self::Restricted(rule) = self {
            if !rule.restriction.admits_all(representatives, inputs)? {
                return Ok(Vec::new());
            }
            return rule
                .rule
                .apply_fragment(store, representatives, inputs, fragment);
        }
        if fragment == &RuleJoinKey::Direct {
            return self.apply(store, representatives, inputs);
        }
        match (self, fragment) {
            (
                Self::Composition(rule),
                RuleJoinKey::Composition {
                    orientation,
                    order: Some(order),
                    ..
                },
            ) => rule.apply_fragment(store, inputs, *orientation, *order),
            (Self::Composition(rule), RuleJoinKey::Composition { order: None, .. }) => {
                rule.apply(store, representatives, inputs)
            }
            (
                Self::Substitution(rule),
                RuleJoinKey::Substitution {
                    orientation, order, ..
                },
            ) => rule.apply_fragment(store, inputs, *orientation, *order),
            (Self::Pattern(rule), RuleJoinKey::Pattern { clause, .. }) => {
                rule.clauses.get(*clause).map_or(Ok(Vec::new()), |clause| {
                    PatternRule::apply_clause(clause, store, representatives, inputs)
                })
            }
            (Self::Absorption(rule), RuleJoinKey::Absorption { .. }) => {
                rule.apply(store, representatives, inputs)
            }
            (Self::CoordinationPhase(rule), RuleJoinKey::Pattern { clause, .. })
                if rule.phase == CoordinationPhase::Mark =>
            {
                let bases = rule
                    .pattern
                    .clauses
                    .get(*clause)
                    .map_or(Ok(Vec::new()), |clause| {
                        PatternRule::apply_clause(clause, store, representatives, inputs)
                    })?;
                let mut outputs = Vec::new();
                for base in bases {
                    rule.marks(store, representatives, &base, &mut outputs)?;
                }
                Ok(outputs)
            }
            (Self::CoordinationPhase(rule), RuleJoinKey::CoordinationPhase { .. }) => {
                rule.apply(store, representatives, inputs)
            }
            _ => Ok(Vec::new()),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::{
        AbsorptionRule, AbsorptionSide, CompositionRestriction, CompositionRule, CoordinationPhase,
        CoordinationPhaseRule, CoreRelation, FunctorSpec, OperatorPattern, Pattern, PatternClause,
        PatternRule, PayloadPattern, ProgramRule, RaisingRule, ReachabilityBindings,
        RepresentativeKernel, RepresentativeTable, RuleApplication, RuleCategoryMeta,
        RuleCategoryQuery, RuleKernel, RuleProgram, SubstitutionRule, combine_output_factors,
        match_one, reachability_application_matches, reachability_build, reachability_conclusions,
        reachability_extend_into, reachability_merge_project, reachability_patterns_match,
        reachability_projection, reachability_separator_projection, reachability_union_projection,
        represent_ground_pattern,
    };
    use crate::term::{OperatorId, OperatorSpec, PayloadKind, SignatureSpec, Term, TermStore};

    fn store() -> Result<TermStore, crate::term::TermError> {
        TermStore::new(
            SignatureSpec {
                name: "rules".to_owned(),
                sorts: vec!["Cat".to_owned(), "Other".to_owned()],
                operators: vec![
                    OperatorSpec {
                        name: "a".to_owned(),
                        sort: 0,
                        arguments: vec![],
                        payload: PayloadKind::None,
                        erased: 0,
                        binding: vec![],
                        shifts: vec![],
                    },
                    OperatorSpec {
                        name: "b".to_owned(),
                        sort: 0,
                        arguments: vec![],
                        payload: PayloadKind::None,
                        erased: 0,
                        binding: vec![],
                        shifts: vec![],
                    },
                    OperatorSpec {
                        name: "c".to_owned(),
                        sort: 0,
                        arguments: vec![],
                        payload: PayloadKind::None,
                        erased: 2,
                        binding: vec![],
                        shifts: vec![],
                    },
                    OperatorSpec {
                        name: "fwd".to_owned(),
                        sort: 0,
                        arguments: vec![0, 0],
                        payload: PayloadKind::None,
                        erased: 3,
                        binding: vec![],
                        shifts: vec![vec![], vec![]],
                    },
                    OperatorSpec {
                        name: "bwd".to_owned(),
                        sort: 0,
                        arguments: vec![0, 0],
                        payload: PayloadKind::None,
                        erased: 4,
                        binding: vec![],
                        shifts: vec![vec![], vec![]],
                    },
                    OperatorSpec {
                        name: "#fvar[Cat]".to_owned(),
                        sort: 0,
                        arguments: vec![],
                        payload: PayloadKind::Free,
                        erased: 5,
                        binding: vec![],
                        shifts: vec![],
                    },
                    OperatorSpec {
                        name: "other".to_owned(),
                        sort: 1,
                        arguments: vec![],
                        payload: PayloadKind::None,
                        erased: 6,
                        binding: vec![],
                        shifts: vec![],
                    },
                    OperatorSpec {
                        name: "mark".to_owned(),
                        sort: 0,
                        arguments: vec![0],
                        payload: PayloadKind::None,
                        erased: 7,
                        binding: vec![],
                        shifts: vec![vec![]],
                    },
                ],
                vocabularies: vec![None, None],
                constant_cores: vec![None, None],
                constants: vec![None, None],
                bound: vec![None, None],
                free: vec![Some(5), None],
                holes: vec![None, None],
                allows_vacuous_binding: true,
            },
            256,
        )
    }

    fn functors() -> FunctorSpec {
        FunctorSpec {
            operators: vec!["fwd".to_owned(), "bwd".to_owned()],
            forward: "fwd".to_owned(),
            backward: "bwd".to_owned(),
            result_path: vec![0],
            argument_path: vec![1],
        }
    }

    fn register(table: &mut RepresentativeTable, terms: &[Term]) {
        let mut pending = terms.to_vec();
        while let Some(term) = pending.pop() {
            table.register(&term, term.clone());
            pending.extend(term.arguments().iter().cloned());
        }
    }

    fn variable(name: &str) -> Pattern {
        Pattern::Term {
            name: name.to_owned(),
            sort: 0,
        }
    }

    #[test]
    fn reachability_matching_respects_declared_sorts() -> Result<(), crate::term::TermError> {
        let store = store()?;
        let a = store.intern(0, vec![], None)?;
        let aa = store.intern(3, vec![a.clone(), a.clone()], None)?;
        let other = Pattern::Term {
            name: "x".to_owned(),
            sort: 1,
        };
        let nested = Pattern::Node {
            operator: OperatorPattern::Fixed(Some(3)),
            arguments: vec![other.clone(), variable("y")],
            payload: PayloadPattern::None,
        };

        assert!(!reachability_patterns_match(
            std::slice::from_ref(&other),
            std::slice::from_ref(&a)
        ));
        assert!(!reachability_patterns_match(
            std::slice::from_ref(&nested),
            std::slice::from_ref(&aa)
        ));
        assert!(reachability_patterns_match(
            &[variable("x")],
            std::slice::from_ref(&a)
        ));
        assert!(!reachability_application_matches(
            std::slice::from_ref(&other),
            &variable("z"),
            std::slice::from_ref(&a),
            &a,
        ));
        Ok(())
    }

    #[test]
    fn reachability_matching_preserves_general_matching() -> Result<(), crate::term::TermError> {
        let store = store()?;
        let a = store.intern(0, vec![], None)?;
        let b = store.intern(1, vec![], None)?;
        let aa = store.intern(3, vec![a.clone(), a.clone()], None)?;
        let ab = store.intern(3, vec![a.clone(), b.clone()], None)?;
        let free_x = store.free(0, "x")?;
        let free_y = store.free(0, "y")?;
        let repeated = Pattern::Node {
            operator: OperatorPattern::Fixed(Some(3)),
            arguments: vec![variable("x"), variable("x")],
            payload: PayloadPattern::None,
        };
        let operator = Pattern::Node {
            operator: OperatorPattern::Variable("operator".to_owned()),
            arguments: Vec::new(),
            payload: PayloadPattern::None,
        };
        let payload = Pattern::Node {
            operator: OperatorPattern::Fixed(Some(5)),
            arguments: Vec::new(),
            payload: PayloadPattern::Variable("payload".to_owned()),
        };

        for (pattern, term) in [
            (&variable("category"), &a),
            (&repeated, &aa),
            (&repeated, &ab),
            (&operator, &a),
            (&payload, &free_x),
        ] {
            assert_eq!(
                reachability_patterns_match(
                    std::slice::from_ref(pattern),
                    std::slice::from_ref(term)
                ),
                match_one(pattern, term, &mut super::Bindings::default()),
            );
        }

        for (patterns, terms) in [
            (vec![operator.clone(), operator], vec![a.clone(), b]),
            (vec![payload.clone(), payload], vec![free_x, free_y]),
            (vec![repeated.clone()], vec![aa]),
            (vec![repeated], vec![ab]),
        ] {
            let mut bindings = super::Bindings::default();
            let general = patterns
                .iter()
                .zip(&terms)
                .all(|(pattern, term)| match_one(pattern, term, &mut bindings));
            assert_eq!(reachability_patterns_match(&patterns, &terms), general);
        }
        Ok(())
    }

    fn cartesian(pools: &[Vec<Term>]) -> Vec<Vec<Term>> {
        pools.iter().fold(vec![Vec::new()], |rows, pool| {
            rows.into_iter()
                .flat_map(|row| {
                    pool.iter().map(move |term| {
                        let mut next = row.clone();
                        next.push(term.clone());
                        next
                    })
                })
                .collect()
        })
    }

    fn assert_binding_thread_matches_cartesian(
        store: &TermStore,
        premises: &[Pattern],
        conclusion: &Pattern,
        pools: &[Vec<Term>],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let schema = vec![(premises.to_vec(), conclusion.clone())];
        let mut expected_outputs = Vec::new();
        let mut actual_outputs = Vec::new();
        let separator = reachability_separator_projection(premises, conclusion);
        for tuple in cartesian(pools) {
            for output in reachability_conclusions(&schema, store, &tuple)? {
                if !expected_outputs.contains(&output) {
                    expected_outputs.push(output);
                }
            }
            let mut bindings = vec![ReachabilityBindings::default()];
            let mut matched = Vec::new();
            for (at, (premise, term)) in premises.iter().zip(&tuple).enumerate() {
                let future = reachability_projection(
                    std::iter::once(conclusion).chain(premises[at + 1..].iter()),
                );
                let retained = reachability_union_projection(&future, &separator);
                reachability_extend_into(
                    premise,
                    term,
                    &ReachabilityBindings::default(),
                    &mut matched,
                );
                let mut next = Vec::new();
                for bindings in &bindings {
                    for row in &matched {
                        if let Some((merged, _)) =
                            reachability_merge_project(bindings, row, &retained)
                        {
                            next.push(merged);
                        }
                    }
                }
                bindings = next;
            }
            for bindings in bindings {
                if let Some(output) = reachability_build(conclusion, store, &bindings)?
                    && !actual_outputs.contains(&output)
                {
                    actual_outputs.push(output);
                }
            }
        }
        assert_eq!(
            actual_outputs.iter().collect::<HashSet<_>>(),
            expected_outputs.iter().collect::<HashSet<_>>(),
        );
        Ok(())
    }

    #[test]
    fn binding_thread_matches_cartesian_relational_evaluation()
    -> Result<(), Box<dyn std::error::Error>> {
        let store = store()?;
        let a = store.intern(0, vec![], None)?;
        let b = store.intern(1, vec![], None)?;
        let c = store.intern(2, vec![], None)?;
        let ca = store.intern(3, vec![c.clone(), a.clone()], None)?;
        let cb = store.intern(3, vec![c.clone(), b.clone()], None)?;
        let cc = store.intern(3, vec![c.clone(), c.clone()], None)?;
        let aa = store.intern(3, vec![a.clone(), a.clone()], None)?;
        let ab = store.intern(3, vec![a.clone(), b.clone()], None)?;
        let bc = store.intern(3, vec![b.clone(), c.clone()], None)?;
        let free_x = store.free(0, "x")?;
        let free_y = store.free(0, "y")?;
        let other = store.intern(6, vec![], None)?;
        let container = Pattern::Node {
            operator: OperatorPattern::Fixed(Some(3)),
            arguments: vec![variable("result"), variable("shared")],
            payload: PayloadPattern::None,
        };
        for (premises, conclusion, pools) in [
            (
                vec![variable("shared"), container.clone()],
                variable("result"),
                vec![vec![a.clone(), b.clone()], vec![ca.clone(), cb.clone(), cc]],
            ),
            (
                vec![container, variable("shared")],
                variable("result"),
                vec![vec![ca.clone(), cb], vec![a.clone(), b.clone()]],
            ),
            (
                vec![
                    Pattern::Node {
                        operator: OperatorPattern::Variable("operator".to_owned()),
                        arguments: Vec::new(),
                        payload: PayloadPattern::None,
                    },
                    Pattern::Node {
                        operator: OperatorPattern::Variable("operator".to_owned()),
                        arguments: Vec::new(),
                        payload: PayloadPattern::None,
                    },
                ],
                Pattern::Node {
                    operator: OperatorPattern::Variable("operator".to_owned()),
                    arguments: Vec::new(),
                    payload: PayloadPattern::None,
                },
                vec![vec![a.clone(), b.clone()], vec![a.clone(), b.clone()]],
            ),
            (
                vec![
                    Pattern::Node {
                        operator: OperatorPattern::Fixed(Some(5)),
                        arguments: Vec::new(),
                        payload: PayloadPattern::Variable("payload".to_owned()),
                    },
                    Pattern::Node {
                        operator: OperatorPattern::Fixed(Some(5)),
                        arguments: Vec::new(),
                        payload: PayloadPattern::Variable("payload".to_owned()),
                    },
                ],
                Pattern::Node {
                    operator: OperatorPattern::Fixed(Some(5)),
                    arguments: Vec::new(),
                    payload: PayloadPattern::Variable("payload".to_owned()),
                },
                vec![vec![free_x.clone(), free_y.clone()], vec![free_x, free_y]],
            ),
            (
                vec![
                    Pattern::Node {
                        operator: OperatorPattern::Fixed(Some(3)),
                        arguments: vec![variable("same"), variable("same")],
                        payload: PayloadPattern::None,
                    },
                    variable("local"),
                ],
                variable("same"),
                vec![vec![aa, ab.clone()], vec![a.clone(), b.clone(), c.clone()]],
            ),
            (
                vec![variable("left"), variable("right")],
                Pattern::Node {
                    operator: OperatorPattern::Fixed(Some(0)),
                    arguments: Vec::new(),
                    payload: PayloadPattern::None,
                },
                vec![vec![a.clone(), b.clone()], vec![b.clone(), c.clone()]],
            ),
            (
                vec![
                    variable("x"),
                    Pattern::Node {
                        operator: OperatorPattern::Fixed(Some(3)),
                        arguments: vec![variable("x"), variable("y")],
                        payload: PayloadPattern::None,
                    },
                    variable("y"),
                ],
                Pattern::Node {
                    operator: OperatorPattern::Fixed(Some(4)),
                    arguments: vec![variable("x"), variable("y")],
                    payload: PayloadPattern::None,
                },
                vec![
                    vec![a.clone(), b.clone()],
                    vec![ab.clone(), bc, ca.clone()],
                    vec![b.clone()],
                ],
            ),
            (
                vec![
                    Pattern::Term {
                        name: "shared".to_owned(),
                        sort: 0,
                    },
                    Pattern::Term {
                        name: "shared".to_owned(),
                        sort: 1,
                    },
                ],
                variable("shared"),
                vec![vec![a.clone()], vec![other]],
            ),
        ] {
            assert_binding_thread_matches_cartesian(&store, &premises, &conclusion, &pools)?;
        }

        let inner = store.intern(3, vec![a.clone(), b.clone()], None)?;
        let outer = store.intern(3, vec![inner, c.clone()], None)?;
        let spine = |core: &str| Pattern::Spine {
            name: "context".to_owned(),
            core: Box::new(variable(core)),
            operators: vec![3],
            slot: 0,
            min_order: 0,
            max_order: Some(2),
        };
        assert_binding_thread_matches_cartesian(
            &store,
            &[spine("left"), spine("right")],
            &variable("left"),
            &[vec![outer.clone()], vec![outer]],
        )?;

        let source = store.intern(3, vec![a, b.clone()], None)?;
        let target = store.intern(3, vec![c.clone(), b], None)?;
        let context = Pattern::Spine {
            name: "replacement-context".to_owned(),
            core: Box::new(variable("source")),
            operators: vec![3],
            slot: 0,
            min_order: 1,
            max_order: Some(1),
        };
        let result = Pattern::Spine {
            name: "replacement-context".to_owned(),
            core: Box::new(variable("replacement")),
            operators: vec![3],
            slot: 0,
            min_order: 1,
            max_order: Some(1),
        };
        let premises = vec![context, variable("replacement")];
        let pools = vec![vec![source.clone()], vec![c]];
        assert_binding_thread_matches_cartesian(&store, &premises, &result, &pools)?;
        let mut bindings = Vec::new();
        reachability_extend_into(
            &premises[0],
            &source,
            &ReachabilityBindings::default(),
            &mut bindings,
        );
        let mut matched = Vec::new();
        let mut joined = Vec::new();
        for bindings in &bindings {
            reachability_extend_into(&premises[1], &pools[1][0], bindings, &mut matched);
            joined.append(&mut matched);
        }
        assert!(joined.iter().any(|bindings| {
            reachability_build(&result, &store, bindings)
                .is_ok_and(|output| output.is_some_and(|output| output.is_identical(&target)))
        }));
        Ok(())
    }

    #[test]
    fn reachability_schemata_use_the_plan_representative_coordinate()
    -> Result<(), Box<dyn std::error::Error>> {
        let store = store()?;
        let plain = store.intern(0, vec![], None)?;
        let featured = store.intern(1, vec![], None)?;
        let pattern = Pattern::Node {
            operator: OperatorPattern::Fixed(Some(3)),
            arguments: vec![
                variable("category"),
                Pattern::Node {
                    operator: OperatorPattern::Fixed(Some(1)),
                    arguments: Vec::new(),
                    payload: PayloadPattern::None,
                },
            ],
            payload: PayloadPattern::None,
        };
        let mut table = RepresentativeTable::default();

        let represented =
            represent_ground_pattern(&pattern, &store, &RepresentativeKernel::Core, &mut table)?;

        assert_eq!(
            represented,
            Pattern::Node {
                operator: OperatorPattern::Fixed(Some(3)),
                arguments: vec![
                    variable("category"),
                    Pattern::Node {
                        operator: OperatorPattern::Fixed(Some(0)),
                        arguments: Vec::new(),
                        payload: PayloadPattern::None,
                    }
                ],
                payload: PayloadPattern::None,
            },
        );
        assert!(table.representative(&featured)?.is_identical(&plain));
        Ok(())
    }

    #[test]
    fn representative_kernel_applies_finite_overrides_before_its_base()
    -> Result<(), Box<dyn std::error::Error>> {
        let store = store()?;
        let a = store.intern(0, vec![], None)?;
        let b = store.intern(1, vec![], None)?;
        let kernel = RepresentativeKernel::Refined {
            base: Box::new(RepresentativeKernel::Identity),
            overrides: std::iter::once((a.clone(), b.clone())).collect(),
        };
        let mut identity = RepresentativeTable::default();
        identity.register_with(&RepresentativeKernel::Identity, &a);
        let mut core = RepresentativeTable::default();
        core.register_with(&RepresentativeKernel::Core, &a);
        let mut refined = RepresentativeTable::default();
        refined.register_with(&kernel, &a);
        refined.register_with(&kernel, &b);

        assert!(identity.representative(&a)?.is_identical(&a));
        assert!(core.representative(&a)?.is_identical(&a.core()));
        assert!(refined.representative(&a)?.is_identical(&b));
        assert!(refined.representative(&b)?.is_identical(&b));
        Ok(())
    }

    #[test]
    fn product_representatives_intersect_both_component_relations()
    -> Result<(), Box<dyn std::error::Error>> {
        let store = store()?;
        let plain = store.intern(0, vec![], None)?;
        let featured = store.intern(1, vec![], None)?;
        let kernel = RepresentativeKernel::Product {
            left: Box::new(RepresentativeKernel::Core),
            right: Box::new(RepresentativeKernel::Identity),
        };
        let mut table = RepresentativeTable::default();
        table.register_with(&kernel, &plain);
        table.register_with(&kernel, &featured);

        assert!(!table.equivalent(&plain, &featured)?);
        assert!(table.equivalent(&plain, &plain)?);
        Ok(())
    }

    #[test]
    fn finite_refinement_preserves_an_opaque_base_class_without_an_override()
    -> Result<(), Box<dyn std::error::Error>> {
        let store = store()?;
        let plain = store.intern(0, vec![], None)?;
        let featured = store.intern(1, vec![], None)?;
        let product = RepresentativeKernel::Product {
            left: Box::new(RepresentativeKernel::Core),
            right: Box::new(RepresentativeKernel::Core),
        };
        let kernel = RepresentativeKernel::Refined {
            base: Box::new(product),
            overrides: std::collections::HashMap::default(),
        };
        let mut table = RepresentativeTable::default();
        table.register_with(&kernel, &plain);
        table.register_with(&kernel, &featured);

        assert!(table.equivalent(&plain, &featured)?);
        Ok(())
    }

    #[test]
    fn subsumption_representatives_preserve_repeated_free_variables()
    -> Result<(), Box<dyn std::error::Error>> {
        let store = store()?;
        let x = store.free(0, "x")?;
        let y = store.free(0, "y")?;
        let z = store.free(0, "z")?;
        let xx = store.intern(3, vec![x.clone(), x], None)?;
        let yy = store.intern(3, vec![y.clone(), y.clone()], None)?;
        let yz = store.intern(3, vec![y, z], None)?;
        let mut table = RepresentativeTable::default();
        for term in [&xx, &yy, &yz] {
            table.register_with(&RepresentativeKernel::Subsumption, term);
        }

        assert!(table.equivalent(&xx, &yy)?);
        assert!(!table.equivalent(&xx, &yz)?);
        Ok(())
    }

    #[test]
    fn category_indexing_reuses_join_keys_for_one_core() -> Result<(), Box<dyn std::error::Error>> {
        let store = store()?;
        let a = store.intern(0, vec![], None)?;
        let b = store.intern(1, vec![], None)?;
        let plain = store.intern(3, vec![a.clone(), a.clone()], None)?;
        let featured = store.intern(3, vec![a, b], None)?;
        assert!(plain.core().is_identical(&featured.core()));
        let program = RuleProgram::new(vec![ProgramRule {
            kernel: RuleKernel::Composition(CompositionRule {
                functors: functors(),
                order: Some(0),
                slash: "fwd".to_owned(),
                crossed: false,
                restriction: CompositionRestriction::Unrestricted,
            }),
            matching: 0,
            representatives: RepresentativeKernel::Core,
            constants: Vec::new(),
        }]);
        let queries = [
            RuleCategoryQuery {
                term: plain,
                scope: 0,
            },
            RuleCategoryQuery {
                term: featured,
                scope: 0,
            },
        ];

        let metadata = program.category_meta_batch(&[(2, true)], &queries, None);

        assert_eq!(metadata[0], metadata[1]);
        assert_eq!(
            program
                .join_cache
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .len(),
            2
        );
        Ok(())
    }

    fn fixed(operator: OperatorId) -> Pattern {
        Pattern::Node {
            operator: OperatorPattern::Fixed(Some(operator)),
            arguments: Vec::new(),
            payload: PayloadPattern::None,
        }
    }

    #[test]
    fn a_marking_constructor_coordinates_a_category_no_pair_names()
    -> Result<(), Box<dyn std::error::Error>> {
        let store = store()?;
        let conj = store.intern(0, vec![], None)?;
        let b = store.intern(1, vec![], None)?;
        let a = store.intern(0, vec![], None)?;
        let raised = store.intern(3, vec![b.clone(), a], None)?;
        let marked = store.intern(7, vec![raised.clone()], None)?;
        let mut representatives = RepresentativeTable::default();
        register(
            &mut representatives,
            &[conj.clone(), raised.clone(), marked.clone()],
        );
        let mark = RuleKernel::CoordinationPhase(CoordinationPhaseRule {
            pattern: PatternRule {
                clauses: vec![PatternClause {
                    premises: vec![fixed(0), variable("X")],
                    conclusion: variable("X"),
                }],
            },
            pairs: Vec::new(),
            marking: Some(7),
            targets: None,
            phase: CoordinationPhase::Mark,
            core_relation: CoreRelation::Same,
        });
        let merge = RuleKernel::CoordinationPhase(CoordinationPhaseRule {
            pattern: PatternRule {
                clauses: vec![PatternClause {
                    premises: vec![variable("X"), variable("X")],
                    conclusion: variable("X"),
                }],
            },
            pairs: Vec::new(),
            marking: Some(7),
            targets: None,
            phase: CoordinationPhase::Merge,
            core_relation: CoreRelation::Same,
        });

        let outputs = mark.apply(&store, &representatives, &[conj, raised.clone()])?;
        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].is_identical(&marked));
        let merged = merge.apply(&store, &representatives, &[raised.clone(), marked.clone()])?;
        assert_eq!(merged.len(), 1);
        assert!(merged[0].is_identical(&raised));
        assert_eq!(
            merge.join_keys(&representatives, &raised, 0)?,
            merge.join_keys(&representatives, &marked, 1)?,
        );
        assert_eq!(
            merge.apply(&store, &representatives, &[b, marked])?,
            Vec::new()
        );
        Ok(())
    }

    #[test]
    fn application_and_composition_share_one_kernel() -> Result<(), Box<dyn std::error::Error>> {
        let store = store()?;
        let a = store.intern(0, vec![], None)?;
        let b = store.intern(1, vec![], None)?;
        let c = store.intern(2, vec![], None)?;
        let principal = store.intern(3, vec![b.clone(), a.clone()], None)?;
        let container = store.intern(3, vec![a.clone(), c.clone()], None)?;
        let expected = store.intern(3, vec![b.clone(), c], None)?;
        let mut representatives = RepresentativeTable::default();
        register(
            &mut representatives,
            &[principal.clone(), container.clone(), a.clone()],
        );
        let application = RuleKernel::Composition(CompositionRule {
            functors: functors(),
            order: Some(0),
            slash: "fwd".to_owned(),
            crossed: false,
            restriction: CompositionRestriction::Unrestricted,
        });
        let composition = RuleKernel::Composition(CompositionRule {
            functors: functors(),
            order: Some(1),
            slash: "fwd".to_owned(),
            crossed: false,
            restriction: CompositionRestriction::Unrestricted,
        });

        assert!(
            application.apply(&store, &representatives, &[principal.clone(), a])?[0]
                .is_identical(&b)
        );
        let factors = application.join_factors(&representatives, &principal, 0)?;
        assert_eq!(factors.len(), 1);
        assert!(factors[0].1.is_identical(&b));
        assert_eq!(
            composition.join_factors(&representatives, &principal, 0)?,
            Vec::new()
        );
        let principal_keys = composition.join_keys(&representatives, &principal, 0)?;
        let container_keys = composition.join_keys(&representatives, &container, 1)?;
        assert_eq!(principal_keys, container_keys);
        assert_eq!(principal_keys.len(), 1);
        let principal_factor = composition
            .output_factors(&representatives, &principal, 0, &principal_keys[0])?
            .into_iter()
            .next()
            .ok_or("the principal has an output factor")?;
        let container_factor = composition
            .output_factors(&representatives, &container, 1, &container_keys[0])?
            .into_iter()
            .next()
            .ok_or("the container has an output factor")?;
        let factored = combine_output_factors(&store, &principal_factor, &container_factor)?
            .ok_or("the factors are complementary")?;
        assert!(factored.is_identical(&expected));
        assert!(
            composition.apply_fragment(
                &store,
                &representatives,
                &[principal.clone(), container.clone()],
                &principal_keys[0],
            )?[0]
                .is_identical(&expected)
        );
        assert!(
            composition.apply(&store, &representatives, &[principal, container])?[0]
                .is_identical(&expected)
        );
        Ok(())
    }

    #[test]
    fn unbounded_composition_factors_each_licensed_depth() -> Result<(), Box<dyn std::error::Error>>
    {
        let store = store()?;
        let a = store.intern(0, vec![], None)?;
        let b = store.intern(1, vec![], None)?;
        let c = store.intern(2, vec![], None)?;
        let d = b.clone();
        let principal = store.intern(3, vec![b.clone(), a.clone()], None)?;
        let shallow = store.intern(3, vec![a.clone(), c.clone()], None)?;
        let inner = store.intern(3, vec![a.clone(), c], None)?;
        let deep = store.intern(3, vec![inner, d], None)?;
        let mut representatives = RepresentativeTable::default();
        register(
            &mut representatives,
            &[principal.clone(), shallow.clone(), deep.clone(), a],
        );
        let rule = RuleKernel::Composition(CompositionRule {
            functors: functors(),
            order: None,
            slash: "fwd".to_owned(),
            crossed: false,
            restriction: CompositionRestriction::Unrestricted,
        });
        let principal_key = rule.join_keys(&representatives, &principal, 0)?[0].clone();
        let principal_factor = rule
            .output_factors(&representatives, &principal, 0, &principal_key)?
            .into_iter()
            .next()
            .ok_or("the unbounded principal has an output factor")?;

        for container in [shallow, deep] {
            let key = rule
                .join_keys(&representatives, &container, 1)?
                .into_iter()
                .find(|key| key == &principal_key)
                .ok_or("the container has the principal join key")?;
            let factors = rule.output_factors(&representatives, &container, 1, &key)?;
            let applied = rule.apply(&store, &representatives, &[principal.clone(), container])?;
            let combined = factors
                .iter()
                .map(|factor| {
                    combine_output_factors(&store, &principal_factor, factor)
                        .map(|output| output.expect("the factors are complementary"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            assert_eq!(combined, applied);
        }
        Ok(())
    }

    #[test]
    fn ccg_wrappers_apply_through_native_pattern_kernels() -> Result<(), Box<dyn std::error::Error>>
    {
        let store = store()?;
        let a = store.intern(0, vec![], None)?;
        let b = store.intern(1, vec![], None)?;
        let c = store.intern(2, vec![], None)?;
        let mut representatives = RepresentativeTable::default();
        register(&mut representatives, &[a.clone(), b.clone(), c.clone()]);
        let retain_right = PatternRule {
            clauses: vec![PatternClause {
                premises: vec![variable("X"), variable("Y")],
                conclusion: variable("Y"),
            }],
        };
        let absorption = RuleKernel::Absorption(AbsorptionRule {
            pattern: retain_right,
            absorbing: vec![a.clone()],
            side: AbsorptionSide::Left,
        });

        assert_eq!(
            absorption.apply(&store, &representatives, &[a.clone(), b.clone()])?,
            vec![b.clone()]
        );
        assert_eq!(
            absorption.apply(&store, &representatives, &[a.clone(), a.clone()])?,
            Vec::new()
        );
        assert_eq!(
            absorption.join_keys(&representatives, &a, 0)?,
            absorption.join_keys(&representatives, &b, 1)?,
        );
        assert_eq!(absorption.join_keys(&representatives, &b, 0)?, Vec::new());
        assert_eq!(absorption.join_keys(&representatives, &a, 1)?, Vec::new());
        let program = RuleProgram::new(vec![ProgramRule {
            kernel: absorption,
            matching: 0,
            representatives: RepresentativeKernel::Identity,
            constants: vec![a.clone()],
        }]);
        let metadata = program.category_meta_batch(
            &[(2, true)],
            &[
                RuleCategoryQuery {
                    term: a.clone(),
                    scope: 0,
                },
                RuleCategoryQuery {
                    term: b.clone(),
                    scope: 0,
                },
            ],
            None,
        );
        assert!(metadata.into_iter().all(|result| result.is_ok()));

        let mark_pattern = PatternRule {
            clauses: vec![PatternClause {
                premises: vec![fixed(0), variable("X")],
                conclusion: variable("X"),
            }],
        };
        let mark = RuleKernel::CoordinationPhase(CoordinationPhaseRule {
            pattern: mark_pattern,
            pairs: vec![(b.clone(), c.clone())],
            marking: None,
            targets: None,
            phase: CoordinationPhase::Mark,
            core_relation: CoreRelation::Same,
        });

        assert_eq!(
            mark.apply(&store, &representatives, &[a, b.clone()])?,
            vec![c.clone()]
        );

        let merge_pattern = PatternRule {
            clauses: vec![PatternClause {
                premises: vec![variable("X"), variable("X")],
                conclusion: variable("X"),
            }],
        };
        let merge = RuleKernel::CoordinationPhase(CoordinationPhaseRule {
            pattern: merge_pattern,
            pairs: vec![(b.clone(), c.clone())],
            marking: None,
            targets: None,
            phase: CoordinationPhase::Merge,
            core_relation: CoreRelation::Same,
        });

        assert_eq!(
            merge.apply(&store, &representatives, &[b.clone(), c.clone()])?,
            vec![b.clone()]
        );
        assert_eq!(
            merge.join_keys(&representatives, &b, 0)?,
            merge.join_keys(&representatives, &c, 1)?,
        );
        assert_eq!(merge.join_keys(&representatives, &c, 0)?, Vec::new());
        Ok(())
    }

    #[test]
    fn substitution_rebuilds_the_container_spine() -> Result<(), Box<dyn std::error::Error>> {
        let store = store()?;
        let a = store.intern(0, vec![], None)?;
        let b = store.intern(1, vec![], None)?;
        let c = store.intern(2, vec![], None)?;
        let principal = store.intern(3, vec![b.clone(), a.clone()], None)?;
        let lifted = store.intern(3, vec![principal, c.clone()], None)?;
        let container = store.intern(3, vec![a.clone(), c.clone()], None)?;
        let nonparallel = store.intern(3, vec![a, b.clone()], None)?;
        let expected = store.intern(3, vec![b, c], None)?;
        let mut representatives = RepresentativeTable::default();
        register(
            &mut representatives,
            &[lifted.clone(), container.clone(), nonparallel.clone()],
        );
        let rule = RuleKernel::Substitution(SubstitutionRule {
            functors: functors(),
            order: Some(1),
            slash: "fwd".to_owned(),
            crossed: false,
        });

        assert!(
            rule.apply(
                &store,
                &representatives,
                &[lifted.clone(), container.clone()],
            )?[0]
                .is_identical(&expected)
        );
        assert_eq!(
            rule.join_keys(&representatives, &lifted, 0)?,
            rule.join_keys(&representatives, &container, 1)?,
        );
        let functor_key = rule.join_keys(&representatives, &lifted, 0)?[0].clone();
        let container_key = rule.join_keys(&representatives, &container, 1)?[0].clone();
        let functor_factor = rule
            .output_factors(&representatives, &lifted, 0, &functor_key)?
            .into_iter()
            .next()
            .ok_or("the substitution functor has an output factor")?;
        let container_factor = rule
            .output_factors(&representatives, &container, 1, &container_key)?
            .into_iter()
            .next()
            .ok_or("the substitution container has an output factor")?;
        let factored = combine_output_factors(&store, &functor_factor, &container_factor)?
            .ok_or("the substitution factors are complementary")?;
        assert!(factored.is_identical(&expected));
        assert_ne!(
            rule.join_keys(&representatives, &lifted, 0)?,
            rule.join_keys(&representatives, &nonparallel, 1)?,
        );

        let unbounded = RuleKernel::Substitution(SubstitutionRule {
            functors: functors(),
            order: None,
            slash: "fwd".to_owned(),
            crossed: false,
        });
        let functor_key = unbounded.join_keys(&representatives, &lifted, 0)?[0].clone();
        let container_key = unbounded.join_keys(&representatives, &container, 1)?[0].clone();
        assert_eq!(functor_key, container_key);
        let functor_factor = unbounded
            .output_factors(&representatives, &lifted, 0, &functor_key)?
            .into_iter()
            .next()
            .ok_or("the unbounded substitution functor has an output factor")?;
        let container_factor = unbounded
            .output_factors(&representatives, &container, 1, &container_key)?
            .into_iter()
            .next()
            .ok_or("the unbounded substitution container has an output factor")?;
        let factored = combine_output_factors(&store, &functor_factor, &container_factor)?
            .ok_or("the unbounded substitution factors are complementary")?;
        assert!(factored.is_identical(&expected));
        Ok(())
    }

    #[test]
    fn a_ground_pattern_builds_its_declared_output() -> Result<(), Box<dyn std::error::Error>> {
        let store = store()?;
        let a = store.intern(0, vec![], None)?;
        let b = store.intern(1, vec![], None)?;
        let mut representatives = RepresentativeTable::default();
        register(&mut representatives, std::slice::from_ref(&a));
        let node = |operator| Pattern::Node {
            operator: OperatorPattern::Fixed(Some(operator)),
            arguments: vec![],
            payload: PayloadPattern::None,
        };
        let rule = RuleKernel::Pattern(PatternRule {
            clauses: vec![PatternClause {
                premises: vec![node(0)],
                conclusion: node(1),
            }],
        });

        assert!(rule.apply(&store, &representatives, &[a])?[0].is_identical(&b));
        Ok(())
    }

    #[test]
    fn pattern_join_keys_preserve_bindings_shared_across_premises()
    -> Result<(), Box<dyn std::error::Error>> {
        let store = store()?;
        let a = store.intern(0, vec![], None)?;
        let b = store.intern(1, vec![], None)?;
        let mut representatives = RepresentativeTable::default();
        register(&mut representatives, &[a.clone(), b.clone()]);
        let rule = RuleKernel::Pattern(PatternRule {
            clauses: vec![PatternClause {
                premises: vec![variable("shared"), variable("shared")],
                conclusion: variable("shared"),
            }],
        });

        assert!(rule.uses_join_keys());
        assert_eq!(
            rule.join_keys(&representatives, &a, 0)?,
            rule.join_keys(&representatives, &a, 1)?,
        );
        assert_ne!(
            rule.join_keys(&representatives, &a, 0)?,
            rule.join_keys(&representatives, &b, 1)?,
        );
        Ok(())
    }

    #[test]
    fn rule_program_parallel_batch_preserves_submission_order()
    -> Result<(), Box<dyn std::error::Error>> {
        let store = store()?;
        let a = store.intern(0, vec![], None)?;
        let b = store.intern(1, vec![], None)?;
        let c = store.intern(2, vec![], None)?;
        let pattern = |output| {
            RuleKernel::Pattern(PatternRule {
                clauses: vec![PatternClause {
                    premises: vec![fixed(0)],
                    conclusion: fixed(output),
                }],
            })
        };
        let program = RuleProgram::new(vec![
            ProgramRule {
                kernel: pattern(1),
                matching: 0,
                representatives: RepresentativeKernel::Identity,
                constants: vec![b.clone()],
            },
            ProgramRule {
                kernel: pattern(2),
                matching: 0,
                representatives: RepresentativeKernel::Identity,
                constants: vec![c.clone()],
            },
        ]);
        let applications = vec![
            RuleApplication {
                rule: 1,
                scope: 0,
                store: store.clone(),
                inputs: vec![a.clone()],
            },
            RuleApplication {
                rule: 0,
                scope: 0,
                store,
                inputs: vec![a],
            },
        ];
        let workers = rayon::ThreadPoolBuilder::new().num_threads(2).build()?;

        let serial = program.apply_batch(&applications, None);
        let parallel = program.apply_batch(&applications, Some(&workers));
        let queries = vec![
            RuleCategoryQuery {
                term: applications[0].inputs[0].clone(),
                scope: 0,
            },
            RuleCategoryQuery {
                term: applications[1].inputs[0].clone(),
                scope: 0,
            },
        ];
        let serial_meta = program.category_meta_batch(&[(1, true), (1, true)], &queries, None);
        let parallel_meta =
            program.category_meta_batch(&[(1, true), (1, true)], &queries, Some(&workers));

        assert_eq!(parallel, serial);
        assert_eq!(parallel_meta, serial_meta);
        assert_eq!(
            parallel_meta[0],
            Ok(RuleCategoryMeta {
                unary: vec![0, 1],
                viable: Vec::new(),
                joins: Vec::new(),
                factors: Vec::new(),
                output_factors: Vec::new(),
            })
        );
        assert_eq!(parallel[0], Ok((1, vec![c])));
        assert_eq!(parallel[1], Ok((0, vec![b])));
        Ok(())
    }

    #[test]
    fn raising_preserves_the_exact_input() -> Result<(), Box<dyn std::error::Error>> {
        let store = store()?;
        let target = store.intern(0, vec![], None)?;
        let input = store.intern(1, vec![], None)?;
        let inner = store.intern(4, vec![target.clone(), input.clone()], None)?;
        let expected = store.intern(3, vec![target.clone(), inner], None)?;
        let rule = RuleKernel::Raising(RaisingRule {
            functors: functors(),
            slash: "fwd".to_owned(),
            targets: Some(vec![target]),
            arguments: None,
        });

        assert!(
            rule.apply(&store, &RepresentativeTable::default(), &[input])?[0]
                .is_identical(&expected)
        );
        Ok(())
    }
}
