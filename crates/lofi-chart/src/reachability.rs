//! Exact root-directed demand planning for the native chart kernel.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};
use std::slice;
use std::sync::{Arc, Mutex};

use crate::chart::Span;
use crate::rule::{
    OperatorPattern, Pattern, PayloadPattern, ReachabilityBindings, ReachabilityProjection,
    ReachabilityProjectionKey, RuleError, reachability_extend_into, reachability_patterns_match,
    reachability_projection_key, reachability_shared_projection,
};
use crate::term::{OperatorId, SortId, Term};

type TermVariable = (String, SortId);

struct ReachabilityHasher(u64);

impl ReachabilityHasher {
    const fn mix(&mut self, value: u64) {
        self.0 ^= value.wrapping_add(0x9e37_79b9_7f4a_7c15);
        self.0 = self.0.rotate_left(27).wrapping_mul(0x3c79_ac49_2ba7_b653);
    }
}

impl Default for ReachabilityHasher {
    fn default() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }
}

impl Hasher for ReachabilityHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 ^= u64::from(*byte);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    fn write_u8(&mut self, value: u8) {
        self.mix(u64::from(value));
    }

    fn write_u16(&mut self, value: u16) {
        self.mix(u64::from(value));
    }

    fn write_u32(&mut self, value: u32) {
        self.mix(u64::from(value));
    }

    fn write_u64(&mut self, value: u64) {
        self.mix(value);
    }

    fn write_usize(&mut self, value: usize) {
        self.mix(u64::try_from(value).unwrap_or(u64::MAX));
    }
}

type FastMap<K, V> = HashMap<K, V, BuildHasherDefault<ReachabilityHasher>>;
type FastSet<T> = HashSet<T, BuildHasherDefault<ReachabilityHasher>>;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum PayloadAtom {
    Operator(Option<OperatorId>),
    Integer(String),
    Variable(String),
}
type TermBindings = FastMap<TermVariable, Pattern>;
type PayloadBindings = FastMap<String, PayloadAtom>;
type Application = (Span, u16);
type PatternRow = Arc<[Pattern]>;
type SharedPattern = Arc<Pattern>;
type PatternRowId = usize;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ApplicationRow {
    premises: PatternRowId,
    conclusion: SharedPattern,
    schema: Option<usize>,
}

type DecompositionRows = (Vec<ApplicationRow>, FastSet<ApplicationRow>);
type ApplicationRows = FastMap<Vec<Span>, DecompositionRows>;
type PremiseExpansion = (PatternRow, SharedPattern, Option<usize>);
type RefinementCacheKey = (PatternRowId, SharedPattern, Option<usize>, Vec<Option<u64>>);
type SharedRefinementCacheKey = (
    PatternRow,
    SharedPattern,
    Option<usize>,
    Vec<Option<u64>>,
    usize,
);
type SharedRefinement = Arc<[(PatternRow, SharedPattern)]>;
type PremiseExpansionCache = Mutex<FastMap<(usize, Pattern), Arc<[PremiseExpansion]>>>;
type RefinedApplication = (PatternRowId, PatternRow, SharedPattern);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct SpanSupport {
    patterns: Vec<Pattern>,
    complete: bool,
}

struct GroundSupport {
    targets: FastSet<Pattern>,
    spans: FastMap<Span, FastSet<Pattern>>,
    nonempty: FastSet<Span>,
}

#[derive(Default)]
struct DemandSet {
    patterns: Vec<SharedPattern>,
    general: Vec<SharedPattern>,
    exact: FastSet<SharedPattern>,
}

#[derive(Default)]
struct ReachabilityEnvelope {
    demands: FastMap<Span, DemandSet>,
    applications: FastMap<Application, ApplicationRows>,
    pattern_rows: Vec<PatternRow>,
    pattern_row_ids: FastMap<PatternRow, PatternRowId>,
    projection_rows: Vec<usize>,
}

#[derive(Clone, Default)]
struct InverseBindings {
    terms: TermBindings,
    payloads: PayloadBindings,
    spines: FastMap<String, Vec<(Pattern, usize)>>,
}

fn term_variable(name: &str, sort: SortId) -> TermVariable {
    (name.to_owned(), sort)
}

fn operator_atom(pattern: &OperatorPattern) -> PayloadAtom {
    match pattern {
        OperatorPattern::Fixed(operator) => PayloadAtom::Operator(*operator),
        OperatorPattern::Variable(name) => PayloadAtom::Variable(name.clone()),
    }
}

fn payload_atom(pattern: &PayloadPattern) -> Option<PayloadAtom> {
    match pattern {
        PayloadPattern::None => None,
        PayloadPattern::Integer(value) => Some(PayloadAtom::Integer(value.clone())),
        PayloadPattern::Variable(name) => Some(PayloadAtom::Variable(name.clone())),
    }
}

/// One rule's finite root-reachability presentation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReachabilityRule {
    /// Premise count.
    pub arity: usize,
    /// Oriented premise/conclusion schemata.
    pub schemata: Vec<(Vec<Pattern>, Pattern)>,
    /// Exhaustive finite conclusions, or unrestricted conclusions.
    pub introduces: Option<Vec<Term>>,
    /// Exhaustive finite categories consumed in at least one premise.
    pub consumes: Option<Vec<Term>>,
}

/// Grammar-owned reachability compilation reused by sentence-specific plans.
#[derive(Debug)]
pub struct ReachabilityTemplate {
    source: Vec<ReachabilityRule>,
    rules: Vec<ReachabilityRule>,
    default_sort: SortId,
    lexical_support_ids: Mutex<FastMap<SpanSupport, u64>>,
    refinements: Mutex<FastMap<SharedRefinementCacheKey, SharedRefinement>>,
    premise_expansions: PremiseExpansionCache,
    binary_projection_cache: Arc<Mutex<BinaryProjectionCache>>,
}

/// One exact sentence-specific root-reachability plan.
#[derive(Clone, Debug)]
pub struct ReachabilityPlan {
    demands: FastMap<Span, Vec<SharedPattern>>,
    applications: FastMap<Application, FastMap<Vec<Span>, Vec<ApplicationRow>>>,
    binary_application_rules: FastMap<(Span, [Span; 2]), Vec<u16>>,
    pattern_rows: Vec<PatternRow>,
    projection_rows: Vec<usize>,
    schemata: Vec<Vec<(Vec<Pattern>, Pattern)>>,
    binary_projection_cache: Arc<Mutex<BinaryProjectionCache>>,
}

#[derive(Debug, Default)]
struct BinaryProjectionCache {
    row_ids: FastMap<PatternRow, usize>,
    rows: Vec<BinaryProjectionRowCache>,
}

#[derive(Debug)]
struct BinaryProjectionRowCache {
    shared: Option<ReachabilityProjection>,
    projections: [BTreeMap<u64, Arc<[ReachabilityProjectionKey]>>; 2],
    axes: [BTreeMap<Arc<[u64]>, ProjectionAxis>; 2],
}

impl BinaryProjectionRowCache {
    fn new(patterns: &PatternRow) -> Self {
        let shared = match patterns.as_ref() {
            [left, right] => Some(reachability_shared_projection(std::iter::once(left), right)),
            _ => None,
        };
        Self {
            shared,
            projections: Default::default(),
            axes: Default::default(),
        }
    }

    fn axis(&mut self, side: usize, pattern: &Pattern, terms: &[Term]) -> ProjectionAxis {
        let mut serials = terms.iter().map(Term::serial).collect::<Vec<_>>();
        serials.sort_unstable();
        serials.dedup();
        if let Some(axis) = self.axes[side].get(serials.as_slice()) {
            return Arc::clone(axis);
        }
        let Some(shared) = self.shared.as_ref() else {
            return Arc::from([]);
        };
        let mut axis = Vec::with_capacity(terms.len());
        for term in terms {
            let projected = self.projections[side]
                .entry(term.serial())
                .or_insert_with(|| {
                    let mut matched = Vec::new();
                    reachability_extend_into(
                        pattern,
                        term,
                        &ReachabilityBindings::default(),
                        &mut matched,
                    );
                    let mut keys = Vec::new();
                    for bindings in matched {
                        let key = reachability_projection_key(&bindings, shared);
                        if !keys.contains(&key) {
                            keys.push(key);
                        }
                    }
                    keys.into()
                });
            if !projected.is_empty() {
                axis.push((term.serial(), Arc::clone(projected)));
            }
        }
        axis.sort_unstable_by_key(|(serial, _)| *serial);
        axis.dedup_by_key(|(serial, _)| *serial);
        let axis = ProjectionAxis::from(axis);
        self.axes[side].insert(serials.into(), Arc::clone(&axis));
        axis
    }
}

#[derive(Clone, Debug)]
struct BinaryPermitRow {
    application: ApplicationRow,
    left: ProjectionAxis,
    right: ProjectionAxis,
}

type ProjectionAxis = Arc<[(u64, Arc<[ReachabilityProjectionKey]>)]>;

fn projected_at(
    axis: &[(u64, Arc<[ReachabilityProjectionKey]>)],
    serial: u64,
) -> Option<&Arc<[ReachabilityProjectionKey]>> {
    axis.binary_search_by_key(&serial, |(held, _)| *held)
        .ok()
        .and_then(|position| axis.get(position).map(|(_, projected)| projected))
}

/// One compact exact permit relation over the category axes of a decomposition.
#[derive(Clone, Debug, Default)]
pub struct BinaryPermits {
    rules: FastMap<u16, Vec<BinaryPermitRow>>,
}

type BinaryPermitKey = (u16, u64, u64);
type BinaryPermitRowId = usize;

/// Request-local matching rows for one binary permit relation.
pub struct BinaryPermitWitnesses<'p> {
    permits: &'p BinaryPermits,
    rows: FastMap<BinaryPermitKey, Box<[BinaryPermitRowId]>>,
}

impl BinaryPermits {
    /// Begin one request-local sequence of exact witness queries.
    #[must_use]
    pub fn witnesses(&self) -> BinaryPermitWitnesses<'_> {
        BinaryPermitWitnesses {
            permits: self,
            rows: FastMap::default(),
        }
    }
}

impl BinaryPermitWitnesses<'_> {
    fn matching_row_ids(&mut self, rule: u16, left: u64, right: u64) -> &[BinaryPermitRowId] {
        let key = (rule, left, right);
        self.rows.entry(key).or_insert_with(|| {
            self.permits
                .rules
                .get(&rule)
                .into_iter()
                .flatten()
                .enumerate()
                .filter_map(|(row_id, row)| {
                    let left_keys = projected_at(&row.left, left)?;
                    let right_keys = projected_at(&row.right, right)?;
                    left_keys
                        .iter()
                        .any(|key| right_keys.contains(key))
                        .then_some(row_id)
                })
                .collect::<Vec<_>>()
                .into_boxed_slice()
        })
    }

    /// Return whether one pair of representative serials belongs to a rule relation.
    #[must_use]
    pub fn permits(&mut self, rule: u16, left: u64, right: u64) -> bool {
        !self.matching_row_ids(rule, left, right).is_empty()
    }

    fn any_matching_application(
        &mut self,
        rule: u16,
        left: u64,
        right: u64,
        mut predicate: impl FnMut(&ApplicationRow) -> bool,
    ) -> bool {
        let key = (rule, left, right);
        let _ = self.matching_row_ids(rule, left, right);
        let Self { permits, rows } = self;
        let Some(rule_rows) = permits.rules.get(&rule) else {
            return false;
        };
        rows.get(&key).is_some_and(|row_ids| {
            row_ids
                .iter()
                .filter_map(|row_id| rule_rows.get(*row_id))
                .any(|row| predicate(&row.application))
        })
    }
}

fn apply_terms(pattern: &Pattern, terms: &TermBindings) -> Pattern {
    if terms.is_empty() {
        return pattern.clone();
    }
    match pattern {
        Pattern::Term { name, sort } => terms
            .get(&term_variable(name, *sort))
            .map_or_else(|| pattern.clone(), |held| apply_terms(held, terms)),
        Pattern::Node {
            operator,
            arguments,
            payload,
        } => Pattern::Node {
            operator: operator.clone(),
            arguments: arguments
                .iter()
                .map(|argument| apply_terms(argument, terms))
                .collect(),
            payload: payload.clone(),
        },
        Pattern::Spine {
            name,
            core,
            operators,
            slot,
            min_order,
            max_order,
        } => Pattern::Spine {
            name: name.clone(),
            core: Box::new(apply_terms(core, terms)),
            operators: operators.clone(),
            slot: *slot,
            min_order: *min_order,
            max_order: *max_order,
        },
    }
}

fn apply_payload(atom: &PayloadAtom, payloads: &PayloadBindings) -> PayloadAtom {
    match atom {
        PayloadAtom::Variable(name) => payloads
            .get(name)
            .map_or_else(|| atom.clone(), |held| apply_payload(held, payloads)),
        PayloadAtom::Operator(_) | PayloadAtom::Integer(_) => atom.clone(),
    }
}

fn occurs(variable: &TermVariable, pattern: &Pattern, terms: &TermBindings) -> bool {
    match pattern {
        Pattern::Term { name, sort } => {
            let held = term_variable(name, *sort);
            variable == &held
                || terms
                    .get(&held)
                    .is_some_and(|pattern| occurs(variable, pattern, terms))
        }
        Pattern::Node { arguments, .. } => arguments
            .iter()
            .any(|argument| occurs(variable, argument, terms)),
        Pattern::Spine { core, .. } => occurs(variable, core, terms),
    }
}

fn bind_term(
    variable: TermVariable,
    pattern: &Pattern,
    terms: &mut TermBindings,
    payloads: &mut PayloadBindings,
) -> bool {
    if let Some(held) = terms.get(&variable).cloned() {
        return unify(&held, pattern, terms, payloads);
    }
    if let Pattern::Term { name, sort } = pattern
        && variable == term_variable(name, *sort)
    {
        return true;
    }
    if occurs(&variable, pattern, terms) {
        return false;
    }
    terms.insert(variable, pattern.clone());
    true
}

fn bind_payload(variable: &str, atom: &PayloadAtom, payloads: &mut PayloadBindings) -> bool {
    let atom = apply_payload(atom, payloads);
    if let Some(held) = payloads.get(variable).cloned() {
        return unify_payload(&held, &atom, payloads);
    }
    if atom == PayloadAtom::Variable(variable.to_owned()) {
        return true;
    }
    payloads.insert(variable.to_owned(), atom);
    true
}

fn unify_payload(left: &PayloadAtom, right: &PayloadAtom, payloads: &mut PayloadBindings) -> bool {
    let left = apply_payload(left, payloads);
    let right = apply_payload(right, payloads);
    match (&left, &right) {
        (PayloadAtom::Variable(name), _) => bind_payload(name, &right, payloads),
        (_, PayloadAtom::Variable(name)) => bind_payload(name, &left, payloads),
        _ => left == right,
    }
}

fn unify(
    left: &Pattern,
    right: &Pattern,
    terms: &mut TermBindings,
    payloads: &mut PayloadBindings,
) -> bool {
    match (left, right) {
        (Pattern::Term { name, sort }, _) => {
            let variable = term_variable(name, *sort);
            if let Some(held) = terms.get(&variable).cloned() {
                unify(&held, right, terms, payloads)
            } else {
                bind_term(variable, right, terms, payloads)
            }
        }
        (_, Pattern::Term { name, sort }) => {
            let variable = term_variable(name, *sort);
            if let Some(held) = terms.get(&variable).cloned() {
                unify(left, &held, terms, payloads)
            } else {
                bind_term(variable, left, terms, payloads)
            }
        }
        (
            Pattern::Node {
                operator: left_operator,
                arguments: left_arguments,
                payload: left_payload,
            },
            Pattern::Node {
                operator: right_operator,
                arguments: right_arguments,
                payload: right_payload,
            },
        ) => {
            if left_arguments.len() != right_arguments.len()
                || !unify_payload(
                    &operator_atom(left_operator),
                    &operator_atom(right_operator),
                    payloads,
                )
            {
                return false;
            }
            match (payload_atom(left_payload), payload_atom(right_payload)) {
                (None, None) => {}
                (Some(left), Some(right)) if unify_payload(&left, &right, payloads) => {}
                _ => return false,
            }
            left_arguments
                .iter()
                .zip(right_arguments)
                .all(|(left, right)| unify(left, right, terms, payloads))
        }
        _ => false,
    }
}

/// Return false only when two patterns cannot unify under the held term bindings.
///
/// The check copies no binding map: it compares the fixed operators and argument
/// counts of the two node structures, following held term bindings, and
/// treats an unbound variable, an operator variable or a spine as compatible,
/// so [`unify`] decides every pair it lets through.
fn may_unify(left: &Pattern, right: &Pattern, terms: &TermBindings) -> bool {
    match (left, right) {
        (Pattern::Term { name, sort }, other) | (other, Pattern::Term { name, sort }) => {
            if terms.is_empty() {
                return true;
            }
            match terms.get(&(name.clone(), *sort)) {
                Some(held) if !matches!(held, Pattern::Term { .. }) => {
                    may_unify(held, other, terms)
                }
                _ => true,
            }
        }
        (
            Pattern::Node {
                operator: left_operator,
                arguments: left_arguments,
                ..
            },
            Pattern::Node {
                operator: right_operator,
                arguments: right_arguments,
                ..
            },
        ) => {
            if left_arguments.len() != right_arguments.len() {
                return false;
            }
            if let (OperatorPattern::Fixed(left_fixed), OperatorPattern::Fixed(right_fixed)) =
                (left_operator, right_operator)
                && left_fixed != right_fixed
            {
                return false;
            }
            left_arguments
                .iter()
                .zip(right_arguments)
                .all(|(left, right)| may_unify(left, right, terms))
        }
        _ => true,
    }
}

fn substitute(pattern: &Pattern, terms: &TermBindings, payloads: &PayloadBindings) -> Pattern {
    match pattern {
        Pattern::Term { name, sort } => terms
            .get(&term_variable(name, *sort))
            .map_or_else(|| pattern.clone(), |held| substitute(held, terms, payloads)),
        Pattern::Spine {
            name,
            core,
            operators,
            slot,
            min_order,
            max_order,
        } => Pattern::Spine {
            name: name.clone(),
            core: Box::new(substitute(core, terms, payloads)),
            operators: operators.clone(),
            slot: *slot,
            min_order: *min_order,
            max_order: *max_order,
        },
        Pattern::Node {
            operator,
            arguments,
            payload,
        } => {
            let operator = match apply_payload(&operator_atom(operator), payloads) {
                PayloadAtom::Operator(operator) => OperatorPattern::Fixed(operator),
                PayloadAtom::Variable(name) => OperatorPattern::Variable(name),
                PayloadAtom::Integer(_) => return apply_terms(pattern, terms),
            };
            let payload = match payload_atom(payload) {
                None => PayloadPattern::None,
                Some(atom) => match apply_payload(&atom, payloads) {
                    PayloadAtom::Integer(value) => PayloadPattern::Integer(value),
                    PayloadAtom::Variable(name) => PayloadPattern::Variable(name),
                    PayloadAtom::Operator(_) => return apply_terms(pattern, terms),
                },
            };
            Pattern::Node {
                operator,
                arguments: arguments
                    .iter()
                    .map(|argument| substitute(argument, terms, payloads))
                    .collect(),
                payload,
            }
        }
    }
}

/// Return whether one exact category satisfies one sort-aware reachability pattern.
fn matches_pattern(pattern: &Pattern, term: &Term) -> bool {
    reachability_patterns_match(slice::from_ref(pattern), slice::from_ref(term))
}

fn canonical_patterns(patterns: &[Pattern], default_sort: SortId) -> Vec<Pattern> {
    fn visit(
        pattern: &Pattern,
        default_sort: SortId,
        terms: &mut FastMap<TermVariable, String>,
        payloads: &mut FastMap<String, String>,
    ) -> Pattern {
        match pattern {
            Pattern::Term { name, sort } => {
                let key = term_variable(name, *sort);
                let next = terms.len();
                let name = terms.entry(key).or_insert_with(|| format!("v{next}"));
                Pattern::Term {
                    name: name.clone(),
                    sort: *sort,
                }
            }
            Pattern::Spine { .. } => {
                let next = terms.len();
                Pattern::Term {
                    name: format!("v{next}"),
                    sort: default_sort,
                }
            }
            Pattern::Node {
                operator,
                arguments,
                payload,
            } => {
                let operator = match operator {
                    OperatorPattern::Fixed(operator) => OperatorPattern::Fixed(*operator),
                    OperatorPattern::Variable(name) => {
                        let next = payloads.len();
                        OperatorPattern::Variable(
                            payloads
                                .entry(name.clone())
                                .or_insert_with(|| format!("p{next}"))
                                .clone(),
                        )
                    }
                };
                let payload = match payload {
                    PayloadPattern::Variable(name) => {
                        let next = payloads.len();
                        PayloadPattern::Variable(
                            payloads
                                .entry(name.clone())
                                .or_insert_with(|| format!("p{next}"))
                                .clone(),
                        )
                    }
                    held => held.clone(),
                };
                Pattern::Node {
                    operator,
                    arguments: arguments
                        .iter()
                        .map(|argument| visit(argument, default_sort, terms, payloads))
                        .collect(),
                    payload,
                }
            }
        }
    }

    let mut terms = FastMap::default();
    let mut payloads = FastMap::default();
    patterns
        .iter()
        .map(|pattern| visit(pattern, default_sort, &mut terms, &mut payloads))
        .collect()
}

fn decompositions(start: usize, stop: usize, arity: usize) -> Vec<Vec<Span>> {
    if arity == 1 {
        return vec![vec![(start, stop)]];
    }
    let mut held = Vec::new();
    for split in start + 1..stop {
        for mut suffix in decompositions(split, stop, arity - 1) {
            let mut row = Vec::with_capacity(arity);
            row.push((start, split));
            row.append(&mut suffix);
            held.push(row);
        }
    }
    held
}

fn term_pattern(term: &Term) -> Pattern {
    Pattern::Node {
        operator: OperatorPattern::Fixed(Some(term.operator())),
        arguments: term.arguments().iter().map(term_pattern).collect(),
        payload: term.payload().map_or(PayloadPattern::None, |value| {
            PayloadPattern::Integer(value.to_string())
        }),
    }
}

fn push_unique<T: PartialEq>(held: &mut Vec<T>, value: T) {
    if !held.contains(&value) {
        held.push(value);
    }
}

fn contains_spine(pattern: &Pattern) -> bool {
    match pattern {
        Pattern::Term { .. } => false,
        Pattern::Node { arguments, .. } => arguments.iter().any(contains_spine),
        Pattern::Spine { .. } => true,
    }
}

fn wildcard_row(arity: usize, sort: SortId) -> Vec<Pattern> {
    (0..arity)
        .map(|slot| Pattern::Term {
            name: format!("v{slot}"),
            sort,
        })
        .collect()
}

fn canonical_schema(
    premises: &[Pattern],
    conclusion: &Pattern,
    default_sort: SortId,
) -> (Vec<Pattern>, Pattern) {
    let mut patterns = Vec::with_capacity(premises.len() + 1);
    patterns.extend_from_slice(premises);
    patterns.push(conclusion.clone());
    let mut canonical = canonical_patterns(&patterns, default_sort);
    let conclusion = canonical.pop().unwrap_or_else(|| conclusion.clone());
    (canonical, conclusion)
}

fn instantiate_schema(
    premises: &[Pattern],
    conclusion: &Pattern,
    selected: &Pattern,
    exact: &Term,
    default_sort: SortId,
) -> Option<(Vec<Pattern>, Pattern)> {
    let mut terms = TermBindings::default();
    let mut payloads = PayloadBindings::default();
    let exact = term_pattern(exact);
    unify(selected, &exact, &mut terms, &mut payloads).then(|| {
        let premises = premises
            .iter()
            .map(|premise| substitute(premise, &terms, &payloads))
            .collect::<Vec<_>>();
        let conclusion = substitute(conclusion, &terms, &payloads);
        canonical_schema(&premises, &conclusion, default_sort)
    })
}

fn declared_schemata(
    rule: &ReachabilityRule,
    default_sort: SortId,
) -> Vec<(Vec<Pattern>, Pattern)> {
    let mut schemata = if rule.schemata.is_empty() {
        vec![(
            wildcard_row(rule.arity, default_sort),
            Pattern::Term {
                name: "result".to_owned(),
                sort: default_sort,
            },
        )]
    } else {
        rule.schemata.clone()
    };

    if let Some(introduces) = &rule.introduces {
        let mut refined = Vec::new();
        for (premises, conclusion) in &schemata {
            if contains_spine(conclusion) || premises.iter().any(contains_spine) {
                push_unique(&mut refined, (premises.clone(), conclusion.clone()));
                continue;
            }
            for exact in introduces {
                if let Some(schema) =
                    instantiate_schema(premises, conclusion, conclusion, exact, default_sort)
                {
                    push_unique(&mut refined, schema);
                }
            }
        }
        schemata = refined;
    }

    if let Some(consumes) = &rule.consumes {
        let mut refined = Vec::new();
        for (premises, conclusion) in &schemata {
            if contains_spine(conclusion) || premises.iter().any(contains_spine) {
                push_unique(&mut refined, (premises.clone(), conclusion.clone()));
                continue;
            }
            for premise in premises {
                for exact in consumes {
                    if let Some(schema) =
                        instantiate_schema(premises, conclusion, premise, exact, default_sort)
                    {
                        push_unique(&mut refined, schema);
                    }
                }
            }
        }
        schemata = refined;
    }

    schemata
}

impl ReachabilityTemplate {
    /// Compile grammar-owned rule schemata once outside sentence planning.
    ///
    /// # Errors
    ///
    /// Returns [`RuleError::TooManyRules`] when the rule inventory exceeds the
    /// native identifier width.
    pub fn new(rules: &[ReachabilityRule], default_sort: SortId) -> Result<Self, RuleError> {
        let maximum_rules = usize::from(u16::MAX) + 1;
        if rules.len() > maximum_rules {
            return Err(RuleError::TooManyRules {
                rules: rules.len(),
                maximum: maximum_rules,
            });
        }
        let source = rules.to_vec();
        let rules = rules
            .iter()
            .map(|rule| ReachabilityRule {
                arity: rule.arity,
                schemata: declared_schemata(rule, default_sort),
                introduces: None,
                consumes: None,
            })
            .collect();
        Ok(Self {
            source,
            rules,
            default_sort,
            lexical_support_ids: Mutex::new(FastMap::default()),
            refinements: Mutex::new(FastMap::default()),
            premise_expansions: Mutex::new(FastMap::default()),
            binary_projection_cache: Arc::new(Mutex::new(BinaryProjectionCache::default())),
        })
    }

    /// Report whether this template owns the supplied lowered declaration.
    #[must_use]
    pub fn matches(&self, rules: &[ReachabilityRule], default_sort: SortId) -> bool {
        self.default_sort == default_sort && self.source == rules
    }

    fn lexical_support_id(&self, support: &SpanSupport) -> u64 {
        let mut identifiers = self
            .lexical_support_ids
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let next = u64::try_from(identifiers.len()).unwrap_or(u64::MAX);
        *identifiers.entry(support.clone()).or_insert(next)
    }

    fn refinement(&self, key: &SharedRefinementCacheKey) -> Option<SharedRefinement> {
        self.refinements
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(key)
            .cloned()
    }

    fn remember_refinement(&self, key: SharedRefinementCacheKey, refinement: SharedRefinement) {
        self.refinements
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .entry(key)
            .or_insert(refinement);
    }

    fn premise_expansions(&self, rule: usize, demand: &Pattern) -> Arc<[PremiseExpansion]> {
        let key = (rule, demand.clone());
        let cached = self
            .premise_expansions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&key)
            .cloned();
        if let Some(rows) = cached {
            return rows;
        }
        let rows = self.rules.get(rule).map_or_else(
            || Arc::from([]),
            |declaration| {
                indexed_premise_demands(declaration, demand, self.default_sort)
                    .into_iter()
                    .map(|(row, conclusion, schema)| {
                        (PatternRow::from(row), Arc::new(conclusion), schema)
                    })
                    .collect()
            },
        );
        self.premise_expansions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .entry(key)
            .or_insert_with(|| Arc::clone(&rows));
        rows
    }

    fn binary_projection_row(&self, row: &PatternRow) -> usize {
        let mut cache = self
            .binary_projection_cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(identifier) = cache.row_ids.get(row) {
            return *identifier;
        }
        let identifier = cache.rows.len();
        cache.rows.push(BinaryProjectionRowCache::new(row));
        cache.row_ids.insert(Arc::clone(row), identifier);
        identifier
    }
}

fn inverse_node_matches(
    operator: &OperatorPattern,
    arguments: &[Pattern],
    payload: &PayloadPattern,
    demand: &Pattern,
    bindings: InverseBindings,
) -> Vec<InverseBindings> {
    let Pattern::Node {
        operator: demand_operator,
        arguments: demand_arguments,
        payload: demand_payload,
    } = demand
    else {
        return Vec::new();
    };
    if arguments.len() != demand_arguments.len() {
        return Vec::new();
    }
    let mut candidate = bindings;
    if !unify_payload(
        &operator_atom(operator),
        &operator_atom(demand_operator),
        &mut candidate.payloads,
    ) {
        return Vec::new();
    }
    match (payload_atom(payload), payload_atom(demand_payload)) {
        (None, None) => {}
        (Some(left), Some(right)) if unify_payload(&left, &right, &mut candidate.payloads) => {}
        _ => return Vec::new(),
    }
    let mut states = vec![candidate];
    for (argument, demand_argument) in arguments.iter().zip(demand_arguments) {
        states = states
            .into_iter()
            .flat_map(|state| inverse_matches(argument, demand_argument, state))
            .collect();
        if states.is_empty() {
            break;
        }
    }
    states
}

fn inverse_matches(
    pattern: &Pattern,
    demand: &Pattern,
    bindings: InverseBindings,
) -> Vec<InverseBindings> {
    match pattern {
        Pattern::Term { .. } => {
            let mut candidate = bindings;
            unify(
                pattern,
                demand,
                &mut candidate.terms,
                &mut candidate.payloads,
            )
            .then_some(candidate)
            .into_iter()
            .collect()
        }
        Pattern::Node {
            operator,
            arguments,
            payload,
        } => inverse_node_matches(operator, arguments, payload, demand, bindings),
        Pattern::Spine {
            name,
            core,
            operators,
            slot,
            min_order,
            max_order,
        } => {
            let mut frames = Vec::new();
            let mut remainders = vec![demand.clone()];
            let mut current = demand;
            while max_order.is_none_or(|maximum| frames.len() < maximum) {
                let Pattern::Node {
                    operator: OperatorPattern::Fixed(Some(operator)),
                    arguments,
                    ..
                } = current
                else {
                    break;
                };
                if !operators.contains(operator) || *slot >= arguments.len() {
                    break;
                }
                frames.push((current.clone(), *slot));
                current = &arguments[*slot];
                remainders.push(current.clone());
            }
            let mut states = Vec::new();
            for order in *min_order..=frames.len() {
                for mut candidate in inverse_matches(core, &remainders[order], bindings.clone()) {
                    if candidate
                        .spines
                        .get(name)
                        .is_some_and(|held| held != &frames[..order])
                    {
                        continue;
                    }
                    candidate
                        .spines
                        .insert(name.clone(), frames[..order].to_vec());
                    states.push(candidate);
                }
            }
            states
        }
    }
}

fn inverse_substitute(pattern: &Pattern, bindings: &InverseBindings) -> Pattern {
    match pattern {
        Pattern::Term { name, sort } => {
            bindings.terms.get(&term_variable(name, *sort)).map_or_else(
                || pattern.clone(),
                |held| inverse_substitute(held, bindings),
            )
        }
        Pattern::Node {
            operator,
            arguments,
            payload,
        } => {
            let resolved_operator =
                match apply_payload(&operator_atom(operator), &bindings.payloads) {
                    PayloadAtom::Operator(operator) => Some(OperatorPattern::Fixed(operator)),
                    PayloadAtom::Variable(name) => Some(OperatorPattern::Variable(name)),
                    PayloadAtom::Integer(_) => None,
                };
            let resolved_payload = payload_atom(payload).map_or(
                Some(PayloadPattern::None),
                |atom| match apply_payload(&atom, &bindings.payloads) {
                    PayloadAtom::Integer(value) => Some(PayloadPattern::Integer(value)),
                    PayloadAtom::Variable(name) => Some(PayloadPattern::Variable(name)),
                    PayloadAtom::Operator(_) => None,
                },
            );
            let (operator, payload) = match (resolved_operator, resolved_payload) {
                (Some(operator), Some(payload)) => (operator, payload),
                _ => (operator.clone(), payload.clone()),
            };
            Pattern::Node {
                operator,
                arguments: arguments
                    .iter()
                    .map(|argument| inverse_substitute(argument, bindings))
                    .collect(),
                payload,
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
            let core = inverse_substitute(core, bindings);
            let Some(frames) = bindings.spines.get(name) else {
                return Pattern::Spine {
                    name: name.clone(),
                    core: Box::new(core),
                    operators: operators.clone(),
                    slot: *slot,
                    min_order: *min_order,
                    max_order: *max_order,
                };
            };
            frames
                .iter()
                .rev()
                .fold(core, |built, (frame, frame_slot)| {
                    let Pattern::Node {
                        operator,
                        mut arguments,
                        payload,
                    } = frame.clone()
                    else {
                        return built;
                    };
                    arguments[*frame_slot] = built;
                    Pattern::Node {
                        operator,
                        arguments,
                        payload,
                    }
                })
        }
    }
}

fn indexed_premise_demands(
    rule: &ReachabilityRule,
    demand: &Pattern,
    default_sort: SortId,
) -> Vec<(Vec<Pattern>, Pattern, Option<usize>)> {
    if rule.schemata.is_empty() {
        let (premises, conclusion) = canonical_schema(
            &wildcard_row(rule.arity, default_sort),
            demand,
            default_sort,
        );
        return vec![(premises, conclusion, None)];
    }
    let mut rows = Vec::new();
    for (schema, (premises, conclusion)) in rule.schemata.iter().enumerate() {
        if contains_spine(conclusion) {
            if !is_ground(demand) {
                push_unique(&mut rows, {
                    let (premises, conclusion) = canonical_schema(
                        &wildcard_row(rule.arity, default_sort),
                        demand,
                        default_sort,
                    );
                    (premises, conclusion, Some(schema))
                });
                continue;
            }
            for bindings in inverse_matches(conclusion, demand, InverseBindings::default()) {
                let substituted = premises
                    .iter()
                    .map(|premise| inverse_substitute(premise, &bindings))
                    .collect::<Vec<_>>();
                let (premises, conclusion) = canonical_schema(&substituted, demand, default_sort);
                push_unique(&mut rows, (premises, conclusion, Some(schema)));
            }
            continue;
        }
        let mut terms = TermBindings::default();
        let mut payloads = PayloadBindings::default();
        if unify(conclusion, demand, &mut terms, &mut payloads) {
            let substituted = premises
                .iter()
                .map(|premise| substitute(premise, &terms, &payloads))
                .collect::<Vec<_>>();
            let (premises, conclusion) = canonical_schema(&substituted, demand, default_sort);
            push_unique(&mut rows, (premises, conclusion, Some(schema)));
        }
    }
    rows
}

fn premise_demands(
    rule: &ReachabilityRule,
    demand: &Pattern,
    default_sort: SortId,
) -> Vec<Vec<Pattern>> {
    let mut rows = Vec::new();
    for (row, _, _) in indexed_premise_demands(rule, demand, default_sort) {
        push_unique(&mut rows, row);
    }
    rows
}

fn generalize(patterns: &[Pattern], default_sort: SortId, next: &mut usize) -> Pattern {
    let Some(first) = patterns.first() else {
        let name = format!("g{next}");
        *next += 1;
        return Pattern::Term {
            name,
            sort: default_sort,
        };
    };
    if patterns.iter().all(|pattern| pattern == first) {
        return first.clone();
    }
    if let Pattern::Node {
        operator,
        arguments,
        payload,
    } = first
        && patterns.iter().all(|pattern| {
            matches!(
                pattern,
                Pattern::Node {
                    operator: held_operator,
                    arguments: held_arguments,
                    payload: held_payload,
                } if held_operator == operator
                    && held_payload == payload
                    && held_arguments.len() == arguments.len()
            )
        })
    {
        return Pattern::Node {
            operator: operator.clone(),
            arguments: (0..arguments.len())
                .map(|position| {
                    generalize(
                        &patterns
                            .iter()
                            .filter_map(|pattern| match pattern {
                                Pattern::Node { arguments, .. } => arguments.get(position).cloned(),
                                Pattern::Term { .. } | Pattern::Spine { .. } => None,
                            })
                            .collect::<Vec<_>>(),
                        default_sort,
                        next,
                    )
                })
                .collect(),
            payload: payload.clone(),
        };
    }
    let name = format!("g{next}");
    *next += 1;
    Pattern::Term {
        name,
        sort: default_sort,
    }
}

fn is_ground(pattern: &Pattern) -> bool {
    match pattern {
        Pattern::Term { .. } | Pattern::Spine { .. } => false,
        Pattern::Node {
            operator,
            arguments,
            payload,
        } => {
            matches!(operator, OperatorPattern::Fixed(_))
                && !matches!(payload, PayloadPattern::Variable(_))
                && arguments.iter().all(is_ground)
        }
    }
}

/// Narrow one row's demands on its lexical children.
///
/// A refinement only narrows the demands a row places on its lexical
/// children. A row whose lexical supports would exceed `refinement_pairs`
/// binding-and-candidate pairs keeps its unrefined form, which demands every
/// category the refinement would have admitted, so the bound trades pruning
/// for planning time without losing a root-reachable item.
fn lexically_refined(
    children: &[Span],
    row: &[Pattern],
    lexical: &FastMap<Span, SpanSupport>,
    default_sort: SortId,
    refinement_pairs: usize,
) -> Vec<Vec<Pattern>> {
    if children.iter().all(|(start, stop)| stop - start != 1) {
        return vec![row.to_vec()];
    }
    let mut rows = Vec::new();
    if children
        .iter()
        .any(|child| child.1 - child.0 == 1 && lexical.get(child).is_none_or(|held| !held.complete))
    {
        rows.push(row.to_vec());
    }
    let mut states = vec![(TermBindings::default(), PayloadBindings::default())];
    for (child, premise) in children.iter().zip(row) {
        if child.1 - child.0 != 1 {
            continue;
        }
        let Some(held) = lexical.get(child) else {
            return rows;
        };
        if states.len().saturating_mul(held.patterns.len()) > refinement_pairs {
            push_unique(&mut rows, row.to_vec());
            return rows;
        }
        let mut next = Vec::new();
        for candidate in &held.patterns {
            for (held_terms, held_payloads) in &states {
                if !may_unify(premise, candidate, held_terms) {
                    continue;
                }
                let mut terms = held_terms.clone();
                let mut payloads = held_payloads.clone();
                if unify(premise, candidate, &mut terms, &mut payloads) {
                    next.push((terms, payloads));
                }
            }
        }
        states = next;
        if states.is_empty() {
            return rows;
        }
    }
    let refined = states
        .into_iter()
        .map(|(terms, payloads)| {
            row.iter()
                .map(|pattern| substitute(pattern, &terms, &payloads))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    if !refined.is_empty() {
        let mut next = 0;
        let generalized = (0..row.len())
            .map(|position| {
                generalize(
                    &refined
                        .iter()
                        .filter_map(|held| held.get(position).cloned())
                        .collect::<Vec<_>>(),
                    default_sort,
                    &mut next,
                )
            })
            .collect::<Vec<_>>();
        push_unique(&mut rows, generalized);
    }
    rows
}

fn lexically_refined_applications(
    children: &[Span],
    premises: &[Pattern],
    conclusion: &Pattern,
    lexical: &FastMap<Span, SpanSupport>,
    default_sort: SortId,
    refinement_pairs: usize,
) -> Vec<(Vec<Pattern>, Pattern)> {
    let mut clause = Vec::with_capacity(premises.len() + 1);
    clause.extend_from_slice(premises);
    clause.push(conclusion.clone());
    lexically_refined(children, &clause, lexical, default_sort, refinement_pairs)
        .into_iter()
        .filter_map(|mut refined| {
            let conclusion = refined.pop()?;
            Some((refined, conclusion))
        })
        .collect()
}

fn lexical_supports(
    lexical: &HashMap<Span, Vec<Term>>,
    rules: &[ReachabilityRule],
    depth: usize,
) -> FastMap<Span, SpanSupport> {
    lexical
        .iter()
        .map(|(span, terms)| {
            let patterns = terms.iter().map(term_pattern).collect::<Vec<_>>();
            let mut closure = patterns.clone();
            let mut frontier = patterns;
            let mut complete = true;
            for _ in 0..depth {
                let mut next = Vec::new();
                for input in &frontier {
                    for rule in rules.iter().filter(|rule| rule.arity == 1) {
                        for (premises, conclusion) in &rule.schemata {
                            let [premise] = premises.as_slice() else {
                                complete = false;
                                continue;
                            };
                            if contains_spine(premise) || contains_spine(conclusion) {
                                complete = false;
                                continue;
                            }
                            let mut term_bindings = TermBindings::default();
                            let mut payload_bindings = PayloadBindings::default();
                            if !unify(premise, input, &mut term_bindings, &mut payload_bindings) {
                                continue;
                            }
                            let output = substitute(conclusion, &term_bindings, &payload_bindings);
                            if !is_ground(&output) {
                                complete = false;
                            } else if !closure.contains(&output) && !next.contains(&output) {
                                next.push(output);
                            }
                        }
                    }
                }
                if next.is_empty() {
                    break;
                }
                closure.extend(next.iter().cloned());
                frontier = next;
            }
            (
                *span,
                SpanSupport {
                    patterns: closure,
                    complete,
                },
            )
        })
        .collect()
}

fn ground_targets(rules: &[ReachabilityRule], roots: &[Term]) -> FastSet<Pattern> {
    let mut targets = roots.iter().map(term_pattern).collect::<FastSet<_>>();
    for rule in rules {
        for (premises, _) in &rule.schemata {
            targets.extend(
                premises
                    .iter()
                    .filter(|premise| is_ground(premise))
                    .cloned(),
            );
        }
    }
    targets
}

impl GroundSupport {
    fn new(
        template: &ReachabilityTemplate,
        tokens: usize,
        rules: &[ReachabilityRule],
        roots: &[Term],
        lexical: &HashMap<Span, Vec<Term>>,
        unary_depth: usize,
    ) -> Self {
        let target_set = ground_targets(rules, roots);
        let targets = target_set.iter().cloned().collect::<Vec<_>>();
        let mut decomposition_cache = FastMap::<(Span, usize), Vec<Vec<Span>>>::default();
        let mut spans = FastMap::<Span, FastSet<Pattern>>::default();
        let mut nonempty = FastSet::default();
        for width in 1..=tokens {
            for start in 0..=tokens - width {
                let span = (start, start + width);
                let lexical_terms = lexical.get(&span).map_or(&[][..], Vec::as_slice);
                if !lexical_terms.is_empty()
                    || rules.iter().filter(|rule| rule.arity >= 2).any(|rule| {
                        decomposition_cache
                            .entry((span, rule.arity))
                            .or_insert_with(|| decompositions(span.0, span.1, rule.arity))
                            .iter()
                            .any(|parts| parts.iter().all(|part| nonempty.contains(part)))
                    })
                {
                    nonempty.insert(span);
                }
                let mut held = targets
                    .iter()
                    .filter(|target| {
                        lexical_terms
                            .iter()
                            .any(|term| matches_pattern(target, term))
                    })
                    .cloned()
                    .collect::<FastSet<_>>();
                for target in &targets {
                    'rules: for (position, rule) in
                        rules.iter().enumerate().filter(|(_, rule)| rule.arity >= 2)
                    {
                        let rows = template.premise_expansions(position, target);
                        let parts_rows = decomposition_cache
                            .entry((span, rule.arity))
                            .or_insert_with(|| decompositions(span.0, span.1, rule.arity));
                        for parts in parts_rows.iter() {
                            if rows.iter().any(|(row, _, _)| {
                                row.len() == parts.len()
                                    && row.iter().zip(parts).all(|(premise, part)| {
                                        if is_ground(premise) {
                                            spans
                                                .get(part)
                                                .is_some_and(|support| support.contains(premise))
                                        } else {
                                            nonempty.contains(part)
                                        }
                                    })
                            }) {
                                held.insert(target.clone());
                                break 'rules;
                            }
                        }
                    }
                }
                for _ in 0..unary_depth {
                    let current = held.clone();
                    let mut changed = false;
                    for target in &targets {
                        if held.contains(target) {
                            continue;
                        }
                        let supported = rules
                            .iter()
                            .enumerate()
                            .filter(|(_, rule)| rule.arity == 1)
                            .any(|(position, _rule)| {
                                template.premise_expansions(position, target).iter().any(
                                    |(row, _, _)| {
                                        let [premise] = row.as_ref() else {
                                            return false;
                                        };
                                        if is_ground(premise) {
                                            current.contains(premise)
                                        } else {
                                            nonempty.contains(&span)
                                        }
                                    },
                                )
                            });
                        if supported {
                            changed |= held.insert(target.clone());
                        }
                    }
                    if !changed {
                        break;
                    }
                }
                if !held.is_empty() {
                    spans.insert(span, held);
                }
            }
        }
        Self {
            targets: target_set,
            spans,
            nonempty,
        }
    }

    fn permits(&self, parts: &[Span], row: &[Pattern]) -> bool {
        row.len() == parts.len()
            && row.iter().zip(parts).all(|(premise, part)| {
                if is_ground(premise) && self.targets.contains(premise) {
                    self.spans
                        .get(part)
                        .is_some_and(|support| support.contains(premise))
                } else {
                    self.nonempty.contains(part)
                }
            })
    }
}

struct DemandUnaryIndex<'a> {
    exact: Vec<(Pattern, Vec<Vec<Pattern>>)>,
    all: Vec<&'a ReachabilityRule>,
    dynamic: Vec<usize>,
}

impl<'a> DemandUnaryIndex<'a> {
    fn new(rules: &'a [ReachabilityRule], default_sort: SortId) -> Self {
        let mut exact = Vec::<(Pattern, Vec<Vec<Pattern>>)>::new();
        let mut all = Vec::new();
        let mut dynamic = Vec::new();

        for rule in rules.iter().filter(|rule| rule.arity == 1) {
            let position = all.len();
            all.push(rule);
            let mut needs_dynamic = rule.schemata.is_empty();
            for (premises, conclusion) in &rule.schemata {
                if premises.iter().any(contains_spine)
                    || contains_spine(conclusion)
                    || !is_ground(conclusion)
                {
                    needs_dynamic = true;
                    continue;
                }
                let row = canonical_patterns(premises, default_sort);
                if let Some((_, rows)) = exact.iter_mut().find(|(held, _)| held == conclusion) {
                    push_unique(rows, row);
                } else {
                    exact.push((conclusion.clone(), vec![row]));
                }
            }
            if needs_dynamic {
                dynamic.push(position);
            }
        }

        Self {
            exact,
            all,
            dynamic,
        }
    }

    fn exact_premise_demands(&self, demand: &Pattern) -> Vec<Vec<Pattern>> {
        if !is_ground(demand) {
            return Vec::new();
        }

        self.exact
            .iter()
            .find(|(conclusion, _)| conclusion == demand)
            .map_or_else(Vec::new, |(_, rows)| rows.clone())
    }

    fn dynamic_premise_demands(
        &self,
        demand: &Pattern,
        default_sort: SortId,
    ) -> Vec<(usize, Vec<Pattern>)> {
        let positions: Box<dyn Iterator<Item = usize> + '_> = if is_ground(demand) {
            Box::new(self.dynamic.iter().copied())
        } else {
            Box::new(0..self.all.len())
        };
        positions
            .flat_map(|position| {
                premise_demands(self.all[position], demand, default_sort)
                    .into_iter()
                    .map(move |row| (position, row))
            })
            .collect()
    }

    const fn len(&self) -> usize {
        self.all.len()
    }
}

fn subsumes(general: &Pattern, specific: &Pattern) -> bool {
    type SubsumptionTerms = Vec<(TermVariable, Pattern)>;
    type SubsumptionPayloads = Vec<(String, PayloadAtom)>;

    fn covers_payload(
        general: &PayloadAtom,
        specific: &PayloadAtom,
        bindings: &mut SubsumptionPayloads,
    ) -> bool {
        match general {
            PayloadAtom::Variable(name) => {
                if let Some((_, held)) = bindings.iter().find(|(held, _)| held == name) {
                    held == specific
                } else {
                    bindings.push((name.clone(), specific.clone()));
                    true
                }
            }
            _ => general == specific,
        }
    }

    fn covers(
        general: &Pattern,
        specific: &Pattern,
        terms: &mut SubsumptionTerms,
        payloads: &mut SubsumptionPayloads,
    ) -> bool {
        match general {
            Pattern::Term { name, sort } => {
                let variable = term_variable(name, *sort);
                if let Some((_, held)) = terms.iter().find(|(held, _)| held == &variable) {
                    held == specific
                } else {
                    terms.push((variable, specific.clone()));
                    true
                }
            }
            Pattern::Node {
                operator,
                arguments,
                payload,
            } => {
                let Pattern::Node {
                    operator: specific_operator,
                    arguments: specific_arguments,
                    payload: specific_payload,
                } = specific
                else {
                    return false;
                };
                arguments.len() == specific_arguments.len()
                    && covers_payload(
                        &operator_atom(operator),
                        &operator_atom(specific_operator),
                        payloads,
                    )
                    && match (payload_atom(payload), payload_atom(specific_payload)) {
                        (None, None) => true,
                        (Some(general), Some(specific)) => {
                            covers_payload(&general, &specific, payloads)
                        }
                        _ => false,
                    }
                    && arguments
                        .iter()
                        .zip(specific_arguments)
                        .all(|(general, specific)| covers(general, specific, terms, payloads))
            }
            Pattern::Spine { .. } => general == specific,
        }
    }

    covers(general, specific, &mut Vec::new(), &mut Vec::new())
}

impl DemandSet {
    fn add(&mut self, pattern: Pattern) -> Option<SharedPattern> {
        let pattern = Arc::new(pattern);
        if self.exact.contains(&pattern) {
            return None;
        }
        if self
            .general
            .iter()
            .any(|existing| subsumes(existing.as_ref(), pattern.as_ref()))
        {
            return None;
        }
        if !is_ground(pattern.as_ref()) {
            let exact = &mut self.exact;
            self.patterns.retain(|existing| {
                let kept = !subsumes(pattern.as_ref(), existing.as_ref());
                if !kept {
                    exact.remove(existing);
                }
                kept
            });
            self.general
                .retain(|existing| !subsumes(pattern.as_ref(), existing.as_ref()));
            self.general.push(Arc::clone(&pattern));
        }
        self.exact.insert(Arc::clone(&pattern));
        self.patterns.push(Arc::clone(&pattern));
        Some(pattern)
    }
}

fn add_demand(demands: &mut FastMap<Span, DemandSet>, span: Span, pattern: Pattern) -> bool {
    demands.entry(span).or_default().add(pattern).is_some()
}

fn demand_unary_closure(
    demands: &mut DemandSet,
    rules: &DemandUnaryIndex<'_>,
    default_sort: SortId,
) {
    // One forward round visits every initial arrival together, so the stored
    // unary graph may contain paths longer than the round count. Exact rows
    // therefore close transitively. Dynamic rows widen per rule so recursive
    // inverse schemata converge to one conservative finite demand.
    let mut frontier = demands.patterns.clone();
    let mut dynamic_summaries = vec![None::<Pattern>; rules.len()];
    while !frontier.is_empty() {
        let mut next = Vec::new();
        for demand in &frontier {
            if matches!(demand.as_ref(), Pattern::Term { .. }) {
                return;
            }
            for row in rules.exact_premise_demands(demand) {
                if let [premise] = row.as_slice()
                    && let Some(premise) = demands.add(premise.clone())
                {
                    next.push(premise);
                }
            }
            for (position, row) in rules.dynamic_premise_demands(demand, default_sort) {
                let [premise] = row.as_slice() else {
                    continue;
                };
                let summary = match &dynamic_summaries[position] {
                    None => premise.clone(),
                    Some(held) if subsumes(held, premise) => continue,
                    Some(held) => {
                        let mut next_variable = 0;
                        generalize(
                            &[held.clone(), premise.clone()],
                            default_sort,
                            &mut next_variable,
                        )
                    }
                };
                dynamic_summaries[position] = Some(summary.clone());
                if let Some(premise) = demands.add(summary) {
                    next.push(premise);
                }
            }
        }
        if next.is_empty() {
            return;
        }
        frontier = next;
    }
}

fn add_application(
    applications: &mut FastMap<Application, ApplicationRows>,
    key: Application,
    parts: &[Span],
    row: PatternRowId,
    output: SharedPattern,
    schema: Option<usize>,
) {
    let (ordered, seen) = applications
        .entry(key)
        .or_default()
        .entry(parts.to_vec())
        .or_default();
    let application = ApplicationRow {
        premises: row,
        conclusion: output,
        schema,
    };
    if seen.insert(application.clone()) {
        ordered.push(application);
    }
}

struct ReachabilityEnvelopeBuilder<'a> {
    template: &'a ReachabilityTemplate,
    rules: &'a [ReachabilityRule],
    default_sort: SortId,
    refinement_pairs: usize,
    lexical: FastMap<Span, SpanSupport>,
    lexical_ids: FastMap<Span, u64>,
    ground: GroundSupport,
    unary: DemandUnaryIndex<'a>,
    envelope: ReachabilityEnvelope,
    decomposition_cache: FastMap<(Span, usize), Arc<[Vec<Span>]>>,
    refinement_cache: FastMap<RefinementCacheKey, Vec<(PatternRowId, SharedPattern)>>,
}

impl ReachabilityEnvelopeBuilder<'_> {
    fn intern_pattern_row(&mut self, row: PatternRow) -> PatternRowId {
        if let Some(identifier) = self.envelope.pattern_row_ids.get(&row) {
            return *identifier;
        }
        let identifier = self.envelope.pattern_rows.len();
        let projection = self.template.binary_projection_row(&row);
        self.envelope.pattern_rows.push(Arc::clone(&row));
        self.envelope.pattern_row_ids.insert(row, identifier);
        self.envelope.projection_rows.push(projection);
        identifier
    }

    fn refined_rows(
        &mut self,
        children: &[Span],
        base_row: PatternRowId,
        row: &PatternRow,
        conclusion: &SharedPattern,
        schema: Option<usize>,
    ) -> Vec<RefinedApplication> {
        let lexical_key = children
            .iter()
            .map(|part| {
                (part.1 - part.0 == 1)
                    .then(|| self.lexical_ids.get(part).copied())
                    .flatten()
            })
            .collect::<Vec<_>>();
        if lexical_key.iter().all(Option::is_none) {
            return vec![(base_row, Arc::clone(row), Arc::clone(conclusion))];
        }
        let cache_key = (
            base_row,
            Arc::clone(conclusion),
            schema,
            lexical_key.clone(),
        );
        if let Some(held) = self.refinement_cache.get(&cache_key).cloned() {
            return held
                .into_iter()
                .filter_map(|(identifier, conclusion)| {
                    self.envelope
                        .pattern_rows
                        .get(identifier)
                        .cloned()
                        .map(|row| (identifier, row, conclusion))
                })
                .collect();
        }
        let shared_key = (
            Arc::clone(row),
            Arc::clone(conclusion),
            schema,
            lexical_key,
            self.refinement_pairs,
        );
        let shared = self.template.refinement(&shared_key).unwrap_or_else(|| {
            let refined = lexically_refined_applications(
                children,
                row.as_ref(),
                conclusion.as_ref(),
                &self.lexical,
                self.default_sort,
                self.refinement_pairs,
            )
            .into_iter()
            .map(|(premises, conclusion)| (PatternRow::from(premises), Arc::new(conclusion)))
            .collect::<Arc<[_]>>();
            self.template
                .remember_refinement(shared_key, Arc::clone(&refined));
            refined
        });
        let refined = shared
            .iter()
            .map(|(premises, conclusion)| {
                let identifier = self.intern_pattern_row(Arc::clone(premises));
                (identifier, Arc::clone(premises), Arc::clone(conclusion))
            })
            .collect::<Vec<_>>();
        self.refinement_cache.insert(
            cache_key,
            refined
                .iter()
                .map(|(identifier, _, conclusion)| (*identifier, Arc::clone(conclusion)))
                .collect(),
        );
        refined
    }

    fn propagate(&mut self, span: Span, unary_depth: usize) {
        let mut demands = self.envelope.demands.remove(&span).unwrap_or_default();
        if unary_depth > 0 {
            demand_unary_closure(&mut demands, &self.unary, self.default_sort);
        }
        for demand in &demands.patterns {
            for (position, rule) in self
                .rules
                .iter()
                .enumerate()
                .filter(|(_, rule)| rule.arity >= 2)
            {
                let Ok(rule_id) = u16::try_from(position) else {
                    continue;
                };
                let rows = self.template.premise_expansions(position, demand);
                let decompositions = self
                    .decomposition_cache
                    .entry((span, rule.arity))
                    .or_insert_with(|| Arc::from(decompositions(span.0, span.1, rule.arity)))
                    .clone();
                for (row, conclusion, schema) in rows.iter() {
                    let base_row = self.intern_pattern_row(Arc::clone(row));
                    for children in decompositions.iter() {
                        for (row_id, refined, conclusion) in
                            self.refined_rows(children, base_row, row, conclusion, *schema)
                        {
                            if !self.ground.permits(children, &refined) {
                                continue;
                            }
                            add_application(
                                &mut self.envelope.applications,
                                (span, rule_id),
                                children,
                                row_id,
                                conclusion,
                                *schema,
                            );
                            for (child, premise) in children.iter().copied().zip(refined.iter()) {
                                for projected in canonical_patterns(
                                    std::slice::from_ref(premise),
                                    self.default_sort,
                                ) {
                                    add_demand(&mut self.envelope.demands, child, projected);
                                }
                            }
                        }
                    }
                }
            }
        }
        self.envelope.demands.insert(span, demands);
    }
}

fn reachability_envelope(
    template: &ReachabilityTemplate,
    tokens: usize,
    roots: &[Term],
    lexical: &HashMap<Span, Vec<Term>>,
    unary_depth: usize,
    refinement_pairs: usize,
) -> ReachabilityEnvelope {
    let rules = template.rules.as_slice();
    let default_sort = template.default_sort;
    let ground = GroundSupport::new(template, tokens, rules, roots, lexical, unary_depth);
    let lexical = lexical_supports(lexical, rules, unary_depth);
    let lexical_ids = lexical
        .iter()
        .map(|(span, support)| (*span, template.lexical_support_id(support)))
        .collect();
    let unary = DemandUnaryIndex::new(rules, default_sort);
    let mut builder = ReachabilityEnvelopeBuilder {
        template,
        rules,
        default_sort,
        refinement_pairs,
        lexical,
        lexical_ids,
        ground,
        unary,
        envelope: ReachabilityEnvelope::default(),
        decomposition_cache: FastMap::default(),
        refinement_cache: FastMap::default(),
    };
    let root_span = (0, tokens);
    for root in roots {
        add_demand(&mut builder.envelope.demands, root_span, term_pattern(root));
    }
    for width in (1..=tokens).rev() {
        for start in 0..=tokens - width {
            let span = (start, start + width);
            if builder.envelope.demands.contains_key(&span) {
                builder.propagate(span, unary_depth);
            }
        }
    }
    builder.envelope
}

impl ReachabilityPlan {
    /// Build one exact top-down plan from roots, schemata, and lexical representatives.
    ///
    /// `refinement_pairs` bounds the binding-and-candidate pairs one lexical
    /// refinement of a rule row enumerates; a row past the bound keeps its
    /// unrefined, wider demands.
    ///
    /// # Errors
    ///
    /// Returns [`RuleError`] when the rule count exceeds the native identifier width.
    pub fn new(
        tokens: usize,
        rules: &[ReachabilityRule],
        roots: &[Term],
        lexical: &HashMap<Span, Vec<Term>>,
        unary_depth: usize,
        default_sort: SortId,
        refinement_pairs: usize,
    ) -> Result<Self, RuleError> {
        let template = ReachabilityTemplate::new(rules, default_sort)?;
        Ok(Self::from_template(
            &template,
            tokens,
            roots,
            lexical,
            unary_depth,
            refinement_pairs,
        ))
    }

    /// Build one sentence-specific plan from a grammar-owned compilation.
    ///
    /// `refinement_pairs` bounds each lexical refinement as in [`Self::new`].
    #[must_use]
    pub fn from_template(
        template: &ReachabilityTemplate,
        tokens: usize,
        roots: &[Term],
        lexical: &HashMap<Span, Vec<Term>>,
        unary_depth: usize,
        refinement_pairs: usize,
    ) -> Self {
        let envelope = reachability_envelope(
            template,
            tokens,
            roots,
            lexical,
            unary_depth,
            refinement_pairs,
        );
        let demands = envelope
            .demands
            .into_iter()
            .map(|(span, held)| (span, held.patterns))
            .collect();
        let applications: FastMap<Application, FastMap<Vec<Span>, Vec<ApplicationRow>>> = envelope
            .applications
            .into_iter()
            .map(|(application, rows)| {
                (
                    application,
                    rows.into_iter()
                        .map(|(parts, (ordered, _))| (parts, ordered))
                        .collect(),
                )
            })
            .collect();
        let mut binary_application_rules = FastMap::<(Span, [Span; 2]), Vec<u16>>::default();
        for ((span, rule), decompositions) in &applications {
            for parts in decompositions.keys() {
                let [left, right] = parts.as_slice() else {
                    continue;
                };
                binary_application_rules
                    .entry((*span, [*left, *right]))
                    .or_default()
                    .push(*rule);
            }
        }
        let pattern_rows = envelope.pattern_rows;
        let projection_rows = envelope.projection_rows;
        let schemata = template
            .rules
            .iter()
            .map(|rule| rule.schemata.clone())
            .collect();
        Self {
            demands,
            applications,
            binary_application_rules,
            pattern_rows,
            projection_rows,
            schemata,
            binary_projection_cache: Arc::clone(&template.binary_projection_cache),
        }
    }

    /// Return whether a representative category is demanded at one span.
    #[must_use]
    pub fn keeps(&self, span: Span, representative: &Term) -> bool {
        self.demands.get(&span).is_some_and(|patterns| {
            patterns
                .iter()
                .any(|pattern| matches_pattern(pattern, representative))
        })
    }

    /// Compile the demanded binary applications over two populated category axes.
    #[must_use]
    pub fn binary_permits(
        &self,
        span: Span,
        parts: &[Span; 2],
        left: &[Term],
        right: &[Term],
    ) -> BinaryPermits {
        let mut rules = FastMap::default();
        let mut materialized: Vec<Option<(ProjectionAxis, ProjectionAxis)>> =
            vec![None; self.pattern_rows.len()];
        let Some(application_rules) = self.binary_application_rules.get(&(span, *parts)) else {
            return BinaryPermits { rules };
        };
        let mut projection_cache = self
            .binary_projection_cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for rule in application_rules {
            let Some(decompositions) = self.applications.get(&(span, *rule)) else {
                continue;
            };
            let Some(pattern_rows) = decompositions.get(parts.as_slice()) else {
                continue;
            };
            let mut permit_rows = Vec::new();
            for application in pattern_rows {
                let Some(patterns) = self.pattern_rows.get(application.premises) else {
                    continue;
                };
                let [left_pattern, right_pattern] = patterns.as_ref() else {
                    continue;
                };
                let Some(projection_row) = self.projection_rows.get(application.premises).copied()
                else {
                    continue;
                };
                let axes = materialized[application.premises]
                    .clone()
                    .unwrap_or_else(|| {
                        let Some(cache) = projection_cache.rows.get_mut(projection_row) else {
                            return (Arc::from([]), Arc::from([]));
                        };
                        let axes = (
                            cache.axis(0, left_pattern, left),
                            cache.axis(1, right_pattern, right),
                        );
                        materialized[application.premises] = Some(axes.clone());
                        axes
                    });
                let (left, right) = axes;
                if !left.is_empty() && !right.is_empty() {
                    permit_rows.push(BinaryPermitRow {
                        application: application.clone(),
                        left,
                        right,
                    });
                }
            }
            if !permit_rows.is_empty() {
                rules.insert(*rule, permit_rows);
            }
        }
        BinaryPermits { rules }
    }

    /// Return whether one exact representative tuple is demanded for an application.
    #[must_use]
    pub fn permits(&self, span: Span, parts: &[Span], rule: u16, representatives: &[Term]) -> bool {
        if parts.len() < 2 {
            return true;
        }
        self.applications
            .get(&(span, rule))
            .and_then(|rows| rows.get(parts))
            .is_some_and(|rows| {
                rows.iter().any(|application| {
                    self.pattern_rows
                        .get(application.premises)
                        .is_some_and(|row| {
                            crate::rule::reachability_patterns_match(row, representatives)
                        })
                })
            })
    }

    /// Return whether one representative conclusion belongs to a demanded application row.
    #[must_use]
    pub fn permits_output(
        &self,
        span: Span,
        parts: &[Span],
        rule: u16,
        representatives: &[Term],
        output: &Term,
    ) -> bool {
        if parts.len() < 2 {
            return self.keeps(span, output);
        }
        let Some(rows) = self
            .applications
            .get(&(span, rule))
            .and_then(|rows| rows.get(parts))
        else {
            return false;
        };
        for application in rows {
            if self.application_permits_output(rule, application, representatives, output) {
                return true;
            }
        }
        false
    }

    /// Return whether a binary permit witness admits one exact conclusion.
    #[must_use]
    pub fn permits_binary_output(
        &self,
        witnesses: &mut BinaryPermitWitnesses<'_>,
        rule: u16,
        representatives: &[Term; 2],
        output: &Term,
    ) -> bool {
        witnesses.any_matching_application(
            rule,
            representatives[0].serial(),
            representatives[1].serial(),
            |application| {
                self.application_permits_output(rule, application, representatives, output)
            },
        )
    }

    fn application_permits_output(
        &self,
        rule: u16,
        application: &ApplicationRow,
        representatives: &[Term],
        output: &Term,
    ) -> bool {
        let Some(row) = self.pattern_rows.get(application.premises) else {
            return false;
        };
        if !crate::rule::reachability_application_matches(
            row,
            application.conclusion.as_ref(),
            representatives,
            output,
        ) {
            return false;
        }
        let Some(schema) = application.schema else {
            return true;
        };
        let Some((premises, conclusion)) = self
            .schemata
            .get(usize::from(rule))
            .and_then(|rows| rows.get(schema))
        else {
            return false;
        };
        crate::rule::reachability_application_matches(premises, conclusion, representatives, output)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::{
        DemandSet, ReachabilityPlan, ReachabilityRule, TermBindings, may_unify, term_pattern,
    };
    use crate::rule::{OperatorPattern, Pattern, PayloadPattern};
    use crate::term::{OperatorSpec, PayloadKind, SignatureSpec, Term, TermStore};

    const REFINEMENT: usize = 1 << 16;

    fn node(operator: u32, arguments: Vec<Pattern>) -> Pattern {
        Pattern::Node {
            operator: OperatorPattern::Fixed(Some(operator)),
            arguments,
            payload: PayloadPattern::None,
        }
    }

    #[test]
    fn the_unification_precheck_refuses_only_clashing_structure() {
        let mut terms = TermBindings::default();

        assert!(!may_unify(&node(0, vec![]), &node(1, vec![]), &terms));
        assert!(!may_unify(
            &node(2, vec![node(0, vec![])]),
            &node(2, vec![node(0, vec![]), node(0, vec![])]),
            &terms
        ));
        assert!(may_unify(&variable("x"), &node(1, vec![]), &terms));
        terms.insert(("x".to_owned(), 0), node(0, vec![]));
        assert!(!may_unify(&variable("x"), &node(1, vec![]), &terms));
        assert!(may_unify(&variable("x"), &node(0, vec![]), &terms));
    }

    #[test]
    fn a_demand_set_keeps_each_pattern_once_and_drops_subsumed_ones() {
        let mut demands = DemandSet::default();

        assert!(demands.add(node(0, vec![])).is_some());
        assert!(demands.add(node(0, vec![])).is_none());
        assert!(demands.add(variable("x")).is_some());
        assert!(demands.add(node(1, vec![])).is_none());
        assert_eq!(demands.patterns.len(), 1);
        assert_eq!(demands.exact.len(), 1);
    }

    fn store() -> Result<TermStore, crate::term::TermError> {
        TermStore::new(
            SignatureSpec {
                name: "reachability".to_owned(),
                sorts: vec!["Cat".to_owned()],
                operators: vec![
                    OperatorSpec {
                        name: "a".to_owned(),
                        sort: 0,
                        arguments: Vec::new(),
                        payload: PayloadKind::None,
                        erased: 0,
                        binding: Vec::new(),
                        shifts: Vec::new(),
                    },
                    OperatorSpec {
                        name: "b".to_owned(),
                        sort: 0,
                        arguments: Vec::new(),
                        payload: PayloadKind::None,
                        erased: 1,
                        binding: Vec::new(),
                        shifts: Vec::new(),
                    },
                    OperatorSpec {
                        name: "c".to_owned(),
                        sort: 0,
                        arguments: Vec::new(),
                        payload: PayloadKind::None,
                        erased: 2,
                        binding: Vec::new(),
                        shifts: Vec::new(),
                    },
                    OperatorSpec {
                        name: "pair".to_owned(),
                        sort: 0,
                        arguments: vec![0, 0],
                        payload: PayloadKind::None,
                        erased: 3,
                        binding: Vec::new(),
                        shifts: vec![Vec::new(), Vec::new()],
                    },
                ],
                vocabularies: vec![None],
                constant_cores: vec![None],
                constants: vec![None],
                bound: vec![None],
                free: vec![None],
                holes: vec![None],
                allows_vacuous_binding: true,
            },
            64,
        )
    }

    fn variable(name: &str) -> Pattern {
        Pattern::Term {
            name: name.to_owned(),
            sort: 0,
        }
    }

    fn pair_rule() -> ReachabilityRule {
        ReachabilityRule {
            arity: 2,
            schemata: vec![(
                vec![variable("left"), variable("right")],
                Pattern::Node {
                    operator: OperatorPattern::Fixed(Some(3)),
                    arguments: vec![variable("left"), variable("right")],
                    payload: PayloadPattern::None,
                },
            )],
            introduces: None,
            consumes: None,
        }
    }

    fn two_output_pair_rule() -> ReachabilityRule {
        ReachabilityRule {
            arity: 2,
            schemata: vec![
                (
                    vec![variable("left"), variable("right")],
                    Pattern::Node {
                        operator: OperatorPattern::Fixed(Some(3)),
                        arguments: vec![variable("left"), variable("right")],
                        payload: PayloadPattern::None,
                    },
                ),
                (
                    vec![variable("left"), variable("right")],
                    Pattern::Node {
                        operator: OperatorPattern::Fixed(Some(3)),
                        arguments: vec![variable("right"), variable("left")],
                        payload: PayloadPattern::None,
                    },
                ),
            ],
            introduces: None,
            consumes: None,
        }
    }

    fn sentence() -> Result<(TermStore, Vec<Term>, ReachabilityPlan), crate::rule::RuleError> {
        let store = store()?;
        let a = store.intern(0, Vec::new(), None)?;
        let b = store.intern(1, Vec::new(), None)?;
        let root = store.intern(3, vec![a.clone(), b.clone()], None)?;
        let lexical = HashMap::from([((0, 1), vec![a.clone()]), ((1, 2), vec![b.clone()])]);
        let plan = ReachabilityPlan::new(
            2,
            &[pair_rule()],
            std::slice::from_ref(&root),
            &lexical,
            0,
            0,
            REFINEMENT,
        )?;
        Ok((store, vec![a, b, root], plan))
    }

    #[test]
    fn one_unary_round_preserves_paths_between_initial_arrivals()
    -> Result<(), Box<dyn std::error::Error>> {
        let store = store()?;
        let a = store.intern(0, Vec::new(), None)?;
        let b = store.intern(1, Vec::new(), None)?;
        let root = store.intern(2, Vec::new(), None)?;
        let unary = |premise: &Term, conclusion: &Term| ReachabilityRule {
            arity: 1,
            schemata: vec![(vec![term_pattern(premise)], term_pattern(conclusion))],
            introduces: None,
            consumes: None,
        };
        let lexical = HashMap::from([((0, 1), vec![a.clone(), b.clone()])]);
        let plan = ReachabilityPlan::new(
            1,
            &[unary(&a, &b), unary(&b, &root)],
            std::slice::from_ref(&root),
            &lexical,
            1,
            0,
            REFINEMENT,
        )?;

        assert!(plan.keeps((0, 1), &a));
        assert!(plan.keeps((0, 1), &b));
        assert!(plan.keeps((0, 1), &root));
        Ok(())
    }

    #[test]
    fn growing_unary_inverse_demands_widen_to_a_finite_pattern()
    -> Result<(), Box<dyn std::error::Error>> {
        let store = store()?;
        let a = store.intern(0, Vec::new(), None)?;
        let root = store.intern(1, Vec::new(), None)?;
        let wrap = |inner: Pattern| Pattern::Node {
            operator: OperatorPattern::Fixed(Some(3)),
            arguments: vec![inner, term_pattern(&a)],
            payload: PayloadPattern::None,
        };
        let rule = ReachabilityRule {
            arity: 1,
            schemata: vec![(vec![wrap(variable("category"))], variable("category"))],
            introduces: None,
            consumes: None,
        };
        let lexical = HashMap::from([((0, 1), vec![root.clone()])]);
        let plan = ReachabilityPlan::new(
            1,
            &[rule],
            std::slice::from_ref(&root),
            &lexical,
            1,
            0,
            REFINEMENT,
        )?;
        let wrapped = store.intern(3, vec![root.clone(), a.clone()], None)?;
        let nested = store.intern(3, vec![wrapped.clone(), a], None)?;

        assert!(plan.keeps((0, 1), &root));
        assert!(plan.keeps((0, 1), &wrapped));
        assert!(plan.keeps((0, 1), &nested));
        Ok(())
    }

    #[test]
    fn root_demands_refine_lexical_spans_and_binary_applications()
    -> Result<(), crate::rule::RuleError> {
        let (store, terms, plan) = sentence()?;
        let c = store.intern(2, Vec::new(), None)?;
        let [a, b, root] = terms.as_slice() else {
            panic!("the fixture supplies two leaves and one root")
        };

        assert!(plan.keeps((0, 1), a));
        assert!(plan.keeps((1, 2), b));
        assert!(plan.keeps((0, 2), root));
        assert!(!plan.keeps((0, 1), &c));
        assert!(!plan.keeps((1, 2), &c));
        assert!(plan.permits((0, 2), &[(0, 1), (1, 2)], 0, &[a.clone(), b.clone()]));
        assert!(!plan.permits((0, 2), &[(0, 1), (1, 2)], 0, &[c.clone(), b.clone()]));
        let inputs = [a.clone(), b.clone()];
        let binary = plan.binary_permits(
            (0, 2),
            &[(0, 1), (1, 2)],
            std::slice::from_ref(a),
            std::slice::from_ref(b),
        );
        let mut witnesses = binary.witnesses();
        assert!(witnesses.permits(0, a.serial(), b.serial()));
        assert!(plan.permits_binary_output(&mut witnesses, 0, &inputs, root));
        assert!(!plan.permits_binary_output(&mut witnesses, 0, &inputs, &c));
        Ok(())
    }

    #[test]
    fn a_lexical_refinement_past_its_cap_keeps_the_wider_demands()
    -> Result<(), Box<dyn std::error::Error>> {
        let store = store()?;
        let a = store.intern(0, Vec::new(), None)?;
        let b = store.intern(1, Vec::new(), None)?;
        let c = store.intern(2, Vec::new(), None)?;
        let paired = store.intern(3, vec![a.clone(), b.clone()], None)?;
        let rule = ReachabilityRule {
            arity: 2,
            schemata: vec![(
                vec![node(3, vec![variable("x"), variable("y")]), variable("y")],
                variable("x"),
            )],
            introduces: None,
            consumes: None,
        };
        let lexical = HashMap::from([((0, 1), vec![paired]), ((1, 2), vec![b.clone(), c.clone()])]);
        let plan = |refinement_pairs| {
            ReachabilityPlan::new(
                2,
                std::slice::from_ref(&rule),
                std::slice::from_ref(&a),
                &lexical,
                0,
                0,
                refinement_pairs,
            )
        };
        let refined = plan(REFINEMENT)?;
        let unrefined = plan(1)?;

        assert!(refined.keeps((1, 2), &b));
        assert!(!refined.keeps((1, 2), &c));
        assert!(unrefined.keeps((1, 2), &b));
        assert!(unrefined.keeps((1, 2), &c));
        Ok(())
    }

    #[test]
    fn binary_witnesses_cache_all_matching_rows_independently_of_query_order()
    -> Result<(), crate::rule::RuleError> {
        let store = store()?;
        let a = store.intern(0, Vec::new(), None)?;
        let b = store.intern(1, Vec::new(), None)?;
        let rejected = store.intern(2, Vec::new(), None)?;
        let forward = store.intern(3, vec![a.clone(), b.clone()], None)?;
        let reverse = store.intern(3, vec![b.clone(), a.clone()], None)?;
        let lexical = HashMap::from([((0, 1), vec![a.clone()]), ((1, 2), vec![b.clone()])]);
        let plan = ReachabilityPlan::new(
            2,
            &[two_output_pair_rule()],
            &[forward.clone(), reverse.clone()],
            &lexical,
            0,
            0,
            REFINEMENT,
        )?;
        let permits = plan.binary_permits(
            (0, 2),
            &[(0, 1), (1, 2)],
            std::slice::from_ref(&a),
            std::slice::from_ref(&b),
        );
        let inputs = [a.clone(), b.clone()];

        let mut reverse_first = permits.witnesses();
        assert!(plan.permits_binary_output(&mut reverse_first, 0, &inputs, &reverse));
        assert!(plan.permits_binary_output(&mut reverse_first, 0, &inputs, &forward));
        assert!(!plan.permits_binary_output(&mut reverse_first, 0, &inputs, &rejected));
        assert!(reverse_first.permits(0, a.serial(), b.serial()));
        assert!(!reverse_first.permits(0, rejected.serial(), b.serial()));
        assert!(!reverse_first.permits(1, a.serial(), b.serial()));

        let mut forward_first = permits.witnesses();
        assert!(plan.permits_binary_output(&mut forward_first, 0, &inputs, &forward));
        assert!(plan.permits_binary_output(&mut forward_first, 0, &inputs, &reverse));
        assert!(!plan.permits_binary_output(&mut forward_first, 0, &inputs, &rejected));
        Ok(())
    }
}
