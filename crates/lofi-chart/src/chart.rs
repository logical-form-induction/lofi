//! Dense-id rule drawing for packed charts.

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::num::NonZeroUsize;

use crate::semiring::Semiring;
pub use rayon::ThreadPool;
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;

/// Dense category identifier minted by the Python category table.
pub type CategoryId = u32;

/// Rule identifier in declaration order.
pub type RuleId = u16;

/// Normal-form tag identifier.
pub type TagId = u8;

/// Interned key used by a binary join.
pub type JoinKeyId = u32;

/// Declared limits on categories admitted to one chart program.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CategoryBounds {
    /// Greatest admitted functor-spine arity, or no bound.
    pub max_arity: Option<NonZeroUsize>,
    /// Greatest admitted term-node count, or no bound.
    pub max_size: Option<NonZeroUsize>,
}

impl CategoryBounds {
    /// Return whether the dimensions fall within both declared bounds.
    #[must_use]
    pub fn admits(self, dimensions: CategoryDimensions) -> bool {
        self.max_arity
            .is_none_or(|maximum| dimensions.arity <= maximum.get())
            && self
                .max_size
                .is_none_or(|maximum| dimensions.size <= maximum.get())
    }
}

/// The two independently bounded dimensions of one exact category.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CategoryDimensions {
    /// Number of functor arguments along the declared result path.
    pub arity: usize,
    /// Number of term nodes, including repeated positions.
    pub size: usize,
}

/// One category and normal-form tag in a chart cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ItemId {
    /// The exact category.
    pub category: CategoryId,
    /// The normal-form tag.
    pub tag: TagId,
}

/// One admitted rule conclusion.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Step {
    /// The applied rule.
    pub rule: RuleId,
    /// The exact conclusion category.
    pub output: CategoryId,
    /// The conclusion's normal-form tag.
    pub tag: TagId,
}

/// One exact native drawing request in a width-level batch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StepRequest {
    /// Exact premise categories in rule order.
    pub inputs: Vec<CategoryId>,
    /// Exact premise normal-form tags in rule order.
    pub tags: Vec<TagId>,
    /// Span receiving the conclusion.
    pub span: Span,
    /// Candidate rules in declaration order.
    pub candidates: Vec<RuleId>,
}

/// One exact category tuple with its populated child-tag tuples.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StepBatchRequest {
    /// Exact premise categories in rule order.
    pub inputs: Vec<CategoryId>,
    /// Exact populated child-tag tuples in deterministic order.
    pub tag_rows: Vec<Vec<TagId>>,
    /// Span receiving the conclusions.
    pub span: Span,
    /// Child spans in premise order.
    pub parts: Vec<Span>,
    /// Candidate rules in declaration order.
    pub candidates: Vec<RuleId>,
}

/// One admitted step from a drawing batch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StepBatchEntry {
    /// Position of the request in its submitted batch.
    pub request: usize,
    /// Position of the premise-tag tuple in its request.
    pub row: usize,
    /// One admitted conclusion for that tag tuple.
    pub step: Step,
}

/// One compiled normal-form plan for an exact category/tag tuple.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StepPlan {
    /// Reuse complete steps memoized for the exact premises and child tags.
    Cached(Vec<Step>),
    /// Draw native rules and forward any boundary-dependent rules.
    Draw {
        /// Complete native rules and their output tags, in declaration order.
        native: Vec<(RuleId, TagId)>,
        /// Boundary-dependent rules in declaration order.
        residual: Vec<RuleId>,
    },
}

/// One nonempty child-tag row and its compiled plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedStepRow {
    /// Position in the request's Cartesian child-tag order.
    pub row: usize,
    /// Compiled work for this row.
    pub plan: StepPlan,
}

/// Static facts about one rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuleSpec {
    /// The rule's premise count.
    pub arity: usize,
    /// The rule's spine order.
    pub order: usize,
    /// Whether forward application enumerates every conclusion.
    pub complete: bool,
}

/// One normal-form decision compiled for an exact child-tag tuple.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormalDecision {
    /// Admit the step without inspecting its exact terms.
    Admit(TagId),
    /// Refuse the step without applying its rule.
    Refuse,
    /// Inspect the exact premises and conclusion before admitting the step.
    Premises(TagId),
}

/// Native rule-selection metadata computed once for an exact category.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CategoryMeta {
    /// Unary rules the rule index admits for this exact category.
    pub unary: Vec<RuleId>,
    /// Whether one category can fill an unjoined binary-rule slot.
    pub viable: HashSet<(RuleId, u8)>,
    /// Join keys by `(rule, slot)`.
    pub joins: HashMap<(RuleId, u8), Vec<JoinKeyId>>,
    /// Unit-weight conclusions fixed by `(rule, principal slot, join key)`.
    pub factors: HashMap<(RuleId, u8, JoinKeyId), CategoryId>,
}

impl CategoryMeta {
    /// Construct metadata for a category refused before rule indexing.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            unary: Vec::new(),
            viable: HashSet::new(),
            joins: HashMap::new(),
            factors: HashMap::new(),
        }
    }
}

/// One indexed pair of positions and its candidate rules.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pair {
    /// Position in the left category sequence.
    pub left: usize,
    /// Position in the right category sequence.
    pub right: usize,
    /// Candidate rules in declaration order.
    pub candidates: Vec<RuleId>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AxisRef {
    One(usize),
    Many { start: u32, len: u32 },
}

impl AxisRef {
    fn first(self, arena: &[usize]) -> Option<usize> {
        match self {
            Self::One(position) => Some(position),
            Self::Many { start, .. } => arena.get(usize::try_from(start).ok()?).copied(),
        }
    }

    fn as_slice<'a>(&'a self, arena: &'a [usize]) -> Result<&'a [usize], ChartError> {
        match self {
            Self::One(position) => Ok(std::slice::from_ref(position)),
            Self::Many { start, len } => {
                let start = usize::try_from(*start).unwrap_or(usize::MAX);
                let len = usize::try_from(*len).unwrap_or(usize::MAX);
                let end = start
                    .checked_add(len)
                    .ok_or(ChartError::InvalidFactoredAxis {
                        start,
                        len,
                        items: arena.len(),
                    })?;
                arena
                    .get(start..end)
                    .ok_or(ChartError::InvalidFactoredAxis {
                        start,
                        len,
                        items: arena.len(),
                    })
            }
        }
    }
}

/// One rule/output relation factored across exact premise axes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FactoredPairProduct {
    /// Rule shared by every represented pair.
    pub rule: RuleId,
    /// Exact conclusion shared by every represented pair.
    pub output: CategoryId,
    axes: [AxisRef; 2],
}

/// Exact residual pairs and factored binary-rule relations for one decomposition.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PairBatch {
    /// Pairs whose conclusions require exact rule application.
    pub pairs: Vec<Pair>,
    /// Complete unit-weight relations grouped by exact conclusion.
    pub products: Vec<FactoredPairProduct>,
    axis_items: Vec<usize>,
}

impl PairBatch {
    /// Construct a batch from residual exact pairs.
    #[must_use]
    pub fn from_pairs(pairs: Vec<Pair>) -> Self {
        Self {
            pairs,
            ..Self::default()
        }
    }

    fn append_axis(&mut self, mut positions: Vec<usize>) -> Result<AxisRef, ChartError> {
        positions.sort_unstable();
        positions.dedup();
        match positions.as_slice() {
            [] => Err(ChartError::EmptyFactoredAxis),
            [position] => Ok(AxisRef::One(*position)),
            _ => {
                let start = u32::try_from(self.axis_items.len()).map_err(|_| {
                    ChartError::FactoredAxisOverflow {
                        items: self.axis_items.len(),
                    }
                })?;
                let len = u32::try_from(positions.len()).map_err(|_| {
                    ChartError::FactoredAxisOverflow {
                        items: positions.len(),
                    }
                })?;
                self.axis_items.extend(positions);
                Ok(AxisRef::Many { start, len })
            }
        }
    }

    /// Append one nonempty exact rule/output rectangle.
    ///
    /// # Errors
    ///
    /// Returns an error when either axis is empty or the shared arena exceeds
    /// its compact offset width.
    pub fn push_product(
        &mut self,
        rule: RuleId,
        output: CategoryId,
        left: Vec<usize>,
        right: Vec<usize>,
    ) -> Result<(), ChartError> {
        if left.is_empty() || right.is_empty() {
            return Err(ChartError::EmptyFactoredAxis);
        }
        let left = self.append_axis(left)?;
        let right = self.append_axis(right)?;
        self.products.push(FactoredPairProduct {
            rule,
            output,
            axes: [left, right],
        });
        Ok(())
    }

    fn axes_of<'a>(
        &'a self,
        product: &'a FactoredPairProduct,
    ) -> Result<[&'a [usize]; 2], ChartError> {
        Ok([
            product.axes[0].as_slice(&self.axis_items)?,
            product.axes[1].as_slice(&self.axis_items)?,
        ])
    }

    /// Borrow the exact left and right position axes of one indexed product.
    ///
    /// # Errors
    ///
    /// Returns an error when a compact range does not belong to this batch.
    pub fn product_axes_at(&self, product: usize) -> Result<[&[usize]; 2], ChartError> {
        let product = self
            .products
            .get(product)
            .ok_or(ChartError::InvalidFactoredProduct {
                product,
                products: self.products.len(),
            })?;
        self.axes_of(product)
    }

    /// Append products from another batch while remapping its arena offsets.
    ///
    /// # Errors
    ///
    /// Returns an error when either arena is malformed or their combined size
    /// exceeds the compact offset width.
    pub fn extend_products(&mut self, mut other: Self) -> Result<(), ChartError> {
        for product in &other.products {
            other.axes_of(product)?;
        }
        let base =
            u32::try_from(self.axis_items.len()).map_err(|_| ChartError::FactoredAxisOverflow {
                items: self.axis_items.len(),
            })?;
        let combined = self
            .axis_items
            .len()
            .checked_add(other.axis_items.len())
            .ok_or(ChartError::FactoredAxisOverflow { items: usize::MAX })?;
        u32::try_from(combined)
            .map_err(|_| ChartError::FactoredAxisOverflow { items: combined })?;
        for product in &mut other.products {
            for axis in &mut product.axes {
                if let AxisRef::Many { start, .. } = axis {
                    *start = start
                        .checked_add(base)
                        .ok_or(ChartError::FactoredAxisOverflow { items: combined })?;
                }
            }
        }
        self.axis_items.extend(other.axis_items);
        self.products.extend(other.products);
        Ok(())
    }

    /// Sort products by their first positions, then rule and output.
    pub fn sort_products(&mut self) {
        let arena = &self.axis_items;
        self.products.sort_unstable_by_key(|product| {
            (
                product.axes[0].first(arena),
                product.axes[1].first(arena),
                product.rule,
                product.output,
            )
        });
    }

    /// Return the number of non-singleton position entries in the shared arena.
    #[must_use]
    pub const fn axis_items(&self) -> usize {
        self.axis_items.len()
    }
}

/// One binary decomposition and its populated exact category axes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PairBatchRequest {
    /// Span receiving the conclusion.
    pub span: Span,
    /// Exact left and right child spans.
    pub parts: [Span; 2],
    /// Left categories in cell arrival order.
    pub left: Vec<CategoryId>,
    /// Right categories in cell arrival order.
    pub right: Vec<CategoryId>,
}

fn factored_output(
    left: &CategoryMeta,
    right: &CategoryMeta,
    rule: RuleId,
) -> Option<(CategoryId, u8)> {
    left.factors
        .iter()
        .find_map(|((held, slot, key), output)| {
            (*held == rule
                && *slot == 0
                && right
                    .joins
                    .get(&(rule, 1))
                    .is_some_and(|keys| keys.contains(key)))
            .then_some((*output, 0))
        })
        .or_else(|| {
            right
                .factors
                .iter()
                .find_map(|((held, slot, key), output)| {
                    (*held == rule
                        && *slot == 1
                        && left
                            .joins
                            .get(&(rule, 0))
                            .is_some_and(|keys| keys.contains(key)))
                    .then_some((*output, 1))
                })
        })
}

type SelectedPairs = HashMap<(usize, usize), Vec<RuleId>>;
type FactoredPairs = HashMap<(RuleId, CategoryId, u8, usize), Vec<usize>>;

fn record_pair(
    selected: &mut SelectedPairs,
    factored: &mut FactoredPairs,
    rule: RuleId,
    left: usize,
    right: usize,
    factor: Option<(CategoryId, u8)>,
) {
    if let Some((output, principal)) = factor {
        let (fixed, partner) = if principal == 0 {
            (left, right)
        } else {
            (right, left)
        };
        factored
            .entry((rule, output, principal, fixed))
            .or_default()
            .push(partner);
    } else {
        selected.entry((left, right)).or_default().push(rule);
    }
}

fn finish_pair_batch(
    selected: SelectedPairs,
    factored: FactoredPairs,
) -> Result<PairBatch, ChartError> {
    let mut pairs = selected
        .into_iter()
        .map(|((left, right), mut candidates)| {
            candidates.sort_unstable();
            candidates.dedup();
            Pair {
                left,
                right,
                candidates,
            }
        })
        .collect::<Vec<_>>();
    pairs.sort_unstable_by_key(|pair| (pair.left, pair.right));
    let mut batch = PairBatch {
        pairs,
        ..PairBatch::default()
    };
    for ((rule, output, principal, fixed), mut partners) in factored {
        partners.sort_unstable();
        partners.dedup();
        let (left, right) = if principal == 0 {
            (vec![fixed], partners)
        } else {
            (partners, vec![fixed])
        };
        batch.push_product(rule, output, left, right)?;
    }
    batch.sort_products();
    Ok(batch)
}

/// One batch of unresolved rule applications at the resolver boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MissBatch {
    inputs: Vec<CategoryId>,
    tags: Vec<TagId>,
    candidates: Vec<RuleId>,
    span: Span,
    permitted: HashSet<RuleId>,
    rules: usize,
    tag_count: usize,
    results: Vec<Step>,
}

impl MissBatch {
    /// Construct a resolver batch from a validated drawing request.
    #[must_use]
    pub fn new(
        program: &ChartProgram,
        inputs: Vec<CategoryId>,
        tags: Vec<TagId>,
        candidates: Vec<RuleId>,
        span: Span,
    ) -> Self {
        let permitted = candidates.iter().copied().collect();
        Self {
            inputs,
            tags,
            candidates,
            span,
            permitted,
            rules: program.rule_count(),
            tag_count: program.tag_count(),
            results: Vec::new(),
        }
    }

    /// Return exact premise categories in rule order.
    #[must_use]
    pub fn inputs(&self) -> &[CategoryId] {
        &self.inputs
    }

    /// Return premise normal-form tags in rule order.
    #[must_use]
    pub fn tags(&self) -> &[TagId] {
        &self.tags
    }

    /// Return candidate rules in declaration order.
    #[must_use]
    pub fn candidates(&self) -> &[RuleId] {
        &self.candidates
    }

    /// Return the half-open conclusion span.
    #[must_use]
    pub const fn span(&self) -> Span {
        self.span
    }

    /// Return admitted steps in resolver insertion order.
    #[must_use]
    pub fn results(&self) -> &[Step] {
        &self.results
    }

    /// Add one resolved and admitted step.
    ///
    /// # Errors
    ///
    /// Returns a boundary error when the rule is unknown or was not a candidate, or
    /// when the tag is outside the declared alphabet.
    pub fn push(&mut self, rule: RuleId, output: CategoryId, tag: TagId) -> Result<(), ChartError> {
        if usize::from(rule) >= self.rules {
            return Err(ChartError::UnknownRule {
                rule,
                rules: self.rules,
            });
        }
        if !self.permitted.contains(&rule) {
            return Err(ChartError::RuleNotCandidate { rule });
        }
        if usize::from(tag) >= self.tag_count {
            return Err(ChartError::UnknownTag {
                tag,
                tags: self.tag_count,
            });
        }
        self.results.push(Step { rule, output, tag });
        Ok(())
    }
}

/// The ordered result of closing unary rules over one chart cell.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnaryClosure {
    /// Admitted edges in drawing order.
    pub edges: Vec<(ItemId, Step)>,
    /// Concluded items after every item they depend on.
    pub concluded: Vec<ItemId>,
    /// Whether a cycle or the depth bound dropped a step.
    pub truncated: bool,
}

/// Errors at the dense-id drawing boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ChartError {
    /// A rule identifier is outside the program.
    UnknownRule {
        /// The supplied identifier.
        rule: RuleId,
        /// The program's rule count.
        rules: usize,
    },
    /// A rule identifier was not admitted for the current resolver batch.
    RuleNotCandidate {
        /// The supplied identifier.
        rule: RuleId,
    },
    /// A category identifier has not been registered.
    UnknownCategory {
        /// The supplied identifier.
        category: CategoryId,
    },
    /// A tag identifier is outside the normal-form alphabet.
    UnknownTag {
        /// The supplied identifier.
        tag: TagId,
        /// The program's tag count.
        tags: usize,
    },
    /// A rule is asked to consume the wrong number of premises.
    WrongArity {
        /// The supplied rule.
        rule: RuleId,
        /// The rule's declared arity.
        expected: usize,
        /// The supplied premise count.
        actual: usize,
    },
    /// A category is registered other than at the next dense identifier.
    NonDenseCategory {
        /// The next identifier the table owns.
        expected: CategoryId,
        /// The supplied identifier.
        supplied: CategoryId,
    },
    /// A rule has no compiled decision for the supplied child tags.
    UnknownDecision {
        /// The supplied rule.
        rule: RuleId,
        /// The supplied child tags.
        tags: Vec<TagId>,
    },
    /// A factored relation supplied an empty premise axis.
    EmptyFactoredAxis,
    /// A factored relation exceeded the compact arena offset width.
    FactoredAxisOverflow {
        /// Number of entries that could not be addressed.
        items: usize,
    },
    /// A factored relation refers outside its owning arena.
    InvalidFactoredAxis {
        /// First requested arena entry.
        start: usize,
        /// Number of requested entries.
        len: usize,
        /// Number of entries owned by the batch.
        items: usize,
    },
    /// A product position lies outside its pair batch.
    InvalidFactoredProduct {
        /// Supplied product position.
        product: usize,
        /// Number of products owned by the batch.
        products: usize,
    },
}

impl fmt::Display for ChartError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownRule { rule, rules } => {
                write!(
                    formatter,
                    "rule id {rule} is outside a program of {rules} rules"
                )
            }
            Self::RuleNotCandidate { rule } => {
                write!(
                    formatter,
                    "rule id {rule} was not a candidate for this miss"
                )
            }
            Self::UnknownCategory { category } => {
                write!(formatter, "category id {category} was never registered")
            }
            Self::UnknownTag { tag, tags } => {
                write!(
                    formatter,
                    "tag id {tag} is outside an alphabet of {tags} tags"
                )
            }
            Self::WrongArity {
                rule,
                expected,
                actual,
            } => write!(
                formatter,
                "rule id {rule} has arity {expected}, not the supplied {actual}"
            ),
            Self::NonDenseCategory { expected, supplied } => write!(
                formatter,
                "category id {supplied} is not the next dense id {expected}"
            ),
            Self::UnknownDecision { rule, tags } => {
                write!(
                    formatter,
                    "rule id {rule} has no decision for child tags {tags:?}"
                )
            }
            Self::EmptyFactoredAxis => write!(formatter, "a factored premise axis is empty"),
            Self::FactoredAxisOverflow { items } => write!(
                formatter,
                "a factored premise arena of {items} items exceeds its offset width"
            ),
            Self::InvalidFactoredAxis { start, len, items } => write!(
                formatter,
                "factored premise range {start}..{} is outside an arena of {items} items",
                start.saturating_add(*len)
            ),
            Self::InvalidFactoredProduct { product, products } => write!(
                formatter,
                "factored product {product} is outside a batch of {products} products"
            ),
        }
    }
}

impl std::error::Error for ChartError {}

fn tag_row_index(tags: usize, row: &[TagId]) -> Option<usize> {
    row.iter().try_fold(0_usize, |index, tag| {
        index.checked_mul(tags)?.checked_add(usize::from(*tag))
    })
}

#[derive(Clone, Debug)]
enum RuleDecisions {
    Dense(Vec<NormalDecision>),
    Sparse(HashMap<Vec<TagId>, NormalDecision>),
}

impl RuleDecisions {
    fn new(tags: usize, arity: usize, rows: HashMap<Vec<TagId>, NormalDecision>) -> Self {
        let expected = u32::try_from(arity)
            .ok()
            .and_then(|power| tags.checked_pow(power));
        let Some(count) = expected.filter(|count| {
            *count == rows.len()
                && rows
                    .keys()
                    .all(|row| row.len() == arity && row.iter().all(|tag| usize::from(*tag) < tags))
        }) else {
            return Self::Sparse(rows);
        };
        let mut dense = vec![NormalDecision::Refuse; count];
        for (row, decision) in &rows {
            if let Some(position) = tag_row_index(tags, row) {
                dense[position] = *decision;
            }
        }
        Self::Dense(dense)
    }

    fn get(&self, tags: usize, row: &[TagId]) -> Option<NormalDecision> {
        match self {
            Self::Dense(decisions) => tag_row_index(tags, row)
                .and_then(|position| decisions.get(position))
                .copied(),
            Self::Sparse(decisions) => decisions.get(row).copied(),
        }
    }
}

type CandidateStepMemo = HashMap<Vec<RuleId>, Vec<Step>>;
type TaggedStepMemo = HashMap<Vec<TagId>, CandidateStepMemo>;
type StepMemo = HashMap<Vec<CategoryId>, TaggedStepMemo>;

/// Dense-id drawing state shared by forest and inside sweeps.
#[derive(Clone, Debug)]
pub struct ChartProgram {
    rules: Vec<RuleSpec>,
    by_arity: HashMap<usize, Vec<RuleId>>,
    unjoined: Vec<RuleId>,
    joined: Vec<RuleId>,
    categories: Vec<CategoryMeta>,
    outputs: Vec<HashMap<Vec<CategoryId>, Vec<CategoryId>>>,
    memo: StepMemo,
    decisions: Vec<RuleDecisions>,
    tags: usize,
    bounds: CategoryBounds,
    refused_categories: usize,
}

impl ChartProgram {
    /// Construct a drawing program from validated rule metadata.
    #[must_use]
    pub fn new(
        rules: Vec<RuleSpec>,
        tags: usize,
        unjoined: Vec<RuleId>,
        joined: Vec<RuleId>,
        decisions: HashMap<(RuleId, Vec<TagId>), NormalDecision>,
        bounds: CategoryBounds,
    ) -> Self {
        let mut by_arity: HashMap<usize, Vec<RuleId>> = HashMap::new();

        for (at, rule) in rules.iter().enumerate() {
            if let Ok(rule_id) = RuleId::try_from(at) {
                by_arity.entry(rule.arity).or_default().push(rule_id);
            }
        }

        let outputs = vec![HashMap::new(); rules.len()];
        let mut rows_by_rule = vec![HashMap::new(); rules.len()];
        for ((rule, children), decision) in decisions {
            let position = usize::from(rule);
            if let Some(rows) = rows_by_rule.get_mut(position) {
                rows.insert(children, decision);
            }
        }
        let compiled_decisions = rules
            .iter()
            .zip(rows_by_rule)
            .map(|(rule, rows)| RuleDecisions::new(tags, rule.arity, rows))
            .collect();

        Self {
            rules,
            by_arity,
            unjoined,
            joined,
            categories: Vec::new(),
            outputs,
            memo: HashMap::new(),
            decisions: compiled_decisions,
            tags,
            bounds,
            refused_categories: 0,
        }
    }

    /// Return the program's rule count.
    #[must_use]
    pub const fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Return the declared normal-form tag count.
    #[must_use]
    pub const fn tag_count(&self) -> usize {
        self.tags
    }

    /// Return how many distinct generated categories the declared bounds refused.
    #[must_use]
    pub const fn refused_categories(&self) -> usize {
        self.refused_categories
    }

    /// Report whether the declared category bounds admit these dimensions.
    #[must_use]
    pub fn admits_category(&self, dimensions: CategoryDimensions) -> bool {
        self.bounds.admits(dimensions)
    }

    /// Return the category bounds owned by this drawing program.
    #[must_use]
    pub const fn category_bounds(&self) -> CategoryBounds {
        self.bounds
    }

    /// Return one rule's metadata.
    ///
    /// # Errors
    ///
    /// Returns [`ChartError::UnknownRule`] when the identifier is outside the program.
    pub fn rule(&self, rule: RuleId) -> Result<RuleSpec, ChartError> {
        self.rules
            .get(usize::from(rule))
            .copied()
            .ok_or(ChartError::UnknownRule {
                rule,
                rules: self.rules.len(),
            })
    }

    /// Return candidate rules of one arity in declaration order.
    #[must_use]
    pub fn rules_of_arity(&self, arity: usize) -> &[RuleId] {
        self.by_arity.get(&arity).map_or(&[], Vec::as_slice)
    }

    /// Return one rule's compiled normal-form decision for exact child tags.
    ///
    /// # Errors
    ///
    /// Returns a boundary error when the rule, a tag, or the decision row is unknown.
    pub fn decision(&self, rule: RuleId, tags: &[TagId]) -> Result<NormalDecision, ChartError> {
        let spec = self.rule(rule)?;
        if spec.arity != tags.len() {
            return Err(ChartError::WrongArity {
                rule,
                expected: spec.arity,
                actual: tags.len(),
            });
        }
        for tag in tags {
            if usize::from(*tag) >= self.tags {
                return Err(ChartError::UnknownTag {
                    tag: *tag,
                    tags: self.tags,
                });
            }
        }
        self.compiled_decision(rule, tags)
            .ok_or_else(|| ChartError::UnknownDecision {
                rule,
                tags: tags.to_vec(),
            })
    }

    fn compiled_decision(&self, rule: RuleId, tags: &[TagId]) -> Option<NormalDecision> {
        self.decisions
            .get(usize::from(rule))
            .and_then(|table| table.get(self.tags, tags))
    }

    /// Return potentially admitted tag rows present on all premise axes.
    ///
    /// Rows follow the Cartesian order of the supplied axes. A row is returned
    /// exactly once when at least one candidate does not refuse it statically.
    ///
    /// # Errors
    ///
    /// Returns a boundary error when a rule, arity, or tag is unknown.
    pub fn admissible_tag_rows(
        &self,
        candidates: &[RuleId],
        rows: &[Vec<TagId>],
    ) -> Result<Vec<(usize, Vec<TagId>)>, ChartError> {
        for tags in rows {
            for tag in tags {
                if usize::from(*tag) >= self.tags {
                    return Err(ChartError::UnknownTag {
                        tag: *tag,
                        tags: self.tags,
                    });
                }
            }
        }

        for rule in candidates {
            let spec = self.rule(*rule)?;
            if rows.iter().any(|tags| tags.len() != spec.arity) {
                return Err(ChartError::WrongArity {
                    rule: *rule,
                    expected: spec.arity,
                    actual: rows.first().map_or(0, Vec::len),
                });
            }
        }
        let mut admitted = Vec::new();
        for (row, tags) in rows.iter().enumerate() {
            for rule in candidates {
                if self.decision(*rule, tags)? != NormalDecision::Refuse {
                    admitted.push((row, tags.clone()));
                    break;
                }
            }
        }
        Ok(admitted)
    }

    /// Compile one exact tagged drawing request.
    ///
    /// # Errors
    ///
    /// Returns a boundary error when a request names an unknown rule, category, or tag.
    pub fn step_plan(
        &self,
        inputs: &[CategoryId],
        tags: &[TagId],
        candidates: &[RuleId],
        normalizes: bool,
    ) -> Result<StepPlan, ChartError> {
        for category in inputs {
            self.category(*category)?;
        }
        for tag in tags {
            if usize::from(*tag) >= self.tags {
                return Err(ChartError::UnknownTag {
                    tag: *tag,
                    tags: self.tags,
                });
            }
        }
        for rule in candidates {
            let spec = self.rule(*rule)?;
            if spec.arity != inputs.len() {
                return Err(ChartError::WrongArity {
                    rule: *rule,
                    expected: spec.arity,
                    actual: inputs.len(),
                });
            }
        }
        self.compiled_step_plan(inputs, tags, candidates, normalizes)
    }

    fn compiled_step_plan(
        &self,
        inputs: &[CategoryId],
        tags: &[TagId],
        candidates: &[RuleId],
        normalizes: bool,
    ) -> Result<StepPlan, ChartError> {
        if let Some(cached) = self.cached(inputs, tags, candidates) {
            return Ok(StepPlan::Cached(cached.to_vec()));
        }

        let mut native = Vec::new();
        let mut residual = Vec::new();
        for rule in candidates {
            let spec = self.rule(*rule)?;
            let decision =
                self.compiled_decision(*rule, tags)
                    .ok_or_else(|| ChartError::UnknownDecision {
                        rule: *rule,
                        tags: tags.to_vec(),
                    })?;
            match decision {
                NormalDecision::Refuse => {}
                NormalDecision::Admit(tag) if spec.complete && !normalizes => {
                    native.push((*rule, tag));
                }
                NormalDecision::Admit(_) | NormalDecision::Premises(_) => {
                    residual.push(*rule);
                }
            }
        }
        Ok(StepPlan::Draw { native, residual })
    }

    /// Compile independent compressed drawing requests in submission order.
    ///
    /// # Errors
    ///
    /// Returns a boundary error when a request names an unknown rule, category, or tag.
    pub fn batch_step_plans(
        &self,
        requests: &[StepBatchRequest],
        normalizes: bool,
        workers: Option<&ThreadPool>,
    ) -> Result<Vec<Vec<PlannedStepRow>>, ChartError> {
        let compile = |request: &StepBatchRequest| {
            let candidates = request
                .candidates
                .iter()
                .copied()
                .map(|rule| {
                    let spec = self.rule(rule)?;
                    if spec.arity != request.inputs.len() {
                        return Err(ChartError::WrongArity {
                            rule,
                            expected: spec.arity,
                            actual: request.inputs.len(),
                        });
                    }
                    Ok(if spec.complete && !normalizes {
                        self.cached_outputs(rule, &request.inputs)
                            .is_none_or(|outputs| !outputs.is_empty())
                            .then_some(rule)
                    } else {
                        Some(rule)
                    })
                })
                .collect::<Result<Vec<_>, ChartError>>()?
                .into_iter()
                .flatten()
                .collect::<Vec<_>>();
            if candidates.is_empty() {
                return Ok(Vec::new());
            }
            for category in &request.inputs {
                self.category(*category)?;
            }
            for tags in &request.tag_rows {
                if tags.len() != request.inputs.len() {
                    return Err(ChartError::WrongArity {
                        rule: candidates[0],
                        expected: request.inputs.len(),
                        actual: tags.len(),
                    });
                }
                for tag in tags {
                    if usize::from(*tag) >= self.tags {
                        return Err(ChartError::UnknownTag {
                            tag: *tag,
                            tags: self.tags,
                        });
                    }
                }
            }
            let mut plans = Vec::new();
            for (row, tags) in request.tag_rows.iter().enumerate() {
                let plan =
                    self.compiled_step_plan(&request.inputs, tags, &candidates, normalizes)?;
                let keeps = match &plan {
                    StepPlan::Cached(steps) => !steps.is_empty(),
                    StepPlan::Draw {
                        native, residual, ..
                    } => !native.is_empty() || !residual.is_empty(),
                };
                if keeps {
                    plans.push(PlannedStepRow { row, plan });
                }
            }
            Ok(plans)
        };

        workers.map_or_else(
            || requests.iter().map(compile).collect(),
            |pool| pool.install(|| requests.par_iter().map(compile).collect()),
        )
    }

    /// Register the next exact category.
    ///
    /// # Errors
    ///
    /// Returns a boundary error when the identifier is not dense or metadata names an
    /// unknown rule.
    pub fn register_category(
        &mut self,
        category: CategoryId,
        meta: CategoryMeta,
        dimensions: CategoryDimensions,
    ) -> Result<bool, ChartError> {
        if !self.bounds.admits(dimensions) {
            self.refused_categories += 1;
            return Ok(false);
        }
        let expected = CategoryId::try_from(self.categories.len()).map_err(|_| {
            ChartError::NonDenseCategory {
                expected: CategoryId::MAX,
                supplied: category,
            }
        })?;

        if category != expected {
            return Err(ChartError::NonDenseCategory {
                expected,
                supplied: category,
            });
        }

        for rule in &meta.unary {
            self.rule(*rule)?;
        }

        for (rule, slot) in &meta.viable {
            self.rule(*rule)?;
            if *slot > 1 {
                return Err(ChartError::WrongArity {
                    rule: *rule,
                    expected: 2,
                    actual: usize::from(*slot) + 1,
                });
            }
        }

        for rule in meta.joins.keys().map(|(rule, _)| rule) {
            self.rule(*rule)?;
        }

        for ((rule, slot, _), output) in &meta.factors {
            self.rule(*rule)?;
            if *slot > 1 {
                return Err(ChartError::WrongArity {
                    rule: *rule,
                    expected: 2,
                    actual: usize::from(*slot) + 1,
                });
            }
            self.category(*output)?;
        }

        self.categories.push(meta);
        Ok(true)
    }

    /// Register one conclusion fixed by a principal category and join key.
    ///
    /// # Errors
    ///
    /// Returns a boundary error when a category, rule, or slot is invalid.
    pub fn register_factor(
        &mut self,
        category: CategoryId,
        rule: RuleId,
        slot: u8,
        key: JoinKeyId,
        output: CategoryId,
    ) -> Result<(), ChartError> {
        self.rule(rule)?;
        if slot > 1 {
            return Err(ChartError::WrongArity {
                rule,
                expected: 2,
                actual: usize::from(slot) + 1,
            });
        }
        self.category(output)?;
        self.categories
            .get_mut(usize::try_from(category).unwrap_or(usize::MAX))
            .ok_or(ChartError::UnknownCategory { category })?
            .factors
            .insert((rule, slot, key), output);
        Ok(())
    }

    /// Return a registered category's metadata.
    ///
    /// # Errors
    ///
    /// Returns [`ChartError::UnknownCategory`] when the identifier is not registered.
    pub fn category(&self, category: CategoryId) -> Result<&CategoryMeta, ChartError> {
        self.categories
            .get(usize::try_from(category).unwrap_or(usize::MAX))
            .ok_or(ChartError::UnknownCategory { category })
    }

    /// Enumerate binary category pairs worth drawing, in Cartesian order.
    ///
    /// # Errors
    ///
    /// Returns [`ChartError::UnknownCategory`] when an input identifier is not registered.
    pub fn pairs(
        &self,
        left: &[CategoryId],
        right: &[CategoryId],
    ) -> Result<Vec<Pair>, ChartError> {
        Ok(self
            .select_pairs(left, right, false, &HashSet::new(), &mut |_, _, _, _| true)?
            .pairs)
    }

    fn select_pairs<F>(
        &self,
        left: &[CategoryId],
        right: &[CategoryId],
        factor_units: bool,
        excluded: &HashSet<RuleId>,
        permits: &mut F,
    ) -> Result<PairBatch, ChartError>
    where
        F: FnMut(RuleId, CategoryId, CategoryId, Option<CategoryId>) -> bool,
    {
        let left_meta = left
            .iter()
            .map(|category| self.category(*category))
            .collect::<Result<Vec<_>, _>>()?;
        let right_meta = right
            .iter()
            .map(|category| self.category(*category))
            .collect::<Result<Vec<_>, _>>()?;
        let mut selected = SelectedPairs::new();
        let mut factored = FactoredPairs::new();

        for rule in &self.unjoined {
            if excluded.contains(rule) {
                continue;
            }
            let left_positions = left_meta
                .iter()
                .enumerate()
                .filter(|(_, meta)| meta.viable.contains(&(*rule, 0)))
                .map(|(position, _)| position)
                .collect::<Vec<_>>();
            let right_positions = right_meta
                .iter()
                .enumerate()
                .filter(|(_, meta)| meta.viable.contains(&(*rule, 1)))
                .map(|(position, _)| position)
                .collect::<Vec<_>>();

            for left_at in &left_positions {
                for right_at in &right_positions {
                    let factor = factor_units
                        .then(|| factored_output(left_meta[*left_at], right_meta[*right_at], *rule))
                        .flatten();
                    if !permits(
                        *rule,
                        left[*left_at],
                        right[*right_at],
                        factor.map(|(output, _)| output),
                    ) {
                        continue;
                    }
                    record_pair(
                        &mut selected,
                        &mut factored,
                        *rule,
                        *left_at,
                        *right_at,
                        factor,
                    );
                }
            }
        }

        let mut right_joins: HashMap<(RuleId, JoinKeyId), Vec<usize>> = HashMap::new();
        for (position, meta) in right_meta.iter().enumerate() {
            for ((rule, slot), keys) in &meta.joins {
                if *slot == 1 && self.joined.contains(rule) && !excluded.contains(rule) {
                    for key in keys {
                        right_joins.entry((*rule, *key)).or_default().push(position);
                    }
                }
            }
        }
        for (left_at, meta) in left_meta.iter().enumerate() {
            for ((rule, slot), keys) in &meta.joins {
                if *slot == 0 && self.joined.contains(rule) && !excluded.contains(rule) {
                    for key in keys {
                        for right_at in right_joins.get(&(*rule, *key)).into_iter().flatten() {
                            let factor = factor_units
                                .then(|| {
                                    factored_output(
                                        left_meta[left_at],
                                        right_meta[*right_at],
                                        *rule,
                                    )
                                })
                                .flatten();
                            if !permits(
                                *rule,
                                left[left_at],
                                right[*right_at],
                                factor.map(|(output, _)| output),
                            ) {
                                continue;
                            }
                            record_pair(
                                &mut selected,
                                &mut factored,
                                *rule,
                                left_at,
                                *right_at,
                                factor,
                            );
                        }
                    }
                }
            }
        }

        finish_pair_batch(selected, factored)
    }

    /// Enumerate a binary category-pair batch while omitting handled rules.
    ///
    /// The permit is consulted before a residual pair or factored partner is
    /// materialized.
    ///
    /// # Errors
    ///
    /// Returns [`ChartError::UnknownCategory`] when an input identifier is not
    /// registered.
    pub fn pair_batch_where_excluding<F>(
        &self,
        left: &[CategoryId],
        right: &[CategoryId],
        factor_units: bool,
        excluded: &HashSet<RuleId>,
        mut permits: F,
    ) -> Result<PairBatch, ChartError>
    where
        F: FnMut(RuleId, CategoryId, CategoryId, Option<CategoryId>) -> bool,
    {
        self.select_pairs(left, right, factor_units, excluded, &mut permits)
    }

    /// Enumerate independent binary category-pair batches in submission order.
    ///
    /// # Errors
    ///
    /// Returns [`ChartError::UnknownCategory`] when an input identifier is not
    /// registered.
    pub fn pair_batches(
        &self,
        batches: &[(Vec<CategoryId>, Vec<CategoryId>)],
        workers: Option<&ThreadPool>,
        factor_units: bool,
    ) -> Result<Vec<PairBatch>, ChartError> {
        let select = |(left, right): &(Vec<CategoryId>, Vec<CategoryId>)| {
            self.select_pairs(
                left,
                right,
                factor_units,
                &HashSet::new(),
                &mut |_, _, _, _| true,
            )
        };

        workers.map_or_else(
            || batches.iter().map(select).collect(),
            |pool| pool.install(|| batches.par_iter().map(select).collect()),
        )
    }

    /// Return indexed unary candidates for one category.
    ///
    /// # Errors
    ///
    /// Returns [`ChartError::UnknownCategory`] when the identifier is not registered.
    pub fn unary_candidates(&self, category: CategoryId) -> Result<&[RuleId], ChartError> {
        let meta = self.category(category)?;
        Ok(&meta.unary)
    }

    /// Return memoized steps for a premise tuple, its tags and candidate rules.
    #[must_use]
    pub fn cached(
        &self,
        inputs: &[CategoryId],
        tags: &[TagId],
        candidates: &[RuleId],
    ) -> Option<&[Step]> {
        self.memo
            .get(inputs)
            .and_then(|row| row.get(tags))
            .and_then(|row| row.get(candidates))
            .map(Vec::as_slice)
    }

    /// Return memoized normalized outputs of one complete rule application.
    #[must_use]
    pub fn cached_outputs(&self, rule: RuleId, inputs: &[CategoryId]) -> Option<&[CategoryId]> {
        self.outputs
            .get(usize::from(rule))
            .and_then(|row| row.get(inputs))
            .map(Vec::as_slice)
    }

    /// Memoize normalized outputs of one complete rule application.
    ///
    /// # Errors
    ///
    /// Returns a boundary error when the rule, its arity, or a category is invalid.
    pub fn remember_outputs(
        &mut self,
        rule: RuleId,
        inputs: Vec<CategoryId>,
        outputs: Vec<CategoryId>,
    ) -> Result<(), ChartError> {
        let spec = self.rule(rule)?;

        if spec.arity != inputs.len() {
            return Err(ChartError::WrongArity {
                rule,
                expected: spec.arity,
                actual: inputs.len(),
            });
        }

        for category in inputs.iter().chain(&outputs) {
            self.category(*category)?;
        }

        self.outputs[usize::from(rule)].insert(inputs, outputs);
        Ok(())
    }

    /// Validate steps returned by the Python resolver.
    ///
    /// # Errors
    ///
    /// Returns a boundary error when any rule, category, tag, or arity is invalid.
    pub fn validate_steps(
        &self,
        inputs: &[CategoryId],
        candidates: &[RuleId],
        steps: &[Step],
    ) -> Result<(), ChartError> {
        for category in inputs {
            self.category(*category)?;
        }

        let permitted: HashSet<RuleId> = candidates.iter().copied().collect();

        for step in steps {
            let spec = self.rule(step.rule)?;

            if spec.arity != inputs.len() || !permitted.contains(&step.rule) {
                return Err(ChartError::WrongArity {
                    rule: step.rule,
                    expected: spec.arity,
                    actual: inputs.len(),
                });
            }

            self.category(step.output)?;

            if usize::from(step.tag) >= self.tags {
                return Err(ChartError::UnknownTag {
                    tag: step.tag,
                    tags: self.tags,
                });
            }
        }

        Ok(())
    }

    /// Memoize complete steps under exact premises, tags and candidate rules.
    pub fn remember(
        &mut self,
        inputs: Vec<CategoryId>,
        tags: Vec<TagId>,
        candidates: Vec<RuleId>,
        steps: Vec<Step>,
    ) {
        self.memo
            .entry(inputs)
            .or_default()
            .entry(tags)
            .or_default()
            .insert(candidates, steps);
    }

    /// Return whether every candidate can be memoized across spans.
    ///
    /// # Errors
    ///
    /// Returns [`ChartError::UnknownRule`] when a candidate identifier is invalid.
    pub fn candidates_are_complete(&self, candidates: &[RuleId]) -> Result<bool, ChartError> {
        candidates.iter().try_fold(true, |complete, rule| {
            Ok(complete && self.rule(*rule)?.complete)
        })
    }
}

/// A half-open token span.
pub type Span = (usize, usize);

fn span_offsets(tokens: usize) -> (Vec<usize>, usize) {
    let mut offsets = vec![0; tokens.saturating_add(2)];
    let mut total = 0;

    for (width, offset) in offsets.iter_mut().enumerate().take(tokens + 1).skip(1) {
        *offset = total;
        total += tokens - width + 1;
    }

    if let Some(last) = offsets.last_mut() {
        *last = total;
    }
    (offsets, total)
}

const fn indexed_span(tokens: usize, offsets: &[usize], span: Span) -> Result<usize, TableError> {
    let (start, end) = span;

    if start >= end || end > tokens {
        return Err(TableError::InvalidSpan { span, tokens });
    }
    Ok(offsets[end - start] + start)
}

/// Errors at the native inside-table boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TableError {
    /// A span lies outside the table's sentence length.
    InvalidSpan {
        /// The invalid span.
        span: Span,
        /// The sentence length owned by the table.
        tokens: usize,
    },
    /// A pair names a position outside the supplied category sequences.
    InvalidPair {
        /// The supplied left position.
        left: usize,
        /// Number of left categories.
        left_categories: usize,
        /// The supplied right position.
        right: usize,
        /// Number of right categories.
        right_categories: usize,
    },
    /// A resolver returned a different number of rows than the submitted batch.
    InvalidStepBatch {
        /// Number of submitted requests.
        requests: usize,
        /// Number of returned result rows.
        results: usize,
    },
    /// A sparse result names a tag row outside its submitted request.
    InvalidStepRow {
        /// Position of the request in the width-level batch.
        request: usize,
        /// Supplied tag-row position.
        row: usize,
        /// Number of tag rows in the request.
        rows: usize,
    },
    /// A factored product does not belong to its returned pair batch.
    InvalidFactoredBatch,
    /// A pending inside row names a category group outside its source cell.
    InvalidSourceGroup {
        /// Source span holding the premise categories.
        span: Span,
        /// Supplied group position.
        group: usize,
        /// Number of groups in the source cell.
        groups: usize,
    },
    /// An item is not present in the forest.
    UnknownItem {
        /// The absent item.
        key: ForestKey,
    },
    /// A restriction names an item outside the cell it restricts.
    ForeignKeep {
        /// The restricted span.
        span: Span,
        /// The item outside that span or absent from it.
        key: ForestKey,
    },
    /// A product alternative contains more logical edges than the table can count.
    EdgeCountOverflow,
    /// The declared chart-cell worker pool could not be constructed.
    ThreadPool {
        /// Number of workers requested by the parse declaration.
        threads: NonZeroUsize,
    },
}

impl fmt::Display for TableError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSpan { span, tokens } => write!(
                formatter,
                "span {span:?} is outside the chart's span index for {tokens} tokens"
            ),
            Self::InvalidPair {
                left,
                left_categories,
                right,
                right_categories,
            } => write!(
                formatter,
                "pair ({left}, {right}) is outside category rows of lengths \
                 ({left_categories}, {right_categories})"
            ),
            Self::InvalidStepBatch { requests, results } => write!(
                formatter,
                "a batch of {requests} step requests returned {results} result rows"
            ),
            Self::InvalidStepRow { request, row, rows } => write!(
                formatter,
                "step request {request} has {rows} tag rows, not row {row}"
            ),
            Self::InvalidFactoredBatch => {
                write!(
                    formatter,
                    "a factored product is outside its pair batch arena"
                )
            }
            Self::InvalidSourceGroup {
                span,
                group,
                groups,
            } => write!(
                formatter,
                "span {span:?} has {groups} category groups, not group {group}"
            ),
            Self::UnknownItem { key } => write!(formatter, "the forest holds no item {key:?}"),
            Self::ForeignKeep { span, key } => {
                write!(formatter, "span {span:?} holds no item {key:?}")
            }
            Self::EdgeCountOverflow => {
                write!(
                    formatter,
                    "a product alternative exceeds the forest edge-count width"
                )
            }
            Self::ThreadPool { threads } => {
                write!(
                    formatter,
                    "could not construct a {threads}-thread chart worker pool"
                )
            }
        }
    }
}

impl std::error::Error for TableError {}

/// One failure raised while sweeping an inside table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SweepError<R, S> {
    /// A drawing or scoring callback failed.
    Resolver(R),
    /// Semiring arithmetic failed.
    Semiring(S),
    /// An internal table access crossed the table boundary.
    Table(TableError),
}

/// One inside-table query that can fail at its span or carrier boundary.
pub type TableResult<T, E> = Result<T, SweepError<TableError, E>>;

fn lift_table_error<R, S>(error: SweepError<TableError, S>) -> SweepError<R, S> {
    match error {
        SweepError::Resolver(error) | SweepError::Table(error) => SweepError::Table(error),
        SweepError::Semiring(error) => SweepError::Semiring(error),
    }
}

/// How one sweep executes independent chart cells.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CellExecution {
    /// Run cell work on the attached caller for a carrier that invokes Python.
    Attached,
    /// Run cell work on a fixed-size native worker pool.
    Parallel(NonZeroUsize),
}

fn worker_pool(execution: CellExecution) -> Result<Option<ThreadPool>, TableError> {
    match execution {
        CellExecution::Attached => Ok(None),
        CellExecution::Parallel(threads) => ThreadPoolBuilder::new()
            .num_threads(threads.get())
            .build()
            .map(Some)
            .map_err(|_| TableError::ThreadPool { threads }),
    }
}

/// The drawing and local-scoring capabilities required by one inside sweep.
pub trait InsideResolver<W> {
    /// An error raised by drawing, scoring, or an interrupt check.
    type Error;

    /// Return rules of one arity in declaration order.
    ///
    /// # Errors
    ///
    /// Returns the resolver's error when the rule metadata cannot be read.
    fn rules_of_arity(&mut self, arity: usize) -> Result<Vec<RuleId>, Self::Error>;

    /// Enumerate indexed binary pairs and admitted factored conclusions.
    ///
    /// # Errors
    ///
    /// Returns the resolver's error when category metadata cannot be read.
    fn pair_batches(
        &mut self,
        batches: &[PairBatchRequest],
        workers: Option<&ThreadPool>,
        factor_units: bool,
    ) -> Result<Vec<PairBatch>, Self::Error>;

    /// Report whether local edge weights are invariant across split points.
    fn local_weight_is_split_invariant(&self) -> bool;

    /// Report whether every local edge weight is the multiplicative identity.
    fn local_weights_are_unit(&self) -> bool;

    /// Return the portion of a decomposition that can affect reachability.
    ///
    /// Equal keys permit rule drawing to be shared while semiring accumulation
    /// continues to use each decomposition's exact source spans.
    fn permit_key(&self, parts: &[Span]) -> Vec<Option<Span>>;

    /// Return the compiled normal-form decision for one rule and child-tag row.
    ///
    /// # Errors
    ///
    /// Returns the resolver's error when the rule or tags are unknown.
    fn decision(&mut self, rule: RuleId, tags: &[TagId]) -> Result<NormalDecision, Self::Error>;

    /// Decide exact premise-sensitive normal-form rows over one known output.
    ///
    /// # Errors
    ///
    /// Returns the resolver's error when a rule, tag, or category is unknown.
    fn admit_rows(
        &mut self,
        rule: RuleId,
        tags: &[TagId],
        input_pairs: &[[CategoryId; 2]],
        output: CategoryId,
    ) -> Result<Vec<bool>, Self::Error>;

    /// Report whether one concluded category remains reachable at this span.
    ///
    /// # Errors
    ///
    /// Returns the resolver's error when the category or reachability program
    /// cannot be read.
    fn keeps(&mut self, span: Span, category: CategoryId) -> Result<bool, Self::Error>;

    /// Report whether one rule may consume an exact premise tuple for a split.
    ///
    /// # Errors
    ///
    /// Returns the resolver's error when the categories or reachability
    /// program cannot be read.
    fn permits(
        &mut self,
        span: Span,
        parts: &[Span],
        rule: RuleId,
        inputs: &[CategoryId],
    ) -> Result<bool, Self::Error>;

    /// Draw admitted steps for exact premise tuples in request order.
    ///
    /// # Errors
    ///
    /// Returns the resolver's error when rule application fails.
    fn steps(
        &mut self,
        requests: &[StepBatchRequest],
        workers: Option<&ThreadPool>,
    ) -> Result<Vec<StepBatchEntry>, Self::Error>;

    /// Close unary rules over one cell.
    ///
    /// # Errors
    ///
    /// Returns the resolver's error when unary drawing fails.
    fn closure(
        &mut self,
        arrival: &[ItemId],
        span: Span,
        depth: usize,
        workers: Option<&ThreadPool>,
    ) -> Result<UnaryClosure, Self::Error>;

    /// Return one rule application's local weight.
    ///
    /// # Errors
    ///
    /// Returns the resolver's error when scoring fails.
    fn local_weight(
        &mut self,
        rule: RuleId,
        inputs: &[CategoryId],
        output: CategoryId,
        span: Span,
        split: Option<usize>,
    ) -> Result<W, Self::Error>;

    /// Check for an interrupt after one complete span width.
    ///
    /// # Errors
    ///
    /// Returns the resolver's interrupt error.
    fn finish_width(&mut self) -> Result<(), Self::Error>;
}

#[derive(Clone, Debug)]
struct InsideCell<W> {
    groups: Vec<CategoryGroup<W>>,
    positions: HashMap<CategoryId, usize>,
    tagged_positions: HashMap<ItemId, (usize, usize)>,
}

impl<W> Default for InsideCell<W> {
    fn default() -> Self {
        Self {
            groups: Vec::new(),
            positions: HashMap::new(),
            tagged_positions: HashMap::new(),
        }
    }
}

impl<W: Clone> InsideCell<W> {
    fn add<S>(&mut self, item: ItemId, weight: W, semiring: &S) -> Result<(), S::Error>
    where
        S: Semiring<Weight = W>,
    {
        if let Some((group, tag)) = self.tagged_positions.get(&item).copied() {
            semiring.plus_assign(&mut self.groups[group].tagged[tag].1, &weight)?;
            return Ok(());
        }
        let group = *self.positions.entry(item.category).or_insert_with(|| {
            let position = self.groups.len();
            self.groups.push(CategoryGroup {
                category: item.category,
                tagged: Vec::new(),
            });
            position
        });
        let tag = self.groups[group].tagged.len();
        self.groups[group].tagged.push((item.tag, weight));
        self.tagged_positions.insert(item, (group, tag));
        Ok(())
    }

    fn weight(&self, item: ItemId) -> Option<&W> {
        self.tagged_positions
            .get(&item)
            .and_then(|(group, tag)| self.groups.get(*group)?.tagged.get(*tag))
            .map(|(_, weight)| weight)
    }

    fn restrict(&mut self, keep: &HashSet<ItemId>) {
        for group in &mut self.groups {
            group.tagged.retain(|(tag, _)| {
                keep.contains(&ItemId {
                    category: group.category,
                    tag: *tag,
                })
            });
        }
        self.groups.retain(|group| !group.tagged.is_empty());
        self.positions.clear();
        self.tagged_positions.clear();

        for (group_at, group) in self.groups.iter().enumerate() {
            self.positions.insert(group.category, group_at);
            for (tag_at, (tag, _)) in group.tagged.iter().enumerate() {
                self.tagged_positions.insert(
                    ItemId {
                        category: group.category,
                        tag: *tag,
                    },
                    (group_at, tag_at),
                );
            }
        }
    }

    fn groups(&self) -> &[CategoryGroup<W>] {
        &self.groups
    }
}

#[derive(Clone, Debug)]
struct CategoryGroup<W> {
    category: CategoryId,
    tagged: Vec<(TagId, W)>,
}

/// Native storage and the single accumulation algorithm for an inside chart.
#[derive(Clone, Debug)]
pub struct InsideTable<W> {
    tokens: usize,
    offsets: Vec<usize>,
    cells: Vec<InsideCell<W>>,
    truncated: bool,
}

impl<W: Clone> InsideTable<W> {
    /// Construct an empty table for a sentence length.
    #[must_use]
    pub fn new(tokens: usize) -> Self {
        let (offsets, total) = span_offsets(tokens);

        Self {
            tokens,
            offsets,
            cells: (0..total).map(|_| InsideCell::default()).collect(),
            truncated: false,
        }
    }

    /// Return the sentence length owned by the table.
    #[must_use]
    pub const fn tokens(&self) -> usize {
        self.tokens
    }

    /// Return whether bounded unary closure omitted a derivation.
    #[must_use]
    pub const fn truncated(&self) -> bool {
        self.truncated
    }

    /// Record that bounded construction omitted a derivation.
    pub const fn mark_truncated(&mut self) {
        self.truncated = true;
    }

    /// Return the number of exact categories held over all spans.
    #[must_use]
    pub fn num_categories(&self) -> usize {
        self.cells.iter().map(|cell| cell.groups.len()).sum()
    }

    fn span_id(&self, span: Span) -> Result<usize, TableError> {
        indexed_span(self.tokens, &self.offsets, span)
    }

    fn cell(&self, span: Span) -> Result<&InsideCell<W>, TableError> {
        let identifier = self.span_id(span)?;
        Ok(&self.cells[identifier])
    }

    fn cell_mut(&mut self, span: Span) -> Result<&mut InsideCell<W>, TableError> {
        let identifier = self.span_id(span)?;
        Ok(&mut self.cells[identifier])
    }

    /// Add one derivation weight to an exact category and tag.
    ///
    /// # Errors
    ///
    /// Returns a table error for an invalid span or the carrier's arithmetic error.
    pub fn add<S>(
        &mut self,
        span: Span,
        item: ItemId,
        weight: W,
        semiring: &S,
    ) -> Result<(), SweepError<TableError, S::Error>>
    where
        S: Semiring<Weight = W>,
    {
        self.cell_mut(span)
            .map_err(SweepError::Table)?
            .add(item, weight, semiring)
            .map_err(SweepError::Semiring)
    }

    /// Keep only the named items at one span.
    ///
    /// # Errors
    ///
    /// Returns a table error when the span is invalid.
    pub fn restrict(&mut self, span: Span, keep: &[ItemId]) -> Result<(), TableError> {
        let keep: HashSet<ItemId> = keep.iter().copied().collect();
        self.cell_mut(span)?.restrict(&keep);
        Ok(())
    }

    /// Return tagged weights at one span in arrival order.
    ///
    /// # Errors
    ///
    /// Returns a table error when the span is invalid.
    pub fn tagged_at(&self, span: Span) -> Result<Vec<(ItemId, W)>, TableError> {
        Ok(self
            .cell(span)?
            .groups()
            .iter()
            .flat_map(|group| {
                group.tagged.iter().map(move |(tag, weight)| {
                    (
                        ItemId {
                            category: group.category,
                            tag: *tag,
                        },
                        weight.clone(),
                    )
                })
            })
            .collect())
    }

    /// Return items at one span in arrival order.
    ///
    /// # Errors
    ///
    /// Returns a table error when the span is invalid.
    pub fn items_at(&self, span: Span) -> Result<Vec<ItemId>, TableError> {
        Ok(self
            .cell(span)?
            .groups()
            .iter()
            .flat_map(|group| {
                group.tagged.iter().map(move |(tag, _)| ItemId {
                    category: group.category,
                    tag: *tag,
                })
            })
            .collect())
    }

    /// Return one tagged weight, when present.
    ///
    /// # Errors
    ///
    /// Returns a table error when the span is invalid.
    pub fn tagged(&self, span: Span, item: ItemId) -> Result<Option<W>, TableError> {
        Ok(self.cell(span)?.weight(item).cloned())
    }

    /// Return category weights summed over tags in category-arrival order.
    ///
    /// # Errors
    ///
    /// Returns a table error when the span is invalid or the carrier cannot add.
    pub fn categories_at<S>(
        &self,
        span: Span,
        semiring: &S,
    ) -> TableResult<Vec<(CategoryId, W)>, S::Error>
    where
        S: Semiring<Weight = W>,
    {
        let groups = self.cell(span).map_err(SweepError::Table)?.groups();
        let mut categories = Vec::with_capacity(groups.len());

        for group in groups {
            let mut total = semiring.zero();

            for (_, weight) in &group.tagged {
                semiring
                    .plus_assign(&mut total, weight)
                    .map_err(SweepError::Semiring)?;
            }
            categories.push((group.category, total));
        }
        Ok(categories)
    }

    /// Return one category's weight summed over tags.
    ///
    /// # Errors
    ///
    /// Returns a table error when the span is invalid or the carrier cannot add.
    pub fn inside<S>(
        &self,
        span: Span,
        category: CategoryId,
        semiring: &S,
    ) -> Result<W, SweepError<TableError, S::Error>>
    where
        S: Semiring<Weight = W>,
    {
        let cell = self.cell(span).map_err(SweepError::Table)?;
        let mut total = semiring.zero();

        if let Some(position) = cell.positions.get(&category).copied() {
            for (_, weight) in &cell.groups[position].tagged {
                semiring
                    .plus_assign(&mut total, weight)
                    .map_err(SweepError::Semiring)?;
            }
        }
        Ok(total)
    }

    /// Fill every cell by width using one drawing resolver and one carrier.
    ///
    /// Returns the root cell's unary closure so the Python goal boundary can
    /// restrict it after the native sweep.
    ///
    /// # Errors
    ///
    /// Returns a resolver or semiring error from the supplied capabilities.
    pub fn sweep<S, R>(
        &mut self,
        active_arities: &[usize],
        unary_depth: usize,
        execution: CellExecution,
        semiring: &S,
        resolver: &mut R,
    ) -> Result<UnaryClosure, SweepError<R::Error, S::Error>>
    where
        W: Send + Sync,
        S: Semiring<Weight = W> + Sync,
        S::Error: Send,
        R: InsideResolver<W>,
    {
        let pool = worker_pool(execution).map_err(SweepError::Table)?;
        let mut root_closure = UnaryClosure {
            edges: Vec::new(),
            concluded: Vec::new(),
            truncated: false,
        };

        for width in 1..=self.tokens {
            let spans: Vec<Span> = (0..=self.tokens - width)
                .map(|start| (start, start + width))
                .collect();
            let plans = cell_plans(&spans, active_arities, pool.as_ref());
            self.fill_width(&plans, pool.as_ref(), semiring, resolver)?;
            for span in spans {
                let arrival = self.items_at(span).map_err(SweepError::Table)?;
                let closure = resolver
                    .closure(&arrival, span, unary_depth, pool.as_ref())
                    .map_err(SweepError::Resolver)?;
                self.apply_closure(span, &arrival, &closure, semiring, resolver)?;

                if span == (0, self.tokens) {
                    root_closure = closure;
                }
            }
            resolver.finish_width().map_err(SweepError::Resolver)?;
        }
        Ok(root_closure)
    }

    fn fill_width<S, R>(
        &mut self,
        plans: &[CellPlan],
        pool: Option<&ThreadPool>,
        semiring: &S,
        resolver: &mut R,
    ) -> Result<(), SweepError<R::Error, S::Error>>
    where
        W: Send + Sync,
        S: Semiring<Weight = W> + Sync,
        S::Error: Send,
        R: InsideResolver<W>,
    {
        let pair_inputs = self.pair_inputs(plans).map_err(SweepError::Table)?;
        let unit_local_weights = resolver.local_weights_are_unit();
        let pair_batches = resolver
            .pair_batches(&pair_inputs, pool, unit_local_weights)
            .map_err(SweepError::Resolver)?;
        if pair_batches.len() != pair_inputs.len() {
            return Err(SweepError::Table(TableError::InvalidStepBatch {
                requests: pair_inputs.len(),
                results: pair_batches.len(),
            }));
        }
        let mut pending_work = Vec::with_capacity(plans.len());
        let mut requests = Vec::new();
        let mut request_splits = Vec::new();
        let mut binary_at = 0;
        for plan in plans {
            let mut pending = Vec::new();
            let mut factorized = Vec::new();
            let mut factor_axes = Vec::new();
            let mut factor_axis_ids = HashMap::new();
            let mut grouped = resolver
                .local_weight_is_split_invariant()
                .then(HashMap::new);
            for parts in &plan.decompositions {
                let pairs = if parts.len() == 2 {
                    let held = pair_batches.get(binary_at).ok_or_else(|| {
                        SweepError::Table(TableError::InvalidStepBatch {
                            requests: pair_inputs.len(),
                            results: binary_at,
                        })
                    })?;
                    binary_at += 1;
                    Some(held)
                } else {
                    None
                };
                let mut output = InsideRequests {
                    requests: &mut requests,
                    splits: &mut request_splits,
                    pending: &mut pending,
                    factorized: &mut factorized,
                    factor_axes: &mut factor_axes,
                    factor_axis_ids: &mut factor_axis_ids,
                    grouped: grouped.as_mut(),
                };
                self.resolve_decomposition(plan.span, parts, pairs, resolver, &mut output)?;
            }
            let cell = self.cell(plan.span).map_err(SweepError::Table)?.clone();
            pending_work.push((plan.span, cell, pending, factorized, factor_axes));
        }
        let step_rows = if requests.is_empty() {
            Vec::new()
        } else {
            resolver
                .steps(&requests, pool)
                .map_err(SweepError::Resolver)?
        };
        let weighted_rows = resolve_inside_rows(&requests, &request_splits, &step_rows, resolver)?;
        let accumulation = InsideAccumulation {
            requests: &requests,
            resolved: &weighted_rows,
            unit_local_weights,
            semiring,
        };
        let accumulate = |(span, cell, pending, factorized, factor_axes): PendingInsideBuild<W>| {
            (
                span,
                self.accumulate_inside_cell(
                    cell,
                    &pending,
                    &factorized,
                    &factor_axes,
                    &accumulation,
                    pool,
                ),
            )
        };
        let completed: Vec<InsideBuild<W, S::Error>> = match pool {
            None => pending_work.into_iter().map(accumulate).collect(),
            Some(workers) => {
                workers.install(|| pending_work.into_par_iter().map(accumulate).collect())
            }
        };
        for (span, cell) in completed {
            let identifier = self.span_id(span).map_err(SweepError::Table)?;
            self.cells[identifier] = cell.map_err(lift_table_error)?;
        }
        Ok(())
    }

    fn pair_inputs(&self, plans: &[CellPlan]) -> Result<Vec<PairBatchRequest>, TableError> {
        let mut inputs = Vec::new();
        for (span, parts) in plans
            .iter()
            .flat_map(|plan| {
                plan.decompositions
                    .iter()
                    .map(move |parts| (plan.span, parts))
            })
            .filter(|(_, parts)| parts.len() == 2)
        {
            let categories = |span| {
                self.cell(span)
                    .map(|cell| cell.groups().iter().map(|group| group.category).collect())
            };
            inputs.push(PairBatchRequest {
                span,
                parts: [parts[0], parts[1]],
                left: categories(parts[0])?,
                right: categories(parts[1])?,
            });
        }
        Ok(inputs)
    }

    fn resolve_decomposition<R, E>(
        &self,
        span: Span,
        parts: &[Span],
        pairs: Option<&PairBatch>,
        resolver: &mut R,
        output: &mut InsideRequests<'_>,
    ) -> Result<(), SweepError<R::Error, E>>
    where
        R: InsideResolver<W>,
    {
        let groups = parts
            .iter()
            .map(|part| {
                self.cell(*part)
                    .map(InsideCell::groups)
                    .map_err(SweepError::Table)
            })
            .collect::<Result<Vec<_>, _>>()?;

        if groups.iter().any(|group| group.is_empty()) {
            return Ok(());
        }

        let permit_key = resolver.permit_key(parts);

        if groups.len() == 2 {
            let Some(batch) = pairs else {
                return Ok(());
            };
            let mut decisions = HashMap::new();
            for pair in &batch.pairs {
                Self::resolve_binary_pair::<R, E>(
                    span,
                    parts,
                    &permit_key,
                    pair,
                    [groups[0], groups[1]],
                    output,
                )?;
            }
            for (product_at, product) in batch.products.iter().enumerate() {
                let axes = batch
                    .product_axes_at(product_at)
                    .map_err(|_| SweepError::Table(TableError::InvalidFactoredBatch))?;
                if !resolver
                    .keeps(span, product.output)
                    .map_err(SweepError::Resolver)?
                {
                    continue;
                }
                Self::resolve_factored_pair_product(
                    &FactoredInsideResolution {
                        parts,
                        product,
                        axes,
                        groups: [groups[0], groups[1]],
                    },
                    resolver,
                    &mut decisions,
                    output,
                )?;
            }
        } else {
            let candidates = resolver
                .rules_of_arity(groups.len())
                .map_err(SweepError::Resolver)?;
            let product = CategoryProduct {
                span,
                parts,
                permit_key: &permit_key,
                groups: &groups,
                candidates: &candidates,
            };
            let mut selected = Vec::with_capacity(product.groups.len());
            Self::category_product(&product, 0, &mut selected, output);
        }
        Ok(())
    }

    fn resolve_factored_pair_product<R, E>(
        resolution: &FactoredInsideResolution<'_, W>,
        resolver: &mut R,
        decisions: &mut HashMap<(RuleId, [TagId; 2]), NormalDecision>,
        output: &mut InsideRequests<'_>,
    ) -> Result<(), SweepError<R::Error, E>>
    where
        R: InsideResolver<W>,
    {
        let parts = resolution.parts;
        let product = resolution.product;
        let groups = resolution.groups;
        let positions = resolution.axes;
        let axis_ids =
            factored_inside_axes(parts, positions, groups, output).map_err(SweepError::Table)?;
        for (left_tag, left_axis) in &axis_ids[0] {
            for (right_tag, right_axis) in &axis_ids[1] {
                let tags = [*left_tag, *right_tag];
                let decision = if let Some(decision) = decisions.get(&(product.rule, tags)) {
                    *decision
                } else {
                    let decision = resolver
                        .decision(product.rule, &tags)
                        .map_err(SweepError::Resolver)?;
                    decisions.insert((product.rule, tags), decision);
                    decision
                };
                match decision {
                    NormalDecision::Admit(tag) => {
                        output.factorized.push(PendingUnitProduct {
                            output: ItemId {
                                category: product.output,
                                tag,
                            },
                            axes: [*left_axis, *right_axis],
                        });
                    }
                    NormalDecision::Premises(tag) => {
                        let mut left_positions = Vec::new();
                        for left in positions[0] {
                            let Some(group) = groups[0].get(*left) else {
                                return Err(SweepError::Table(TableError::InvalidPair {
                                    left: *left,
                                    left_categories: groups[0].len(),
                                    right: positions[1].first().copied().unwrap_or(0),
                                    right_categories: groups[1].len(),
                                }));
                            };
                            if group.tagged.iter().any(|(held, _)| held == left_tag) {
                                left_positions.push(*left);
                            }
                        }
                        let mut right_positions = Vec::new();
                        for right in positions[1] {
                            let Some(group) = groups[1].get(*right) else {
                                return Err(SweepError::Table(TableError::InvalidPair {
                                    left: positions[0].first().copied().unwrap_or(0),
                                    left_categories: groups[0].len(),
                                    right: *right,
                                    right_categories: groups[1].len(),
                                }));
                            };
                            if group.tagged.iter().any(|(held, _)| held == right_tag) {
                                right_positions.push(*right);
                            }
                        }
                        if left_positions.is_empty() || right_positions.is_empty() {
                            continue;
                        }
                        let mut by_rights = HashMap::<Vec<usize>, Vec<usize>>::new();
                        for left in left_positions {
                            let input_pairs = right_positions
                                .iter()
                                .map(|right| [groups[0][left].category, groups[1][*right].category])
                                .collect::<Vec<_>>();
                            let admitted = resolver
                                .admit_rows(product.rule, &tags, &input_pairs, product.output)
                                .map_err(SweepError::Resolver)?;
                            let admitted = admitted_pair_rights(&right_positions, &admitted)
                                .map_err(SweepError::Table)?;
                            if !admitted.is_empty() {
                                by_rights.entry(admitted).or_default().push(left);
                            }
                        }
                        let rectangles = admitted_pair_rectangles(by_rights);
                        for (left, right) in rectangles {
                            let left_axis =
                                factored_inside_axis(parts[0], &left, *left_tag, groups[0], output)
                                    .map_err(SweepError::Table)?;
                            let right_axis = factored_inside_axis(
                                parts[1], &right, *right_tag, groups[1], output,
                            )
                            .map_err(SweepError::Table)?;
                            if let (Some(left_axis), Some(right_axis)) = (left_axis, right_axis) {
                                output.factorized.push(PendingUnitProduct {
                                    output: ItemId {
                                        category: product.output,
                                        tag,
                                    },
                                    axes: [left_axis, right_axis],
                                });
                            }
                        }
                    }
                    NormalDecision::Refuse => {}
                }
            }
        }
        Ok(())
    }

    fn resolve_binary_pair<R, E>(
        span: Span,
        parts: &[Span],
        permit_key: &[Option<Span>],
        pair: &Pair,
        groups: [&[CategoryGroup<W>]; 2],
        output: &mut InsideRequests<'_>,
    ) -> Result<(), SweepError<R::Error, E>>
    where
        R: InsideResolver<W>,
    {
        let Some(left_group) = groups[0].get(pair.left) else {
            return Err(SweepError::Table(TableError::InvalidPair {
                left: pair.left,
                left_categories: groups[0].len(),
                right: pair.right,
                right_categories: groups[1].len(),
            }));
        };
        let Some(right_group) = groups[1].get(pair.right) else {
            return Err(SweepError::Table(TableError::InvalidPair {
                left: pair.left,
                left_categories: groups[0].len(),
                right: pair.right,
                right_categories: groups[1].len(),
            }));
        };
        let selected = [left_group, right_group];
        let positions = [pair.left, pair.right];
        if !pair.candidates.is_empty() {
            Self::resolve_groups(
                span,
                parts,
                permit_key,
                &selected,
                &positions,
                &pair.candidates,
                output,
            );
        }
        Ok(())
    }

    fn category_product(
        product: &CategoryProduct<'_, W>,
        at: usize,
        selected: &mut Vec<usize>,
        output: &mut InsideRequests<'_>,
    ) {
        if at == product.groups.len() {
            let chosen: Vec<&CategoryGroup<W>> = selected
                .iter()
                .enumerate()
                .map(|(part, position)| &product.groups[part][*position])
                .collect();
            Self::resolve_groups(
                product.span,
                product.parts,
                product.permit_key,
                &chosen,
                selected,
                product.candidates,
                output,
            );
            return;
        }

        for position in 0..product.groups[at].len() {
            selected.push(position);
            Self::category_product(product, at + 1, selected, output);
            selected.pop();
        }
    }

    fn resolve_groups(
        span: Span,
        parts: &[Span],
        permit_key: &[Option<Span>],
        groups: &[&CategoryGroup<W>],
        positions: &[usize],
        candidates: &[RuleId],
        output: &mut InsideRequests<'_>,
    ) {
        let tag_axes: Vec<Vec<TagId>> = groups
            .iter()
            .map(|group| group.tagged.iter().map(|(tag, _)| *tag).collect())
            .collect();
        let tag_rows = tag_tuples(&tag_axes);
        let group = GroupResolution {
            span,
            parts,
            permit_key,
            groups,
            positions,
            candidates,
        };
        Self::resolve_group_rows(&group, tag_rows, output);
    }

    fn resolve_group_rows(
        group: &GroupResolution<'_, W>,
        tag_rows: Vec<Vec<TagId>>,
        output: &mut InsideRequests<'_>,
    ) {
        let inputs: Vec<CategoryId> = group
            .groups
            .iter()
            .map(|category| category.category)
            .collect();
        let split = (group.parts.len() == 2).then_some(group.parts[0].1);
        let key = StepGroupKey {
            inputs: inputs.clone(),
            candidates: group.candidates.to_vec(),
            permit_key: group.permit_key.to_vec(),
        };
        let request = if let Some(request) = output
            .grouped
            .as_deref()
            .and_then(|grouped| grouped.get(&key).copied())
        {
            let held = &mut output.requests[request].tag_rows;
            for tags in tag_rows {
                if !held.contains(&tags) {
                    held.push(tags);
                }
            }
            request
        } else {
            let request = output.requests.len();
            output.requests.push(StepBatchRequest {
                inputs,
                tag_rows,
                span: group.span,
                parts: group.parts.to_vec(),
                candidates: group.candidates.to_vec(),
            });
            output.splits.push(split);
            if let Some(grouped) = output.grouped.as_deref_mut() {
                grouped.insert(key, request);
            }
            request
        };
        output.pending.push(PendingInside {
            request,
            sources: group
                .parts
                .iter()
                .copied()
                .zip(group.positions.iter().copied())
                .collect(),
        });
    }

    fn accumulate_inside_cell<S>(
        &self,
        mut cell: InsideCell<W>,
        pending: &[PendingInside],
        factorized: &[PendingUnitProduct],
        factor_axes: &[Vec<UnitSource>],
        accumulation: &InsideAccumulation<'_, W, S>,
        workers: Option<&ThreadPool>,
    ) -> Result<InsideCell<W>, SweepError<TableError, S::Error>>
    where
        W: Send + Sync,
        S: Semiring<Weight = W> + Sync,
        S::Error: Send,
    {
        const CHUNK_SIZE: usize = 256;

        if let Some(pool) = workers.filter(|_| pending.len() > CHUNK_SIZE) {
            let partials = pool.install(|| {
                pending
                    .par_chunks(CHUNK_SIZE)
                    .map(|chunk| {
                        self.accumulate_inside_chunk(InsideCell::default(), chunk, accumulation)
                    })
                    .collect::<Vec<_>>()
            });
            for partial in partials {
                for group in partial?.groups {
                    for (tag, weight) in group.tagged {
                        cell.add(
                            ItemId {
                                category: group.category,
                                tag,
                            },
                            weight,
                            accumulation.semiring,
                        )
                        .map_err(SweepError::Semiring)?;
                    }
                }
            }
        } else {
            cell = self.accumulate_inside_chunk(cell, pending, accumulation)?;
        }
        self.accumulate_factored(cell, factorized, factor_axes, accumulation.semiring)
    }

    fn accumulate_factored<S>(
        &self,
        mut cell: InsideCell<W>,
        pending: &[PendingUnitProduct],
        axes: &[Vec<UnitSource>],
        semiring: &S,
    ) -> Result<InsideCell<W>, SweepError<TableError, S::Error>>
    where
        S: Semiring<Weight = W>,
    {
        let mut products = HashMap::<[usize; 2], W>::new();
        let mut axis_totals: Vec<Option<W>> = vec![None; axes.len()];
        for held in pending {
            let source_weight = |source: UnitSource| {
                let cell = self.cell(source.span).map_err(SweepError::Table)?;
                let group = cell.groups.get(source.group).ok_or_else(|| {
                    SweepError::Table(TableError::InvalidSourceGroup {
                        span: source.span,
                        group: source.group,
                        groups: cell.groups.len(),
                    })
                })?;
                Ok::<_, SweepError<TableError, S::Error>>(
                    group
                        .tagged
                        .iter()
                        .find(|(tag, _)| *tag == source.tag)
                        .map(|(_, weight)| weight),
                )
            };
            let weight = if let Some(weight) = products.get(&held.axes) {
                weight.clone()
            } else {
                let mut totals = Vec::with_capacity(2);
                for identifier in held.axes {
                    let total = if let Some(total) = &axis_totals[identifier] {
                        Some(total.clone())
                    } else {
                        let factor_axis = &axes[identifier];
                        let mut total = None;
                        for source in factor_axis {
                            let Some(weight) = source_weight(*source)? else {
                                continue;
                            };
                            if let Some(sum) = &mut total {
                                semiring
                                    .plus_assign(sum, weight)
                                    .map_err(SweepError::Semiring)?;
                            } else {
                                total = Some(weight.clone());
                            }
                        }
                        if let Some(total) = &total {
                            axis_totals[identifier] = Some(total.clone());
                        }
                        total
                    };
                    let Some(total) = total else {
                        totals.clear();
                        break;
                    };
                    totals.push(total);
                }
                let [left, right] = totals.as_slice() else {
                    continue;
                };
                let weight = semiring.times(left, right).map_err(SweepError::Semiring)?;
                products.insert(held.axes, weight.clone());
                weight
            };
            cell.add(held.output, weight, semiring)
                .map_err(SweepError::Semiring)?;
        }
        Ok(cell)
    }

    fn accumulate_inside_chunk<S>(
        &self,
        mut cell: InsideCell<W>,
        pending: &[PendingInside],
        accumulation: &InsideAccumulation<'_, W, S>,
    ) -> Result<InsideCell<W>, SweepError<TableError, S::Error>>
    where
        S: Semiring<Weight = W>,
    {
        let mut chunk = InsideChunk {
            accumulation,
            factors: Vec::new(),
            factor_positions: HashMap::new(),
        };
        for held in pending {
            self.accumulate_pending(&mut cell, held, &mut chunk)?;
        }

        for factor in chunk.factors {
            let weight = accumulation
                .semiring
                .times(&factor.left, &factor.rights)
                .map_err(SweepError::Semiring)?;
            cell.add(factor.output, weight, accumulation.semiring)
                .map_err(SweepError::Semiring)?;
        }
        Ok(cell)
    }

    fn accumulate_pending<S>(
        &self,
        cell: &mut InsideCell<W>,
        held: &PendingInside,
        chunk: &mut InsideChunk<'_, W, S>,
    ) -> Result<(), SweepError<TableError, S::Error>>
    where
        S: Semiring<Weight = W>,
    {
        let request = chunk.request(held.request)?;
        let rows = chunk.rows(held.request)?;
        let groups = held
            .sources
            .iter()
            .map(|(span, group)| {
                let source = self.cell(*span).map_err(SweepError::Table)?;
                source.groups.get(*group).ok_or_else(|| {
                    SweepError::Table(TableError::InvalidSourceGroup {
                        span: *span,
                        group: *group,
                        groups: source.groups.len(),
                    })
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        if chunk.accumulation.unit_local_weights && groups.len() == 2 {
            return chunk.accumulate_unit(held, request, rows, &groups);
        }

        let mut child_weights = vec![None; request.tag_rows.len()];
        for row in rows {
            let tags = request_tags(held.request, request, row.row)?;
            if child_weights[row.row].is_none() {
                let mut product = chunk.accumulation.semiring.one();
                let mut available = true;
                for (group, tag) in groups.iter().zip(tags) {
                    let Some((_, below)) = group.tagged.iter().find(|(held, _)| held == tag) else {
                        available = false;
                        break;
                    };
                    product = chunk
                        .accumulation
                        .semiring
                        .times(&product, below)
                        .map_err(SweepError::Semiring)?;
                }
                if available {
                    child_weights[row.row] = Some(product);
                }
            }
            let Some(children) = child_weights[row.row].as_ref() else {
                continue;
            };
            let weight = chunk
                .accumulation
                .semiring
                .times(&row.local, children)
                .map_err(SweepError::Semiring)?;
            cell.add(
                ItemId {
                    category: row.step.output,
                    tag: row.step.tag,
                },
                weight,
                chunk.accumulation.semiring,
            )
            .map_err(SweepError::Semiring)?;
        }
        Ok(())
    }

    /// Apply one exact unary closure to a populated cell.
    ///
    /// # Errors
    ///
    /// Returns a resolver, table, or carrier error.
    pub fn apply_closure<S, R>(
        &mut self,
        span: Span,
        arrival: &[ItemId],
        closure: &UnaryClosure,
        semiring: &S,
        resolver: &mut R,
    ) -> Result<(), SweepError<R::Error, S::Error>>
    where
        S: Semiring<Weight = W>,
        R: InsideResolver<W>,
    {
        let arrival: HashSet<ItemId> = arrival.iter().copied().collect();
        let initial: HashMap<ItemId, W> = self
            .cell(span)
            .map_err(SweepError::Table)?
            .groups
            .iter()
            .flat_map(|group| {
                group.tagged.iter().map(|(tag, weight)| {
                    (
                        ItemId {
                            category: group.category,
                            tag: *tag,
                        },
                        weight.clone(),
                    )
                })
            })
            .collect();
        let mut incoming: HashMap<ItemId, Vec<usize>> = HashMap::new();

        for (position, (_, step)) in closure.edges.iter().enumerate() {
            incoming
                .entry(ItemId {
                    category: step.output,
                    tag: step.tag,
                })
                .or_default()
                .push(position);
        }

        let mut final_weights: HashMap<ItemId, W> = HashMap::new();
        let mut edge_weights: Vec<Option<W>> = vec![None; closure.edges.len()];

        for item in &closure.concluded {
            let mut total = if arrival.contains(item) {
                initial.get(item).cloned()
            } else {
                None
            };

            for position in incoming.get(item).into_iter().flatten() {
                let (child, step) = closure.edges[*position];
                let below = final_weights
                    .get(&child)
                    .or_else(|| initial.get(&child))
                    .cloned();
                let below = below.unwrap_or_else(|| semiring.zero());
                let local = resolver
                    .local_weight(step.rule, &[child.category], step.output, span, None)
                    .map_err(SweepError::Resolver)?;
                let weight = semiring
                    .times(&local, &below)
                    .map_err(SweepError::Semiring)?;
                edge_weights[*position] = Some(weight.clone());

                if let Some(held) = total.as_mut() {
                    semiring
                        .plus_assign(held, &weight)
                        .map_err(SweepError::Semiring)?;
                } else {
                    total = Some(weight);
                }
            }

            if let Some(weight) = total {
                final_weights.insert(*item, weight);
            }
        }

        for ((_, step), weight) in closure.edges.iter().zip(edge_weights) {
            if let Some(weight) = weight {
                self.cell_mut(span)
                    .map_err(SweepError::Table)?
                    .add(
                        ItemId {
                            category: step.output,
                            tag: step.tag,
                        },
                        weight,
                        semiring,
                    )
                    .map_err(SweepError::Semiring)?;
            }
        }

        self.truncated |= closure.truncated;
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct CellPlan {
    span: Span,
    decompositions: Vec<Vec<Span>>,
}

fn cell_plan(span: Span, active_arities: &[usize]) -> CellPlan {
    CellPlan {
        span,
        decompositions: active_arities
            .iter()
            .flat_map(|arity| span_decompositions(span, *arity))
            .collect(),
    }
}

fn cell_plans(
    spans: &[Span],
    active_arities: &[usize],
    workers: Option<&ThreadPool>,
) -> Vec<CellPlan> {
    workers.map_or_else(
        || {
            spans
                .iter()
                .map(|span| cell_plan(*span, active_arities))
                .collect()
        },
        |pool| {
            pool.install(|| {
                spans
                    .par_iter()
                    .map(|span| cell_plan(*span, active_arities))
                    .collect()
            })
        },
    )
}

#[derive(Clone, Debug)]
struct PendingInside {
    request: usize,
    sources: Vec<(Span, usize)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct UnitSource {
    span: Span,
    group: usize,
    tag: TagId,
}

#[derive(Clone, Debug)]
struct PendingUnitProduct {
    output: ItemId,
    axes: [usize; 2],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct UnitFactorKey {
    output: ItemId,
    left_span: Span,
    left: ItemId,
}

#[derive(Clone, Debug)]
struct UnitFactor<W> {
    output: ItemId,
    left: W,
    rights: W,
}

struct InsideRequests<'a> {
    requests: &'a mut Vec<StepBatchRequest>,
    splits: &'a mut Vec<Option<usize>>,
    pending: &'a mut Vec<PendingInside>,
    factorized: &'a mut Vec<PendingUnitProduct>,
    factor_axes: &'a mut Vec<Vec<UnitSource>>,
    factor_axis_ids: &'a mut HashMap<Vec<UnitSource>, usize>,
    grouped: Option<&'a mut HashMap<StepGroupKey, usize>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct StepGroupKey {
    inputs: Vec<CategoryId>,
    candidates: Vec<RuleId>,
    permit_key: Vec<Option<Span>>,
}

#[derive(Clone, Debug)]
struct ResolvedInsideStep<W> {
    row: usize,
    step: Step,
    local: W,
}

struct ResolvedInsideRows<W> {
    offsets: Vec<usize>,
    steps: Vec<ResolvedInsideStep<W>>,
}

impl<W> ResolvedInsideRows<W> {
    fn get(&self, request: usize) -> Option<&[ResolvedInsideStep<W>]> {
        let start = *self.offsets.get(request)?;
        let end = *self.offsets.get(request + 1)?;
        self.steps.get(start..end)
    }
}

type ResolvedInsideResult<W, R, S> = Result<ResolvedInsideRows<W>, SweepError<R, S>>;

type InsideBuild<W, E> = (Span, Result<InsideCell<W>, SweepError<TableError, E>>);
type PendingInsideBuild<W> = (
    Span,
    InsideCell<W>,
    Vec<PendingInside>,
    Vec<PendingUnitProduct>,
    Vec<Vec<UnitSource>>,
);
struct InsideAccumulation<'a, W, S> {
    requests: &'a [StepBatchRequest],
    resolved: &'a ResolvedInsideRows<W>,
    unit_local_weights: bool,
    semiring: &'a S,
}

struct InsideChunk<'a, W, S> {
    accumulation: &'a InsideAccumulation<'a, W, S>,
    factors: Vec<UnitFactor<W>>,
    factor_positions: HashMap<UnitFactorKey, usize>,
}

impl<'a, W, S> InsideChunk<'a, W, S>
where
    W: Clone,
    S: Semiring<Weight = W>,
{
    fn request(
        &self,
        position: usize,
    ) -> Result<&'a StepBatchRequest, SweepError<TableError, S::Error>> {
        self.accumulation.requests.get(position).ok_or_else(|| {
            SweepError::Table(TableError::InvalidStepBatch {
                requests: self.accumulation.requests.len(),
                results: position.saturating_add(1),
            })
        })
    }

    fn rows(
        &self,
        position: usize,
    ) -> Result<&'a [ResolvedInsideStep<W>], SweepError<TableError, S::Error>> {
        self.accumulation.resolved.get(position).ok_or_else(|| {
            SweepError::Table(TableError::InvalidStepBatch {
                requests: self.accumulation.requests.len(),
                results: position.saturating_add(1),
            })
        })
    }

    fn accumulate_unit(
        &mut self,
        held: &PendingInside,
        request: &StepBatchRequest,
        rows: &[ResolvedInsideStep<W>],
        groups: &[&CategoryGroup<W>],
    ) -> Result<(), SweepError<TableError, S::Error>> {
        for row in rows {
            let tags = request_tags(held.request, request, row.row)?;
            let Some((left_tag, left)) = groups[0]
                .tagged
                .iter()
                .find(|(held, _)| tags.first() == Some(held))
            else {
                continue;
            };
            let Some((_, right)) = groups[1]
                .tagged
                .iter()
                .find(|(held, _)| tags.get(1) == Some(held))
            else {
                continue;
            };
            let key = UnitFactorKey {
                output: ItemId {
                    category: row.step.output,
                    tag: row.step.tag,
                },
                left_span: held.sources[0].0,
                left: ItemId {
                    category: groups[0].category,
                    tag: *left_tag,
                },
            };
            if let Some(position) = self.factor_positions.get(&key).copied() {
                self.accumulation
                    .semiring
                    .plus_assign(&mut self.factors[position].rights, right)
                    .map_err(SweepError::Semiring)?;
            } else {
                self.factor_positions.insert(key, self.factors.len());
                self.factors.push(UnitFactor {
                    output: key.output,
                    left: left.clone(),
                    rights: right.clone(),
                });
            }
        }
        Ok(())
    }
}

fn request_tags<E>(
    request_at: usize,
    request: &StepBatchRequest,
    row: usize,
) -> Result<&[TagId], SweepError<TableError, E>> {
    request.tag_rows.get(row).map(Vec::as_slice).ok_or_else(|| {
        SweepError::Table(TableError::InvalidStepRow {
            request: request_at,
            row,
            rows: request.tag_rows.len(),
        })
    })
}

struct CategoryProduct<'a, W> {
    span: Span,
    parts: &'a [Span],
    permit_key: &'a [Option<Span>],
    groups: &'a [&'a [CategoryGroup<W>]],
    candidates: &'a [RuleId],
}

struct GroupResolution<'a, W> {
    span: Span,
    parts: &'a [Span],
    permit_key: &'a [Option<Span>],
    groups: &'a [&'a CategoryGroup<W>],
    positions: &'a [usize],
    candidates: &'a [RuleId],
}

struct FactoredInsideResolution<'a, W> {
    parts: &'a [Span],
    product: &'a FactoredPairProduct,
    axes: [&'a [usize]; 2],
    groups: [&'a [CategoryGroup<W>]; 2],
}

fn factored_inside_axes<W>(
    parts: &[Span],
    positions: [&[usize]; 2],
    groups: [&[CategoryGroup<W>]; 2],
    output: &mut InsideRequests<'_>,
) -> Result<[Vec<(TagId, usize)>; 2], TableError> {
    let mut axes = [Vec::<(TagId, Vec<UnitSource>)>::new(), Vec::new()];
    let first = [
        positions[0].first().copied().unwrap_or(0),
        positions[1].first().copied().unwrap_or(0),
    ];
    for (axis, positions) in positions.into_iter().enumerate() {
        for position in positions {
            let Some(group) = groups[axis].get(*position) else {
                return Err(TableError::InvalidPair {
                    left: first[0],
                    left_categories: groups[0].len(),
                    right: first[1],
                    right_categories: groups[1].len(),
                });
            };
            for (tag, _) in &group.tagged {
                let tag_at = axes[axis]
                    .iter()
                    .position(|(held, _)| held == tag)
                    .unwrap_or_else(|| {
                        axes[axis].push((*tag, Vec::new()));
                        axes[axis].len() - 1
                    });
                axes[axis][tag_at].1.push(UnitSource {
                    span: parts[axis],
                    group: *position,
                    tag: *tag,
                });
            }
        }
    }
    let mut axis_ids = [Vec::new(), Vec::new()];
    for (held, identifiers) in axes.iter().zip(&mut axis_ids) {
        for (tag, sources) in held {
            let identifier = if let Some(identifier) = output.factor_axis_ids.get(sources) {
                *identifier
            } else {
                let identifier = output.factor_axes.len();
                output.factor_axis_ids.insert(sources.clone(), identifier);
                output.factor_axes.push(sources.clone());
                identifier
            };
            identifiers.push((*tag, identifier));
        }
    }
    Ok(axis_ids)
}

fn factored_inside_axis<W>(
    span: Span,
    positions: &[usize],
    tag: TagId,
    groups: &[CategoryGroup<W>],
    output: &mut InsideRequests<'_>,
) -> Result<Option<usize>, TableError> {
    let mut sources = Vec::new();
    for position in positions {
        let Some(group) = groups.get(*position) else {
            return Err(TableError::InvalidSourceGroup {
                span,
                group: *position,
                groups: groups.len(),
            });
        };
        if group.tagged.iter().any(|(held, _)| *held == tag) {
            sources.push(UnitSource {
                span,
                group: *position,
                tag,
            });
        }
    }
    if sources.is_empty() {
        return Ok(None);
    }
    if let Some(identifier) = output.factor_axis_ids.get(&sources) {
        return Ok(Some(*identifier));
    }
    let identifier = output.factor_axes.len();
    output.factor_axis_ids.insert(sources.clone(), identifier);
    output.factor_axes.push(sources);
    Ok(Some(identifier))
}

fn admitted_pair_rights(right: &[usize], admitted: &[bool]) -> Result<Vec<usize>, TableError> {
    if admitted.len() != right.len() {
        return Err(TableError::InvalidStepBatch {
            requests: right.len(),
            results: admitted.len(),
        });
    }
    Ok(right
        .iter()
        .zip(admitted)
        .filter_map(|(right, admitted)| admitted.then_some(*right))
        .collect())
}

fn admitted_pair_rectangles(
    by_rights: HashMap<Vec<usize>, Vec<usize>>,
) -> Vec<(Vec<usize>, Vec<usize>)> {
    let mut rectangles = by_rights
        .into_iter()
        .map(|(right, left)| (left, right))
        .collect::<Vec<_>>();
    rectangles.sort_unstable();
    rectangles
}

fn resolve_inside_rows<W, R, E>(
    requests: &[StepBatchRequest],
    splits: &[Option<usize>],
    step_rows: &[StepBatchEntry],
    resolver: &mut R,
) -> ResolvedInsideResult<W, R::Error, E>
where
    W: Clone,
    R: InsideResolver<W>,
{
    if requests.len() != splits.len() {
        return Err(SweepError::Table(TableError::InvalidStepBatch {
            requests: requests.len(),
            results: splits.len(),
        }));
    }
    let mut offsets = vec![0; requests.len() + 1];
    let mut steps = Vec::with_capacity(step_rows.len());
    let mut active = None;
    let mut locals: HashMap<(RuleId, CategoryId), W> = HashMap::new();

    for entry in step_rows {
        if active.is_some_and(|request| request > entry.request) {
            return Err(SweepError::Table(TableError::InvalidStepBatch {
                requests: requests.len(),
                results: entry.request,
            }));
        }
        if active != Some(entry.request) {
            active = Some(entry.request);
            locals.clear();
        }
        let Some(request) = requests.get(entry.request) else {
            return Err(SweepError::Table(TableError::InvalidStepBatch {
                requests: requests.len(),
                results: entry.request.saturating_add(1),
            }));
        };
        if entry.row >= request.tag_rows.len() {
            return Err(SweepError::Table(TableError::InvalidStepRow {
                request: entry.request,
                row: entry.row,
                rows: request.tag_rows.len(),
            }));
        }
        let local = match locals.entry((entry.step.rule, entry.step.output)) {
            Entry::Occupied(entry) => entry.get().clone(),
            Entry::Vacant(vacant) => {
                let weight = resolver
                    .local_weight(
                        entry.step.rule,
                        &request.inputs,
                        entry.step.output,
                        request.span,
                        splits[entry.request],
                    )
                    .map_err(SweepError::Resolver)?;
                vacant.insert(weight).clone()
            }
        };
        offsets[entry.request + 1] += 1;
        steps.push(ResolvedInsideStep {
            row: entry.row,
            step: entry.step,
            local,
        });
    }
    for request in 0..requests.len() {
        offsets[request + 1] += offsets[request];
    }
    Ok(ResolvedInsideRows { offsets, steps })
}

fn tag_tuples(axes: &[Vec<TagId>]) -> Vec<Vec<TagId>> {
    fn visit(axes: &[Vec<TagId>], at: usize, row: &mut Vec<TagId>, rows: &mut Vec<Vec<TagId>>) {
        if at == axes.len() {
            rows.push(row.clone());
            return;
        }
        for tag in &axes[at] {
            row.push(*tag);
            visit(axes, at + 1, row, rows);
            row.pop();
        }
    }

    let mut rows = Vec::new();
    visit(axes, 0, &mut Vec::with_capacity(axes.len()), &mut rows);
    rows
}

fn span_decompositions(span: Span, arity: usize) -> Vec<Vec<Span>> {
    fn collect(
        span: Span,
        remaining: usize,
        cursor: usize,
        parts: &mut Vec<Span>,
        held: &mut Vec<Vec<Span>>,
    ) {
        if remaining == 1 {
            parts.push((cursor, span.1));
            held.push(parts.clone());
            parts.pop();
            return;
        }

        let tail = remaining - 1;
        let last_cut = span.1.saturating_sub(tail);

        for cut in cursor + 1..=last_cut {
            parts.push((cursor, cut));
            collect(span, tail, cut, parts, held);
            parts.pop();
        }
    }

    if arity == 0 || span.1 - span.0 < arity {
        return Vec::new();
    }

    let mut held = Vec::new();
    collect(
        span,
        arity,
        span.0,
        &mut Vec::with_capacity(arity),
        &mut held,
    );
    held
}

/// One exact packed-forest item.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ForestKey {
    /// The item's half-open token span.
    pub span: Span,
    /// The item's exact category and normal-form tag.
    pub item: ItemId,
}

/// One packed derivation edge read from native storage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ForestEdge<L, W> {
    /// The interned edge label.
    pub label: L,
    /// Premise items in rule order.
    pub children: Vec<ForestKey>,
    /// The edge's exact conclusion category.
    pub output: CategoryId,
    /// Local weight excluding premise inside weights.
    pub weight: W,
    /// Binary split point, when the edge is binary.
    pub split: Option<usize>,
    /// Rule spine order.
    pub order: usize,
}

/// One exact Cartesian product of packed derivation edges.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProductAlternative<L, W> {
    /// The interned edge label shared by every represented edge.
    pub label: L,
    /// Exact premise-item alternatives in rule order.
    pub axes: Vec<Vec<ForestKey>>,
    /// The exact conclusion category shared by every represented edge.
    pub output: CategoryId,
    /// Local weight excluding premise inside weights.
    pub weight: W,
    /// Binary split point, when the represented edges are binary.
    pub split: Option<usize>,
    /// Rule spine order.
    pub order: usize,
}

impl<L, W> ProductAlternative<L, W> {
    fn edge_count(&self) -> Option<usize> {
        self.axes
            .iter()
            .try_fold(1_usize, |count, axis| count.checked_mul(axis.len()))
    }
}

#[derive(Clone, Debug)]
struct ForestItem<W> {
    key: ForestKey,
    inside: W,
    alternatives: Vec<usize>,
}

#[derive(Clone, Debug, Default)]
struct ForestCell {
    items: Vec<usize>,
}

#[derive(Clone, Debug)]
struct ForestCategoryGroup<W> {
    category: CategoryId,
    tagged: Vec<(ForestKey, W)>,
}

/// Native storage for a packed forest and its product alternatives.
#[derive(Clone, Debug)]
pub struct ForestTable<L, W> {
    tokens: usize,
    offsets: Vec<usize>,
    cells: Vec<ForestCell>,
    positions: HashMap<ForestKey, usize>,
    items: Vec<ForestItem<W>>,
    alternatives: Vec<ProductAlternative<L, W>>,
    edge_count: usize,
    truncated: bool,
}

impl<L: Clone, W: Clone> ForestTable<L, W> {
    /// Construct an empty forest for a sentence length.
    #[must_use]
    pub fn new(tokens: usize) -> Self {
        let (offsets, total) = span_offsets(tokens);
        Self {
            tokens,
            offsets,
            cells: (0..total).map(|_| ForestCell::default()).collect(),
            positions: HashMap::new(),
            items: Vec::new(),
            alternatives: Vec::new(),
            edge_count: 0,
            truncated: false,
        }
    }

    /// Return the sentence length owned by the forest.
    #[must_use]
    pub const fn tokens(&self) -> usize {
        self.tokens
    }

    /// Return the number of present items.
    #[must_use]
    pub fn num_items(&self) -> usize {
        self.positions.len()
    }

    /// Return the number of edges belonging to present items.
    #[must_use]
    pub const fn num_edges(&self) -> usize {
        self.edge_count
    }

    /// Return whether bounded construction omitted a derivation.
    #[must_use]
    pub const fn truncated(&self) -> bool {
        self.truncated
    }

    /// Record that bounded construction omitted a derivation.
    pub const fn mark_truncated(&mut self) {
        self.truncated = true;
    }

    fn span_id(&self, span: Span) -> Result<usize, TableError> {
        indexed_span(self.tokens, &self.offsets, span)
    }

    /// Return whether the forest holds one exact item.
    #[must_use]
    pub fn contains(&self, key: ForestKey) -> bool {
        self.positions.contains_key(&key)
    }

    /// Add one edge and its inside contribution.
    ///
    /// # Errors
    ///
    /// Returns a table error for an invalid conclusion span or the carrier's error.
    pub fn add<S>(
        &mut self,
        key: ForestKey,
        edge: ForestEdge<L, W>,
        semiring: &S,
    ) -> TableResult<(), S::Error>
    where
        S: Semiring<Weight = W>,
    {
        self.add_product(
            key,
            ProductAlternative {
                label: edge.label,
                axes: edge.children.into_iter().map(|child| vec![child]).collect(),
                output: edge.output,
                weight: edge.weight,
                split: edge.split,
                order: edge.order,
            },
            semiring,
        )
    }

    /// Add one exact Cartesian product of edges and its inside contribution.
    ///
    /// Repeated items within one premise axis denote one choice and are retained
    /// once in first-arrival order. Repeated alternatives remain distinct edge
    /// occurrences.
    ///
    /// # Errors
    ///
    /// Returns a table error for an invalid conclusion span or an edge-count
    /// overflow, or the carrier's arithmetic error.
    pub fn add_product<S>(
        &mut self,
        key: ForestKey,
        mut alternative: ProductAlternative<L, W>,
        semiring: &S,
    ) -> TableResult<(), S::Error>
    where
        S: Semiring<Weight = W>,
    {
        let cell = self.span_id(key.span).map_err(SweepError::Table)?;
        for axis in &mut alternative.axes {
            let mut seen = HashSet::new();
            axis.retain(|child| seen.insert(*child));
        }
        let edges = alternative
            .edge_count()
            .ok_or(SweepError::Table(TableError::EdgeCountOverflow))?;
        if edges == 0 {
            return Ok(());
        }
        self.edge_count
            .checked_add(edges)
            .ok_or(SweepError::Table(TableError::EdgeCountOverflow))?;
        let contribution = self
            .product_contribution(&alternative, semiring)
            .map_err(SweepError::Semiring)?;

        let position = if let Some(position) = self.positions.get(&key).copied() {
            semiring
                .plus_assign(&mut self.items[position].inside, &contribution)
                .map_err(SweepError::Semiring)?;
            position
        } else {
            let position = self.items.len();
            self.positions.insert(key, position);
            self.items.push(ForestItem {
                key,
                inside: contribution,
                alternatives: Vec::new(),
            });
            self.cells[cell].items.push(position);
            position
        };

        self.store_product(position, alternative)
            .map_err(SweepError::Table)?;
        Ok(())
    }

    fn store_product(
        &mut self,
        position: usize,
        alternative: ProductAlternative<L, W>,
    ) -> Result<(), TableError> {
        let edges = alternative
            .edge_count()
            .ok_or(TableError::EdgeCountOverflow)?;
        let next_edges = self
            .edge_count
            .checked_add(edges)
            .ok_or(TableError::EdgeCountOverflow)?;
        let alternative_position = self.alternatives.len();
        self.alternatives.push(alternative);
        self.items[position].alternatives.push(alternative_position);
        self.edge_count = next_edges;
        Ok(())
    }

    fn product_contribution<S>(
        &self,
        alternative: &ProductAlternative<L, W>,
        semiring: &S,
    ) -> Result<W, S::Error>
    where
        S: Semiring<Weight = W>,
    {
        let mut contribution = alternative.weight.clone();
        for axis in &alternative.axes {
            let mut children = axis.iter();
            let Some(first) = children.next() else {
                return Ok(semiring.zero());
            };
            let mut total = self.inside(*first, semiring);
            for child in children {
                semiring.plus_assign(&mut total, &self.inside(*child, semiring))?;
            }
            contribution = semiring.times(&contribution, &total)?;
        }
        Ok(contribution)
    }

    /// Return items covering one span in arrival order.
    ///
    /// # Errors
    ///
    /// Returns a table error when the span is invalid.
    pub fn items_at(&self, span: Span) -> Result<Vec<ForestKey>, TableError> {
        let cell = self.span_id(span)?;
        Ok(self.cells[cell]
            .items
            .iter()
            .map(|position| self.items[*position].key)
            .collect())
    }

    /// Return one item's edges in insertion order.
    ///
    /// # Errors
    ///
    /// Returns a table error when the item is absent.
    pub fn edges(&self, key: ForestKey) -> Result<Vec<ForestEdge<L, W>>, TableError> {
        let position = self
            .positions
            .get(&key)
            .copied()
            .ok_or(TableError::UnknownItem { key })?;
        let mut edges = Vec::new();
        for alternative in &self.items[position].alternatives {
            Self::expand_product(
                &self.alternatives[*alternative],
                0,
                &mut Vec::new(),
                &mut edges,
            );
        }
        Ok(edges)
    }

    /// Return one item's exact Cartesian edge products in insertion order.
    ///
    /// # Errors
    ///
    /// Returns a table error when the item is absent.
    pub fn products(&self, key: ForestKey) -> Result<Vec<ProductAlternative<L, W>>, TableError> {
        let position = self
            .positions
            .get(&key)
            .copied()
            .ok_or(TableError::UnknownItem { key })?;
        Ok(self.items[position]
            .alternatives
            .iter()
            .map(|alternative| self.alternatives[*alternative].clone())
            .collect())
    }

    fn expand_product(
        alternative: &ProductAlternative<L, W>,
        at: usize,
        children: &mut Vec<ForestKey>,
        edges: &mut Vec<ForestEdge<L, W>>,
    ) {
        if at == alternative.axes.len() {
            edges.push(ForestEdge {
                label: alternative.label.clone(),
                children: children.clone(),
                output: alternative.output,
                weight: alternative.weight.clone(),
                split: alternative.split,
                order: alternative.order,
            });
            return;
        }
        for child in &alternative.axes[at] {
            children.push(*child);
            Self::expand_product(alternative, at + 1, children, edges);
            children.pop();
        }
    }

    /// Return one item's inside weight, or zero when absent.
    pub fn inside<S>(&self, key: ForestKey, semiring: &S) -> W
    where
        S: Semiring<Weight = W>,
    {
        self.positions
            .get(&key)
            .and_then(|position| self.items.get(*position))
            .map_or_else(|| semiring.zero(), |item| item.inside.clone())
    }

    /// Return category weights summed over exact tags in first-arrival order.
    ///
    /// # Errors
    ///
    /// Returns a table or carrier error.
    pub fn categories_at<S>(
        &self,
        span: Span,
        semiring: &S,
    ) -> TableResult<Vec<(CategoryId, W)>, S::Error>
    where
        S: Semiring<Weight = W>,
    {
        let groups = self.groups(span).map_err(SweepError::Table)?;
        let mut categories = Vec::with_capacity(groups.len());

        for group in groups {
            let mut total = semiring.zero();

            for (_, weight) in group.tagged {
                semiring
                    .plus_assign(&mut total, &weight)
                    .map_err(SweepError::Semiring)?;
            }
            categories.push((group.category, total));
        }
        Ok(categories)
    }

    /// Recompute item weights in the supplied dependency order.
    ///
    /// # Errors
    ///
    /// Returns the carrier's arithmetic error.
    pub fn recompute_inside<S>(&mut self, keys: &[ForestKey], semiring: &S) -> Result<(), S::Error>
    where
        S: Semiring<Weight = W>,
    {
        for key in keys {
            let Some(position) = self.positions.get(key).copied() else {
                continue;
            };
            let alternatives = self.items[position].alternatives.clone();
            let mut total = semiring.zero();

            for alternative in alternatives {
                let contribution =
                    self.product_contribution(&self.alternatives[alternative], semiring)?;
                semiring.plus_assign(&mut total, &contribution)?;
            }
            self.items[position].inside = total;
        }
        Ok(())
    }

    /// Restrict one cell and settle edges whose premises were removed.
    ///
    /// # Errors
    ///
    /// Returns a table error for an invalid span or unknown kept item, or a
    /// carrier error while recomputing surviving inside weights.
    pub fn restrict<S>(
        &mut self,
        span: Span,
        keep: &[ForestKey],
        semiring: &S,
        truncates: bool,
    ) -> TableResult<(), S::Error>
    where
        S: Semiring<Weight = W>,
    {
        let cell = self.span_id(span).map_err(SweepError::Table)?;
        let present = self.cells[cell].items.clone();

        if present.is_empty() {
            return Ok(());
        }

        let present_keys: HashSet<ForestKey> = present
            .iter()
            .map(|position| self.items[*position].key)
            .collect();
        let kept: HashSet<ForestKey> = keep.iter().copied().collect();

        if let Some(key) = kept.difference(&present_keys).next().copied() {
            return Err(SweepError::Table(TableError::ForeignKeep { span, key }));
        }

        if kept.len() == present_keys.len() {
            return Ok(());
        }

        let mut surviving = Vec::new();

        for position in present {
            let key = self.items[position].key;

            if kept.contains(&key) {
                surviving.push(position);
            } else {
                self.positions.remove(&key);
                let removed = self.items[position]
                    .alternatives
                    .iter()
                    .try_fold(0_usize, |count, alternative| {
                        let edges = self.alternatives[*alternative]
                            .edge_count()
                            .ok_or(TableError::EdgeCountOverflow)?;
                        count
                            .checked_add(edges)
                            .ok_or(TableError::EdgeCountOverflow)
                    })
                    .map_err(SweepError::Table)?;
                self.edge_count -= removed;
                self.items[position].alternatives.clear();
            }
        }

        loop {
            let before = surviving.len();
            let mut following = Vec::new();

            for position in surviving {
                let alternatives = self.items[position].alternatives.clone();
                let mut retained = Vec::new();
                for alternative in alternatives {
                    let positions = &self.positions;
                    let held = &mut self.alternatives[alternative];
                    let before = held
                        .edge_count()
                        .ok_or(SweepError::Table(TableError::EdgeCountOverflow))?;
                    for axis in &mut held.axes {
                        axis.retain(|child| positions.contains_key(child));
                    }
                    let after = held
                        .edge_count()
                        .ok_or(SweepError::Table(TableError::EdgeCountOverflow))?;
                    self.edge_count -= before - after;
                    if after > 0 {
                        retained.push(alternative);
                    }
                }
                self.items[position].alternatives = retained;

                if self.items[position].alternatives.is_empty() {
                    self.positions.remove(&self.items[position].key);
                } else {
                    following.push(position);
                }
            }

            if following.len() == before {
                surviving = following;
                break;
            }
            surviving = following;
        }

        self.cells[cell].items = surviving;
        let keys: Vec<ForestKey> = self.cells[cell]
            .items
            .iter()
            .map(|position| self.items[*position].key)
            .collect();
        self.recompute_inside(&keys, semiring)
            .map_err(SweepError::Semiring)?;
        self.truncated |= truncates;
        Ok(())
    }

    fn groups(&self, span: Span) -> Result<Vec<ForestCategoryGroup<W>>, TableError> {
        let cell = self.span_id(span)?;
        let mut groups: Vec<ForestCategoryGroup<W>> = Vec::new();
        let mut positions: HashMap<CategoryId, usize> = HashMap::new();

        for item_position in &self.cells[cell].items {
            let item = &self.items[*item_position];
            let position = *positions.entry(item.key.item.category).or_insert_with(|| {
                let position = groups.len();
                groups.push(ForestCategoryGroup {
                    category: item.key.item.category,
                    tagged: Vec::new(),
                });
                position
            });
            groups[position]
                .tagged
                .push((item.key, item.inside.clone()));
        }
        Ok(groups)
    }

    fn weighted_items(&self, span: Span) -> Result<Vec<(ItemId, W)>, TableError> {
        let cell = self.span_id(span)?;
        Ok(self.cells[cell]
            .items
            .iter()
            .map(|position| {
                let item = &self.items[*position];
                (item.key.item, item.inside.clone())
            })
            .collect())
    }
}

/// Drawing, scoring and per-cell restriction callbacks for a forest sweep.
pub trait ForestResolver<W, L>: InsideResolver<W> {
    /// Return the stored label and spine order for one rule.
    ///
    /// # Errors
    ///
    /// Returns the resolver's metadata error.
    fn label_order(&mut self, rule: RuleId) -> Result<(L, usize), Self::Error>;

    /// Restrict the root cell after unary closure, when requested.
    ///
    /// # Errors
    ///
    /// Returns the resolver's root-matching error.
    fn root_keep(
        &mut self,
        span: Span,
        items: &[(ItemId, W)],
        closure: &UnaryClosure,
    ) -> Result<Option<Vec<ItemId>>, Self::Error>;

    /// Restrict one closed cell through the declared pruner, when requested.
    ///
    /// # Errors
    ///
    /// Returns the resolver's pruning error.
    fn prune_keep(
        &mut self,
        span: Span,
        items: &[(ItemId, W)],
    ) -> Result<Option<Vec<ItemId>>, Self::Error>;
}

#[derive(Clone)]
struct ForestLocal<L, W> {
    label: L,
    order: usize,
    weight: W,
}

struct FactoredForestRectangle<'a, W> {
    span: Span,
    split: usize,
    product: &'a FactoredPairProduct,
    tag: TagId,
    inputs: [CategoryId; 2],
    sources: [&'a [(ForestKey, W)]; 2],
}

struct ForestCategoryProduct<'a, W> {
    span: Span,
    parts: &'a [Span],
    groups: &'a [Vec<ForestCategoryGroup<W>>],
    candidates: &'a [RuleId],
}

type ForestFactorAxis<W> = Vec<(TagId, Vec<(ForestKey, W)>)>;

fn factored_forest_axes<W: Clone>(
    positions: [&[usize]; 2],
    groups: [&[ForestCategoryGroup<W>]; 2],
) -> Result<[ForestFactorAxis<W>; 2], TableError> {
    let mut axes = [ForestFactorAxis::new(), ForestFactorAxis::new()];
    let first = [
        positions[0].first().copied().unwrap_or(0),
        positions[1].first().copied().unwrap_or(0),
    ];
    for (axis, positions) in positions.into_iter().enumerate() {
        for position in positions {
            let Some(group) = groups[axis].get(*position) else {
                return Err(TableError::InvalidPair {
                    left: first[0],
                    left_categories: groups[0].len(),
                    right: first[1],
                    right_categories: groups[1].len(),
                });
            };
            for (key, weight) in &group.tagged {
                let tag = key.item.tag;
                let tag_at = axes[axis]
                    .iter()
                    .position(|(held, _)| *held == tag)
                    .unwrap_or_else(|| {
                        axes[axis].push((tag, Vec::new()));
                        axes[axis].len() - 1
                    });
                axes[axis][tag_at].1.push((*key, weight.clone()));
            }
        }
    }
    Ok(axes)
}

fn factored_forest_axis<W: Clone>(
    positions: &[usize],
    tag: TagId,
    groups: &[ForestCategoryGroup<W>],
) -> Result<Vec<(ForestKey, W)>, TableError> {
    let mut sources = Vec::new();
    for position in positions {
        let Some(group) = groups.get(*position) else {
            return Err(TableError::InvalidPair {
                left: *position,
                left_categories: groups.len(),
                right: *position,
                right_categories: groups.len(),
            });
        };
        if let Some((key, weight)) = group.tagged.iter().find(|(key, _)| key.item.tag == tag) {
            sources.push((*key, weight.clone()));
        }
    }
    Ok(sources)
}

struct ForestTagProduct<'a, W> {
    span: Span,
    parts: &'a [Span],
    split: Option<usize>,
    groups: &'a [&'a ForestCategoryGroup<W>],
    inputs: &'a [CategoryId],
    candidates: &'a [RuleId],
}

struct ForestTagState<L, W> {
    tags: Vec<TagId>,
    children: Vec<ForestKey>,
    below: Vec<W>,
    locals: HashMap<(RuleId, CategoryId), ForestLocal<L, W>>,
}

#[derive(Clone)]
struct ForestEmission<L, W> {
    item: ItemId,
    alternative: ProductAlternative<L, W>,
    below: Vec<Vec<W>>,
}

struct BuiltForestItem<L, W> {
    item: ItemId,
    inside: W,
    alternatives: Vec<ProductAlternative<L, W>>,
}

struct BuiltForestCell<L, W> {
    items: Vec<BuiltForestItem<L, W>>,
    positions: HashMap<ItemId, usize>,
}

type ForestBuild<L, W, E> = (Span, Result<BuiltForestCell<L, W>, E>);

fn accumulate_forest_cell<L, W, S>(
    emissions: Vec<ForestEmission<L, W>>,
    semiring: &S,
) -> Result<BuiltForestCell<L, W>, S::Error>
where
    L: Clone,
    W: Clone,
    S: Semiring<Weight = W>,
{
    let mut built = BuiltForestCell {
        items: Vec::new(),
        positions: HashMap::new(),
    };

    for emission in emissions {
        let mut contribution = emission.alternative.weight.clone();

        for axis in &emission.below {
            let mut weights = axis.iter();
            let Some(first) = weights.next() else {
                contribution = semiring.zero();
                break;
            };
            let mut total = first.clone();
            for weight in weights {
                semiring.plus_assign(&mut total, weight)?;
            }
            contribution = semiring.times(&contribution, &total)?;
        }

        if let Some(position) = built.positions.get(&emission.item).copied() {
            let item = &mut built.items[position];
            semiring.plus_assign(&mut item.inside, &contribution)?;
            item.alternatives.push(emission.alternative);
        } else {
            built.positions.insert(emission.item, built.items.len());
            built.items.push(BuiltForestItem {
                item: emission.item,
                inside: contribution,
                alternatives: vec![emission.alternative],
            });
        }
    }
    Ok(built)
}

impl<L: Clone, W: Clone> ForestTable<L, W> {
    fn merge_built_cell<S>(
        &mut self,
        span: Span,
        built: BuiltForestCell<L, W>,
        semiring: &S,
    ) -> TableResult<(), S::Error>
    where
        S: Semiring<Weight = W>,
    {
        let cell = self.span_id(span).map_err(SweepError::Table)?;
        let added = built
            .items
            .iter()
            .flat_map(|item| &item.alternatives)
            .try_fold(0_usize, |count, alternative| {
                let edges = alternative
                    .edge_count()
                    .ok_or(TableError::EdgeCountOverflow)?;
                count
                    .checked_add(edges)
                    .ok_or(TableError::EdgeCountOverflow)
            })
            .map_err(SweepError::Table)?;
        self.edge_count
            .checked_add(added)
            .ok_or(SweepError::Table(TableError::EdgeCountOverflow))?;

        for built_item in built.items {
            let key = ForestKey {
                span,
                item: built_item.item,
            };
            let position = if let Some(position) = self.positions.get(&key).copied() {
                semiring
                    .plus_assign(&mut self.items[position].inside, &built_item.inside)
                    .map_err(SweepError::Semiring)?;
                position
            } else {
                let position = self.items.len();
                self.positions.insert(key, position);
                self.items.push(ForestItem {
                    key,
                    inside: built_item.inside,
                    alternatives: Vec::new(),
                });
                self.cells[cell].items.push(position);
                position
            };

            for alternative in built_item.alternatives {
                self.store_product(position, alternative)
                    .map_err(SweepError::Table)?;
            }
        }
        Ok(())
    }

    /// Fill every cell by width and apply the resolver's per-cell restrictions.
    ///
    /// # Errors
    ///
    /// Returns a resolver, table or carrier error.
    pub fn sweep<S, R>(
        &mut self,
        active_arities: &[usize],
        unary_depth: usize,
        execution: CellExecution,
        semiring: &S,
        resolver: &mut R,
    ) -> Result<(), SweepError<R::Error, S::Error>>
    where
        L: Send + Sync,
        W: Send + Sync,
        S: Semiring<Weight = W> + Sync,
        S::Error: Send,
        R: ForestResolver<W, L>,
    {
        let pool = worker_pool(execution).map_err(SweepError::Table)?;

        for width in 1..=self.tokens {
            let spans: Vec<Span> = (0..=self.tokens - width)
                .map(|start| (start, start + width))
                .collect();
            let plans: Vec<CellPlan> = pool.as_ref().map_or_else(
                || {
                    spans
                        .iter()
                        .map(|span| cell_plan(*span, active_arities))
                        .collect()
                },
                |workers| {
                    workers.install(|| {
                        spans
                            .par_iter()
                            .map(|span| cell_plan(*span, active_arities))
                            .collect()
                    })
                },
            );
            let mut work = Vec::with_capacity(plans.len());

            for plan in &plans {
                let mut emissions = Vec::new();

                for parts in &plan.decompositions {
                    self.resolve_forest_decomposition(plan.span, parts, resolver, &mut emissions)?;
                }
                work.push((plan.span, emissions));
            }

            let built: Vec<ForestBuild<L, W, S::Error>> = match &pool {
                None => work
                    .into_iter()
                    .map(|(span, emissions)| (span, accumulate_forest_cell(emissions, semiring)))
                    .collect(),
                Some(workers) => workers.install(|| {
                    work.into_par_iter()
                        .map(|(span, emissions)| {
                            (span, accumulate_forest_cell(emissions, semiring))
                        })
                        .collect()
                }),
            };

            for (span, cell) in built {
                self.merge_built_cell(span, cell.map_err(SweepError::Semiring)?, semiring)
                    .map_err(lift_table_error)?;
            }

            for span in spans {
                let arrival: Vec<ItemId> = self
                    .items_at(span)
                    .map_err(SweepError::Table)?
                    .into_iter()
                    .map(|key| key.item)
                    .collect();
                let closure = resolver
                    .closure(&arrival, span, unary_depth, pool.as_ref())
                    .map_err(SweepError::Resolver)?;
                self.close_forest(span, &closure, semiring, resolver)?;

                let items = self.weighted_items(span).map_err(SweepError::Table)?;

                if let Some(keep) = resolver
                    .root_keep(span, &items, &closure)
                    .map_err(SweepError::Resolver)?
                {
                    self.restrict_items(span, &keep, semiring, false)?;
                }

                let items = self.weighted_items(span).map_err(SweepError::Table)?;

                if let Some(keep) = resolver
                    .prune_keep(span, &items)
                    .map_err(SweepError::Resolver)?
                {
                    self.restrict_items(span, &keep, semiring, true)?;
                }
            }
            resolver.finish_width().map_err(SweepError::Resolver)?;
        }
        Ok(())
    }

    fn restrict_items<S, R>(
        &mut self,
        span: Span,
        keep: &[ItemId],
        semiring: &S,
        truncates: bool,
    ) -> Result<(), SweepError<R, S::Error>>
    where
        S: Semiring<Weight = W>,
    {
        let keys: Vec<ForestKey> = keep
            .iter()
            .copied()
            .map(|item| ForestKey { span, item })
            .collect();
        self.restrict(span, &keys, semiring, truncates)
            .map_err(lift_table_error)
    }

    fn resolve_forest_decomposition<R, E>(
        &self,
        span: Span,
        parts: &[Span],
        resolver: &mut R,
        emissions: &mut Vec<ForestEmission<L, W>>,
    ) -> Result<(), SweepError<R::Error, E>>
    where
        R: ForestResolver<W, L>,
    {
        let groups = parts
            .iter()
            .map(|part| self.groups(*part).map_err(SweepError::Table))
            .collect::<Result<Vec<_>, _>>()?;

        if groups.iter().any(Vec::is_empty) {
            return Ok(());
        }

        if groups.len() == 2 {
            let left: Vec<CategoryId> = groups[0].iter().map(|group| group.category).collect();
            let right: Vec<CategoryId> = groups[1].iter().map(|group| group.category).collect();
            let batches = resolver
                .pair_batches(
                    &[PairBatchRequest {
                        span,
                        parts: [parts[0], parts[1]],
                        left,
                        right,
                    }],
                    None,
                    resolver.local_weights_are_unit(),
                )
                .map_err(SweepError::Resolver)?;
            let batch = batches.into_iter().next().ok_or_else(|| {
                SweepError::Table(TableError::InvalidStepBatch {
                    requests: 1,
                    results: 0,
                })
            })?;

            for pair in &batch.pairs {
                let Some(left_group) = groups[0].get(pair.left) else {
                    return Err(SweepError::Table(TableError::InvalidPair {
                        left: pair.left,
                        left_categories: groups[0].len(),
                        right: pair.right,
                        right_categories: groups[1].len(),
                    }));
                };
                let Some(right_group) = groups[1].get(pair.right) else {
                    return Err(SweepError::Table(TableError::InvalidPair {
                        left: pair.left,
                        left_categories: groups[0].len(),
                        right: pair.right,
                        right_categories: groups[1].len(),
                    }));
                };
                Self::resolve_forest_groups(
                    span,
                    parts,
                    &[left_group, right_group],
                    &pair.candidates,
                    resolver,
                    emissions,
                )?;
            }
            for (product_at, product) in batch.products.iter().enumerate() {
                let positions = batch
                    .product_axes_at(product_at)
                    .map_err(|_| SweepError::Table(TableError::InvalidFactoredBatch))?;
                Self::resolve_factored_forest_product(
                    span,
                    parts,
                    product,
                    positions,
                    [&groups[0], &groups[1]],
                    resolver,
                    emissions,
                )?;
            }
        } else {
            let candidates = resolver
                .rules_of_arity(groups.len())
                .map_err(SweepError::Resolver)?;
            let product = ForestCategoryProduct {
                span,
                parts,
                groups: &groups,
                candidates: &candidates,
            };
            Self::forest_category_product(&product, 0, &mut Vec::new(), resolver, emissions)?;
        }
        Ok(())
    }

    fn resolve_factored_forest_product<R, E>(
        span: Span,
        parts: &[Span],
        product: &FactoredPairProduct,
        positions: [&[usize]; 2],
        groups: [&[ForestCategoryGroup<W>]; 2],
        resolver: &mut R,
        emissions: &mut Vec<ForestEmission<L, W>>,
    ) -> Result<(), SweepError<R::Error, E>>
    where
        R: ForestResolver<W, L>,
    {
        let axes = factored_forest_axes(positions, groups).map_err(SweepError::Table)?;
        for (left_tag, left_sources) in &axes[0] {
            for (right_tag, right_sources) in &axes[1] {
                let tags = [*left_tag, *right_tag];
                match resolver
                    .decision(product.rule, &tags)
                    .map_err(SweepError::Resolver)?
                {
                    NormalDecision::Admit(tag) => {
                        let inputs = [
                            groups[0][positions[0][0]].category,
                            groups[1][positions[1][0]].category,
                        ];
                        Self::emit_factored_forest_rectangle(
                            &FactoredForestRectangle {
                                span,
                                split: parts[0].1,
                                product,
                                tag,
                                inputs,
                                sources: [left_sources, right_sources],
                            },
                            resolver,
                            emissions,
                        )?;
                    }
                    NormalDecision::Premises(tag) => {
                        let [left_positions, right_positions] = [0, 1].map(|axis| {
                            positions[axis]
                                .iter()
                                .copied()
                                .filter(|position| {
                                    groups[axis][*position]
                                        .tagged
                                        .iter()
                                        .any(|(key, _)| key.item.tag == tags[axis])
                                })
                                .collect::<Vec<_>>()
                        });
                        if left_positions.is_empty() || right_positions.is_empty() {
                            continue;
                        }
                        let mut by_rights = HashMap::<Vec<usize>, Vec<usize>>::new();
                        for left in left_positions {
                            let input_pairs = right_positions
                                .iter()
                                .map(|right| [groups[0][left].category, groups[1][*right].category])
                                .collect::<Vec<_>>();
                            let admitted = resolver
                                .admit_rows(product.rule, &tags, &input_pairs, product.output)
                                .map_err(SweepError::Resolver)?;
                            let admitted = admitted_pair_rights(&right_positions, &admitted)
                                .map_err(SweepError::Table)?;
                            if !admitted.is_empty() {
                                by_rights.entry(admitted).or_default().push(left);
                            }
                        }
                        let rectangles = admitted_pair_rectangles(by_rights);
                        for (left, right) in rectangles {
                            let left_sources = factored_forest_axis(&left, *left_tag, groups[0])
                                .map_err(SweepError::Table)?;
                            let right_sources = factored_forest_axis(&right, *right_tag, groups[1])
                                .map_err(SweepError::Table)?;
                            let inputs =
                                [groups[0][left[0]].category, groups[1][right[0]].category];
                            Self::emit_factored_forest_rectangle(
                                &FactoredForestRectangle {
                                    span,
                                    split: parts[0].1,
                                    product,
                                    tag,
                                    inputs,
                                    sources: [&left_sources, &right_sources],
                                },
                                resolver,
                                emissions,
                            )?;
                        }
                    }
                    NormalDecision::Refuse => {}
                }
            }
        }
        Ok(())
    }

    fn emit_factored_forest_rectangle<R, E>(
        rectangle: &FactoredForestRectangle<'_, W>,
        resolver: &mut R,
        emissions: &mut Vec<ForestEmission<L, W>>,
    ) -> Result<(), SweepError<R::Error, E>>
    where
        R: ForestResolver<W, L>,
    {
        let FactoredForestRectangle {
            span,
            split,
            product,
            tag,
            inputs,
            sources: [left, right],
        } = *rectangle;
        let weight = resolver
            .local_weight(product.rule, &inputs, product.output, span, Some(split))
            .map_err(SweepError::Resolver)?;
        let (label, order) = resolver
            .label_order(product.rule)
            .map_err(SweepError::Resolver)?;
        emissions.push(ForestEmission {
            item: ItemId {
                category: product.output,
                tag,
            },
            alternative: ProductAlternative {
                label,
                axes: vec![
                    left.iter().map(|(key, _)| *key).collect(),
                    right.iter().map(|(key, _)| *key).collect(),
                ],
                output: product.output,
                weight,
                split: Some(split),
                order,
            },
            below: vec![
                left.iter().map(|(_, weight)| weight.clone()).collect(),
                right.iter().map(|(_, weight)| weight.clone()).collect(),
            ],
        });
        Ok(())
    }

    fn forest_category_product<R, E>(
        product: &ForestCategoryProduct<'_, W>,
        at: usize,
        selected: &mut Vec<usize>,
        resolver: &mut R,
        emissions: &mut Vec<ForestEmission<L, W>>,
    ) -> Result<(), SweepError<R::Error, E>>
    where
        R: ForestResolver<W, L>,
    {
        if at == product.groups.len() {
            let chosen: Vec<&ForestCategoryGroup<W>> = selected
                .iter()
                .enumerate()
                .map(|(part, position)| &product.groups[part][*position])
                .collect();
            return Self::resolve_forest_groups(
                product.span,
                product.parts,
                &chosen,
                product.candidates,
                resolver,
                emissions,
            );
        }

        for position in 0..product.groups[at].len() {
            selected.push(position);
            Self::forest_category_product(product, at + 1, selected, resolver, emissions)?;
            selected.pop();
        }
        Ok(())
    }

    fn resolve_forest_groups<R, E>(
        span: Span,
        parts: &[Span],
        groups: &[&ForestCategoryGroup<W>],
        candidates: &[RuleId],
        resolver: &mut R,
        emissions: &mut Vec<ForestEmission<L, W>>,
    ) -> Result<(), SweepError<R::Error, E>>
    where
        R: ForestResolver<W, L>,
    {
        let inputs: Vec<CategoryId> = groups.iter().map(|group| group.category).collect();
        let product = ForestTagProduct {
            span,
            parts,
            split: (parts.len() == 2).then_some(parts[0].1),
            groups,
            inputs: &inputs,
            candidates,
        };
        let mut state = ForestTagState {
            tags: Vec::with_capacity(groups.len()),
            children: Vec::with_capacity(groups.len()),
            below: Vec::with_capacity(groups.len()),
            locals: HashMap::new(),
        };
        Self::forest_tag_product(&product, 0, &mut state, resolver, emissions)
    }

    fn forest_tag_product<R, E>(
        product: &ForestTagProduct<'_, W>,
        at: usize,
        state: &mut ForestTagState<L, W>,
        resolver: &mut R,
        emissions: &mut Vec<ForestEmission<L, W>>,
    ) -> Result<(), SweepError<R::Error, E>>
    where
        R: ForestResolver<W, L>,
    {
        if at == product.groups.len() {
            let batch_request = StepBatchRequest {
                inputs: product.inputs.to_vec(),
                tag_rows: vec![state.tags.clone()],
                span: product.span,
                parts: product.parts.to_vec(),
                candidates: product.candidates.to_vec(),
            };
            let rows = resolver
                .steps(std::slice::from_ref(&batch_request), None)
                .map_err(SweepError::Resolver)?;
            let steps = rows
                .into_iter()
                .filter(|entry| entry.request == 0 && entry.row == 0)
                .map(|entry| entry.step);

            for step in steps {
                let local = match state.locals.entry((step.rule, step.output)) {
                    Entry::Occupied(entry) => entry.get().clone(),
                    Entry::Vacant(entry) => {
                        let weight = resolver
                            .local_weight(
                                step.rule,
                                product.inputs,
                                step.output,
                                product.span,
                                product.split,
                            )
                            .map_err(SweepError::Resolver)?;
                        let (label, order) = resolver
                            .label_order(step.rule)
                            .map_err(SweepError::Resolver)?;
                        entry
                            .insert(ForestLocal {
                                label,
                                order,
                                weight,
                            })
                            .clone()
                    }
                };
                emissions.push(ForestEmission {
                    item: ItemId {
                        category: step.output,
                        tag: step.tag,
                    },
                    alternative: ProductAlternative {
                        label: local.label,
                        axes: state
                            .children
                            .iter()
                            .copied()
                            .map(|child| vec![child])
                            .collect(),
                        output: step.output,
                        weight: local.weight,
                        split: product.split,
                        order: local.order,
                    },
                    below: state
                        .below
                        .iter()
                        .cloned()
                        .map(|weight| vec![weight])
                        .collect(),
                });
            }
            return Ok(());
        }

        for (key, weight) in &product.groups[at].tagged {
            state.tags.push(key.item.tag);
            state.children.push(*key);
            state.below.push(weight.clone());
            Self::forest_tag_product(product, at + 1, state, resolver, emissions)?;
            state.below.pop();
            state.children.pop();
            state.tags.pop();
        }
        Ok(())
    }

    /// Apply one exact unary closure to a populated forest cell.
    ///
    /// # Errors
    ///
    /// Returns a resolver, table, or carrier error.
    pub fn close_forest<S, R>(
        &mut self,
        span: Span,
        closure: &UnaryClosure,
        semiring: &S,
        resolver: &mut R,
    ) -> Result<(), SweepError<R::Error, S::Error>>
    where
        S: Semiring<Weight = W>,
        R: ForestResolver<W, L>,
    {
        for (child, step) in &closure.edges {
            let weight = resolver
                .local_weight(step.rule, &[child.category], step.output, span, None)
                .map_err(SweepError::Resolver)?;
            let (label, order) = resolver
                .label_order(step.rule)
                .map_err(SweepError::Resolver)?;
            self.add(
                ForestKey {
                    span,
                    item: ItemId {
                        category: step.output,
                        tag: step.tag,
                    },
                },
                ForestEdge {
                    label,
                    children: vec![ForestKey { span, item: *child }],
                    output: step.output,
                    weight,
                    split: None,
                    order,
                },
                semiring,
            )
            .map_err(lift_table_error)?;
        }
        let concluded: Vec<ForestKey> = closure
            .concluded
            .iter()
            .copied()
            .map(|item| ForestKey { span, item })
            .collect();
        self.recompute_inside(&concluded, semiring)
            .map_err(SweepError::Semiring)?;
        self.truncated |= closure.truncated;
        Ok(())
    }
}

/// Unary closure in dense ids, parameterized by the one rule resolver.
///
/// # Errors
///
/// Returns the error produced by the supplied resolver.
pub fn unary_closure<E, F>(arrival: &[ItemId], depth: usize, mut draw: F) -> Result<UnaryClosure, E>
where
    F: FnMut(&[ItemId]) -> Result<Vec<Vec<Step>>, E>,
{
    let mut canonical_arrival = arrival.to_vec();
    canonical_arrival.sort_unstable_by_key(|item| (item.category, item.tag));
    let mut edges = Vec::new();
    let mut category_graph: HashMap<CategoryId, HashSet<CategoryId>> = HashMap::new();
    let mut visited = HashSet::new();
    let mut scheduled: HashSet<ItemId> = canonical_arrival.iter().copied().collect();
    let mut order = canonical_arrival.clone();
    let mut frontier = canonical_arrival;
    let mut truncated = false;

    for _ in 0..depth {
        let mut following = Vec::new();
        let children = frontier
            .into_iter()
            .filter(|child| visited.insert(*child))
            .collect::<Vec<_>>();
        let rows = draw(&children)?;

        for (child, steps) in children.into_iter().zip(rows) {
            for step in steps {
                let reaches = reaches(&category_graph, step.output, child.category);

                if reaches {
                    truncated = true;
                    continue;
                }

                let output = ItemId {
                    category: step.output,
                    tag: step.tag,
                };
                edges.push((child, step));
                category_graph
                    .entry(child.category)
                    .or_default()
                    .insert(step.output);

                if scheduled.insert(output) {
                    following.push(output);
                    order.push(output);
                }
            }
        }

        frontier = following;

        if frontier.is_empty() {
            break;
        }
    }

    if !frontier.is_empty() {
        for steps in draw(&frontier)? {
            if !steps.is_empty() {
                truncated = true;
                break;
            }
        }
    }

    let concluded: HashSet<ItemId> = edges
        .iter()
        .map(|(_, step)| ItemId {
            category: step.output,
            tag: step.tag,
        })
        .collect();
    let dependency = dependency_order(&edges, &order);
    let ordered = dependency
        .into_iter()
        .filter(|item| concluded.contains(item))
        .collect();

    Ok(UnaryClosure {
        edges,
        concluded: ordered,
        truncated,
    })
}

fn reaches(
    graph: &HashMap<CategoryId, HashSet<CategoryId>>,
    start: CategoryId,
    target: CategoryId,
) -> bool {
    let mut pending = vec![start];
    let mut seen = HashSet::new();

    while let Some(current) = pending.pop() {
        if current == target {
            return true;
        }

        if seen.insert(current) {
            pending.extend(graph.get(&current).into_iter().flatten().copied());
        }
    }

    false
}

fn dependency_order(edges: &[(ItemId, Step)], arrival: &[ItemId]) -> Vec<ItemId> {
    let mut graph: HashMap<ItemId, HashSet<ItemId>> = HashMap::new();
    let mut indegree: HashMap<ItemId, usize> = HashMap::new();

    for (child, step) in edges {
        let output = ItemId {
            category: step.output,
            tag: step.tag,
        };
        let first_dependency = graph.entry(*child).or_default().insert(output);
        graph.entry(output).or_default();
        indegree.entry(*child).or_insert(0);

        if first_dependency {
            *indegree.entry(output).or_insert(0) += 1;
        }
    }

    let mut ready: Vec<ItemId> = arrival
        .iter()
        .copied()
        .filter(|item| indegree.get(item).is_some_and(|degree| *degree == 0))
        .collect();
    let mut ordered = Vec::new();
    let mut at = 0;

    while let Some(child) = ready.get(at).copied() {
        at += 1;
        ordered.push(child);

        for output in graph.get(&child).into_iter().flatten() {
            if let Some(degree) = indegree.get_mut(output) {
                *degree -= 1;

                if *degree == 0 {
                    ready.push(*output);
                }
            }
        }
    }

    ordered
}
