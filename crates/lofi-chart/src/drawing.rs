//! Rule drawing over dense chart identifiers for exact scoped categories.
//!
//! A [`DrawingProgram`] owns the dense category inventory of one chart
//! program: it interns exact categories under their [`Scope`], compiles their
//! candidate-index metadata through a [`ScopedRuleProgram`], applies complete
//! rules and their output-factor fragments, caches the results, and filters
//! steps by root reachability. A [`DrawingResolver`] drives a chart sweep
//! through the program, asking a [`DrawingHost`] only for what the program
//! cannot compute: local edge weights, residual rule applications and the items
//! a cell keeps.

use std::collections::{HashMap, HashSet};
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex, MutexGuard};

use crate::chart::{
    CategoryBounds, CategoryDimensions, CategoryMeta, ChartError, ChartProgram, ForestResolver,
    InsideResolver, ItemId, MissBatch, NormalDecision, Pair, PairBatch, PairBatchRequest,
    PlannedStepRow, RuleSpec, Span, Step, StepBatchEntry, StepBatchRequest, StepPlan, StepRequest,
    ThreadPool, UnaryClosure, unary_closure,
};
use crate::program::{DrawError, DrawResult, HostError, Scope, ScopedRuleProgram};
use crate::reachability::{
    BinaryPermitWitnesses, BinaryPermits, ReachabilityPlan, ReachabilityRule,
};
use crate::rule::{
    RepresentativeKernel, RepresentativeTable, RuleJoinKey, RuleOutputFactor,
    combine_output_factors,
};
use crate::semiring::Semiring;
use crate::term::{SortId, Term};

type StepBatchPlans = Vec<Vec<PlannedStepRow>>;
type ResidualStepWork = (usize, usize, StepPlan);

/// Rule applications the program cannot draw itself.
pub trait ResidualRules {
    /// Resolve one batch of residual rule applications to admitted steps.
    ///
    /// # Errors
    ///
    /// Returns the callback's failure.
    fn resolve(&mut self, batch: MissBatch) -> Result<Vec<Step>, HostError>;
}

/// The caller-supplied part of an inside sweep.
pub trait DrawingHost<W>: ResidualRules {
    /// Decide, per input pair, whether a step with exact premises is admitted.
    ///
    /// # Errors
    ///
    /// Returns the callback's failure.
    fn admit_rows(
        &mut self,
        rule: u16,
        tags: &[u8],
        input_pairs: &[[u32; 2]],
        output: u32,
    ) -> Result<Vec<bool>, HostError>;

    /// Return one rule edge's local weight.
    ///
    /// # Errors
    ///
    /// Returns the callback's failure.
    fn weight(
        &mut self,
        rule: u16,
        inputs: &[u32],
        output: u32,
        span: Span,
        split: Option<usize>,
    ) -> Result<W, HostError>;

    /// Run between span widths, for example to observe an interruption.
    ///
    /// # Errors
    ///
    /// Returns the callback's failure.
    fn finish_width(&mut self) -> Result<(), HostError>;
}

/// The caller-supplied part of a forest sweep.
pub trait ForestHost<W>: DrawingHost<W> {
    /// Return the items a root cell keeps, or `None` to keep them all.
    ///
    /// # Errors
    ///
    /// Returns the callback's failure.
    fn root_keep(
        &mut self,
        span: Span,
        items: &[(ItemId, W)],
        closure: &UnaryClosure,
    ) -> Result<Option<Vec<ItemId>>, HostError>;

    /// Return the items a pruned cell keeps, or `None` to keep them all.
    ///
    /// # Errors
    ///
    /// Returns the callback's failure.
    fn prune_keep(
        &mut self,
        span: Span,
        items: &[(ItemId, W)],
    ) -> Result<Option<Vec<ItemId>>, HostError>;
}

/// The rule-drawing program shared by both chart retention modes.
pub struct DrawingProgram<O> {
    inner: Mutex<ChartProgram>,
    categories: Mutex<Categories<O>>,
    fragment_outputs: Mutex<HashMap<RawFragmentKey, Vec<Term>>>,
    reachability: Mutex<Option<Arc<Reachability<O>>>>,
    category_measure: HashMap<String, usize>,
    shapes: Vec<(usize, bool)>,
    rules: Arc<ScopedRuleProgram<O>>,
    normalizes: bool,
}

/// One rule declaration of a drawing program.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DrawingRule {
    /// Premise count.
    pub arity: usize,
    /// Declared spine order recorded on edges.
    pub order: usize,
    /// Whether the program draws every conclusion of the rule.
    pub complete: bool,
    /// Normal-form decision per child-tag row.
    pub decisions: Vec<(Vec<u8>, NormalDecision)>,
}

/// The exact inputs of a root-directed reachability plan.
pub struct ReachabilityInputs<O> {
    /// The matching representative function the plan compares categories under.
    pub kernel: RepresentativeKernel,
    /// The representatives already registered while lowering the inputs.
    pub table: RepresentativeTable,
    /// The scope every input category belongs to.
    pub scope: Arc<O>,
    /// Sentence length in tokens.
    pub tokens: usize,
    /// Unary closure depth per cell.
    pub unary_depth: usize,
    /// The most binding-and-candidate pairs one lexical refinement enumerates.
    pub refinement_pairs: usize,
    /// Root category representatives.
    pub roots: Vec<Term>,
    /// Lexical category representatives per one-token span.
    pub lexical: HashMap<Span, Vec<Term>>,
    /// One reachability rule per chart rule.
    pub rules: Vec<ReachabilityRule>,
    /// The sort a pattern metavariable takes when it declares none.
    pub default_sort: SortId,
}

fn record_step(entries: &mut Vec<StepBatchEntry>, request: usize, row: usize, step: Step) {
    entries.push(StepBatchEntry { request, row, step });
}

#[derive(Debug)]
struct Reachability<O> {
    plan: ReachabilityPlan,
    kernel: RepresentativeKernel,
    representatives: Mutex<RepresentativeTable>,
    registry: Arc<O>,
}

impl<O: Scope> Reachability<O> {
    fn keeps(&self, span: Span, term: &Term, registry: &Arc<O>) -> DrawResult<O, bool> {
        if !Arc::ptr_eq(&self.registry, registry) {
            return Err(DrawError::invalid(
                "a reachable category belongs to another term signature",
            ));
        }
        let representative = {
            let mut representatives = self.representatives.lock().map_err(|_| {
                DrawError::internal("the reachability representative lock is poisoned")
            })?;
            represent(&self.kernel, &mut representatives, term)?
        };
        Ok(self.plan.keeps(span, &representative))
    }

    fn representatives(&self, terms: &[(Term, Arc<O>)]) -> DrawResult<O, Vec<Term>> {
        for (_, registry) in terms {
            if !Arc::ptr_eq(&self.registry, registry) {
                return Err(DrawError::invalid(
                    "a reachable premise belongs to another term signature",
                ));
            }
        }
        {
            let mut table = self.representatives.lock().map_err(|_| {
                DrawError::internal("the reachability representative lock is poisoned")
            })?;
            terms
                .iter()
                .map(|(term, _)| represent(&self.kernel, &mut table, term))
                .collect()
        }
    }

    fn permits(&self, span: Span, parts: &[Span], rule: u16, representatives: &[Term]) -> bool {
        self.plan.permits(span, parts, rule, representatives)
    }

    fn binary_permits(
        &self,
        span: Span,
        parts: &[Span; 2],
        left: &[Term],
        right: &[Term],
    ) -> BinaryPermits {
        self.plan.binary_permits(span, parts, left, right)
    }

    fn permits_output(
        &self,
        span: Span,
        parts: &[Span],
        rule: u16,
        inputs: &[Term],
        output: &Term,
    ) -> bool {
        self.plan.permits_output(span, parts, rule, inputs, output)
    }

    fn permits_binary_output(
        &self,
        witnesses: &mut BinaryPermitWitnesses<'_>,
        rule: u16,
        inputs: &[Term; 2],
        output: &Term,
    ) -> bool {
        self.plan
            .permits_binary_output(witnesses, rule, inputs, output)
    }
}

struct Categories<O> {
    terms: Vec<(Term, Arc<O>)>,
    output_factors: Vec<Vec<CategoryOutputFactor>>,
    identifiers: HashMap<Term, u32>,
    join_ids: HashMap<(u16, RuleJoinKey), u32>,
    join_keys: Vec<(u16, RuleJoinKey)>,
    rejected: HashSet<Term>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct CategoryOutputFactor {
    rule: u16,
    slot: u8,
    key: u32,
    factor: RuleOutputFactor,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct OutputFactorGroupKey {
    rule: u16,
    key: u32,
    factor: RuleOutputFactor,
    representative: u64,
    scope: usize,
}

#[derive(Clone)]
struct OutputFactorGroup<O> {
    positions: Vec<usize>,
    representative: Option<Term>,
    owner: Term,
    registry: Arc<O>,
}

struct OutputFactorCandidate<O> {
    rule: u16,
    left: Vec<usize>,
    right: Vec<usize>,
    output: Term,
    registry: Arc<O>,
}

struct ReachableOutputFactorCandidate {
    rule: u16,
    left_group: u32,
    right_group: u32,
    output: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct OutputFactorPairKey {
    rule: u16,
    scope: usize,
    left: RuleOutputFactor,
    right: RuleOutputFactor,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ScopedFactorOutput {
    scope: usize,
    term: Term,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FactorIdentity {
    Pending,
    Refused,
    Known(u32),
}

struct CachedFactorOutput<O> {
    output: ScopedFactorOutput,
    representative: Option<Term>,
    registry: Arc<O>,
    identity: FactorIdentity,
}

/// The combined output-factor conclusions one sweep has interned.
struct FactorOutputCache<O> {
    combined: HashMap<OutputFactorPairKey, u32>,
    ids: HashMap<ScopedFactorOutput, u32>,
    outputs: Vec<CachedFactorOutput<O>>,
}

impl<O: Scope> FactorOutputCache<O> {
    fn new() -> Self {
        Self {
            combined: HashMap::new(),
            ids: HashMap::new(),
            outputs: Vec::new(),
        }
    }

    fn get(&self, output: u32) -> DrawResult<O, &CachedFactorOutput<O>> {
        self.outputs
            .get(usize::try_from(output).unwrap_or(usize::MAX))
            .ok_or_else(|| DrawError::internal("a factor candidate has no interned output"))
    }

    fn get_mut(&mut self, output: u32) -> DrawResult<O, &mut CachedFactorOutput<O>> {
        self.outputs
            .get_mut(usize::try_from(output).unwrap_or(usize::MAX))
            .ok_or_else(|| DrawError::internal("an interned factor output disappeared"))
    }

    fn representative(&self, output: u32) -> DrawResult<O, &Term> {
        self.get(output)?.representative.as_ref().ok_or_else(|| {
            DrawError::internal("a cached factor output lost its reachability representative")
        })
    }

    /// Return the interned conclusion two complementary output factors build.
    fn combined(
        &mut self,
        left_key: &OutputFactorGroupKey,
        right_key: &OutputFactorGroupKey,
        left: &OutputFactorGroup<O>,
    ) -> DrawResult<O, u32> {
        let pair_key = OutputFactorPairKey {
            rule: left_key.rule,
            scope: left_key.scope,
            left: left_key.factor.clone(),
            right: right_key.factor.clone(),
        };
        if let Some(output) = self.combined.get(&pair_key) {
            return Ok(*output);
        }
        let store = left
            .registry
            .store_for(&left.owner)
            .ok_or_else(|| DrawError::invalid("a grouped rule premise has no live term store"))?;
        let term = combine_output_factors(&store, &left_key.factor, &right_key.factor)
            .map_err(|error| DrawError::invalid(error.to_string()))?
            .ok_or_else(|| {
                DrawError::internal("a shared rule join has incompatible output factors")
            })?;
        let output = ScopedFactorOutput {
            scope: left_key.scope,
            term,
        };
        let identifier = if let Some(identifier) = self.ids.get(&output) {
            *identifier
        } else {
            let identifier = u32::try_from(self.outputs.len()).map_err(|_| {
                DrawError::invalid("the factored-output inventory exceeds the id width")
            })?;
            self.ids.insert(output.clone(), identifier);
            self.outputs.push(CachedFactorOutput {
                output,
                representative: None,
                registry: left.registry.clone(),
                identity: FactorIdentity::Pending,
            });
            identifier
        };
        self.combined.insert(pair_key, identifier);
        Ok(identifier)
    }

    /// Give every candidate output its reachability representative.
    fn represent(
        &mut self,
        candidates: &[ReachableOutputFactorCandidate],
        reachability: &Reachability<O>,
    ) -> DrawResult<O, ()> {
        let mut missing = Vec::new();
        let mut scheduled = HashSet::new();
        for candidate in candidates {
            if self.get(candidate.output)?.representative.is_none()
                && scheduled.insert(candidate.output)
            {
                missing.push(candidate.output);
            }
        }
        let terms = missing
            .iter()
            .map(|output| {
                let cached = self.get(*output)?;
                Ok((cached.output.term.clone(), cached.registry.clone()))
            })
            .collect::<DrawResult<O, Vec<_>>>()?;
        let representatives = reachability.representatives(&terms)?;
        if representatives.len() != missing.len() {
            return Err(DrawError::internal(
                "the reachability kernel returned an inconsistent output representative count",
            ));
        }
        for (output, representative) in missing.into_iter().zip(representatives) {
            self.get_mut(output)?.representative = Some(representative);
        }
        Ok(())
    }

    /// Give every candidate output its dense category identity.
    fn identify<F>(
        &mut self,
        candidates: &[ReachableOutputFactorCandidate],
        identify: F,
    ) -> DrawResult<O, ()>
    where
        F: FnOnce(&[(Term, Arc<O>)]) -> DrawResult<O, Vec<Option<u32>>>,
    {
        let mut pending = Vec::new();
        let mut scheduled = HashSet::new();
        for candidate in candidates {
            if self.get(candidate.output)?.identity == FactorIdentity::Pending
                && scheduled.insert(candidate.output)
            {
                pending.push(candidate.output);
            }
        }
        let terms = pending
            .iter()
            .map(|output| {
                let cached = self.get(*output)?;
                Ok((cached.output.term.clone(), cached.registry.clone()))
            })
            .collect::<DrawResult<O, Vec<_>>>()?;
        let identifiers = identify(&terms)?;
        if identifiers.len() != pending.len() {
            return Err(DrawError::internal(
                "the category inventory returned an inconsistent output count",
            ));
        }
        for (output, identifier) in pending.into_iter().zip(identifiers) {
            self.get_mut(output)?.identity =
                identifier.map_or(FactorIdentity::Refused, FactorIdentity::Known);
        }
        Ok(())
    }
}

enum CategoryIdentifier {
    Known(u32),
    Rejected,
    Pending(usize),
}

type PendingCategory<O> = (usize, Term, Arc<O>, CategoryDimensions);
type PendingFactor<O> = (u32, u16, u8, u32, Term, Arc<O>);
type CoreCategory<O> = (Term, Arc<O>);
type PendingInventory<O> = (Vec<CategoryIdentifier>, Vec<CoreCategory<O>>);
type FragmentJob<O> = (Vec<(u16, RuleJoinKey)>, Vec<CoreCategory<O>>);
type FragmentOutputRow<O> = Vec<(u16, Vec<Term>, Arc<O>)>;
type CompleteFragmentOutputs<O> = (Vec<FragmentOutputRow<O>>, Vec<Option<u32>>);
type CompletePairPosition = (usize, usize, [u32; 2]);
type CachedCompleteRow = (usize, usize, [u32; 2], u16, Vec<u32>);
type OutputsByLeft = HashMap<(u16, u32, usize), Vec<usize>>;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct RawFragmentKey {
    rule: u16,
    fragment: RuleJoinKey,
    scope: usize,
    inputs: [u64; 2],
}

struct CompletePairPlan<O> {
    jobs: Vec<FragmentJob<O>>,
    positions: Vec<CompletePairPosition>,
    cached: Vec<CachedCompleteRow>,
    residual: Vec<Pair>,
}

struct CompletePairContext<'a, 'p, O> {
    request: &'a PairBatchRequest,
    workers: Option<&'a ThreadPool>,
    reachability: &'a Reachability<O>,
    witnesses: &'a mut BinaryPermitWitnesses<'p>,
}

struct BinaryOutputPermit<'a, 'p, O> {
    reachability: &'a Reachability<O>,
    witnesses: &'a mut BinaryPermitWitnesses<'p>,
    representatives: &'a HashMap<u32, Term>,
    cache: HashMap<(u16, u64, u64, u64), bool>,
}

impl<O: Scope> BinaryOutputPermit<'_, '_, O> {
    fn permits_representatives(&mut self, rule: u16, inputs: &[Term; 2], output: &Term) -> bool {
        let key = (
            rule,
            inputs[0].serial(),
            inputs[1].serial(),
            output.serial(),
        );
        *self.cache.entry(key).or_insert_with(|| {
            self.reachability
                .permits_binary_output(self.witnesses, rule, inputs, output)
        })
    }

    fn permits_pair(&mut self, rule: u16, inputs: [u32; 2]) -> DrawResult<O, bool> {
        let representative = |identifier: u32| {
            self.representatives.get(&identifier).ok_or_else(|| {
                DrawError::internal(format!(
                    "category {identifier} has no reachability representative"
                ))
            })
        };
        Ok(self.witnesses.permits(
            rule,
            representative(inputs[0])?.serial(),
            representative(inputs[1])?.serial(),
        ))
    }

    fn permits(&mut self, rule: u16, inputs: [u32; 2], output: u32) -> DrawResult<O, bool> {
        let representative = |identifier: u32| {
            self.representatives.get(&identifier).ok_or_else(|| {
                DrawError::internal(format!(
                    "category {identifier} has no reachability representative"
                ))
            })
        };
        let inputs = [
            representative(inputs[0])?.clone(),
            representative(inputs[1])?.clone(),
        ];
        let output = representative(output)?;
        Ok(self.permits_representatives(rule, &inputs, output))
    }
}

fn positive_bound<O>(name: &str, value: Option<usize>) -> DrawResult<O, Option<NonZeroUsize>> {
    value
        .map(|bound| {
            NonZeroUsize::new(bound)
                .ok_or_else(|| DrawError::invalid(format!("{name} must be positive")))
        })
        .transpose()
}

/// Return one term's matching representative, registering it first when new.
///
/// # Errors
///
/// Returns [`DrawError::Invalid`] when the kernel assigns no representative.
pub fn represent<O>(
    kernel: &RepresentativeKernel,
    table: &mut RepresentativeTable,
    term: &Term,
) -> DrawResult<O, Term> {
    if !table.contains(term) {
        table.register_with(kernel, term);
    }
    table
        .representative(term)
        .map_err(|error| DrawError::invalid(error.to_string()))
}

impl<O: Scope> DrawingProgram<O> {
    fn inner(&self) -> DrawResult<O, MutexGuard<'_, ChartProgram>> {
        self.inner
            .lock()
            .map_err(|_| DrawError::internal("the chart program lock is poisoned"))
    }

    fn categories(&self) -> DrawResult<O, MutexGuard<'_, Categories<O>>> {
        self.categories
            .lock()
            .map_err(|_| DrawError::internal("the chart category lock is poisoned"))
    }

    fn fragment_outputs(
        &self,
    ) -> DrawResult<O, MutexGuard<'_, HashMap<RawFragmentKey, Vec<Term>>>> {
        self.fragment_outputs
            .lock()
            .map_err(|_| DrawError::internal("the rule fragment cache lock is poisoned"))
    }

    fn extend_category_representatives(
        &self,
        reachability: &Reachability<O>,
        representatives: &mut HashMap<u32, Term>,
        represented_categories: &mut usize,
    ) -> DrawResult<O, ()> {
        let categories = self.categories()?;
        if *represented_categories > categories.terms.len() {
            return Err(DrawError::internal(
                "the represented category prefix exceeds the category inventory",
            ));
        }
        let start = *represented_categories;
        let end = categories.terms.len();
        let terms = categories
            .terms
            .iter()
            .skip(start)
            .cloned()
            .collect::<Vec<_>>();
        drop(categories);
        let fresh = reachability.representatives(&terms)?;
        if fresh.len() != terms.len() {
            return Err(DrawError::internal(
                "the reachability kernel returned an inconsistent representative count",
            ));
        }
        for (offset, representative) in fresh.into_iter().enumerate() {
            let identifier = u32::try_from(start + offset).map_err(|_| {
                DrawError::invalid("the chart category inventory exceeds the id width")
            })?;
            representatives.insert(identifier, representative);
        }
        *represented_categories = end;
        Ok(())
    }

    fn install_reachability(&self, reachability: Reachability<O>) -> DrawResult<O, ()> {
        let mut slot = self
            .reachability
            .lock()
            .map_err(|_| DrawError::internal("the chart reachability lock is poisoned"))?;
        *slot = Some(Arc::new(reachability));
        drop(slot);
        Ok(())
    }

    fn keeps(&self, span: Span, category: u32) -> DrawResult<O, bool> {
        let reachability = self
            .reachability
            .lock()
            .map_err(|_| DrawError::internal("the chart reachability lock is poisoned"))?
            .clone();
        let Some(reachability) = reachability else {
            return Ok(true);
        };
        let (term, registry) = {
            let categories = self.categories()?;
            categories
                .terms
                .get(usize::try_from(category).unwrap_or(usize::MAX))
                .cloned()
                .ok_or_else(|| DrawError::Chart(ChartError::UnknownCategory { category }))?
        };
        reachability.keeps(span, &term, &registry)
    }

    fn reachable_steps(&self, span: Span, steps: Vec<Step>) -> DrawResult<O, Vec<Step>> {
        let mut reachable = Vec::with_capacity(steps.len());
        for step in steps {
            if self.keeps(span, step.output)? {
                reachable.push(step);
            }
        }
        Ok(reachable)
    }

    fn permits(
        &self,
        span: Span,
        parts: &[Span],
        rule: u16,
        inputs: &[u32],
    ) -> DrawResult<O, bool> {
        let reachability = self
            .reachability
            .lock()
            .map_err(|_| DrawError::internal("the chart reachability lock is poisoned"))?
            .clone();
        let Some(reachability) = reachability else {
            return Ok(true);
        };
        let categories = self.categories()?;
        let terms = inputs
            .iter()
            .map(|category| {
                categories
                    .terms
                    .get(usize::try_from(*category).unwrap_or(usize::MAX))
                    .cloned()
                    .ok_or_else(|| {
                        DrawError::Chart(ChartError::UnknownCategory {
                            category: *category,
                        })
                    })
            })
            .collect::<DrawResult<O, Vec<_>>>()?;
        let representatives = reachability.representatives(&terms)?;
        Ok(reachability.permits(span, parts, rule, &representatives))
    }

    fn permits_output(
        &self,
        span: Span,
        parts: &[Span],
        rule: u16,
        inputs: &[u32],
        output: u32,
    ) -> DrawResult<O, bool> {
        let reachability = self
            .reachability
            .lock()
            .map_err(|_| DrawError::internal("the chart reachability lock is poisoned"))?
            .clone();
        let Some(reachability) = reachability else {
            return Ok(true);
        };
        let categories = self.categories()?;
        let mut identifiers = inputs.to_vec();
        identifiers.push(output);
        let terms = identifiers
            .iter()
            .map(|category| {
                categories
                    .terms
                    .get(usize::try_from(*category).unwrap_or(usize::MAX))
                    .cloned()
                    .ok_or_else(|| {
                        DrawError::Chart(ChartError::UnknownCategory {
                            category: *category,
                        })
                    })
            })
            .collect::<DrawResult<O, Vec<_>>>()?;
        drop(categories);
        let mut representatives = reachability.representatives(&terms)?;
        let Some(output) = representatives.pop() else {
            return Ok(false);
        };
        Ok(reachability.permits_output(span, parts, rule, &representatives, &output))
    }

    fn identify_core(&self, term: Term, registry: Arc<O>) -> DrawResult<O, Option<u32>> {
        self.identify_cores(&[(term, registry)], None)
            .map(|mut identifiers| identifiers.pop().flatten())
    }

    fn category_arity(&self, term: &Term) -> DrawResult<O, usize> {
        let mut arity = 0;
        let mut current = term;

        while let Some(slot) = self.category_measure.get(current.operator_name()).copied() {
            let result = current.arguments().get(slot).ok_or_else(|| {
                DrawError::invalid(format!(
                    "constructor {:?} declares result slot {slot}, but this term has {} children",
                    current.operator_name(),
                    current.arguments().len()
                ))
            })?;
            arity += 1;
            current = result;
        }
        Ok(arity)
    }

    fn identify_cores(
        &self,
        terms: &[(Term, Arc<O>)],
        workers: Option<&ThreadPool>,
    ) -> DrawResult<O, Vec<Option<u32>>> {
        let (identifiers, pending) = self.pending_categories(terms)?;
        let mut pending_ids = vec![None; pending.len()];
        let admitted = self.partition_categories(pending)?;
        let pending_factors = self.register_categories(admitted, workers, &mut pending_ids)?;
        self.register_factors(pending_factors, workers)?;

        identifiers
            .into_iter()
            .map(|identifier| match identifier {
                CategoryIdentifier::Known(identifier) => Ok(Some(identifier)),
                CategoryIdentifier::Rejected => Ok(None),
                CategoryIdentifier::Pending(position) => {
                    pending_ids.get(position).copied().ok_or_else(|| {
                        DrawError::internal("a pending chart category was not registered")
                    })
                }
            })
            .collect()
    }

    fn pending_categories(&self, terms: &[(Term, Arc<O>)]) -> DrawResult<O, PendingInventory<O>> {
        let mut pending = Vec::new();
        let mut pending_at = HashMap::new();
        let categories = self.categories()?;
        let identifiers = terms
            .iter()
            .map(|(term, registry)| {
                if let Some(identifier) = categories.identifiers.get(term).copied() {
                    return CategoryIdentifier::Known(identifier);
                }
                if categories.rejected.contains(term) {
                    return CategoryIdentifier::Rejected;
                }
                let position = *pending_at.entry(term.clone()).or_insert_with(|| {
                    let position = pending.len();
                    pending.push((term.clone(), registry.clone()));
                    position
                });
                CategoryIdentifier::Pending(position)
            })
            .collect();
        drop(categories);
        Ok((identifiers, pending))
    }

    fn partition_categories(
        &self,
        pending: Vec<(Term, Arc<O>)>,
    ) -> DrawResult<O, Vec<PendingCategory<O>>> {
        let mut admitted = Vec::new();
        let mut categories = self.categories()?;
        let mut program = self.inner()?;
        for (position, (term, registry)) in pending.into_iter().enumerate() {
            let dimensions = CategoryDimensions {
                arity: self.category_arity(&term)?,
                size: term.size(),
            };
            if program.admits_category(dimensions) {
                admitted.push((position, term, registry, dimensions));
                continue;
            }

            let identifier = u32::try_from(categories.terms.len()).map_err(|_| {
                DrawError::invalid("the chart category inventory exceeds the id width")
            })?;
            let registered = program
                .register_category(identifier, CategoryMeta::empty(), dimensions)
                .map_err(DrawError::Chart)?;
            if registered {
                return Err(DrawError::internal(
                    "a category admitted after its declared bounds refused it",
                ));
            }
            categories.rejected.insert(term);
        }
        drop(program);
        drop(categories);
        Ok(admitted)
    }

    fn register_categories(
        &self,
        admitted: Vec<PendingCategory<O>>,
        workers: Option<&ThreadPool>,
        pending_ids: &mut [Option<u32>],
    ) -> DrawResult<O, Vec<PendingFactor<O>>> {
        let admitted_terms = admitted
            .iter()
            .map(|(_, term, registry, _)| (term.clone(), registry.clone()))
            .collect::<Vec<_>>();
        let metas = self
            .rules
            .category_meta_batch(&self.shapes, &admitted_terms, workers)?;
        let mut categories = self.categories()?;
        let mut program = self.inner()?;
        let mut pending_factors = Vec::new();
        for ((position, term, registry, dimensions), meta) in admitted.into_iter().zip(metas) {
            let identifier = u32::try_from(categories.terms.len()).map_err(|_| {
                DrawError::invalid("the chart category inventory exceeds the id width")
            })?;
            let mut joins = HashMap::new();
            for (rule, slot, keys) in meta.joins {
                let mut identifiers = Vec::with_capacity(keys.len());
                for key in keys.iter() {
                    let key = (rule, key.clone());
                    let key_id = if let Some(identifier) = categories.join_ids.get(&key) {
                        *identifier
                    } else {
                        let identifier =
                            u32::try_from(categories.join_keys.len()).map_err(|_| {
                                DrawError::invalid(
                                    "the chart join-key inventory exceeds the id width",
                                )
                            })?;
                        categories.join_ids.insert(key.clone(), identifier);
                        categories.join_keys.push(key);
                        identifier
                    };
                    identifiers.push(key_id);
                }
                joins.insert((rule, slot), identifiers);
            }
            let factors = meta
                .factors
                .into_iter()
                .map(|(rule, slot, key, output)| {
                    let key = categories
                        .join_ids
                        .get(&(rule, key))
                        .copied()
                        .ok_or_else(|| {
                            DrawError::internal("a factored rule output has no registered join key")
                        })?;
                    Ok((identifier, rule, slot, key, output, registry.clone()))
                })
                .collect::<DrawResult<O, Vec<_>>>()?;
            let output_factors = meta
                .output_factors
                .into_iter()
                .map(|(rule, slot, key, factor)| {
                    let key = categories
                        .join_ids
                        .get(&(rule, key))
                        .copied()
                        .ok_or_else(|| {
                            DrawError::internal("a grouped rule output has no registered join key")
                        })?;
                    Ok(CategoryOutputFactor {
                        rule,
                        slot,
                        key,
                        factor,
                    })
                })
                .collect::<DrawResult<O, Vec<_>>>()?;
            let admitted = program
                .register_category(
                    identifier,
                    CategoryMeta {
                        unary: meta.unary,
                        viable: meta.viable.into_iter().collect(),
                        joins,
                        factors: HashMap::new(),
                    },
                    dimensions,
                )
                .map_err(DrawError::Chart)?;
            if !admitted {
                return Err(DrawError::internal(
                    "a category refused after its declared bounds admitted it",
                ));
            }
            pending_factors.extend(factors);
            categories.identifiers.insert(term.clone(), identifier);
            categories.terms.push((term, registry));
            categories.output_factors.push(output_factors);
            pending_ids[position] = Some(identifier);
        }
        drop(program);
        drop(categories);
        Ok(pending_factors)
    }

    fn register_factors(
        &self,
        pending_factors: Vec<PendingFactor<O>>,
        workers: Option<&ThreadPool>,
    ) -> DrawResult<O, ()> {
        if !pending_factors.is_empty() {
            let outputs = pending_factors
                .iter()
                .map(|(_, _, _, _, output, registry)| (output.clone(), registry.clone()))
                .collect::<Vec<_>>();
            let output_ids = self.identify_cores(&outputs, workers)?;
            let mut program = self.inner()?;
            for ((category, rule, slot, key, _, _), output) in
                pending_factors.into_iter().zip(output_ids)
            {
                if let Some(output) = output {
                    program
                        .register_factor(category, rule, slot, key, output)
                        .map_err(DrawError::Chart)?;
                }
            }
        }
        Ok(())
    }

    fn draw_native(&self, inputs: &[u32], native: &[(u16, u8)]) -> DrawResult<O, Vec<Step>> {
        let (missing, mut outputs_by_rule) = {
            let program = self.inner()?;
            let mut missing = Vec::new();
            let mut outputs_by_rule = HashMap::new();
            for (rule, _) in native {
                if let Some(outputs) = program.cached_outputs(*rule, inputs) {
                    outputs_by_rule.insert(*rule, outputs.to_vec());
                } else {
                    missing.push(*rule);
                }
            }
            (missing, outputs_by_rule)
        };

        if !missing.is_empty() {
            let terms = {
                let categories = self.categories()?;
                inputs
                    .iter()
                    .map(|identifier| {
                        categories
                            .terms
                            .get(usize::try_from(*identifier).unwrap_or(usize::MAX))
                            .cloned()
                            .ok_or_else(|| {
                                DrawError::Chart(ChartError::UnknownCategory {
                                    category: *identifier,
                                })
                            })
                    })
                    .collect::<DrawResult<O, Vec<_>>>()?
            };
            let native_outputs = self.rules.apply(&missing, &terms)?;
            for (rule, outputs, registry) in native_outputs {
                let mut identifiers = Vec::with_capacity(outputs.len());
                for output in outputs {
                    if let Some(identifier) = self.identify_core(output, registry.clone())? {
                        identifiers.push(identifier);
                    }
                }
                self.inner()?
                    .remember_outputs(rule, inputs.to_vec(), identifiers.clone())
                    .map_err(DrawError::Chart)?;
                outputs_by_rule.insert(rule, identifiers);
            }
        }
        Ok(native
            .iter()
            .flat_map(|(rule, tag)| {
                outputs_by_rule[rule]
                    .iter()
                    .copied()
                    .map(move |output| Step {
                        rule: *rule,
                        output,
                        tag: *tag,
                    })
            })
            .collect())
    }

    fn finish_step_plan<H: ResidualRules>(
        &self,
        request: &StepRequest,
        plan: StepPlan,
        callbacks: &mut H,
    ) -> DrawResult<O, Vec<Step>> {
        let (native, residual) = match plan {
            StepPlan::Cached(steps) => return Ok(steps),
            StepPlan::Draw { native, residual } => (native, residual),
        };

        let mut results = self.draw_native(&request.inputs, &native)?;

        if !residual.is_empty() {
            let inner = {
                let program = self.inner()?;
                MissBatch::new(
                    &program,
                    request.inputs.clone(),
                    request.tags.clone(),
                    residual,
                    request.span,
                )
            };
            results.extend(callbacks.resolve(inner).map_err(DrawError::Host)?);
        }

        let mut program = self.inner()?;
        program
            .validate_steps(&request.inputs, &request.candidates, &results)
            .map_err(DrawError::Chart)?;

        if program
            .candidates_are_complete(&request.candidates)
            .map_err(DrawError::Chart)?
        {
            program.remember(
                request.inputs.clone(),
                request.tags.clone(),
                request.candidates.clone(),
                results.clone(),
            );
        }
        drop(program);

        Ok(results)
    }

    fn draw_steps<H: ResidualRules>(
        &self,
        inputs: Vec<u32>,
        tags: Vec<u8>,
        span: (usize, usize),
        candidates: &[u16],
        callbacks: &mut H,
    ) -> DrawResult<O, Vec<Step>> {
        let request = StepRequest {
            inputs,
            tags,
            span,
            candidates: candidates.to_vec(),
        };
        let plan = {
            let program = self.inner()?;
            program
                .step_plan(
                    &request.inputs,
                    &request.tags,
                    &request.candidates,
                    self.normalizes,
                )
                .map_err(DrawError::Chart)?
        };
        let steps = self.finish_step_plan(&request, plan, callbacks)?;
        self.reachable_steps(span, steps)
    }

    fn batch_step_plans(
        &self,
        requests: &[StepBatchRequest],
        workers: Option<&ThreadPool>,
    ) -> DrawResult<O, StepBatchPlans> {
        let program = self.inner()?;
        program
            .batch_step_plans(requests, self.normalizes, workers)
            .map_err(DrawError::Chart)
    }

    fn draw_steps_batch<H: ResidualRules>(
        &self,
        requests: &[StepBatchRequest],
        callbacks: &mut H,
        workers: Option<&ThreadPool>,
    ) -> DrawResult<O, Vec<StepBatchEntry>> {
        let plans = self.batch_step_plans(requests, workers)?;
        let mut jobs: Vec<(Vec<u32>, Vec<u16>)> = Vec::new();
        let mut job_at: HashMap<Vec<u32>, usize> = HashMap::new();
        {
            let program = self.inner()?;
            for (request, row_plans) in requests.iter().zip(&plans) {
                let native = row_plans
                    .iter()
                    .filter_map(|planned| match &planned.plan {
                        StepPlan::Draw { native, .. } => Some(native.as_slice()),
                        StepPlan::Cached(_) => None,
                    })
                    .flatten();
                for (rule, _) in native {
                    if program.cached_outputs(*rule, &request.inputs).is_some() {
                        continue;
                    }
                    let position = *job_at.entry(request.inputs.clone()).or_insert_with(|| {
                        let position = jobs.len();
                        jobs.push((request.inputs.clone(), Vec::new()));
                        position
                    });
                    if !jobs[position].1.contains(rule) {
                        jobs[position].1.push(*rule);
                    }
                }
            }
            drop(program);
        }
        let native_jobs = {
            let categories = self.categories()?;
            jobs.iter()
                .map(|(inputs, candidates)| {
                    let terms = inputs
                        .iter()
                        .map(|identifier| {
                            categories
                                .terms
                                .get(usize::try_from(*identifier).unwrap_or(usize::MAX))
                                .cloned()
                                .ok_or_else(|| {
                                    DrawError::Chart(ChartError::UnknownCategory {
                                        category: *identifier,
                                    })
                                })
                        })
                        .collect::<DrawResult<O, Vec<_>>>()?;
                    Ok((candidates.clone(), terms))
                })
                .collect::<DrawResult<O, Vec<_>>>()?
        };
        let output_rows = self.rules.apply_batches(&native_jobs, workers)?;
        let output_terms = output_rows
            .iter()
            .flat_map(|row| row.iter())
            .flat_map(|(_, terms, registry)| {
                terms.iter().cloned().map(|term| (term, registry.clone()))
            })
            .collect::<Vec<_>>();
        let output_identifiers = self.identify_cores(&output_terms, workers)?;
        let mut identifier_at = 0;

        let mut program = self.inner()?;
        for ((inputs, _), outputs) in jobs.into_iter().zip(output_rows) {
            for (rule, terms, _) in outputs {
                let end = identifier_at + terms.len();
                let identifiers = output_identifiers
                    .get(identifier_at..end)
                    .ok_or_else(|| {
                        DrawError::internal(
                            "a native rule batch returned an inconsistent output count",
                        )
                    })?
                    .iter()
                    .flatten()
                    .copied()
                    .collect();
                identifier_at = end;
                program
                    .remember_outputs(rule, inputs.clone(), identifiers)
                    .map_err(DrawError::Chart)?;
            }
        }
        drop(program);
        self.finish_step_batch(requests, plans, callbacks)
    }

    fn plan_complete_pairs(
        &self,
        request: &PairBatchRequest,
        pairs: Vec<Pair>,
    ) -> DrawResult<O, CompletePairPlan<O>> {
        let categories = self.categories()?;
        let program = self.inner()?;
        let mut plan = CompletePairPlan::default();
        for mut pair in pairs {
            let mut complete = Vec::new();
            pair.candidates.retain(|rule| {
                let is_complete = program
                    .rule(*rule)
                    .is_ok_and(|declaration| declaration.complete);
                if is_complete {
                    complete.push(*rule);
                }
                !is_complete
            });
            if !pair.candidates.is_empty() {
                plan.residual.push(pair.clone());
            }
            let identifiers = [request.left[pair.left], request.right[pair.right]];
            let missing = Self::missing_pair_fragments(
                &program,
                &categories,
                &mut plan.cached,
                &pair,
                identifiers,
                complete,
            )?;
            if missing.is_empty() {
                continue;
            }
            let terms = identifiers
                .iter()
                .map(|identifier| {
                    categories
                        .terms
                        .get(usize::try_from(*identifier).unwrap_or(usize::MAX))
                        .cloned()
                        .ok_or_else(|| {
                            DrawError::Chart(ChartError::UnknownCategory {
                                category: *identifier,
                            })
                        })
                })
                .collect::<DrawResult<O, Vec<_>>>()?;
            plan.jobs.push((missing, terms));
            plan.positions.push((pair.left, pair.right, identifiers));
        }
        drop(program);
        drop(categories);
        Ok(plan)
    }

    fn missing_pair_fragments(
        program: &ChartProgram,
        categories: &Categories<O>,
        cached: &mut Vec<CachedCompleteRow>,
        pair: &Pair,
        identifiers: [u32; 2],
        complete: Vec<u16>,
    ) -> DrawResult<O, Vec<(u16, RuleJoinKey)>> {
        let mut missing = Vec::new();
        for rule in complete {
            if let Some(outputs) = program.cached_outputs(rule, &identifiers) {
                cached.push((pair.left, pair.right, identifiers, rule, outputs.to_vec()));
                continue;
            }
            let left_keys = program
                .category(identifiers[0])
                .map_err(DrawError::Chart)?
                .joins
                .get(&(rule, 0));
            let right_keys = program
                .category(identifiers[1])
                .map_err(DrawError::Chart)?
                .joins
                .get(&(rule, 1));
            let shared = left_keys
                .into_iter()
                .flatten()
                .filter(|key| right_keys.is_some_and(|keys| keys.contains(key)))
                .copied()
                .collect::<Vec<_>>();
            if shared.is_empty() {
                missing.push((rule, RuleJoinKey::Direct));
                continue;
            }
            for key in shared {
                let (held_rule, fragment) = categories
                    .join_keys
                    .get(usize::try_from(key).unwrap_or(usize::MAX))
                    .ok_or_else(|| {
                        DrawError::internal("a category pair names an unknown rule fragment")
                    })?;
                if *held_rule != rule {
                    return Err(DrawError::internal(
                        "a category pair names a fragment owned by another rule",
                    ));
                }
                missing.push((rule, fragment.clone()));
            }
        }
        Ok(missing)
    }

    fn raw_fragment_key(
        rule: u16,
        fragment: &RuleJoinKey,
        terms: &[CoreCategory<O>],
    ) -> DrawResult<O, RawFragmentKey> {
        let [left, right] = terms else {
            return Err(DrawError::internal(
                "a complete binary fragment needs exactly two premises",
            ));
        };
        if !Arc::ptr_eq(&left.1, &right.1) {
            return Err(DrawError::invalid(
                "native rule premises belong to different term signatures",
            ));
        }
        Ok(RawFragmentKey {
            rule,
            fragment: fragment.clone(),
            scope: Arc::as_ptr(&left.1).addr(),
            inputs: [left.0.serial(), right.0.serial()],
        })
    }

    fn apply_raw_fragment_jobs(
        &self,
        jobs: &[FragmentJob<O>],
        workers: Option<&ThreadPool>,
    ) -> DrawResult<O, Vec<FragmentOutputRow<O>>> {
        let mut job_keys = Vec::with_capacity(jobs.len());
        let mut missing = Vec::<(RawFragmentKey, FragmentJob<O>)>::new();
        let mut scheduled = HashSet::new();
        {
            let cache = self.fragment_outputs()?;
            for (fragments, terms) in jobs {
                let mut keys = Vec::with_capacity(fragments.len());
                for (rule, fragment) in fragments {
                    let key = Self::raw_fragment_key(*rule, fragment, terms)?;
                    if !cache.contains_key(&key) && scheduled.insert(key.clone()) {
                        missing.push((
                            key.clone(),
                            (vec![(*rule, fragment.clone())], terms.clone()),
                        ));
                    }
                    keys.push(key);
                }
                job_keys.push(keys);
            }
            drop(cache);
        }

        if !missing.is_empty() {
            let missing_jobs = missing
                .iter()
                .map(|(_, job)| job.clone())
                .collect::<Vec<_>>();
            let fresh_rows = self.rules.apply_fragment_batches(&missing_jobs, workers)?;
            if fresh_rows.len() != missing.len() {
                return Err(DrawError::internal(
                    "a native rule batch returned an inconsistent row count",
                ));
            }
            let mut cache = self.fragment_outputs()?;
            for ((key, _), row) in missing.into_iter().zip(fresh_rows) {
                let mut outputs = row.into_iter();
                let Some((rule, terms, _)) = outputs.next() else {
                    return Err(DrawError::internal(
                        "a native rule fragment returned no result row",
                    ));
                };
                if outputs.next().is_some() || rule != key.rule {
                    return Err(DrawError::internal(
                        "a native rule fragment returned an inconsistent result row",
                    ));
                }
                cache.entry(key).or_insert(terms);
            }
        }

        let cache = self.fragment_outputs()?;
        jobs.iter()
            .zip(job_keys)
            .map(|((_, terms), keys)| {
                let registry = terms
                    .first()
                    .map(|(_, registry)| registry.clone())
                    .ok_or_else(|| {
                        DrawError::internal("a complete binary fragment lost its premise signature")
                    })?;
                let mut row: FragmentOutputRow<O> = Vec::new();
                for key in keys {
                    let outputs = cache.get(&key).ok_or_else(|| {
                        DrawError::internal("a rule fragment cache entry is missing")
                    })?;
                    if let Some((_, held, _)) = row
                        .iter_mut()
                        .find(|(held_rule, _, _)| *held_rule == key.rule)
                    {
                        for output in outputs {
                            if !held.contains(output) {
                                held.push(output.clone());
                            }
                        }
                    } else {
                        row.push((key.rule, outputs.clone(), registry.clone()));
                    }
                }
                Ok(row)
            })
            .collect()
    }

    fn apply_complete_pair_fragments(
        &self,
        context: &mut CompletePairContext<'_, '_, O>,
        jobs: &[FragmentJob<O>],
        positions: &[CompletePairPosition],
        representatives: &mut HashMap<u32, Term>,
    ) -> DrawResult<O, CompleteFragmentOutputs<O>> {
        let output_rows = self.apply_raw_fragment_jobs(jobs, context.workers)?;
        let output_terms = output_rows
            .iter()
            .flat_map(|row| row.iter())
            .flat_map(|(_, terms, registry)| {
                terms.iter().cloned().map(|term| (term, registry.clone()))
            })
            .collect::<Vec<_>>();
        let output_representatives = context.reachability.representatives(&output_terms)?;
        if output_rows.len() != positions.len() {
            return Err(DrawError::internal(
                "a native rule batch returned an inconsistent row count",
            ));
        }
        let admitted = {
            let mut permit = BinaryOutputPermit {
                reachability: context.reachability,
                witnesses: &mut *context.witnesses,
                representatives,
                cache: HashMap::new(),
            };
            let mut output_at = 0;
            let mut admitted = Vec::with_capacity(output_terms.len());
            for ((_, _, inputs), row) in positions.iter().zip(&output_rows) {
                let input_representatives = [
                    representatives.get(&inputs[0]).cloned().ok_or_else(|| {
                        DrawError::internal(format!(
                            "category {} has no reachability representative",
                            inputs[0]
                        ))
                    })?,
                    representatives.get(&inputs[1]).cloned().ok_or_else(|| {
                        DrawError::internal(format!(
                            "category {} has no reachability representative",
                            inputs[1]
                        ))
                    })?,
                ];
                for (rule, terms, _) in row {
                    for _ in terms {
                        let output = output_representatives.get(output_at).ok_or_else(|| {
                            DrawError::internal(
                                "a native rule batch returned an inconsistent output count",
                            )
                        })?;
                        admitted.push(permit.permits_representatives(
                            *rule,
                            &input_representatives,
                            output,
                        ));
                        output_at += 1;
                    }
                }
            }
            if output_at != output_representatives.len() {
                return Err(DrawError::internal(
                    "a native rule batch returned an inconsistent output count",
                ));
            }
            admitted
        };
        let retained_terms = output_terms
            .iter()
            .zip(&admitted)
            .filter(|(_, admitted)| **admitted)
            .map(|(term, _)| term.clone())
            .collect::<Vec<_>>();
        let retained_identifiers = self.identify_cores(&retained_terms, context.workers)?;
        let mut retained_at = 0;
        let output_identifiers =
            admitted
                .iter()
                .map(|admitted| {
                    if !admitted {
                        return Ok(None);
                    }
                    let identifier = retained_identifiers.get(retained_at).copied().ok_or_else(|| {
                    DrawError::internal(
                        "a native rule batch returned an inconsistent retained output count",
                    )
                })?;
                    retained_at += 1;
                    Ok(identifier)
                })
                .collect::<DrawResult<O, Vec<_>>>()?;
        if retained_at != retained_identifiers.len() {
            return Err(DrawError::internal(
                "a native rule batch returned an inconsistent retained output count",
            ));
        }
        representatives.extend(
            output_identifiers
                .iter()
                .zip(output_representatives)
                .filter_map(|(identifier, representative)| {
                    identifier.map(|identifier| (identifier, representative))
                }),
        );
        Ok((output_rows, output_identifiers))
    }

    fn resolved_complete_pairs(
        plan: &CompletePairPlan<O>,
        output_rows: Vec<FragmentOutputRow<O>>,
        output_identifiers: &[Option<u32>],
        permit: &mut BinaryOutputPermit<'_, '_, O>,
    ) -> DrawResult<O, OutputsByLeft> {
        let mut identifier_at = 0;
        let mut by_left = OutputsByLeft::new();
        for (left, right, inputs, rule, outputs) in &plan.cached {
            for output in outputs.iter().copied() {
                if permit.permits(*rule, *inputs, output)? {
                    by_left
                        .entry((*rule, output, *left))
                        .or_default()
                        .push(*right);
                }
            }
        }
        for ((left, right, _), outputs) in plan.positions.iter().zip(output_rows) {
            for (rule, terms, _) in outputs {
                let end = identifier_at + terms.len();
                let identifiers = output_identifiers
                    .get(identifier_at..end)
                    .ok_or_else(|| {
                        DrawError::internal(
                            "a native rule batch returned an inconsistent output count",
                        )
                    })?
                    .iter()
                    .flatten()
                    .copied()
                    .collect::<Vec<_>>();
                identifier_at = end;
                for output in identifiers {
                    by_left
                        .entry((rule, output, *left))
                        .or_default()
                        .push(*right);
                }
            }
        }
        Ok(by_left)
    }

    fn add_complete_pair_products(
        batch: &mut PairBatch,
        by_left: OutputsByLeft,
    ) -> DrawResult<O, ()> {
        let mut rectangles = HashMap::<(u16, u32, Vec<usize>), Vec<usize>>::new();
        for ((rule, output, left), mut rights) in by_left {
            rights.sort_unstable();
            rights.dedup();
            rectangles
                .entry((rule, output, rights))
                .or_default()
                .push(left);
        }
        for ((rule, output, right), left) in rectangles {
            batch
                .push_product(rule, output, left, right)
                .map_err(DrawError::Chart)?;
        }
        batch.sort_products();
        Ok(())
    }

    fn output_factor_groups(
        categories: &Categories<O>,
        identifiers: &[u32],
        slot: u8,
        representatives: Option<&HashMap<u32, Term>>,
    ) -> DrawResult<O, Vec<(OutputFactorGroupKey, OutputFactorGroup<O>)>> {
        let mut group_at = HashMap::<OutputFactorGroupKey, usize>::new();
        let mut groups = Vec::<(OutputFactorGroupKey, OutputFactorGroup<O>)>::new();
        for (position, identifier) in identifiers.iter().copied().enumerate() {
            let index = usize::try_from(identifier).unwrap_or(usize::MAX);
            let (owner, registry) = categories.terms.get(index).ok_or_else(|| {
                DrawError::Chart(ChartError::UnknownCategory {
                    category: identifier,
                })
            })?;
            let representative = representatives
                .map(|representatives| {
                    representatives.get(&identifier).cloned().ok_or_else(|| {
                        DrawError::internal(format!(
                            "category {identifier} has no reachability representative"
                        ))
                    })
                })
                .transpose()?;
            let factors = categories.output_factors.get(index).ok_or_else(|| {
                DrawError::internal(format!(
                    "category {identifier} has no output-factor metadata"
                ))
            })?;
            for factor in factors.iter().filter(|factor| factor.slot == slot) {
                let scope = Arc::as_ptr(registry).addr();
                let key = OutputFactorGroupKey {
                    rule: factor.rule,
                    key: factor.key,
                    factor: factor.factor.clone(),
                    representative: representative.as_ref().map_or(0, Term::serial),
                    scope,
                };
                let next = groups.len();
                let at = *group_at.entry(key.clone()).or_insert(next);
                if at == next {
                    groups.push((
                        key,
                        OutputFactorGroup {
                            positions: Vec::new(),
                            representative: representative.clone(),
                            owner: owner.clone(),
                            registry: registry.clone(),
                        },
                    ));
                }
                groups[at].1.positions.push(position);
            }
        }
        for (_, group) in &mut groups {
            group.positions.sort_unstable();
            group.positions.dedup();
        }
        Ok(groups)
    }

    fn complete_output_factor_rules(
        categories: &Categories<O>,
        program: &ChartProgram,
        request: &PairBatchRequest,
    ) -> DrawResult<O, HashSet<u16>> {
        let axis_joins = |identifiers: &[u32], slot: u8| {
            let mut joins = HashSet::new();
            for identifier in identifiers {
                let meta = program.category(*identifier).map_err(DrawError::Chart)?;
                for ((rule, held_slot), keys) in &meta.joins {
                    if *held_slot == slot {
                        joins.extend(keys.iter().map(|key| (*rule, *key)));
                    }
                }
            }
            Ok::<_, DrawError<O>>(joins)
        };
        let left_joins = axis_joins(&request.left, 0)?;
        let right_joins = axis_joins(&request.right, 1)?;
        let shared = left_joins
            .intersection(&right_joins)
            .copied()
            .collect::<HashSet<_>>();
        let mut complete = shared.iter().map(|(rule, _)| *rule).collect::<HashSet<_>>();
        for (identifiers, slot) in [(&request.left, 0_u8), (&request.right, 1_u8)] {
            for identifier in identifiers {
                let index = usize::try_from(*identifier).unwrap_or(usize::MAX);
                let meta = program.category(*identifier).map_err(DrawError::Chart)?;
                let factors = categories.output_factors.get(index).ok_or_else(|| {
                    DrawError::internal(format!(
                        "category {identifier} has no output-factor metadata"
                    ))
                })?;
                for ((rule, held_slot), keys) in &meta.joins {
                    if *held_slot != slot || !complete.contains(rule) {
                        continue;
                    }
                    if keys.iter().any(|key| {
                        shared.contains(&(*rule, *key))
                            && !factors.iter().any(|factor| {
                                factor.rule == *rule && factor.slot == slot && factor.key == *key
                            })
                    }) {
                        complete.remove(rule);
                    }
                }
            }
        }
        Ok(complete)
    }

    fn factor_candidates(
        left_groups: &[(OutputFactorGroupKey, OutputFactorGroup<O>)],
        right_groups: &[(OutputFactorGroupKey, OutputFactorGroup<O>)],
        complete_rules: &HashSet<u16>,
        witnesses: &mut BinaryPermitWitnesses<'_>,
        cache: &mut FactorOutputCache<O>,
    ) -> DrawResult<O, (Vec<ReachableOutputFactorCandidate>, HashSet<u16>)> {
        let mut right_by_join = HashMap::<(u16, u32), Vec<usize>>::new();
        for (at, (key, _)) in right_groups.iter().enumerate() {
            right_by_join
                .entry((key.rule, key.key))
                .or_default()
                .push(at);
        }
        let mut handled = HashSet::new();
        let mut candidates = Vec::new();
        for (left_at, (left_key, left)) in left_groups.iter().enumerate() {
            if !complete_rules.contains(&left_key.rule) {
                continue;
            }
            let Some(right_keys) = right_by_join.get(&(left_key.rule, left_key.key)) else {
                continue;
            };
            handled.insert(left_key.rule);
            for right_at in right_keys {
                let (right_key, right) = &right_groups[*right_at];
                let left_representative = left.representative.as_ref().ok_or_else(|| {
                    DrawError::internal("a grouped premise lost its reachability representative")
                })?;
                let right_representative = right.representative.as_ref().ok_or_else(|| {
                    DrawError::internal("a grouped premise lost its reachability representative")
                })?;
                if left_key.scope != right_key.scope
                    || !witnesses.permits(
                        left_key.rule,
                        left_representative.serial(),
                        right_representative.serial(),
                    )
                {
                    continue;
                }
                let output = cache.combined(left_key, right_key, left)?;
                candidates.push(ReachableOutputFactorCandidate {
                    rule: left_key.rule,
                    left_group: u32::try_from(left_at).map_err(|_| {
                        DrawError::invalid("the left factor-group inventory exceeds the id width")
                    })?,
                    right_group: u32::try_from(*right_at).map_err(|_| {
                        DrawError::invalid("the right factor-group inventory exceeds the id width")
                    })?,
                    output,
                });
            }
        }
        Ok((candidates, handled))
    }

    fn group_representative(
        groups: &[(OutputFactorGroupKey, OutputFactorGroup<O>)],
        at: u32,
    ) -> DrawResult<O, &OutputFactorGroup<O>> {
        groups
            .get(usize::try_from(at).unwrap_or(usize::MAX))
            .map(|(_, group)| group)
            .ok_or_else(|| DrawError::internal("a factor candidate has no premise group"))
    }

    fn pre_cartesian_products(
        &self,
        request: &PairBatchRequest,
        workers: Option<&ThreadPool>,
        reachability: &Reachability<O>,
        witnesses: &mut BinaryPermitWitnesses<'_>,
        representatives: &mut HashMap<u32, Term>,
        cache: &mut FactorOutputCache<O>,
    ) -> DrawResult<O, (PairBatch, HashSet<u16>)> {
        let (left_groups, right_groups, complete_rules) = {
            let categories = self.categories()?;
            let program = self.inner()?;
            (
                Self::output_factor_groups(&categories, &request.left, 0, Some(representatives))?,
                Self::output_factor_groups(&categories, &request.right, 1, Some(representatives))?,
                Self::complete_output_factor_rules(&categories, &program, request)?,
            )
        };
        let (mut candidates, handled) = Self::factor_candidates(
            &left_groups,
            &right_groups,
            &complete_rules,
            witnesses,
            cache,
        )?;
        if candidates.is_empty() {
            return Ok((PairBatch::default(), handled));
        }

        cache.represent(&candidates, reachability)?;
        let mut permit = BinaryOutputPermit {
            reachability,
            witnesses,
            representatives,
            cache: HashMap::new(),
        };
        let mut admissions = Vec::with_capacity(candidates.len());
        for candidate in &candidates {
            let representative = |group: &OutputFactorGroup<O>| {
                group.representative.clone().ok_or_else(|| {
                    DrawError::internal("a cached factor candidate lost its premise representative")
                })
            };
            let left = representative(Self::group_representative(
                &left_groups,
                candidate.left_group,
            )?)?;
            let right = representative(Self::group_representative(
                &right_groups,
                candidate.right_group,
            )?)?;
            let output = cache.representative(candidate.output)?;
            admissions.push(permit.permits_representatives(candidate.rule, &[left, right], output));
        }
        let mut admissions = admissions.into_iter();
        candidates.retain(|_| admissions.next().unwrap_or(false));
        cache.identify(&candidates, |terms| self.identify_cores(terms, workers))?;
        let mut by_left = OutputsByLeft::new();
        for candidate in candidates {
            let FactorIdentity::Known(identifier) = cache.get(candidate.output)?.identity else {
                continue;
            };
            representatives.insert(identifier, cache.representative(candidate.output)?.clone());
            let left = &Self::group_representative(&left_groups, candidate.left_group)?.positions;
            let right =
                &Self::group_representative(&right_groups, candidate.right_group)?.positions;
            for left in left {
                by_left
                    .entry((candidate.rule, identifier, *left))
                    .or_default()
                    .extend(right.iter().copied());
            }
        }
        let mut batch = PairBatch::default();
        Self::add_complete_pair_products(&mut batch, by_left)?;
        Ok((batch, handled))
    }

    fn unrestricted_pre_cartesian_products(
        &self,
        request: &PairBatchRequest,
        workers: Option<&ThreadPool>,
    ) -> DrawResult<O, (PairBatch, HashSet<u16>)> {
        let (left_groups, right_groups, complete_rules) = {
            let categories = self.categories()?;
            let program = self.inner()?;
            (
                Self::output_factor_groups(&categories, &request.left, 0, None)?,
                Self::output_factor_groups(&categories, &request.right, 1, None)?,
                Self::complete_output_factor_rules(&categories, &program, request)?,
            )
        };
        let mut right_by_join = HashMap::<(u16, u32), Vec<usize>>::new();
        for (at, (key, _)) in right_groups.iter().enumerate() {
            right_by_join
                .entry((key.rule, key.key))
                .or_default()
                .push(at);
        }
        let mut handled = HashSet::new();
        let mut candidates = Vec::new();
        for (left_key, left) in &left_groups {
            if !complete_rules.contains(&left_key.rule) {
                continue;
            }
            let Some(right_keys) = right_by_join.get(&(left_key.rule, left_key.key)) else {
                continue;
            };
            handled.insert(left_key.rule);
            for right_at in right_keys {
                let (right_key, right) = &right_groups[*right_at];
                if left_key.scope != right_key.scope {
                    continue;
                }
                let store = left.registry.store_for(&left.owner).ok_or_else(|| {
                    DrawError::invalid("a grouped rule premise has no live term store")
                })?;
                let output = combine_output_factors(&store, &left_key.factor, &right_key.factor)
                    .map_err(|error| DrawError::invalid(error.to_string()))?
                    .ok_or_else(|| {
                        DrawError::internal("a shared rule join has incompatible output factors")
                    })?;
                candidates.push(OutputFactorCandidate {
                    rule: left_key.rule,
                    left: left.positions.clone(),
                    right: right.positions.clone(),
                    output,
                    registry: left.registry.clone(),
                });
            }
        }
        let output_terms = candidates
            .iter()
            .map(|candidate| (candidate.output.clone(), candidate.registry.clone()))
            .collect::<Vec<_>>();
        let output_identifiers = self.identify_cores(&output_terms, workers)?;
        let mut by_left = OutputsByLeft::new();
        for (candidate, identifier) in candidates.into_iter().zip(output_identifiers) {
            let Some(identifier) = identifier else {
                continue;
            };
            for left in candidate.left {
                by_left
                    .entry((candidate.rule, identifier, left))
                    .or_default()
                    .extend(candidate.right.iter().copied());
            }
        }
        let mut batch = PairBatch::default();
        Self::add_complete_pair_products(&mut batch, by_left)?;
        Ok((batch, handled))
    }

    fn factor_complete_pairs(
        &self,
        context: &mut CompletePairContext<'_, '_, O>,
        mut batch: PairBatch,
        representatives: &mut HashMap<u32, Term>,
    ) -> DrawResult<O, PairBatch> {
        let plan = self.plan_complete_pairs(context.request, std::mem::take(&mut batch.pairs))?;
        batch.pairs.clone_from(&plan.residual);
        let (output_rows, output_identifiers) = self.apply_complete_pair_fragments(
            context,
            &plan.jobs,
            &plan.positions,
            representatives,
        )?;
        let mut permit = BinaryOutputPermit {
            reachability: context.reachability,
            witnesses: &mut *context.witnesses,
            representatives,
            cache: HashMap::new(),
        };
        let by_left =
            Self::resolved_complete_pairs(&plan, output_rows, &output_identifiers, &mut permit)?;
        Self::add_complete_pair_products(&mut batch, by_left)?;
        Ok(batch)
    }

    /// Move the complete rules of every pair into exact products, as the rooted
    /// batch does, so both batches emit a cell's steps in one order.
    fn factor_unrestricted_complete_pairs(
        &self,
        request: &PairBatchRequest,
        workers: Option<&ThreadPool>,
        mut batch: PairBatch,
    ) -> DrawResult<O, PairBatch> {
        let plan = self.plan_complete_pairs(request, std::mem::take(&mut batch.pairs))?;
        batch.pairs.clone_from(&plan.residual);
        let output_rows = self.apply_raw_fragment_jobs(&plan.jobs, workers)?;
        if output_rows.len() != plan.positions.len() {
            return Err(DrawError::internal(
                "a native rule batch returned an inconsistent row count",
            ));
        }
        let output_terms = output_rows
            .iter()
            .flat_map(|row| row.iter())
            .flat_map(|(_, terms, registry)| {
                terms.iter().cloned().map(|term| (term, registry.clone()))
            })
            .collect::<Vec<_>>();
        let output_identifiers = self.identify_cores(&output_terms, workers)?;
        if output_identifiers.len() != output_terms.len() {
            return Err(DrawError::internal(
                "the category inventory returned an inconsistent output count",
            ));
        }
        let mut by_left = OutputsByLeft::new();
        for (left, right, _, rule, outputs) in &plan.cached {
            for output in outputs.iter().copied() {
                by_left
                    .entry((*rule, output, *left))
                    .or_default()
                    .push(*right);
            }
        }
        let mut identifiers = output_identifiers.into_iter();
        for ((left, right, _), row) in plan.positions.iter().zip(output_rows) {
            for (rule, terms, _) in row {
                for output in identifiers.by_ref().take(terms.len()).flatten() {
                    by_left
                        .entry((rule, output, *left))
                        .or_default()
                        .push(*right);
                }
            }
        }
        Self::add_complete_pair_products(&mut batch, by_left)?;
        Ok(batch)
    }

    fn finish_step_batch<H: ResidualRules>(
        &self,
        requests: &[StepBatchRequest],
        plans: StepBatchPlans,
        callbacks: &mut H,
    ) -> DrawResult<O, Vec<StepBatchEntry>> {
        let mut entries = Vec::new();
        let mut residual_work: Vec<ResidualStepWork> = Vec::new();
        {
            let mut program = self.inner()?;
            for (request_at, (request, row_plans)) in requests.iter().zip(plans).enumerate() {
                let memoizes = program
                    .candidates_are_complete(&request.candidates)
                    .map_err(DrawError::Chart)?;
                for planned in row_plans {
                    let PlannedStepRow { row, plan } = planned;
                    let steps = match plan {
                        StepPlan::Cached(steps) => steps,
                        StepPlan::Draw { native, residual } if residual.is_empty() => {
                            let mut steps = Vec::new();
                            for (rule, tag) in native {
                                let outputs = program
                                    .cached_outputs(rule, &request.inputs)
                                    .ok_or_else(|| {
                                        DrawError::internal(
                                            "a native rule plan is missing its cached outputs",
                                        )
                                    })?;
                                steps.extend(outputs.iter().copied().map(|output| Step {
                                    rule,
                                    output,
                                    tag,
                                }));
                            }
                            if memoizes {
                                let tags = request.tag_rows.get(row).ok_or_else(|| {
                                    DrawError::internal("a native rule plan lost its child-tag row")
                                })?;
                                program.remember(
                                    request.inputs.clone(),
                                    tags.clone(),
                                    request.candidates.clone(),
                                    steps.clone(),
                                );
                            }
                            steps
                        }
                        plan @ StepPlan::Draw { .. } => {
                            residual_work.push((request_at, row, plan));
                            continue;
                        }
                    };
                    for step in steps {
                        record_step(&mut entries, request_at, row, step);
                    }
                }
            }
            drop(program);
        }

        for (request_at, row, plan) in residual_work {
            let request = requests.get(request_at).ok_or_else(|| {
                DrawError::internal("a residual rule plan lost its batch request")
            })?;
            let tags = request.tag_rows.get(row).cloned().ok_or_else(|| {
                DrawError::internal("a residual rule plan lost its child-tag row")
            })?;
            let exact = StepRequest {
                inputs: request.inputs.clone(),
                tags,
                span: request.span,
                candidates: request.candidates.clone(),
            };
            let steps = self.finish_step_plan(&exact, plan, callbacks)?;
            for step in steps {
                record_step(&mut entries, request_at, row, step);
            }
        }
        let mut reachable = Vec::with_capacity(entries.len());
        for entry in entries {
            let span = requests
                .get(entry.request)
                .ok_or_else(|| DrawError::internal("a resolved rule step lost its batch request"))?
                .span;
            if self.keeps(span, entry.step.output)? {
                reachable.push(entry);
            }
        }
        reachable.sort_by_key(|entry| (entry.request, entry.row));
        Ok(reachable)
    }

    fn draw_closure<H: ResidualRules>(
        &self,
        items: &[ItemId],
        span: Span,
        depth: usize,
        callbacks: &mut H,
        workers: Option<&ThreadPool>,
    ) -> DrawResult<O, UnaryClosure> {
        unary_closure(items, depth, |children| {
            let requests = {
                let program = self.inner()?;
                children
                    .iter()
                    .map(|child| {
                        Ok(StepBatchRequest {
                            inputs: vec![child.category],
                            tag_rows: vec![vec![child.tag]],
                            span,
                            parts: vec![span],
                            candidates: program
                                .unary_candidates(child.category)
                                .map_err(DrawError::Chart)?
                                .to_vec(),
                        })
                    })
                    .collect::<DrawResult<O, Vec<_>>>()?
            };
            self.draw_steps_batch(&requests, callbacks, workers)
                .map(|entries| {
                    let mut rows = vec![Vec::new(); requests.len()];
                    for entry in entries {
                        if entry.row == 0 {
                            rows[entry.request].push(entry.step);
                        }
                    }
                    rows
                })
        })
    }
}

/// How a scorer's local edge weights vary across spans and splits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeScoring {
    /// A weight may depend on its span or split.
    Dependent,
    /// A weight is invariant across spans and splits.
    SpanFree,
    /// Every rule edge has the carrier's multiplicative identity.
    UnitEdges,
}

enum EdgeWeights<W> {
    Dependent,
    SpanFree(HashMap<(u16, Vec<u32>, u32), W>),
    Unit(W),
}

fn edge_weights<W: Clone, S: Semiring<Weight = W>>(
    scoring: EdgeScoring,
    semiring: &S,
) -> EdgeWeights<W> {
    match scoring {
        EdgeScoring::Dependent => EdgeWeights::Dependent,
        EdgeScoring::SpanFree => EdgeWeights::SpanFree(HashMap::new()),
        EdgeScoring::UnitEdges => EdgeWeights::Unit(semiring.one()),
    }
}

/// The inside and forest resolver that draws rule steps through a [`DrawingProgram`].
///
/// Rule outputs, reachability and output-factor products come from the
/// program; local weights, residual rules and kept items come from the host.
pub struct DrawingResolver<'a, O, W, H> {
    program: &'a DrawingProgram<O>,
    callbacks: H,
    weights: EdgeWeights<W>,
    reachability: HashMap<(Span, u32), bool>,
    category_representatives: HashMap<u32, Term>,
    represented_categories: usize,
    factor_outputs: FactorOutputCache<O>,
}

impl<O: Scope, W, H> DrawingResolver<'_, O, W, H> {
    fn reachable_pair_batch(
        &mut self,
        batch: &PairBatchRequest,
        workers: Option<&ThreadPool>,
        factor_units: bool,
        reachability: &Reachability<O>,
    ) -> DrawResult<O, PairBatch> {
        let program = self.program;
        let representatives = &mut self.category_representatives;
        let cache = &mut self.factor_outputs;
        let left = batch
            .left
            .iter()
            .map(|identifier| representatives[identifier].clone())
            .collect::<Vec<_>>();
        let right = batch
            .right
            .iter()
            .map(|identifier| representatives[identifier].clone())
            .collect::<Vec<_>>();
        let permits = reachability.binary_permits(batch.span, &batch.parts, &left, &right);
        let mut witnesses = permits.witnesses();
        let (products, excluded) = if factor_units {
            program.pre_cartesian_products(
                batch,
                workers,
                reachability,
                &mut witnesses,
                representatives,
                cache,
            )?
        } else {
            (PairBatch::default(), HashSet::new())
        };
        let mut permit = BinaryOutputPermit {
            reachability,
            witnesses: &mut witnesses,
            representatives,
            cache: HashMap::new(),
        };
        let mut permit_error = None;
        let mut selected = {
            let held = program.inner()?;
            held.pair_batch_where_excluding(
                &batch.left,
                &batch.right,
                factor_units,
                &excluded,
                |rule, left, right, output| {
                    let admitted = match output {
                        Some(output) => permit.permits(rule, [left, right], output),
                        None => permit.permits_pair(rule, [left, right]),
                    };
                    match admitted {
                        Ok(admitted) => admitted,
                        Err(error) => {
                            permit_error = Some(error);
                            false
                        }
                    }
                },
            )
        }
        .map_err(DrawError::Chart)?;
        if let Some(error) = permit_error {
            return Err(error);
        }
        drop(permit);
        selected
            .extend_products(products)
            .map_err(DrawError::Chart)?;
        let selected = if factor_units {
            let mut context = CompletePairContext {
                request: batch,
                workers,
                reachability,
                witnesses: &mut witnesses,
            };
            program.factor_complete_pairs(&mut context, selected, representatives)?
        } else {
            selected
        };
        Ok(selected)
    }

    fn unrestricted_pair_batches(
        &self,
        batches: &[PairBatchRequest],
        workers: Option<&ThreadPool>,
        factor_units: bool,
    ) -> DrawResult<O, Vec<PairBatch>> {
        if factor_units {
            return batches
                .iter()
                .map(|batch| {
                    let (products, excluded) = self
                        .program
                        .unrestricted_pre_cartesian_products(batch, workers)?;
                    let mut selected = self
                        .program
                        .inner()?
                        .pair_batch_where_excluding(
                            &batch.left,
                            &batch.right,
                            factor_units,
                            &excluded,
                            |_, _, _, _| true,
                        )
                        .map_err(DrawError::Chart)?;
                    selected
                        .extend_products(products)
                        .map_err(DrawError::Chart)?;
                    self.program
                        .factor_unrestricted_complete_pairs(batch, workers, selected)
                })
                .collect();
        }
        let category_batches = batches
            .iter()
            .map(|batch| (batch.left.clone(), batch.right.clone()))
            .collect::<Vec<_>>();
        self.program
            .inner()?
            .pair_batches(&category_batches, workers, factor_units)
            .map_err(DrawError::Chart)
    }
}

impl<O: Scope, W: Clone, H: DrawingHost<W>> InsideResolver<W> for DrawingResolver<'_, O, W, H> {
    type Error = DrawError<O>;

    fn rules_of_arity(&mut self, arity: usize) -> DrawResult<O, Vec<u16>> {
        Ok(self.program.inner()?.rules_of_arity(arity).to_vec())
    }

    fn pair_batches(
        &mut self,
        batches: &[PairBatchRequest],
        workers: Option<&ThreadPool>,
        factor_units: bool,
    ) -> DrawResult<O, Vec<PairBatch>> {
        let factor_units = factor_units && !self.program.normalizes;
        let reachability = self
            .program
            .reachability
            .lock()
            .map_err(|_| DrawError::internal("the chart reachability lock is poisoned"))?
            .clone();
        if let Some(reachability) = reachability {
            self.program.extend_category_representatives(
                &reachability,
                &mut self.category_representatives,
                &mut self.represented_categories,
            )?;
            batches
                .iter()
                .map(|batch| self.reachable_pair_batch(batch, workers, factor_units, &reachability))
                .collect()
        } else {
            self.unrestricted_pair_batches(batches, workers, factor_units)
        }
    }

    fn local_weight_is_split_invariant(&self) -> bool {
        matches!(
            self.weights,
            EdgeWeights::SpanFree(_) | EdgeWeights::Unit(_)
        )
    }

    fn local_weights_are_unit(&self) -> bool {
        matches!(self.weights, EdgeWeights::Unit(_))
    }

    fn permit_key(&self, parts: &[Span]) -> Vec<Option<Span>> {
        parts
            .iter()
            .map(|part| (part.1 - part.0 == 1).then_some(*part))
            .collect()
    }

    fn decision(&mut self, rule: u16, tags: &[u8]) -> DrawResult<O, NormalDecision> {
        self.program
            .inner()?
            .decision(rule, tags)
            .map_err(DrawError::Chart)
    }

    fn admit_rows(
        &mut self,
        rule: u16,
        tags: &[u8],
        input_pairs: &[[u32; 2]],
        output: u32,
    ) -> DrawResult<O, Vec<bool>> {
        self.callbacks
            .admit_rows(rule, tags, input_pairs, output)
            .map_err(DrawError::Host)
    }

    fn keeps(&mut self, span: Span, category: u32) -> DrawResult<O, bool> {
        if let Some(keeps) = self.reachability.get(&(span, category)) {
            return Ok(*keeps);
        }
        let keeps = self.program.keeps(span, category)?;
        self.reachability.insert((span, category), keeps);
        Ok(keeps)
    }

    fn permits(
        &mut self,
        span: Span,
        parts: &[Span],
        rule: u16,
        inputs: &[u32],
    ) -> DrawResult<O, bool> {
        self.program.permits(span, parts, rule, inputs)
    }

    fn steps(
        &mut self,
        requests: &[StepBatchRequest],
        workers: Option<&ThreadPool>,
    ) -> DrawResult<O, Vec<StepBatchEntry>> {
        let mut reachable = requests.to_vec();
        for request in &mut reachable {
            if request.inputs.len() == 2 {
                continue;
            }
            let mut candidates = Vec::with_capacity(request.candidates.len());
            for rule in &request.candidates {
                if self
                    .program
                    .permits(request.span, &request.parts, *rule, &request.inputs)?
                {
                    candidates.push(*rule);
                }
            }
            request.candidates = candidates;
        }
        let entries = self
            .program
            .draw_steps_batch(&reachable, &mut self.callbacks, workers)?;
        let mut permitted = Vec::with_capacity(entries.len());
        for entry in entries {
            let Some(request) = reachable.get(entry.request) else {
                return Err(DrawError::internal(
                    "a resolved rule step lost its batch request",
                ));
            };
            if request.inputs.len() != 2
                || self.program.permits_output(
                    request.span,
                    &request.parts,
                    entry.step.rule,
                    &request.inputs,
                    entry.step.output,
                )?
            {
                permitted.push(entry);
            }
        }
        Ok(permitted)
    }

    fn closure(
        &mut self,
        arrival: &[ItemId],
        span: Span,
        depth: usize,
        workers: Option<&ThreadPool>,
    ) -> DrawResult<O, UnaryClosure> {
        self.program
            .draw_closure(arrival, span, depth, &mut self.callbacks, workers)
    }

    fn local_weight(
        &mut self,
        rule: u16,
        inputs: &[u32],
        output: u32,
        span: Span,
        split: Option<usize>,
    ) -> DrawResult<O, W> {
        let key = (rule, inputs.to_vec(), output);

        match &self.weights {
            EdgeWeights::Unit(weight) => return Ok(weight.clone()),
            EdgeWeights::SpanFree(scores) => {
                if let Some(weight) = scores.get(&key) {
                    return Ok(weight.clone());
                }
            }
            EdgeWeights::Dependent => {}
        }

        let weight = self
            .callbacks
            .weight(rule, inputs, output, span, split)
            .map_err(DrawError::Host)?;

        if let EdgeWeights::SpanFree(scores) = &mut self.weights {
            scores.insert(key, weight.clone());
        }
        Ok(weight)
    }

    fn finish_width(&mut self) -> DrawResult<O, ()> {
        self.callbacks.finish_width().map_err(DrawError::Host)
    }
}

impl<O: Scope, W: Clone, H: ForestHost<W>> ForestResolver<W, u32> for DrawingResolver<'_, O, W, H> {
    fn label_order(&mut self, rule: u16) -> DrawResult<O, (u32, usize)> {
        let order = self
            .program
            .inner()?
            .rule(rule)
            .map_err(DrawError::Chart)?
            .order;
        Ok((u32::from(rule), order))
    }

    fn root_keep(
        &mut self,
        span: Span,
        items: &[(ItemId, W)],
        closure: &UnaryClosure,
    ) -> DrawResult<O, Option<Vec<ItemId>>> {
        self.callbacks
            .root_keep(span, items, closure)
            .map_err(DrawError::Host)
    }

    fn prune_keep(
        &mut self,
        span: Span,
        items: &[(ItemId, W)],
    ) -> DrawResult<O, Option<Vec<ItemId>>> {
        self.callbacks
            .prune_keep(span, items)
            .map_err(DrawError::Host)
    }
}

impl<O> Default for Categories<O> {
    fn default() -> Self {
        Self {
            terms: Vec::new(),
            output_factors: Vec::new(),
            identifiers: HashMap::new(),
            join_ids: HashMap::new(),
            join_keys: Vec::new(),
            rejected: HashSet::new(),
        }
    }
}

impl<O> Default for CompletePairPlan<O> {
    fn default() -> Self {
        Self {
            jobs: Vec::new(),
            positions: Vec::new(),
            cached: Vec::new(),
            residual: Vec::new(),
        }
    }
}

impl<O: Scope> DrawingProgram<O> {
    /// Build a drawing program over one ordered rule inventory.
    ///
    /// # Errors
    ///
    /// Returns [`DrawError::Invalid`] when rules or tags exceed the dense id
    /// widths, a decision row has the wrong arity or names an unknown tag or
    /// repeats, the category measure repeats a constructor, a category bound is
    /// zero, or an arity bound is declared without a category measure.
    pub fn new(
        rules: Vec<DrawingRule>,
        tags: usize,
        program: Arc<ScopedRuleProgram<O>>,
        normalizes: bool,
        category_measure: Vec<(String, usize)>,
        category_bounds: (Option<usize>, Option<usize>),
    ) -> DrawResult<O, Self> {
        if rules.len() > usize::from(u16::MAX) + 1 {
            return Err(DrawError::invalid(format!(
                "a chart program has at most {} rules, not {}",
                usize::from(u16::MAX) + 1,
                rules.len()
            )));
        }
        if tags > usize::from(u8::MAX) + 1 {
            return Err(DrawError::invalid(format!(
                "a chart program has at most {} tags, not {tags}",
                usize::from(u8::MAX) + 1
            )));
        }
        let shapes = rules
            .iter()
            .map(|rule| (rule.arity, rule.complete))
            .collect::<Vec<_>>();
        let (unjoined, joined) = program.binary_partition(&shapes)?;
        let mut decisions = HashMap::new();
        let mut specs = Vec::with_capacity(rules.len());
        for (position, declared) in rules.into_iter().enumerate() {
            let rule = u16::try_from(position)
                .map_err(|_| DrawError::invalid(format!("rule position {position} exceeds u16")))?;
            for (children, decision) in declared.decisions {
                if children.len() != declared.arity {
                    return Err(DrawError::invalid(format!(
                        "rule id {rule} has arity {}, not decision arity {}",
                        declared.arity,
                        children.len()
                    )));
                }
                let output = match decision {
                    NormalDecision::Admit(output) | NormalDecision::Premises(output) => {
                        Some(output)
                    }
                    NormalDecision::Refuse => None,
                };
                if children
                    .iter()
                    .chain(output.as_ref())
                    .any(|tag| usize::from(*tag) >= tags)
                {
                    return Err(DrawError::invalid(format!(
                        "rule id {rule} has a decision outside {tags} normal-form tags"
                    )));
                }
                if decisions.insert((rule, children), decision).is_some() {
                    return Err(DrawError::invalid(format!(
                        "rule id {rule} repeats a child-tag decision"
                    )));
                }
            }
            specs.push(RuleSpec {
                arity: declared.arity,
                order: declared.order,
                complete: declared.complete,
            });
        }
        let mut measure = HashMap::new();
        for (operator, slot) in category_measure {
            if measure.insert(operator.clone(), slot).is_some() {
                return Err(DrawError::invalid(format!(
                    "a category measure repeats constructor {operator:?}"
                )));
            }
        }
        if category_bounds.0.is_some() && measure.is_empty() {
            return Err(DrawError::invalid(
                "max_category_arity needs the grammar's category constructors",
            ));
        }
        let inner = ChartProgram::new(
            specs,
            tags,
            unjoined,
            joined,
            decisions,
            CategoryBounds {
                max_arity: positive_bound("max_category_arity", category_bounds.0)?,
                max_size: positive_bound("max_category_size", category_bounds.1)?,
            },
        );

        Ok(Self {
            inner: Mutex::new(inner),
            categories: Mutex::new(Categories::default()),
            fragment_outputs: Mutex::new(HashMap::new()),
            reachability: Mutex::new(None),
            category_measure: measure,
            shapes,
            rules: program,
            normalizes,
        })
    }

    /// Install the exact per-span category demands for one root-directed parse.
    ///
    /// # Errors
    ///
    /// Returns [`DrawError::Invalid`] when the reachability rules do not cover
    /// the chart rules or their template cannot be compiled.
    pub fn configure_reachability(&self, inputs: ReachabilityInputs<O>) -> DrawResult<O, ()> {
        if inputs.rules.len() != self.shapes.len() {
            return Err(DrawError::invalid(format!(
                "{} reachability rules were supplied for {} chart rules",
                inputs.rules.len(),
                self.shapes.len()
            )));
        }
        let template = self
            .rules
            .reachability_template(&inputs.rules, inputs.default_sort)?;
        let plan = ReachabilityPlan::from_template(
            &template,
            inputs.tokens,
            &inputs.roots,
            &inputs.lexical,
            inputs.unary_depth,
            inputs.refinement_pairs,
        );
        self.install_reachability(Reachability {
            plan,
            kernel: inputs.kernel,
            representatives: Mutex::new(inputs.table),
            registry: inputs.scope,
        })
    }

    /// Return whether one registered category remains root-reachable at a span.
    ///
    /// # Errors
    ///
    /// Returns a refusal when the category is unknown or a lock is poisoned.
    pub fn reachable(&self, span: Span, category: u32) -> DrawResult<O, bool> {
        self.keeps(span, category)
    }

    /// Return the dense identifier of one exact category, registering it when new.
    ///
    /// `None` means the declared category bounds refuse the category.
    ///
    /// # Errors
    ///
    /// Returns a refusal when a rule refuses the category or a lock is poisoned.
    pub fn identify(&self, term: Term, scope: Arc<O>) -> DrawResult<O, Option<u32>> {
        self.identify_core(term, scope)
    }

    /// Return how many distinct exact categories the declared bounds refused.
    ///
    /// # Errors
    ///
    /// Returns [`DrawError::Internal`] when the program lock is poisoned.
    pub fn refused_categories(&self) -> DrawResult<O, usize> {
        Ok(self.inner()?.refused_categories())
    }

    /// Return one registered category's dense identifier when present.
    ///
    /// # Errors
    ///
    /// Returns [`DrawError::Internal`] when the category lock is poisoned.
    pub fn lookup(&self, term: &Term) -> DrawResult<O, Option<u32>> {
        Ok(self.categories()?.identifiers.get(term).copied())
    }

    /// Return the exact category at one dense identifier, with its scope.
    ///
    /// # Errors
    ///
    /// Returns [`DrawError::Chart`] when the identifier is unknown.
    pub fn term(&self, identifier: u32) -> DrawResult<O, (Term, Arc<O>)> {
        let categories = self.categories()?;
        categories
            .terms
            .get(usize::try_from(identifier).unwrap_or(usize::MAX))
            .cloned()
            .ok_or(DrawError::Chart(ChartError::UnknownCategory {
                category: identifier,
            }))
    }

    /// Return candidate rule identifiers of one arity.
    ///
    /// # Errors
    ///
    /// Returns [`DrawError::Internal`] when the program lock is poisoned.
    pub fn rules_of_arity(&self, arity: usize) -> DrawResult<O, Vec<u16>> {
        Ok(self.inner()?.rules_of_arity(arity).to_vec())
    }

    /// Return memoized normalized outputs for one complete rule application.
    ///
    /// # Errors
    ///
    /// Returns [`DrawError::Chart`] when the rule or a category is unknown.
    pub fn cached_outputs(&self, rule: u16, inputs: &[u32]) -> DrawResult<O, Option<Vec<u32>>> {
        let program = self.inner()?;
        program.rule(rule).map_err(DrawError::Chart)?;
        for category in inputs {
            program.category(*category).map_err(DrawError::Chart)?;
        }
        Ok(program.cached_outputs(rule, inputs).map(<[u32]>::to_vec))
    }

    /// Memoize normalized outputs for one complete rule application.
    ///
    /// # Errors
    ///
    /// Returns [`DrawError::Chart`] when the rule or a category is unknown.
    pub fn remember_outputs(
        &self,
        rule: u16,
        inputs: Vec<u32>,
        outputs: Vec<u32>,
    ) -> DrawResult<O, ()> {
        self.inner()?
            .remember_outputs(rule, inputs, outputs)
            .map_err(DrawError::Chart)
    }

    /// Enumerate indexed binary pairs in Cartesian order.
    ///
    /// # Errors
    ///
    /// Returns [`DrawError::Chart`] when a category is unknown.
    pub fn pairs(&self, left: &[u32], right: &[u32]) -> DrawResult<O, Vec<Pair>> {
        self.inner()?.pairs(left, right).map_err(DrawError::Chart)
    }

    /// Draw one exact premise tuple, resolving residual rules through the host.
    ///
    /// # Errors
    ///
    /// Returns a refusal from the chart, a rule kernel or the host.
    pub fn steps<H: ResidualRules>(
        &self,
        inputs: Vec<u32>,
        tags: Vec<u8>,
        span: Span,
        candidates: &[u16],
        callbacks: &mut H,
    ) -> DrawResult<O, Vec<Step>> {
        self.draw_steps(inputs, tags, span, candidates, callbacks)
    }

    /// Close unary rules over one cell up to the supplied depth.
    ///
    /// # Errors
    ///
    /// Returns a refusal from the chart, a rule kernel or the host.
    pub fn closure<H: ResidualRules>(
        &self,
        items: &[ItemId],
        span: Span,
        depth: usize,
        callbacks: &mut H,
    ) -> DrawResult<O, UnaryClosure> {
        self.draw_closure(items, span, depth, callbacks, None)
    }

    /// Return the number of rules and of normal-form tags.
    ///
    /// # Errors
    ///
    /// Returns [`DrawError::Internal`] when the program lock is poisoned.
    pub fn counts(&self) -> DrawResult<O, (usize, usize)> {
        let inner = self.inner()?;
        Ok((inner.rule_count(), inner.tag_count()))
    }
}

impl<'a, O: Scope, W: Clone, H> DrawingResolver<'a, O, W, H> {
    /// Start a resolver for one sweep over a drawing program.
    pub fn new<S: Semiring<Weight = W>>(
        program: &'a DrawingProgram<O>,
        callbacks: H,
        scoring: EdgeScoring,
        semiring: &S,
    ) -> Self {
        Self {
            program,
            callbacks,
            weights: edge_weights(scoring, semiring),
            reachability: HashMap::new(),
            category_representatives: HashMap::new(),
            represented_categories: 0,
            factor_outputs: FactorOutputCache::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semiring::{Count, Counting};
    use crate::term::TermStore;

    #[derive(Debug)]
    struct Owner;

    impl Scope for Owner {
        fn store_for(&self, _term: &Term) -> Option<TermStore> {
            None
        }
    }

    #[test]
    fn unit_scoring_answers_every_edge_with_the_identity() {
        assert!(matches!(
            edge_weights(EdgeScoring::UnitEdges, &Counting),
            EdgeWeights::Unit(Count::Small(1))
        ));
        assert!(matches!(
            edge_weights::<Count, _>(EdgeScoring::SpanFree, &Counting),
            EdgeWeights::SpanFree(scores) if scores.is_empty()
        ));
        assert!(matches!(
            edge_weights::<Count, _>(EdgeScoring::Dependent, &Counting),
            EdgeWeights::Dependent
        ));
    }

    #[test]
    fn an_empty_factor_cache_refuses_an_unknown_output() {
        let cache = FactorOutputCache::<Owner>::new();
        assert!(matches!(cache.get(0), Err(DrawError::Internal(_))));
        assert!(matches!(
            cache.representative(0),
            Err(DrawError::Internal(_))
        ));
    }

    #[test]
    fn a_zero_category_bound_is_refused() {
        assert!(matches!(
            positive_bound::<Owner>("max_category_size", Some(0)),
            Err(DrawError::Invalid(_))
        ));
        assert!(matches!(
            positive_bound::<Owner>("max_category_size", None),
            Ok(None)
        ));
    }
}
