//! Behavioral tests for the dense-id chart-drawing kernel.

use std::collections::{HashMap, HashSet};
use std::convert::Infallible;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};

use lofi_chart::chart::{
    CategoryBounds, CategoryDimensions, CategoryMeta, CellExecution, ChartError, ChartProgram,
    ForestEdge, ForestKey, ForestResolver, ForestTable, InsideResolver, InsideTable, ItemId,
    MissBatch, NormalDecision, Pair, PairBatch, PairBatchRequest, ProductAlternative, RuleSpec,
    Span, Step, StepBatchEntry, StepBatchRequest, StepPlan, SweepError, UnaryClosure,
    unary_closure,
};
use lofi_chart::semiring::{
    Boolean, Count, Counting, Expectation, KBest, LogInside, Semiring, Viterbi,
};

#[test]
fn factored_pair_axes_reject_empty_rows_and_remap_owned_ranges()
-> Result<(), Box<dyn std::error::Error>> {
    let mut selected = PairBatch::default();
    assert_eq!(
        selected.push_product(0, 1, Vec::new(), vec![0]),
        Err(ChartError::EmptyFactoredAxis)
    );
    assert_eq!(
        selected.push_product(0, 1, vec![0], Vec::new()),
        Err(ChartError::EmptyFactoredAxis)
    );
    assert_eq!(selected.products.len(), 0);
    assert_eq!(selected.axis_items(), 0);

    selected.push_product(1, 8, vec![2, 0, 2], vec![3, 1])?;
    let mut incoming = PairBatch::default();
    incoming.push_product(0, 7, vec![4, 2], vec![5])?;
    selected.extend_products(incoming)?;
    selected.sort_products();

    let rows = selected
        .products
        .iter()
        .enumerate()
        .map(|(product_at, product)| {
            let [left, right] = selected.product_axes_at(product_at)?;
            Ok::<_, ChartError>((product.rule, product.output, left.to_vec(), right.to_vec()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(
        rows,
        vec![(1, 8, vec![0, 2], vec![1, 3]), (0, 7, vec![2, 4], vec![5]),]
    );
    assert_eq!(
        selected.product_axes_at(2),
        Err(ChartError::InvalidFactoredProduct {
            product: 2,
            products: 2,
        })
    );
    Ok(())
}

#[derive(Default)]
struct Resolver {
    lexical_permits: bool,
    requests: usize,
    widths: usize,
}

#[derive(Default)]
struct CountingOperations {
    times: AtomicUsize,
}

impl Semiring for CountingOperations {
    type Weight = Count;
    type Error = Infallible;

    fn zero(&self) -> Self::Weight {
        Counting.zero()
    }

    fn one(&self) -> Self::Weight {
        Counting.one()
    }

    fn plus_assign(
        &self,
        accumulator: &mut Self::Weight,
        weight: &Self::Weight,
    ) -> Result<(), Self::Error> {
        Counting.plus_assign(accumulator, weight)
    }

    fn times(
        &self,
        left: &Self::Weight,
        right: &Self::Weight,
    ) -> Result<Self::Weight, Self::Error> {
        self.times.fetch_add(1, Ordering::Relaxed);
        Counting.times(left, right)
    }
}

struct MultiOutputResolver {
    kept: Option<u32>,
    permitted: bool,
    mixed: bool,
}

impl InsideResolver<Count> for MultiOutputResolver {
    type Error = ChartError;

    fn rules_of_arity(&mut self, arity: usize) -> Result<Vec<u16>, Self::Error> {
        Ok((arity == 2).then_some(0).into_iter().collect())
    }

    fn pair_batches(
        &mut self,
        batches: &[PairBatchRequest],
        _workers: Option<&rayon::ThreadPool>,
        _factor_units: bool,
    ) -> Result<Vec<PairBatch>, Self::Error> {
        if !self.permitted {
            return Ok(batches.iter().map(|_| PairBatch::default()).collect());
        }
        batches
            .iter()
            .map(|batch| -> Result<PairBatch, Self::Error> {
                let pairs = batch
                    .left
                    .iter()
                    .enumerate()
                    .flat_map(|(left, _)| {
                        batch.right.iter().enumerate().map(move |(right, _)| Pair {
                            left,
                            right,
                            candidates: Vec::new(),
                        })
                    })
                    .collect::<Vec<_>>();
                let mut selected = PairBatch::from_pairs(pairs);
                for output in [1, 2]
                    .into_iter()
                    .filter(|output| self.kept.is_none_or(|kept| kept == *output))
                {
                    selected.push_product(
                        0,
                        output,
                        (0..batch.left.len()).collect(),
                        (0..batch.right.len()).collect(),
                    )?;
                }
                Ok(selected)
            })
            .collect()
    }

    fn local_weight_is_split_invariant(&self) -> bool {
        true
    }

    fn local_weights_are_unit(&self) -> bool {
        true
    }

    fn permit_key(&self, parts: &[Span]) -> Vec<Option<Span>> {
        parts.iter().copied().map(Some).collect()
    }

    fn decision(&mut self, _rule: u16, tags: &[u8]) -> Result<NormalDecision, Self::Error> {
        Ok(if !self.mixed || tags == [0, 0] {
            NormalDecision::Admit(0)
        } else if tags == [2, 2] {
            NormalDecision::Premises(2)
        } else {
            NormalDecision::Refuse
        })
    }

    fn admit_rows(
        &mut self,
        _rule: u16,
        tags: &[u8],
        input_pairs: &[[u32; 2]],
        _output: u32,
    ) -> Result<Vec<bool>, Self::Error> {
        Ok(input_pairs
            .iter()
            .map(|inputs| {
                !self.mixed || tags != [2, 2] || matches!(inputs, [10 | 11, 20 | 21] | [12, 22])
            })
            .collect())
    }

    fn keeps(&mut self, _span: Span, category: u32) -> Result<bool, Self::Error> {
        Ok(self.kept.is_none_or(|kept| kept == category))
    }

    fn permits(
        &mut self,
        _span: Span,
        _parts: &[Span],
        _rule: u16,
        _inputs: &[u32],
    ) -> Result<bool, Self::Error> {
        Ok(self.permitted)
    }

    fn steps(
        &mut self,
        _requests: &[StepBatchRequest],
        _workers: Option<&rayon::ThreadPool>,
    ) -> Result<Vec<StepBatchEntry>, Self::Error> {
        Ok(Vec::new())
    }

    fn closure(
        &mut self,
        _arrival: &[ItemId],
        _span: Span,
        _depth: usize,
        _workers: Option<&rayon::ThreadPool>,
    ) -> Result<UnaryClosure, Self::Error> {
        Ok(UnaryClosure {
            edges: Vec::new(),
            concluded: Vec::new(),
            truncated: false,
        })
    }

    fn local_weight(
        &mut self,
        _rule: u16,
        _inputs: &[u32],
        _output: u32,
        _span: Span,
        _split: Option<usize>,
    ) -> Result<Count, Self::Error> {
        Ok(Count::Small(1))
    }

    fn finish_width(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }
}

impl ForestResolver<Count, u16> for MultiOutputResolver {
    fn label_order(&mut self, rule: u16) -> Result<(u16, usize), Self::Error> {
        Ok((rule, 0))
    }

    fn root_keep(
        &mut self,
        _span: Span,
        _items: &[(ItemId, Count)],
        _closure: &UnaryClosure,
    ) -> Result<Option<Vec<ItemId>>, Self::Error> {
        Ok(None)
    }

    fn prune_keep(
        &mut self,
        _span: Span,
        _items: &[(ItemId, Count)],
    ) -> Result<Option<Vec<ItemId>>, Self::Error> {
        Ok(None)
    }
}

impl InsideResolver<Count> for Resolver {
    type Error = Infallible;

    fn rules_of_arity(&mut self, arity: usize) -> Result<Vec<u16>, Self::Error> {
        Ok((arity == 2).then_some(1).into_iter().collect())
    }

    fn pair_batches(
        &mut self,
        batches: &[PairBatchRequest],
        _workers: Option<&rayon::ThreadPool>,
        _factor_units: bool,
    ) -> Result<Vec<PairBatch>, Self::Error> {
        Ok(batches
            .iter()
            .map(|batch| {
                let pairs = batch
                    .left
                    .iter()
                    .enumerate()
                    .flat_map(|(left_at, _)| {
                        batch
                            .right
                            .iter()
                            .enumerate()
                            .map(move |(right_at, _)| Pair {
                                left: left_at,
                                right: right_at,
                                candidates: vec![1],
                            })
                    })
                    .collect::<Vec<_>>();
                PairBatch::from_pairs(pairs)
            })
            .collect())
    }

    fn local_weight_is_split_invariant(&self) -> bool {
        true
    }

    fn local_weights_are_unit(&self) -> bool {
        false
    }

    fn permit_key(&self, parts: &[Span]) -> Vec<Option<Span>> {
        parts
            .iter()
            .map(|part| (!self.lexical_permits || part.1 - part.0 == 1).then_some(*part))
            .collect()
    }

    fn decision(&mut self, _rule: u16, _tags: &[u8]) -> Result<NormalDecision, Self::Error> {
        Ok(NormalDecision::Admit(0))
    }

    fn admit_rows(
        &mut self,
        _rule: u16,
        _tags: &[u8],
        input_pairs: &[[u32; 2]],
        _output: u32,
    ) -> Result<Vec<bool>, Self::Error> {
        Ok(vec![true; input_pairs.len()])
    }

    fn keeps(&mut self, _span: Span, _category: u32) -> Result<bool, Self::Error> {
        Ok(true)
    }

    fn permits(
        &mut self,
        _span: Span,
        _parts: &[Span],
        _rule: u16,
        _inputs: &[u32],
    ) -> Result<bool, Self::Error> {
        Ok(true)
    }

    fn steps(
        &mut self,
        requests: &[StepBatchRequest],
        _workers: Option<&rayon::ThreadPool>,
    ) -> Result<Vec<StepBatchEntry>, Self::Error> {
        self.requests += requests.len();
        Ok(requests
            .iter()
            .enumerate()
            .flat_map(|(request_at, request)| {
                request
                    .tag_rows
                    .iter()
                    .enumerate()
                    .filter_map(move |(row, tags)| {
                        (request.inputs == [0, 0]
                            && (tags.as_slice() == [0, 0] || tags.as_slice() == [2, 0])
                            && request.candidates.contains(&1))
                        .then_some(StepBatchEntry {
                            request: request_at,
                            row,
                            step: Step {
                                rule: 1,
                                output: 1,
                                tag: 1,
                            },
                        })
                    })
            })
            .collect())
    }

    fn closure(
        &mut self,
        arrival: &[ItemId],
        _span: Span,
        depth: usize,
        _workers: Option<&rayon::ThreadPool>,
    ) -> Result<UnaryClosure, Self::Error> {
        unary_closure(arrival, depth, |items| {
            Ok::<_, Infallible>(
                items
                    .iter()
                    .map(|item| {
                        if item.category == 1 {
                            vec![Step {
                                rule: 0,
                                output: 2,
                                tag: 2,
                            }]
                        } else {
                            Vec::new()
                        }
                    })
                    .collect(),
            )
        })
    }

    fn local_weight(
        &mut self,
        rule: u16,
        _inputs: &[u32],
        _output: u32,
        _span: Span,
        _split: Option<usize>,
    ) -> Result<Count, Self::Error> {
        Ok(Count::Small(if rule == 0 { 7 } else { 5 }))
    }

    fn finish_width(&mut self) -> Result<(), Self::Error> {
        self.widths += 1;
        Ok(())
    }
}

impl ForestResolver<Count, u16> for Resolver {
    fn label_order(&mut self, rule: u16) -> Result<(u16, usize), Self::Error> {
        Ok((rule, usize::from(rule)))
    }

    fn root_keep(
        &mut self,
        _span: Span,
        _items: &[(ItemId, Count)],
        _closure: &UnaryClosure,
    ) -> Result<Option<Vec<ItemId>>, Self::Error> {
        Ok(None)
    }

    fn prune_keep(
        &mut self,
        _span: Span,
        _items: &[(ItemId, Count)],
    ) -> Result<Option<Vec<ItemId>>, Self::Error> {
        Ok(None)
    }
}

impl InsideResolver<f64> for Resolver {
    type Error = Infallible;

    fn rules_of_arity(&mut self, arity: usize) -> Result<Vec<u16>, Self::Error> {
        <Self as InsideResolver<Count>>::rules_of_arity(self, arity)
    }

    fn pair_batches(
        &mut self,
        batches: &[PairBatchRequest],
        workers: Option<&rayon::ThreadPool>,
        factor_units: bool,
    ) -> Result<Vec<PairBatch>, Self::Error> {
        <Self as InsideResolver<Count>>::pair_batches(self, batches, workers, factor_units)
    }

    fn local_weight_is_split_invariant(&self) -> bool {
        <Self as InsideResolver<Count>>::local_weight_is_split_invariant(self)
    }

    fn local_weights_are_unit(&self) -> bool {
        <Self as InsideResolver<Count>>::local_weights_are_unit(self)
    }

    fn permit_key(&self, parts: &[Span]) -> Vec<Option<Span>> {
        <Self as InsideResolver<Count>>::permit_key(self, parts)
    }

    fn decision(&mut self, rule: u16, tags: &[u8]) -> Result<NormalDecision, Self::Error> {
        <Self as InsideResolver<Count>>::decision(self, rule, tags)
    }

    fn admit_rows(
        &mut self,
        rule: u16,
        tags: &[u8],
        input_pairs: &[[u32; 2]],
        output: u32,
    ) -> Result<Vec<bool>, Self::Error> {
        <Self as InsideResolver<Count>>::admit_rows(self, rule, tags, input_pairs, output)
    }

    fn keeps(&mut self, span: Span, category: u32) -> Result<bool, Self::Error> {
        <Self as InsideResolver<Count>>::keeps(self, span, category)
    }

    fn permits(
        &mut self,
        span: Span,
        parts: &[Span],
        rule: u16,
        inputs: &[u32],
    ) -> Result<bool, Self::Error> {
        <Self as InsideResolver<Count>>::permits(self, span, parts, rule, inputs)
    }

    fn steps(
        &mut self,
        requests: &[StepBatchRequest],
        workers: Option<&rayon::ThreadPool>,
    ) -> Result<Vec<StepBatchEntry>, Self::Error> {
        <Self as InsideResolver<Count>>::steps(self, requests, workers)
    }

    fn closure(
        &mut self,
        arrival: &[ItemId],
        span: Span,
        depth: usize,
        workers: Option<&rayon::ThreadPool>,
    ) -> Result<UnaryClosure, Self::Error> {
        <Self as InsideResolver<Count>>::closure(self, arrival, span, depth, workers)
    }

    fn local_weight(
        &mut self,
        rule: u16,
        _inputs: &[u32],
        _output: u32,
        _span: Span,
        _split: Option<usize>,
    ) -> Result<f64, Self::Error> {
        Ok(if rule == 0 { 7.0 } else { 5.0 })
    }

    fn finish_width(&mut self) -> Result<(), Self::Error> {
        self.widths += 1;
        Ok(())
    }
}

fn program() -> ChartProgram {
    ChartProgram::new(
        vec![
            RuleSpec {
                arity: 1,
                order: 0,
                complete: true,
            },
            RuleSpec {
                arity: 2,
                order: 0,
                complete: true,
            },
            RuleSpec {
                arity: 2,
                order: 1,
                complete: true,
            },
        ],
        3,
        vec![1],
        vec![2],
        HashMap::new(),
        CategoryBounds::default(),
    )
}

#[test]
fn miss_batch_owns_its_request_and_admitted_steps() {
    let mut program = program();
    for category in 0..10 {
        assert_eq!(
            program.register_category(
                category,
                CategoryMeta::empty(),
                CategoryDimensions { arity: 0, size: 1 },
            ),
            Ok(true)
        );
    }
    let mut batch = MissBatch::new(&program, vec![3, 5], vec![1, 0], vec![2, 1], (4, 7));

    assert_eq!(batch.inputs(), [3, 5]);
    assert_eq!(batch.tags(), [1, 0]);
    assert_eq!(batch.candidates(), [2, 1]);
    assert_eq!(batch.span(), (4, 7));
    assert_eq!(batch.push(1, 8, 0), Ok(()));
    assert_eq!(batch.push(2, 9, 1), Ok(()));
    assert_eq!(
        batch.results(),
        [
            Step {
                rule: 1,
                output: 8,
                tag: 0,
            },
            Step {
                rule: 2,
                output: 9,
                tag: 1,
            },
        ],
        "the boundary retains accepted steps in resolver insertion order"
    );
}

#[test]
fn miss_batch_push_refuses_each_invalid_identifier_without_mutation() {
    let mut program = program();
    assert_eq!(
        program.register_category(
            0,
            CategoryMeta::empty(),
            CategoryDimensions { arity: 0, size: 1 },
        ),
        Ok(true)
    );
    let mut batch = MissBatch::new(&program, vec![0], vec![0], vec![1], (0, 1));

    assert_eq!(
        batch.push(3, 0, 0),
        Err(ChartError::UnknownRule { rule: 3, rules: 3 })
    );
    assert_eq!(
        batch.push(0, 0, 0),
        Err(ChartError::RuleNotCandidate { rule: 0 })
    );
    assert_eq!(
        batch.push(1, 0, 3),
        Err(ChartError::UnknownTag { tag: 3, tags: 3 })
    );
    assert!(
        batch.results().is_empty(),
        "a refused step never mutates the resolver batch"
    );
}

fn log_inside_snapshot(threads: usize) -> Vec<(Span, Vec<(ItemId, f64)>)> {
    let semiring = LogInside;
    let mut table = InsideTable::new(4);

    for position in 0..4 {
        assert!(
            table
                .add(
                    (position, position + 1),
                    ItemId {
                        category: 0,
                        tag: 0,
                    },
                    f64::from(u32::try_from(position).unwrap_or_default()),
                    &semiring,
                )
                .is_ok(),
            "each lexical position is in the four-token chart"
        );
    }
    let mut resolver = Resolver::default();
    let execution =
        CellExecution::Parallel(NonZeroUsize::new(threads).unwrap_or(NonZeroUsize::MIN));
    assert!(
        table
            .sweep(&[2], 3, execution, &semiring, &mut resolver)
            .is_ok(),
        "the log-inside carrier and resolver are infallible"
    );

    (1..=4)
        .flat_map(|width| (0..=4 - width).map(move |start| (start, start + width)))
        .map(|span| (span, table.tagged_at(span).unwrap_or_default()))
        .collect()
}

#[test]
fn parallel_inside_cells_are_bitwise_deterministic() {
    let serial = log_inside_snapshot(1);

    assert_eq!(log_inside_snapshot(2), serial);
    assert_eq!(log_inside_snapshot(8), serial);
}

type ForestSnapshot = Vec<(Span, Vec<(ForestKey, Count, Vec<ForestEdge<u16, Count>>)>)>;

fn forest_snapshot(threads: usize) -> ForestSnapshot {
    let semiring = Counting;
    let mut table = ForestTable::new(4);

    for position in 0..4 {
        let key = ForestKey {
            span: (position, position + 1),
            item: ItemId {
                category: 0,
                tag: 0,
            },
        };
        assert!(
            table
                .add(
                    key,
                    ForestEdge {
                        label: 10,
                        children: Vec::new(),
                        output: 0,
                        weight: Count::Small(u64::try_from(position + 2).unwrap_or_default()),
                        split: None,
                        order: 0,
                    },
                    &semiring,
                )
                .is_ok(),
            "each lexical position is in the four-token forest"
        );
    }
    let mut resolver = Resolver::default();
    let execution =
        CellExecution::Parallel(NonZeroUsize::new(threads).unwrap_or(NonZeroUsize::MIN));
    assert!(
        table
            .sweep(&[2], 3, execution, &semiring, &mut resolver)
            .is_ok(),
        "the counting carrier and resolver are infallible"
    );

    (1..=4)
        .flat_map(|width| (0..=4 - width).map(move |start| (start, start + width)))
        .map(|span| {
            let items = table.items_at(span).unwrap_or_default();
            let snapshot = items
                .into_iter()
                .map(|key| {
                    (
                        key,
                        table.inside(key, &semiring),
                        table.edges(key).unwrap_or_default(),
                    )
                })
                .collect();
            (span, snapshot)
        })
        .collect()
}

#[test]
fn parallel_forest_cells_are_bitwise_deterministic() {
    let serial = forest_snapshot(1);

    assert_eq!(forest_snapshot(2), serial);
    assert_eq!(forest_snapshot(8), serial);
}

#[test]
fn indexed_pairs_merge_head_and_join_rules_in_declaration_order() {
    let mut chart = program();
    assert!(
        chart
            .register_category(
                0,
                CategoryMeta {
                    unary: vec![0],
                    viable: HashSet::from([(1, 0)]),
                    joins: HashMap::from([((2, 0), vec![7])]),
                    factors: HashMap::new(),
                },
                CategoryDimensions { arity: 0, size: 1 },
            )
            .is_ok(),
        "the first category is dense"
    );
    assert!(
        chart
            .register_category(
                1,
                CategoryMeta {
                    unary: vec![],
                    viable: HashSet::from([(1, 1)]),
                    joins: HashMap::from([((2, 1), vec![7])]),
                    factors: HashMap::new(),
                },
                CategoryDimensions { arity: 0, size: 1 },
            )
            .is_ok(),
        "the second category is dense"
    );

    let pairs = chart.pairs(&[0, 0], &[1, 1]).unwrap_or_default();
    let repeated = chart.pairs(&[0, 0], &[1, 1]).unwrap_or_default();
    let without_joined = chart
        .pair_batch_where_excluding(
            &[0, 0],
            &[1, 1],
            false,
            &HashSet::from([2]),
            |_, _, _, _| true,
        )
        .unwrap_or_default();

    assert_eq!(pairs.len(), 4, "each Cartesian occurrence is admitted");
    assert!(
        pairs.iter().all(|pair| pair.candidates == [1, 2]),
        "rule order is preserved"
    );
    assert_eq!(repeated, pairs, "memoization preserves exact pair ordering");
    assert!(
        without_joined
            .pairs
            .iter()
            .all(|pair| pair.candidates == [1]),
        "pre-factored joined rules leave only exact residual candidates"
    );
}

#[test]
fn complete_outputs_are_memoized_independently_of_normal_form_tags() {
    let mut chart = program();

    for category in 0..3 {
        assert!(
            chart
                .register_category(
                    category,
                    CategoryMeta {
                        unary: vec![0],
                        viable: HashSet::new(),
                        joins: HashMap::new(),
                        factors: HashMap::new(),
                    },
                    CategoryDimensions { arity: 0, size: 1 },
                )
                .is_ok(),
            "the category sequence is dense"
        );
    }

    assert!(
        chart.remember_outputs(1, vec![0, 1], vec![2]).is_ok(),
        "the rule and categories are registered"
    );

    assert_eq!(
        chart.cached_outputs(1, &[0, 1]),
        Some(&[2][..]),
        "rule outputs do not depend on child tags"
    );
    assert_eq!(
        chart.cached(&[0, 1], &[0, 1], &[1]),
        None,
        "admitted steps remain tag-aware"
    );
}

#[test]
fn memoized_steps_are_scoped_to_the_exact_candidate_rules() {
    let mut chart = program();
    let kept = Step {
        rule: 2,
        output: 7,
        tag: 1,
    };

    chart.remember(vec![3, 5], vec![1, 0], vec![1, 2], vec![kept]);
    chart.remember(vec![3, 5], vec![1, 0], vec![1], Vec::new());

    assert_eq!(
        chart.cached(&[3, 5], &[1, 0], &[1, 2]),
        Some(&[kept][..]),
        "a narrower request cannot replace the complete candidate row"
    );
    assert_eq!(
        chart.cached(&[3, 5], &[1, 0], &[1]),
        Some(&[][..]),
        "the narrower request keeps its own result"
    );
    assert_eq!(
        chart.cached(&[3, 5], &[1, 0], &[2]),
        None,
        "an unseen candidate set has no cached result"
    );
}

#[test]
fn normal_form_decisions_are_exactly_tag_aware() {
    let decisions = HashMap::from([
        ((0, vec![0]), NormalDecision::Admit(1)),
        ((0, vec![1]), NormalDecision::Refuse),
        ((0, vec![2]), NormalDecision::Premises(0)),
    ]);
    let chart = ChartProgram::new(
        vec![RuleSpec {
            arity: 1,
            order: 0,
            complete: true,
        }],
        3,
        Vec::new(),
        Vec::new(),
        decisions,
        CategoryBounds::default(),
    );

    assert_eq!(chart.decision(0, &[0]), Ok(NormalDecision::Admit(1)));
    assert_eq!(chart.decision(0, &[1]), Ok(NormalDecision::Refuse));
    assert_eq!(chart.decision(0, &[2]), Ok(NormalDecision::Premises(0)));
    assert!(matches!(
        chart.decision(0, &[3]),
        Err(ChartError::UnknownTag { tag: 3, tags: 3 })
    ));
}

#[test]
fn an_arbitrary_arity_decision_remains_exact_when_its_dense_size_overflows() {
    let arity = usize::BITS.try_into().unwrap_or_default();
    let children = vec![0; arity];
    let decisions = HashMap::from([((0, children.clone()), NormalDecision::Admit(1))]);
    let chart = ChartProgram::new(
        vec![RuleSpec {
            arity,
            order: 0,
            complete: true,
        }],
        2,
        Vec::new(),
        Vec::new(),
        decisions,
        CategoryBounds::default(),
    );

    assert_eq!(
        chart.decision(0, &children),
        Ok(NormalDecision::Admit(1)),
        "an overflowing dense shape uses the same exact per-rule decision relation"
    );
}

#[test]
fn admissible_tag_rows_intersect_present_axes_in_cartesian_order() {
    let decisions = HashMap::from([
        ((0, vec![2, 1]), NormalDecision::Refuse),
        ((0, vec![2, 2]), NormalDecision::Admit(0)),
        ((0, vec![0, 1]), NormalDecision::Premises(1)),
        ((0, vec![0, 2]), NormalDecision::Refuse),
    ]);
    let chart = ChartProgram::new(
        vec![RuleSpec {
            arity: 2,
            order: 0,
            complete: true,
        }],
        3,
        Vec::new(),
        Vec::new(),
        decisions,
        CategoryBounds::default(),
    );

    assert_eq!(
        chart.admissible_tag_rows(&[0], &[vec![2, 1], vec![2, 2], vec![0, 1], vec![0, 2]],),
        Ok(vec![(1, vec![2, 2]), (2, vec![0, 1])]),
        "only present non-refused rows survive in their original Cartesian order"
    );
}

#[test]
fn batch_plans_keep_cartesian_order_and_pair_each_rule_with_its_output_tag() {
    let decisions = HashMap::from([
        ((0, vec![0, 0]), NormalDecision::Admit(1)),
        ((0, vec![0, 1]), NormalDecision::Refuse),
        ((0, vec![1, 0]), NormalDecision::Premises(0)),
        ((0, vec![1, 1]), NormalDecision::Admit(0)),
    ]);
    let mut chart = ChartProgram::new(
        vec![RuleSpec {
            arity: 2,
            order: 0,
            complete: true,
        }],
        2,
        vec![0],
        Vec::new(),
        decisions,
        CategoryBounds::default(),
    );
    for category in 0..2 {
        assert_eq!(
            chart.register_category(
                category,
                CategoryMeta::empty(),
                CategoryDimensions { arity: 0, size: 1 },
            ),
            Ok(true)
        );
    }
    let requests = [StepBatchRequest {
        inputs: vec![0, 1],
        tag_rows: vec![vec![1, 1], vec![0, 1], vec![0, 0], vec![1, 0]],
        span: (0, 2),
        parts: vec![(0, 1), (1, 2)],
        candidates: vec![0],
    }];

    let plans = chart.batch_step_plans(&requests, false, None);
    assert!(
        plans.is_ok(),
        "the request names registered categories and complete decisions"
    );
    let plans = plans.unwrap_or_default();
    let rows = &plans[0];

    assert_eq!(
        rows.iter().map(|row| row.row).collect::<Vec<_>>(),
        vec![0, 2, 3],
        "a refused row is removed without changing the surviving Cartesian order"
    );
    assert_eq!(
        rows[0].plan,
        StepPlan::Draw {
            native: vec![(0, 0)],
            residual: Vec::new(),
        }
    );
    assert_eq!(
        rows[1].plan,
        StepPlan::Draw {
            native: vec![(0, 1)],
            residual: Vec::new(),
        }
    );
    assert_eq!(
        rows[2].plan,
        StepPlan::Draw {
            native: Vec::new(),
            residual: vec![0],
        }
    );
}

#[test]
fn unary_closure_orders_convergent_steps_and_drops_cycles() {
    let arrival = [ItemId {
        category: 0,
        tag: 0,
    }];
    let closure = unary_closure(&arrival, 4, |items| {
        Ok::<_, ChartError>(
            items
                .iter()
                .map(|item| match item.category {
                    0 => vec![
                        Step {
                            rule: 0,
                            output: 1,
                            tag: 1,
                        },
                        Step {
                            rule: 1,
                            output: 1,
                            tag: 1,
                        },
                        Step {
                            rule: 0,
                            output: 2,
                            tag: 1,
                        },
                    ],
                    1 => vec![Step {
                        rule: 0,
                        output: 2,
                        tag: 1,
                    }],
                    2 => vec![Step {
                        rule: 0,
                        output: 0,
                        tag: 1,
                    }],
                    _ => vec![],
                })
                .collect(),
        )
    })
    .unwrap_or(UnaryClosure {
        edges: Vec::new(),
        concluded: Vec::new(),
        truncated: false,
    });

    assert_eq!(
        closure.edges.len(),
        4,
        "parallel rules and the other acyclic steps are retained"
    );
    assert_eq!(
        closure.concluded,
        vec![
            ItemId {
                category: 1,
                tag: 1,
            },
            ItemId {
                category: 2,
                tag: 1,
            },
        ],
        "premises precede convergent conclusions"
    );
    assert!(
        closure.truncated,
        "the return to category zero is a dropped cycle"
    );
}

#[test]
fn unary_cycle_selection_is_independent_of_cell_insertion_order() {
    let close = |arrival: &[ItemId]| {
        unary_closure(arrival, 1, |items| {
            Ok::<_, ChartError>(
                items
                    .iter()
                    .map(|item| {
                        vec![Step {
                            rule: 0,
                            output: 1 - item.category,
                            tag: 0,
                        }]
                    })
                    .collect(),
            )
        })
        .unwrap_or(UnaryClosure {
            edges: Vec::new(),
            concluded: Vec::new(),
            truncated: false,
        })
    };
    let low = ItemId {
        category: 0,
        tag: 0,
    };
    let high = ItemId {
        category: 1,
        tag: 0,
    };

    assert_eq!(
        close(&[low, high]),
        close(&[high, low]),
        "cycle truncation follows canonical dense identifiers, not storage order"
    );
}

#[test]
fn category_registration_rejects_a_gap() {
    let mut chart = program();
    let result = chart.register_category(
        1,
        CategoryMeta {
            unary: vec![],
            viable: HashSet::new(),
            joins: HashMap::new(),
            factors: HashMap::new(),
        },
        CategoryDimensions { arity: 0, size: 1 },
    );
    assert!(result.is_err(), "the first category id must be zero");
    let error = result
        .err()
        .unwrap_or(ChartError::UnknownCategory { category: u32::MAX });

    assert_eq!(
        error,
        ChartError::NonDenseCategory {
            expected: 0,
            supplied: 1,
        },
        "the boundary reports both ids"
    );
}

#[test]
fn category_bounds_refuse_only_categories_outside_a_declared_dimension() {
    let mut chart = ChartProgram::new(
        Vec::new(),
        1,
        Vec::new(),
        Vec::new(),
        HashMap::new(),
        CategoryBounds {
            max_arity: NonZeroUsize::new(2),
            max_size: NonZeroUsize::new(5),
        },
    );
    let meta = || CategoryMeta {
        unary: Vec::new(),
        viable: HashSet::new(),
        joins: HashMap::new(),
        factors: HashMap::new(),
    };

    assert_eq!(
        chart.register_category(0, meta(), CategoryDimensions { arity: 2, size: 5 }),
        Ok(true),
        "a category exactly on both bounds is retained"
    );
    assert_eq!(
        chart.register_category(1, meta(), CategoryDimensions { arity: 2, size: 6 }),
        Ok(false),
        "size is enforced independently"
    );
    assert_eq!(
        chart.register_category(1, meta(), CategoryDimensions { arity: 3, size: 5 }),
        Ok(false),
        "functor-spine arity is enforced independently"
    );
    assert_eq!(chart.refused_categories(), 2);
    assert_eq!(
        chart.register_category(1, meta(), CategoryDimensions { arity: 1, size: 3 }),
        Ok(true),
        "a refused category consumes no dense identifier"
    );
}

#[test]
fn inside_sweep_preserves_exact_tags_and_closes_unary_weights() {
    let semiring = Counting;
    let mut table = InsideTable::new(2);
    let left = table.add(
        (0, 1),
        ItemId {
            category: 0,
            tag: 0,
        },
        Count::Small(2),
        &semiring,
    );
    let alternate_left = table.add(
        (0, 1),
        ItemId {
            category: 0,
            tag: 2,
        },
        Count::Small(11),
        &semiring,
    );
    let right = table.add(
        (1, 2),
        ItemId {
            category: 0,
            tag: 0,
        },
        Count::Small(3),
        &semiring,
    );
    assert!(
        left.is_ok() && alternate_left.is_ok() && right.is_ok(),
        "both lexical spans are valid"
    );

    let mut resolver = Resolver::default();
    let swept = table.sweep(
        &[2],
        3,
        CellExecution::Parallel(NonZeroUsize::MIN),
        &semiring,
        &mut resolver,
    );
    assert!(
        swept.is_ok(),
        "the resolver and counting carrier are infallible"
    );

    assert_eq!(
        table.tagged_at((0, 2)),
        Ok(vec![
            (
                ItemId {
                    category: 1,
                    tag: 1,
                },
                Count::Small(195),
            ),
            (
                ItemId {
                    category: 2,
                    tag: 2,
                },
                Count::Small(1365),
            ),
        ]),
        "binary and unary conclusions retain their exact normal-form tags"
    );
    assert_eq!(
        resolver.widths, 2,
        "the resolver checks once after each width"
    );
}

#[test]
fn inside_sweep_shares_rule_drawing_across_interior_splits() {
    let semiring = Counting;
    let make_table = || {
        let mut table = InsideTable::new(6);
        for start in 0..6 {
            for stop in start + 1..=6 {
                assert!(
                    table
                        .add(
                            (start, stop),
                            ItemId {
                                category: 0,
                                tag: 0,
                            },
                            Count::Small(1),
                            &semiring,
                        )
                        .is_ok(),
                    "the seeded span is valid"
                );
            }
        }
        table
    };
    let mut exact = make_table();
    let mut shared = make_table();
    let mut exact_resolver = Resolver::default();
    let mut shared_resolver = Resolver {
        lexical_permits: true,
        ..Resolver::default()
    };

    assert!(
        exact
            .sweep(
                &[2],
                0,
                CellExecution::Attached,
                &semiring,
                &mut exact_resolver,
            )
            .is_ok(),
        "the exact-split sweep is infallible"
    );
    assert!(
        shared
            .sweep(
                &[2],
                0,
                CellExecution::Attached,
                &semiring,
                &mut shared_resolver,
            )
            .is_ok(),
        "the permit-equivalent sweep is infallible"
    );

    assert_eq!(
        (0..6)
            .flat_map(|start| (start + 1..=6).map(move |stop| (start, stop)))
            .map(|span| shared.tagged_at(span))
            .collect::<Vec<_>>(),
        (0..6)
            .flat_map(|start| (start + 1..=6).map(move |stop| (start, stop)))
            .map(|span| exact.tagged_at(span))
            .collect::<Vec<_>>(),
        "sharing preserves every exact item and inside weight"
    );
    assert!(
        shared_resolver.requests < exact_resolver.requests,
        "interior decompositions share one rule-drawing request"
    );
}

#[test]
fn inside_sweep_factors_child_products_across_rule_outputs() {
    let semiring = CountingOperations::default();
    let mut table = InsideTable::new(2);
    for (span, category, weight) in [((0, 1), 0, 2), ((1, 2), 0, 3), ((1, 2), 3, 7)] {
        assert!(
            table
                .add(
                    span,
                    ItemId { category, tag: 0 },
                    Count::Small(weight),
                    &semiring,
                )
                .is_ok(),
            "the lexical item has a valid span"
        );
    }

    let mut resolver = MultiOutputResolver {
        kept: None,
        permitted: true,
        mixed: false,
    };
    assert!(
        table
            .sweep(&[2], 0, CellExecution::Attached, &semiring, &mut resolver,)
            .is_ok(),
        "the exact counting sweep is infallible"
    );

    assert_eq!(
        table.categories_at((0, 2), &semiring),
        Ok(vec![(1, Count::Small(20)), (2, Count::Small(20))]),
        "factoring preserves each exact conclusion and its derivation count"
    );
    assert_eq!(
        semiring.times.load(Ordering::Relaxed),
        1,
        "one child product is shared by every exact output"
    );
}

#[test]
fn premise_products_preserve_sparse_raised_feature_rows_inside_and_forest()
-> Result<(), Box<dyn std::error::Error>> {
    let semiring = Counting;
    let mut inside = InsideTable::new(2);
    let mut forest = ForestTable::new(2);
    for (span, categories) in [((0, 1), 10..=12), ((1, 2), 20..=22)] {
        for category in categories {
            for tag in 0..=2 {
                let item = ItemId { category, tag };
                assert!(
                    inside.add(span, item, Count::Small(1), &semiring).is_ok(),
                    "the inside lexical item has a valid span"
                );
                assert!(
                    forest
                        .add(
                            ForestKey { span, item },
                            ForestEdge {
                                label: 9,
                                children: Vec::new(),
                                output: category,
                                weight: Count::Small(1),
                                split: None,
                                order: 0,
                            },
                            &semiring,
                        )
                        .is_ok(),
                    "the forest lexical item has a valid span"
                );
            }
        }
    }

    let resolver = || MultiOutputResolver {
        kept: Some(1),
        permitted: true,
        mixed: true,
    };
    assert!(
        inside
            .sweep(&[2], 0, CellExecution::Attached, &semiring, &mut resolver(),)
            .is_ok(),
        "the premise-sensitive inside sweep is infallible"
    );
    assert!(
        forest
            .sweep(&[2], 0, CellExecution::Attached, &semiring, &mut resolver(),)
            .is_ok(),
        "the premise-sensitive forest sweep is infallible"
    );

    let admitted = [
        (
            ItemId {
                category: 1,
                tag: 0,
            },
            Count::Small(9),
        ),
        (
            ItemId {
                category: 1,
                tag: 2,
            },
            Count::Small(5),
        ),
    ];
    assert_eq!(inside.tagged_at((0, 2)), Ok(admitted.to_vec()));
    assert_eq!(
        require_table(forest.items_at((0, 2)))?
            .into_iter()
            .map(|key| (key.item, forest.inside(key, &semiring)))
            .collect::<Vec<_>>(),
        admitted.to_vec()
    );

    let premise_key = ForestKey {
        span: (0, 2),
        item: ItemId {
            category: 1,
            tag: 2,
        },
    };
    let products = require_table(forest.products(premise_key))?;
    assert_eq!(
        products
            .iter()
            .map(|product| product.axes.iter().map(Vec::len).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
        vec![vec![2, 2], vec![1, 1]],
        "equal admitted-right sets form disjoint rectangles"
    );
    assert_eq!(
        require_table(forest.edges(premise_key))?.len(),
        5,
        "rectangle expansion preserves exact edge multiplicity"
    );
    Ok(())
}

#[test]
fn inside_sweep_filters_factored_outputs_before_insertion() {
    let semiring = Counting;
    let mut table = InsideTable::new(2);
    for span in [(0, 1), (1, 2)] {
        assert!(
            table
                .add(
                    span,
                    ItemId {
                        category: 0,
                        tag: 0,
                    },
                    Count::Small(1),
                    &semiring,
                )
                .is_ok(),
            "the lexical item has a valid span"
        );
    }

    let mut resolver = MultiOutputResolver {
        kept: Some(1),
        permitted: true,
        mixed: false,
    };
    assert!(
        table
            .sweep(&[2], 0, CellExecution::Attached, &semiring, &mut resolver)
            .is_ok(),
        "the reachability-filtered sweep is infallible"
    );

    assert_eq!(
        table.categories_at((0, 2), &semiring),
        Ok(vec![(1, Count::Small(1))]),
        "an unreachable factored conclusion never enters the chart cell"
    );
}

#[test]
fn inside_sweep_filters_factored_rules_by_joint_premise_demand() {
    let semiring = Counting;
    let mut table = InsideTable::new(2);
    for span in [(0, 1), (1, 2)] {
        assert!(
            table
                .add(
                    span,
                    ItemId {
                        category: 0,
                        tag: 0,
                    },
                    Count::Small(1),
                    &semiring,
                )
                .is_ok(),
            "the lexical item has a valid span"
        );
    }

    let mut resolver = MultiOutputResolver {
        kept: None,
        permitted: false,
        mixed: false,
    };
    assert!(
        table
            .sweep(&[2], 0, CellExecution::Attached, &semiring, &mut resolver)
            .is_ok(),
        "the joint-demand-filtered sweep is infallible"
    );

    assert_eq!(
        table.categories_at((0, 2), &semiring),
        Ok(Vec::new()),
        "a factored rule outside the joint premise demand never enters the chart"
    );
}

#[test]
fn inside_table_sums_tags_in_category_arrival_order() {
    let semiring = Counting;
    let mut table = InsideTable::new(1);

    for (category, tag, weight) in [(4, 1, 2), (7, 0, 5), (4, 2, 3)] {
        assert!(
            table
                .add(
                    (0, 1),
                    ItemId { category, tag },
                    Count::Small(weight),
                    &semiring,
                )
                .is_ok(),
            "the lexical span is valid"
        );
    }

    assert_eq!(
        table.categories_at((0, 1), &semiring),
        Ok(vec![(4, Count::Small(5)), (7, Count::Small(5))]),
        "category order follows first arrival and each total sums exact tags"
    );
    assert_eq!(table.num_categories(), 2);
}

#[test]
fn inside_table_reports_invalid_spans_without_creating_cells() {
    let semiring = Counting;
    let mut table = InsideTable::new(2);
    let result = table.add(
        (1, 3),
        ItemId {
            category: 0,
            tag: 0,
        },
        Count::Small(1),
        &semiring,
    );

    assert!(
        matches!(result, Err(SweepError::Table(_))),
        "invalid spans are table-boundary errors"
    );
    assert_eq!(table.num_categories(), 0);
}

#[test]
fn forest_table_preserves_child_order_exact_tags_and_inside_weights() {
    let semiring = Counting;
    let mut table = ForestTable::new(2);
    let left = ForestKey {
        span: (0, 1),
        item: ItemId {
            category: 0,
            tag: 0,
        },
    };
    let right = ForestKey {
        span: (1, 2),
        item: ItemId {
            category: 0,
            tag: 0,
        },
    };

    for (key, label, weight) in [(left, 10, 2), (right, 11, 3)] {
        assert!(
            table
                .add(
                    key,
                    ForestEdge {
                        label,
                        children: Vec::new(),
                        output: 0,
                        weight: Count::Small(weight),
                        split: None,
                        order: 0,
                    },
                    &semiring,
                )
                .is_ok(),
            "the lexical edge has a valid span"
        );
    }

    let mut resolver = Resolver::default();
    assert!(
        table
            .sweep(
                &[2],
                3,
                CellExecution::Parallel(NonZeroUsize::MIN),
                &semiring,
                &mut resolver,
            )
            .is_ok(),
        "the resolver and counting carrier are infallible"
    );
    let binary = ForestKey {
        span: (0, 2),
        item: ItemId {
            category: 1,
            tag: 1,
        },
    };
    let unary = ForestKey {
        span: (0, 2),
        item: ItemId {
            category: 2,
            tag: 2,
        },
    };

    assert_eq!(table.inside(binary, &semiring), Count::Small(30));
    assert_eq!(table.inside(unary, &semiring), Count::Small(210));
    assert_eq!(
        table.edges(binary),
        Ok(vec![ForestEdge {
            label: 1,
            children: vec![left, right],
            output: 1,
            weight: Count::Small(5),
            split: Some(1),
            order: 1,
        }]),
        "one CSR child row preserves rule order"
    );
    assert_eq!(table.items_at((0, 2)), Ok(vec![binary, unary]));
    assert_eq!(table.num_items(), 4);
    assert_eq!(table.num_edges(), 4);
}

const fn forest_key(span: Span, category: u32) -> ForestKey {
    ForestKey {
        span,
        item: ItemId { category, tag: 0 },
    }
}

fn require_table<T, E: std::fmt::Debug>(result: Result<T, E>) -> Result<T, String> {
    result.map_err(|error| format!("the product fixture is invalid: {error:?}"))
}

type ForestValue<W> = (W, Vec<ForestEdge<u16, W>>);

fn product_and_flat<W, S>(
    semiring: &S,
    weights: &[W; 5],
) -> Result<(ForestValue<W>, ForestValue<W>), String>
where
    W: Clone,
    S: Semiring<Weight = W>,
    S::Error: std::fmt::Debug,
{
    let left = [forest_key((0, 1), 0), forest_key((0, 1), 1)];
    let right = [forest_key((1, 2), 2), forest_key((1, 2), 3)];
    let parent = forest_key((0, 2), 4);
    let mut product = ForestTable::new(2);
    let mut flat = ForestTable::new(2);

    for (key, weight) in left
        .into_iter()
        .chain(right)
        .zip(weights[..4].iter().cloned())
    {
        for table in [&mut product, &mut flat] {
            require_table(table.add(
                key,
                ForestEdge {
                    label: 0,
                    children: Vec::new(),
                    output: key.item.category,
                    weight: weight.clone(),
                    split: None,
                    order: 0,
                },
                semiring,
            ))?;
        }
    }
    let local = weights[4].clone();
    require_table(product.add_product(
        parent,
        ProductAlternative {
            label: 7,
            axes: vec![left.to_vec(), right.to_vec()],
            output: parent.item.category,
            weight: local.clone(),
            split: Some(1),
            order: 2,
        },
        semiring,
    ))?;
    for first in left {
        for second in right {
            require_table(flat.add(
                parent,
                ForestEdge {
                    label: 7,
                    children: vec![first, second],
                    output: parent.item.category,
                    weight: local.clone(),
                    split: Some(1),
                    order: 2,
                },
                semiring,
            ))?;
        }
    }
    Ok((
        (
            product.inside(parent, semiring),
            require_table(product.edges(parent))?,
        ),
        (
            flat.inside(parent, semiring),
            require_table(flat.edges(parent))?,
        ),
    ))
}

#[test]
fn product_alternatives_equal_flat_edges_under_every_native_semiring()
-> Result<(), Box<dyn std::error::Error>> {
    let (product, flat) = product_and_flat(
        &Counting,
        &[
            Count::Small(2),
            Count::Small(3),
            Count::Small(5),
            Count::Small(7),
            Count::Small(11),
        ],
    )?;
    assert_eq!(product, flat);

    let (product, flat) = product_and_flat(&Boolean, &[true; 5])?;
    assert_eq!(product, flat);

    let (product, flat) = product_and_flat(&Viterbi, &[1.0, 2.0, 3.0, 4.0, 5.0])?;
    assert_eq!(product, flat);

    let logs = [0.2_f64, 0.3, 0.5, 0.7, 0.9].map(f64::ln);
    let (product, flat) = product_and_flat(&LogInside, &logs)?;
    assert!((product.0 - flat.0).abs() <= 1e-12);
    assert_eq!(product.1, flat.1);

    let width = NonZeroUsize::new(3).unwrap_or(NonZeroUsize::MIN);
    let ranked = [1.0, 2.0, 3.0, 4.0, 5.0].map(|weight| vec![weight].into_boxed_slice());
    let (product, flat) = product_and_flat(&KBest::new(width), &ranked)?;
    assert_eq!(product, flat);

    let expected = [
        (2.0, vec![1.0].into_boxed_slice()),
        (3.0, vec![2.0].into_boxed_slice()),
        (5.0, vec![3.0].into_boxed_slice()),
        (7.0, vec![4.0].into_boxed_slice()),
        (11.0, vec![5.0].into_boxed_slice()),
    ];
    let (product, flat) = product_and_flat(&Expectation::new(NonZeroUsize::MIN), &expected)?;
    assert_eq!(product, flat);
    Ok(())
}

#[test]
fn product_alternatives_preserve_order_and_edge_occurrence_multiplicity()
-> Result<(), Box<dyn std::error::Error>> {
    let semiring = Counting;
    let left = [forest_key((0, 1), 0), forest_key((0, 1), 1)];
    let right = forest_key((1, 2), 2);
    let parent = forest_key((0, 2), 3);
    let mut table = ForestTable::new(2);
    for (key, weight) in [(left[0], 2), (left[1], 3), (right, 5)] {
        require_table(table.add(
            key,
            ForestEdge {
                label: 0,
                children: Vec::new(),
                output: key.item.category,
                weight: Count::Small(weight),
                split: None,
                order: 0,
            },
            &semiring,
        ))?;
    }
    let alternative = ProductAlternative {
        label: 7,
        axes: vec![vec![left[0], left[0], left[1]], vec![right]],
        output: parent.item.category,
        weight: Count::Small(1),
        split: Some(1),
        order: 0,
    };
    require_table(table.add_product(parent, alternative.clone(), &semiring))?;
    require_table(table.add_product(parent, alternative, &semiring))?;

    let edges = require_table(table.edges(parent))?;
    assert_eq!(edges.len(), 4);
    assert_eq!(edges[0].children, vec![left[0], right]);
    assert_eq!(edges[1].children, vec![left[1], right]);
    assert_eq!(edges[2], edges[0]);
    assert_eq!(edges[3], edges[1]);
    assert_eq!(table.inside(parent, &semiring), Count::Small(50));
    assert_eq!(table.num_edges(), 7);
    Ok(())
}

#[test]
fn restricting_product_axes_matches_restricting_flat_edges()
-> Result<(), Box<dyn std::error::Error>> {
    let semiring = Counting;
    let first = forest_key((0, 1), 0);
    let second = forest_key((0, 1), 1);
    let parent = forest_key((0, 1), 2);
    let mut product = ForestTable::new(1);
    let mut flat = ForestTable::new(1);
    for (key, weight) in [(first, 2), (second, 3)] {
        for table in [&mut product, &mut flat] {
            require_table(table.add(
                key,
                ForestEdge {
                    label: 0,
                    children: Vec::new(),
                    output: key.item.category,
                    weight: Count::Small(weight),
                    split: None,
                    order: 0,
                },
                &semiring,
            ))?;
        }
    }
    require_table(product.add_product(
        parent,
        ProductAlternative {
            label: 1,
            axes: vec![vec![first, second]],
            output: parent.item.category,
            weight: Count::Small(5),
            split: None,
            order: 0,
        },
        &semiring,
    ))?;
    for child in [first, second] {
        require_table(flat.add(
            parent,
            ForestEdge {
                label: 1,
                children: vec![child],
                output: parent.item.category,
                weight: Count::Small(5),
                split: None,
                order: 0,
            },
            &semiring,
        ))?;
    }

    require_table(product.restrict((0, 1), &[first, parent], &semiring, true))?;
    require_table(flat.restrict((0, 1), &[first, parent], &semiring, true))?;

    assert_eq!(product.items_at((0, 1)), flat.items_at((0, 1)));
    assert_eq!(
        product.inside(parent, &semiring),
        flat.inside(parent, &semiring)
    );
    assert_eq!(product.edges(parent), flat.edges(parent));
    assert_eq!(product.num_edges(), flat.num_edges());
    Ok(())
}
