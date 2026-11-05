from functools import cache
from itertools import pairwise
from math import exp, inf, log
from typing import TYPE_CHECKING, Final, cast, override

import pytest
import torch
from tests.support.rules import ccg_signature, declared_rules

from lofi._native import SpanDependent, SpanFree, Term
from lofi.deduce import (
    Edge,
    ItemKey,
    PackedForest,
    ParsePolicy,
    Reachability,
    Retain,
    RootGoal,
    SpanIndex,
    parse,
)
from lofi.deduce import LogInside as SymbolicLogInside
from lofi.judge import (
    CompositionRole,
    EisnerNormalForm,
    HockenmaierBiskNormalForm,
    NoNormalForm,
    NormalFormRole,
    RuleSet,
)
from lofi.lexicon import Entry
from lofi.nn.parsers import (
    LogSumExpReduction,
    MaximumReduction,
    NormalFormTransitions,
    ScoreChannels,
    TaggedDecoder,
    TaggedInsideKernel,
    TaggedRuleScores,
    TagTransition,
    TensorChart,
    span_marginals,
)
from lofi.nn.parsers import _tagged as tagged_module
from lofi.spec import (
    ApplicationSpec,
    ConstantPatternSpec,
    PatternRuleSpec,
    RuleSetSpec,
    TermPatternSpec,
)
from lofi.terms import TargetSet

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from lofi._types import JsonValue
    from lofi.deduce import Cell, Derivation
    from lofi.judge import (
        Direction,
        NormalForm,
        PatternRule,
        Rule,
        RuleProbe,
        RuleRestriction,
        SchematicSequent,
        Slot,
    )
    from lofi.lexicon import Clamp
    from lofi.nn.parsers._chart import _TensorChartAccess
    from lofi.nn.parsers._ports import LogWeightReduction
    from lofi.spec import PatternSpec
    from lofi.terms import Functors, TermMatching, TermRuleKernel

FORWARD: Final = "/"
BACKWARD: Final = "\\"

#: Absolute tolerance for log weights.
TOLERANCE = 1e-9

#: How many rounds of unary rule application one cell takes.
DEPTH = 2


class ToyRule:
    def __init__(
        self,
        name: str,
        matching: TermMatching,
        /,
        *,
        slash: str,
        order: int,
    ) -> None:
        self._name = name
        self._matching = matching
        self._slash = slash
        self._order = order
        self._functors = FUNCTORS
        compiled = pattern_kernel((None, None), None)
        self._lowering = compiled.lowering
        self._reachability_schemata = compiled.reachability_schemata

    @property
    def name(self) -> str:
        return self._name

    @property
    def arity(self) -> int:
        return 2

    @property
    def directions(self) -> frozenset[Direction]:
        return frozenset({"forward", "check"})

    @property
    def spine_order(self) -> int:
        return 0

    @property
    def residual_slots(self) -> frozenset[int]:
        return frozenset()

    @property
    def forward_is_complete(self) -> bool:
        return True

    @property
    def matching(self) -> TermMatching:
        return self._matching

    @property
    def restriction(self) -> RuleRestriction | None:
        return None

    @property
    def probe(self) -> RuleProbe | None:
        return None

    @property
    def lowering(self) -> TermRuleKernel:
        return self._lowering

    @property
    def normal_form_role(self) -> NormalFormRole:
        direction = "forward" if self._slash == FORWARD else "backward"
        principal = 0 if direction == "forward" else 1
        return NormalFormRole(
            composition=CompositionRole(
                direction, principal_slot=principal, order=self._order
            )
        )

    @property
    def reachability_schemata(self) -> tuple[SchematicSequent, ...]:
        return self._reachability_schemata

    @property
    def functors(self) -> Functors[Term]:
        return self._functors

    @property
    def order(self) -> int:
        return self._order

    @property
    def slash(self) -> str:
        return self._slash

    def apply(self, inputs: tuple[Term, ...], /) -> Iterator[Term]:
        if len(inputs) == 2 and inputs[0] == inputs[1]:
            yield inputs[0]

    def admits(self, inputs: tuple[Term, ...], output: Term, /) -> bool:
        return (
            len(inputs) == 2
            and all(item == inputs[0] for item in inputs)
            and inputs[0] == output
        )

    def residuate(
        self, output: Term, known: tuple[Slot[Term], ...], /
    ) -> Iterator[Term]:
        del output, known
        return
        yield

    def to_spec(self) -> Mapping[str, JsonValue]:
        return {"rule": "composition", "direction": "unrestricted"}


class DuplicateRule(ToyRule):
    @override
    def apply(self, inputs: tuple[Term, ...], /) -> Iterator[Term]:
        for output in super().apply(inputs):
            yield output
            yield output


class NaryRule(ToyRule):
    def __init__(
        self,
        name: str,
        matching: TermMatching,
        /,
        *,
        arity: int,
        slash: str,
        order: int,
    ) -> None:
        super().__init__(name, matching, slash=slash, order=order)
        self._arity = arity
        compiled = pattern_kernel((None,) * arity, None)
        self._lowering = compiled.lowering
        self._reachability_schemata = compiled.reachability_schemata

    @property
    @override
    def arity(self) -> int:
        return self._arity

    @override
    def apply(self, inputs: tuple[Term, ...], /) -> Iterator[Term]:
        if len(inputs) == self._arity and len(set(inputs)) == 1:
            yield inputs[0]

    @override
    def admits(self, inputs: tuple[Term, ...], output: Term, /) -> bool:
        return len(inputs) == self._arity and all(item == output for item in inputs)


class RaisingRule(ToyRule):
    """A unary rule lifting the argument category to the functor category."""

    @property
    @override
    def arity(self) -> int:
        return 1

    @property
    @override
    def lowering(self) -> TermRuleKernel:
        return pattern_kernel(("NP",), "T").lowering

    @property
    @override
    def reachability_schemata(self) -> tuple[SchematicSequent, ...]:
        return pattern_kernel(("NP",), "T").reachability_schemata

    @property
    @override
    def normal_form_role(self) -> NormalFormRole:
        direction = "forward" if self.slash == FORWARD else "backward"
        return NormalFormRole(raising=direction)

    @override
    def apply(self, inputs: tuple[Term, ...], /) -> Iterator[Term]:
        if inputs == (ARGUMENT,):
            yield FUNCTOR

    @override
    def admits(self, inputs: tuple[Term, ...], output: Term, /) -> bool:
        return inputs == (ARGUMENT,) and output == FUNCTOR

    @override
    def to_spec(self) -> Mapping[str, JsonValue]:
        return {"rule": "type-raising"}


CATEGORY_NAMES: Final = (
    "X",
    "Y",
    "NP",
    "T",
    "a",
    "b",
    "c",
    "d",
    "c0",
    "c1",
    "c2",
    "c3",
    "c4",
)
CATEGORY_SIGNATURE: Final = ccg_signature(*CATEGORY_NAMES, name="tagged-parser-tests")


def _pattern(name: str | None, /) -> PatternSpec:
    """Return a constant pattern, or the shared category variable for ``None``."""
    if name is None:
        return TermPatternSpec(name="category", sort="Cat")

    return ConstantPatternSpec(name=name)


@cache
def pattern_kernel(
    premises: tuple[str | None, ...], conclusion: str | None
) -> PatternRule:
    """Compile one pattern rule over the shared category signature.

    Parameters
    ----------
    premises : tuple[str | None, ...]
        Each premise's constant category, ``None`` for the shared variable.
    conclusion : str | None
        The conclusion's constant category, ``None`` for the shared variable.
    """
    (rule,) = declared_rules(
        CATEGORY_SIGNATURE,
        RuleSetSpec(
            name="tagged-parser-tests",
            rules=(
                PatternRuleSpec(
                    name="kernel",
                    premises=tuple(_pattern(name) for name in premises),
                    conclusion=_pattern(conclusion),
                ),
            ),
        ),
    ).rule_set

    return cast("PatternRule", rule)


CATEGORY_CONTEXT: Final = declared_rules(
    CATEGORY_SIGNATURE,
    RuleSetSpec(name="tagged-parser-tests", rules=(ApplicationSpec(view=FORWARD),)),
).context
CATEGORY_STORE: Final = CATEGORY_CONTEXT.store
CATEGORY_BY_NAME: Final = {
    name: CATEGORY_CONTEXT.syntax.parse(CATEGORY_STORE, name) for name in CATEGORY_NAMES
}
NAME_OF: Final = {term: name for name, term in CATEGORY_BY_NAME.items()}
FUNCTORS: Final = CATEGORY_CONTEXT.functors[FORWARD][0]
MATCHING: Final = pattern_kernel((None, None), None).matching
CATEGORIES: Final = (CATEGORY_BY_NAME["X"],)
ARGUMENT: Final = CATEGORY_BY_NAME["NP"]
FUNCTOR: Final = CATEGORY_BY_NAME["T"]
RAISED: Final = (ARGUMENT, FUNCTOR)
RULES: Final = (
    ToyRule("forward-application", MATCHING, slash=FORWARD, order=0),
    ToyRule("forward-composition", MATCHING, slash=FORWARD, order=1),
    ToyRule("backward-application", MATCHING, slash=BACKWARD, order=0),
    ToyRule("backward-composition", MATCHING, slash=BACKWARD, order=1),
)


def ruleset(
    normal_form: NormalForm[Term], rules: tuple[ToyRule, ...] = RULES
) -> RuleSet[Term]:
    return RuleSet("toy", rules, MATCHING, normal_form)


def test_the_toy_rule_serializes_its_composition_direction() -> None:
    assert RULES[0].to_spec() == {
        "rule": "composition",
        "direction": "unrestricted",
    }


def scores(
    index: SpanIndex, rules: int, /, *, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    leaves = torch.randn(1, index.n, 1, generator=generator, dtype=torch.float64)
    steps = torch.randn(
        1,
        index.num_spans,
        index.max_splits,
        rules,
        generator=generator,
        dtype=torch.float64,
    )

    return (leaves, steps)


def dense_combine_reference(
    kernel: TaggedInsideKernel[Term],
    chart: _TensorChartAccess,
    rule_scores: TaggedRuleScores,
    gathers: tagged_module._Gathers,
    batch: int,
    /,
    *,
    width: int,
) -> torch.Tensor:
    """Return the former dense output-state reduction for comparison."""
    start, stop = chart.bounds(width)
    spans = stop - start
    left, right = chart.read_width_children(width)
    left = left.permute(2, 0, 1, 3)
    right = right.permute(2, 0, 1, 3)
    joined = (
        left.index_select(-1, gathers.left_states)
        + right.index_select(-1, gathers.right_states)
        + rule_scores.log_scores[:, start:stop]
    )
    base_mask = rule_scores.mask[:, start:stop]
    output_states = joined.new_tensor(
        [edge.output for edge in kernel.transitions.edges], dtype=torch.long
    )
    outputs = [
        kernel.reduction.reduce(
            joined.reshape(batch, spans, -1),
            (base_mask & (output_states == state)).reshape(batch, spans, -1),
            dim=-1,
        )
        for state in range(kernel.transitions.chart_states)
    ]

    return torch.stack(outputs, dim=-1)


@pytest.mark.parametrize("reduction", [LogSumExpReduction(), MaximumReduction()])
def test_sparse_output_reduction_matches_the_dense_sweep(
    reduction: LogWeightReduction, monkeypatch: pytest.MonkeyPatch
) -> None:
    index = SpanIndex(4)
    categories = (CATEGORY_BY_NAME["X"], CATEGORY_BY_NAME["Y"])
    kernel = TaggedInsideKernel(
        reduction,
        ruleset(EisnerNormalForm[Term]()),
        categories,
        unary_depth=0,
    )
    generator = torch.Generator().manual_seed(23)
    leaves = torch.randn(
        2,
        index.n,
        len(categories),
        dtype=torch.float64,
        generator=generator,
        requires_grad=True,
    )
    steps = torch.randn(
        2,
        index.num_spans,
        index.max_splits,
        len(kernel.transitions.rules),
        dtype=torch.float64,
        generator=generator,
        requires_grad=True,
    )
    expanded = kernel.transitions.mask_scores(steps, index)
    two_token_span = index.span_id(0, 2)
    outputs = {edge.output for edge in kernel.transitions.edges}

    assert len(outputs) < kernel.transitions.chart_states
    assert not bool(expanded.mask[:, two_token_span, 1:].any())

    sparse = kernel.inside(ScoreChannels(leaves, steps), index).inside
    sparse_finite = torch.isfinite(sparse)
    sparse_objective = sparse.masked_fill(~sparse_finite, 0.0).sum()
    sparse_gradients = torch.autograd.grad(sparse_objective, (leaves, steps))

    dense_leaves = leaves.detach().clone().requires_grad_()
    dense_steps = steps.detach().clone().requires_grad_()
    monkeypatch.setattr(TaggedInsideKernel, "_combine", dense_combine_reference)
    dense = kernel.inside(ScoreChannels(dense_leaves, dense_steps), index).inside
    dense_finite = torch.isfinite(dense)
    dense_objective = dense.masked_fill(~dense_finite, 0.0).sum()
    dense_gradients = torch.autograd.grad(dense_objective, (dense_leaves, dense_steps))

    assert torch.equal(sparse_finite, dense_finite)
    assert torch.allclose(sparse, dense)
    assert all(
        torch.allclose(sparse_gradient, dense_gradient)
        for sparse_gradient, dense_gradient in zip(
            sparse_gradients, dense_gradients, strict=True
        )
    )


def test_the_inside_sweep_uses_one_packed_tensor_chart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stores = 0

    def packed_store(
        index: SpanIndex,
        like: torch.Tensor,
        /,
        *,
        batch: int,
        width: int,
    ) -> TensorChart:
        nonlocal stores
        stores += 1

        return TensorChart(index, like, batch=batch, width=width)

    monkeypatch.setattr(tagged_module, "TensorChart", packed_store)
    index = SpanIndex(4)
    leaves, steps = scores(index, len(RULES), seed=29)
    kernel = TaggedInsideKernel(
        LogSumExpReduction(), ruleset(NoNormalForm[Term]()), CATEGORIES, unary_depth=3
    )

    kernel.inside(ScoreChannels(leaves, steps), index)

    assert stores == 1


def unique_outputs(
    rule_set: RuleSet[Term], rule: Rule[Term], inputs: tuple[Term, Term], /
) -> tuple[Term, ...]:
    found = []

    for output in rule.apply(inputs):
        if not any(rule_set.matching.equivalent(output, held) for held in found):
            found.append(output)

    return tuple(found)


def add_symbolic_steps(
    forest: PackedForest[Term, float],
    rule_set: RuleSet[Term],
    steps: torch.Tensor,
    index: SpanIndex,
    cell: tuple[int, int],
) -> None:
    left, right = cell
    parent = index.span_id(left, right)

    for split_column, split in enumerate(index.splits(left, right)):
        left_keys = forest.items_at((left, split))
        right_keys = forest.items_at((split, right))

        for left_key in left_keys:
            for right_key in right_keys:
                inputs = (forest.term(left_key), forest.term(right_key))
                child_tags = (left_key.nf_tag, right_key.nf_tag)

                for rule_column, rule in enumerate(rule_set.of_arity(2)):
                    for output in unique_outputs(rule_set, rule, inputs):
                        if not rule_set.normal_form.admits(
                            rule, child_tags, inputs, output
                        ):
                            continue

                        tag = rule_set.normal_form.tag(rule, child_tags)
                        key = ItemKey((left, right), output, tag)
                        edge = Edge(
                            rule.name,
                            (left_key, right_key),
                            output,
                            steps[0, parent, split_column, rule_column].item(),
                            split=split,
                        )
                        forest.add(key, edge)


def symbolic_partition(
    rule_set: RuleSet[Term],
    leaves: torch.Tensor,
    steps: torch.Tensor,
    index: SpanIndex,
) -> float:
    algebra = SymbolicLogInside()
    forest = PackedForest[Term, float](index, algebra)
    lexical_tag = rule_set.normal_form.lexical_tag()

    for token in range(index.n):
        key = ItemKey((token, token + 1), CATEGORIES[0], lexical_tag)
        forest.add(
            key,
            Edge("lexical", (), CATEGORIES[0], leaves[0, token, 0].item()),
        )

    for width in range(2, index.n + 1):
        for left, right in index.spans_by_length(min_length=width):
            if right - left == width:
                add_symbolic_steps(forest, rule_set, steps, index, (left, right))

    total = algebra.zero

    for key in forest.items_at((0, index.n)):
        total = algebra.plus(total, forest.inside(key))

    return total


@pytest.mark.parametrize(
    "normal_form",
    [NoNormalForm[Term](), EisnerNormalForm[Term](), HockenmaierBiskNormalForm[Term]()],
)
def test_tagged_log_partition_matches_the_symbolic_chart(
    normal_form: NormalForm[Term],
) -> None:
    index = SpanIndex(4)
    rule_set = ruleset(normal_form)
    leaves, steps = scores(index, len(RULES), seed=31)
    kernel = TaggedInsideKernel(
        LogSumExpReduction(), rule_set, CATEGORIES, unary_depth=3
    )

    tensor = kernel.log_partition(ScoreChannels(leaves, steps), index)
    symbolic = symbolic_partition(rule_set, leaves, steps, index)

    assert tensor.item() == pytest.approx(symbolic, abs=1e-5)


def test_duplicate_matching_outputs_do_not_inflate_the_partition() -> None:
    index = SpanIndex(2)
    duplicate = DuplicateRule(
        "duplicate",
        MATCHING,
        slash=FORWARD,
        order=0,
    )
    rule_set = ruleset(NoNormalForm[Term](), (duplicate,))
    leaves = torch.zeros(1, index.n, 1, dtype=torch.float64)
    steps = torch.zeros(1, index.num_spans, index.max_splits, 1, dtype=torch.float64)
    kernel = TaggedInsideKernel(
        LogSumExpReduction(), rule_set, CATEGORIES, unary_depth=3
    )

    tensor = kernel.log_partition(ScoreChannels(leaves, steps), index)
    symbolic = symbolic_partition(rule_set, leaves, steps, index)

    assert len(kernel.transitions.transitions) == 1
    assert tensor.item() == pytest.approx(0.0)
    assert tensor.item() == pytest.approx(symbolic)


@pytest.mark.parametrize("arity", [3, 4])
def test_an_nary_rule_reaches_exactly_the_symbolic_visible_states(arity: int) -> None:
    rule = NaryRule(f"combine-{arity}", MATCHING, arity=arity, slash=FORWARD, order=0)
    index = SpanIndex(arity)
    kernel = TaggedInsideKernel(
        LogSumExpReduction(),
        ruleset(NoNormalForm[Term](), (rule,)),
        CATEGORIES,
        unary_depth=0,
    )

    assert tensor_states(kernel, index) == symbolic_states(
        kernel.transitions.rule_set, CATEGORIES, index, 0
    )
    assert kernel.transitions.chart_states > kernel.transitions.states


def test_a_four_premise_step_reads_its_leaves_and_rule_score_once() -> None:
    arity = 4
    rule = NaryRule("combine-four", MATCHING, arity=arity, slash=FORWARD, order=0)
    index = SpanIndex(arity)
    kernel = TaggedInsideKernel(
        LogSumExpReduction(),
        ruleset(NoNormalForm[Term](), (rule,)),
        CATEGORIES,
        unary_depth=0,
    )
    leaves = torch.arange(1, arity + 1, dtype=torch.float64).reshape(1, arity, 1)
    steps = torch.zeros(
        1,
        index.num_spans,
        index.max_splits,
        len(kernel.transitions.rules),
        dtype=torch.float64,
    )
    root = index.span_id(0, arity)
    final_split = arity - 2
    steps[0, root, final_split, 0] = 7.0
    channels = ScoreChannels(leaves, steps)

    partition = kernel.log_partition(channels, index)
    counts = kernel.expected_counts(channels, index)

    assert partition.item() == leaves.sum().item() + 7.0
    assert counts.leaf.eq(1.0).all()
    assert counts.rule.sum().item() == 1.0
    assert counts.rule[0, root, final_split, 0].item() == 1.0


def test_hidden_prefix_states_decode_as_one_four_premise_step() -> None:
    arity = 4
    rule = NaryRule("combine-four", MATCHING, arity=arity, slash=FORWARD, order=0)
    index = SpanIndex(arity)
    kernel = TaggedInsideKernel(
        LogSumExpReduction(),
        ruleset(NoNormalForm[Term](), (rule,)),
        CATEGORIES,
        unary_depth=0,
    )
    leaves = torch.zeros(1, arity, len(CATEGORIES), dtype=torch.float64)
    steps = torch.zeros(
        1,
        index.num_spans,
        index.max_splits,
        len(kernel.transitions.rules),
        dtype=torch.float64,
    )

    tree = TaggedDecoder(kernel).k_best(ScoreChannels(leaves, steps), index)[0][0]

    assert tree.rule == 0
    assert len(tree.children) == arity
    assert all(child.rule is None for child in tree.children)


def test_a_zero_length_index_is_refused_before_transition_masking() -> None:
    index = SpanIndex(0)
    kernel = TaggedInsideKernel(
        LogSumExpReduction(),
        ruleset(NoNormalForm[Term](), (RULES[0],)),
        CATEGORIES,
        unary_depth=3,
    )
    leaves = torch.empty(1, 0, 1)
    steps = torch.empty(1, 0, 0, 1)

    with pytest.raises(ValueError, match=r"transition masking needs.*at least one"):
        kernel.transitions.mask_scores(steps, index)

    with pytest.raises(ValueError, match=r"tagged chart needs.*at least one"):
        kernel.inside(ScoreChannels(leaves, steps), index)


def test_tag_transition_is_a_mask_over_split_and_rule_scores() -> None:
    index = SpanIndex(3)
    _, steps = scores(index, len(RULES), seed=37)
    kernel = TaggedInsideKernel(
        LogSumExpReduction(),
        ruleset(EisnerNormalForm[Term]()),
        CATEGORIES,
        unary_depth=3,
    )
    expanded = kernel.transitions.mask_scores(steps, index)
    root = index.span_id(0, index.n)
    live_split = 0
    blocked = []
    admitted = []

    for column, transition in enumerate(kernel.transitions.transitions):
        rule = RULES[transition.rule]
        left_tag = kernel.transitions.tags[transition.premises[0][1]]

        if rule.slash == FORWARD and left_tag == "FC":
            blocked.append(column)

        if rule.slash == FORWARD and left_tag == "OTHER":
            admitted.append(column)

    assert blocked
    assert admitted
    assert not bool(expanded.mask[0, root, live_split, blocked].any())
    assert bool(expanded.mask[0, root, live_split, admitted].all())
    assert bool(expanded.log_scores[0, root, live_split, blocked].eq(-inf).all())
    assert torch.equal(
        expanded.log_scores[0, root, live_split, admitted],
        steps[0, root, live_split].index_select(
            0,
            torch.tensor(
                [kernel.transitions.transitions[column].rule for column in admitted]
            ),
        ),
    )


def test_a_tag_transition_is_an_immutable_value() -> None:
    premises = ((2, 3), (4, 5), (6, 7))
    transition = TagTransition(1, premises, (8, 9), admitted=True)
    same = TagTransition(1, premises, (8, 9), admitted=True)
    different = TagTransition(1, premises, (8, 10), admitted=True)

    assert transition == same
    assert transition != different
    assert hash(transition) == hash(same)
    assert repr(transition) == (
        "TagTransition(1, ((2, 3), (4, 5), (6, 7)), (8, 9), admitted=True)"
    )
    assert (
        transition.rule,
        transition.premises,
        transition.output,
        transition.admitted,
    ) == (1, premises, (8, 9), True)

    field = "_rule"

    with pytest.raises(AttributeError, match="tag transition is immutable"):
        setattr(transition, field, 9)

    with pytest.raises(AttributeError, match="tag transition is immutable"):
        delattr(transition, field)


@pytest.mark.parametrize(
    "normal_form",
    [NoNormalForm[Term](), EisnerNormalForm[Term](), HockenmaierBiskNormalForm[Term]()],
)
def test_an_unambiguous_sentence_has_zero_derivation_entropy(
    normal_form: NormalForm[Term],
) -> None:
    index = SpanIndex(2)
    only = (RULES[0],)
    leaves, steps = scores(index, len(only), seed=41)
    kernel = TaggedInsideKernel(
        LogSumExpReduction(),
        ruleset(normal_form, only),
        CATEGORIES,
        unary_depth=3,
    )

    entropy = kernel.derivation_entropy(ScoreChannels(leaves, steps), index)

    assert entropy.item() == pytest.approx(0.0, abs=TOLERANCE)


def test_uniform_three_token_scores_have_positive_entropy() -> None:
    index = SpanIndex(3)
    only = (RULES[0],)
    leaves = torch.zeros(1, index.n, 1, dtype=torch.float64)
    steps = torch.zeros(1, index.num_spans, index.max_splits, 1, dtype=torch.float64)
    kernel = TaggedInsideKernel(
        LogSumExpReduction(),
        ruleset(NoNormalForm[Term](), only),
        CATEGORIES,
        unary_depth=3,
    )

    entropy = kernel.derivation_entropy(ScoreChannels(leaves, steps), index)

    assert entropy.item() == pytest.approx(log(2.0), abs=TOLERANCE)


def test_entropy_remains_available_under_no_grad() -> None:
    index = SpanIndex(3)
    only = (RULES[0],)
    leaves = torch.zeros(1, index.n, 1, dtype=torch.float64)
    steps = torch.zeros(1, index.num_spans, index.max_splits, 1, dtype=torch.float64)
    kernel = TaggedInsideKernel(
        LogSumExpReduction(),
        ruleset(NoNormalForm[Term](), only),
        CATEGORIES,
        unary_depth=3,
    )

    with torch.no_grad():
        entropy = kernel.derivation_entropy(ScoreChannels(leaves, steps), index)

    assert entropy.item() == pytest.approx(log(2.0), abs=TOLERANCE)
    assert not entropy.requires_grad


def test_an_item_the_chart_derives_nothing_for_has_zero_entropy() -> None:
    index = SpanIndex(3)
    only = (RULES[0],)
    leaves = torch.zeros(1, index.n, 1, dtype=torch.float64)
    steps = torch.full(
        (1, index.num_spans, index.max_splits, 1), -inf, dtype=torch.float64
    )
    kernel = TaggedInsideKernel(
        LogSumExpReduction(),
        ruleset(NoNormalForm[Term](), only),
        CATEGORIES,
        unary_depth=3,
    )
    channels = ScoreChannels(leaves, steps)

    entropy = kernel.derivation_entropy(channels, index)

    assert kernel.log_partition(channels, index).item() == -inf
    assert entropy.item() == 0.0


def test_ragged_items_read_their_own_root_spans() -> None:
    index = SpanIndex(4)
    only = (RULES[0],)
    leaves = torch.zeros(2, index.n, 1, dtype=torch.float64)
    steps = torch.zeros(2, index.num_spans, index.max_splits, 1, dtype=torch.float64)
    kernel = TaggedInsideKernel(
        LogSumExpReduction(),
        ruleset(NoNormalForm[Term](), only),
        CATEGORIES,
        unary_depth=3,
    )

    partition = kernel.log_partition(
        ScoreChannels(leaves, steps), index, lengths=torch.tensor([2, 4])
    )

    assert partition.exp().tolist() == pytest.approx([1.0, 5.0])


def test_a_normal_form_blocked_step_has_zero_score_gradient() -> None:
    index = SpanIndex(3)
    only_composition = (RULES[1],)
    leaves = torch.zeros(1, index.n, 1, dtype=torch.float64)
    steps = torch.zeros(
        1,
        index.num_spans,
        index.max_splits,
        1,
        dtype=torch.float64,
        requires_grad=True,
    )
    kernel = TaggedInsideKernel(
        LogSumExpReduction(),
        ruleset(EisnerNormalForm[Term](), only_composition),
        CATEGORIES,
        unary_depth=3,
    )

    kernel.log_partition(ScoreChannels(leaves, steps), index).backward()

    assert steps.grad is not None
    root = index.span_id(0, index.n)
    assert steps.grad[0, root, 0, 0].item() == pytest.approx(1.0)
    assert steps.grad[0, root, 1, 0].item() == pytest.approx(0.0)


def test_n40_item_multiplication_is_exactly_the_declared_tag_alphabet() -> None:
    index = SpanIndex(40)
    forms = (
        NoNormalForm[Term](),
        EisnerNormalForm[Term](),
        HockenmaierBiskNormalForm[Term](),
    )

    for normal_form in forms:
        transitions = TaggedInsideKernel(
            LogSumExpReduction(),
            ruleset(normal_form),
            CATEGORIES,
            unary_depth=3,
        ).transitions
        unrefined = index.num_spans * len(CATEGORIES)
        measured = transitions.item_count(index) // unrefined

        assert measured == transitions.item_multiplier
        assert measured == len(normal_form.alphabet)


def test_the_chart_names_its_category_and_tag_axes() -> None:
    index = SpanIndex(3)
    _, steps = scores(index, len(RULES), seed=43)
    leaves = torch.zeros(1, index.n, 2, dtype=steps.dtype)
    categories = (CATEGORY_BY_NAME["X"], CATEGORY_BY_NAME["Y"])
    kernel = TaggedInsideKernel(
        LogSumExpReduction(),
        ruleset(EisnerNormalForm[Term]()),
        categories,
        unary_depth=3,
    )
    chart = kernel.inside(ScoreChannels(leaves, steps), index)

    assert chart.axes == ("item", "span", "category", "tag")
    assert chart.index is index
    assert chart.inside.shape == (
        1,
        index.num_spans,
        len(categories),
        len(kernel.transitions.tags),
    )
    assert bool(chart.inside[0, index.span_id(0, index.n)].isfinite().any(dim=-1).all())


def test_matching_equivalent_category_columns_are_refused() -> None:
    with pytest.raises(ValueError, match="positions 0 and 1 are equivalent"):
        TaggedInsideKernel(
            LogSumExpReduction(),
            ruleset(NoNormalForm[Term]()),
            (CATEGORIES[0], CATEGORIES[0]),
            unary_depth=3,
        )


class OneCategoryEach:
    """A lexical source admitting every category at every token, unweighted."""

    def __init__(self, categories: tuple[Term, ...], /) -> None:
        self._categories = categories

    @property
    def is_re_callable(self) -> bool:
        return True

    @property
    def clamps(self) -> tuple[Clamp[Term], ...]:
        return ()

    @property
    def forms(self) -> tuple[str, ...]:
        return ()

    def entries(self, form: str) -> tuple[Entry[Term], ...]:
        del form
        return ()

    def axioms(
        self, tokens: Sequence[str], /
    ) -> Iterator[tuple[int, Entry[Term], float]]:
        for position, token in enumerate(tokens):
            for category in self._categories:
                yield (position, Entry(token, category), 0.0)


class Unweighted:
    """An edge scorer giving every axiom and every step log weight zero."""

    lowering = SpanFree()

    def axiom_weight(self, position: int, entry: Entry[Term], /) -> float:
        del position, entry
        return 0.0

    def edge_weight(
        self,
        rule: Rule[Term],
        inputs: tuple[Term, ...],
        output: Term,
        span: Cell,
        split: int | None,
        /,
    ) -> float:
        del rule, inputs, output, span, split
        return 0.0


def symbolic_states(
    rule_set: RuleSet[Term],
    categories: tuple[Term, ...],
    index: SpanIndex,
    depth: int,
) -> dict[Cell, set[tuple[Term, str]]]:
    """Return the category and tag of every item the symbolic chart holds."""
    forest = parse(
        tuple(f"w{position}" for position in range(index.n)),
        policy=ParsePolicy(Retain.FOREST, Reachability.ALL),
        rules=rule_set,
        source=OneCategoryEach(categories),
        semiring=SymbolicLogInside(),
        scorer=Unweighted(),
        goal=RootGoal(TargetSet(categories, MATCHING)),
        unary_depth=depth,
    )

    return {
        span: {(key.reading, key.nf_tag) for key in forest.items_at(span)}
        for span in index.spans_by_length()
    }


def tensor_states(
    kernel: TaggedInsideKernel[Term], index: SpanIndex
) -> dict[Cell, set[tuple[Term, str]]]:
    """Return the category and tag of every state the tensor chart derives."""
    transitions = kernel.transitions
    leaves = torch.zeros(1, index.n, len(transitions.categories), dtype=torch.float64)
    steps = torch.zeros(
        1,
        index.num_spans,
        index.max_splits,
        len(transitions.rules),
        dtype=torch.float64,
    )
    unary = (
        None
        if not transitions.unary_rules
        else torch.zeros(
            1, index.num_spans, len(transitions.unary_rules), dtype=torch.float64
        )
    )
    chart = kernel.inside(ScoreChannels(leaves, steps, unary), index).inside

    return {
        span: {
            (transitions.categories[category], transitions.tags[tag])
            for category in range(len(transitions.categories))
            for tag in range(len(transitions.tags))
            if bool(chart[0, index.span_id(*span), category, tag].isfinite())
        }
        for span in index.spans_by_length()
    }


def raising_rules() -> tuple[ToyRule, ...]:
    return (
        RaisingRule("type-raising", MATCHING, slash=FORWARD, order=0),
        ToyRule("forward-composition", MATCHING, slash=FORWARD, order=1),
        ToyRule("backward-composition", MATCHING, slash=BACKWARD, order=1),
    )


def raising_kernel(depth: int) -> TaggedInsideKernel[Term]:
    return TaggedInsideKernel(
        LogSumExpReduction(),
        ruleset(HockenmaierBiskNormalForm[Term](), raising_rules()),
        RAISED,
        unary_depth=depth,
    )


def unary_scores(
    kernel: TaggedInsideKernel[Term], index: SpanIndex, /, *, seed: int
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)

    return torch.randn(
        1,
        index.num_spans,
        len(kernel.transitions.unary_rules),
        generator=generator,
        dtype=torch.float64,
    )


def raising_channels(
    kernel: TaggedInsideKernel[Term], index: SpanIndex, /, *, seed: int
) -> ScoreChannels:
    generator = torch.Generator().manual_seed(seed)
    leaves = torch.randn(
        1, index.n, len(RAISED), generator=generator, dtype=torch.float64
    )
    steps = torch.randn(
        1,
        index.num_spans,
        index.max_splits,
        len(kernel.transitions.rules),
        generator=generator,
        dtype=torch.float64,
    )

    return ScoreChannels(leaves, steps, unary_scores(kernel, index, seed=seed + 1))


@pytest.mark.parametrize("n", [1, 2, 3])
def test_the_tensor_and_symbolic_charts_reach_the_same_states(n: int) -> None:
    depth = DEPTH
    index = SpanIndex(n)
    kernel = raising_kernel(depth)

    assert tensor_states(kernel, index) == symbolic_states(
        kernel.transitions.rule_set, RAISED, index, depth
    )


@pytest.mark.parametrize("n", [1, 2, 3])
def test_a_type_raised_derivation_reaches_the_hockenmaier_bisk_tr_state(
    n: int,
) -> None:
    depth = DEPTH
    index = SpanIndex(n)
    kernel = raising_kernel(depth)
    lifted = (FUNCTOR, "FTR")

    tensor = tensor_states(kernel, index)
    symbolic = symbolic_states(kernel.transitions.rule_set, RAISED, index, depth)

    assert lifted in tensor[(0, index.n)]
    assert lifted in symbolic[(0, index.n)]


def test_a_depth_of_zero_leaves_the_raised_state_out_of_reach() -> None:
    index = SpanIndex(2)
    kernel = raising_kernel(0)

    reached = tensor_states(kernel, index)

    assert (FUNCTOR, "FTR") not in reached[(0, 1)]
    assert kernel.transitions.truncated


def test_a_closure_that_applies_every_step_is_not_truncated() -> None:
    index = SpanIndex(2)
    kernel = raising_kernel(DEPTH)
    channels = raising_channels(kernel, index, seed=11)

    chart = kernel.inside(channels, index)

    assert not chart.truncated
    assert not kernel.transitions.truncated


def test_a_unary_rule_needs_a_unary_score_channel() -> None:
    index = SpanIndex(2)
    kernel = raising_kernel(DEPTH)
    channels = raising_channels(kernel, index, seed=13)

    with pytest.raises(ValueError, match=r"1 unary rules need a unary score channel"):
        kernel.inside(ScoreChannels(channels.leaf, channels.rule), index)


def test_a_rule_set_declaring_no_unary_rule_takes_no_unary_channel() -> None:
    index = SpanIndex(2)
    kernel = TaggedInsideKernel(
        LogSumExpReduction(),
        ruleset(NoNormalForm[Term]()),
        CATEGORIES,
        unary_depth=3,
    )
    leaves, steps = scores(index, len(RULES), seed=17)
    spare = torch.zeros(1, index.num_spans, 1, dtype=steps.dtype)

    with pytest.raises(ValueError, match=r"declaring no unary rule takes no unary"):
        kernel.inside(ScoreChannels(leaves, steps, spare), index)


def test_a_unary_channel_of_the_wrong_shape_is_refused() -> None:
    index = SpanIndex(2)
    kernel = raising_kernel(DEPTH)
    channels = raising_channels(kernel, index, seed=19)
    wrong = torch.zeros(1, index.num_spans, 2, dtype=torch.float64)

    with pytest.raises(ValueError, match=r"unary scores have shape"):
        kernel.inside(ScoreChannels(channels.leaf, channels.rule, wrong), index)


def test_a_unary_channel_that_is_not_floating_point_is_refused() -> None:
    index = SpanIndex(2)
    kernel = raising_kernel(DEPTH)
    counts = torch.zeros(1, index.num_spans, 1, dtype=torch.int64)

    with pytest.raises(TypeError, match=r"unary scores are log weights"):
        kernel.transitions.checked_unary_scores(counts, index, 1)


def test_a_raising_step_lifts_the_cell_by_its_own_score() -> None:
    index = SpanIndex(1)
    kernel = raising_kernel(1)
    transitions = kernel.transitions
    leaves = torch.tensor([[[log(2.0), -inf]]], dtype=torch.float64)
    steps = torch.zeros(
        1,
        index.num_spans,
        index.max_splits,
        len(transitions.rules),
        dtype=torch.float64,
    )
    unary = torch.tensor([[[log(3.0)]]], dtype=torch.float64)

    chart = kernel.inside(ScoreChannels(leaves, steps, unary), index).inside
    raised = chart[0, 0, RAISED.index(FUNCTOR), transitions.tags.index("FTR")]

    assert raised.exp().item() == pytest.approx(6.0)


def test_the_channels_must_agree_on_how_many_items_they_cover() -> None:
    index = SpanIndex(2)
    kernel = TaggedInsideKernel(
        LogSumExpReduction(),
        ruleset(NoNormalForm[Term]()),
        CATEGORIES,
        unary_depth=3,
    )
    leaves, steps = scores(index, len(RULES), seed=29)

    with pytest.raises(ValueError, match=r"lexical scores for 2 items came with"):
        kernel.transitions.checked_channels(
            ScoreChannels(leaves.expand(2, -1, -1), steps), index
        )


def test_the_goal_columns_come_back_in_the_order_the_goals_name_them() -> None:
    kernel = raising_kernel(DEPTH)

    assert kernel.transitions.goal_categories(None) == (0, 1)
    assert kernel.transitions.goal_categories((FUNCTOR, ARGUMENT, FUNCTOR)) == (1, 0)


def test_an_empty_goal_sequence_is_refused() -> None:
    kernel = raising_kernel(DEPTH)

    with pytest.raises(ValueError, match=r"at least one goal category"):
        kernel.transitions.goal_categories(())


def test_the_expected_counts_read_every_channel_the_partition_read() -> None:
    index = SpanIndex(3)
    kernel = raising_kernel(DEPTH)
    channels = raising_channels(kernel, index, seed=31)

    supplied = channels.unary

    assert supplied is not None

    counts = kernel.expected_counts(channels, index)

    assert counts.leaf.shape == channels.leaf.shape
    assert counts.rule.shape == channels.rule.shape
    assert counts.unary is not None
    assert counts.unary.shape == supplied.shape
    assert counts.leaf.sum().item() == pytest.approx(float(index.n))


def test_the_expected_binary_count_is_one_less_than_the_length() -> None:
    index = SpanIndex(4)
    kernel = raising_kernel(DEPTH)
    channels = raising_channels(kernel, index, seed=37)

    counts = kernel.expected_counts(channels, index)

    assert counts.rule.sum().item() == pytest.approx(float(index.n - 1))


def test_the_span_marginals_are_the_counts_of_the_steps_that_build_them() -> None:
    index = SpanIndex(3)
    kernel = raising_kernel(DEPTH)
    channels = raising_channels(kernel, index, seed=41)

    counts = kernel.expected_counts(channels, index)
    marginals = span_marginals(counts, index)

    reference = enumerated_marginals(kernel, channels, RAISED, index)

    assert marginals.shape == (1, index.num_spans)
    assert marginals[0, : index.n].tolist() == pytest.approx([1.0] * index.n)
    assert marginals[0, index.span_id(0, index.n)].item() == pytest.approx(1.0)
    assert [marginals[0, index.span_id(*span)].item() for span in reference] == (
        pytest.approx(list(reference.values()), abs=TOLERANCE)
    )


def test_the_expected_counts_stay_attached_where_a_gradient_is_wanted() -> None:
    index = SpanIndex(3)
    kernel = raising_kernel(DEPTH)
    channels = raising_channels(kernel, index, seed=43)
    channels.rule.requires_grad_()

    counts = kernel.expected_counts(channels, index)

    assert counts.rule.requires_grad

    with torch.no_grad():
        detached = kernel.expected_counts(channels, index)

    assert not detached.rule.requires_grad


def test_a_unary_score_moves_the_partition_it_enters() -> None:
    index = SpanIndex(2)
    kernel = raising_kernel(DEPTH)
    channels = raising_channels(kernel, index, seed=47)
    supplied = channels.unary

    assert supplied is not None

    lifted = ScoreChannels(channels.leaf, channels.rule, supplied + log(2.0))

    before = kernel.log_partition(channels, index)
    after = kernel.log_partition(lifted, index)

    assert after.item() > before.item()


@pytest.mark.parametrize("n", [1, 2, 3])
def test_the_unary_partition_matches_the_symbolic_chart(n: int) -> None:
    index = SpanIndex(n)
    depth = DEPTH
    kernel = raising_kernel(depth)
    transitions = kernel.transitions
    leaves = torch.zeros(1, index.n, len(RAISED), dtype=torch.float64)
    steps = torch.zeros(
        1,
        index.num_spans,
        index.max_splits,
        len(transitions.rules),
        dtype=torch.float64,
    )
    unary = torch.zeros(
        1, index.num_spans, len(transitions.unary_rules), dtype=torch.float64
    )
    forest = parse(
        tuple(f"w{position}" for position in range(index.n)),
        policy=ParsePolicy(Retain.FOREST, Reachability.ALL),
        rules=transitions.rule_set,
        source=OneCategoryEach(RAISED),
        semiring=SymbolicLogInside(),
        scorer=Unweighted(),
        goal=RootGoal(TargetSet(RAISED, MATCHING)),
        unary_depth=depth,
    )
    algebra = SymbolicLogInside()
    total = algebra.zero

    for key in forest.items_at((0, index.n)):
        total = algebra.plus(total, forest.inside(key))

    tensor = kernel.log_partition(ScoreChannels(leaves, steps, unary), index)

    assert tensor.item() == pytest.approx(total, abs=TOLERANCE)


def test_the_expected_counts_can_be_differentiated_in_their_turn() -> None:
    index = SpanIndex(3)
    kernel = TaggedInsideKernel(
        LogSumExpReduction(),
        ruleset(NoNormalForm[Term]()),
        CATEGORIES,
        unary_depth=3,
    )
    leaves, steps = scores(index, len(RULES), seed=53)
    steps.requires_grad_()

    counts = kernel.expected_counts(ScoreChannels(leaves, steps), index)
    # the total expected count is fixed by the length, so square it to move it
    (counts.rule * counts.rule).sum().backward()

    assert steps.grad is not None
    assert bool(torch.isfinite(steps.grad).all())
    assert steps.grad.abs().sum().item() > 0.0


def test_the_unary_counts_can_be_differentiated_in_their_turn() -> None:
    index = SpanIndex(3)
    kernel = raising_kernel(DEPTH)
    channels = raising_channels(kernel, index, seed=59)
    supplied = channels.unary

    assert supplied is not None

    supplied.requires_grad_()

    counts = kernel.expected_counts(channels, index)

    assert counts.unary is not None

    (counts.unary * counts.unary).sum().backward()

    assert supplied.grad is not None
    assert bool(torch.isfinite(supplied.grad).all())
    assert supplied.grad.abs().sum().item() > 0.0


def test_the_entropy_can_be_differentiated_in_its_turn() -> None:
    index = SpanIndex(3)
    kernel = raising_kernel(DEPTH)
    channels = raising_channels(kernel, index, seed=61)
    channels.rule.requires_grad_()

    kernel.derivation_entropy(channels, index).sum().backward()

    assert channels.rule.grad is not None
    assert bool(torch.isfinite(channels.rule.grad).all())
    assert channels.rule.grad.abs().sum().item() > 0.0


class LiftRule(ToyRule):
    """A unary rule lifting one named category to another."""

    def __init__(
        self,
        name: str,
        matching: TermMatching,
        /,
        *,
        source: Term,
        output: Term,
    ) -> None:
        super().__init__(name, matching, slash=FORWARD, order=0)
        self._source = source
        self._output = output
        compiled = pattern_kernel((NAME_OF[source],), NAME_OF[output])
        self._lowering = compiled.lowering
        self._reachability_schemata = compiled.reachability_schemata

    @property
    @override
    def arity(self) -> int:
        return 1

    @property
    @override
    def normal_form_role(self) -> NormalFormRole:
        return NormalFormRole(raising="forward")

    @override
    def apply(self, inputs: tuple[Term, ...], /) -> Iterator[Term]:
        if inputs == (self._source,):
            yield self._output

    @override
    def admits(self, inputs: tuple[Term, ...], output: Term, /) -> bool:
        return inputs == (self._source,) and output == self._output

    @override
    def to_spec(self) -> Mapping[str, JsonValue]:
        return {"rule": "type-raising"}


DIAMOND: Final = tuple(CATEGORY_BY_NAME[name] for name in ("a", "b", "c", "d"))
LADDER: Final = tuple(CATEGORY_BY_NAME[name] for name in ("c0", "c1", "c2", "c3", "c4"))


def diamond_rules() -> tuple[ToyRule, ...]:
    """Return two unary paths of unequal length between the same two categories."""
    return (
        LiftRule("left", MATCHING, source=DIAMOND[0], output=DIAMOND[1]),
        LiftRule("right", MATCHING, source=DIAMOND[0], output=DIAMOND[2]),
        LiftRule("join", MATCHING, source=DIAMOND[1], output=DIAMOND[2]),
        LiftRule("top", MATCHING, source=DIAMOND[2], output=DIAMOND[3]),
        ToyRule("forward-application", MATCHING, slash=FORWARD, order=0),
    )


def ladder_rules() -> tuple[ToyRule, ...]:
    """Return one unary chain long enough to feel a change of one in the depth."""
    return tuple(
        LiftRule(f"lift-{output}", MATCHING, source=source, output=output)
        for source, output in pairwise(LADDER)
    )


def base_channels(
    kernel: TaggedInsideKernel[Term], index: SpanIndex, base: Term
) -> ScoreChannels:
    """Return unweighted channels admitting one category alone at every token."""
    transitions = kernel.transitions
    leaves = torch.full(
        (1, index.n, len(transitions.categories)), -inf, dtype=torch.float64
    )
    leaves[0, :, transitions.categories.index(base)] = 0.0
    steps = torch.zeros(
        1,
        index.num_spans,
        index.max_splits,
        len(transitions.rules),
        dtype=torch.float64,
    )
    unary = torch.zeros(
        1, index.num_spans, len(transitions.unary_rules), dtype=torch.float64
    )

    return ScoreChannels(leaves, steps, unary)


def forest_partition(forest: PackedForest[Term, float], n: int) -> float:
    """Return the total weight of every item spanning the whole input."""
    algebra = SymbolicLogInside()
    total = algebra.zero

    for key in forest.items_at((0, n)):
        total = algebra.plus(total, forest.inside(key))

    return total


def tokens_of(index: SpanIndex) -> tuple[str, ...]:
    return tuple(f"w{position}" for position in range(index.n))


@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_a_diamond_of_unary_rules_closes_as_the_symbolic_chart_does(
    n: int, depth: int
) -> None:
    index = SpanIndex(n)
    rule_set = ruleset(NoNormalForm[Term](), diamond_rules())
    kernel = TaggedInsideKernel(
        LogSumExpReduction(), rule_set, DIAMOND, unary_depth=depth
    )
    forest = parse(
        tokens_of(index),
        policy=ParsePolicy(Retain.FOREST, Reachability.ALL),
        rules=rule_set,
        source=OneCategoryEach((DIAMOND[0],)),
        semiring=SymbolicLogInside(),
        scorer=Unweighted(),
        goal=RootGoal(TargetSet(DIAMOND, MATCHING)),
        unary_depth=depth,
    )

    tensor = kernel.log_partition(base_channels(kernel, index, DIAMOND[0]), index)

    assert tensor.item() == pytest.approx(
        forest_partition(forest, index.n), abs=TOLERANCE
    )
    assert kernel.transitions.truncated == forest.truncated


def test_the_declared_depth_closes_what_the_symbolic_depth_closes() -> None:
    index = SpanIndex(1)
    rule_set = ruleset(NoNormalForm[Term](), ladder_rules())
    kernel = TaggedInsideKernel(LogSumExpReduction(), rule_set, LADDER, unary_depth=3)
    forest = parse(
        tokens_of(index),
        policy=ParsePolicy(Retain.FOREST, Reachability.ALL),
        rules=rule_set,
        source=OneCategoryEach((LADDER[0],)),
        semiring=SymbolicLogInside(),
        scorer=Unweighted(),
        goal=RootGoal(TargetSet(LADDER, MATCHING)),
        unary_depth=3,
    )

    tensor = kernel.log_partition(base_channels(kernel, index, LADDER[0]), index)

    assert tensor.item() == pytest.approx(
        forest_partition(forest, index.n), abs=TOLERANCE
    )
    assert kernel.transitions.truncated == forest.truncated


def test_the_transitions_and_the_kernel_use_one_declared_depth() -> None:
    rule_set = ruleset(NoNormalForm[Term](), ladder_rules())

    alone = NormalFormTransitions(rule_set, LADDER, unary_depth=3)
    through = TaggedInsideKernel(
        LogSumExpReduction(), rule_set, LADDER, unary_depth=3
    ).transitions

    assert alone.closure.depth == through.closure.depth


def test_a_padded_position_is_bracketed_by_nothing() -> None:
    index = SpanIndex(4)
    only = (RULES[0],)
    kernel = TaggedInsideKernel(
        LogSumExpReduction(),
        ruleset(NoNormalForm[Term](), only),
        CATEGORIES,
        unary_depth=3,
    )
    leaves = torch.zeros(2, index.n, 1, dtype=torch.float64)
    steps = torch.zeros(2, index.num_spans, index.max_splits, 1, dtype=torch.float64)

    found = kernel.constituent_marginals(
        ScoreChannels(leaves, steps), index, lengths=torch.tensor([2, 4])
    )

    assert found[0, : index.n].tolist() == pytest.approx([1.0, 1.0, 0.0, 0.0])
    assert found[1, : index.n].tolist() == pytest.approx([1.0] * index.n)
    assert found[0, index.span_id(0, 2)].item() == pytest.approx(1.0)
    assert found[0, index.span_id(0, index.n)].item() == pytest.approx(0.0)


type Columns = tuple[dict[str, int], dict[str, int]]
"""The binary and unary score columns of each rule, by name."""

type Application = tuple["Rule[Term]", "Cell", int | None]
"""One symbolic rule application: its rule, its span and its cut point."""


def rule_columns(kernel: TaggedInsideKernel[Term]) -> Columns:
    """Return the binary and unary score columns, by rule name."""
    transitions = kernel.transitions

    return (
        {rule.name: column for column, rule in enumerate(transitions.rules)},
        {rule.name: column for column, rule in enumerate(transitions.unary_rules)},
    )


def edge_score(
    columns: Columns, channels: ScoreChannels, index: SpanIndex, edge: Application
) -> float:
    """Return the score the tensor chart reads for one symbolic rule application."""
    rule, span, split = edge
    start, end = span
    held = index.span_id(start, end)
    unary = channels.unary

    if split is None:
        if unary is None:
            message = f"{rule.name!r} has no unary channel to read"
            raise AssertionError(message)

        return unary[0, held, columns[1][rule.name]].item()

    return channels.rule[0, held, split - start - 1, columns[0][rule.name]].item()


class ChannelSource:
    """A lexical source weighting every category at every token from the channel."""

    def __init__(self, leaf: torch.Tensor, categories: tuple[Term, ...], /) -> None:
        self._leaf = leaf
        self._categories = categories

    @property
    def is_re_callable(self) -> bool:
        return True

    @property
    def clamps(self) -> tuple[Clamp[Term], ...]:
        return ()

    @property
    def forms(self) -> tuple[str, ...]:
        return ()

    def entries(self, form: str) -> tuple[Entry[Term], ...]:
        del form
        return ()

    def axioms(
        self, tokens: Sequence[str], /
    ) -> Iterator[tuple[int, Entry[Term], float]]:
        for position, token in enumerate(tokens):
            for column, category in enumerate(self._categories):
                yield (
                    position,
                    Entry(token, category),
                    self._leaf[0, position, column].item(),
                )


class ChannelScorer:
    """An edge scorer reading the same channels the tensor chart reads."""

    lowering = SpanDependent()

    def __init__(
        self,
        kernel: TaggedInsideKernel[Term],
        channels: ScoreChannels,
        index: SpanIndex,
        /,
    ) -> None:
        self._columns = rule_columns(kernel)
        self._channels = channels
        self._index = index

    def axiom_weight(self, position: int, entry: Entry[Term], /) -> float:
        del position, entry
        return 0.0

    def edge_weight(
        self,
        rule: Rule[Term],
        inputs: tuple[Term, ...],
        output: Term,
        span: Cell,
        split: int | None,
        /,
    ) -> float:
        del inputs, output

        return edge_score(
            self._columns, self._channels, self._index, (rule, span, split)
        )


def derivation_total(derivation: Derivation[Term, float]) -> float:
    """Return the summed weight of one derivation and its premises."""
    return derivation.edge.weight + sum(
        derivation_total(child) for child in derivation.children
    )


def derivation_brackets(derivation: Derivation[Term, float]) -> frozenset[Cell]:
    """Return the spans one derivation builds from two premises."""
    held: frozenset[Cell] = (
        frozenset({derivation.key.span})
        if len(derivation.children) == 2
        else frozenset()
    )

    return held.union(*(derivation_brackets(child) for child in derivation.children))


def enumerated_marginals(
    kernel: TaggedInsideKernel[Term],
    channels: ScoreChannels,
    categories: tuple[Term, ...],
    index: SpanIndex,
) -> dict[Cell, float]:
    """Return each span's bracketing share of every symbolic derivation's weight."""
    forest = parse(
        tokens_of(index),
        policy=ParsePolicy(Retain.FOREST, Reachability.ALL),
        rules=kernel.transitions.rule_set,
        source=ChannelSource(channels.leaf, categories),
        semiring=SymbolicLogInside(),
        scorer=ChannelScorer(kernel, channels, index),
        goal=RootGoal(TargetSet(categories, MATCHING)),
        unary_depth=DEPTH,
    )
    found = [
        (exp(derivation_total(derivation)), derivation_brackets(derivation))
        for key in forest.items_at((0, index.n))
        for derivation in forest.derivations(key)
    ]
    mass = sum(weight for weight, _ in found)

    return {
        span: sum(weight for weight, held in found if span in held) / mass
        for span in index.spans_by_length(min_length=2)
    }


@pytest.mark.parametrize("n", [2, 3])
def test_the_constituent_marginals_are_the_share_of_trees_holding_the_span(
    n: int,
) -> None:
    index = SpanIndex(n)
    kernel = raising_kernel(DEPTH)
    channels = raising_channels(kernel, index, seed=n + 101)
    reference = enumerated_marginals(kernel, channels, RAISED, index)

    found = kernel.constituent_marginals(channels, index)

    assert reference
    assert [found[0, index.span_id(*span)].item() for span in reference] == (
        pytest.approx(list(reference.values()), abs=TOLERANCE)
    )


class PairRule(ToyRule):
    """A binary rule combining two named categories into a third."""

    def __init__(
        self,
        name: str,
        matching: TermMatching,
        /,
        *,
        left: Term,
        right: Term,
        output: Term,
    ) -> None:
        super().__init__(name, matching, slash=FORWARD, order=0)
        self._premises = (left, right)
        self._output = output
        compiled = pattern_kernel((NAME_OF[left], NAME_OF[right]), NAME_OF[output])
        self._lowering = compiled.lowering
        self._reachability_schemata = compiled.reachability_schemata

    @override
    def apply(self, inputs: tuple[Term, ...], /) -> Iterator[Term]:
        if inputs == self._premises:
            yield self._output

    @override
    def admits(self, inputs: tuple[Term, ...], output: Term, /) -> bool:
        return inputs == self._premises and output == self._output


PAIRED: Final = tuple(CATEGORY_BY_NAME[name] for name in ("a", "b", "c"))


def paired_kernel() -> TaggedInsideKernel[Term]:
    """Return a grammar whose only goal derivation of three tokens is (a a) a."""
    a, b, c = PAIRED

    return TaggedInsideKernel(
        MaximumReduction(),
        ruleset(
            NoNormalForm[Term](),
            (
                PairRule("pair", MATCHING, left=a, right=a, output=b),
                PairRule("close", MATCHING, left=b, right=a, output=c),
            ),
        ),
        PAIRED,
        unary_depth=0,
    )


def zero_channels(
    kernel: TaggedInsideKernel[Term], index: SpanIndex, /
) -> ScoreChannels:
    """Return channels giving every category, step and lift log weight zero."""
    transitions = kernel.transitions
    unary = (
        None
        if not transitions.unary_rules
        else torch.zeros(
            1, index.num_spans, len(transitions.unary_rules), dtype=torch.float64
        )
    )

    return ScoreChannels(
        torch.zeros(1, index.n, len(transitions.categories), dtype=torch.float64),
        torch.zeros(
            1,
            index.num_spans,
            index.max_splits,
            len(transitions.rules),
            dtype=torch.float64,
        ),
        unary,
    )


def supported_categories(
    kernel: TaggedInsideKernel[Term], support: torch.Tensor, index: SpanIndex
) -> dict[Cell, set[Term]]:
    """Return the categories a support tensor marks at each span."""
    categories = kernel.transitions.categories

    return {
        span: {
            category
            for column, category in enumerate(categories)
            if bool(support[0, index.span_id(*span), column].any())
        }
        for span in index.spans_by_length()
    }


def test_a_category_derivable_off_every_goal_derivation_is_unsupported() -> None:
    index = SpanIndex(3)
    kernel = paired_kernel()
    channels = zero_channels(kernel, index)
    a, b, c = PAIRED

    inside = kernel.inside(channels, index).inside
    support = kernel.goal_support(channels, index, goals=(c,))

    assert bool(inside[0, index.span_id(1, 3), PAIRED.index(b)].isfinite().any())
    assert supported_categories(kernel, support, index) == {
        (0, 1): {a},
        (1, 2): {a},
        (2, 3): {a},
        (0, 2): {b},
        (1, 3): set(),
        (0, 3): {c},
    }


def goal_derivation_states(
    kernel: TaggedInsideKernel[Term],
    categories: tuple[Term, ...],
    goals: tuple[Term, ...],
    index: SpanIndex,
    depth: int,
) -> set[tuple[Cell, Term, str]]:
    """Return every item some symbolic derivation of a goal at the root uses."""
    forest = parse(
        tokens_of(index),
        policy=ParsePolicy(Retain.FOREST, Reachability.ALL),
        rules=kernel.transitions.rule_set,
        source=OneCategoryEach(categories),
        semiring=SymbolicLogInside(),
        scorer=Unweighted(),
        goal=RootGoal(TargetSet(categories, MATCHING)),
        unary_depth=depth,
    )
    found: set[tuple[Cell, Term, str]] = set()
    pending = [
        derivation
        for key in forest.items_at((0, index.n))
        if key.reading in goals
        for derivation in forest.derivations(key)
    ]

    while pending:
        derivation = pending.pop()
        key = derivation.key
        found.add((key.span, key.reading, key.nf_tag))
        pending.extend(derivation.children)

    return found


def tensor_goal_states(
    kernel: TaggedInsideKernel[Term],
    goals: tuple[Term, ...],
    index: SpanIndex,
    channels: ScoreChannels | None = None,
) -> set[tuple[Cell, Term, str]]:
    """Return every state the tensor goal support marks."""
    transitions = kernel.transitions
    support = kernel.goal_support(
        zero_channels(kernel, index) if channels is None else channels,
        index,
        goals=goals,
    )

    return {
        (span, category, transitions.tags[tag])
        for span in index.spans_by_length()
        for column, category in enumerate(transitions.categories)
        for tag in range(len(transitions.tags))
        if bool(support[0, index.span_id(*span), column, tag])
    }


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("goal", [ARGUMENT, FUNCTOR])
def test_goal_support_under_unary_closure_is_the_symbolic_goal_derivations(
    n: int, goal: Term
) -> None:
    index = SpanIndex(n)
    kernel = raising_kernel(DEPTH)

    assert tensor_goal_states(kernel, (goal,), index) == goal_derivation_states(
        kernel, RAISED, (goal,), index, DEPTH
    )


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_goal_support_through_a_unary_diamond_is_the_symbolic_goal_derivations(
    n: int, depth: int
) -> None:
    index = SpanIndex(n)
    kernel = TaggedInsideKernel(
        MaximumReduction(),
        ruleset(NoNormalForm[Term](), diamond_rules()),
        DIAMOND,
        unary_depth=depth,
    )
    goals = (DIAMOND[2],)

    assert tensor_goal_states(kernel, goals, index) == goal_derivation_states(
        kernel, DIAMOND, goals, index, depth
    )


@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("depth", [1, 2, 3, 4])
@pytest.mark.parametrize(
    ("rules", "categories", "goal"),
    [
        (diamond_rules(), DIAMOND, DIAMOND[2]),
        (ladder_rules(), LADDER, LADDER[-1]),
        (ladder_rules(), LADDER, LADDER[2]),
    ],
)
def test_goal_support_lifts_only_premises_the_closure_depth_admits(
    n: int,
    depth: int,
    rules: tuple[ToyRule, ...],
    categories: tuple[Term, ...],
    goal: Term,
) -> None:
    index = SpanIndex(n)
    kernel = TaggedInsideKernel(
        MaximumReduction(),
        ruleset(NoNormalForm[Term](), rules),
        categories,
        unary_depth=depth,
    )
    channels = base_channels(kernel, index, categories[0])

    assert tensor_goal_states(
        kernel, (goal,), index, channels
    ) == goal_derivation_states(kernel, (categories[0],), (goal,), index, depth)


@pytest.mark.parametrize("n", [3, 4, 5])
def test_goal_support_through_hidden_prefixes_is_the_symbolic_goal_derivations(
    n: int,
) -> None:
    rule = NaryRule("combine-3", MATCHING, arity=3, slash=FORWARD, order=0)
    index = SpanIndex(n)
    kernel = TaggedInsideKernel(
        MaximumReduction(),
        ruleset(NoNormalForm[Term](), (rule,)),
        CATEGORIES,
        unary_depth=0,
    )

    assert tensor_goal_states(kernel, CATEGORIES, index) == goal_derivation_states(
        kernel, CATEGORIES, CATEGORIES, index, 0
    )


def test_a_span_every_goal_derivation_brackets_is_required() -> None:
    index = SpanIndex(3)
    kernel = paired_kernel()
    channels = zero_channels(kernel, index)
    goals = (PAIRED[2],)

    required = {
        span: kernel.requires_span(channels, index, span, goals=goals).tolist()
        for span in index.spans_by_length()
    }

    assert required == {
        (0, 1): [False],
        (1, 2): [False],
        (2, 3): [False],
        (0, 2): [True],
        (1, 3): [False],
        (0, 3): [True],
    }


def test_every_span_is_required_of_an_input_with_no_goal_derivation() -> None:
    index = SpanIndex(4)
    kernel = paired_kernel()
    channels = zero_channels(kernel, index)
    goals = (PAIRED[2],)

    assert kernel.log_partition(channels, index, goals=goals).item() == -inf
    assert all(
        kernel.requires_span(channels, index, span, goals=goals).tolist() == [True]
        for span in index.spans_by_length()
    )
