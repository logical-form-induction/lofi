import math
from functools import cache
from itertools import product
from math import inf
from typing import TYPE_CHECKING, NamedTuple, cast

import pytest
import torch
from tests.support.rules import ccg_signature, declared_rules

from lofi._native import SpanDependent, Term
from lofi.deduce import LogInside as SymbolicLogInside
from lofi.deduce import (
    MarginalScorer,
    PackedForest,
    ParsePolicy,
    Reachability,
    Retain,
    RootGoal,
    SpanIndex,
    outside_from_edges,
    parse,
)
from lofi.judge import NoNormalForm, NormalFormRole, RuleSet
from lofi.lexicon import Entry
from lofi.nn import parsers
from lofi.run._benchmark_fixture import packaged_pinned_chart
from lofi.spec import ConstantPatternSpec, PatternRuleSpec, RuleSetSpec
from lofi.terms import TargetSet

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from lofi._types import JsonValue
    from lofi.deduce import Cell, ItemKey
    from lofi.judge import (
        Direction,
        PatternRule,
        Rule,
        RuleProbe,
        RuleRestriction,
        SchematicSequent,
        Slot,
    )
    from lofi.lexicon import Clamp
    from lofi.terms import TermMatching, TermRuleKernel


def test_the_published_surface_scores_a_batch_under_either_reduction() -> None:
    index = SpanIndex(4)
    leaves = torch.zeros(2, index.n)
    splits = torch.zeros(2, index.num_spans, index.max_splits)
    summed = parsers.InsideKernel(parsers.LogSumExpReduction())
    best = parsers.InsideKernel(parsers.MaximumReduction())

    assert isinstance(summed.reduction, parsers.LogWeightReduction)
    assert summed.log_partition(leaves, splits, index).exp().tolist() == pytest.approx(
        [5.0, 5.0]
    )
    assert best.log_partition(leaves, splits, index).exp().tolist() == pytest.approx(
        [1.0, 1.0]
    )


def test_the_published_surface_reads_the_chart_of_every_span() -> None:
    index = SpanIndex(3)
    leaves = torch.zeros(1, index.n)
    splits = torch.zeros(1, index.num_spans, index.max_splits)

    chart = parsers.InsideKernel(parsers.LogSumExpReduction()).inside(
        leaves, splits, index
    )

    assert chart[0, index.span_id(0, 1)].exp().item() == pytest.approx(1.0)
    assert chart[0, index.span_id(0, 2)].exp().item() == pytest.approx(1.0)
    assert chart[0, index.span_id(0, 3)].exp().item() == pytest.approx(2.0)


def test_the_published_surface_stores_a_cell_and_gathers_its_children() -> None:
    index = SpanIndex(3)
    chart = parsers.TensorChart(index, torch.zeros(()), batch=1, width=2)

    chart.write(0, 1, torch.tensor([[1.0, 2.0]]))
    chart.write(1, 3, torch.tensor([[3.0, 4.0]]))
    left, right = chart.read_children(0, 3)

    assert chart.read(0, 1).tolist() == [[1.0, 2.0]]
    assert left[0].tolist() == [[1.0, 2.0]]
    assert right[0].tolist() == [[3.0, 4.0]]


def test_the_published_surface_carries_a_scalar_beside_every_cell() -> None:
    index = SpanIndex(3)
    vectors = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    leaves = torch.zeros(1, index.n)
    splits = torch.zeros(1, index.num_spans, index.max_splits)

    chart = parsers.SplitChannel(
        lambda left, right: left + right,
        weigher=parsers.MarginalSplitWeigher(),
    ).sweep(vectors, leaves, splits, index)

    assert isinstance(chart, parsers.ChanneledChart)
    assert chart.inside[0, index.span_id(0, 3)].exp().item() == pytest.approx(2.0)
    assert chart.alpha[0, index.span_id(0, 3)].exp().tolist() == pytest.approx(
        [0.5, 0.5]
    )
    assert chart.cells[index.span_id(0, 3), 0].tolist() == pytest.approx([2.0, 2.0])


def test_the_published_surface_reads_a_posterior_off_a_scalar_chart() -> None:
    index = SpanIndex(3)
    leaves = torch.zeros(1, index.n)
    splits = torch.zeros(1, index.num_spans, index.max_splits)

    inside = parsers.InsideKernel(parsers.LogSumExpReduction()).inside(
        leaves, splits, index
    )
    alpha = parsers.log_split_posterior(inside, splits, index)

    assert alpha[0, index.span_id(0, 3)].exp().tolist() == pytest.approx([0.5, 0.5])


def test_the_published_surface_reads_an_outside_chart_off_an_inside_one() -> None:
    index = SpanIndex(3)
    leaves = torch.zeros(1, index.n)
    splits = torch.zeros(1, index.num_spans, index.max_splits)

    inside = parsers.InsideKernel(parsers.LogSumExpReduction()).inside(
        leaves, splits, index
    )
    sweep = parsers.OutsideKernel(parsers.LogSumExpReduction())
    outside = sweep.outside(inside, splits, index)

    assert isinstance(sweep.reduction, parsers.LogWeightReduction)
    assert outside[0, index.span_id(0, 3)].exp().item() == pytest.approx(1.0)
    assert outside[0, index.span_id(0, 2)].exp().item() == pytest.approx(1.0)
    assert outside[0, index.span_id(1, 2)].exp().item() == pytest.approx(2.0)


def test_the_published_surface_reads_a_posterior_over_the_contexts_of_a_span() -> None:
    index = SpanIndex(3)
    leaves = torch.zeros(1, index.n)
    splits = torch.zeros(1, index.num_spans, index.max_splits)

    inside = parsers.InsideKernel(parsers.LogSumExpReduction()).inside(
        leaves, splits, index
    )
    outside = parsers.OutsideKernel(parsers.LogSumExpReduction()).outside(
        inside, splits, index
    )
    posterior = parsers.log_parent_posterior(inside, outside, splits, index)

    assert posterior[0, index.span_id(0, 1)].exp().tolist() == pytest.approx([0.5, 0.5])
    assert posterior[0, index.span_id(0, 3)].exp().tolist() == pytest.approx([0.0, 0.0])


def test_the_published_surface_names_the_parent_behind_a_posterior_column() -> None:
    index = SpanIndex(3)
    contexts = parsers.parent_contexts(index)
    leaf = index.span_id(0, 1)

    assert isinstance(contexts, parsers.ParentContexts)
    assert contexts.parents[leaf] == (index.span_id(0, 2), index.span_id(0, 3))
    assert contexts.siblings[leaf] == (index.span_id(1, 2), index.span_id(1, 3))
    assert contexts.columns[leaf] == (0, 0)
    assert contexts.parents[index.span_id(0, 3)] == ()


def test_the_published_surface_reads_the_marginals_off_the_two_charts() -> None:
    index = SpanIndex(3)
    leaves = torch.zeros(1, index.n)
    splits = torch.zeros(1, index.num_spans, index.max_splits)

    inside = parsers.InsideKernel(parsers.LogSumExpReduction()).inside(
        leaves, splits, index
    )
    outside = parsers.OutsideKernel(parsers.LogSumExpReduction()).outside(
        inside, splits, index
    )
    spans = parsers.log_constituent_marginals(inside, outside, index).exp()
    rules = parsers.log_rule_marginals(inside, outside, splits, index).exp()

    assert spans[0, index.n :].sum().item() == pytest.approx(2.0)
    assert spans[0, index.span_id(0, 2)].item() == pytest.approx(0.5)
    assert rules[0, index.span_id(0, 3)].tolist() == pytest.approx([0.5, 0.5])


def test_the_published_surface_reads_alpha_off_the_gradient_of_the_partition() -> None:
    index = SpanIndex(3)
    leaves = torch.zeros(1, index.n, dtype=torch.float64)
    splits = torch.zeros(1, index.num_spans, index.max_splits, dtype=torch.float64)

    counts = parsers.rule_marginals_by_gradient(leaves, splits, index)
    alpha = parsers.log_split_posterior_by_gradient(leaves, splits, index)

    assert counts[0, index.span_id(0, 3)].tolist() == pytest.approx([0.5, 0.5])
    assert alpha[0, index.span_id(0, 3)].exp().tolist() == pytest.approx([0.5, 0.5])


def test_the_published_surface_reports_the_entropy_of_the_chart() -> None:
    index = SpanIndex(3)
    leaves = torch.zeros(1, index.n, dtype=torch.float64)
    splits = torch.zeros(1, index.num_spans, index.max_splits, dtype=torch.float64)

    inside = parsers.InsideKernel(parsers.LogSumExpReduction()).inside(
        leaves, splits, index
    )
    outside = parsers.OutsideKernel(parsers.LogSumExpReduction()).outside(
        inside, splits, index
    )

    assert parsers.derivation_entropy(
        inside, outside, splits, index
    ).item() == pytest.approx(math.log(2.0))


def test_the_published_surface_decodes_trees_off_the_chart() -> None:
    index = SpanIndex(3)
    leaves = torch.zeros(1, index.n, dtype=torch.float64)
    splits = torch.zeros(1, index.num_spans, index.max_splits, dtype=torch.float64)
    splits[0, index.span_id(0, 3), 0] = 1.0

    inside = parsers.InsideKernel(parsers.LogSumExpReduction()).inside(
        leaves, splits, index
    )
    outside = parsers.OutsideKernel(parsers.LogSumExpReduction()).outside(
        inside, splits, index
    )
    best = parsers.k_best(leaves, splits, index, k=2)
    bracketing = parsers.mbr_trees(inside, outside, splits, index)

    assert isinstance(best[0][0], parsers.BracketedTree)
    assert best[0][0].constituents == ((1, 3), (0, 3))
    assert bracketing[0] == best[0][0]


class Slashes:
    @property
    def forward_slash(self) -> str:
        return "/"

    @property
    def backward_slash(self) -> str:
        return "\\"

    @property
    def result_path(self) -> tuple[int, ...]:
        return (0,)

    @property
    def argument_path(self) -> tuple[int, ...]:
        return (1,)

    @property
    def category_sort(self) -> str:
        """Return the sort the functor constructors build."""
        return "Cat"


STEPS = (
    ("aa-a", "A", "A", "A"),
    ("ab-a", "A", "B", "A"),
    ("ab-b", "A", "B", "B"),
    ("ba-a", "B", "A", "A"),
    ("ba-b", "B", "A", "B"),
    ("bb-b", "B", "B", "B"),
)


def _step_spec(name: str, left: str, right: str, output: str) -> PatternRuleSpec:
    """Declare one binary rule over constant categories."""
    return PatternRuleSpec(
        name=name,
        premises=(ConstantPatternSpec(name=left), ConstantPatternSpec(name=right)),
        conclusion=ConstantPatternSpec(name=output),
    )


PARSER_RULES = declared_rules(
    ccg_signature("X", "A", "B", name="parser-differential"),
    RuleSetSpec(
        name="parser-differential",
        rules=(
            _step_spec("join", "X", "X", "X"),
            *(_step_spec(*step) for step in STEPS),
        ),
    ),
)
PARSER_STORE = PARSER_RULES.context.store
PARSER_MATCHING = cast("TermMatching", PARSER_RULES.rule_set.matching)
COMPILED = {rule.name: cast("PatternRule", rule) for rule in PARSER_RULES.rule_set}
CATEGORY_BY_NAME = {
    name: PARSER_RULES.context.syntax.parse(PARSER_STORE, name)
    for name in ("X", "A", "B")
}


class Join:
    @property
    def name(self) -> str:
        return "join"

    @property
    def arity(self) -> int:
        return 2

    @property
    def directions(self) -> frozenset[Direction]:
        return frozenset({"forward", "check"})

    @property
    def normal_form_role(self) -> NormalFormRole:
        return NormalFormRole()

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
        return PARSER_MATCHING

    @property
    def restriction(self) -> RuleRestriction | None:
        return None

    @property
    def probe(self) -> RuleProbe | None:
        return None

    @property
    def lowering(self) -> TermRuleKernel:
        return COMPILED["join"].lowering

    @property
    def reachability_schemata(self) -> tuple[SchematicSequent, ...]:
        return COMPILED["join"].reachability_schemata

    @property
    def functors(self) -> Slashes:
        return Slashes()

    @property
    def order(self) -> int:
        return 1

    @property
    def slash(self) -> str:
        return "/"

    def apply(self, inputs: tuple[Term, ...], /) -> Iterator[Term]:
        if len(inputs) == 2 and inputs[0] == inputs[1]:
            yield inputs[0]

    def admits(self, inputs: tuple[Term, ...], output: Term, /) -> bool:
        return len(inputs) == 2 and all(item == output for item in inputs)

    def residuate(
        self, output: Term, known: tuple[Slot[Term], ...], /
    ) -> Iterator[Term]:
        del output, known
        return
        yield

    def to_spec(self) -> Mapping[str, JsonValue]:
        return {"rule": "composition", "direction": "unrestricted"}


def tagged_kernel() -> parsers.TaggedInsideKernel[Term]:
    return parsers.TaggedInsideKernel(
        parsers.LogSumExpReduction(),
        RuleSet("toy", (Join(),), PARSER_MATCHING, NoNormalForm[Term]()),
        (CATEGORY_BY_NAME["X"],),
        unary_depth=3,
    )


def flat_channels(index: SpanIndex) -> parsers.ScoreChannels:
    return parsers.ScoreChannels(
        torch.zeros(1, index.n, 1, dtype=torch.float64),
        torch.zeros(1, index.num_spans, index.max_splits, 1, dtype=torch.float64),
    )


def test_the_toy_rule_serializes_its_composition_direction() -> None:
    assert Join().to_spec() == {
        "rule": "composition",
        "direction": "unrestricted",
    }


def test_the_published_surface_refines_the_chart_by_normal_form_tag() -> None:
    index = SpanIndex(3)
    kernel = tagged_kernel()

    chart = kernel.inside(flat_channels(index), index)

    assert chart.axes == ("item", "span", "category", "tag")
    assert not chart.truncated
    assert kernel.log_partition(
        flat_channels(index), index
    ).exp().item() == pytest.approx(2.0)


def test_the_published_surface_reads_the_span_marginals_off_the_counts() -> None:
    index = SpanIndex(3)
    kernel = tagged_kernel()
    channels = flat_channels(index)

    counts = kernel.expected_counts(channels, index)
    marginals = parsers.span_marginals(counts, index)

    assert marginals[0, : index.n].tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert marginals[0, index.span_id(0, 2)].item() == pytest.approx(0.5)


def test_the_published_surface_decodes_derivations_over_the_normal_form() -> None:
    index = SpanIndex(3)
    kernel = tagged_kernel()
    channels = flat_channels(index)
    decoder = parsers.TaggedDecoder(kernel, k=2)

    best = decoder.k_best(channels, index)
    risk = decoder.mbr_trees(channels, index)

    assert isinstance(best[0][0], parsers.TaggedTree)
    assert len(best[0]) == 2
    assert risk[0] is not None
    assert risk[0].constituents() in (((0, 3), (0, 2)), ((0, 3), (1, 3)))


def test_the_published_surface_names_the_unary_closure_a_chart_applies() -> None:
    kernel = tagged_kernel()
    closure = kernel.transitions.closure

    assert isinstance(closure, parsers.UnaryClosure)
    assert closure.retained == ()
    assert not closure.truncated


#: How many random score draws the tensor engines are compared with the symbolic
#: chart over, per input length.
DRAWS = 200

#: The input lengths the engines are compared at.
LENGTHS = (4, 6, 8)

#: How many derivations each engine ranks, and the ranking is compared over.
KEPT = 3

#: Absolute tolerance the engines must agree to, on log weights and probabilities.
TOLERANCE = 1e-5

#: How many of the draws have the partition of each engine differentiated numerically.
DIFFERENTIATED = 4

#: The acceptance fixture compares exactly twenty repository sentences.
FIXTURE_ITEMS = 20

type Shape = tuple["Cell", Term, str, str | None, tuple["Shape", ...]]
"""A derivation's span, category, tag and rule name, nested as its premises are."""


class Step:
    """A binary rule concluding one category from one ordered pair of categories."""

    def __init__(self, name: str, left: str, right: str, output: str, /) -> None:
        self._name = name
        self._inputs = (CATEGORY_BY_NAME[left], CATEGORY_BY_NAME[right])
        self._output = CATEGORY_BY_NAME[output]
        compiled = COMPILED[name]
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
    def normal_form_role(self) -> NormalFormRole:
        return NormalFormRole()

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
        return PARSER_MATCHING

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
    def reachability_schemata(self) -> tuple[SchematicSequent, ...]:
        return self._reachability_schemata

    def apply(self, inputs: tuple[Term, ...], /) -> Iterator[Term]:
        if inputs == self._inputs:
            yield self._output

    def admits(self, inputs: tuple[Term, ...], output: Term, /) -> bool:
        return inputs == self._inputs and output == self._output

    def residuate(
        self, output: Term, known: tuple[Slot[Term], ...], /
    ) -> Iterator[Term]:
        del output, known
        return
        yield

    def to_spec(self) -> Mapping[str, JsonValue]:
        return {"rule": self._name}


CATEGORIES = (CATEGORY_BY_NAME["A"], CATEGORY_BY_NAME["B"])
GRAMMAR = tuple(Step(*step) for step in STEPS)


class Reference[D](NamedTuple):
    """What one engine reads off every draw, with derivations under identity D."""

    partition: torch.Tensor
    rule_marginals: torch.Tensor
    constituents: torch.Tensor
    best: tuple[tuple[D, ...], ...]


class Source:
    """A lexical source admitting every category at every token, from one item."""

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
                    self._leaf[position, column].item(),
                )


class Scorer:
    """An edge scorer reading one item of the rule channel a tensor chart reads."""

    lowering = SpanDependent()

    def __init__(
        self, rules: torch.Tensor, columns: Mapping[str, int], index: SpanIndex, /
    ) -> None:
        self._rules = rules
        self._columns = columns
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
        start, end = span

        if split is None:
            message = f"{rule.name!r} is binary and takes a split"
            raise AssertionError(message)

        return self._rules[
            self._index.span_id(start, end), split - start - 1, self._columns[rule.name]
        ].item()


def drawn(
    index: SpanIndex, categories: int, rules: int, /, *, seed: int
) -> parsers.ScoreChannels:
    generator = torch.Generator().manual_seed(seed)
    leaf = torch.randn(
        DRAWS, index.n, categories, generator=generator, dtype=torch.float64
    )
    steps = torch.randn(
        DRAWS,
        index.num_spans,
        index.max_splits,
        rules,
        generator=generator,
        dtype=torch.float64,
    )

    return parsers.ScoreChannels(leaf, steps)


def ranked(
    forest: PackedForest[Term, float],
    key: ItemKey[Term],
    memo: dict[ItemKey[Term], list[tuple[float, Shape]]],
) -> list[tuple[float, Shape]]:
    """Return an item's heaviest few derivations over the symbolic forest's edges."""
    known = memo.get(key)

    if known is not None:
        return known

    found = []

    for edge in forest.edges(key):
        # an axiom has no split and names no rule
        rule = None if edge.split is None else edge.rule_name

        for below in product(*(ranked(forest, child, memo) for child in edge.children)):
            total = edge.weight + sum(weight for weight, _ in below)
            shape = (
                key.span,
                key.reading,
                key.nf_tag,
                rule,
                tuple(s for _, s in below),
            )
            found.append((total, shape))

    memo[key] = sorted(found, key=lambda pair: -pair[0])[:KEPT]

    return memo[key]


def symbolic_reference[D](
    transitions: parsers.NormalFormTransitions[Term],
    channels: parsers.ScoreChannels,
    index: SpanIndex,
    identity: Callable[[Shape], D],
) -> tuple[Reference[D], torch.Tensor]:
    """Return what the symbolic chart reads off every draw, and its split posterior.

    The rule set, the categories and the rule column order are read from the tensor
    engine's compiled transitions, so both engines address one score tensor alike.
    The split posterior is the one the symbolic marginal scorer reads off the
    forest, laid out as the rule channel is without its rule axis.
    """
    categories = transitions.categories
    columns = {rule.name: column for column, rule in enumerate(transitions.rules)}
    goal = RootGoal(TargetSet(categories, PARSER_MATCHING))
    tokens = tuple(f"w{position}" for position in range(index.n))
    partition = torch.empty(DRAWS, dtype=torch.float64)
    rule_marginals = torch.zeros_like(channels.rule)
    alpha = torch.zeros_like(channels.rule[..., 0])
    constituents = torch.zeros_like(channels.rule[..., 0, 0])
    best = []

    for draw in range(DRAWS):
        forest = parse(
            tokens,
            policy=ParsePolicy(Retain.FOREST, Reachability.ALL),
            rules=transitions.rule_set,
            source=Source(channels.leaf[draw], categories),
            semiring=SymbolicLogInside(),
            scorer=Scorer(channels.rule[draw], columns, index),
            goal=goal,
            unary_depth=3,
        )
        outside = outside_from_edges(forest, goal)
        partition[draw] = outside.partition

        for i, j in index.spans_by_length():
            span = index.span_id(i, j)
            constituents[draw, span] = sum(
                outside.constituent_marginal(key) for key in forest.items_at((i, j))
            )

            for parent in forest.items_at((i, j)):
                for edge, marginal in outside.rule_marginals(parent):
                    if edge.split is None:
                        # an axiom has no split and reads no rule score
                        continue

                    column = columns[edge.rule_name]
                    rule_marginals[draw, span, edge.split - i - 1, column] += marginal

        posterior = MarginalScorer(forest, outside).probabilities

        for (i, j), row in posterior.rows.items():
            for split, probability in row.items():
                alpha[draw, index.span_id(i, j), split - i - 1] = probability

        memo: dict[ItemKey[Term], list[tuple[float, Shape]]] = {}
        roots = [
            pair
            for key in forest.items_at((0, index.n))
            if goal.accepts(forest, key)
            for pair in ranked(forest, key, memo)
        ]
        roots.sort(key=lambda pair: -pair[0])
        best.append(tuple(identity(shape) for _, shape in roots[:KEPT]))

    return (Reference(partition, rule_marginals, constituents, tuple(best)), alpha)


def brackets(shape: Shape) -> frozenset[Cell]:
    """Return the spans a derivation builds from two premises."""
    span, _, _, _, children = shape
    held = frozenset({span}) if len(children) == 2 else frozenset()

    return held.union(*(brackets(child) for child in children))


def shape_of(
    kernel: parsers.TaggedInsideKernel[Term], tree: parsers.TaggedTree
) -> Shape:
    """Return a decoded derivation in the terms the symbolic chart names."""
    transitions = kernel.transitions
    rule = None if tree.rule is None else transitions.rules[tree.rule].name

    return (
        tree.span,
        transitions.categories[tree.category],
        transitions.tags[tree.tag],
        rule,
        tuple(shape_of(kernel, child) for child in tree.children),
    )


def disagreements[D](
    found: tuple[tuple[D, ...], ...], reference: tuple[tuple[D, ...], ...]
) -> list[int]:
    """Return the draws whose rankings differ, by position."""
    return [
        draw
        for draw, (held, expected) in enumerate(zip(found, reference, strict=True))
        if held != expected
    ]


def grammar_kernel() -> parsers.TaggedInsideKernel[Term]:
    return parsers.TaggedInsideKernel(
        parsers.LogSumExpReduction(),
        RuleSet("table", GRAMMAR, PARSER_MATCHING, NoNormalForm[Term]()),
        CATEGORIES,
        unary_depth=3,
    )


@cache
def tagged_oracle(
    n: int,
) -> tuple[parsers.TaggedInsideKernel[Term], parsers.ScoreChannels, Reference[Shape]]:
    """Return the tagged kernel, a draw of its channels and the symbolic reference."""
    index = SpanIndex(n)
    kernel = grammar_kernel()
    channels = drawn(index, len(CATEGORIES), len(GRAMMAR), seed=n)
    reference, _ = symbolic_reference(kernel.transitions, channels, index, lambda s: s)

    return (kernel, channels, reference)


def tagged_readouts(
    kernel: parsers.TaggedInsideKernel[Term],
    channels: parsers.ScoreChannels,
    index: SpanIndex,
) -> Reference[Shape]:
    """Return what the tagged tensor engine reads off every draw of the channels."""
    partition = kernel.log_partition(channels, index)
    counts = kernel.expected_counts(channels, index)
    constituents = kernel.constituent_marginals(channels, index)
    best = parsers.TaggedDecoder(kernel, k=KEPT).k_best(channels, index)

    return Reference(
        partition,
        counts.rule,
        constituents,
        tuple(tuple(shape_of(kernel, tree) for tree in trees) for trees in best),
    )


@cache
def scalar_oracle(
    n: int,
) -> tuple[torch.Tensor, torch.Tensor, Reference[frozenset[Cell]], torch.Tensor]:
    """Return a draw of leaf and split scores, the symbolic reference and its alpha."""
    index = SpanIndex(n)
    channels = drawn(index, 1, 1, seed=n)
    reference, alpha = symbolic_reference(
        tagged_kernel().transitions, channels, index, brackets
    )

    return (channels.leaf[..., 0], channels.rule[..., 0], reference, alpha)


def scalar_readouts(
    leaves: torch.Tensor, splits: torch.Tensor, index: SpanIndex
) -> Reference[frozenset[Cell]]:
    """Return what the scalar tensor engine reads off every draw of the scores."""
    inside = parsers.InsideKernel(parsers.LogSumExpReduction()).inside(
        leaves, splits, index
    )
    outside = parsers.OutsideKernel(parsers.LogSumExpReduction()).outside(
        inside, splits, index
    )
    best = parsers.k_best(leaves, splits, index, k=KEPT)

    return Reference(
        inside[:, index.span_id(0, index.n)],
        parsers.log_rule_marginals(inside, outside, splits, index).exp().unsqueeze(-1),
        parsers.log_constituent_marginals(inside, outside, index).exp(),
        tuple(tuple(frozenset(tree.constituents) for tree in trees) for trees in best),
    )


def test_twenty_repository_sentences_match_symbolic_log_inside() -> None:
    sentences = packaged_pinned_chart().sentences[:FIXTURE_ITEMS]
    lengths = {len(sentence) for sentence in sentences}

    if len(sentences) != FIXTURE_ITEMS or len(lengths) != 1:
        pytest.fail("the twenty-sentence chart fixture no longer has one fixed length")

    n = lengths.pop()
    index = SpanIndex(n)
    leaves = torch.tensor(
        [[math.log1p(len(token)) for token in sentence] for sentence in sentences],
        dtype=torch.float64,
    )
    splits = torch.zeros(
        FIXTURE_ITEMS,
        index.num_spans,
        index.max_splits,
        dtype=torch.float64,
    )

    for item, sentence in enumerate(sentences):
        for i, j in index.spans_by_length(min_length=2):
            span = index.span_id(i, j)

            for column, k in enumerate(index.splits(i, j)):
                splits[item, span, column] = math.log1p(
                    len(sentence[k - 1]) + len(sentence[k])
                )

    transitions = tagged_kernel().transitions
    goal = RootGoal(TargetSet((CATEGORY_BY_NAME["X"],), PARSER_MATCHING))
    symbolic = []

    for item, sentence in enumerate(sentences):
        forest = parse(
            sentence,
            policy=ParsePolicy(Retain.FOREST, Reachability.ALL),
            rules=transitions.rule_set,
            source=Source(leaves[item].unsqueeze(-1), (CATEGORY_BY_NAME["X"],)),
            semiring=SymbolicLogInside(),
            scorer=Scorer(splits[item].unsqueeze(-1), {"join": 0}, index),
            goal=goal,
            unary_depth=3,
        )
        symbolic.append(outside_from_edges(forest, goal).partition)

    neural = parsers.InsideKernel(parsers.LogSumExpReduction()).log_partition(
        leaves, splits, index
    )

    torch.testing.assert_close(
        neural,
        torch.tensor(symbolic, dtype=torch.float64),
        atol=TOLERANCE,
        rtol=0.0,
    )


def leaky_outside(
    inside: torch.Tensor, splits: torch.Tensor, index: SpanIndex
) -> torch.Tensor:
    """Return the outside chart that mixes each parent by its inside split posterior.

    Each child's outside is its parent's outside times the posterior of the split
    that builds the child, so the recursion reuses the inside weight of the very span
    whose outside it is computing.
    """
    alpha = parsers.log_split_posterior(inside, splits, index)
    outside = torch.full_like(inside, -inf)
    outside[:, index.span_id(0, index.n)] = 0.0

    for i, j in index.spans_by_length_desc(min_length=2):
        parent = index.span_id(i, j)

        for column, k in enumerate(index.splits(i, j)):
            weight = outside[:, parent] + alpha[:, parent, column]

            for child in (index.span_id(i, k), index.span_id(k, j)):
                outside[:, child] = torch.logaddexp(outside[:, child], weight)

    return outside


def central_differences(
    partition: Callable[[torch.Tensor], torch.Tensor],
    scores: torch.Tensor,
    directions: torch.Tensor,
) -> torch.Tensor:
    """Return the derivative of a partition along each direction, numerically.

    The step is the cube root of the dtype's epsilon, where the truncation error of a
    central difference and its rounding error balance.
    """
    step = torch.finfo(scores.dtype).eps ** (1 / 3)
    shifted = torch.cat(
        (
            scores.unsqueeze(0) + step * directions,
            scores.unsqueeze(0) - step * directions,
        )
    )
    values = partition(shifted)

    return (values[: len(directions)] - values[len(directions) :]) / (2 * step)


def difference_tolerance(dtype: torch.dtype) -> float:
    """Return the tolerance a central difference of the partition is held to.

    A central difference reaches the two-thirds power of the dtype's epsilon, so the
    square root leaves it room for the growth of the partition with the input length.
    """
    return torch.finfo(dtype).eps ** 0.5


def unit_directions(mask: torch.Tensor) -> torch.Tensor:
    """Return one indicator tensor per true position of a mask, in row-major order."""
    positions = mask.nonzero()
    directions = torch.zeros((len(positions), *mask.shape), dtype=torch.float64)
    directions[(torch.arange(len(positions)), *positions.unbind(1))] = 1.0

    return directions


def span_directions(index: SpanIndex) -> torch.Tensor:
    """Return one indicator per span over every real split of that span."""
    real = torch.tensor(index.split_mask, dtype=torch.float64)
    directions = torch.zeros((index.num_spans, *real.shape), dtype=torch.float64)

    for span in range(index.num_spans):
        directions[span, span] = real[span]

    return directions


@pytest.mark.parametrize("n", LENGTHS)
def test_the_tagged_engine_reads_the_symbolic_partition_off_every_draw(n: int) -> None:
    kernel, channels, reference = tagged_oracle(n)

    found = tagged_readouts(kernel, channels, SpanIndex(n))

    torch.testing.assert_close(
        found.partition, reference.partition, atol=TOLERANCE, rtol=0.0
    )


@pytest.mark.parametrize("n", LENGTHS)
def test_the_tagged_engine_reads_every_symbolic_rule_marginal_off_every_draw(
    n: int,
) -> None:
    kernel, channels, reference = tagged_oracle(n)

    found = tagged_readouts(kernel, channels, SpanIndex(n))

    torch.testing.assert_close(
        found.rule_marginals, reference.rule_marginals, atol=TOLERANCE, rtol=0.0
    )


@pytest.mark.parametrize("n", LENGTHS)
def test_the_tagged_engine_reads_every_symbolic_constituent_marginal_off_every_draw(
    n: int,
) -> None:
    kernel, channels, reference = tagged_oracle(n)

    found = tagged_readouts(kernel, channels, SpanIndex(n))

    torch.testing.assert_close(
        found.constituents, reference.constituents, atol=TOLERANCE, rtol=0.0
    )


@pytest.mark.parametrize("n", LENGTHS)
def test_the_tagged_engine_decodes_the_symbolic_chart_s_best_few_on_every_draw(
    n: int,
) -> None:
    kernel, channels, reference = tagged_oracle(n)

    found = tagged_readouts(kernel, channels, SpanIndex(n))

    assert all(len(trees) == KEPT for trees in reference.best)
    assert disagreements(found.best, reference.best) == []


@pytest.mark.parametrize("n", LENGTHS)
def test_the_scalar_engine_reads_the_symbolic_partition_off_every_draw(n: int) -> None:
    leaves, splits, reference, _ = scalar_oracle(n)

    found = scalar_readouts(leaves, splits, SpanIndex(n))

    torch.testing.assert_close(
        found.partition, reference.partition, atol=TOLERANCE, rtol=0.0
    )


@pytest.mark.parametrize("n", LENGTHS)
def test_the_scalar_engine_reads_every_symbolic_rule_marginal_off_every_draw(
    n: int,
) -> None:
    leaves, splits, reference, _ = scalar_oracle(n)

    found = scalar_readouts(leaves, splits, SpanIndex(n))

    torch.testing.assert_close(
        found.rule_marginals, reference.rule_marginals, atol=TOLERANCE, rtol=0.0
    )


@pytest.mark.parametrize("n", LENGTHS)
def test_the_scalar_engine_reads_every_symbolic_split_posterior_off_every_draw(
    n: int,
) -> None:
    index = SpanIndex(n)
    leaves, splits, _, alpha = scalar_oracle(n)
    inside = parsers.InsideKernel(parsers.LogSumExpReduction()).inside(
        leaves, splits, index
    )

    found = parsers.log_split_posterior(inside, splits, index).exp()

    torch.testing.assert_close(found, alpha, atol=TOLERANCE, rtol=0.0)


@pytest.mark.parametrize("n", LENGTHS)
def test_the_scalar_engine_reads_every_symbolic_constituent_marginal_off_every_draw(
    n: int,
) -> None:
    leaves, splits, reference, _ = scalar_oracle(n)

    found = scalar_readouts(leaves, splits, SpanIndex(n))

    torch.testing.assert_close(
        found.constituents, reference.constituents, atol=TOLERANCE, rtol=0.0
    )


@pytest.mark.parametrize("n", LENGTHS)
def test_the_scalar_engine_decodes_the_symbolic_chart_s_best_few_on_every_draw(
    n: int,
) -> None:
    leaves, splits, reference, _ = scalar_oracle(n)

    found = scalar_readouts(leaves, splits, SpanIndex(n))

    assert all(len(trees) == KEPT for trees in reference.best)
    assert disagreements(found.best, reference.best) == []


@pytest.mark.parametrize("n", LENGTHS)
def test_reusing_the_inside_posterior_in_the_outside_pass_fails_every_draw(
    n: int,
) -> None:
    index = SpanIndex(n)
    leaves, splits, reference, _ = scalar_oracle(n)
    inside = parsers.InsideKernel(parsers.LogSumExpReduction()).inside(
        leaves, splits, index
    )
    root = inside[:, index.span_id(0, index.n)].unsqueeze(-1)

    leaked = (inside + leaky_outside(inside, splits, index) - root).exp()
    gap = (leaked - reference.constituents).abs().amax(dim=1)

    assert bool((gap > TOLERANCE).all())


@pytest.mark.parametrize("n", LENGTHS)
def test_the_tagged_engine_s_expected_counts_are_the_derivatives_of_its_partition(
    n: int,
) -> None:
    index = SpanIndex(n)
    kernel, channels, _ = tagged_oracle(n)
    real = torch.tensor(index.split_mask).unsqueeze(-1).expand_as(channels.rule[0])
    tolerance = difference_tolerance(channels.rule.dtype)

    for draw in range(DIFFERENTIATED):
        leaf = channels.leaf[draw]
        rules = channels.rule[draw]
        counts = kernel.expected_counts(
            parsers.ScoreChannels(leaf.unsqueeze(0), rules.unsqueeze(0)), index
        )
        by_rule = central_differences(
            lambda shifted, leaf=leaf: kernel.log_partition(
                parsers.ScoreChannels(leaf.expand(len(shifted), -1, -1), shifted), index
            ),
            rules,
            unit_directions(real),
        )
        by_leaf = central_differences(
            lambda shifted, rules=rules: kernel.log_partition(
                parsers.ScoreChannels(shifted, rules.expand(len(shifted), -1, -1, -1)),
                index,
            ),
            leaf,
            unit_directions(torch.ones_like(leaf, dtype=torch.bool)),
        )

        torch.testing.assert_close(
            counts.rule[0][real], by_rule, atol=tolerance, rtol=0.0
        )
        torch.testing.assert_close(
            counts.leaf[0].reshape(-1), by_leaf, atol=tolerance, rtol=0.0
        )


@pytest.mark.parametrize("n", LENGTHS)
def test_the_outside_pass_reads_the_derivatives_of_the_scalar_partition(
    n: int,
) -> None:
    index = SpanIndex(n)
    leaves, splits, _, _ = scalar_oracle(n)
    real = torch.tensor(index.split_mask)
    tolerance = difference_tolerance(splits.dtype)
    kernel = parsers.InsideKernel(parsers.LogSumExpReduction())

    for draw in range(DIFFERENTIATED):
        leaf = leaves[draw].unsqueeze(0)
        split = splits[draw].unsqueeze(0)
        inside = kernel.inside(leaf, split, index)
        outside = parsers.OutsideKernel(parsers.LogSumExpReduction()).outside(
            inside, split, index
        )
        rules = parsers.log_rule_marginals(inside, outside, split, index).exp()
        spans = parsers.log_constituent_marginals(inside, outside, index).exp()
        by_gradient = parsers.rule_marginals_by_gradient(leaf, split, index)

        def partition(shifted: torch.Tensor, leaf: torch.Tensor = leaf) -> torch.Tensor:
            return kernel.log_partition(leaf.expand(len(shifted), -1), shifted, index)

        by_rule = central_differences(partition, splits[draw], unit_directions(real))
        by_span = central_differences(partition, splits[draw], span_directions(index))

        torch.testing.assert_close(rules[0][real], by_rule, atol=tolerance, rtol=0.0)
        torch.testing.assert_close(
            by_gradient[0][real], by_rule, atol=tolerance, rtol=0.0
        )
        torch.testing.assert_close(
            spans[0, index.n :], by_span[index.n :], atol=tolerance, rtol=0.0
        )
