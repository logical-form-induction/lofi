"""Benchmarks the symbolic chart engine over the pinned sentence fixture."""

from typing import TYPE_CHECKING, Final, NamedTuple

import pytest

from lofi.deduce import ParsePolicy, Reachability, Retain, SpanFree, Viterbi, parse
from lofi.judge import (
    EisnerNormalForm,
    HockenmaierBiskNormalForm,
    NoNormalForm,
    RuleIndex,
    RuleSet,
)

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture
    from tests.benchmarks.conftest import PinnedChart, WallClockBudget

    from lofi.deduce import Cell, PackedForest
    from lofi.judge import NormalForm, Rule
    from lofi.lexicon import Entry
    from lofi.terms import Signature, Term

type Category = Term
type Forest = PackedForest[Category, float]


class OperatorHeads:
    """Head codes of category terms, read off the operator name at each root.

    Parameters
    ----------
    signature : Signature
        The signature whose operator codes are the head codes.
    """

    __slots__ = ("_by_name", "_codes")

    def __init__(self, signature: Signature) -> None:
        self._by_name = {
            operator.name: signature.code(operator) for operator in signature.operators
        }
        self._codes = frozenset(self._by_name.values())

    @property
    def codes(self) -> frozenset[int]:
        """Return every operator code."""
        return self._codes

    def term(self, term: Category, /) -> int:
        """Return the code of a term's root operator."""
        return self._by_name[term.op.name]

    def named(self, name: str, /) -> int:
        """Return the code of a named operator."""
        return self._by_name[name]


class Unweighted:
    """Scores every axiom and every edge at log probability zero."""

    __slots__ = ()
    lowering = SpanFree()

    def axiom_weight(self, position: int, entry: Entry[Category], /) -> float:
        """Return zero."""
        del position, entry

        return 0.0

    def edge_weight(
        self,
        rule: Rule[Category],
        inputs: tuple[Category, ...],
        output: Category,
        span: Cell,
        split: int | None,
        /,
    ) -> float:
        """Return zero."""
        del rule, inputs, output, span, split

        return 0.0


UNWEIGHTED: Final = Unweighted()


class Charts(NamedTuple):
    """The pinned fixture with an indexed copy of its rule set.

    Attributes
    ----------
    pinned
        The pinned sentences and the grammar they parse under.
    heads
        The head codes the index dispatches on.
    index
        The binary-rule index over the grammar's rules.
    indexed
        The grammar's rule set with the index attached.
    """

    pinned: PinnedChart
    heads: OperatorHeads
    index: RuleIndex[Category]
    indexed: RuleSet[Category]


def _with(
    pinned: PinnedChart,
    index: RuleIndex[Category],
    normal_form: NormalForm[Category],
) -> RuleSet[Category]:
    """Return the grammar's rules under an index and a normal form."""
    rules = pinned.grammar.rules

    return RuleSet(rules.name, rules.rules, rules.matching, normal_form, index=index)


def _without_index(pinned: PinnedChart) -> RuleSet[Category]:
    """Return the grammar's rules without the compiled dispatch index."""
    rules = pinned.grammar.rules

    return RuleSet(
        rules.name,
        rules.rules,
        rules.matching,
        rules.normal_form,
        rules.restriction_mode,
    )


@pytest.fixture(scope="module")
def charts(pinned_chart: PinnedChart) -> Charts:
    """Return the pinned fixture with its rules indexed by head code."""
    heads = OperatorHeads(pinned_chart.grammar.signature)
    index = RuleIndex(pinned_chart.grammar.rules.rules, heads)

    return Charts(
        pinned_chart, heads, index, _with(pinned_chart, index, NoNormalForm())
    )


def _parse_all(pinned: PinnedChart, rules: RuleSet[Category]) -> tuple[Forest, ...]:
    """Parse every pinned sentence under a rule set."""
    return tuple(
        parse(
            sentence,
            rules=rules,
            source=pinned.lexicon,
            semiring=Viterbi(),
            scorer=UNWEIGHTED,
            goal=pinned.grammar.goal,
            unary_depth=pinned.grammar.budget.unary_rounds,
            policy=ParsePolicy(Retain.FOREST, Reachability.ALL),
        )
        for sentence in pinned.sentences
    )


def _all_rooted(pinned: PinnedChart, forests: tuple[Forest, ...]) -> bool:
    """Report whether every forest derives the grammar's goal."""
    goal = pinned.grammar.goal

    return len(forests) == len(pinned.sentences) and all(
        tuple(goal.roots(forest)) for forest in forests
    )


def _working_point(charts: Charts) -> tuple[int, int, int]:
    """Return the fixture's token count, the grammar's beam and its rule count."""
    (n,) = charts.pinned.pinned.lengths

    return n, charts.pinned.grammar.budget.beam, len(charts.indexed)


class Trials(NamedTuple):
    """Rule trials counted over the binary splits of a set of forests.

    Attributes
    ----------
    pairs
        Premise pairs, one per left item and right item at a split.
    distinct
        Premise pairs counted once per forest and pair of exact categories,
        which is how often the sweep draws a pair's steps.
    indexed
        Rule trials with the index selecting each pair's candidates.
    unindexed
        Rule trials with every binary rule tried on every pair.
    items
        Chart items over every cell.
    cells
        Cells over every forest.
    """

    pairs: int
    distinct: int
    indexed: int
    unindexed: int
    items: int
    cells: int


def _trials(
    forests: tuple[Forest, ...],
    index: RuleIndex[Category],
    binary_rules: int,
) -> Trials:
    """Count the rule trials the engine's binary sweep makes over the forests."""
    pairs = 0
    distinct = 0
    indexed = 0
    items = 0
    cells = 0

    for forest in forests:
        drawn: set[tuple[Category, Category]] = set()

        for span in forest.index.regions_by_size():
            items += len(forest.items_at(span))
            cells += 1

        for _, left, right in forest.index.binary_sweep():
            lefts = tuple(forest.term(key) for key in forest.items_at(left))
            rights = tuple(forest.term(key) for key in forest.items_at(right))
            pairs += len(lefts) * len(rights)
            drawn.update((one, other) for one in lefts for other in rights)
            indexed += sum(
                len(index.candidates(one, other)) for one in lefts for other in rights
            )

        distinct += len(drawn)

    return Trials(pairs, distinct, indexed, pairs * binary_rules, items, cells)


def _multiplication(forests: tuple[Forest, ...]) -> float:
    """Return chart items per distinct span and reading over the forests."""
    items = sum(forest.num_items for forest in forests)
    packed = sum(
        len(
            {
                (key.span, key.reading)
                for span in forest.index.regions_by_size()
                for key in forest.items_at(span)
            }
        )
        for forest in forests
    )

    return items / packed


def test_the_fixture_is_at_the_working_point_the_budgets_are_stated_at(
    charts: Charts,
) -> None:
    n, beam, rule_count = _working_point(charts)

    assert (n, beam, rule_count) == (20, 8, 20)
    assert charts.pinned.pinned.count == len(charts.pinned.sentences) == 200
    assert charts.pinned.grammar.fingerprint == charts.pinned.pinned.fingerprint
    assert all(len(sentence) == n for sentence in charts.pinned.sentences)


def test_the_indexed_parse_roots_every_pinned_sentence(charts: Charts) -> None:
    forests = _parse_all(charts.pinned, charts.indexed)

    assert _all_rooted(charts.pinned, forests)


def test_the_pinned_sentences_parse_within_the_chart_budgets(
    benchmark: BenchmarkFixture,
    harness_floor: float,
    charts: Charts,
    wall_clock_budget: WallClockBudget,
) -> None:
    forests = benchmark(_parse_all, charts.pinned, charts.indexed)
    measured = benchmark.stats

    if measured is None:
        pytest.fail("the sweep ran but pytest-benchmark timed nothing")

    n, beam, rule_count = _working_point(charts)
    sentences = len(forests)
    seconds = (measured.stats.median - harness_floor) / sentences
    split_points = sum(1 for _ in forests[0].index.binary_sweep())
    # Every vocabulary atom has the category-constant operator head, so the
    # conservative head index admits every binary rule for each premise pair.
    design_trials = split_points * beam * beam * rule_count
    counted = _trials(forests, charts.index, rule_count)
    pinned = charts.pinned.pinned
    benchmark.extra_info["grammar"] = pinned.grammar
    benchmark.extra_info["seed"] = pinned.seed
    benchmark.extra_info["sentences"] = pinned.count
    benchmark.extra_info["lengths"] = dict(pinned.lengths)
    benchmark.extra_info["tokens"] = n
    benchmark.extra_info["beam"] = beam
    benchmark.extra_info["rules"] = rule_count
    benchmark.extra_info["ms_per_sentence"] = seconds * 1e3
    benchmark.extra_info["mean_ms_per_sentence"] = (
        (measured.stats.mean - harness_floor) / sentences * 1e3
    )
    benchmark.extra_info["sentences_per_second"] = 1 / seconds
    benchmark.extra_info["split_points"] = split_points
    benchmark.extra_info["split_points_per_second"] = split_points / seconds
    benchmark.extra_info["design_trials_per_sentence"] = design_trials
    benchmark.extra_info["design_trials_per_second"] = design_trials / seconds
    benchmark.extra_info["trials_per_sentence"] = counted.indexed / sentences
    benchmark.extra_info["trials_per_second"] = counted.indexed / sentences / seconds
    benchmark.extra_info["mean_items_per_cell"] = counted.items / counted.cells
    benchmark.extra_info["budget"] = "chart-construction"

    # enforce the 80 ms per-sentence budget
    if wall_clock_budget.enforced:
        assert seconds * 1e3 <= 80

        # enforce the 12 sentences per second, 16,000 split points per second and
        # 3.1 million trials per second that the wall time derives
        assert 1 / seconds >= 12
        assert split_points / seconds >= 16_000
        assert design_trials / seconds >= 3.1e6


def test_the_constant_category_index_preserves_forests_conservatively(
    charts: Charts,
) -> None:
    plain = _without_index(charts.pinned)
    forests = _parse_all(charts.pinned, plain)
    _, _, rule_count = _working_point(charts)
    counted = _trials(forests, charts.index, rule_count)
    indexed = _parse_all(charts.pinned, charts.indexed)

    assert plain.index is None
    assert _all_rooted(charts.pinned, forests)
    assert [forest.num_items for forest in indexed] == [
        forest.num_items for forest in forests
    ]
    assert [forest.num_edges for forest in indexed] == [
        forest.num_edges for forest in forests
    ]
    assert counted.pairs > 0
    assert counted.indexed == counted.unindexed
    assert counted.indexed / counted.pairs == rule_count


def test_the_constant_category_index_records_its_full_candidate_workload(
    benchmark: BenchmarkFixture, harness_floor: float, charts: Charts
) -> None:
    plain = _without_index(charts.pinned)
    forests = benchmark.pedantic(
        _parse_all, args=(charts.pinned, plain), rounds=1, iterations=1
    )
    measured = benchmark.stats

    if measured is None:
        pytest.fail("the sweep ran but pytest-benchmark timed nothing")

    _, _, rule_count = _working_point(charts)
    counted = _trials(forests, charts.index, rule_count)

    sentences = len(forests)
    benchmark.extra_info["ms_per_sentence_unindexed"] = (
        (measured.stats.median - harness_floor) / sentences * 1e3
    )
    benchmark.extra_info["pairs_per_sentence"] = counted.pairs / sentences
    benchmark.extra_info["indexed_trials_per_sentence"] = counted.indexed / sentences
    benchmark.extra_info["unindexed_trials_per_sentence"] = (
        counted.unindexed / sentences
    )
    benchmark.extra_info["mean_candidates"] = counted.indexed / counted.pairs
    benchmark.extra_info["mean_items_per_cell"] = counted.items / counted.cells


@pytest.mark.parametrize(
    ("normal_form", "bound"),
    [
        (NoNormalForm(), 1),
        (EisnerNormalForm(), 4),
        (HockenmaierBiskNormalForm(), 10),
    ],
    ids=["none", "eisner", "hockenmaier-bisk"],
)
def test_normal_form_item_multiplication_respects_its_alphabet_and_bound(
    charts: Charts,
    normal_form: NormalForm[Category],
    bound: int,
) -> None:
    rules = _with(charts.pinned, charts.index, normal_form)

    forests = _parse_all(charts.pinned, rules)
    multiplication = _multiplication(forests)

    assert _all_rooted(charts.pinned, forests)
    assert 1 <= multiplication <= len(normal_form.alphabet)
    assert multiplication <= bound


@pytest.mark.parametrize(
    ("normal_form", "bound"),
    [
        (NoNormalForm(), 1),
        (EisnerNormalForm(), 4),
        # order four: the ten-tag alphabet the tenfold bound covers
        (HockenmaierBiskNormalForm(), 10),
    ],
    ids=["none", "eisner", "hockenmaier-bisk"],
)
def test_normal_form_item_multiplication_is_within_its_bound(
    benchmark: BenchmarkFixture,
    harness_floor: float,
    charts: Charts,
    normal_form: NormalForm[Category],
    bound: int,
) -> None:
    rules = _with(charts.pinned, charts.index, normal_form)
    forests = benchmark.pedantic(
        _parse_all, args=(charts.pinned, rules), rounds=1, iterations=1
    )
    measured = benchmark.stats

    if measured is None:
        pytest.fail("the sweep ran but pytest-benchmark timed nothing")

    multiplication = _multiplication(forests)
    benchmark.extra_info["normal_form"] = normal_form.name
    benchmark.extra_info["alphabet"] = len(normal_form.alphabet)
    benchmark.extra_info["items"] = sum(forest.num_items for forest in forests)
    benchmark.extra_info["item_multiplication"] = multiplication
    benchmark.extra_info["item_multiplication_bound"] = bound
    benchmark.extra_info["ms_per_sentence"] = (
        (measured.stats.median - harness_floor) / len(forests) * 1e3
    )
