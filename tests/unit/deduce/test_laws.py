import math
import operator
from typing import TYPE_CHECKING, override

import pytest
from hypothesis import strategies as st

from lofi._checks import LawViolationError
from lofi._native import PythonCarrier
from lofi.deduce._forest import PackedForest
from lofi.deduce._goals import CoverGoal, RootGoal
from lofi.deduce._pruners import TopKPruner
from lofi.deduce._semiring import (
    Boolean,
    Counting,
    Derivations,
    Expectation,
    Expected,
    Forest,
    KBest,
    LogInside,
    Ranked,
    Truth,
    Viterbi,
)
from lofi.deduce._spans import SpanIndex
from lofi.deduce._values import Edge, ItemKey
from lofi.deduce.laws import (
    CHART_LAWS,
    GOAL_LAWS,
    PRUNER_LAWS,
    SEMIRING_LAWS,
    SPAN_INDEX_LAWS,
    ChartFixture,
    GoalFixture,
    PrunerFixture,
    SemiringFixture,
    SpanIndexFixture,
    a_goal_does_not_write_to_the_chart,
    a_read_is_a_stable_snapshot,
    accepts_only_what_it_enumerates,
    add_packs_on_the_item_key,
    an_absent_item_weighs_zero,
    check_chart_laws,
    check_goal_laws,
    check_pruner_laws,
    check_semiring_laws,
    check_span_index_laws,
    inside_multiplies_in_the_premises,
    keeping_is_idempotent,
    keeping_repeats_itself,
    keeps_a_subsequence_of_the_cell,
    plus_is_a_commutative_monoid,
    zero_annihilates,
)
from lofi.terms import TargetSet

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from lofi.deduce._ports import Chart, Semiring
    from lofi.deduce._spans import Cell
    from lofi.terms import Matching

EXAMPLES = 50

COUNTS = st.integers(min_value=0, max_value=1000)
LOG_WEIGHTS = st.one_of(
    st.just(-math.inf),
    st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False),
)
RANKED: st.SearchStrategy[Ranked] = st.lists(
    st.floats(min_value=-20.0, max_value=20.0), max_size=3
).map(lambda scores: tuple(sorted(scores, reverse=True)))
DERIVATIONS: st.SearchStrategy[Derivations[str]] = st.frozensets(
    st.lists(st.sampled_from("abc"), max_size=2).map(tuple), max_size=3
)
EXPECTED: st.SearchStrategy[Expected] = st.tuples(
    st.floats(min_value=0.0, max_value=1.0),
    st.tuples(
        st.floats(min_value=-1.0, max_value=1.0),
        st.floats(min_value=-1.0, max_value=1.0),
    ),
)


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)


def _close_ranked(a: Ranked, b: Ranked) -> bool:
    return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b, strict=True))


def _close_expected(a: Expected, b: Expected) -> bool:
    return _close(a[0], b[0]) and all(
        _close(x, y) for x, y in zip(a[1], b[1], strict=True)
    )


class _TruthyLogInside:
    """Treats log zero using Python truthiness."""

    __slots__ = ()

    @property
    def name(self) -> str:
        """Return the algebra name."""
        return "truthy-log-inside"

    @property
    def log_probability(self) -> bool:
        """Report whether weights are log probabilities."""
        return True

    @property
    def zero(self) -> float:
        """Return log zero."""
        return LogInside().zero

    @property
    def one(self) -> float:
        """Return log one."""
        return LogInside().one

    @property
    def lowering(self) -> PythonCarrier:
        """Select Python-carried arithmetic."""
        return PythonCarrier()

    def plus(self, a: float, b: float, /) -> float:
        """Add alternatives in log space."""
        return LogInside().plus(a, b)

    def times(self, a: float, b: float, /) -> float:
        """Multiply antecedents in log space."""
        return LogInside().times(a, b)

    def is_zero(self, a: float, /) -> bool:
        """Use the deliberately invalid truthiness test."""
        return not a


class _SubtractingCounting:
    """Subtracts alternative derivation counts."""

    __slots__ = ()

    @property
    def name(self) -> str:
        """Return the algebra name."""
        return "subtracting-counting"

    @property
    def log_probability(self) -> bool:
        """Report whether weights are log probabilities."""
        return False

    @property
    def zero(self) -> int:
        """Return additive zero."""
        return 0

    @property
    def one(self) -> int:
        """Return multiplicative one."""
        return 1

    @property
    def lowering(self) -> PythonCarrier:
        """Select Python-carried arithmetic."""
        return PythonCarrier()

    def plus(self, a: int, b: int, /) -> int:
        """Subtract alternatives to violate the law."""
        return a - b

    def times(self, a: int, b: int, /) -> int:
        """Multiply antecedent counts."""
        return a * b

    def is_zero(self, a: int, /) -> bool:
        """Return whether the count is zero."""
        return a == 0


BOOLEAN = SemiringFixture(Boolean(), st.booleans(), max_examples=EXAMPLES)
LOG_INSIDE = SemiringFixture(LogInside(), LOG_WEIGHTS, _close, EXAMPLES)
VITERBI = SemiringFixture(Viterbi(), LOG_WEIGHTS, _close, EXAMPLES)
COUNTING = SemiringFixture(Counting(), COUNTS, max_examples=EXAMPLES)
K_BEST = SemiringFixture(KBest(3), RANKED, _close_ranked, EXAMPLES)
FOREST = SemiringFixture(Forest[str](), DERIVATIONS, max_examples=EXAMPLES)
EXPECTATION = SemiringFixture(Expectation(2), EXPECTED, _close_expected, EXAMPLES)
SPANS = SpanIndexFixture(max_n=16, max_examples=25)


@pytest.mark.parametrize("law", SEMIRING_LAWS, ids=lambda law: law.__name__)
def test_each_semiring_law_holds_of_boolean(
    law: Callable[[SemiringFixture[Truth]], None],
) -> None:
    law(BOOLEAN)


@pytest.mark.parametrize("law", SEMIRING_LAWS, ids=lambda law: law.__name__)
def test_each_semiring_law_holds_of_log_inside(
    law: Callable[[SemiringFixture[float]], None],
) -> None:
    law(LOG_INSIDE)


@pytest.mark.parametrize("law", SEMIRING_LAWS, ids=lambda law: law.__name__)
def test_each_semiring_law_holds_of_viterbi(
    law: Callable[[SemiringFixture[float]], None],
) -> None:
    law(VITERBI)


@pytest.mark.parametrize("law", SEMIRING_LAWS, ids=lambda law: law.__name__)
def test_each_semiring_law_holds_of_counting(
    law: Callable[[SemiringFixture[int]], None],
) -> None:
    law(COUNTING)


@pytest.mark.parametrize("law", SEMIRING_LAWS, ids=lambda law: law.__name__)
def test_each_semiring_law_holds_of_k_best(
    law: Callable[[SemiringFixture[Ranked]], None],
) -> None:
    law(K_BEST)


@pytest.mark.parametrize("law", SEMIRING_LAWS, ids=lambda law: law.__name__)
def test_each_semiring_law_holds_of_the_forest(
    law: Callable[[SemiringFixture[Derivations[str]]], None],
) -> None:
    law(FOREST)


@pytest.mark.parametrize("law", SEMIRING_LAWS, ids=lambda law: law.__name__)
def test_each_semiring_law_holds_of_expectation(
    law: Callable[[SemiringFixture[Expected]], None],
) -> None:
    law(EXPECTATION)


@pytest.mark.parametrize("law", SPAN_INDEX_LAWS, ids=lambda law: law.__name__)
def test_each_span_index_law_holds_of_the_index(
    law: Callable[[SpanIndexFixture], None],
) -> None:
    law(SPANS)


def test_zero_is_not_falsiness() -> None:
    semiring = LogInside()

    assert semiring.is_zero(semiring.one) is False
    assert semiring.is_zero(semiring.zero) is True

    with pytest.raises(LawViolationError, match="annihilate"):
        zero_annihilates(SemiringFixture(_TruthyLogInside(), LOG_WEIGHTS, _close, 50))


def test_the_semiring_suite_runs_every_law() -> None:
    check_semiring_laws(SemiringFixture(Counting(), COUNTS, max_examples=25))
    check_semiring_laws(SemiringFixture(LogInside(), LOG_WEIGHTS, _close, 25))


def test_the_suite_catches_a_plus_that_is_not_a_commutative_monoid() -> None:
    with pytest.raises(LawViolationError, match="plus is not"):
        plus_is_a_commutative_monoid(
            SemiringFixture(_SubtractingCounting(), COUNTS, max_examples=50)
        )


def test_the_suite_catches_a_weight_wider_than_its_algebra() -> None:
    with pytest.raises(LawViolationError, match="identity for plus"):
        plus_is_a_commutative_monoid(
            SemiringFixture(KBest(1), RANKED, _close_ranked, EXAMPLES)
        )


def test_the_span_suite_runs_every_law() -> None:
    check_span_index_laws(SpanIndexFixture(max_n=12, max_examples=10))


class _Identity:
    """Compares category strings directly."""

    @property
    def name(self) -> str:
        """Return the relation's name."""
        return "identity"

    @property
    def has_key(self) -> bool:
        """Return whether every class has a representative."""
        return True

    def key(self, t: str, /) -> str:
        """Return the category itself, which is its own representative."""
        return t

    def equivalent(self, a: str, b: str, /) -> bool:
        """Return whether two categories are the same string."""
        return a == b


IDENTITY: Matching[str] = _Identity()
HALF = math.log(0.5)


class _FirstOnly(PackedForest[str, float]):
    """Keeps only the first edge for each item."""

    __slots__ = ()

    @override
    def add(self, key: ItemKey[str], edge: Edge[str, float], /) -> None:
        if self.edges(key):
            return

        super().add(key, edge)


class _Local(PackedForest[str, float]):
    """Ignores premise weights during insertion."""

    __slots__ = ()

    @override
    def add(self, key: ItemKey[str], edge: Edge[str, float], /) -> None:
        super().add(
            key,
            Edge(edge.rule_name, (), edge.output, edge.weight, edge.split, edge.order),
        )


class _OneItemPerSpan(PackedForest[str, float]):
    """Packs items by span only."""

    __slots__ = ()

    @override
    def add(self, key: ItemKey[str], edge: Edge[str, float], /) -> None:
        held = self.items_at(key.span)

        super().add(held[0] if held else key, edge)


class _Neutral(PackedForest[str, float]):
    """Uses one as the missing-item weight."""

    __slots__ = ()

    @override
    def inside(self, key: ItemKey[str], /) -> float:
        return super().inside(key) if self.edges(key) else self.semiring.one


class _Generous(RootGoal[str]):
    """Accepts items that are not enumerated as roots."""

    __slots__ = ()

    @override
    def accepts[W](self, chart: Chart[str, W], key: ItemKey[str], /) -> bool:
        return True


class _Recording(RootGoal[str]):
    """Mutates the chart while reading roots."""

    __slots__ = ()

    @override
    def roots[W](self, chart: Chart[str, W], /) -> Iterator[ItemKey[str]]:
        seen = ItemKey((0, chart.index.n), "SEEN", "goal")
        chart.add(seen, Edge("goal", (), "SEEN", chart.inside(seen)))

        yield from super().roots(chart)


def _chart(n: int, spans: Sequence[Cell]) -> PackedForest[str, float]:
    forest = PackedForest[str, float](SpanIndex(n), LogInside())

    for span in spans:
        forest.add(ItemKey(span, "NP", "lex"), Edge("lex", (), "NP", HALF))

    return forest


def _chart_fixture[W](
    semiring: Semiring[W],
    weights: st.SearchStrategy[W],
    equal: Callable[[W, W], bool] = operator.eq,
) -> ChartFixture[str, W]:
    return ChartFixture(
        lambda: PackedForest[str, W](SpanIndex(3), semiring),
        semiring,
        (0, 2),
        "NP",
        "N",
        weights,
        equal,
        25,
    )


BOOLEAN_CHART = _chart_fixture(Boolean(), st.booleans())
LOG_INSIDE_CHART = _chart_fixture(LogInside(), LOG_WEIGHTS, _close)
VITERBI_CHART = _chart_fixture(Viterbi(), LOG_WEIGHTS, _close)
COUNTING_CHART = _chart_fixture(Counting(), COUNTS)
K_BEST_CHART = _chart_fixture(KBest(3), RANKED, _close_ranked)
FOREST_CHART = _chart_fixture(Forest[str](), DERIVATIONS)
EXPECTATION_CHART = _chart_fixture(Expectation(2), EXPECTED, _close_expected)

SPANNED = _chart(2, [(0, 1), (1, 2), (0, 2)])
UNSPANNED = _chart(3, [(0, 1), (1, 2), (2, 3)])
BARE = _chart(2, [])

ROOT_GOAL = GoalFixture(
    RootGoal(TargetSet(("NP",), IDENTITY)),
    (SPANNED, UNSPANNED, BARE),
)
COVER_GOAL = GoalFixture(CoverGoal(3, penalty=1.0), (SPANNED, UNSPANNED, BARE))


@pytest.mark.parametrize("law", CHART_LAWS, ids=lambda law: law.__name__)
def test_each_chart_law_holds_of_the_packed_forest(
    law: Callable[[ChartFixture[str, float]], None],
) -> None:
    law(LOG_INSIDE_CHART)


def test_the_chart_suite_runs_every_law_under_every_algebra() -> None:
    check_chart_laws(BOOLEAN_CHART)
    check_chart_laws(LOG_INSIDE_CHART)
    check_chart_laws(VITERBI_CHART)
    check_chart_laws(COUNTING_CHART)
    check_chart_laws(K_BEST_CHART)
    check_chart_laws(FOREST_CHART)
    check_chart_laws(EXPECTATION_CHART)


@pytest.mark.parametrize("law", GOAL_LAWS, ids=lambda law: law.__name__)
def test_each_goal_law_holds_of_the_root_goal(
    law: Callable[[GoalFixture[str, float]], None],
) -> None:
    law(ROOT_GOAL)


@pytest.mark.parametrize("law", GOAL_LAWS, ids=lambda law: law.__name__)
def test_each_goal_law_holds_of_the_cover_goal(
    law: Callable[[GoalFixture[str, float]], None],
) -> None:
    law(COVER_GOAL)


def test_the_goal_suite_runs_every_law() -> None:
    check_goal_laws(ROOT_GOAL)
    check_goal_laws(COVER_GOAL)


def test_the_chart_suite_catches_a_forest_that_drops_a_second_analysis() -> None:
    fixture = LOG_INSIDE_CHART._replace(
        make=lambda: _FirstOnly(SpanIndex(3), LogInside())
    )

    with pytest.raises(LawViolationError, match="edges"):
        add_packs_on_the_item_key(fixture)


def test_the_chart_suite_catches_a_forest_that_ignores_its_premises() -> None:
    fixture = LOG_INSIDE_CHART._replace(make=lambda: _Local(SpanIndex(3), LogInside()))

    add_packs_on_the_item_key(fixture)

    with pytest.raises(LawViolationError, match="folded wrong"):
        inside_multiplies_in_the_premises(fixture)


def test_the_chart_suite_catches_a_forest_keyed_by_span_alone() -> None:
    fixture = LOG_INSIDE_CHART._replace(
        make=lambda: _OneItemPerSpan(SpanIndex(3), LogInside())
    )

    with pytest.raises(LawViolationError, match="second item"):
        a_read_is_a_stable_snapshot(fixture)


def test_the_chart_suite_catches_a_forest_weighing_an_absent_item_at_one() -> None:
    fixture = LOG_INSIDE_CHART._replace(
        make=lambda: _Neutral(SpanIndex(3), LogInside())
    )

    with pytest.raises(LawViolationError, match="does not hold weighs"):
        an_absent_item_weighs_zero(fixture)


def test_the_goal_suite_catches_a_goal_accepting_what_it_does_not_enumerate() -> None:
    generous = GoalFixture(
        _Generous(TargetSet(("NP",), IDENTITY)),
        (SPANNED, UNSPANNED),
    )

    with pytest.raises(LawViolationError, match="differently"):
        accepts_only_what_it_enumerates(generous)


def test_the_goal_suite_catches_a_goal_writing_to_the_chart_it_reads() -> None:
    recording = GoalFixture(
        _Recording(TargetSet(("NP",), IDENTITY)),
        (_chart(2, [(0, 2)]),),
    )

    with pytest.raises(LawViolationError, match="changed what the chart holds"):
        a_goal_does_not_write_to_the_chart(recording)


def test_the_goal_suite_refuses_a_fixture_offering_no_chart() -> None:
    with pytest.raises(LawViolationError, match="no chart was offered"):
        check_goal_laws(GoalFixture(RootGoal(TargetSet(("NP",), IDENTITY)), ()))


def _pruner_cell(
    span: Cell, *weights: float
) -> tuple[Cell, tuple[tuple[ItemKey[str], float], ...]]:
    return (
        span,
        tuple(
            (ItemKey(span, f"X{at}", "step"), weight)
            for at, weight in enumerate(weights)
        ),
    )


PRUNER_CELLS = (
    _pruner_cell((0, 2), math.log(0.1), math.log(0.9), math.log(0.5)),
    _pruner_cell((0, 1), math.log(0.5)),
    _pruner_cell((1, 3), math.log(0.5), math.log(0.5), -math.inf),
    _pruner_cell((2, 3)),
)
BEAM = PrunerFixture(TopKPruner[str](2), PRUNER_CELLS)


class _Inventive:
    """Keeps an item no cell ever offered."""

    def keep(
        self, span: Cell, items: Sequence[tuple[ItemKey[str], float]], /
    ) -> tuple[ItemKey[str], ...]:
        del items

        return (ItemKey(span, "invented", "step"),)


class _Reordering:
    """Keeps every item, heaviest first."""

    def keep(
        self, span: Cell, items: Sequence[tuple[ItemKey[str], float]], /
    ) -> tuple[ItemKey[str], ...]:
        del span

        return tuple(key for key, _ in sorted(items, key=lambda pair: -pair[1]))


class _Restless:
    """Keeps one fewer item on each read of a cell."""

    def __init__(self) -> None:
        self._reads = 0

    def keep(
        self, span: Cell, items: Sequence[tuple[ItemKey[str], float]], /
    ) -> tuple[ItemKey[str], ...]:
        del span
        self._reads += 1

        return tuple(key for key, _ in items[: max(len(items) - self._reads, 0)])


class _Thinning:
    """Drops the lightest item of every cell it reads."""

    def keep(
        self, span: Cell, items: Sequence[tuple[ItemKey[str], float]], /
    ) -> tuple[ItemKey[str], ...]:
        del span

        if not items:
            return ()

        lightest = min(range(len(items)), key=lambda at: items[at][1])

        return tuple(key for at, (key, _) in enumerate(items) if at != lightest)


@pytest.mark.parametrize("law", PRUNER_LAWS, ids=lambda law: law.__name__)
def test_each_pruner_law_holds_of_the_top_k_beam(
    law: Callable[[PrunerFixture[str, float]], None],
) -> None:
    law(BEAM)


def test_the_pruner_suite_runs_every_law() -> None:
    check_pruner_laws(BEAM)
    check_pruner_laws(PrunerFixture(TopKPruner[str](1), PRUNER_CELLS))


def test_the_pruner_suite_catches_a_pruner_keeping_what_it_was_not_offered() -> None:
    with pytest.raises(LawViolationError, match="not a subsequence"):
        keeps_a_subsequence_of_the_cell(BEAM._replace(pruner=_Inventive()))


def test_the_pruner_suite_catches_a_pruner_that_reorders_the_cell() -> None:
    with pytest.raises(LawViolationError, match="not a subsequence"):
        keeps_a_subsequence_of_the_cell(BEAM._replace(pruner=_Reordering()))


def test_the_pruner_suite_catches_a_pruner_that_answers_differently_twice() -> None:
    with pytest.raises(LawViolationError, match="on a second read"):
        keeping_repeats_itself(BEAM._replace(pruner=_Restless()))


def test_the_pruner_suite_catches_a_pruner_that_keeps_thinning_a_cell() -> None:
    with pytest.raises(LawViolationError, match="dropped an item"):
        keeping_is_idempotent(BEAM._replace(pruner=_Thinning()))


def test_the_pruner_suite_refuses_a_fixture_offering_no_cell() -> None:
    with pytest.raises(LawViolationError, match="no cell was offered"):
        check_pruner_laws(PrunerFixture(TopKPruner[str](2), ()))
