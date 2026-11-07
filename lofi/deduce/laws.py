"""Deduction-component conformance laws."""

import operator
from typing import TYPE_CHECKING, Final, NamedTuple, NoReturn

from hypothesis import given, settings
from hypothesis import strategies as st

from lofi._checks import LawViolationError
from lofi.deduce._spans import SpanIndex
from lofi.deduce._values import Edge, ItemKey

if TYPE_CHECKING:
    from collections.abc import Callable

    from lofi.deduce._ports import CellPruner, Chart, Goal, Semiring
    from lofi.deduce._spans import Cell

__all__ = [
    "CHART_LAWS",
    "GOAL_LAWS",
    "PRUNER_LAWS",
    "SEMIRING_LAWS",
    "SPAN_INDEX_LAWS",
    "ChartFixture",
    "GoalFixture",
    "PrunerFixture",
    "SemiringFixture",
    "SpanIndexFixture",
    "a_goal_does_not_write_to_the_chart",
    "a_read_is_a_stable_snapshot",
    "accepts_only_what_it_enumerates",
    "add_packs_on_the_item_key",
    "an_absent_item_weighs_zero",
    "check_chart_laws",
    "check_goal_laws",
    "check_pruner_laws",
    "check_semiring_laws",
    "check_span_index_laws",
    "gather_vectors_agree_with_splits",
    "inside_multiplies_in_the_premises",
    "keeping_is_idempotent",
    "keeping_repeats_itself",
    "keeps_a_subsequence_of_the_cell",
    "plus_is_a_commutative_monoid",
    "roots_are_items_the_chart_holds",
    "span_id_is_a_bijection",
    "span_round_trip",
    "times_distributes_over_plus",
    "times_is_a_monoid",
    "zero_annihilates",
]


class SemiringFixture[W](NamedTuple):
    """Inputs for semiring-law checks.

    Attributes
    ----------
    semiring : Semiring[W]
        The algebra under test.
    values : st.SearchStrategy[W]
        A strategy drawing weights of the algebra.
    equal : Callable[[W, W], bool]
        Weight equality, exact by default.
    max_examples : int
        The example budget each law gives Hypothesis.
    """

    semiring: Semiring[W]
    values: st.SearchStrategy[W]
    equal: Callable[[W, W], bool] = operator.eq
    max_examples: int = 200


class SpanIndexFixture(NamedTuple):
    """Inputs for span-index law checks.

    Attributes
    ----------
    max_n : int
        The longest input length to draw.
    max_examples : int
        The example budget each law gives Hypothesis.
    """

    max_n: int = 24
    max_examples: int = 50


class ChartFixture[T, W](NamedTuple):
    """Inputs for chart-law checks.

    Attributes
    ----------
    make : Callable[[], Chart[T, W]]
        Factory for an empty chart.
    semiring : Semiring[W]
        The algebra the chart under test accumulates in.
    span : Cell
        A span of that input the laws record at, at least two tokens wide.
    term : T
        A conclusion the laws record.
    other : T
        A second conclusion, which the chart keys apart from the first.
    weights : st.SearchStrategy[W]
        A strategy drawing weights of the algebra.
    equal : Callable[[W, W], bool]
        Weight equality, exact by default.
    max_examples : int
        The example budget each law gives Hypothesis.
    """

    make: Callable[[], Chart[T, W]]
    semiring: Semiring[W]
    span: Cell
    term: T
    other: T
    weights: st.SearchStrategy[W]
    equal: Callable[[W, W], bool] = operator.eq
    max_examples: int = 50


class GoalFixture[T, W](NamedTuple):
    """Inputs for goal-law checks.

    Attributes
    ----------
    goal : Goal[T]
        The goal under test.
    charts : tuple[Chart[T, W], ...]
        Nonempty chart cases, including one without accepted roots.
    """

    goal: Goal[T]
    charts: tuple[Chart[T, W], ...]


class PrunerFixture[T, W](NamedTuple):
    """Inputs for cell-pruner law checks.

    Attributes
    ----------
    pruner : CellPruner[T, W]
        The pruner under test.
    cells : tuple[tuple[Cell, tuple[tuple[ItemKey[T], W], ...]], ...]
        Nonempty cell cases, each a span and the items of that span with their
        inside weights, in arrival order.
    """

    pruner: CellPruner[T, W]
    cells: tuple[tuple[Cell, tuple[tuple[ItemKey[T], W], ...]], ...]


def _fail(detail: str) -> NoReturn:
    """Raise a law violation.

    Parameters
    ----------
    detail : str
        Failure description and witness.

    Raises
    ------
    LawViolationError
        Always.
    """
    raise LawViolationError(detail)


def plus_is_a_commutative_monoid[W](fixture: SemiringFixture[W]) -> None:
    """Check that plus is commutative, associative, and has zero as its identity.

    Parameters
    ----------
    fixture : SemiringFixture[W]
        The algebra under test and its strategy.

    Raises
    ------
    LawViolationError
        If a drawn case violates the law.
    """
    semiring = fixture.semiring
    equal = fixture.equal

    @settings(max_examples=fixture.max_examples)
    @given(fixture.values, fixture.values, fixture.values)
    def law(a: W, b: W, c: W) -> None:
        if not equal(semiring.plus(a, b), semiring.plus(b, a)):
            _fail(f"plus is not commutative on {a!r} and {b!r}")

        left = semiring.plus(semiring.plus(a, b), c)
        right = semiring.plus(a, semiring.plus(b, c))

        if not equal(left, right):
            _fail(f"plus is not associative on {a!r}, {b!r} and {c!r}")

        if not equal(semiring.plus(a, semiring.zero), a):
            _fail(f"zero is not a right identity for plus on {a!r}")

        if not equal(semiring.plus(semiring.zero, a), a):
            _fail(f"zero is not a left identity for plus on {a!r}")

    law()


def times_is_a_monoid[W](fixture: SemiringFixture[W]) -> None:
    """Check that times is associative and has one as its identity.

    Parameters
    ----------
    fixture : SemiringFixture[W]
        The algebra under test and its strategy.

    Raises
    ------
    LawViolationError
        If a drawn case violates the law.
    """
    semiring = fixture.semiring
    equal = fixture.equal

    @settings(max_examples=fixture.max_examples)
    @given(fixture.values, fixture.values, fixture.values)
    def law(a: W, b: W, c: W) -> None:
        left = semiring.times(semiring.times(a, b), c)
        right = semiring.times(a, semiring.times(b, c))

        if not equal(left, right):
            _fail(f"times is not associative on {a!r}, {b!r} and {c!r}")

        if not equal(semiring.times(a, semiring.one), a):
            _fail(f"one is not a right identity for times on {a!r}")

        if not equal(semiring.times(semiring.one, a), a):
            _fail(f"one is not a left identity for times on {a!r}")

    law()


def times_distributes_over_plus[W](fixture: SemiringFixture[W]) -> None:
    """Check that times distributes over plus from the left and from the right.

    Parameters
    ----------
    fixture : SemiringFixture[W]
        The algebra under test and its strategy.

    Raises
    ------
    LawViolationError
        If a drawn case violates the law.
    """
    semiring = fixture.semiring
    equal = fixture.equal

    @settings(max_examples=fixture.max_examples)
    @given(fixture.values, fixture.values, fixture.values)
    def law(a: W, b: W, c: W) -> None:
        from_left = semiring.times(a, semiring.plus(b, c))
        expanded_left = semiring.plus(semiring.times(a, b), semiring.times(a, c))

        if not equal(from_left, expanded_left):
            _fail(f"times does not distribute from the left on {a!r}")

        from_right = semiring.times(semiring.plus(b, c), a)
        expanded_right = semiring.plus(semiring.times(b, a), semiring.times(c, a))

        if not equal(from_right, expanded_right):
            _fail(f"times does not distribute from the right on {a!r}")

    law()


def zero_annihilates[W](fixture: SemiringFixture[W]) -> None:
    """Check that multiplying by zero yields zero, on either side.

    Parameters
    ----------
    fixture : SemiringFixture[W]
        The algebra under test and its strategy.

    Raises
    ------
    LawViolationError
        If a drawn case violates the law.
    """
    semiring = fixture.semiring

    @settings(max_examples=fixture.max_examples)
    @given(fixture.values)
    def law(a: W) -> None:
        if not semiring.is_zero(semiring.times(semiring.zero, a)):
            _fail(f"zero does not annihilate from the left on {a!r}")

        if not semiring.is_zero(semiring.times(a, semiring.zero)):
            _fail(f"zero does not annihilate from the right on {a!r}")

    law()


def span_round_trip(fixture: SpanIndexFixture) -> None:
    """Check that a span survives the round trip through its dense id.

    Parameters
    ----------
    fixture : SpanIndexFixture
        The input lengths to draw over.

    Raises
    ------
    LawViolationError
        If a span does not come back from its own id.
    """

    @settings(max_examples=fixture.max_examples)
    @given(st.integers(min_value=1, max_value=fixture.max_n))
    def law(n: int) -> None:
        index = SpanIndex(n)

        for span in index.spans_by_length():
            recovered = index.span_of(index.span_id(*span))

            if recovered != span:
                _fail(f"span {span} came back from its id as {recovered}")

    law()


def span_id_is_a_bijection(fixture: SpanIndexFixture) -> None:
    """Check that the span ids are exactly the span range, each used once.

    Parameters
    ----------
    fixture : SpanIndexFixture
        The input lengths to draw over.

    Raises
    ------
    LawViolationError
        If the ids are not a bijection onto the span range.
    """

    @settings(max_examples=fixture.max_examples)
    @given(st.integers(min_value=1, max_value=fixture.max_n))
    def law(n: int) -> None:
        index = SpanIndex(n)
        ids = [index.span_id(i, j) for i, j in index.spans_by_length()]

        if len(ids) != index.num_spans:
            _fail(f"{len(ids)} spans were enumerated for {index.num_spans} ids")

        if set(ids) != set(range(index.num_spans)):
            _fail(f"the span ids at n = {n} are not the whole span range")

    law()


def gather_vectors_agree_with_splits(fixture: SpanIndexFixture) -> None:
    """Check that the gather vectors address exactly the cells a sweep iterates.

    Parameters
    ----------
    fixture : SpanIndexFixture
        The input lengths to draw over.

    Raises
    ------
    LawViolationError
        If a gather row disagrees with the split enumeration.
    """

    @settings(max_examples=fixture.max_examples)
    @given(st.integers(min_value=1, max_value=fixture.max_n))
    def law(n: int) -> None:
        index = SpanIndex(n)
        left_rows = index.left_child_ids
        right_rows = index.right_child_ids
        masks = index.split_mask

        for i, j in index.spans_by_length():
            sid = index.span_id(i, j)
            mask = masks[sid]
            splits = list(index.splits(i, j))
            # store real split IDs before padding.
            live = (True,) * len(splits)
            expected_mask = live + (False,) * (index.max_splits - len(splits))

            if mask != expected_mask:
                _fail(f"the split mask of span {(i, j)} is {mask}")

            left = [c for c, keep in zip(left_rows[sid], mask, strict=True) if keep]
            right = [c for c, keep in zip(right_rows[sid], mask, strict=True) if keep]

            if left != [index.span_id(i, k) for k in splits]:
                _fail(f"the left children of span {(i, j)} are {left}")

            if right != [index.span_id(k, j) for k in splits]:
                _fail(f"the right children of span {(i, j)} are {right}")

    law()


def add_packs_on_the_item_key[T, W](fixture: ChartFixture[T, W]) -> None:
    """Check that many steps under one key leave one item and every step.

    Parameters
    ----------
    fixture : ChartFixture[T, W]
        The chart under test and its strategy.

    Raises
    ------
    LawViolationError
        If a drawn case violates the law.
    """
    semiring = fixture.semiring
    key = ItemKey(fixture.span, fixture.term)

    @settings(max_examples=fixture.max_examples)
    @given(st.lists(fixture.weights, min_size=1, max_size=8))
    def law(weights: list[W]) -> None:
        chart = fixture.make()

        for weight in weights:
            chart.add(key, Edge("axiom", (), fixture.term, weight))

        items = chart.items_at(fixture.span)

        if items != (key,):
            _fail(f"{len(weights)} steps under one key left the items {items!r}")

        recorded = chart.edges(key)

        if len(recorded) != len(weights):
            _fail(f"{len(weights)} steps under one key left {len(recorded)} edges")

        total = semiring.zero

        for weight in weights:
            total = semiring.plus(total, weight)

        if not fixture.equal(chart.inside(key), total):
            _fail(f"the inside weight of an item reached {len(weights)} ways is wrong")

    law()


def inside_multiplies_in_the_premises[T, W](fixture: ChartFixture[T, W]) -> None:
    """Check that a step contributes its own weight times its premises'.

    Parameters
    ----------
    fixture : ChartFixture[T, W]
        The chart under test and its strategy.

    Raises
    ------
    LawViolationError
        If a drawn case violates the law.
    """
    semiring = fixture.semiring
    start, end = fixture.span
    left = ItemKey((start, start + 1), fixture.term)
    right = ItemKey((start + 1, end), fixture.other)
    parent = ItemKey(fixture.span, fixture.term)

    @settings(max_examples=fixture.max_examples)
    @given(fixture.weights, fixture.weights, fixture.weights)
    def law(a: W, b: W, c: W) -> None:
        chart = fixture.make()
        chart.add(left, Edge("axiom", (), fixture.term, a))
        chart.add(right, Edge("axiom", (), fixture.other, b))
        chart.add(
            parent,
            Edge("step", (left, right), fixture.term, c, split=start + 1),
        )

        want = semiring.times(semiring.times(c, a), b)

        if not fixture.equal(chart.inside(parent), want):
            _fail(f"a step over premises {a!r} and {b!r} weighing {c!r} folded wrong")

    law()


def a_read_is_a_stable_snapshot[T, W](fixture: ChartFixture[T, W]) -> None:
    """Check that a cell is not mutated under a caller reading it.

    Parameters
    ----------
    fixture : ChartFixture[T, W]
        The chart under test and its strategy.

    Raises
    ------
    LawViolationError
        If a drawn case violates the law.
    """
    key = ItemKey(fixture.span, fixture.term)
    second = ItemKey(fixture.span, fixture.other)

    @settings(max_examples=fixture.max_examples)
    @given(fixture.weights, fixture.weights)
    def law(a: W, b: W) -> None:
        chart = fixture.make()
        chart.add(key, Edge("axiom", (), fixture.term, a))

        items = chart.items_at(fixture.span)
        edges = chart.edges(key)

        chart.add(second, Edge("axiom", (), fixture.other, b))
        chart.add(key, Edge("step", (), fixture.term, b))

        if items != (key,):
            _fail(f"an items-at result became {items!r} under a later insertion")

        if len(edges) != 1:
            _fail(f"an edges result grew to {len(edges)} under a later insertion")

        if len(chart.items_at(fixture.span)) != len(items) + 1:
            _fail("a second item at one span did not reach the chart")

        if len(chart.edges(key)) != len(edges) + 1:
            _fail("a second step under one item did not reach the chart")

    law()


def an_absent_item_weighs_zero[T, W](fixture: ChartFixture[T, W]) -> None:
    """Check that an item the chart does not hold weighs zero and has no steps.

    Parameters
    ----------
    fixture : ChartFixture[T, W]
        The chart under test and its strategy.

    Raises
    ------
    LawViolationError
        If the chart answers for an item it does not hold.
    """
    chart = fixture.make()
    absent = ItemKey(fixture.span, fixture.other, "absent")

    if not fixture.semiring.is_zero(chart.inside(absent)):
        _fail(f"an item the chart does not hold weighs {chart.inside(absent)!r}")

    if chart.edges(absent) != ():
        _fail("an item the chart does not hold has edges")

    if chart.items_at(fixture.span) != ():
        _fail("a chart holding nothing named items at a span")

    held = True

    try:
        chart.term(absent)
    except KeyError:
        held = False

    if held:
        _fail("the chart named a term for an item it does not hold")


def _items_of[T, W](chart: Chart[T, W]) -> tuple[ItemKey[T], ...]:
    """Return every chart item, narrowest span first.

    Parameters
    ----------
    chart : Chart[T, W]
        The chart to read.

    Returns
    -------
    tuple[ItemKey[T], ...]
        The items, in span order and then in the order they arrived.
    """
    return tuple(
        key for span in chart.index.spans_by_length() for key in chart.items_at(span)
    )


def _census[T, W](chart: Chart[T, W]) -> tuple[tuple[int, int], ...]:
    """Return each chart item's width and edge count.

    Parameters
    ----------
    chart : Chart[T, W]
        The chart to read.

    Returns
    -------
    tuple[tuple[int, int], ...]
        Span width and edge count for each item.
    """
    return tuple(
        (key.span[1] - key.span[0], len(chart.edges(key))) for key in _items_of(chart)
    )


def roots_are_items_the_chart_holds[T, W](fixture: GoalFixture[T, W]) -> None:
    """Check that every goal root exists in the chart.

    Parameters
    ----------
    fixture : GoalFixture[T, W]
        The goal under test and the charts to read it over.

    Raises
    ------
    LawViolationError
        If a root is not an item of the chart it was drawn from.
    """
    if not fixture.charts:
        _fail("no chart was offered, so nothing was checked")

    for chart in fixture.charts:
        roots = tuple(fixture.goal.roots(chart))
        grouped = tuple(
            key for group in fixture.goal.root_groups(chart) for key in group
        )

        if grouped != roots:
            _fail("the goal's root groups do not flatten to its roots")

        for key in roots:
            if key not in chart.items_at(key.span):
                _fail(f"the goal ended at {key!r}, which the chart does not hold")


def accepts_only_what_it_enumerates[T, W](fixture: GoalFixture[T, W]) -> None:
    """Check that acceptance and enumeration agree, item for item.

    Parameters
    ----------
    fixture : GoalFixture[T, W]
        The goal under test and the charts to read it over.

    Raises
    ------
    LawViolationError
        If an item is accepted and not enumerated, or the reverse.
    """
    for chart in fixture.charts:
        roots = tuple(fixture.goal.roots(chart))

        for key in _items_of(chart):
            if fixture.goal.accepts(chart, key) != (key in roots):
                _fail(f"the goal accepts and enumerates {key!r} differently")


def a_goal_does_not_write_to_the_chart[T, W](fixture: GoalFixture[T, W]) -> None:
    """Check that reading a goal leaves the chart as it found it.

    Parameters
    ----------
    fixture : GoalFixture[T, W]
        The goal under test and the charts to read it over.

    Raises
    ------
    LawViolationError
        If reading the goal changes what a chart holds.
    """
    for chart in fixture.charts:
        before = _census(chart)
        groups = fixture.goal.root_groups(chart)
        fixture.goal.combination_weight(len(groups))

        for group in groups:
            for root in group:
                fixture.goal.root_weight(root)

        for key in tuple(fixture.goal.roots(chart)):
            fixture.goal.accepts(chart, key)

        for key in _items_of(chart):
            fixture.goal.accepts(chart, key)

            for edge in chart.edges(key):
                fixture.goal.admits_edge(chart, key, edge)

        if _census(chart) != before:
            _fail("reading the goal changed what the chart holds")


def keeps_a_subsequence_of_the_cell[T, W](fixture: PrunerFixture[T, W]) -> None:
    """Check that a pruner returns offered items, once each, in arrival order.

    Parameters
    ----------
    fixture : PrunerFixture[T, W]
        The pruner under test and the cells to run it over.

    Raises
    ------
    LawViolationError
        If a kept item was not offered, repeats, or arrives out of order.
    """
    if not fixture.cells:
        _fail("no cell was offered, so nothing was checked")

    for span, items in fixture.cells:
        kept = tuple(fixture.pruner.keep(span, items))
        offered = tuple(key for key, _ in items)
        remaining = iter(offered)

        if len(set(kept)) != len(kept):
            _fail(f"the pruner kept an item of {span!r} twice")

        if not all(key in remaining for key in kept):
            _fail(f"the items kept at {span!r} are not a subsequence of the cell")


def keeping_repeats_itself[T, W](fixture: PrunerFixture[T, W]) -> None:
    """Check that a pruner reading one cell twice keeps the same items.

    Parameters
    ----------
    fixture : PrunerFixture[T, W]
        The pruner under test and the cells to run it over.

    Raises
    ------
    LawViolationError
        If two reads of one cell disagree.
    """
    for span, items in fixture.cells:
        first = tuple(fixture.pruner.keep(span, items))
        second = tuple(fixture.pruner.keep(span, items))

        if first != second:
            _fail(f"the pruner kept different items of {span!r} on a second read")


def keeping_is_idempotent[T, W](fixture: PrunerFixture[T, W]) -> None:
    """Check that pruning a pruned cell drops nothing further.

    Parameters
    ----------
    fixture : PrunerFixture[T, W]
        The pruner under test and the cells to run it over.

    Raises
    ------
    LawViolationError
        If a second pass over what the first kept drops an item.
    """
    for span, items in fixture.cells:
        kept = tuple(fixture.pruner.keep(span, items))
        weights = dict(items)
        again = tuple(
            fixture.pruner.keep(span, tuple((key, weights[key]) for key in kept))
        )

        if again != kept:
            _fail(f"pruning what the pruner kept at {span!r} dropped an item")


SEMIRING_LAWS: Final = (
    plus_is_a_commutative_monoid,
    times_is_a_monoid,
    times_distributes_over_plus,
    zero_annihilates,
)
"""Semiring laws in conformance order."""

SPAN_INDEX_LAWS: Final = (
    span_round_trip,
    span_id_is_a_bijection,
    gather_vectors_agree_with_splits,
)
"""Span-index laws in conformance order."""

CHART_LAWS: Final = (
    add_packs_on_the_item_key,
    inside_multiplies_in_the_premises,
    a_read_is_a_stable_snapshot,
    an_absent_item_weighs_zero,
)
"""Chart laws in conformance order."""

GOAL_LAWS: Final = (
    roots_are_items_the_chart_holds,
    accepts_only_what_it_enumerates,
    a_goal_does_not_write_to_the_chart,
)
"""Goal laws in conformance order."""

PRUNER_LAWS: Final = (
    keeps_a_subsequence_of_the_cell,
    keeping_repeats_itself,
    keeping_is_idempotent,
)
"""Cell-pruner laws in conformance order."""


def check_semiring_laws[W](fixture: SemiringFixture[W]) -> None:
    """Run every semiring law against one algebra.

    Parameters
    ----------
    fixture : SemiringFixture[W]
        The algebra under test and its strategy.

    Raises
    ------
    LawViolationError
        On the first law the algebra fails.
    """
    for law in SEMIRING_LAWS:
        law(fixture)


def check_span_index_laws(fixture: SpanIndexFixture) -> None:
    """Run every span-index law.

    Parameters
    ----------
    fixture : SpanIndexFixture
        The input lengths to draw over.

    Raises
    ------
    LawViolationError
        On the first law the index fails.
    """
    for law in SPAN_INDEX_LAWS:
        law(fixture)


def check_chart_laws[T, W](fixture: ChartFixture[T, W]) -> None:
    """Run every chart law against one packed forest.

    Parameters
    ----------
    fixture : ChartFixture[T, W]
        The chart under test and its strategy.

    Raises
    ------
    LawViolationError
        On the first law the chart fails.
    """
    for law in CHART_LAWS:
        law(fixture)


def check_goal_laws[T, W](fixture: GoalFixture[T, W]) -> None:
    """Run every goal law against one goal.

    Parameters
    ----------
    fixture : GoalFixture[T, W]
        The goal under test and the charts to read it over.

    Raises
    ------
    LawViolationError
        On the first law the goal fails.
    """
    for law in GOAL_LAWS:
        law(fixture)


def check_pruner_laws[T, W](fixture: PrunerFixture[T, W]) -> None:
    """Run every cell-pruner law against one pruner.

    Parameters
    ----------
    fixture : PrunerFixture[T, W]
        The pruner under test and the cells to run it over.

    Raises
    ------
    LawViolationError
        On the first law the pruner fails.
    """
    for law in PRUNER_LAWS:
        law(fixture)
