import copy
import pickle
from math import log
from typing import TYPE_CHECKING, Final

import pytest

from lofi.deduce._forest import PackedForest
from lofi.deduce._goals import CoverGoal, PenaltyForm, RootGoal, Tiling
from lofi.deduce._ports import Goal
from lofi.deduce._semiring import LogInside
from lofi.deduce._spans import SpanIndex
from lofi.deduce._values import Edge, ItemKey
from lofi.terms import TargetSet

if TYPE_CHECKING:
    from lofi.deduce._spans import Cell
    from lofi.terms import Matching


class Identity:
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


class Erasure:
    """Compares categories after removing bracketed features."""

    @property
    def name(self) -> str:
        """Return the relation's name."""
        return "feature-erasure"

    @property
    def has_key(self) -> bool:
        """Return whether every class has a representative."""
        return True

    def key(self, t: str, /) -> str:
        """Return the category with any bracketed feature removed."""
        head, bracket, _ = t.partition("[")

        return head if bracket else t

    def equivalent(self, a: str, b: str, /) -> bool:
        """Return whether two categories agree once features are dropped."""
        return self.key(a) == self.key(b)


IDENTITY: Final[Matching[str]] = Identity()
ERASURE: Final[Matching[str]] = Erasure()

DET = ItemKey((0, 1), "NP/N", "lex")
NOUN = ItemKey((1, 2), "N", "lex")
PHRASE = ItemKey((0, 2), "NP", "step")


def spanning() -> PackedForest[str, float]:
    forest = PackedForest[str, float](SpanIndex(2), LogInside())
    forest.add(DET, Edge("lex", (), "NP/N", log(0.5)))
    forest.add(NOUN, Edge("lex", (), "N", log(1.0)))
    forest.add(PHRASE, Edge("fa", (DET, NOUN), "NP", log(0.5), split=1))

    return forest


def unspanned() -> PackedForest[str, float]:
    forest = PackedForest[str, float](SpanIndex(3), LogInside())

    for i, category in enumerate(("NP", "S\\NP", "PP")):
        forest.add(
            ItemKey((i, i + 1), category, "lex"),
            Edge("lex", (), category, log(0.5)),
        )

    forest.add(
        ItemKey((1, 3), "S\\NP", "step"),
        Edge(
            "fa",
            (ItemKey((1, 2), "S\\NP", "lex"), ItemKey((2, 3), "PP", "lex")),
            "S\\NP",
            log(0.5),
            split=2,
        ),
    )

    return forest


def tied() -> PackedForest[str, float]:
    forest = PackedForest[str, float](SpanIndex(3), LogInside())

    for span in ((0, 1), (1, 2), (2, 3), (0, 2), (1, 3)):
        forest.add(ItemKey(span, "X", "lex"), Edge("lex", (), "X", log(0.5)))

    return forest


def roots_through[W](goal: Goal[str], chart: PackedForest[str, W]) -> tuple[str, ...]:
    return tuple(key.reading for key in goal.roots(chart))


def spans_through[W](goal: Goal[str], chart: PackedForest[str, W]) -> tuple[Cell, ...]:
    return tuple(key.span for key in goal.roots(chart))


def test_a_root_goal_accepts_one_item_spanning_the_input() -> None:
    goal = RootGoal(TargetSet(("NP",), IDENTITY))
    chart = spanning()

    assert isinstance(goal, Goal)
    assert goal.accepts(chart, PHRASE)
    assert tuple(goal.roots(chart)) == (PHRASE,)


def test_a_root_goal_refuses_an_item_that_does_not_span_the_input() -> None:
    goal = RootGoal(TargetSet(("NP",), IDENTITY))
    chart = spanning()

    assert not goal.accepts(chart, DET)
    assert not goal.accepts(chart, ItemKey((1, 2), "NP", "lex"))


def test_a_root_goal_refuses_a_category_it_was_not_given() -> None:
    chart = spanning()

    assert tuple(RootGoal(TargetSet(("S",), IDENTITY)).roots(chart)) == ()
    assert not RootGoal(TargetSet((), IDENTITY)).accepts(chart, PHRASE)


def test_a_root_goal_reads_the_designated_category_under_the_matching() -> None:
    forest = PackedForest[str, float](SpanIndex(1), LogInside())
    featured = ItemKey((0, 1), "S[dcl]", "lex")
    forest.add(featured, Edge("lex", (), "S[dcl]", log(0.5)))

    assert RootGoal(TargetSet(("S",), ERASURE)).accepts(forest, featured)
    assert not RootGoal(TargetSet(("S",), IDENTITY)).accepts(forest, featured)


def test_a_root_goal_over_a_chart_with_no_spanning_item_ends_nowhere() -> None:
    assert tuple(RootGoal(TargetSet(("S",), IDENTITY)).roots(unspanned())) == ()


def test_a_root_goal_exposes_its_target_set() -> None:
    targets = TargetSet(["S", "NP"], IDENTITY)
    goal = RootGoal(targets)

    assert goal.targets is targets
    assert repr(goal) == (
        "RootGoal(targets=TargetSet(categories=('S', 'NP'), matching='identity'), "
        "weights=(0.0, 0.0))"
    )


def test_a_root_goal_returns_the_declared_weight_of_each_matching_root() -> None:
    targets = TargetSet(("S", "NP"), IDENTITY)
    goal = RootGoal(targets, (-0.25, -1.5))

    assert goal.root_weight(ItemKey((0, 2), "NP", "root")) == -1.5
    assert goal.combination_weight(1) == 0.0

    with pytest.raises(ValueError, match="one weight per target"):
        RootGoal(targets, (0.0,))


@pytest.mark.parametrize("field", ["_targets"])
def test_a_root_goal_refuses_private_slot_mutation_and_deletion(field: str) -> None:
    """Freeze root-goal fields after construction."""
    goal = RootGoal(TargetSet(("NP",), IDENTITY))

    with pytest.raises(AttributeError, match="immutable"):
        goal.__setattr__(field, ())

    with pytest.raises(AttributeError, match="immutable"):
        goal.__delattr__(field)


def test_a_root_goal_reconstructs_through_standard_copy_protocols() -> None:
    """Reconstruct a root goal without changing its behavior."""
    goal = RootGoal(TargetSet(("NP",), IDENTITY))
    shallow = copy.copy(goal)
    deep = copy.deepcopy(goal)
    restored = pickle.loads(pickle.dumps(goal))

    for rebuilt in (shallow, deep, restored):
        assert rebuilt.targets.categories == goal.targets.categories
        assert rebuilt.targets.matching_name == goal.targets.matching_name
        assert tuple(rebuilt.roots(spanning())) == (PHRASE,)

    assert shallow.targets is goal.targets
    assert deep.targets is goal.targets


def test_a_cover_goal_tiles_an_input_no_single_item_spans() -> None:
    goal = CoverGoal(3)
    chart = unspanned()

    assert isinstance(goal, Goal)
    assert goal.fragments(chart) == ((0, 1), (1, 3))
    assert roots_through(goal, chart) == ("NP", "S\\NP")


def test_a_cover_goal_covers_a_spanning_input_in_one_fragment() -> None:
    goal = CoverGoal(4)
    chart = spanning()

    assert goal.fragments(chart) == ((0, 2),)
    assert tuple(goal.roots(chart)) == (PHRASE,)
    assert not goal.accepts(chart, DET)


def test_a_cover_goal_takes_the_longest_leading_fragment_of_a_tie() -> None:
    goal = CoverGoal(3)
    chart = tied()

    assert goal.fragments(chart) == ((0, 2), (2, 3))
    assert goal.accepts(chart, ItemKey((0, 2), "X", "lex"))
    assert not goal.accepts(chart, ItemKey((0, 1), "X", "lex"))


def test_a_cover_goal_ends_nowhere_when_no_tiling_fits_the_bound() -> None:
    chart = unspanned()

    assert CoverGoal(1).fragments(chart) == ()
    assert tuple(CoverGoal(1).roots(chart)) == ()
    assert CoverGoal(2).fragments(chart) == ((0, 1), (1, 3))


def test_a_cover_goal_ends_nowhere_when_a_token_carries_nothing() -> None:
    forest = PackedForest[str, float](SpanIndex(2), LogInside())
    forest.add(DET, Edge("lex", (), "NP/N", log(0.5)))

    assert CoverGoal(2).fragments(forest) == ()


def test_a_cover_goal_over_an_empty_input_covers_nothing() -> None:
    forest = PackedForest[str, float](SpanIndex(0), LogInside())

    assert CoverGoal(2).fragments(forest) == ()
    assert tuple(CoverGoal(2).roots(forest)) == ()


def test_a_cover_goal_ends_at_every_item_of_a_fragment() -> None:
    chart = unspanned()
    second = ItemKey((1, 3), "NP", "step")
    chart.add(
        second,
        Edge(
            "ba",
            (ItemKey((1, 2), "S\\NP", "lex"), ItemKey((2, 3), "PP", "lex")),
            "NP",
            log(0.25),
            split=2,
        ),
    )
    goal = CoverGoal(3)

    assert roots_through(goal, chart) == ("NP", "S\\NP", "NP")
    assert goal.accepts(chart, second)


def test_a_cover_goal_charges_one_penalty_for_each_fragment_after_the_first() -> None:
    goal = CoverGoal(4, penalty=1.5)

    assert goal.penalty == 1.5
    assert goal.max_fragments == 4
    assert goal.penalty_for(0) == 0.0
    assert goal.penalty_for(1) == 0.0
    assert goal.penalty_for(3) == -3.0
    assert CoverGoal(4).penalty_for(3) == 0.0
    assert goal.combination_weight(3) == -3.0
    assert goal.root_weight(PHRASE) == 0.0
    assert repr(goal) == (
        "CoverGoal(max_fragments=4, penalty=1.5, "
        "penalty_form=<PenaltyForm.AFTER_FIRST: 'after-first'>, "
        "tiling=<Tiling.FEWEST: 'fewest-fragments'>)"
    )


def test_a_cover_goal_declares_its_penalty_form_and_tiling_policy() -> None:
    chart = unspanned()
    lexical = CoverGoal(
        3,
        penalty=0.5,
        penalty_form=PenaltyForm.EACH,
        tiling=Tiling.LEXICAL,
    )

    assert lexical.fragments(chart) == ((0, 1), (1, 2), (2, 3))
    assert lexical.penalty_for(3) == -1.5


@pytest.mark.parametrize("field", ["_max_fragments", "_penalty"])
def test_a_cover_goal_refuses_private_slot_mutation_and_deletion(field: str) -> None:
    """Freeze cover-goal fields after construction."""
    goal = CoverGoal(3, penalty=1.5)

    with pytest.raises(AttributeError, match="immutable"):
        goal.__setattr__(field, 0)

    with pytest.raises(AttributeError, match="immutable"):
        goal.__delattr__(field)


def test_a_cover_goal_reconstructs_through_standard_copy_protocols() -> None:
    """Reconstruct a cover goal without changing its behavior."""
    goal = CoverGoal(3, penalty=1.5)

    for rebuilt in (
        copy.copy(goal),
        copy.deepcopy(goal),
        pickle.loads(pickle.dumps(goal)),
    ):
        assert rebuilt.max_fragments == goal.max_fragments
        assert rebuilt.penalty == goal.penalty
        assert rebuilt.fragments(unspanned()) == ((0, 1), (1, 3))
        assert rebuilt.combination_weight(2) == -1.5


def test_a_cover_goal_refuses_a_bound_below_one_and_a_negative_penalty() -> None:
    with pytest.raises(ValueError, match="at least one"):
        CoverGoal(0)

    with pytest.raises(ValueError, match="negative"):
        CoverGoal(-1)

    with pytest.raises(ValueError, match="non-negative"):
        CoverGoal(2, penalty=-1.0)

    with pytest.raises(ValueError, match="not finite"):
        CoverGoal(2, penalty=float("inf"))


def test_the_two_goals_answer_the_same_two_calls_over_one_chart() -> None:
    chart = unspanned()
    root: Goal[str] = RootGoal(TargetSet(("S",), IDENTITY))
    cover: Goal[str] = CoverGoal(3)

    assert spans_through(root, chart) == ()
    assert spans_through(cover, chart) == ((0, 1), (1, 3))

    for goal in (root, cover):
        seeds = tuple(goal.roots(chart))

        assert all(goal.accepts(chart, key) for key in seeds)
        assert seeds == tuple(goal.roots(chart))


def test_root_and_cover_goals_admit_every_forest_edge() -> None:
    chart = spanning()
    edge = chart.edges(PHRASE)[0]

    assert RootGoal(TargetSet(("NP",), IDENTITY)).admits_edge(chart, PHRASE, edge)
    assert CoverGoal(2).admits_edge(chart, PHRASE, edge)


def test_goals_group_alternative_roots_by_required_component() -> None:
    chart = unspanned()
    root = RootGoal(TargetSet(("S",), IDENTITY))
    cover = CoverGoal(2, penalty=1.5)

    assert root.root_groups(chart) == ()
    assert root.combination_weight(0) == 0.0
    assert cover.root_groups(chart) == (
        chart.items_at((0, 1)),
        chart.items_at((1, 3)),
    )
    assert cover.combination_weight(2) == -1.5
