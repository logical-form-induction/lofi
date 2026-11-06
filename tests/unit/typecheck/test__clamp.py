import copy
import pickle
from math import exp, isinf, log
from typing import TYPE_CHECKING, Final, cast, override

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from tests.language import language

from lofi.deduce import (
    Cell,
    CoverGoal,
    Edge,
    Goal,
    ItemKey,
    LogInside,
    PackedForest,
    RootGoal,
    SpanIndex,
    outside_from_edges,
)
from lofi.deduce.laws import GoalFixture, check_goal_laws
from lofi.lexicon import Clamp, ClampMode
from lofi.terms import (
    FeatureErasure,
    Identity,
    Operator,
    Signature,
    TargetSet,
    Term,
    TermStore,
)
from lofi.typecheck import (
    CategoryProjection,
    ClampedGoal,
    ClampProjection,
    ClampProjectionFixture,
    PairProjection,
    check_clamp_projection_laws,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from lofi.deduce import Chart, Derivation
    from lofi.terms import Matching


type Reading = tuple[Term, Term]
type StringReading = tuple[str, str]
type Forest = PackedForest[StringReading, float]


class Equality[T]:
    """Compares values by equality."""

    @property
    def name(self) -> str:
        """Return the relation's name."""
        return "equality"

    @property
    def has_key(self) -> bool:
        """Return whether every class has a representative."""
        return True

    def key(self, term: T, /) -> T:
        """Return the reading itself."""
        return term

    def equivalent(self, left: T, right: T, /) -> bool:
        """Return whether two readings are equal."""
        return left == right


class RejectRule(RootGoal[StringReading]):
    """Root goal excluding one rule name."""

    __slots__ = ("_rejected",)

    def __init__(self, targets: TargetSet[StringReading], rejected: str) -> None:
        super().__init__(targets)
        object.__setattr__(self, "_rejected", rejected)

    @override
    def admits_edge[W](
        self,
        chart: Chart[StringReading, W],
        parent: ItemKey[StringReading],
        edge: Edge[StringReading, W],
        /,
    ) -> bool:
        """Reject edges with the configured rule name."""
        del chart, parent

        return edge.rule_name != self._rejected


SIGNATURE: Final = Signature(
    "clamp-categories",
    ("Cat",),
    (
        Operator("NP", "Cat"),
        Operator("S", "Cat"),
        Operator("fwd", "Cat", arg_sorts=("Cat", "Cat")),
    ),
)
CATEGORY_STORE: Final = TermStore(SIGNATURE)
LANGUAGE: Final = language(("john", "mary", "run", "sleep"))
STORE: Final = LANGUAGE.store
NP: Final = CATEGORY_STORE.op(SIGNATURE.by_name("NP"))
S: Final = CATEGORY_STORE.op(SIGNATURE.by_name("S"))
VP: Final = CATEGORY_STORE.op(SIGNATURE.by_name("fwd"), S, NP)
PAIRS: Final[Matching[Reading]] = Equality()
STRING_PAIRS: Final[Matching[StringReading]] = Equality()
TERMS: Final[Matching[Term]] = Identity()
STRINGS: Final[Matching[str]] = Equality()
PAIR: Final = PairProjection[Term]()
TOLERANCE: Final = 1e-9


def predicate(name: str, variable: str) -> Term:
    """Build the one-place abstraction applying a constant to its variable."""
    body = STORE.op(
        STORE.signature.by_name(LANGUAGE.calculus.app),
        STORE.const(LANGUAGE.term_sort, name),
        STORE.fvar(LANGUAGE.term_sort, variable),
    )

    return STORE.bind(LANGUAGE.calculus.lam, variable, body)


SLEEP: Final = predicate("sleep", "x")
RUN: Final = predicate("run", "x")
JOHN: Final = STORE.const(LANGUAGE.term_sort, "john")
MARY: Final = STORE.const(LANGUAGE.term_sort, "mary")

A: Final = ItemKey((0, 1), (NP, JOHN), "lex")
B: Final = ItemKey((1, 2), (VP, SLEEP), "lex")
C: Final = ItemKey((2, 3), (NP, MARY), "lex")
AB: Final = ItemKey((0, 2), (S, SLEEP), "step")
BC: Final = ItemKey((1, 3), (VP, RUN), "step")
XY: Final = ItemKey((0, 2), (S, RUN), "step")
ROOT: Final = ItemKey((0, 3), (S, SLEEP), "step")
ROOT_TARGETS: Final = TargetSet(((S, SLEEP),), PAIRS)


def crossing() -> PackedForest[Reading, float]:
    """Pack two alternatives under one root, one of which crosses (0, 2)."""
    forest = PackedForest[Reading, float](SpanIndex(3), LogInside())
    forest.add(A, Edge("a", (), A.reading, log(0.5)))
    forest.add(B, Edge("b", (), B.reading, log(0.6)))
    forest.add(C, Edge("c", (), C.reading, log(0.7)))
    forest.add(AB, Edge("ab", (A, B), AB.reading, log(0.4), split=1))
    forest.add(BC, Edge("bc", (B, C), BC.reading, log(0.2), split=2))
    forest.add(ROOT, Edge("root-left", (AB, C), ROOT.reading, log(0.3), split=2))
    forest.add(ROOT, Edge("root-right", (A, BC), ROOT.reading, log(0.1), split=1))

    return forest


def competing() -> PackedForest[Reading, float]:
    """Pack two alternatives under one root that differ in the (0, 2) reading."""
    forest = PackedForest[Reading, float](SpanIndex(3), LogInside())
    forest.add(A, Edge("a", (), A.reading, log(0.5)))
    forest.add(B, Edge("b", (), B.reading, log(0.6)))
    forest.add(C, Edge("c", (), C.reading, log(0.7)))
    forest.add(AB, Edge("ab", (A, B), AB.reading, log(0.4), split=1))
    forest.add(XY, Edge("xy", (A, B), XY.reading, log(0.2), split=1))
    forest.add(ROOT, Edge("root-ab", (AB, C), ROOT.reading, log(0.3), split=2))
    forest.add(ROOT, Edge("root-xy", (XY, C), ROOT.reading, log(0.1), split=2))

    return forest


def snapshot[R](
    forest: PackedForest[R, float],
) -> tuple[tuple[ItemKey[R], float, tuple[Edge[R, float], ...]], ...]:
    """Capture every item, inside weight and edge."""
    return tuple(
        (key, forest.inside(key), forest.edges(key))
        for span in forest.index.spans_by_length()
        for key in forest.items_at(span)
    )


def _topmost_at[R, W](
    derivation: Derivation[R, W],
    span: Cell,
) -> ItemKey[R] | None:
    """Return the highest node covering a span, or None if none does."""
    pending = [derivation]

    while pending:
        node = pending.pop()

        if node.key.span == span:
            return node.key

        pending.extend(node.children)

    return None


def _independently_kept[T](
    forest: Chart[tuple[T, T], float],
    base: Goal[tuple[T, T]],
    clamp: Clamp[T],
) -> int:
    """Count derivations whose topmost node at the clamped span passes the clamp.

    The count walks the derivations of the base goal's roots and reads each
    node's pair directly, so it consults no clamped goal.
    """
    kept = 0

    for root in base.roots(forest):
        for derivation in forest.derivations(root):
            node = _topmost_at(derivation, clamp.span)

            if node is None:
                continue

            category, meaning = node.reading
            named = (not clamp.category or category in clamp.category) and (
                clamp.meaning is None or meaning == clamp.meaning
            )

            if named is not clamp.negate:
                kept += 1

    return kept


def test_the_goal_holds_the_clamps_it_was_built_from() -> None:
    clamp = Clamp((0, 2), meaning=predicate("sleep", "y"))
    base = RootGoal(ROOT_TARGETS)
    goal = ClampedGoal(base, (clamp,), PAIR, TERMS)

    assert goal.clamps == (clamp,)
    assert goal.clamps[0] is clamp
    assert goal.base is base
    assert goal.projection is PAIR
    assert goal.matching is TERMS
    assert isinstance(goal, Goal)


def test_an_alternative_crossing_the_clamped_span_contributes_no_mass() -> None:
    forest = crossing()
    before = snapshot(forest)
    unconstrained = outside_from_edges(forest, RootGoal(ROOT_TARGETS))
    goal = ClampedGoal(
        RootGoal(ROOT_TARGETS),
        (Clamp((0, 2), meaning=predicate("sleep", "z")),),
        PAIR,
        TERMS,
    )
    result = outside_from_edges(forest, goal)
    left = 0.3 * 0.4 * 0.5 * 0.6 * 0.7
    right = 0.1 * 0.5 * 0.2 * 0.6 * 0.7

    assert exp(unconstrained.partition) == pytest.approx(left + right, rel=TOLERANCE)
    assert exp(result.partition) == pytest.approx(left, rel=TOLERANCE)
    assert exp(result.outside(AB)) == pytest.approx(0.3 * 0.7, rel=TOLERANCE)
    assert isinf(result.outside(BC))
    assert exp(result.outside(A)) == pytest.approx(0.3 * 0.4 * 0.6 * 0.7, rel=TOLERANCE)
    assert result.constituent_marginal(AB) == pytest.approx(1.0, rel=TOLERANCE)
    assert result.constituent_marginal(BC) == pytest.approx(0.0, abs=TOLERANCE)
    assert result.rule_marginal(ROOT, forest.edges(ROOT)[0]) == pytest.approx(
        1.0, rel=TOLERANCE
    )
    assert result.rule_marginal(ROOT, forest.edges(ROOT)[1]) == pytest.approx(
        0.0, abs=TOLERANCE
    )
    assert snapshot(forest) == before
    assert len(tuple(forest.derivations(ROOT))) == 2
    assert _independently_kept(forest, RootGoal(ROOT_TARGETS), goal.clamps[0]) == 1


def test_the_unique_containing_child_is_tested_when_it_covers_the_span() -> None:
    forest = competing()
    goal = ClampedGoal(
        RootGoal(ROOT_TARGETS),
        (Clamp((0, 2), meaning=predicate("sleep", "w")),),
        PAIR,
        TERMS,
    )
    result = outside_from_edges(forest, goal)
    ab_edge, xy_edge = forest.edges(ROOT)

    assert goal.admits_edge(forest, ROOT, ab_edge)
    assert not goal.admits_edge(forest, ROOT, xy_edge)
    assert exp(result.partition) == pytest.approx(
        0.3 * 0.4 * 0.5 * 0.6 * 0.7, rel=TOLERANCE
    )
    assert result.constituent_marginal(XY) == pytest.approx(0.0, abs=TOLERANCE)
    assert result.constituent_marginal(AB) == pytest.approx(1.0, rel=TOLERANCE)


def test_a_category_clamp_tests_the_category_component() -> None:
    forest = competing()
    goal = ClampedGoal(
        RootGoal(ROOT_TARGETS),
        (Clamp((1, 2), (VP, NP)),),
        PAIR,
        TERMS,
    )
    rejecting = ClampedGoal(
        RootGoal(ROOT_TARGETS), (Clamp((1, 2), (NP,)),), PAIR, TERMS
    )
    ab_edge = forest.edges(AB)[0]

    assert goal.admits_edge(forest, AB, ab_edge)
    assert not rejecting.admits_edge(forest, AB, ab_edge)
    assert isinf(outside_from_edges(forest, rejecting).partition)


def test_a_clamp_naming_both_requires_both() -> None:
    forest = competing()
    both = ClampedGoal(
        RootGoal(ROOT_TARGETS),
        (Clamp((0, 2), (S,), SLEEP),),
        PAIR,
        TERMS,
    )
    wrong_category = ClampedGoal(
        RootGoal(ROOT_TARGETS),
        (Clamp((0, 2), (NP,), SLEEP),),
        PAIR,
        TERMS,
    )
    ab_edge, xy_edge = forest.edges(ROOT)

    assert both.admits_edge(forest, ROOT, ab_edge)
    assert not both.admits_edge(forest, ROOT, xy_edge)
    assert not wrong_category.admits_edge(forest, ROOT, ab_edge)


def test_a_negated_clamp_forbids_the_named_reading_at_a_constituent() -> None:
    forest = competing()
    goal = ClampedGoal(
        RootGoal(ROOT_TARGETS),
        (Clamp((0, 2), meaning=SLEEP, negate=True),),
        PAIR,
        TERMS,
    )
    result = outside_from_edges(forest, goal)
    ab_edge, xy_edge = forest.edges(ROOT)

    assert not goal.admits_edge(forest, ROOT, ab_edge)
    assert goal.admits_edge(forest, ROOT, xy_edge)
    assert exp(result.partition) == pytest.approx(
        0.1 * 0.2 * 0.5 * 0.6 * 0.7, rel=TOLERANCE
    )


def test_a_negated_clamp_still_rejects_a_crossing_partition() -> None:
    forest = crossing()
    goal = ClampedGoal(
        RootGoal(ROOT_TARGETS),
        (Clamp((0, 2), meaning=RUN, negate=True),),
        PAIR,
        TERMS,
    )
    left_edge, right_edge = forest.edges(ROOT)

    assert goal.admits_edge(forest, ROOT, left_edge)
    assert not goal.admits_edge(forest, ROOT, right_edge)


def test_a_negative_bracket_excludes_the_named_constituent() -> None:
    forest = crossing()
    goal = ClampedGoal(
        RootGoal(ROOT_TARGETS),
        (Clamp((0, 2), require_constituent=False),),
        PAIR,
        TERMS,
    )
    result = outside_from_edges(forest, goal)
    left_edge, right_edge = forest.edges(ROOT)

    assert not goal.admits_edge(forest, ROOT, left_edge)
    assert goal.admits_edge(forest, ROOT, right_edge)
    assert exp(result.partition) == pytest.approx(
        0.1 * 0.5 * 0.2 * 0.6 * 0.7, rel=TOLERANCE
    )


def test_a_soft_clamp_does_not_filter_the_symbolic_forest() -> None:
    forest = crossing()
    base = RootGoal(ROOT_TARGETS)
    goal = ClampedGoal(
        base,
        (
            Clamp(
                (0, 2),
                meaning=RUN,
                mode=ClampMode.SOFT,
                penalty=0.5,
            ),
        ),
        PAIR,
        TERMS,
    )

    assert goal.root_groups(forest) == base.root_groups(forest)
    assert all(goal.admits_edge(forest, ROOT, edge) for edge in forest.edges(ROOT))


def test_a_meaning_clamp_uses_its_declared_equivalence() -> None:
    class Universal(Equality[Term]):
        """One equivalence class over the fixture meanings."""

        @override
        @property
        def name(self) -> str:
            """Return the relation's name."""
            return "universal"

        @override
        def equivalent(self, left: Term, right: Term, /) -> bool:
            """Return true for every pair."""
            del left, right
            return True

    forest = competing()
    goal = ClampedGoal(
        RootGoal(ROOT_TARGETS),
        (Clamp((0, 2), meaning=RUN),),
        PAIR,
        TERMS,
        meaning_matching=Universal(),
    )

    assert all(goal.admits_edge(forest, ROOT, edge) for edge in forest.edges(ROOT))


def test_a_same_span_unary_interior_is_not_tested_again() -> None:
    forest = competing()
    raised = ItemKey((0, 2), (VP, RUN), "raised")
    forest.add(raised, Edge("raise", (AB,), raised.reading, log(0.9)))
    forest.add(ROOT, Edge("root-raised", (raised, C), ROOT.reading, log(0.5), split=2))
    goal = ClampedGoal(
        RootGoal(ROOT_TARGETS), (Clamp((0, 2), meaning=SLEEP),), PAIR, TERMS
    )
    raise_edge = forest.edges(raised)[0]
    root_raised = forest.edges(ROOT)[2]

    assert goal.admits_edge(forest, raised, raise_edge)
    assert not goal.admits_edge(forest, ROOT, root_raised)
    assert goal.admits_edge(forest, AB, forest.edges(AB)[0])


def test_a_root_span_clamp_filters_the_root_directly() -> None:
    forest = competing()
    other = ItemKey((0, 3), (S, RUN), "step")
    forest.add(other, Edge("root-other", (XY, C), other.reading, log(0.5), split=2))
    targets = TargetSet(((S, SLEEP), (S, RUN)), PAIRS)
    goal = ClampedGoal(RootGoal(targets), (Clamp((0, 3), meaning=SLEEP),), PAIR, TERMS)
    result = outside_from_edges(forest, goal)

    assert tuple(RootGoal(targets).roots(forest)) == (ROOT, other)
    assert tuple(goal.roots(forest)) == (ROOT,)
    assert goal.root_groups(forest) == ((ROOT,),)
    assert goal.accepts(forest, ROOT)
    assert not goal.accepts(forest, other)
    assert goal.admits_edge(forest, other, forest.edges(other)[0])
    assert result.constituent_marginal(other) == pytest.approx(0.0, abs=TOLERANCE)
    assert result.constituent_marginal(ROOT) == pytest.approx(1.0, rel=TOLERANCE)


class BothReadings(RootGoal[Reading]):
    """Root goal requiring one root per target reading, as separate groups."""

    __slots__ = ()

    @override
    def root_groups[W](
        self,
        chart: Chart[Reading, W],
        /,
    ) -> tuple[tuple[ItemKey[Reading], ...], ...]:
        """Return one group per target, each holding the roots matching it."""
        return tuple(
            tuple(
                key
                for key in chart.items_at((0, chart.index.n))
                if self.matching.equivalent(key.reading, category)
            )
            for category in self.categories
        )


def test_a_goal_component_losing_every_root_leaves_no_parse() -> None:
    forest = competing()
    other = ItemKey((0, 3), (S, RUN), "step")
    forest.add(other, Edge("root-other", (XY, C), other.reading, log(0.5), split=2))
    base = BothReadings(TargetSet(((S, SLEEP), (S, RUN)), PAIRS))
    goal = ClampedGoal(base, (Clamp((0, 3), meaning=SLEEP),), PAIR, TERMS)

    assert base.root_groups(forest) == ((ROOT,), (other,))
    assert goal.root_groups(forest) == ()


def test_a_root_span_clamp_no_root_passes_leaves_no_parse() -> None:
    forest = competing()
    goal = ClampedGoal(
        RootGoal(ROOT_TARGETS), (Clamp((0, 3), meaning=RUN),), PAIR, TERMS
    )

    assert goal.root_groups(forest) == ()
    assert tuple(goal.roots(forest)) == ()
    assert isinf(outside_from_edges(forest, goal).partition)


def test_a_clamp_reaching_past_the_input_is_refused() -> None:
    forest = competing()
    goal = ClampedGoal(RootGoal(ROOT_TARGETS), (Clamp((2, 4), (NP,)),), PAIR, TERMS)

    with pytest.raises(ValueError, match=r"reaches past the 3-token input"):
        goal.root_groups(forest)


def test_the_base_goal_root_weight_and_edge_admission_are_kept() -> None:
    forest = string_forest()
    targets = TargetSet((("S", "m"),), STRING_PAIRS)
    base = RejectRule(targets, "root-left")
    goal = ClampedGoal(base, (Clamp((0, 1), ("A",)),), PairProjection[str](), STRINGS)
    left, right = forest.edges(ItemKey((0, 3), ("S", "m"), "step"))

    root = goal.root_groups(forest)[0][0]
    assert goal.root_weight(root) == base.root_weight(root)
    assert goal.combination_weight(1) == base.combination_weight(1)
    assert not goal.admits_edge(forest, ItemKey((0, 3), ("S", "m"), "step"), left)
    assert goal.admits_edge(forest, ItemKey((0, 3), ("S", "m"), "step"), right)


def test_a_cover_fragment_penalty_and_fragment_clamps_are_honored() -> None:
    forest = PackedForest[StringReading, float](SpanIndex(3), LogInside())
    left = ItemKey((0, 2), ("X", "m"), "step")
    right = ItemKey((2, 3), ("C", "k"), "lex")
    a = ItemKey((0, 1), ("A", "m"), "lex")
    b = ItemKey((1, 2), ("B", "m"), "lex")
    forest.add(a, Edge("a", (), a.reading, 0.0))
    forest.add(b, Edge("b", (), b.reading, 0.0))
    forest.add(right, Edge("c", (), right.reading, 0.0))
    forest.add(left, Edge("x", (a, b), left.reading, 0.0, split=1))
    base = CoverGoal(2, penalty=0.5)
    projection = PairProjection[str]()
    inside = ClampedGoal(base, (Clamp((0, 2), ("X",)),), projection, STRINGS)
    crossing_fragments = ClampedGoal(
        base, (Clamp((1, 3), ("X",)),), projection, STRINGS
    )
    failing = ClampedGoal(base, (Clamp((2, 3), ("A",)),), projection, STRINGS)

    assert inside.root_groups(forest) == ((left,), (right,))
    assert inside.combination_weight(2) == base.penalty_for(2)
    assert crossing_fragments.root_groups(forest) == ()
    assert failing.root_groups(forest) == ()
    assert inside.accepts(forest, left)
    assert not failing.accepts(forest, right)


def _cover_forest() -> Forest:
    """Build a three-token forest with a two-fragment cover and no spanning item."""
    forest = PackedForest[StringReading, float](SpanIndex(3), LogInside())
    left = ItemKey((0, 2), ("X", "m"), "step")
    right = ItemKey((2, 3), ("C", "k"), "lex")
    a = ItemKey((0, 1), ("A", "m"), "lex")
    b = ItemKey((1, 2), ("B", "m"), "lex")
    forest.add(a, Edge("a", (), a.reading, 0.0))
    forest.add(b, Edge("b", (), b.reading, 0.0))
    forest.add(right, Edge("c", (), right.reading, 0.0))
    forest.add(left, Edge("x", (a, b), left.reading, 0.0, split=1))

    return forest


def _two_root_forest() -> PackedForest[Reading, float]:
    """Build the competing forest with a second spanning root reading RUN."""
    forest = competing()
    other = ItemKey((0, 3), (S, RUN), "step")
    forest.add(other, Edge("root-other", (XY, C), other.reading, log(0.5), split=2))

    return forest


def test_the_goal_laws_hold_of_clamped_root_and_cover_goals() -> None:
    both = BothReadings(TargetSet(((S, SLEEP), (S, RUN)), PAIRS))
    root_goals = (
        ClampedGoal(
            RootGoal(ROOT_TARGETS), (Clamp((0, 2), meaning=SLEEP),), PAIR, TERMS
        ),
        ClampedGoal(RootGoal(ROOT_TARGETS), (Clamp((0, 3), meaning=RUN),), PAIR, TERMS),
        ClampedGoal(both, (Clamp((0, 3), meaning=SLEEP),), PAIR, TERMS),
        ClampedGoal(both, (Clamp((1, 2), (VP,)),), PAIR, TERMS),
    )
    cover = CoverGoal(2, penalty=0.5)
    cover_goals = (
        ClampedGoal(cover, (Clamp((0, 2), ("X",)),), PairProjection[str](), STRINGS),
        ClampedGoal(cover, (Clamp((1, 3), ("X",)),), PairProjection[str](), STRINGS),
        ClampedGoal(cover, (Clamp((2, 3), ("A",)),), PairProjection[str](), STRINGS),
    )
    empty = PackedForest[Reading, float](SpanIndex(3), LogInside())

    for goal in root_goals:
        check_goal_laws(GoalFixture(goal, (crossing(), _two_root_forest(), empty)))

    for string_goal in cover_goals:
        check_goal_laws(GoalFixture(string_goal, (_cover_forest(), string_forest())))


def test_acceptance_agrees_with_enumeration_when_a_component_is_emptied() -> None:
    forest = _two_root_forest()
    both = BothReadings(TargetSet(((S, SLEEP), (S, RUN)), PAIRS))
    goal = ClampedGoal(both, (Clamp((0, 3), meaning=SLEEP),), PAIR, TERMS)
    cover = CoverGoal(2, penalty=0.5)
    crossing_fragments = ClampedGoal(
        cover, (Clamp((1, 3), ("X",)),), PairProjection[str](), STRINGS
    )
    cover_forest = _cover_forest()

    assert both.accepts(forest, ROOT)
    assert tuple(goal.roots(forest)) == ()
    assert not goal.accepts(forest, ROOT)
    assert cover.accepts(cover_forest, ItemKey((2, 3), ("C", "k"), "lex"))
    assert tuple(crossing_fragments.roots(cover_forest)) == ()
    assert not crossing_fragments.accepts(
        cover_forest, ItemKey((2, 3), ("C", "k"), "lex")
    )


def test_a_clamp_reads_the_exact_featured_category_of_an_item() -> None:
    signature = Signature(
        "featured",
        ("Cat",),
        (
            Operator("NP", "Cat"),
            Operator("S", "Cat"),
            Operator("S[dcl]", "Cat", erases_to="S"),
        ),
    )
    store = TermStore(signature)
    noun_phrase = store.op(signature.by_name("NP"))
    sentence = store.op(signature.by_name("S"))
    declarative = store.op(signature.by_name("S[dcl]"))
    erasure = FeatureErasure()
    forest = PackedForest[Term, float](SpanIndex(2), LogInside())
    a = ItemKey((0, 1), noun_phrase, "lex")
    b = ItemKey((1, 2), declarative, "lex")
    root = ItemKey((0, 2), declarative, "step")
    forest.add(a, Edge("a", (), noun_phrase, 0.0))
    forest.add(b, Edge("b", (), declarative, 0.0))
    forest.add(root, Edge("s", (a, b), declarative, 0.0, split=1))
    base = RootGoal(TargetSet((declarative,), erasure))
    projection = CategoryProjection[Term]()
    inner = ClampedGoal(base, (Clamp((1, 2), (declarative,)),), projection, erasure)
    at_root = ClampedGoal(base, (Clamp((0, 2), (declarative,)),), projection, erasure)
    exact = ClampedGoal(base, (Clamp((1, 2), (declarative,)),), projection, TERMS)
    bare = ClampedGoal(base, (Clamp((1, 2), (sentence,)),), projection, TERMS)

    assert b.reading is declarative
    assert inner.admits_edge(forest, root, forest.edges(root)[0])
    assert exp(outside_from_edges(forest, inner).partition) == pytest.approx(1.0)
    assert tuple(at_root.roots(forest)) == (root,)
    assert exact.admits_edge(forest, root, forest.edges(root)[0])
    assert not bare.admits_edge(forest, root, forest.edges(root)[0])


def test_the_shipped_projections_pass_the_projection_laws_over_chart_readings() -> None:
    check_clamp_projection_laws(
        ClampProjectionFixture(PAIR, (A.reading, B.reading, ROOT.reading))
    )
    check_clamp_projection_laws(
        ClampProjectionFixture(CategoryProjection[Term](), (NP, S, VP))
    )


def test_a_partition_with_no_child_containing_the_span_is_rejected() -> None:
    forest = PackedForest[StringReading, float](SpanIndex(3), LogInside())
    a = ItemKey((0, 1), ("A", "m"), "lex")
    b = ItemKey((1, 2), ("B", "m"), "lex")
    c = ItemKey((2, 3), ("C", "m"), "lex")
    root = ItemKey((0, 3), ("S", "m"), "step")

    for key in (a, b, c):
        forest.add(key, Edge("lex", (), key.reading, 0.0))

    forest.add(root, Edge("ternary", (a, b, c), root.reading, 0.0))
    goal = ClampedGoal(
        RootGoal(TargetSet((root.reading,), STRING_PAIRS)),
        (Clamp((1, 3), ("B",)),),
        PairProjection[str](),
        STRINGS,
    )

    assert not goal.admits_edge(forest, root, forest.edges(root)[0])


def test_several_clamps_are_all_enforced() -> None:
    forest = competing()
    goal = ClampedGoal(
        RootGoal(ROOT_TARGETS),
        (Clamp((0, 2), meaning=SLEEP), Clamp((2, 3), (NP,))),
        PAIR,
        TERMS,
    )
    stricter = ClampedGoal(
        RootGoal(ROOT_TARGETS),
        (Clamp((0, 2), meaning=SLEEP), Clamp((2, 3), (S,))),
        PAIR,
        TERMS,
    )
    ab_edge, xy_edge = forest.edges(ROOT)

    assert goal.admits_edge(forest, ROOT, ab_edge)
    assert not goal.admits_edge(forest, ROOT, xy_edge)
    assert not stricter.admits_edge(forest, ROOT, ab_edge)


def test_a_category_projection_serves_category_only_readings() -> None:
    forest = PackedForest[str, float](SpanIndex(2), LogInside())
    a = ItemKey((0, 1), "A", "lex")
    b = ItemKey((1, 2), "B", "lex")
    root = ItemKey((0, 2), "S", "step")
    forest.add(a, Edge("a", (), "A", 0.0))
    forest.add(b, Edge("b", (), "B", 0.0))
    forest.add(root, Edge("s", (a, b), "S", 0.0, split=1))
    projection = CategoryProjection[str]()
    base = RootGoal(TargetSet(("S",), Equality[str]()))
    goal = ClampedGoal(base, (Clamp((0, 1), ("A",)),), projection, STRINGS)
    rejecting = ClampedGoal(base, (Clamp((0, 1), ("B",)),), projection, STRINGS)

    assert not projection.has_meaning
    assert projection.category("A") == "A"
    assert projection.meaning("A") is None
    assert isinstance(projection, ClampProjection)
    assert goal.admits_edge(forest, root, forest.edges(root)[0])
    assert not rejecting.admits_edge(forest, root, forest.edges(root)[0])
    assert repr(projection) == "CategoryProjection()"


def test_a_pair_projection_reads_both_components() -> None:
    assert PAIR.has_meaning
    assert PAIR.category((NP, JOHN)) is NP
    assert PAIR.meaning((NP, JOHN)) is JOHN
    assert isinstance(PAIR, ClampProjection)
    assert repr(PAIR) == "PairProjection()"


def test_construction_refuses_clamps_the_goal_cannot_enforce() -> None:
    base = RootGoal(ROOT_TARGETS)

    with pytest.raises(ValueError, match="narrows nothing"):
        ClampedGoal(base, (), PAIR, TERMS)

    with pytest.raises(ValueError, match=r"covers no position"):
        ClampedGoal(base, (Clamp((2, 2), (NP,)),), PAIR, TERMS)

    with pytest.raises(ValueError, match=r"covers no position"):
        ClampedGoal(base, (Clamp((2, 1), (NP,)),), PAIR, TERMS)

    with pytest.raises(ValueError, match=r"start is negative"):
        ClampedGoal(base, (Clamp((-1, 1), (NP,)),), PAIR, TERMS)

    with pytest.raises(TypeError):
        ClampedGoal(
            base, (Clamp(cast("tuple[int, int]", (0, 1.5)), (NP,)),), PAIR, TERMS
        )

    with pytest.raises(ValueError, match="names the category or the meaning"):
        ClampedGoal(base, (Clamp((0, 1)),), PAIR, TERMS)

    with pytest.raises(ValueError, match="readings hold none"):
        ClampedGoal(
            RootGoal(TargetSet((S,), Equality[Term]())),
            (Clamp((0, 1), meaning=SLEEP),),
            CategoryProjection[Term](),
            TERMS,
        )


def test_a_clamped_goal_is_immutable_and_copies_to_itself() -> None:
    goal = ClampedGoal(RootGoal(ROOT_TARGETS), (Clamp((0, 2), (S,)),), PAIR, TERMS)

    with pytest.raises(AttributeError, match="immutable"):
        goal.__setattr__("_clamps", ())

    with pytest.raises(AttributeError, match="immutable"):
        goal.__delattr__("_clamps")

    assert copy.copy(goal) is goal
    assert copy.deepcopy(goal) is goal
    assert repr(goal) == (
        f"ClampedGoal({RootGoal(ROOT_TARGETS)!r}, clamps=1, matching='identity')"
    )

    with pytest.raises(TypeError, match="cannot be pickled"):
        pickle.dumps(goal)


def string_forest() -> Forest:
    """Build a three-token string forest with two alternatives under one root."""
    forest = PackedForest[StringReading, float](SpanIndex(3), LogInside())
    a = ItemKey((0, 1), ("A", "m"), "lex")
    b = ItemKey((1, 2), ("B", "m"), "lex")
    c = ItemKey((2, 3), ("C", "m"), "lex")
    ab = ItemKey((0, 2), ("AB", "m"), "step")
    bc = ItemKey((1, 3), ("BC", "m"), "step")
    root = ItemKey((0, 3), ("S", "m"), "step")

    for key in (a, b, c):
        forest.add(key, Edge("lex", (), key.reading, 0.0))

    forest.add(ab, Edge("ab", (a, b), ab.reading, 0.0, split=1))
    forest.add(bc, Edge("bc", (b, c), bc.reading, 0.0, split=2))
    forest.add(root, Edge("root-left", (ab, c), root.reading, 0.0, split=2))
    forest.add(root, Edge("root-right", (a, bc), root.reading, 0.0, split=1))

    return forest


CATEGORIES: Final = ("A", "B", "C")
MEANINGS: Final = ("m", "k")


@st.composite
def generated_forests(draw: st.DrawFn) -> tuple[Forest, Clamp[str]]:
    """Draw a unit-weighted forest over paired readings and a clamp inside it."""
    n = draw(st.integers(1, 4))
    forest = PackedForest[StringReading, float](SpanIndex(n), LogInside())
    readings = st.tuples(st.sampled_from(CATEGORIES), st.sampled_from(MEANINGS))

    for span in forest.index.spans_by_length():
        i, j = span
        held = tuple(dict.fromkeys(draw(st.lists(readings, min_size=1, max_size=2))))
        keys = tuple(ItemKey(span, reading, "step") for reading in held)

        for position, key in enumerate(keys):
            if j - i == 1:
                forest.add(key, Edge("lex", (), key.reading, 0.0))

            else:
                splits = draw(
                    st.lists(st.integers(i + 1, j - 1), min_size=1, max_size=j - i - 1)
                )

                for split in dict.fromkeys(splits):
                    left = draw(st.sampled_from(forest.items_at((i, split))))
                    right = draw(st.sampled_from(forest.items_at((split, j))))
                    forest.add(
                        key,
                        Edge(
                            f"step-{split}",
                            (left, right),
                            key.reading,
                            0.0,
                            split=split,
                        ),
                    )

            if position and draw(st.booleans()):
                forest.add(key, Edge("unary", (keys[0],), key.reading, 0.0))

    start = draw(st.integers(0, n - 1))
    end = draw(st.integers(start + 1, n))
    category = draw(st.frozensets(st.sampled_from(CATEGORIES), max_size=2))
    meaning = draw(st.one_of(st.none(), st.sampled_from(MEANINGS)))
    negate = draw(st.booleans())

    if not category and meaning is None:
        meaning = draw(st.sampled_from(MEANINGS))

    return forest, Clamp((start, end), category, meaning, negate=negate)


def _all_roots(forest: Forest) -> RootGoal[StringReading]:
    """Return a root goal accepting every spanning item."""
    n = forest.index.n

    return RootGoal(
        TargetSet(tuple(key.reading for key in forest.items_at((0, n))), STRING_PAIRS)
    )


@settings(max_examples=200, deadline=None)
@given(generated_forests())
def test_the_admitted_packed_count_matches_independent_derivation_filtering(
    case: tuple[Forest, Clamp[str]],
) -> None:
    forest, clamp = case
    base = _all_roots(forest)
    goal = ClampedGoal(base, (clamp,), PairProjection[str](), STRINGS)
    before = snapshot(forest)
    packed = exp(outside_from_edges(forest, goal).partition)
    expected = _independently_kept(forest, base, clamp)

    assert round(packed) == expected
    assert abs(packed - expected) < 1e-6 * max(1, expected)
    assert snapshot(forest) == before


@settings(max_examples=100, deadline=None)
@given(generated_forests())
def test_clamped_roots_are_the_base_roots_passing_at_the_root_span(
    case: tuple[Forest, Clamp[str]],
) -> None:
    forest, clamp = case
    base = _all_roots(forest)
    goal = ClampedGoal(base, (clamp,), PairProjection[str](), STRINGS)
    roots: Iterator[ItemKey[StringReading]] = goal.roots(forest)
    held = tuple(roots)

    for root in base.roots(forest):
        assert goal.accepts(forest, root) == (root in held)

    if clamp.span != (0, forest.index.n):
        assert held == tuple(base.roots(forest))
