from typing import TYPE_CHECKING, Final, NamedTuple

import pytest
from tests.language import language

from lofi.deduce import Edge, ItemKey, LogInside, PackedForest, RootGoal, SpanIndex
from lofi.lexicon import Clamp
from lofi.terms import (
    Identity,
    Operator,
    Signature,
    TargetSet,
    Term,
    TermStore,
)
from lofi.typecheck import ClampedGoal, PairProjection

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture
    from tests.benchmarks.conftest import WallClockBudget

    from lofi.terms import Matching

#: Nanoseconds one clamped edge admission may take.
LOOKUP_NS: Final = 11_000


type Reading = tuple[Term, Term]


class PairEquality:
    """Compares readings by equality of both components."""

    @property
    def name(self) -> str:
        """Return the relation's name."""
        return "pair-equality"

    @property
    def has_key(self) -> bool:
        """Return whether every class has a representative."""
        return True

    def key(self, term: Reading, /) -> Reading:
        """Return the reading itself."""
        return term

    def equivalent(self, left: Reading, right: Reading, /) -> bool:
        """Return whether two readings are equal."""
        return left == right


class Lookup(NamedTuple):
    """One fixed edge admission under a clamped goal."""

    forest: PackedForest[Reading, float]
    goal: ClampedGoal[Term, Reading]
    parent: ItemKey[Reading]
    edge: Edge[Reading, float]


PAIRS: Final[Matching[Reading]] = PairEquality()


@pytest.fixture(scope="module")
def lookup() -> Lookup:
    """Build a spanning edge whose child covers a clamped span exactly."""
    signature = Signature(
        "clamp-benchmark-categories",
        ("Cat",),
        (
            Operator("NP", "Cat"),
            Operator("S", "Cat"),
        ),
    )
    categories = TermStore(signature)
    declared = language(("john", "mary", "sleep"))
    store = declared.store
    noun_phrase = categories.op(signature.by_name("NP"))
    sentence = categories.op(signature.by_name("S"))
    application = store.signature.by_name(declared.calculus.app)
    sleep = store.bind(
        declared.calculus.lam,
        "x",
        store.op(
            application,
            store.const(declared.term_sort, "sleep"),
            store.fvar(declared.term_sort, "x"),
        ),
    )
    john = store.const(declared.term_sort, "john")
    mary = store.const(declared.term_sort, "mary")
    forest = PackedForest[Reading, float](SpanIndex(3), LogInside())
    a = ItemKey((0, 1), (noun_phrase, john), "lex")
    b = ItemKey((1, 2), (sentence, sleep), "lex")
    c = ItemKey((2, 3), (noun_phrase, mary), "lex")
    ab = ItemKey((0, 2), (sentence, sleep), "step")
    root = ItemKey((0, 3), (sentence, sleep), "step")

    for key in (a, b, c):
        forest.add(key, Edge("lex", (), key.reading, 0.0))

    forest.add(ab, Edge("ab", (a, b), ab.reading, 0.0, split=1))
    forest.add(root, Edge("root", (ab, c), root.reading, 0.0, split=2))
    goal = ClampedGoal(
        RootGoal(TargetSet((root.reading,), PAIRS)),
        (
            Clamp((0, 2), (sentence, noun_phrase), sleep),
            Clamp((2, 3), (noun_phrase,)),
            Clamp((0, 1), meaning=john),
            Clamp((1, 2), (sentence,), negate=True),
        ),
        PairProjection[Term](),
        Identity[Term](),
    )

    return Lookup(forest, goal, root, forest.edges(root)[0])


def test_clamp_lookup_on_one_edge_is_within_budget(
    benchmark: BenchmarkFixture,
    lookup: Lookup,
    wall_clock_budget: WallClockBudget,
) -> None:
    benchmark(
        lookup.goal.admits_edge,
        lookup.forest,
        lookup.parent,
        lookup.edge,
    )

    measured = benchmark.stats

    if measured is None:
        pytest.fail("clamp lookup ran but pytest-benchmark timed nothing")

    nanoseconds = measured.stats.median * 1e9
    benchmark.extra_info["clamps"] = len(lookup.goal.clamps)
    benchmark.extra_info["median_ns_per_lookup"] = nanoseconds
    benchmark.extra_info["mean_ns_per_lookup"] = measured.stats.mean * 1e9
    benchmark.extra_info["budget_ns"] = LOOKUP_NS

    if wall_clock_budget.enforced:
        assert nanoseconds <= LOOKUP_NS


def test_clamp_lookup_admits_the_priced_edge(lookup: Lookup) -> None:
    admitted = lookup.goal.admits_edge(lookup.forest, lookup.parent, lookup.edge)

    assert admitted
