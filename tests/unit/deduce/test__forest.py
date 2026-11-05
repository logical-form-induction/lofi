import pickle
from math import exp, isclose, log
from typing import TYPE_CHECKING, override

import pytest

from lofi._native import ForestTable, PythonCarrier
from lofi.deduce._forest import PackedForest
from lofi.deduce._semiring import (
    Boolean,
    Counting,
    Expectation,
    Forest,
    KBest,
    LogInside,
    Viterbi,
)
from lofi.deduce._spans import SpanIndex
from lofi.deduce._values import Edge, ItemKey, PrunedMass

if TYPE_CHECKING:
    from lofi.deduce._ports import Chart, Semiring

DEEP = 20


class Counted:
    """Counts hash calls."""

    hashes = 0

    def __init__(self, name: str) -> None:
        self._name = name

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether two categories have the same name."""
        if type(other) is not Counted:
            return NotImplemented

        return self._name == other._name

    @override
    def __hash__(self) -> int:
        """Return the name's hash, counting the call."""
        Counted.hashes += 1

        return hash(self._name)

    @override
    def __repr__(self) -> str:
        """Return a rendering naming the category."""
        return f"Counted({self._name!r})"


class Tallied:
    """Counts semiring calls."""

    def __init__(self) -> None:
        self._inner = LogInside()
        self.plus_calls = 0
        self.times_calls = 0

    @property
    def name(self) -> str:
        """Return the name the algebra is selected by."""
        return self._inner.name

    @property
    def log_probability(self) -> bool:
        """Report whether weights are log probabilities."""
        return self._inner.log_probability

    @property
    def zero(self) -> float:
        """Return the additive identity."""
        return self._inner.zero

    @property
    def one(self) -> float:
        """Return the multiplicative identity."""
        return self._inner.one

    @property
    def lowering(self) -> PythonCarrier:
        """Select Python-carried arithmetic."""
        return PythonCarrier()

    def plus(self, a: float, b: float, /) -> float:
        """Return the log-sum of two weights, counting the call.

        Parameters
        ----------
        a : float
        b : float
        """
        self.plus_calls += 1

        return self._inner.plus(a, b)

    def times(self, a: float, b: float, /) -> float:
        """Return the sum of two log weights, counting the call.

        Parameters
        ----------
        a : float
        b : float
        """
        self.times_calls += 1

        return self._inner.times(a, b)

    def is_zero(self, a: float, /) -> bool:
        """Return whether a weight is the additive identity.

        Parameters
        ----------
        a : float
        """
        return self._inner.is_zero(a)


DET = ItemKey((0, 1), "NP/N", "lex")
NOUN = ItemKey((1, 2), "N", "lex")
NAME = ItemKey((1, 2), "NP", "lex")
PHRASE = ItemKey((0, 2), "NP", "step")


def filled() -> PackedForest[str, float]:
    forest = PackedForest[str, float](SpanIndex(2), LogInside())
    forest.add(DET, Edge("lex", (), "NP/N", log(0.5)))
    forest.add(NOUN, Edge("lex", (), "N", log(1.0)))
    forest.add(NAME, Edge("lex", (), "NP", log(0.25)))
    forest.add(PHRASE, Edge("fa", (DET, NOUN), "NP", log(0.5), split=1))
    forest.add(PHRASE, Edge("ba", (DET, NAME), "NP", log(0.5), split=1))

    return forest


def read_through[W](chart: Chart[str, W], key: ItemKey[str]) -> tuple[int, int]:
    return (len(chart.items_at(key.span)), len(chart.edges(key)))


def hashes_of_last_insertion(depth: int) -> int:
    """Return how often a term is hashed by the last insertion of a chain.

    Parameters
    ----------
    depth : int
        How many binary steps the chain runs to.

    Returns
    -------
    int
        The number of times a category was hashed by the last insertion.
    """
    n = depth + 1
    forest = PackedForest[Counted, float](SpanIndex(n), LogInside())
    tag = Counted("X")
    leaves = [ItemKey((i, i + 1), tag) for i in range(n)]

    for leaf in leaves:
        forest.add(leaf, Edge("lex", (), tag, log(0.5)))

    spine = leaves[0]

    for k in range(1, n):
        parent = ItemKey((0, k + 1), tag)
        edge = Edge("step", (spine, leaves[k]), tag, log(0.5), split=k)

        if k == n - 1:
            Counted.hashes = 0
            forest.add(parent, edge)

            return Counted.hashes

        forest.add(parent, edge)
        spine = parent

    msg = f"a chain of depth {depth} has no last insertion"
    raise AssertionError(msg)


def test_a_thousand_steps_under_one_key_leave_one_item() -> None:
    forest = PackedForest[str, float](SpanIndex(2), LogInside())
    key = ItemKey((0, 2), "NP", "step")
    reached = 1000

    for _ in range(reached):
        forest.add(key, Edge("fa", (), "NP", log(0.001)))

    assert forest.items_at((0, 2)) == (key,)
    assert len(forest.edges(key)) == reached
    assert forest.num_items == 1
    assert forest.num_edges == reached
    assert isclose(exp(forest.inside(key)), reached * 0.001)


def test_native_forest_storage_keeps_child_order_and_exact_tags() -> None:
    semiring = Counting()
    table = ForestTable(2, semiring, semiring.lowering)
    left = ((0, 1), (0, 0))
    right = ((1, 2), (1, 0))
    parent = ((0, 2), (2, 1))
    table.add(left, (10, [], 0, 2, None, 0))
    table.add(right, (11, [], 1, 3, None, 0))
    table.add(parent, (12, [left, right], 2, 5, 1, 7))

    assert table.items_at((0, 2)) == [(2, 1)]
    assert table.inside(parent) == 30
    assert table.edges(parent) == [(12, [left, right], 2, 5, 1, 7)]
    assert table.num_items == 3
    assert table.num_edges == 3


@pytest.mark.parametrize("span", [(-1, 1), (0, 0), (1, 0), (0, 3), (2, 3)])
def test_an_item_outside_the_span_index_is_refused_before_mutation(
    span: tuple[int, int],
) -> None:
    algebra = Tallied()
    forest = PackedForest[str, float](SpanIndex(2), algebra)
    forest.add(DET, Edge("lex", (), "NP/N", log(0.5)))
    algebra.plus_calls = 0
    algebra.times_calls = 0
    invalid = ItemKey(span, "NP", "step")

    with pytest.raises(ValueError, match="outside the forest span index"):
        forest.add(invalid, Edge("fa", (DET,), "NP", log(0.5)))

    assert forest.items_at((0, 1)) == (DET,)
    assert forest.items_at(span) == ()
    assert forest.num_items == 1
    assert forest.num_edges == 1
    assert algebra.plus_calls == 0
    assert algebra.times_calls == 0


def test_the_forest_multiplies_the_premises_into_a_step() -> None:
    forest = filled()
    both = 0.5 * 0.5 * 1.0 + 0.5 * 0.5 * 0.25

    assert isclose(exp(forest.inside(PHRASE)), both)
    assert isclose(exp(forest.inside(DET)), 0.5)
    assert read_through(forest, PHRASE) == (1, 2)
    assert read_through(forest, NOUN) == (2, 1)


def test_a_step_over_a_premise_the_forest_lacks_contributes_zero() -> None:
    forest = PackedForest[str, float](SpanIndex(2), LogInside())
    absent = ItemKey((0, 1), "N", "lex")
    key = ItemKey((0, 2), "NP", "step")

    forest.add(NOUN, Edge("lex", (), "N", log(1.0)))
    forest.add(key, Edge("fa", (absent, NOUN), "NP", log(0.5), split=1))

    assert forest.inside(key) == LogInside().zero
    assert len(forest.edges(key)) == 1


def test_the_forest_answers_for_an_item_it_does_not_hold() -> None:
    forest = filled()
    absent = ItemKey((0, 2), "S", "step")

    assert forest.inside(absent) == LogInside().zero
    assert forest.edges(absent) == ()
    assert forest.items_at((0, 1)) == (DET,)

    with pytest.raises(KeyError):
        forest.term(absent)


def test_the_forest_names_the_exact_term_of_an_item() -> None:
    forest = PackedForest[str, float](SpanIndex(1), LogInside())
    key = ItemKey((0, 1), "S[dcl]")

    forest.add(key, Edge("lex", (), "S[dcl]", log(0.5)))

    assert forest.term(key) == "S[dcl]"


def test_two_feature_variants_are_two_items_with_their_own_weights() -> None:
    forest = PackedForest[str, float](SpanIndex(1), LogInside())

    for drawn, weight in (("S[dcl]", 0.25), ("S[b]", 0.5)):
        forest.add(ItemKey((0, 1), drawn), Edge("lex", (), drawn, log(weight)))

    assert forest.items_at((0, 1)) == (
        ItemKey((0, 1), "S[dcl]"),
        ItemKey((0, 1), "S[b]"),
    )
    assert [(term, exp(weight)) for term, weight in forest.categories_at((0, 1))] == [
        ("S[dcl]", pytest.approx(0.25)),
        ("S[b]", pytest.approx(0.5)),
    ]


def test_a_category_sums_its_weight_over_normal_form_tags() -> None:
    forest = PackedForest[str, int](SpanIndex(1), Counting())

    for tag, count in (("FC", 2), ("OTHER", 3), ("FC", 4)):
        forest.add(ItemKey((0, 1), "S", tag), Edge("lex", (), "S", count))

    forest.add(ItemKey((0, 1), "NP", "OTHER"), Edge("lex", (), "NP", 1))

    assert forest.categories_at((0, 1)) == (("S", 9), ("NP", 1))
    assert forest.categories_at((0, 0)) == ()


def test_an_items_at_result_is_a_snapshot_an_insertion_does_not_change() -> None:
    forest = PackedForest[str, float](SpanIndex(1), LogInside())
    first = ItemKey((0, 1), "N", "lex")
    forest.add(first, Edge("lex", (), "N", log(0.5)))

    seen: list[ItemKey[str]] = []

    for key in forest.items_at((0, 1)):
        seen.append(key)
        forest.add(ItemKey((0, 1), "NP", "lex"), Edge("lex", (), "NP", log(0.5)))

    assert seen == [first]
    assert len(forest.items_at((0, 1))) == 2


def test_an_edges_result_is_a_snapshot_an_insertion_does_not_change() -> None:
    forest = filled()
    edges = forest.edges(PHRASE)

    forest.add(PHRASE, Edge("cc", (DET, NOUN), "NP", log(0.1), split=1))

    assert len(edges) == 2
    assert len(forest.edges(PHRASE)) == 3


def test_insertion_hashes_the_premises_and_not_the_subtree_beneath_them() -> None:
    counts = [hashes_of_last_insertion(depth) for depth in range(1, DEEP + 1)]

    assert len(set(counts)) == 1
    assert counts[0] < DEEP


def test_insertion_calls_plus_once_and_times_once_per_premise() -> None:
    algebra = Tallied()
    forest = PackedForest[str, float](SpanIndex(2), algebra)
    key = ItemKey((0, 2), "NP", "step")

    forest.add(DET, Edge("lex", (), "NP/N", log(0.5)))
    forest.add(NOUN, Edge("lex", (), "N", log(1.0)))
    forest.add(key, Edge("fa", (DET, NOUN), "NP", log(0.5), split=1))

    algebra.plus_calls = 0
    algebra.times_calls = 0
    forest.add(key, Edge("ba", (DET, NOUN), "NP", log(0.25), split=1))

    assert algebra.plus_calls == 1
    assert algebra.times_calls == 2


def test_the_forest_reads_back_one_tree_for_each_way_an_item_was_reached() -> None:
    forest = filled()
    trees = tuple(forest.derivations(PHRASE))

    assert len(trees) == 2
    assert {tree.edge.rule_name for tree in trees} == {"fa", "ba"}
    assert all(tree.key == PHRASE for tree in trees)
    assert all(len(tree.children) == 2 for tree in trees)
    assert all(child.children == () for tree in trees for child in tree.children)


def test_the_forest_reads_back_the_product_of_two_ambiguous_premises() -> None:
    forest = filled()
    top = ItemKey((0, 2), "S", "step")

    forest.add(NOUN, Edge("lex2", (), "N", log(0.5)))
    forest.add(top, Edge("tr", (PHRASE,), "S", log(0.5)))

    assert len(tuple(forest.derivations(top))) == 3


def test_a_cycle_through_a_unary_step_yields_no_tree() -> None:
    forest = PackedForest[str, float](SpanIndex(1), LogInside())
    key = ItemKey((0, 1), "N", "lex")

    forest.add(key, Edge("lex", (), "N", log(0.5)))
    forest.add(key, Edge("tr", (key,), "N", log(0.5)))

    trees = tuple(forest.derivations(key))

    assert len(trees) == 1
    assert trees[0].edge.rule_name == "lex"


def test_reading_derivations_of_an_absent_item_yields_nothing() -> None:
    assert tuple(filled().derivations(ItemKey((0, 2), "S", "step"))) == ()


@pytest.mark.parametrize(
    "algebra",
    [
        Boolean(),
        LogInside(),
        Viterbi(),
        Counting(),
        KBest(2),
        Forest[str](),
        Expectation(1),
    ],
    ids=lambda algebra: algebra.name,
)
def test_the_forest_aggregates_under_every_shipped_algebra[W](
    algebra: Semiring[W],
) -> None:
    forest = PackedForest[str, W](SpanIndex(2), algebra)
    key = ItemKey((0, 2), "NP", "step")

    assert forest.inside(key) == algebra.zero

    forest.add(DET, Edge("lex", (), "NP/N", algebra.one))
    forest.add(NOUN, Edge("lex", (), "N", algebra.one))
    forest.add(key, Edge("fa", (DET, NOUN), "NP", algebra.one, split=1))
    forest.add(key, Edge("ba", (DET, NOUN), "NP", algebra.one, split=1))

    assert forest.inside(key) == algebra.plus(algebra.one, algebra.one)
    assert read_through(forest, key) == (1, 2)


def test_the_forest_renders_its_length_and_its_item_count() -> None:
    assert repr(filled()) == "PackedForest(n=2, items=4)"


def test_the_forest_names_the_index_and_the_algebra_it_was_built_over() -> None:
    index = SpanIndex(2)
    algebra = LogInside()
    forest = PackedForest[str, float](index, algebra)

    assert forest.index is index
    assert forest.semiring is algebra
    assert forest.num_items == 0
    assert forest.num_edges == 0
    assert not forest.truncated


def test_truncation_is_sticky() -> None:
    forest = PackedForest[str, float](SpanIndex(1), LogInside())

    forest.mark_truncated()
    forest.mark_truncated()

    assert forest.truncated


def test_pruned_weight_accumulates_per_cell_in_the_forests_algebra() -> None:
    forest = PackedForest[str, float](SpanIndex(2), LogInside())

    forest.record_pruned((0, 2), log(0.25))
    forest.record_pruned((0, 2), log(0.5))
    forest.record_pruned((1, 2), log(0.125))

    assert set(forest.pruned) == {(0, 2), (1, 2)}
    assert isclose(forest.pruned[0, 2], log(0.75))
    assert forest.pruned_mass is None

    with pytest.raises(ValueError, match="outside the forest span index"):
        forest.record_pruned((1, 3), 0.0)


def test_pruned_records_survive_a_pickle_round_trip() -> None:
    forest = PackedForest[str, float](SpanIndex(1), Viterbi())
    forest.add(ItemKey((0, 1), "N", "lex"), Edge("lex", (), "N", -1.0))
    forest.record_pruned((0, 1), -2.0)
    forest.record_pruned_mass(PrunedMass(log(0.5), log(0.25)))

    restored = pickle.loads(pickle.dumps(forest))

    assert dict(restored.pruned) == {(0, 1): -2.0}
    assert restored.pruned_mass == PrunedMass(log(0.5), log(0.25))
    assert isclose(restored.pruned_mass.lost, 0.25)
    assert isclose(restored.pruned_mass.lost_fraction, 0.5)


def chained() -> PackedForest[str, float]:
    forest = PackedForest[str, float](SpanIndex(1), LogInside())
    lexical = ItemKey((0, 1), "N", "lex")
    raised = ItemKey((0, 1), "NP", "tr")
    twice = ItemKey((0, 1), "S/(S\\NP)", "tr")

    forest.add(lexical, Edge("lex", (), "N", log(0.5)))
    forest.add(raised, Edge("tc", (lexical,), "NP", log(0.5)))
    forest.add(twice, Edge("tr", (raised,), "S/(S\\NP)", log(0.5)))

    return forest


def test_restricting_a_cell_drops_the_items_it_does_not_name() -> None:
    forest = filled()

    forest.restrict((1, 2), (NOUN,))

    assert forest.items_at((1, 2)) == (NOUN,)
    assert forest.num_items == 3
    assert forest.truncated


def test_a_wider_span_reaches_only_what_the_restriction_kept() -> None:
    forest = PackedForest[str, float](SpanIndex(2), LogInside())
    forest.add(DET, Edge("lex", (), "NP/N", log(0.5)))
    forest.add(NOUN, Edge("lex", (), "N", log(1.0)))
    forest.add(NAME, Edge("lex", (), "NP", log(0.25)))

    forest.restrict((1, 2), (NOUN,))

    for right in forest.items_at((1, 2)):
        forest.add(PHRASE, Edge("fa", (DET, right), "NP", log(0.5), split=1))

    assert tuple(edge.rule_name for edge in forest.edges(PHRASE)) == ("fa",)
    assert isclose(exp(forest.inside(PHRASE)), 0.5 * 0.5 * 1.0)


def test_restricting_a_cell_to_everything_it_holds_changes_nothing() -> None:
    forest = filled()
    before = exp(forest.inside(PHRASE))

    forest.restrict((0, 2), (PHRASE,))
    forest.restrict((1, 2), (NOUN, NAME))

    assert forest.num_items == 4
    assert forest.num_edges == 5
    assert isclose(exp(forest.inside(PHRASE)), before)
    assert not forest.truncated


def test_an_item_left_with_no_step_goes_with_the_premise_it_lost() -> None:
    forest = chained()
    lexical, raised, twice = forest.items_at((0, 1))

    forest.restrict((0, 1), (raised, twice))

    assert forest.items_at((0, 1)) == ()
    assert forest.num_items == 0
    assert forest.num_edges == 0
    assert forest.edges(lexical) == ()


def test_a_cell_keeps_an_item_whose_premise_survived_the_restriction() -> None:
    forest = chained()
    lexical, raised, twice = forest.items_at((0, 1))

    forest.restrict((0, 1), (lexical, raised))

    assert forest.items_at((0, 1)) == (lexical, raised)
    assert isclose(exp(forest.inside(raised)), 0.25)
    assert forest.edges(twice) == ()


def test_restricting_refuses_an_item_the_span_does_not_hold() -> None:
    forest = filled()

    with pytest.raises(ValueError, match="holds no item"):
        forest.restrict((1, 2), (NOUN, PHRASE))

    assert forest.items_at((1, 2)) == (NOUN, NAME)
    assert not forest.truncated


def test_restricting_a_span_the_forest_never_reached_changes_nothing() -> None:
    forest = PackedForest[str, float](SpanIndex(2), LogInside())

    forest.restrict((0, 2), ())

    assert forest.num_items == 0
    assert not forest.truncated
