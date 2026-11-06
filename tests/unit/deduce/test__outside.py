"""Tests for packed-forest inside-outside inference."""

import copy
import pickle
from math import exp, isclose, log
from typing import TYPE_CHECKING, Final, override

import pytest
from hypothesis import given, seed, settings
from hypothesis import strategies as st
from tests.conformance._suite import Conformance, ConformanceSuite

from lofi._native import PythonCarrier
from lofi.deduce import (
    CoverGoal,
    Edge,
    EdgeProduct,
    IndependentRuleWeights,
    ItemKey,
    LogInside,
    MarginalScorer,
    Outside,
    PackedForest,
    RootGoal,
    SharedRuleWeights,
    SpanIndex,
    SplitLogWeights,
    Viterbi,
    goal_partition,
    outside_from_edges,
    target_mass,
)
from lofi.terms import TargetSet

if TYPE_CHECKING:
    from lofi.deduce import Chart, OutsideRuleSource
    from lofi.terms import Matching


class Identity[T]:
    """Direct category-string comparison."""

    @property
    def name(self) -> str:
        """Return the relation's name."""
        return "identity"

    @property
    def has_key(self) -> bool:
        """Return whether every class has a representative."""
        return True

    def key(self, term: T, /) -> T:
        """Return the category itself."""
        return term

    def equivalent(self, left: T, right: T, /) -> bool:
        """Return whether two categories are equal."""
        return left == right


class FeatureErasure(Identity[str]):
    """String matching after bracketed feature removal."""

    @property
    @override
    def name(self) -> str:
        """Return the relation's name."""
        return "feature-erasure"

    @override
    def key(self, term: str, /) -> str:
        """Return the category without a bracketed feature."""
        head, bracket, _ = term.partition("[")

        return head if bracket else term

    @override
    def equivalent(self, left: str, right: str, /) -> bool:
        """Return whether categories match after feature removal."""
        return self.key(left) == self.key(right)


class RejectRule(RootGoal[str]):
    """Root goal excluding one rule name."""

    __slots__ = ("_rejected",)

    def __init__(self, rejected: str) -> None:
        super().__init__(TargetSet(("S",), IDENTITY))
        object.__setattr__(self, "_rejected", rejected)

    @override
    def admits_edge[W](
        self,
        chart: Chart[str, W],
        parent: ItemKey[str],
        edge: Edge[str, W],
        /,
    ) -> bool:
        """Reject edges with the configured rule name."""
        del chart, parent

        return edge.rule_name != self._rejected


class WrongAlgebra:
    """Log operations under a declared name and log-probability capability."""

    __slots__ = ("_log_probability", "_name")

    def __init__(self, name: str = "wrong", *, log_probability: bool = False) -> None:
        self._name = name
        self._log_probability = log_probability

    @property
    def name(self) -> str:
        """Return the declared algebra name."""
        return self._name

    @property
    def log_probability(self) -> bool:
        """Report the declared log-probability capability."""
        return self._log_probability

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
        """Return whether the weight is log zero."""
        return LogInside().is_zero(a)


IDENTITY: Final[Matching[str]] = Identity()
INTEGER_IDENTITY: Final[Matching[int]] = Identity()
TOLERANCE: Final = 1e-9
A = ItemKey((0, 1), "A", "lex")
B = ItemKey((1, 2), "B", "lex")
C = ItemKey((2, 3), "C", "lex")
AB = ItemKey((0, 2), "AB", "step")
BC = ItemKey((1, 3), "BC", "step")
ROOT = ItemKey((0, 3), "S", "step")


def ambiguous(
    c_weight: float = 0.7, bc_weight: float = 0.2
) -> PackedForest[str, float]:
    """Build a hand-checkable three-token forest."""
    forest = PackedForest[str, float](SpanIndex(3), LogInside())
    forest.add(A, Edge("a", (), "A", log(0.5)))
    forest.add(B, Edge("b", (), "B", log(0.6)))
    forest.add(C, Edge("c", (), "C", log(c_weight)))
    forest.add(AB, Edge("ab", (A, B), "AB", log(0.4), split=1))
    forest.add(BC, Edge("bc", (B, C), "BC", log(bc_weight), split=2))
    forest.add(ROOT, Edge("root-left", (AB, C), "S", log(0.3), split=2))
    forest.add(ROOT, Edge("root-right", (A, BC), "S", log(0.1), split=1))

    return forest


def snapshot(
    forest: PackedForest[str, float],
) -> tuple[tuple[ItemKey[str], float, tuple[Edge[str, float], ...]], ...]:
    """Capture every item, inside weight, and edge."""
    return tuple(
        (key, forest.inside(key), forest.edges(key))
        for span in forest.index.spans_by_length()
        for key in forest.items_at(span)
    )


def test_hand_checked_inside_outside_values_and_marginals() -> None:
    forest = ambiguous()
    result = outside_from_edges(forest, RootGoal(TargetSet(("S",), IDENTITY)))
    left_mass = 0.3 * 0.4 * 0.5 * 0.6 * 0.7
    right_mass = 0.1 * 0.5 * 0.2 * 0.6 * 0.7
    partition = left_mass + right_mass

    assert isinstance(result, Outside)
    assert exp(result.partition) == pytest.approx(
        partition, rel=TOLERANCE, abs=TOLERANCE
    )
    assert exp(result.inside(AB)) == pytest.approx(
        0.4 * 0.5 * 0.6, rel=TOLERANCE, abs=TOLERANCE
    )
    assert exp(result.outside(AB)) == pytest.approx(
        0.3 * 0.7, rel=TOLERANCE, abs=TOLERANCE
    )
    assert exp(result.outside(BC)) == pytest.approx(
        0.1 * 0.5, rel=TOLERANCE, abs=TOLERANCE
    )
    assert exp(result.outside(A)) == pytest.approx(0.0588, rel=TOLERANCE, abs=TOLERANCE)
    assert exp(result.outside(B)) == pytest.approx(0.049, rel=TOLERANCE, abs=TOLERANCE)
    assert exp(result.outside(C)) == pytest.approx(0.042, rel=TOLERANCE, abs=TOLERANCE)
    assert result.constituent_marginal(ROOT) == pytest.approx(
        1.0, rel=TOLERANCE, abs=TOLERANCE
    )
    assert result.constituent_marginal(AB) == pytest.approx(
        left_mass / partition, rel=TOLERANCE, abs=TOLERANCE
    )
    assert result.constituent_marginal(BC) == pytest.approx(
        right_mass / partition, rel=TOLERANCE, abs=TOLERANCE
    )
    assert result.rule_marginal(ROOT, forest.edges(ROOT)[0]) == pytest.approx(
        left_mass / partition, rel=TOLERANCE, abs=TOLERANCE
    )
    assert result.rule_marginal(ROOT, forest.edges(ROOT)[1]) == pytest.approx(
        right_mass / partition, rel=TOLERANCE, abs=TOLERANCE
    )


def test_product_outside_matches_flat_edges_without_expanding_storage() -> None:
    semiring = LogInside()
    left = (
        ItemKey((0, 1), "A-1", "lex"),
        ItemKey((0, 1), "A-2", "lex"),
    )
    right = (
        ItemKey((1, 2), "B-1", "lex"),
        ItemKey((1, 2), "B-2", "lex"),
    )
    root = ItemKey((0, 2), "S", "step")
    product = PackedForest[str, float](SpanIndex(2), semiring)
    flat = PackedForest[str, float](SpanIndex(2), semiring)

    for key, weight in zip((*left, *right), (0.2, 0.3, 0.4, 0.5), strict=True):
        edge = Edge(key.reading.lower(), (), key.reading, log(weight))
        product.add(key, edge)
        flat.add(key, edge)

    alternative = EdgeProduct(
        "combine",
        (left, right),
        "S",
        log(0.7),
        split=1,
    )
    product.add_product(root, alternative)

    for edge in alternative.edges():
        flat.add(root, edge)

    product_result = outside_from_edges(
        product,
        RootGoal(TargetSet(("S",), IDENTITY)),
    )
    flat_result = outside_from_edges(flat, RootGoal(TargetSet(("S",), IDENTITY)))

    assert product.products(root) == (alternative,)
    assert product.edges(root) == flat.edges(root)
    rebuilt = pickle.loads(pickle.dumps(product))
    assert rebuilt.products(root) == (alternative,)
    assert product_result.partition == pytest.approx(flat_result.partition)
    for key in (*left, *right, root):
        assert product_result.inside(key) == pytest.approx(flat_result.inside(key))
        assert product_result.outside(key) == pytest.approx(flat_result.outside(key))
    product_marginals = product_result.rule_marginals(root)
    flat_marginals = flat_result.rule_marginals(root)
    assert tuple(edge for edge, _ in product_marginals) == tuple(
        edge for edge, _ in flat_marginals
    )
    assert tuple(weight for _, weight in product_marginals) == pytest.approx(
        tuple(weight for _, weight in flat_marginals)
    )


def test_target_mass_is_one_for_an_unambiguous_matching_root() -> None:
    forest = PackedForest[str, float](SpanIndex(2), LogInside())
    left = ItemKey((0, 1), "A", "lex")
    right = ItemKey((1, 2), "B", "lex")
    root = ItemKey((0, 2), "S", "step")
    forest.add(left, Edge("a", (), "A", log(0.5)))
    forest.add(right, Edge("b", (), "B", log(0.5)))
    forest.add(root, Edge("combine", (left, right), "S", 0.0, split=1))
    roots = TargetSet(("S",), IDENTITY)
    result = outside_from_edges(forest, RootGoal(roots))

    assert len(tuple(forest.derivations(root))) == 1
    assert target_mass(forest, result, roots, (0, 2)) == pytest.approx(
        1.0,
        rel=TOLERANCE,
        abs=TOLERANCE,
    )
    assert target_mass(
        forest,
        result,
        TargetSet(("NP",), IDENTITY),
        (0, 2),
    ) == pytest.approx(0.0, abs=TOLERANCE)


def test_root_category_weights_enter_the_partition_and_root_marginals() -> None:
    forest = PackedForest[str, float](SpanIndex(1), LogInside())
    sentence = ItemKey((0, 1), "S", "lex")
    fragment = ItemKey((0, 1), "NP", "lex")
    forest.add(sentence, Edge("s", (), "S", log(0.5)))
    forest.add(fragment, Edge("np", (), "NP", log(0.5)))
    goal = RootGoal(TargetSet(("S", "NP"), IDENTITY), (0.0, log(0.5)))
    result = outside_from_edges(forest, goal)

    assert exp(result.partition) == pytest.approx(0.5 + 0.25)
    assert result.constituent_marginal(sentence) == pytest.approx(2.0 / 3.0)
    assert result.constituent_marginal(fragment) == pytest.approx(1.0 / 3.0)


def test_the_goal_partition_is_the_partition_an_outside_pass_reports() -> None:
    forest = PackedForest[str, float](SpanIndex(2), LogInside())
    first = ItemKey((0, 1), "A", "lex")
    second = ItemKey((1, 2), "B", "lex")
    forest.add(first, Edge("a", (), "A", log(0.5)))
    forest.add(second, Edge("b", (), "B", log(0.2)))
    goals = (CoverGoal(2, penalty=1.0), RootGoal(TargetSet(("S",), IDENTITY)))

    for goal in goals:
        assert (
            goal_partition(forest, goal) == outside_from_edges(forest, goal).partition
        )

    with pytest.raises(ValueError, match="log-probability weights"):
        goal_partition(PackedForest[str, float](SpanIndex(1), Viterbi()), goals[1])


def test_target_mass_uses_the_target_sets_matching() -> None:
    forest = PackedForest[str, float](SpanIndex(1), LogInside())
    featured = ItemKey((0, 1), "S[dcl]", "lex")
    forest.add(featured, Edge("lex", (), "S[dcl]", 0.0))
    targets = TargetSet(("S",), FeatureErasure())
    result = outside_from_edges(forest, RootGoal(targets))

    assert target_mass(forest, result, targets, (0, 1)) == pytest.approx(
        1.0,
        rel=TOLERANCE,
        abs=TOLERANCE,
    )


def test_target_mass_at_width_one_is_the_conditional_item_mass() -> None:
    forest = PackedForest[str, float](SpanIndex(1), LogInside())
    p = ItemKey((0, 1), "P", "lex")
    q = ItemKey((0, 1), "Q", "lex")
    forest.add(p, Edge("p", (), "P", log(0.25)))
    forest.add(q, Edge("q", (), "Q", log(0.75)))
    result = outside_from_edges(
        forest,
        RootGoal(TargetSet(("P", "Q"), IDENTITY)),
    )

    assert result.constituent_marginal(p) + result.constituent_marginal(
        q
    ) == pytest.approx(1.0, rel=TOLERANCE, abs=TOLERANCE)
    assert target_mass(
        forest,
        result,
        TargetSet(("P",), IDENTITY),
        (0, 1),
    ) == pytest.approx(0.25, rel=TOLERANCE, abs=TOLERANCE)


def interior_alternatives() -> PackedForest[str, float]:
    """Build a three-token forest with two competing interior spans."""
    forest = PackedForest[str, float](SpanIndex(3), LogInside())
    x = ItemKey((0, 2), "X", "step")
    y = ItemKey((0, 2), "Y", "step")
    z = ItemKey((1, 3), "Z", "step")

    for key in (A, B, C):
        forest.add(key, Edge(key.reading.lower(), (), key.reading, 0.0))

    forest.add(x, Edge("x", (A, B), "X", log(0.2), split=1))
    forest.add(y, Edge("y", (A, B), "Y", log(0.3), split=1))
    forest.add(z, Edge("z", (B, C), "Z", log(0.5), split=2))
    forest.add(ROOT, Edge("root-x", (x, C), "S", 0.0, split=2))
    forest.add(ROOT, Edge("root-y", (y, C), "S", 0.0, split=2))
    forest.add(ROOT, Edge("root-z", (A, z), "S", 0.0, split=1))

    return forest


def test_target_mass_factors_constituency_and_conditional_interior_mass() -> None:
    forest = interior_alternatives()
    result = outside_from_edges(
        forest,
        RootGoal(TargetSet(("S",), IDENTITY)),
    )
    x, y = forest.items_at((0, 2))
    constituent = sum(result.constituent_marginal(key) for key in (x, y))
    conditional = result.constituent_marginal(x) / constituent

    assert constituent == pytest.approx(0.5, rel=TOLERANCE, abs=TOLERANCE)
    assert conditional == pytest.approx(0.4, rel=TOLERANCE, abs=TOLERANCE)
    assert target_mass(
        forest,
        result,
        TargetSet(("X",), IDENTITY),
        (0, 2),
    ) == pytest.approx(
        constituent * conditional,
        rel=TOLERANCE,
        abs=TOLERANCE,
    )


def test_target_mass_rejects_an_outside_result_for_another_chart() -> None:
    source = ambiguous()
    result = outside_from_edges(
        source,
        RootGoal(TargetSet(("S",), IDENTITY)),
    )

    with pytest.raises(ValueError, match="computed from another chart"):
        target_mass(
            ambiguous(),
            result,
            TargetSet(("S",), IDENTITY),
            (0, 3),
        )


@seed(20260914)
@settings(max_examples=100, deadline=None, database=None)
@given(st.lists(st.integers(1, 20), min_size=2, max_size=8).map(tuple))
def test_target_mass_is_monotone_under_generated_target_supersets(
    weights: tuple[int, ...],
) -> None:
    forest = PackedForest[int, float](SpanIndex(1), LogInside())
    categories = tuple(range(len(weights)))

    for category, weight in zip(categories, weights, strict=True):
        key = ItemKey((0, 1), category, "lex")
        forest.add(key, Edge("lex", (), category, log(weight)))

    result = outside_from_edges(
        forest,
        RootGoal(TargetSet(categories, INTEGER_IDENTITY)),
    )
    cutoff = max(1, len(categories) // 2)
    subset = TargetSet(categories[:cutoff], INTEGER_IDENTITY)
    superset = TargetSet(categories[: cutoff + 1], INTEGER_IDENTITY)

    assert target_mass(forest, result, subset, (0, 1)) <= target_mass(
        forest,
        result,
        superset,
        (0, 1),
    )


def test_unambiguous_nonlexical_marginals_sum_to_n_minus_one() -> None:
    forest = ambiguous()
    rejected = RejectRule("root-right")
    result = outside_from_edges(forest, rejected)

    assert sum(
        result.constituent_marginal(key) for key in (AB, BC, ROOT)
    ) == pytest.approx(forest.index.n - 1, rel=TOLERANCE, abs=TOLERANCE)


def test_a_childs_inside_weight_does_not_enter_its_own_outside_value() -> None:
    first = ambiguous()
    second = ambiguous()
    second.add(AB, Edge("extra-ab", (A, B), "AB", log(0.8), split=1))
    goal = RootGoal(TargetSet(("S",), IDENTITY))

    first_result = outside_from_edges(first, goal)
    second_result = outside_from_edges(second, goal)

    assert first_result.inside(AB) != second_result.inside(AB)
    assert first_result.outside(AB) == second_result.outside(AB)


def test_edge_admission_changes_results_without_mutating_the_forest() -> None:
    forest = ambiguous()
    before = snapshot(forest)
    before_bytes = pickle.dumps(forest)
    unrestricted = outside_from_edges(
        forest,
        RootGoal(TargetSet(("S",), IDENTITY)),
    )
    restricted = outside_from_edges(forest, RejectRule("root-right"))

    assert exp(restricted.partition) < exp(unrestricted.partition)
    assert restricted.constituent_marginal(BC) == 0.0
    assert restricted.rule_marginal(ROOT, forest.edges(ROOT)[1]) == 0.0
    assert snapshot(forest) == before
    assert pickle.dumps(forest) == before_bytes


def test_admitted_inside_weights_do_not_reuse_unconstrained_chart_values() -> None:
    forest = ambiguous()
    restricted = outside_from_edges(forest, RejectRule("ab"))
    right_mass = 0.1 * 0.5 * 0.2 * 0.6 * 0.7

    assert forest.inside(AB) != -float("inf")
    assert restricted.inside(AB) == -float("inf")
    assert exp(restricted.partition) == pytest.approx(
        right_mass, rel=TOLERANCE, abs=TOLERANCE
    )
    assert restricted.constituent_marginal(AB) == 0.0


def test_empty_and_wrong_semiring_forests_are_reported() -> None:
    empty = PackedForest[str, float](SpanIndex(1), LogInside())
    result = outside_from_edges(empty, RootGoal(TargetSet(("S",), IDENTITY)))

    assert result.partition == -float("inf")
    assert result.constituent_marginal(ROOT) == 0.0
    assert repr(result) == "Outside(partition=-inf, items=0)"

    wrong = PackedForest[str, float](SpanIndex(1), WrongAlgebra())

    with pytest.raises(ValueError, match="require log-probability weights"):
        outside_from_edges(wrong, RootGoal(TargetSet(("S",), IDENTITY)))


def test_outside_reads_the_declared_capability_rather_than_the_name() -> None:
    goal = RootGoal(TargetSet(("S",), IDENTITY))
    renamed = PackedForest[str, float](
        SpanIndex(1), WrongAlgebra("tempered", log_probability=True)
    )
    renamed.add(ItemKey((0, 1), "S", "lex"), Edge("s", (), "S", log(0.5)))
    misnamed = PackedForest[str, float](SpanIndex(1), WrongAlgebra("log-inside"))

    assert outside_from_edges(renamed, goal).partition == pytest.approx(log(0.5))

    with pytest.raises(ValueError, match="require log-probability weights"):
        outside_from_edges(misnamed, goal)


def test_an_unadmitted_edge_has_zero_rule_marginal() -> None:
    forest = ambiguous()
    result = outside_from_edges(forest, RootGoal(TargetSet(("S",), IDENTITY)))
    unknown = Edge("unknown", (A, BC), "S", log(0.9), split=1)

    assert isclose(result.rule_marginal(ROOT, unknown), 0.0)


def test_cover_partition_multiplies_fragments_and_applies_the_penalty() -> None:
    forest = PackedForest[str, float](SpanIndex(2), LogInside())
    first = ItemKey((0, 1), "A", "lex")
    second = ItemKey((1, 2), "B", "lex")
    forest.add(first, Edge("a", (), "A", log(0.5)))
    forest.add(second, Edge("b", (), "B", log(0.2)))

    result = outside_from_edges(forest, CoverGoal(2, penalty=1.0))

    assert exp(result.partition) == pytest.approx(0.5 * 0.2 * exp(-1.0))
    assert result.constituent_marginal(first) == pytest.approx(1.0)
    assert result.constituent_marginal(second) == pytest.approx(1.0)
    assert exp(result.outside(first)) == pytest.approx(0.2 * exp(-1.0))
    assert exp(result.outside(second)) == pytest.approx(0.5 * exp(-1.0))


@pytest.mark.parametrize(
    "field",
    ["_admitted", "_chart", "_inside", "_outside", "_partition", "_semiring"],
)
def test_an_outside_result_refuses_private_slot_mutation_and_deletion(
    field: str,
) -> None:
    """Freeze outside-result fields after construction."""
    result = outside_from_edges(
        ambiguous(),
        RootGoal(TargetSet(("S",), IDENTITY)),
    )

    with pytest.raises(AttributeError, match="immutable"):
        result.__setattr__(field, 0.0)

    with pytest.raises(AttributeError, match="immutable"):
        result.__delattr__(field)


def test_an_outside_result_reconstructs_through_standard_copy_protocols() -> None:
    """Reconstruct an outside result without changing its values."""
    forest = ambiguous()
    result = outside_from_edges(forest, RootGoal(TargetSet(("S",), IDENTITY)))
    shallow = copy.copy(result)
    deep = copy.deepcopy(result)
    restored = pickle.loads(pickle.dumps(result))

    assert result.is_for(forest)
    assert shallow.is_for(forest)
    assert deep.is_for(forest)
    assert not result.is_for(ambiguous())

    for rebuilt in (shallow, deep, restored):
        assert rebuilt.partition == result.partition

        for key in (A, B, C, AB, BC, ROOT):
            assert rebuilt.inside(key) == result.inside(key)
            assert rebuilt.outside(key) == result.outside(key)
            assert rebuilt.constituent_marginal(key) == pytest.approx(
                result.constituent_marginal(key)
            )

    source_fields = result.__reduce__()[1][0]

    for rebuilt in (shallow, deep):
        rebuilt_fields = rebuilt.__reduce__()[1][0]
        source_chart, source_semiring, _, source_inside, _, source_admitted = (
            source_fields
        )
        rebuilt_chart, rebuilt_semiring, _, rebuilt_inside, _, rebuilt_admitted = (
            rebuilt_fields
        )

        assert rebuilt_chart is source_chart
        assert rebuilt_semiring is source_semiring
        assert all(
            new[0] is old[0]
            for new, old in zip(rebuilt_inside, source_inside, strict=True)
        )
        assert all(
            new_edges is old_edges
            for (_, new_edges), (_, old_edges) in zip(
                rebuilt_admitted,
                source_admitted,
                strict=True,
            )
        )

    _, _, _, restored_inside, restored_outside, restored_admitted = (
        restored.__reduce__()[1][0]
    )
    restored_keys = {key: key for key, _ in restored_inside}

    assert all(restored_keys[key] is key for key, _ in restored_outside)
    assert all(restored_keys[key] is key for key, _ in restored_admitted)
    assert all(
        restored_keys[child] is child
        for _, products in restored_admitted
        for product in products
        for axis in product.axes
        for child in axis
    )


def test_an_outside_result_enumerates_only_admitted_rule_marginals() -> None:
    forest = ambiguous()
    result = outside_from_edges(forest, RejectRule("root-right"))

    assert result.rule_marginals(ROOT) == (
        (
            forest.edges(ROOT)[0],
            result.rule_marginal(ROOT, forest.edges(ROOT)[0]),
        ),
    )


def test_intersection_keeps_the_inside_chart_order() -> None:
    forest = PackedForest[int, float](SpanIndex(1), LogInside())
    keys = tuple(ItemKey((0, 1), reading, "lex") for reading in (3, 1, 2))

    for key in keys:
        forest.add(key, Edge("lex", (), key.reading, 0.0))

    result = outside_from_edges(
        forest,
        RootGoal(TargetSet((2, 3), INTEGER_IDENTITY)),
    )

    assert result.intersection((0, 1)) == (keys[0], keys[2])


SENTENCE: Final = RootGoal(TargetSet(("S",), IDENTITY))
ATTENTION: Final = SplitLogWeights(
    {
        (0, 2): {1: 0.0},
        (1, 3): {2: 0.0},
        (0, 3): {1: log(0.25), 2: log(0.75)},
    }
)
MOVED_CONTENT: Final = 0.9


def chart_derived_attention(forest: PackedForest[str, float]) -> SplitLogWeights:
    """Build split weights out of the forest's own inside-outside posteriors."""
    posteriors = MarginalScorer(forest, outside_from_edges(forest, SENTENCE))

    return SplitLogWeights(
        {
            span: {split: log(mass) for split, mass in row.items()}
            for span, row in posteriors.probabilities.rows.items()
        }
    )


def test_a_rule_source_declares_its_tag_and_where_its_weights_come_from() -> None:
    shared = SharedRuleWeights()
    independent = IndependentRuleWeights(ATTENTION)

    assert (shared.name, shared.chart_derived) == ("shared-with-inside", True)
    assert (independent.name, independent.chart_derived) == ("independent", False)
    assert independent.attention == ATTENTION


def test_the_two_rule_sources_differ_numerically_on_one_chart() -> None:
    forest = ambiguous()
    shared = outside_from_edges(forest, SENTENCE)
    independent = outside_from_edges(
        forest,
        SENTENCE,
        rule_source=IndependentRuleWeights(ATTENTION),
    )

    # the independent sweep reads 0.75 at the root's right split, not the 0.3
    # the inside pass recorded, and 0.25 at its left split, not 0.1
    assert exp(independent.outside(AB)) == pytest.approx(
        0.75 * 0.7, rel=TOLERANCE, abs=TOLERANCE
    )
    assert exp(independent.outside(BC)) == pytest.approx(
        0.25 * 0.5, rel=TOLERANCE, abs=TOLERANCE
    )
    assert exp(independent.outside(C)) == pytest.approx(
        0.75 * 0.4 * 0.5 * 0.6 + 0.25 * 0.5 * 1.0 * 0.6,
        rel=TOLERANCE,
        abs=TOLERANCE,
    )
    assert independent.outside(AB) != shared.outside(AB)
    assert independent.outside(BC) != shared.outside(BC)
    assert independent.outside(C) != shared.outside(C)
    assert independent.inside(AB) == shared.inside(AB)


@pytest.mark.parametrize(
    "source",
    [SharedRuleWeights(), IndependentRuleWeights(ATTENTION)],
    ids=lambda source: source.name,
)
def test_no_rule_source_carries_a_spans_own_inside_weight_into_its_outside(
    source: SharedRuleWeights | IndependentRuleWeights,
) -> None:
    base = outside_from_edges(ambiguous(), SENTENCE, rule_source=source)
    moved = outside_from_edges(ambiguous(MOVED_CONTENT), SENTENCE, rule_source=source)
    restructured = outside_from_edges(
        ambiguous(bc_weight=MOVED_CONTENT), SENTENCE, rule_source=source
    )

    assert moved.inside(C) != base.inside(C)
    assert moved.inside(BC) != base.inside(BC)
    assert moved.partition != base.partition
    assert moved.outside(C) == base.outside(C)
    assert moved.outside(BC) == base.outside(BC)
    assert restructured.inside(BC) != base.inside(BC)
    assert restructured.outside(BC) == base.outside(BC)


def test_weights_read_off_the_chart_do_carry_a_spans_inside_into_its_outside() -> None:
    settled = ambiguous()
    moved_forest = ambiguous(bc_weight=MOVED_CONTENT)
    settled_result = outside_from_edges(
        settled,
        SENTENCE,
        rule_source=IndependentRuleWeights(chart_derived_attention(settled)),
    )
    moved = outside_from_edges(
        moved_forest,
        SENTENCE,
        rule_source=IndependentRuleWeights(chart_derived_attention(moved_forest)),
    )
    held = outside_from_edges(
        moved_forest, SENTENCE, rule_source=IndependentRuleWeights(ATTENTION)
    )

    assert moved.outside(BC) != settled_result.outside(BC)
    assert held.outside(BC) == outside_from_edges(
        settled, SENTENCE, rule_source=IndependentRuleWeights(ATTENTION)
    ).outside(BC)


def test_an_axiom_weighs_what_the_chart_recorded_under_either_source() -> None:
    axiom = Edge[str, float]("c", (), "C", log(0.7))
    independent = IndependentRuleWeights(ATTENTION)

    assert independent.rule_weight(C, axiom) == log(0.7)
    assert SharedRuleWeights().rule_weight(C, axiom) == log(0.7)


def test_the_independent_source_refuses_a_step_with_no_split_point() -> None:
    unary = Edge[str, float]("raise", (A,), "AB", log(0.5))

    with pytest.raises(ValueError, match=r"'raise' at \(0, 2\) has no split point"):
        IndependentRuleWeights(ATTENTION).rule_weight(AB, unary)


def test_the_independent_source_refuses_an_undeclared_split() -> None:
    attention = SplitLogWeights({(0, 3): {1: log(0.25), 2: log(0.75)}})

    with pytest.raises(ValueError, match=r"no independent outside weight"):
        outside_from_edges(
            ambiguous(),
            SENTENCE,
            rule_source=IndependentRuleWeights(attention),
        )


def test_an_independent_source_round_trips_through_pickle_and_copy() -> None:
    source = IndependentRuleWeights(ATTENTION)
    restored = pickle.loads(pickle.dumps(source))

    assert restored == source
    assert copy.copy(source) == source
    assert copy.deepcopy(source) == source
    assert hash(restored) == hash(source)
    assert repr(source) == "IndependentRuleWeights(spans=3)"
    assert SharedRuleWeights() == pickle.loads(pickle.dumps(SharedRuleWeights()))

    with pytest.raises(AttributeError, match="immutable"):
        source.__setattr__("_attention", ATTENTION)

    with pytest.raises(AttributeError, match="immutable"):
        source.__delattr__("_attention")


ONE_SPLIT: Final = SplitLogWeights({(0, 2): {1: 0.0}})


class TestASharedRuleSource(ConformanceSuite):
    """Tests the shared rule source against the value laws."""

    @pytest.fixture
    def conformance(self) -> Conformance[OutsideRuleSource]:
        """Return the conformance the laws run the shared source through."""
        return Conformance(
            build=SharedRuleWeights,
            unequal=IndependentRuleWeights(ATTENTION),
            operations=(
                ("name", lambda source: source.name),
                ("chart_derived", lambda source: source.chart_derived),
                (
                    "rule_weight",
                    lambda source: source.rule_weight(
                        C, Edge[str, float]("c", (), "C", log(0.7))
                    ),
                ),
            ),
        )


class TestAnIndependentRuleSource(ConformanceSuite):
    """Tests the independent rule source against the value laws."""

    @pytest.fixture
    def conformance(self) -> Conformance[IndependentRuleWeights]:
        """Return the conformance the laws run the independent source through."""
        return Conformance(
            build=lambda: IndependentRuleWeights(ATTENTION),
            unequal=IndependentRuleWeights(ONE_SPLIT),
            operations=(
                ("name", lambda source: source.name),
                ("chart_derived", lambda source: source.chart_derived),
                ("attention", lambda source: source.attention.spans),
                (
                    "rule_weight",
                    lambda source: source.rule_weight(
                        ROOT,
                        Edge[str, float]("root-left", (AB, C), "S", log(0.3), split=2),
                    ),
                ),
            ),
        )
