import math

from lofi import deduce
from lofi.terms import TargetSet


def test_the_published_surface_indexes_spans() -> None:
    index = deduce.SpanIndex(3)
    goal: deduce.Cell = (0, 3)

    assert index.num_spans == 6
    assert index.span_of(index.span_id(*goal)) == goal
    assert list(index.binary_sweep())[-1] == (goal, (0, 2), (2, 3))


def test_the_published_surface_exports_a_semiring() -> None:
    semiring = deduce.LogInside()

    assert isinstance(semiring, deduce.Semiring)
    assert semiring.times(math.log(0.5), math.log(0.5)) == math.log(0.5) * 2


def test_the_published_surface_exports_every_algebra() -> None:
    boolean = deduce.Boolean()
    log_inside = deduce.LogInside()
    viterbi = deduce.Viterbi()
    counting = deduce.Counting()
    k_best = deduce.KBest(2)
    forest = deduce.Forest[str]()
    expectation = deduce.Expectation(1)

    assert boolean.plus(boolean.zero, boolean.one) is True
    assert viterbi.plus(-1.0, -2.0) == -1.0
    assert counting.plus(counting.one, counting.one) == 2
    assert k_best.plus((-1.0,), (-2.0,)) == (-1.0, -2.0)
    assert forest.times(forest.one, frozenset({("a",)})) == frozenset({("a",)})
    assert expectation.plus(expectation.one, expectation.one) == (2.0, (0.0,))

    assert (
        boolean.name,
        log_inside.name,
        viterbi.name,
        counting.name,
        k_best.name,
        forest.name,
        expectation.name,
    ) == (
        "boolean",
        "log-inside",
        "viterbi",
        "counting",
        "k-best",
        "forest",
        "expectation",
    )


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


def test_the_published_surface_packs_a_forest_and_reaches_a_goal() -> None:
    erasure = Erasure()
    chart = deduce.PackedForest[str, float](deduce.SpanIndex(2), deduce.LogInside())
    left = deduce.ItemKey((0, 1), "S[dcl]/NP")
    right = deduce.ItemKey((1, 2), "NP")
    top = deduce.ItemKey((0, 2), "S[dcl]")

    chart.add(left, deduce.Edge("lex", (), "S[dcl]/NP", math.log(0.5)))
    chart.add(right, deduce.Edge("lex", (), "NP", math.log(0.5)))
    chart.add(top, deduce.Edge("fa", (left, right), "S[dcl]", math.log(0.5), split=1))

    targets = TargetSet(("S",), erasure)
    root = deduce.RootGoal(targets)
    cover = deduce.CoverGoal(2, penalty=1.0)

    assert isinstance(root, deduce.Goal)
    assert isinstance(cover, deduce.Goal)
    assert tuple(root.roots(chart)) == (top,)
    assert cover.fragments(chart) == ((0, 2),)
    assert math.isclose(math.exp(chart.inside(top)), 0.125)
    assert next(chart.derivations(top)).edge.rule_name == "fa"

    outside = deduce.outside_from_edges(chart, root)

    assert outside.constituent_marginal(top) == 1.0
    assert deduce.target_mass(chart, outside, targets, (0, 2)) == 1.0


def test_the_published_surface_exports_split_scorers() -> None:
    weights = deduce.SplitLogWeights({(0, 3): {1: math.log(0.25), 2: math.log(0.75)}})

    assert weights.spans == ((0, 3),)
    assert deduce.SplitProbabilities({(0, 2): {1: 1.0}}).probability((0, 2), 1) == 1.0
    assert deduce.LocalAttendScorer
    assert deduce.MarginalScorer
    assert repr(deduce.SpanFree()) == "SpanFree()"
    assert repr(deduce.SpanDependent()) == "SpanDependent()"
    assert repr(deduce.UnitEdges()) == "UnitEdges()"


def test_the_published_surface_exports_astar_search() -> None:
    assert deduce.AStarContext
    assert deduce.AStarOptions
    assert deduce.AStarResult
    assert deduce.astar_parse
    assert deduce.supertag_outside


def test_the_published_surface_exports_derivation_count_reports() -> None:
    assert deduce.DerivationCountRow
    assert deduce.derivation_count_report


def test_the_published_surface_beams_a_cell() -> None:
    pruner = deduce.TopKPruner[str](1)
    forest = deduce.PackedForest[str, float](deduce.SpanIndex(2), deduce.Viterbi())
    lighter = deduce.ItemKey((0, 2), "S", "step")
    heavier = deduce.ItemKey((0, 2), "NP", "step")

    forest.add(lighter, deduce.Edge("fa", (), "S", math.log(0.25)))
    forest.add(heavier, deduce.Edge("fa", (), "NP", math.log(0.75)))
    cell = tuple((key, forest.inside(key)) for key in forest.items_at((0, 2)))

    assert isinstance(pruner, deduce.CellPruner)
    assert pruner.keep((0, 2), cell) == (heavier,)

    forest.restrict((0, 2), pruner.keep((0, 2), cell))

    assert forest.items_at((0, 2)) == (heavier,)
    assert forest.truncated
