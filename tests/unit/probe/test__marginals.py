from math import fsum, log
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import pytest
import torch
from scripts.fixtures.make_chart_fixture import (
    PinnedFingerprint,
    check_pinned_fixture,
    fixture_grammar,
    read_sentences,
)

from lofi._native import SpanFree
from lofi.deduce import (
    Edge,
    ItemKey,
    LogInside,
    PackedForest,
    ParsePolicy,
    Reachability,
    Retain,
    RootGoal,
    SpanIndex,
    outside_from_edges,
    parse,
)
from lofi.nn.sites import ComponentId, Role, Site
from lofi.probe._graph import ComputationGraph, GraphEdge, GraphNode
from lofi.probe._marginals import DerivationWeights, SpanAggregate, SpanWeighting
from lofi.terms import Identity, TargetSet

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from lofi.deduce import Cell, Chart, Outside
    from lofi.judge import Rule
    from lofi.lexicon import Entry
    from lofi.terms import Term

PARSER: Final = ComponentId("cky-parser")
FIXTURES: Final = Path(__file__).resolve().parents[3] / "tests/benchmarks/fixtures"

A = ItemKey((0, 1), "A", "lex")
B = ItemKey((1, 2), "B", "lex")
C = ItemKey((2, 3), "C", "lex")
AB = ItemKey((0, 2), "AB", "step")
BC = ItemKey((1, 3), "BC", "step")
ROOT = ItemKey((0, 3), "S", "step")

# the two bracketings of the three-token forest, and the mass on each
LEFT_MASS: Final = 0.3 * 0.4 * 0.5 * 0.6 * 0.7
RIGHT_MASS: Final = 0.1 * 0.5 * 0.2 * 0.6 * 0.7
LEFT_POSTERIOR: Final = LEFT_MASS / (LEFT_MASS + RIGHT_MASS)


def a_site(key: ItemKey[str]) -> Site:
    return Site(PARSER, Role.CHART_CELL, key.span)


def sites_of[T, W](chart: Chart[T, W]) -> dict[ItemKey[T], Site]:
    return {
        key: Site(PARSER, Role.CHART_CELL, key.span)
        for span in chart.index.spans_by_length()
        for key in chart.items_at(span)
    }


def ambiguous() -> PackedForest[str, float]:
    forest = PackedForest[str, float](SpanIndex(3), LogInside())
    forest.add(A, Edge("a", (), "A", log(0.5)))
    forest.add(B, Edge("b", (), "B", log(0.6)))
    forest.add(C, Edge("c", (), "C", log(0.7)))
    forest.add(AB, Edge("ab", (A, B), "AB", log(0.4), split=1))
    forest.add(BC, Edge("bc", (B, C), "BC", log(0.2), split=2))
    forest.add(ROOT, Edge("root-left", (AB, C), "S", log(0.3), split=2))
    forest.add(ROOT, Edge("root-right", (A, BC), "S", log(0.1), split=1))

    return forest


def a_goal() -> RootGoal[str]:
    return RootGoal(TargetSet(("S",), Identity[str]()))


def a_graph() -> ComputationGraph[str]:
    return ComputationGraph.from_forest(ambiguous(), sites_of(ambiguous()))


def node(key: ItemKey[str]) -> GraphNode[str]:
    return GraphNode(PARSER, key)


def known() -> Mapping[GraphNode[str], float]:
    return {
        node(A): 1.0,
        node(B): 1.0,
        node(C): 1.0,
        node(AB): LEFT_POSTERIOR,
        node(BC): 1.0 - LEFT_POSTERIOR,
        node(ROOT): 1.0,
    }


def sum_at[T](weights: DerivationWeights[T], spans: Iterator[Cell]) -> float:
    return fsum(weights.at(span) for span in spans)


def test_an_aggregate_holds_both_means_and_the_count() -> None:
    aggregate = SpanAggregate(1.5, 2.0, 4)

    assert aggregate.weighted == 1.5
    assert aggregate.uniform == 2.0
    assert aggregate.count == 4
    assert repr(aggregate) == "SpanAggregate(weighted=1.5, uniform=2.0, count=4)"


def test_an_aggregate_is_equal_on_its_two_means_and_its_count() -> None:
    aggregate = SpanAggregate(1.5, 2.0, 4)

    assert aggregate == SpanAggregate(1.5, 2.0, 4)
    assert hash(aggregate) == hash(SpanAggregate(1.5, 2.0, 4))
    assert aggregate != SpanAggregate(1.0, 2.0, 4)
    assert aggregate != SpanAggregate(1.5, 1.0, 4)
    assert aggregate != SpanAggregate(1.5, 2.0, 3)
    assert aggregate != (1.5, 2.0, 4)


def test_an_aggregate_refuses_a_mean_that_is_not_a_number_or_no_spans() -> None:
    with pytest.raises(ValueError, match="not finite"):
        SpanAggregate(float("nan"), 2.0, 4)

    with pytest.raises(TypeError, match="not a number"):
        SpanAggregate(1.5, cast("float", "2.0"), 4)

    with pytest.raises(ValueError, match="no spans"):
        SpanAggregate(1.5, 2.0, 0)

    with pytest.raises(ValueError, match="negative"):
        SpanAggregate(1.5, 2.0, -1)


def test_aggregating_over_weights_takes_the_weighted_and_the_uniform_mean() -> None:
    values = torch.tensor([[1.0, 3.0], [5.0, 7.0]])
    weights = torch.tensor([[1.0, 0.0], [2.0, 1.0]])
    aggregate = SpanAggregate.over(values, weights)

    assert aggregate.weighted == pytest.approx((1.0 + 10.0 + 7.0) / 4.0)
    assert aggregate.uniform == pytest.approx(4.0)
    assert aggregate.count == 4


def test_aggregating_leaves_out_masked_positions() -> None:
    values = torch.tensor([1.0, 3.0, 100.0])
    weights = torch.tensor([1.0, 3.0, 1.0])
    live = torch.tensor([True, True, False])
    aggregate = SpanAggregate.over(values, weights, live)

    assert aggregate.weighted == pytest.approx(2.5)
    assert aggregate.uniform == pytest.approx(2.0)
    assert aggregate.count == 2


def test_weights_that_do_not_cover_the_values_are_refused() -> None:
    values = torch.tensor([1.0, 3.0])

    with pytest.raises(ValueError, match="cannot weigh values"):
        SpanAggregate.over(values, torch.tensor([1.0, 1.0, 1.0]))

    with pytest.raises(ValueError, match="does not cover values"):
        SpanAggregate.over(values, torch.ones(2), torch.tensor([True]))


def test_aggregating_over_nothing_live_or_no_mass_is_refused() -> None:
    values = torch.tensor([1.0, 3.0])

    with pytest.raises(ValueError, match="nothing to aggregate"):
        SpanAggregate.over(values, torch.ones(2), torch.tensor([False, False]))

    with pytest.raises(ValueError, match="no mass"):
        SpanAggregate.over(values, torch.zeros(2))

    with pytest.raises(ValueError, match="negative or not finite"):
        SpanAggregate.over(values, torch.tensor([1.0, -1.0]))

    with pytest.raises(ValueError, match="negative or not finite"):
        SpanAggregate.over(values, torch.tensor([1.0, float("nan")]))


def test_the_weights_hold_one_marginal_per_step() -> None:
    graph = a_graph()
    weights = DerivationWeights(graph, known())

    assert weights.graph is graph
    assert weights.nodes == graph.nodes
    assert len(weights) == 6
    assert set(weights) == set(graph.nodes)
    assert dict(weights.items()) == dict(known())
    assert weights.of(node(AB)) == pytest.approx(LEFT_POSTERIOR)
    assert repr(weights) == f"DerivationWeights({graph!r}, steps=6)"


def test_a_step_without_a_marginal_or_a_marginal_without_a_step_is_refused() -> None:
    graph = a_graph()
    marginals = dict(known())
    stray = node(ItemKey((0, 3), "T", "step"))

    del marginals[node(BC)]

    with pytest.raises(ValueError, match="has no marginal"):
        DerivationWeights(graph, marginals)

    with pytest.raises(ValueError, match="not a node here"):
        DerivationWeights(graph, {**known(), stray: 0.5})


def test_a_marginal_that_is_not_a_probability_is_refused() -> None:
    graph = a_graph()

    with pytest.raises(ValueError, match="not a probability"):
        DerivationWeights(graph, {**known(), node(AB): 1.5})

    with pytest.raises(ValueError, match="not a probability"):
        DerivationWeights(graph, {**known(), node(AB): -0.5})

    with pytest.raises(ValueError, match="not finite"):
        DerivationWeights(graph, {**known(), node(AB): float("nan")})

    with pytest.raises(TypeError, match="not a number"):
        DerivationWeights(graph, {**known(), node(AB): cast("float", "0.5")})


def test_a_marginal_off_the_unit_interval_by_rounding_is_clamped() -> None:
    weights = DerivationWeights(
        a_graph(), {**known(), node(AB): 1.0 + 1e-12, node(BC): -1e-12}
    )

    assert weights.of(node(AB)) == 1.0
    assert weights.of(node(BC)) == 0.0


def test_the_weights_are_over_a_computation_graph() -> None:
    with pytest.raises(TypeError, match="computation graph"):
        DerivationWeights(cast("ComputationGraph[str]", ambiguous()), known())


def test_looking_up_a_step_the_graph_does_not_hold_says_so() -> None:
    weights = DerivationWeights(a_graph(), known())

    with pytest.raises(KeyError, match="not a step"):
        weights.of(node(ItemKey((0, 3), "T", "step")))


def test_the_marginals_at_a_span_sum_over_the_items_concluded_there() -> None:
    forest = ambiguous()
    forest.add(ItemKey((0, 3), "T", "step"), Edge("up", (ROOT,), "T", log(0.5)))
    graph = ComputationGraph.from_forest(forest, sites_of(forest))
    weights = DerivationWeights(
        graph, {**known(), node(ItemKey((0, 3), "T", "step")): 0.25}
    )

    assert weights.at((0, 3)) == pytest.approx(1.25)
    assert weights.at((0, 2)) == pytest.approx(LEFT_POSTERIOR)
    assert weights.at((2, 3)) == 1.0
    assert weights.at((0, 1)) == 1.0
    assert weights.at((1, 1)) == 0.0


def test_the_weights_are_equal_on_their_graph_and_their_marginals() -> None:
    weights = DerivationWeights(a_graph(), known())

    assert weights == DerivationWeights(a_graph(), known())
    assert hash(weights) == hash(DerivationWeights(a_graph(), known()))
    assert weights != DerivationWeights(a_graph(), {**known(), node(AB): 0.5})
    assert weights != known()


def test_the_weights_copy_the_marginals_they_were_given() -> None:
    marginals = dict(known())
    weights = DerivationWeights(a_graph(), marginals)

    marginals[node(AB)] = 0.0

    assert weights.of(node(AB)) == pytest.approx(LEFT_POSTERIOR)


def test_aggregating_under_the_marginals_differs_from_the_uniform_mean() -> None:
    weights = DerivationWeights(a_graph(), known())
    values = {node(AB): 10.0, node(BC): 0.0}
    aggregate = weights.aggregate(values)

    assert aggregate.uniform == pytest.approx(5.0)
    assert aggregate.weighted == pytest.approx(10.0 * LEFT_POSTERIOR)
    assert aggregate.count == 2
    assert aggregate.weighted != aggregate.uniform


def test_aggregating_over_no_steps_or_no_mass_is_refused() -> None:
    weights = DerivationWeights(a_graph(), {**known(), node(AB): 0.0, node(BC): 0.0})

    with pytest.raises(ValueError, match="no steps"):
        weights.aggregate({})

    with pytest.raises(ValueError, match="no mass"):
        weights.aggregate({node(AB): 1.0, node(BC): 2.0})

    with pytest.raises(ValueError, match="not finite"):
        weights.aggregate({node(ROOT): float("inf")})

    with pytest.raises(KeyError, match="not a step"):
        weights.aggregate({node(ItemKey((0, 3), "T", "step")): 1.0})


def test_the_marginals_of_a_known_posterior_are_read_off_the_outside_pass() -> None:
    forest = ambiguous()
    outside = outside_from_edges(forest, a_goal())
    weights = DerivationWeights.from_outside(
        ComputationGraph.from_forest(forest, sites_of(forest)), forest, outside
    )

    assert weights.of(node(AB)) == pytest.approx(LEFT_POSTERIOR)
    assert weights.of(node(BC)) == pytest.approx(1.0 - LEFT_POSTERIOR)
    assert weights.of(node(ROOT)) == pytest.approx(1.0)
    assert weights.of(node(A)) == pytest.approx(1.0)
    assert all(
        weights.of(step) == pytest.approx(outside.constituent_marginal(step.key))
        for step in weights
    )


def test_span_marginals_sum_to_one_less_than_the_length_over_proper_spans() -> None:
    forest = ambiguous()
    weights = DerivationWeights.from_outside(
        ComputationGraph.from_forest(forest, sites_of(forest)),
        forest,
        outside_from_edges(forest, a_goal()),
    )
    length = forest.index.n

    assert sum_at(weights, forest.index.spans_by_length(min_length=2)) == pytest.approx(
        length - 1
    )
    assert sum_at(weights, forest.index.spans_by_length()) == pytest.approx(
        2 * length - 1
    )


def test_an_outside_pass_over_another_chart_is_refused() -> None:
    forest = ambiguous()
    other = outside_from_edges(ambiguous(), a_goal())

    with pytest.raises(ValueError, match="another chart"):
        DerivationWeights.from_outside(
            ComputationGraph.from_forest(forest, sites_of(forest)), forest, other
        )


def test_a_graph_that_does_not_match_the_forest_is_refused() -> None:
    forest = ambiguous()
    outside = outside_from_edges(forest, a_goal())
    nodes = {key: node(key) for key in (A, B, C, AB, BC, ROOT)}
    edges = {
        GraphEdge(nodes[source], nodes[target], Role.CHART_CELL)
        for source, target in ((A, AB), (B, AB), (B, BC), (C, BC), (AB, ROOT))
    }
    fewer = ComputationGraph(
        [step for key, step in nodes.items() if key != BC],
        {edge for edge in edges if BC not in (edge.source.key, edge.target.key)},
    )
    stray = node(ItemKey((0, 3), "T", "step"))
    more = ComputationGraph([*nodes.values(), stray], edges)

    with pytest.raises(ValueError, match="no step of the graph concludes"):
        DerivationWeights.from_outside(fewer, forest, outside)

    with pytest.raises(ValueError, match="the forest does not hold"):
        DerivationWeights.from_outside(more, forest, outside)


class Unweighted[T]:
    lowering = SpanFree()

    def axiom_weight(self, position: int, entry: Entry[T], /) -> float:
        del position, entry

        return 0.0

    def edge_weight(
        self,
        rule: Rule[T],
        inputs: tuple[T, ...],
        output: T,
        span: Cell,
        split: int | None,
        /,
    ) -> float:
        del rule, inputs, output, span, split

        return 0.0


def pinned_weights() -> Iterator[
    tuple[
        PackedForest[Term, float],
        Outside[Term],
        DerivationWeights[Term],
    ]
]:
    sentences_path = FIXTURES / "chart_200.jsonl"
    fingerprint_path = FIXTURES / "chart_200.fingerprint"

    if not sentences_path.is_file() or not fingerprint_path.is_file():
        pytest.skip("the pinned chart fixture is absent")

    recorded = PinnedFingerprint.read(fingerprint_path)
    assembly = fixture_grammar(recorded.grammar).assemble()
    check_pinned_fixture(assembly.grammar, sentences_path, fingerprint_path)

    for sentence in read_sentences(sentences_path):
        forest = parse(
            sentence,
            rules=assembly.grammar.rules,
            source=assembly.lexicon,
            semiring=LogInside(),
            scorer=Unweighted(),
            goal=assembly.grammar.goal,
            unary_depth=assembly.grammar.budget.unary_rounds,
            policy=ParsePolicy(Retain.FOREST, Reachability.ALL),
        )
        outside = outside_from_edges(forest, assembly.grammar.goal)
        graph = ComputationGraph.from_forest(forest, sites_of(forest))

        yield forest, outside, DerivationWeights.from_outside(graph, forest, outside)


def test_every_pinned_chart_puts_one_less_than_its_length_on_proper_spans() -> None:
    parsed = 0

    for forest, outside, weights in pinned_weights():
        length = forest.index.n
        parsed += 1

        assert sum_at(
            weights, forest.index.spans_by_length(min_length=2)
        ) == pytest.approx(length - 1)
        assert sum_at(weights, forest.index.spans_by_length()) == pytest.approx(
            2 * length - 1
        )

        for step in weights:
            assert weights.of(step) == pytest.approx(
                outside.constituent_marginal(step.key)
            )

    assert parsed == 200


def test_a_uniform_weighting_over_the_pinned_charts_favors_disbelieved_spans() -> None:
    uniform_shares: list[float] = []
    marginal_shares: list[float] = []

    for forest, _, weights in pinned_weights():
        proper = tuple(
            span
            for span in forest.index.spans_by_length(min_length=2)
            if forest.items_at(span)
        )
        disbelieved = tuple(span for span in proper if weights.at(span) < 0.5)

        uniform_shares.append(len(disbelieved) / len(proper))
        marginal_shares.append(
            sum_at(weights, iter(disbelieved)) / (forest.index.n - 1)
        )

    assert fsum(uniform_shares) / len(uniform_shares) > fsum(marginal_shares) / len(
        marginal_shares
    )
    assert all(
        uniform >= marginal
        for uniform, marginal in zip(uniform_shares, marginal_shares, strict=True)
    )


def test_a_weighting_names_its_axes_and_holds_its_weights() -> None:
    weighting = SpanWeighting(torch.tensor([[0.5, 1.0]]), axes=("item", "span"))

    assert weighting.axes == ("item", "span")
    assert torch.equal(weighting.tensor, torch.tensor([[0.5, 1.0]]))
    assert repr(weighting) == "SpanWeighting(axes=('item', 'span'), shape=(1, 2))"


def test_a_weighting_is_detached_from_the_graph_it_was_read_off() -> None:
    read = torch.tensor([0.5, 1.0], requires_grad=True)
    weighting = SpanWeighting(read, axes=("span",))

    assert not weighting.tensor.requires_grad
    assert read.requires_grad


def test_a_weighting_refuses_a_negative_or_non_finite_weight_and_a_non_tensor() -> None:
    with pytest.raises(ValueError, match="negative or not finite"):
        SpanWeighting(torch.tensor([0.5, -1.0]), axes=("span",))

    with pytest.raises(ValueError, match="negative or not finite"):
        SpanWeighting(torch.tensor([0.5, float("inf")]), axes=("span",))

    with pytest.raises(TypeError, match="is a tensor"):
        SpanWeighting(cast("torch.Tensor", [0.5, 1.0]), axes=("span",))


def test_a_weighting_refuses_axis_names_that_do_not_cover_its_axes() -> None:
    weights = torch.ones(2, 3)

    with pytest.raises(ValueError, match="needs 2 axis names"):
        SpanWeighting(weights, axes=("item",))

    with pytest.raises(ValueError, match="declares an axis twice"):
        SpanWeighting(weights, axes=("item", "item"))

    with pytest.raises(ValueError, match="empty"):
        SpanWeighting(weights, axes=("item", ""))

    with pytest.raises(TypeError, match="not a string"):
        SpanWeighting(weights, axes=cast("tuple[str, str]", ("item", 1)))


def test_a_batch_of_weights_is_laid_out_by_item_and_span() -> None:
    forest = ambiguous()
    weights = DerivationWeights.from_outside(
        ComputationGraph.from_forest(forest, sites_of(forest)),
        forest,
        outside_from_edges(forest, a_goal()),
    )
    index = SpanIndex(4)
    weighting = SpanWeighting.from_weights(
        [weights, weights], index, axes=("item", "span")
    )
    expected = torch.zeros(index.num_spans, dtype=torch.float64)

    for span, marginal in (
        ((0, 1), 1.0),
        ((1, 2), 1.0),
        ((2, 3), 1.0),
        ((0, 2), LEFT_POSTERIOR),
        ((1, 3), 1.0 - LEFT_POSTERIOR),
        ((0, 3), 1.0),
    ):
        expected[index.span_id(*span)] = marginal

    assert weighting.axes == ("item", "span")
    assert tuple(weighting.tensor.shape) == (2, index.num_spans)
    assert torch.allclose(weighting.tensor[0], expected)
    assert torch.allclose(weighting.tensor[1], expected)
    assert weighting.tensor[0, index.span_id(0, 4)] == 0.0


def test_a_step_beyond_the_index_is_refused() -> None:
    weights = DerivationWeights(a_graph(), known())

    with pytest.raises(ValueError, match="does not address"):
        SpanWeighting.from_weights([weights], SpanIndex(2), axes=("item", "span"))


def test_aligned_broadcasts_the_weights_over_a_readings_trailing_axes() -> None:
    weighting = SpanWeighting(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), axes=("i", "s"))
    aligned = weighting.aligned(("i", "s", "width"), (2, 2, 3))

    assert tuple(aligned.shape) == (2, 2, 3)
    assert torch.equal(aligned[1, 0], torch.full((3,), 3.0))
    assert torch.equal(aligned[0, 1], torch.full((3,), 2.0))
    assert torch.equal(weighting.aligned(("i", "s"), (2, 2)), weighting.tensor)


def test_aligned_refuses_a_reading_whose_leading_axes_are_not_the_weightings() -> None:
    weighting = SpanWeighting(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), axes=("i", "s"))

    with pytest.raises(ValueError, match="cannot weigh a reading over"):
        weighting.aligned(("s", "i", "width"), (2, 2, 3))

    with pytest.raises(ValueError, match="cannot weigh a reading of shape"):
        weighting.aligned(("i", "s", "width"), (2, 3, 3))

    with pytest.raises(ValueError, match="cannot weigh a reading over"):
        weighting.aligned(("i",), (2,))
