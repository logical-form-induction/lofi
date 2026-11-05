from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scripts.fixtures.make_chart_fixture import (
    PinnedFingerprint,
    check_pinned_fixture,
    fixture_grammar,
    read_sentences,
)

from lofi._native import SpanFree
from lofi.deduce import (
    Counting,
    Edge,
    ItemKey,
    PackedForest,
    ParsePolicy,
    Reachability,
    Retain,
    SpanIndex,
    Viterbi,
    parse,
)
from lofi.nn.sites import ComponentId, Role, Site
from lofi.probe._graph import ComputationGraph, GraphEdge, GraphNode

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from lofi.deduce import Cell, Chart
    from lofi.judge import Rule
    from lofi.lexicon import Entry
    from lofi.terms import Term

PARSER = ComponentId("cky-parser")
FIXTURES: Final = Path(__file__).resolve().parents[3] / "tests/benchmarks/fixtures"

DET = ItemKey((0, 1), "NP/N", "lex")
NOUN = ItemKey((1, 2), "N", "lex")
NAME = ItemKey((1, 2), "NP", "lex")
PHRASE = ItemKey((0, 2), "NP", "step")
SENTENCE = ItemKey((0, 2), "S", "step")

LEFT = GraphNode(PARSER, DET)
RIGHT = GraphNode(PARSER, NOUN)
WHOLE = GraphNode(PARSER, PHRASE)
NODES = (LEFT, RIGHT, WHOLE)
EDGES = (
    GraphEdge(LEFT, WHOLE, Role.CHART_CELL),
    GraphEdge(RIGHT, WHOLE, Role.CHART_CELL),
)


def a_site(key: ItemKey[str]) -> Site:
    return Site(PARSER, Role.CHART_CELL, key.span)


def sites_of(chart: Chart[str, int]) -> dict[ItemKey[str], Site]:
    return {
        key: a_site(key)
        for span in chart.index.spans_by_length()
        for key in chart.items_at(span)
    }


def a_forest(*steps: tuple[ItemKey[str], Edge[str, int]]) -> PackedForest[str, int]:
    forest = PackedForest[str, int](SpanIndex(2), Counting())

    for key, edge in steps:
        forest.add(key, edge)

    return forest


def lexical(key: ItemKey[str]) -> tuple[ItemKey[str], Edge[str, int]]:
    return (key, Edge("lex", (), key.reading, 1))


def combined(
    key: ItemKey[str], rule: str, *children: ItemKey[str]
) -> tuple[ItemKey[str], Edge[str, int]]:
    return (key, Edge(rule, children, key.reading, 1, split=1))


def test_a_node_is_a_component_and_the_item_it_concluded() -> None:
    assert WHOLE.component == PARSER
    assert WHOLE.key == PHRASE
    assert WHOLE.span == (0, 2)


def test_a_node_concludes_an_item_and_not_a_bare_span() -> None:
    with pytest.raises(TypeError, match="concludes a forest item"):
        GraphNode(PARSER, cast("ItemKey[str]", (0, 2)))


def test_a_span_that_covers_nothing_is_refused() -> None:
    with pytest.raises(ValueError, match="covers nothing"):
        GraphNode(PARSER, ItemKey((2, 2), "N"))

    with pytest.raises(ValueError, match="covers nothing"):
        GraphNode(PARSER, ItemKey((3, 1), "N"))


def test_a_span_counting_from_the_end_is_refused() -> None:
    with pytest.raises(ValueError, match="span's start is negative"):
        GraphNode(PARSER, ItemKey((-1, 4), "N"))


def test_a_span_that_is_not_a_pair_is_refused() -> None:
    with pytest.raises(TypeError, match="a start and an end"):
        GraphNode(PARSER, ItemKey(cast("tuple[int, int]", (0, 2, 4)), "N"))


def test_a_node_names_the_component_that_took_the_step() -> None:
    with pytest.raises(ValueError, match="names the component"):
        GraphNode(ComponentId(""), PHRASE)


def test_nodes_naming_the_same_step_are_one_node() -> None:
    rebuilt = GraphNode(PARSER, ItemKey((0, 2), "NP", "step"))

    assert rebuilt == WHOLE
    assert hash(rebuilt) == hash(WHOLE)
    assert GraphNode(PARSER, SENTENCE) != WHOLE
    assert GraphNode(PARSER, ItemKey((0, 2), "NP", "other")) != WHOLE
    assert GraphNode(ComponentId("chart"), PHRASE) != WHOLE
    assert rebuilt != (PARSER, PHRASE)


def test_a_node_cannot_be_rebound() -> None:
    for held in ("component", "key", "span"):
        with pytest.raises(AttributeError):
            setattr(WHOLE, held, None)

    assert not hasattr(WHOLE, "__dict__")


def test_an_edge_says_which_step_fed_which_and_in_what_role() -> None:
    edge = EDGES[0]

    assert edge.source == LEFT
    assert edge.target == WHOLE
    assert edge.role is Role.CHART_CELL


def test_an_edge_is_directed_so_reversing_it_makes_a_different_edge() -> None:
    assert GraphEdge(LEFT, WHOLE, Role.CHART_CELL) != GraphEdge(
        WHOLE, LEFT, Role.CHART_CELL
    )


def test_edges_recording_the_same_three_fields_are_one_edge() -> None:
    edge = GraphEdge(LEFT, WHOLE, Role.CHART_CELL)

    assert edge == EDGES[0]
    assert hash(edge) == hash(EDGES[0])
    assert edge != GraphEdge(LEFT, WHOLE, Role.SPLIT_DISTRIBUTION)
    assert edge != (LEFT, WHOLE, Role.CHART_CELL)


def test_a_role_cannot_arrive_at_an_edge_as_a_string() -> None:
    with pytest.raises(TypeError, match="role is a Role"):
        GraphEdge(LEFT, WHOLE, cast("Role", "chart_cell"))


def test_an_edge_runs_between_nodes_and_not_between_items() -> None:
    with pytest.raises(TypeError, match="runs between graph nodes"):
        GraphEdge(cast("GraphNode[str]", DET), WHOLE, Role.CHART_CELL)


def test_a_graph_holds_the_steps_and_edges_it_was_given() -> None:
    graph = ComputationGraph(NODES, EDGES)

    assert graph.nodes == frozenset(NODES)
    assert graph.edges == frozenset(EDGES)


def test_a_graph_publishes_where_a_steps_value_can_be_read() -> None:
    inside = Site(PARSER, Role.CHART_CELL, (0, 2))
    graph = ComputationGraph(NODES, EDGES, {WHOLE: inside})

    assert graph.site_of(WHOLE) == inside
    assert graph.site_of(LEFT) is None


def test_a_site_published_for_a_step_that_is_not_in_the_graph_is_refused() -> None:
    stray = GraphNode(PARSER, SENTENCE)

    with pytest.raises(ValueError, match="is not a node here"):
        ComputationGraph(NODES, EDGES, {stray: Site(PARSER, Role.CHART_CELL, (0, 2))})


def test_a_step_is_published_at_a_site_and_not_at_an_attribute_path() -> None:
    with pytest.raises(TypeError, match="published at a Site"):
        ComputationGraph(
            NODES, EDGES, cast("dict[GraphNode[str], Site]", {WHOLE: "chart"})
        )


def test_an_edge_reaching_outside_the_graph_is_refused() -> None:
    stray = GraphNode(PARSER, SENTENCE)

    with pytest.raises(ValueError, match="is not a node here"):
        ComputationGraph(NODES, [*EDGES, GraphEdge(stray, WHOLE, Role.CHART_CELL)])


def test_a_cyclic_edge_set_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="the edges cycle"):
        ComputationGraph(
            [LEFT, RIGHT],
            [
                GraphEdge(LEFT, RIGHT, Role.CHART_CELL),
                GraphEdge(RIGHT, LEFT, Role.CHART_CELL),
            ],
        )


def test_a_step_that_feeds_itself_is_refused() -> None:
    with pytest.raises(ValueError, match="the edges cycle"):
        ComputationGraph([LEFT], [GraphEdge(LEFT, LEFT, Role.CHART_CELL)])


def test_a_longer_cycle_is_refused_and_the_message_names_a_node_on_it() -> None:
    middle = GraphNode(PARSER, NAME)

    with pytest.raises(ValueError, match="through GraphNode"):
        ComputationGraph(
            [LEFT, middle, RIGHT],
            [
                GraphEdge(LEFT, middle, Role.CHART_CELL),
                GraphEdge(middle, RIGHT, Role.CHART_CELL),
                GraphEdge(RIGHT, LEFT, Role.CHART_CELL),
            ],
        )


def test_a_graph_with_no_edges_at_all_is_acyclic() -> None:
    assert ComputationGraph(NODES, []).edges == frozenset()


def test_a_graph_holds_nodes_and_not_the_items_beneath_them() -> None:
    with pytest.raises(TypeError, match="holds graph nodes"):
        ComputationGraph(cast("list[GraphNode[str]]", [DET]), [])


@settings(max_examples=50)
@given(order=st.permutations(list(range(len(NODES)))))
def test_shuffling_the_node_list_leaves_the_edge_set_identical(
    order: list[int],
) -> None:
    shuffled = [NODES[position] for position in order]
    graph = ComputationGraph(shuffled, EDGES)

    assert graph.edges == frozenset(EDGES)
    assert graph == ComputationGraph(NODES, EDGES)
    assert hash(graph) == hash(ComputationGraph(NODES, EDGES))


def test_graphs_recording_the_same_steps_edges_and_sites_are_one_graph() -> None:
    inside = Site(PARSER, Role.CHART_CELL, (0, 2))
    graph = ComputationGraph(NODES, EDGES, {WHOLE: inside})

    assert graph == ComputationGraph(NODES, EDGES, {WHOLE: inside})
    assert graph != ComputationGraph(NODES, EDGES)
    assert graph != ComputationGraph(NODES, EDGES[:1], {WHOLE: inside})
    assert graph != frozenset(NODES)


def test_a_graph_cannot_be_rebound() -> None:
    graph = ComputationGraph(NODES, EDGES)

    for held in ("nodes", "edges"):
        with pytest.raises(AttributeError):
            setattr(graph, held, None)

    assert not hasattr(graph, "__dict__")


def test_the_repr_of_a_graph_says_how_much_of_each_thing_it_holds() -> None:
    shown = repr(ComputationGraph(NODES, EDGES, {WHOLE: Site(PARSER, Role.CHART_CELL)}))

    assert shown == "ComputationGraph(nodes=3, edges=2, sites=1)"


def test_a_forest_graph_has_one_node_per_item_at_that_items_site() -> None:
    forest = a_forest(
        lexical(DET), lexical(NOUN), lexical(NAME), combined(PHRASE, "fa", DET, NOUN)
    )
    sites = sites_of(forest)
    graph = ComputationGraph.from_forest(forest, sites)

    assert len(graph.nodes) == forest.num_items == 4
    assert graph.nodes == frozenset(GraphNode(PARSER, key) for key in sites)

    for node in graph.nodes:
        assert graph.site_of(node) == sites[node.key]


def test_a_forest_graph_draws_only_the_edges_the_forest_holds() -> None:
    # NAME sits beside NOUN over the second token, so a graph reconstructed from
    # span adjacency would feed PHRASE from both; the forest fed it from NOUN only
    forest = a_forest(
        lexical(DET), lexical(NOUN), lexical(NAME), combined(PHRASE, "fa", DET, NOUN)
    )
    graph = ComputationGraph.from_forest(forest, sites_of(forest))
    beside = GraphNode(PARSER, NAME)

    assert graph.edges == frozenset(EDGES)
    assert GraphEdge(beside, WHOLE, Role.CHART_CELL) not in graph.edges
    assert beside in graph.nodes


def test_repeated_alternatives_with_one_premise_and_one_conclusion_are_one_edge() -> (
    None
):
    forest = a_forest(
        lexical(DET),
        lexical(NOUN),
        combined(PHRASE, "fa", DET, NOUN),
        combined(PHRASE, "ba", DET, NOUN),
        combined(PHRASE, "fa", DET, DET),
    )
    graph = ComputationGraph.from_forest(forest, sites_of(forest))

    assert len(forest.edges(PHRASE)) == 3
    assert graph.edges == frozenset(EDGES)


def test_distinct_readings_and_tags_over_one_span_are_distinct_nodes() -> None:
    retagged = ItemKey((0, 2), "NP", "other")
    forest = a_forest(
        lexical(DET),
        lexical(NOUN),
        combined(PHRASE, "fa", DET, NOUN),
        combined(SENTENCE, "ba", DET, NOUN),
        combined(retagged, "fa", DET, NOUN),
    )
    graph = ComputationGraph.from_forest(forest, sites_of(forest))
    over_the_pair = {node for node in graph.nodes if node.span == (0, 2)}

    assert over_the_pair == {
        WHOLE,
        GraphNode(PARSER, SENTENCE),
        GraphNode(PARSER, retagged),
    }
    assert len(graph.edges) == 6


def test_an_edge_role_is_the_role_its_source_is_published_under() -> None:
    forest = a_forest(lexical(DET), lexical(NOUN), combined(PHRASE, "fa", DET, NOUN))
    sites = sites_of(forest)
    sites[NOUN] = Site(PARSER, Role.SPLIT_DISTRIBUTION, (1, 2))
    graph = ComputationGraph.from_forest(forest, sites)

    assert graph.edges == {
        GraphEdge(LEFT, WHOLE, Role.CHART_CELL),
        GraphEdge(RIGHT, WHOLE, Role.SPLIT_DISTRIBUTION),
    }


def test_a_node_takes_its_component_from_its_site() -> None:
    forest = a_forest(lexical(DET), lexical(NOUN), combined(PHRASE, "fa", DET, NOUN))
    sites = sites_of(forest)
    chart = ComponentId("chart")
    sites[PHRASE] = Site(chart, Role.CHART_CELL, (0, 2))
    graph = ComputationGraph.from_forest(forest, sites)

    assert GraphNode(chart, PHRASE) in graph.nodes
    assert WHOLE not in graph.nodes


def test_an_item_published_at_no_site_is_refused() -> None:
    forest = a_forest(lexical(DET), lexical(NOUN), combined(PHRASE, "fa", DET, NOUN))
    sites = sites_of(forest)
    del sites[NOUN]

    with pytest.raises(ValueError, match="published at no site"):
        ComputationGraph.from_forest(forest, sites)


def test_a_site_published_for_an_item_the_forest_does_not_hold_is_refused() -> None:
    forest = a_forest(lexical(DET), lexical(NOUN), combined(PHRASE, "fa", DET, NOUN))
    sites = sites_of(forest)
    sites[SENTENCE] = a_site(SENTENCE)

    with pytest.raises(ValueError, match="which the forest does not hold"):
        ComputationGraph.from_forest(forest, sites)


def test_an_item_published_somewhere_other_than_a_site_is_refused() -> None:
    forest = a_forest(lexical(DET), lexical(NOUN), combined(PHRASE, "fa", DET, NOUN))
    sites = sites_of(forest)
    sites[NOUN] = cast("Site", "chart.cells[1]")

    with pytest.raises(TypeError, match="published at a Site"):
        ComputationGraph.from_forest(forest, sites)


def test_an_alternative_consuming_an_absent_item_is_refused() -> None:
    forest = a_forest(lexical(DET), combined(PHRASE, "fa", DET, NOUN))

    with pytest.raises(ValueError, match="which the forest does not hold"):
        ComputationGraph.from_forest(forest, sites_of(forest))


def test_a_forest_whose_alternatives_cycle_is_refused() -> None:
    forest = a_forest(
        lexical(DET),
        lexical(NOUN),
        combined(PHRASE, "fa", DET, NOUN),
        (SENTENCE, Edge("up", (PHRASE,), "S", 1)),
        (PHRASE, Edge("down", (SENTENCE,), "NP", 1)),
    )

    with pytest.raises(ValueError, match="the edges cycle"):
        ComputationGraph.from_forest(forest, sites_of(forest))


def test_the_forest_graph_of_no_items_is_empty() -> None:
    graph = ComputationGraph.from_forest(a_forest(), {})

    assert graph.nodes == frozenset()
    assert graph.edges == frozenset()


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


def pinned_forests() -> Iterator[PackedForest[Term, float]]:
    sentences_path = FIXTURES / "chart_200.jsonl"
    fingerprint_path = FIXTURES / "chart_200.fingerprint"

    if not sentences_path.is_file() or not fingerprint_path.is_file():
        pytest.skip("the pinned chart fixture is absent")

    recorded = PinnedFingerprint.read(fingerprint_path)
    assembly = fixture_grammar(recorded.grammar).assemble()
    check_pinned_fixture(assembly.grammar, sentences_path, fingerprint_path)

    for sentence in read_sentences(sentences_path):
        yield parse(
            sentence,
            rules=assembly.grammar.rules,
            source=assembly.lexicon,
            semiring=Viterbi(),
            scorer=Unweighted(),
            goal=assembly.grammar.goal,
            unary_depth=assembly.grammar.budget.unary_rounds,
            policy=ParsePolicy(Retain.FOREST, Reachability.ALL),
        )


def dataflow_of[T, W](
    chart: Chart[T, W], sites: Mapping[ItemKey[T], Site]
) -> frozenset[GraphEdge[T]]:
    return frozenset(
        GraphEdge(
            GraphNode(PARSER, premise), GraphNode(PARSER, key), sites[premise].role
        )
        for span in chart.index.spans_by_length()
        for key in chart.items_at(span)
        for alternative in chart.edges(key)
        for premise in alternative.children
    )


def test_every_pinned_forest_is_covered_exactly_and_acyclically() -> None:
    parsed = 0

    for forest in pinned_forests():
        sites = {
            key: Site(PARSER, Role.CHART_CELL, key.span)
            for span in forest.index.spans_by_length()
            for key in forest.items_at(span)
        }
        graph = ComputationGraph.from_forest(forest, sites)
        parsed += 1

        assert len(graph.nodes) == forest.num_items == len(sites)
        assert graph.edges == dataflow_of(forest, sites)

        for node in graph.nodes:
            assert graph.site_of(node) == sites[node.key]

    assert parsed == 200
