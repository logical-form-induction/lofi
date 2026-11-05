import pickle
from collections.abc import Mapping, MutableMapping
from itertools import count, pairwise

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from lofi.data._derivation import (
    Dependency,
    Derivation,
    DerivationNode,
    GraphEdge,
    GraphNode,
    NodeId,
)
from lofi.data._values import TokenSpan

MAX_EXAMPLES = 50
MAX_DEPTH = 3

LABELS = ("S", "NP", "N", "S/NP", "(S\\NP)/NP")
RULES = ("apply_forward", "compose_forward", "type_raise", "coordinate")
FEATURE_KEYS = ("number", "case", "form")
FEATURE_VALUES = ("sg", "pl", "nom", "acc", "bare")


def features() -> st.SearchStrategy[dict[str, str]]:
    return st.dictionaries(
        st.sampled_from(FEATURE_KEYS),
        st.sampled_from(FEATURE_VALUES),
        max_size=3,
    )


def _split(
    draw: st.DrawFn,
    start: int,
    end: int,
    depth: int,
) -> tuple[tuple[int, int], ...]:
    if depth >= MAX_DEPTH or draw(st.integers(min_value=0, max_value=3)) == 0:
        return ()

    if end - start == 1 or draw(st.integers(min_value=0, max_value=3)) == 0:
        return ((start, end),)

    cuts = sorted(
        draw(
            st.sets(
                st.integers(min_value=start + 1, max_value=end - 1),
                min_size=1,
                max_size=2,
            )
        )
    )
    points = (start, *cuts, end)

    return tuple(pairwise(points))


@st.composite
def derivations(draw: st.DrawFn) -> Derivation[str]:
    width = draw(st.integers(min_value=1, max_value=6))
    nodes: dict[NodeId, DerivationNode[str]] = {}
    ids = count()

    def build(start: int, end: int, depth: int) -> NodeId:
        node_id = next(ids)
        children = tuple(
            build(child_start, child_end, depth + 1)
            for child_start, child_end in _split(draw, start, end, depth)
        )
        nodes[node_id] = DerivationNode(
            TokenSpan(start, end),
            draw(st.sampled_from(LABELS)),
            rule=draw(st.none() | st.sampled_from(RULES)),
            children=children,
            features=draw(st.none() | features()),
        )

        return node_id

    root = build(0, width, 0)
    scored = draw(
        st.dictionaries(
            st.sampled_from(sorted(nodes)),
            st.floats(min_value=0.0, max_value=1.0),
            max_size=2,
        )
    )

    return Derivation(root, nodes, scored)


def leaf(start: int, end: int, label: str = "N") -> DerivationNode[str]:
    return DerivationNode(TokenSpan(start, end), label)


@given(derivation=derivations())
@settings(max_examples=MAX_EXAMPLES)
def test_the_leaves_tile_the_derivation_span_exactly_once(
    derivation: Derivation[str],
) -> None:
    covered: list[int] = []

    for node in derivation.leaves():
        covered.extend(range(node.span.start, node.span.end))

    assert covered == list(range(derivation.span.start, derivation.span.end))


@given(derivation=derivations())
@settings(max_examples=MAX_EXAMPLES)
def test_a_non_leaf_span_is_the_union_of_its_children(
    derivation: Derivation[str],
) -> None:
    for node_id in derivation.nodes:
        node = derivation.node(node_id)
        children = derivation.children_of(node_id)

        if not children:
            continue

        covered = sorted(
            position
            for child in children
            for position in range(child.span.start, child.span.end)
        )

        assert covered == list(range(node.span.start, node.span.end))


@given(derivation=derivations())
@settings(max_examples=MAX_EXAMPLES)
def test_every_node_is_reached_from_the_root_exactly_once(
    derivation: Derivation[str],
) -> None:
    reached: list[NodeId] = []
    pending = [derivation.root]

    while pending:
        current = pending.pop()
        reached.append(current)
        pending.extend(derivation.node(current).children)

    assert sorted(reached) == sorted(derivation.nodes)
    assert len(derivation) == len(reached)


def test_preorder_is_root_first_and_left_to_right() -> None:
    derivation = Derivation(
        40,
        {
            13: leaf(1, 2),
            2: leaf(2, 3),
            40: DerivationNode(TokenSpan(0, 3), "S", children=(7, 2)),
            11: leaf(0, 1),
            7: DerivationNode(TokenSpan(0, 2), "NP", children=(11, 13)),
        },
    )

    assert derivation.preorder == (40, 7, 11, 13, 2)


@given(derivation=derivations())
@settings(max_examples=MAX_EXAMPLES)
def test_a_derivation_survives_pickling_equal_and_equally_hashed(
    derivation: Derivation[str],
) -> None:
    restored = pickle.loads(pickle.dumps(derivation))

    assert restored is not derivation
    assert restored == derivation
    assert hash(restored) == hash(derivation)
    assert restored.leaves() == derivation.leaves()


def test_a_unary_type_change_node_and_a_coordination_node_are_the_same_shape() -> None:
    type_change = DerivationNode(TokenSpan(0, 1), "NP", rule="n_to_np", children=(1,))
    coordination = DerivationNode(
        TokenSpan(0, 5),
        "NP",
        rule="coordinate",
        children=(1, 2, 3),
    )

    assert type(type_change) is type(coordination)
    assert len(type_change.children) == 1
    assert len(coordination.children) == 3
    assert not type_change.is_lexical
    assert not coordination.is_lexical


def test_a_derivation_holds_an_n_ary_node_over_its_children() -> None:
    nodes = {
        0: DerivationNode(TokenSpan(0, 3), "NP", rule="coordinate", children=(1, 2, 3)),
        1: leaf(0, 1),
        2: leaf(1, 2, "conj"),
        3: leaf(2, 3),
    }
    derivation = Derivation(0, nodes)

    assert len(derivation.children_of(0)) == 3
    assert [node.span.start for node in derivation.leaves()] == [0, 1, 2]


def test_a_lexical_node_is_the_one_with_no_children() -> None:
    assert leaf(0, 1).is_lexical
    assert not DerivationNode(TokenSpan(0, 1), "NP", children=(1,)).is_lexical


def test_a_node_may_name_no_rule() -> None:
    unlabeled = DerivationNode(TokenSpan(0, 2), "S", children=(1, 2))

    assert unlabeled.rule is None


def test_a_node_refuses_an_empty_rule_name() -> None:
    with pytest.raises(ValueError, match="empty rule"):
        DerivationNode(TokenSpan(0, 1), "S", rule="")


def test_two_nodes_differing_only_in_their_rule_are_different_nodes() -> None:
    applied = DerivationNode(TokenSpan(0, 2), "S", rule="apply_forward", children=(1,))
    composed = DerivationNode(
        TokenSpan(0, 2),
        "S",
        rule="compose_forward",
        children=(1,),
    )

    assert applied != composed
    assert len({applied, composed}) == 2


def test_a_feature_bearing_node_round_trips_through_construction() -> None:
    node = DerivationNode(
        TokenSpan(0, 1),
        "S",
        features={"form": "bare", "number": "sg"},
    )
    restored = DerivationNode(
        node.span,
        node.label,
        rule=node.rule,
        children=node.children,
        features=node.features,
    )

    assert restored == node
    assert hash(restored) == hash(node)
    assert dict(restored.features) == {"form": "bare", "number": "sg"}


def test_two_nodes_differing_only_in_their_features_are_different_nodes() -> None:
    singular = DerivationNode(TokenSpan(0, 1), "NP", features={"number": "sg"})
    plural = DerivationNode(TokenSpan(0, 1), "NP", features={"number": "pl"})

    assert singular != plural
    assert len({singular, plural}) == 2


def test_feature_order_does_not_change_a_node() -> None:
    one = DerivationNode(
        TokenSpan(0, 1), "NP", features={"case": "nom", "number": "sg"}
    )
    other = DerivationNode(
        TokenSpan(0, 1),
        "NP",
        features={"number": "sg", "case": "nom"},
    )

    assert one == other
    assert hash(one) == hash(other)


def test_a_node_copies_the_features_it_is_given() -> None:
    given_features = {"number": "sg"}
    node = DerivationNode(TokenSpan(0, 1), "NP", features=given_features)
    given_features["number"] = "pl"

    assert dict(node.features) == {"number": "sg"}


def test_a_node_hands_out_no_writable_feature_map() -> None:
    node = DerivationNode(TokenSpan(0, 1), "NP", features={"number": "sg"})

    assert isinstance(node.features, Mapping)
    assert not isinstance(node.features, MutableMapping)


def test_a_node_refuses_an_empty_feature_key() -> None:
    with pytest.raises(ValueError, match="empty key"):
        DerivationNode(TokenSpan(0, 1), "NP", features={"": "sg"})


def test_confidence_stays_outside_a_derivation_identity() -> None:
    nodes = {0: DerivationNode(TokenSpan(0, 1), "NP", children=(1,)), 1: leaf(0, 1)}
    confident = Derivation(0, nodes, {0: 0.99})
    doubtful = Derivation(0, nodes, {0: 0.11})

    assert confident == doubtful
    assert hash(confident) == hash(doubtful)
    assert confident.confidence_of(0) == pytest.approx(0.99)
    assert doubtful.confidence_of(0) == pytest.approx(0.11)


def test_an_unscored_node_reports_no_confidence() -> None:
    derivation = Derivation(0, {0: leaf(0, 1)})

    assert derivation.confidence_of(0) is None


def test_a_derivation_refuses_a_confidence_for_a_node_it_does_not_hold() -> None:
    with pytest.raises(ValueError, match="scores node 7"):
        Derivation(0, {0: leaf(0, 1)}, {7: 0.5})


def test_a_derivation_refuses_children_that_leave_a_gap() -> None:
    nodes = {
        0: DerivationNode(TokenSpan(0, 3), "S", children=(1, 2)),
        1: leaf(0, 1),
        2: leaf(2, 3),
    }

    with pytest.raises(ValueError, match="does not continue from 1"):
        Derivation(0, nodes)


def test_a_derivation_refuses_children_that_overlap() -> None:
    nodes = {
        0: DerivationNode(TokenSpan(0, 3), "S", children=(1, 2)),
        1: leaf(0, 2),
        2: leaf(1, 3),
    }

    with pytest.raises(ValueError, match="does not continue from 2"):
        Derivation(0, nodes)


def test_a_derivation_refuses_children_that_stop_short_of_their_parent() -> None:
    nodes = {
        0: DerivationNode(TokenSpan(0, 3), "S", children=(1,)),
        1: leaf(0, 2),
    }

    with pytest.raises(ValueError, match="reaching only to 2"):
        Derivation(0, nodes)


def test_a_derivation_refuses_a_child_it_does_not_hold() -> None:
    nodes = {0: DerivationNode(TokenSpan(0, 1), "S", children=(4,))}

    with pytest.raises(ValueError, match="names node 4"):
        Derivation(0, nodes)


def test_a_derivation_refuses_a_root_it_does_not_hold() -> None:
    with pytest.raises(ValueError, match="names node 3"):
        Derivation(3, {0: leaf(0, 1)})


def test_a_derivation_refuses_a_cycle() -> None:
    nodes = {
        0: DerivationNode(TokenSpan(0, 1), "S", children=(1,)),
        1: DerivationNode(TokenSpan(0, 1), "NP", children=(0,)),
    }

    with pytest.raises(ValueError, match="reaches node 0 twice"):
        Derivation(0, nodes)


def test_a_derivation_refuses_a_node_shared_by_two_parents() -> None:
    nodes = {
        0: DerivationNode(TokenSpan(0, 2), "S", children=(1, 2)),
        1: DerivationNode(TokenSpan(0, 1), "NP", children=(3,)),
        2: DerivationNode(TokenSpan(1, 2), "VP", children=(3,)),
        3: leaf(0, 1),
    }

    with pytest.raises(ValueError, match="reaches node 3 twice"):
        Derivation(0, nodes)


def test_a_derivation_refuses_a_node_unreachable_from_its_root() -> None:
    with pytest.raises(ValueError, match=r"unreachable from its root: \[1\]"):
        Derivation(0, {0: leaf(0, 1), 1: leaf(1, 2)})


def test_a_node_refuses_an_empty_span() -> None:
    with pytest.raises(ValueError, match="covers no token"):
        DerivationNode(TokenSpan(2, 2), "S")


def test_a_node_refuses_a_negative_child_id() -> None:
    with pytest.raises(ValueError, match="negative child id"):
        DerivationNode(TokenSpan(0, 1), "S", children=(-1,))


def test_a_node_refuses_the_same_child_twice() -> None:
    with pytest.raises(ValueError, match="one child twice"):
        DerivationNode(TokenSpan(0, 2), "S", children=(1, 1))


def test_a_derivation_copies_the_nodes_it_is_given() -> None:
    nodes = {0: leaf(0, 1)}
    derivation = Derivation(0, nodes)
    nodes[1] = leaf(1, 2)

    assert len(derivation) == 1
    assert sorted(derivation.nodes) == [0]


def test_a_derivation_hands_out_no_writable_node_map() -> None:
    derivation = Derivation(0, {0: leaf(0, 1)})

    assert isinstance(derivation.nodes, Mapping)
    assert not isinstance(derivation.nodes, MutableMapping)


def test_a_derivation_refuses_an_attribute_write() -> None:
    derivation = Derivation(0, {0: leaf(0, 1)})

    for field, value in (("root", 4), ("nodes", {}), ("span", None)):
        with pytest.raises(AttributeError):
            setattr(derivation, field, value)

    assert derivation.root == 0


def test_a_node_refuses_an_attribute_write() -> None:
    node = leaf(0, 1)

    for field, value in (("label", "S"), ("span", None), ("children", ())):
        with pytest.raises(AttributeError):
            setattr(node, field, value)

    assert node.label == "N"


def test_a_derived_quantity_is_not_writable() -> None:
    node = leaf(0, 1)

    for field in ("is_lexical", "rule"):
        with pytest.raises(AttributeError):
            setattr(node, field, False)

    assert node.is_lexical


def test_asking_for_a_node_a_derivation_does_not_hold_raises() -> None:
    derivation = Derivation(0, {0: leaf(0, 1)})

    with pytest.raises(KeyError):
        derivation.node(9)


def test_children_are_read_in_the_order_the_corpus_listed_them() -> None:
    nodes = {
        0: DerivationNode(TokenSpan(0, 2), "S", children=(2, 1)),
        1: leaf(0, 1, "NP"),
        2: leaf(1, 2, "VP"),
    }
    derivation = Derivation(0, nodes)

    assert [node.label for node in derivation.children_of(0)] == ["VP", "NP"]
    assert [node.label for node in derivation.leaves()] == ["NP", "VP"]


def test_a_dependency_contains_its_positions_and_label() -> None:
    dependency = Dependency(1, 0, "nsubj", 1)

    assert dependency.head == 1
    assert dependency.dependent == 0
    assert dependency.label == "nsubj"
    assert dependency.slot == 1


def test_a_dependency_without_a_slot_records_none() -> None:
    assert Dependency(1, 0, "nsubj").slot is None


def test_a_dependency_survives_pickling_equal_and_equally_hashed() -> None:
    dependency = Dependency(3, 1, "dobj", 2)
    restored = pickle.loads(pickle.dumps(dependency))

    assert restored == dependency
    assert hash(restored) == hash(dependency)
    assert {dependency, restored} == {dependency}


@pytest.mark.parametrize(
    ("head", "dependent", "label", "slot", "message"),
    [
        (-1, 0, "nsubj", None, "negative head"),
        (0, -2, "nsubj", None, "negative dependent"),
        (0, 1, "", None, "has no label"),
        (0, 1, "nsubj", -1, "negative slot"),
    ],
)
def test_a_dependency_refuses_a_malformed_field(
    head: int,
    dependent: int,
    label: str,
    slot: int | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        Dependency(head, dependent, label, slot)


def test_a_dependency_refuses_an_attribute_write() -> None:
    dependency = Dependency(1, 0, "nsubj")

    for field, value in (("head", 2), ("dependent", 3), ("label", "obj")):
        with pytest.raises(AttributeError):
            setattr(dependency, field, value)

    assert dependency.head == 1


def test_a_graph_node_contains_its_type_label_anchor_and_properties() -> None:
    node = GraphNode(
        "b1",
        "situation",
        label="sleep",
        anchor=TokenSpan(2, 3),
        properties={"tense": "present"},
    )

    assert node.identifier == "b1"
    assert node.node_type == "situation"
    assert node.label == "sleep"
    assert node.anchor == TokenSpan(2, 3)
    assert dict(node.properties) == {"tense": "present"}


def test_an_unanchored_graph_node_records_no_anchor() -> None:
    assert GraphNode("b0", "concept").anchor is None


def test_a_graph_node_and_a_graph_edge_survive_pickling() -> None:
    node = GraphNode("x1", "entity", label="child", anchor=TokenSpan(1, 2))
    edge = GraphEdge("b1", "x1", "instance-of", label="agent")

    assert pickle.loads(pickle.dumps(node)) == node
    assert pickle.loads(pickle.dumps(edge)) == edge
    assert hash(pickle.loads(pickle.dumps(edge))) == hash(edge)


def test_a_graph_edge_joins_two_nodes_by_identifier() -> None:
    edge = GraphEdge("b1", "x1", "instance-of", properties={"role": "agent"})

    assert edge.source == "b1"
    assert edge.target == "x1"
    assert edge.edge_type == "instance-of"
    assert edge.label is None
    assert dict(edge.properties) == {"role": "agent"}


def test_two_edges_differing_only_in_direction_are_different_edges() -> None:
    forward = GraphEdge("b1", "x1", "part-of")
    backward = GraphEdge("x1", "b1", "part-of")

    assert forward != backward
    assert len({forward, backward}) == 2


def test_a_graph_node_refuses_an_empty_identifier_or_type() -> None:
    with pytest.raises(ValueError, match="no identifier"):
        GraphNode("", "entity")

    with pytest.raises(ValueError, match="no node type"):
        GraphNode("x1", "")


def test_a_graph_node_refuses_an_anchor_covering_no_token() -> None:
    with pytest.raises(ValueError, match="anchored to no token"):
        GraphNode("x1", "entity", anchor=TokenSpan(2, 2))


def test_a_graph_edge_refuses_a_missing_endpoint_or_type() -> None:
    with pytest.raises(ValueError, match="no source"):
        GraphEdge("", "x1", "part-of")

    with pytest.raises(ValueError, match="no target"):
        GraphEdge("b1", "", "part-of")

    with pytest.raises(ValueError, match="no edge type"):
        GraphEdge("b1", "x1", "")


def test_a_graph_value_is_not_equal_to_an_unrelated_object() -> None:
    assert GraphNode("x1", "entity") != "x1"
    assert GraphEdge("b1", "x1", "part-of") != ("b1", "x1", "part-of")
    assert leaf(0, 1) != TokenSpan(0, 1)
    assert Derivation(0, {0: leaf(0, 1)}) != 0


def test_a_meaning_graph_may_share_a_node_between_two_edges() -> None:
    referent = GraphNode("x1", "entity", label="child", anchor=TokenSpan(1, 2))
    situation = GraphNode("b1", "situation", label="sleep", anchor=TokenSpan(2, 3))
    edges = (
        GraphEdge("b1", "x1", "agent"),
        GraphEdge("b1", "x1", "theme"),
    )

    assert len({referent, situation}) == 2
    assert len({edge.target for edge in edges}) == 1
    assert len(set(edges)) == 2
