from datetime import UTC, datetime
from typing import Final

import pytest
from hypothesis import given
from hypothesis import strategies as st
from lairs.records import annotation as annotation_records
from lairs.records import defs as defs_records
from lairs.records import graph as graph_records
from lairs.records import judgment as judgment_records
from lairs.records import segmentation as segmentation_records

from lofi.data import (
    RESPONSE_MISSING,
    AnnotatorId,
    Categorical,
    Derivation,
    DerivationNode,
    GoldGraph,
    ItemId,
    Relational,
    RelationalScalar,
    Scalar,
    SpanConstraints,
    TokenSpan,
)
from lofi.data.corpora import SignatureLabels
from lofi.data.corpora.layers import _translate as translate_module
from lofi.data.corpora.layers._translate import (
    EMITTED_ARMS,
    GRAPH_LIFT,
    LAYER_LIFTS,
    RELATION_LIFT,
    RELATION_NAMES,
    RESPONSE_LIFTS,
    GraphContext,
    LayerContext,
    LayerLift,
    LayersSkip,
    NodeSource,
    RecordSkippedError,
    RelationPair,
    TokenReading,
    TreeLift,
    annotator_of,
    dependencies_of,
    derivation_of,
    features_of,
    graph_of,
    graph_start,
    item_id_of,
    lift_tree,
    lower_tree,
    pair_response_of,
    provenance_of,
    relation_pair_of,
    relational_edge_of,
    relational_layer_of,
    response_of,
    scale_of,
    span_constraints_of,
    supervision_of,
    tokenization_of,
    tokens_of,
)
from lofi.terms import Operator, Signature

CREATED: Final = datetime(2026, 1, 1, tzinfo=UTC)

TARGET: Final = "at://did:plc:test/pub.layers.expression.expression/one"

HYPOTHESIS: Final = "at://did:plc:test/pub.layers.expression.expression/two"

HALF: Final = 500

SEVEN: Final = 7

CONFIDENCE_CEILING: Final = 1000

DEEP: Final = 4000


def uuid_of(value: str) -> defs_records.Uuid:
    return defs_records.Uuid(value=value)


def node(
    value: str,
    *,
    label: str | None = "X",
    parent: str | None = None,
    children: tuple[str, ...] = (),
    token: int | None = None,
) -> annotation_records.Annotation:
    return annotation_records.Annotation(
        uuid=uuid_of(value),
        label=label,
        parentId=None if parent is None else uuid_of(parent),
        childIds=tuple(uuid_of(child) for child in children),
        tokenIndex=token,
    )


def scored(
    value: str,
    label: str,
    confidence: int,
    *,
    children: tuple[str, ...] = (),
    token: int | None = None,
) -> annotation_records.Annotation:
    return annotation_records.Annotation(
        uuid=uuid_of(value),
        label=label,
        childIds=tuple(uuid_of(child) for child in children),
        tokenIndex=token,
        confidence=confidence,
    )


def layer(
    kind: str,
    annotations: tuple[annotation_records.Annotation, ...],
    *,
    subkind: str | None = None,
    tokenization: str | None = None,
) -> annotation_records.AnnotationLayer:
    return annotation_records.AnnotationLayer(
        createdAt=CREATED,
        expression=TARGET,
        kind=kind,
        subkind=subkind,
        annotations=annotations,
        tokenizationId=None if tokenization is None else uuid_of(tokenization),
    )


def dependency_layer(
    annotations: tuple[annotation_records.Annotation, ...],
) -> annotation_records.AnnotationLayer:
    return layer("relation", annotations, subkind="dependency")


def arc(
    value: str,
    label: str | None,
    head: int | None,
    dependent: int | None,
) -> annotation_records.Annotation:
    return annotation_records.Annotation(
        uuid=uuid_of(value),
        label=label,
        headIndex=head,
        targetIndex=dependent,
    )


def graph_node(
    label: str | None = "sleep",
    node_type: str = "situation",
    properties: tuple[tuple[str, str], ...] = (),
) -> graph_records.GraphNode:
    return graph_records.GraphNode(
        createdAt=CREATED,
        nodeType=node_type,
        label=label,
        properties=defs_records.FeatureMap(
            entries=tuple(
                defs_records.Feature(key=key, value=value) for key, value in properties
            )
        )
        if properties
        else None,
    )


def graph_edge(
    source: str,
    target: str,
    edge_type: str = "denotes",
    label: str | None = None,
    properties: tuple[tuple[str, str], ...] = (),
) -> graph_records.GraphEdge:
    return graph_records.GraphEdge(
        createdAt=CREATED,
        edgeType=edge_type,
        label=label,
        source=defs_records.ObjectRef(recordRef=source),
        target=defs_records.ObjectRef(recordRef=target),
        properties=defs_records.FeatureMap(
            entries=tuple(
                defs_records.Feature(key=key, value=value) for key, value in properties
            )
        )
        if properties
        else None,
    )


def segmentation(
    tokens: tuple[segmentation_records.Token, ...],
) -> segmentation_records.Segmentation:
    return segmentation_records.Segmentation(
        createdAt=CREATED,
        expression=TARGET,
        tokenizations=(
            segmentation_records.Tokenization(
                uuid=uuid_of("t"),
                kind="whitespace",
                tokens=tokens,
            ),
        ),
    )


def token(
    index: int,
    text: str,
    span: tuple[int, int] | None = None,
) -> segmentation_records.Token:
    return segmentation_records.Token(
        tokenIndex=index,
        text=text,
        textSpan=None
        if span is None
        else defs_records.Span(byteStart=span[0], byteEnd=span[1]),
    )


def experiment(
    *,
    task: str | None,
    minimum: int | None = 1,
    maximum: int | None = 7,
    labels: tuple[str, ...] = (),
    features: tuple[tuple[str, str], ...] = (),
) -> judgment_records.ExperimentDef:
    return judgment_records.ExperimentDef(
        name="a study",
        createdAt=CREATED,
        taskType=task,
        scaleMin=minimum,
        scaleMax=maximum,
        labels=labels,
        features=defs_records.FeatureMap(
            entries=tuple(
                defs_records.Feature(key=key, value=value) for key, value in features
            )
        )
        if features
        else None,
    )


def judgment(
    *,
    value: int | None = None,
    label: str | None = None,
    target: str = TARGET,
    object_id: str | None = None,
) -> judgment_records.Judgment:
    return judgment_records.Judgment(
        item=defs_records.ObjectRef(
            recordRef=target,
            objectId=None if object_id is None else uuid_of(object_id),
        ),
        scalarValue=value,
        categoricalValue=label,
    )


def metadata(agent: str) -> defs_records.AnnotationMetadata:
    return defs_records.AnnotationMetadata(
        tool="fixture",
        agent=defs_records.AgentRef(id=agent),
    )


def relation_edge(
    relation: str,
    agent: str | None = None,
) -> graph_records.GraphEdge:
    return graph_records.GraphEdge(
        createdAt=CREATED,
        edgeType=relation,
        source=defs_records.ObjectRef(recordRef=TARGET),
        target=defs_records.ObjectRef(recordRef=HYPOTHESIS),
        metadata=None if agent is None else metadata(agent),
    )


def relation_layer(
    annotation: annotation_records.Annotation,
    agent: str | None = None,
) -> annotation_records.AnnotationLayer:
    return annotation_records.AnnotationLayer(
        createdAt=CREATED,
        expression=TARGET,
        kind="relation",
        annotations=(annotation,),
        metadata=None if agent is None else metadata(agent),
    )


def inference_experiment() -> judgment_records.ExperimentDef:
    return judgment_records.ExperimentDef(
        name="a study",
        createdAt=CREATED,
        taskType="categorical",
        measureType="inference",
        labels=tuple(sorted(RELATION_NAMES)),
    )


def pair_annotation(
    value: str,
    label: str | None = "entails",
    hypothesis: str = HYPOTHESIS,
) -> annotation_records.Annotation:
    return annotation_records.Annotation(
        uuid=uuid_of(value),
        label=label,
        arguments=(
            annotation_records.ArgumentRef(
                role="hypothesis",
                target=defs_records.ObjectRef(recordRef=hypothesis),
            ),
        ),
    )


def test_the_relation_layer_decoder_has_its_exact_public_name() -> None:
    assert "relational_layer_of" in translate_module.__all__
    assert "pair_response_of" in translate_module.__all__
    assert "relational_of" not in translate_module.__all__


def licensing(expression: str | None) -> defs_records.Licensing:
    return defs_records.Licensing(
        expression=expression,
        licenses=(defs_records.LicenseRef(spdx="CC0-1.0", appliesTo="annotations"),),
    )


def test_a_feature_map_reads_as_its_entries() -> None:
    mapped = features_of(
        defs_records.FeatureMap(
            entries=(
                defs_records.Feature(key="a", value="1"),
                defs_records.Feature(key="b", value="2"),
            )
        )
    )

    assert mapped == {"a": "1", "b": "2"}
    assert features_of(None) == {}


def test_an_annotation_with_no_source_method_has_no_provenance() -> None:
    assert provenance_of(None, licensing("CC0-1.0"), (), None) is None
    assert provenance_of("", licensing("CC0-1.0"), (), None) is None


def test_an_annotation_with_no_license_expression_has_no_provenance() -> None:
    assert provenance_of("manual-native", licensing(None), (), None) is None
    assert provenance_of("manual-native", None, (), None) is None


def test_provenance_includes_license_eprints_and_reproduction() -> None:
    provenance = provenance_of(
        "automatic",
        licensing("CC0-1.0"),
        ("at://did:plc:test/pub.layers.eprint.eprint/one",),
        defs_records.ReproducibilityInfo(
            codeUri="https://example.invalid/code",
            commitHash="abc",
            command="run",
            environment="py314",
            randomSeed=11,
        ),
    )

    assert provenance is not None
    assert provenance.source_method == "automatic"
    assert provenance.licensing.expression == "CC0-1.0"
    assert [
        (reference.identifier, reference.applies_to)
        for reference in provenance.licensing.licenses
    ] == [("CC0-1.0", "annotations")]
    assert provenance.eprint_refs == ("at://did:plc:test/pub.layers.eprint.eprint/one",)
    reproduction = provenance.reproducibility

    assert reproduction is not None
    assert reproduction.code_uri == "https://example.invalid/code"
    assert reproduction.commit_hash == "abc"
    assert reproduction.command == "run"
    assert reproduction.environment == "py314"
    assert reproduction.random_seed == 11


def test_an_expression_with_no_segmentation_has_no_tokens() -> None:
    assert tokens_of("a b", ()) == ()


def test_an_expression_with_no_segmentation_names_no_tokenization() -> None:
    assert tokenization_of(()) is None


def test_the_tokenization_named_is_the_one_the_tokens_were_read_from() -> None:
    first = segmentation((token(0, "a"), token(1, "b")))
    second = segmentation_records.Segmentation(
        createdAt=CREATED,
        expression=TARGET,
        tokenizations=(
            segmentation_records.Tokenization(
                uuid=uuid_of("other"),
                kind="whitespace",
                tokens=(token(0, "ab"),),
            ),
        ),
    )
    read = tokens_of("a b", (first, second))

    assert tokenization_of((first, second)) == "t"
    assert [each.surface for each in read] == ["a", "b"]


def test_tokens_are_read_in_position_order_whatever_order_they_arrive_in() -> None:
    read = tokens_of("a b", (segmentation((token(1, "b"), token(0, "a"))),))

    assert [each.surface for each in read] == ["a", "b"]
    assert [each.index for each in read] == [0, 1]


def test_a_byte_span_survives_as_a_byte_span() -> None:
    read = tokens_of("孩子 睡觉", (segmentation((token(0, "孩子", (0, 6)),)),))

    assert read[0].span is not None
    assert read[0].span.substring("孩子 睡觉") == "孩子"


def test_a_tokenization_whose_positions_skip_is_refused() -> None:
    with pytest.raises(RecordSkippedError, match="contiguous"):
        tokens_of("a b", (segmentation((token(0, "a"), token(2, "b"))),))


def test_a_byte_span_into_an_expression_with_no_text_is_refused() -> None:
    with pytest.raises(RecordSkippedError, match="holding no text"):
        tokens_of(None, (segmentation((token(0, "a", (0, 1)),)),))


def test_tokens_without_spans_read_out_of_an_expression_with_no_text() -> None:
    read = tokens_of(None, (segmentation((token(0, "a"), token(1, "b"))),))

    assert [each.surface for each in read] == ["a", "b"]
    assert [each.span for each in read] == [None, None]


def test_a_tree_layer_spans_every_node_from_its_leaves() -> None:
    analyzed = derivation_of(
        layer(
            "tree",
            (
                node("r", label="S", children=("l", "m")),
                node("l", label="A", parent="r", token=0),
                node("m", label="B", parent="r", token=1),
            ),
        )
    )
    derivation = analyzed.derivation
    root = derivation.node(derivation.root)

    assert root.label == "S"
    assert (root.span.start, root.span.end) == (0, 2)
    assert derivation.children_of(derivation.root)[0].label == "A"


def test_a_tree_layer_preserves_confidence_on_the_derivation() -> None:
    analyzed = derivation_of(
        layer(
            "tree",
            (
                scored("r", "S", HALF, children=("l",)),
                node("l", label="A", parent="r", token=0),
            ),
        )
    )

    assert analyzed.derivation.confidence_of(analyzed.derivation.root) == 0.5


def test_a_tree_layer_with_no_root_is_refused() -> None:
    with pytest.raises(RecordSkippedError, match="one root"):
        derivation_of(layer("tree", ()))


def test_a_tree_layer_with_two_roots_is_refused() -> None:
    with pytest.raises(RecordSkippedError, match="one root"):
        derivation_of(layer("tree", (node("a", token=0), node("b", token=1))))


def test_a_tree_layer_naming_a_node_it_does_not_hold_is_refused() -> None:
    with pytest.raises(RecordSkippedError, match="does not hold"):
        derivation_of(layer("tree", (node("r", children=("gone",)),)))


def test_a_node_that_is_its_own_parent_is_a_cycle_rather_than_a_missing_root() -> None:
    with pytest.raises(RecordSkippedError, match="cycle"):
        derivation_of(layer("tree", (node("r", parent="r", token=0),)))


def test_a_tree_layer_whose_nodes_form_a_cycle_is_refused() -> None:
    with pytest.raises(RecordSkippedError, match="cycle"):
        derivation_of(
            layer(
                "tree",
                (
                    node("r", children=("a",)),
                    node("a", parent="r", children=("b",)),
                    node("b", parent="a", children=("r",)),
                ),
            )
        )


def test_a_tree_layer_whose_leaf_anchors_no_token_is_refused() -> None:
    with pytest.raises(RecordSkippedError, match="anchors no token"):
        derivation_of(
            layer("tree", (node("r", children=("l",)), node("l", parent="r")))
        )


def test_a_tree_layer_whose_node_has_no_label_is_refused() -> None:
    with pytest.raises(RecordSkippedError, match="no label"):
        derivation_of(layer("tree", (node("r", label=None, token=0),)))


def test_a_tree_layer_whose_children_leave_a_gap_is_refused() -> None:
    with pytest.raises(ValueError, match="does not continue"):
        derivation_of(
            layer(
                "tree",
                (
                    node("r", children=("l", "m")),
                    node("l", parent="r", token=0),
                    node("m", parent="r", token=2),
                ),
            )
        )


def test_a_relation_layer_reads_as_the_arcs_it_records() -> None:
    related = dependencies_of(dependency_layer((arc("d", "nsubj", 1, 0),)))

    assert [
        (relation.head, relation.dependent, relation.label)
        for relation in related.dependencies
    ] == [(1, 0, "nsubj")]


def test_a_relation_layer_whose_arc_names_no_head_is_refused() -> None:
    with pytest.raises(RecordSkippedError, match="no head"):
        dependencies_of(dependency_layer((arc("d", "nsubj", None, 0),)))


def test_the_arc_the_wire_roots_governs_itself() -> None:
    related = dependencies_of(
        dependency_layer((arc("r", "root", -1, 2), arc("d", "nsubj", 2, 0)))
    )
    rooted, other = related.dependencies

    assert (rooted.head, rooted.dependent) == (2, 2)
    assert (other.head, other.dependent) == (2, 0)


def test_the_root_arc_is_the_one_arc_that_governs_itself() -> None:
    related = dependencies_of(
        dependency_layer(
            (arc("a", "det", 1, 0), arc("r", "root", -1, 1), arc("b", "obj", 1, 2))
        )
    )
    roots = [
        relation.dependent
        for relation in related.dependencies
        if relation.head == relation.dependent
    ]

    assert roots == [1]


def test_an_arc_governing_from_no_position_at_all_is_refused() -> None:
    with pytest.raises(RecordSkippedError, match="no position"):
        dependencies_of(dependency_layer((arc("d", "nsubj", -2, 0),)))


@pytest.mark.parametrize("relation", sorted(RELATION_NAMES))
def test_a_relation_layer_names_a_directed_expression_pair(relation: str) -> None:
    annotation = pair_annotation("pair", relation)
    paired = relation_layer(annotation, "annotator-a")
    reading = relational_layer_of(paired, annotation)

    assert reading is not None
    assert reading.pair == RelationPair(ItemId(TARGET), ItemId(HYPOTHESIS))
    assert reading.supervision == Relational(
        ItemId(HYPOTHESIS),
        relation,
        AnnotatorId("annotator-a"),
    )


def test_an_unlabeled_relation_annotation_is_an_elicitation_carrier() -> None:
    annotation = pair_annotation("pair", None)
    paired = relation_layer(annotation)

    assert relation_pair_of(paired, annotation) == RelationPair(
        ItemId(TARGET), ItemId(HYPOTHESIS)
    )
    assert relational_layer_of(paired, annotation) is None


@pytest.mark.parametrize(
    "arguments",
    [
        (),
        (
            annotation_records.ArgumentRef(
                role="premise",
                target=defs_records.ObjectRef(recordRef=HYPOTHESIS),
            ),
        ),
        (
            annotation_records.ArgumentRef(
                role="hypothesis",
                target=defs_records.ObjectRef(recordRef=HYPOTHESIS),
            ),
            annotation_records.ArgumentRef(
                role="hypothesis",
                target=defs_records.ObjectRef(recordRef=TARGET),
            ),
        ),
    ],
)
def test_a_relation_annotation_has_exactly_one_hypothesis_argument(
    arguments: tuple[annotation_records.ArgumentRef, ...],
) -> None:
    annotation = annotation_records.Annotation(
        uuid=uuid_of("pair"),
        label="entails",
        arguments=arguments,
    )

    with pytest.raises(RecordSkippedError, match="exactly one hypothesis"):
        relation_pair_of(relation_layer(annotation), annotation)


def test_a_relation_annotation_names_the_hypothesis_by_record_reference() -> None:
    annotation = annotation_records.Annotation(
        uuid=uuid_of("pair"),
        label="entails",
        arguments=(
            annotation_records.ArgumentRef(
                role="hypothesis",
                target=defs_records.ObjectRef(localId=uuid_of("local")),
            ),
        ),
    )

    with pytest.raises(RecordSkippedError, match="no expression record"):
        relation_pair_of(relation_layer(annotation), annotation)


@pytest.mark.parametrize("relation", sorted(RELATION_NAMES))
def test_a_relation_graph_edge_runs_from_premise_to_hypothesis(
    relation: str,
) -> None:
    reading = relational_edge_of(relation_edge(relation, "annotator-a"))

    assert reading.pair == RelationPair(ItemId(TARGET), ItemId(HYPOTHESIS))
    assert reading.supervision == Relational(
        ItemId(HYPOTHESIS),
        relation,
        AnnotatorId("annotator-a"),
    )


def test_an_inference_judgment_assigns_a_relation_to_its_pair() -> None:
    pair = RelationPair(ItemId(TARGET), ItemId(HYPOTHESIS))
    supervision = pair_response_of(
        judgment(label="entails", target="at://layer/pair", object_id="pair"),
        inference_experiment(),
        pair,
        AnnotatorId("annotator-a"),
    )

    assert supervision == Relational(
        ItemId(HYPOTHESIS), "entails", AnnotatorId("annotator-a")
    )


@pytest.mark.parametrize(
    "held",
    [
        judgment(value=1, target="at://layer/pair", object_id="pair"),
        judgment_records.Judgment(
            item=defs_records.ObjectRef(
                recordRef="at://layer/pair", objectId=uuid_of("pair")
            ),
            continuousValue=judgment_records.DecimalValue(coefficient=73, exponent=-2),
        ),
    ],
)
def test_a_categorical_inference_judgment_rejects_a_numeric_arm(
    held: judgment_records.Judgment,
) -> None:
    with pytest.raises(RecordSkippedError, match="numeric arm"):
        pair_response_of(
            held,
            inference_experiment(),
            RelationPair(ItemId(TARGET), ItemId(HYPOTHESIS)),
            None,
        )


@pytest.mark.parametrize("relation", ["entailment", "contradiction", "unknown"])
def test_no_pair_route_accepts_a_relation_alias(relation: str) -> None:
    annotation = pair_annotation("pair", relation)
    paired = relation_layer(annotation)

    with pytest.raises(RecordSkippedError, match="canonical inference vocabulary"):
        relational_layer_of(paired, annotation)

    with pytest.raises(RecordSkippedError, match="canonical inference vocabulary"):
        relational_edge_of(relation_edge(relation))

    with pytest.raises(RecordSkippedError, match="canonical inference vocabulary"):
        pair_response_of(
            judgment(label=relation, target="at://layer/pair", object_id="pair"),
            inference_experiment(),
            RelationPair(ItemId(TARGET), ItemId(HYPOTHESIS)),
            None,
        )


def test_a_continuous_inference_judgment_reads_the_exact_decimal_arm_once() -> None:
    pair = RelationPair(ItemId(TARGET), ItemId(HYPOTHESIS))
    continuous = judgment_records.ExperimentDef(
        name="continuous inference",
        createdAt=CREATED,
        taskType="continuous",
        measureType="inference",
        scaleMin=0,
        scaleMax=1,
    )
    response = pair_response_of(
        judgment_records.Judgment(
            item=defs_records.ObjectRef(
                recordRef="at://layer/pair", objectId=uuid_of("pair")
            ),
            continuousValue=judgment_records.DecimalValue(coefficient=73, exponent=-2),
        ),
        continuous,
        pair,
        AnnotatorId("annotator-a"),
    )

    assert isinstance(response, RelationalScalar)
    assert response.other == pair.hypothesis
    assert response.relation == "entails"
    assert response.value == 0.73
    assert response.scale.kind == "continuous"
    assert response.annotator == "annotator-a"


def test_a_continuous_inference_judgment_rejects_the_integer_scalar_arm() -> None:
    continuous = judgment_records.ExperimentDef(
        name="continuous inference",
        createdAt=CREATED,
        taskType="continuous",
        measureType="inference",
        scaleMin=0,
        scaleMax=1,
    )

    with pytest.raises(RecordSkippedError, match="integer scalar arm"):
        pair_response_of(
            judgment(value=1, target="at://layer/pair", object_id="pair"),
            continuous,
            RelationPair(ItemId(TARGET), ItemId(HYPOTHESIS)),
            None,
        )


def test_a_continuous_inference_judgment_rejects_the_categorical_arm() -> None:
    continuous = judgment_records.ExperimentDef(
        name="continuous inference",
        createdAt=CREATED,
        taskType="continuous",
        measureType="inference",
        scaleMin=0,
        scaleMax=1,
    )

    with pytest.raises(RecordSkippedError, match="categorical arm"):
        pair_response_of(
            judgment(label="entails", target="at://layer/pair", object_id="pair"),
            continuous,
            RelationPair(ItemId(TARGET), ItemId(HYPOTHESIS)),
            None,
        )


def test_a_graph_edge_names_the_object_at_its_other_end() -> None:
    other = "at://did:plc:test/pub.layers.graph.graphNode/one"

    assert graph_start(graph_edge(TARGET, other), TARGET) == other
    assert graph_start(graph_edge(other, TARGET), TARGET) == other
    assert graph_start(graph_edge(other, other), TARGET) is None


def test_a_graph_lifts_the_nodes_and_the_edges_the_attached_one_reaches() -> None:
    nodes = {
        "at://n/one": graph_node("sleep", properties=(("tense", "present"),)),
        "at://n/two": graph_node("child", node_type="entity"),
        "at://n/away": graph_node("elsewhere"),
    }
    edges = {
        "at://e/one": graph_edge("at://n/one", "at://n/two", "agent", label="who"),
        "at://e/attach": graph_edge(TARGET, "at://n/one"),
    }
    graph = graph_of(GraphContext("at://n/one", nodes, edges))

    assert isinstance(graph, GoldGraph)
    assert [held.identifier for held in graph.nodes] == ["at://n/one", "at://n/two"]
    assert graph.nodes[0].node_type == "situation"
    assert graph.nodes[0].label == "sleep"
    assert dict(graph.nodes[0].properties) == {"tense": "present"}
    assert graph.nodes[1].node_type == "entity"
    assert [(edge.source, edge.target) for edge in graph.edges] == [
        ("at://n/one", "at://n/two")
    ]
    assert graph.edges[0].edge_type == "agent"
    assert graph.edges[0].label == "who"


def test_a_graph_is_reached_through_an_edge_in_either_direction() -> None:
    nodes = {"at://n/one": graph_node(), "at://n/two": graph_node("child")}
    edges = {"at://e/one": graph_edge("at://n/two", "at://n/one", "agent")}
    graph = graph_of(GraphContext("at://n/one", nodes, edges))

    assert [held.identifier for held in graph.nodes] == ["at://n/one", "at://n/two"]


def test_a_graph_leaves_out_a_node_no_edge_reaches() -> None:
    nodes = {"at://n/one": graph_node(), "at://n/loose": graph_node("nobody")}
    graph = graph_of(GraphContext("at://n/one", nodes, {}))

    assert [held.identifier for held in graph.nodes] == ["at://n/one"]
    assert graph.edges == ()


def test_a_graph_edge_leaving_the_graph_it_reaches_is_not_one_of_its_edges() -> None:
    nodes = {"at://n/one": graph_node()}
    edges = {"at://e/attach": graph_edge(TARGET, "at://n/one")}
    graph = graph_of(GraphContext("at://n/one", nodes, edges))

    assert [held.identifier for held in graph.nodes] == ["at://n/one"]
    assert graph.edges == ()


def test_a_graph_edge_whose_end_is_no_record_reaches_nothing() -> None:
    nodes = {"at://n/one": graph_node(), "at://n/two": graph_node("child")}
    edges = {
        "at://e/one": graph_records.GraphEdge(
            createdAt=CREATED,
            edgeType="agent",
            source=defs_records.ObjectRef(recordRef="at://n/one"),
            target=defs_records.ObjectRef(localId=uuid_of("two")),
        )
    }
    graph = graph_of(GraphContext("at://n/one", nodes, edges))

    assert [held.identifier for held in graph.nodes] == ["at://n/one"]
    assert graph.edges == ()


def test_an_edge_attaching_an_expression_to_no_graph_node_is_refused() -> None:
    with pytest.raises(RecordSkippedError, match="no graph node"):
        graph_of(GraphContext("at://n/absent", {}, {}))


def test_a_graph_node_stating_no_type_is_refused_by_the_value_it_becomes() -> None:
    nodes = {"at://n/one": graph_node(node_type="")}

    with pytest.raises(ValueError, match="no node type"):
        graph_of(GraphContext("at://n/one", nodes, {}))


def test_a_token_tag_layer_becomes_width_one_total_constraints() -> None:
    tagged = span_constraints_of(
        layer(
            "token-tag",
            (
                node("t-one", label="N", token=0),
                node("t-two", label="NP", token=0),
                node("t-three", label="S\\NP", token=1),
            ),
        ),
        2,
    )

    assert tagged == SpanConstraints.from_positions(
        (frozenset({"N", "NP"}), frozenset({"S\\NP"}))
    )


def test_a_token_tag_layer_carrying_confidence_is_refused_by_name() -> None:
    with pytest.raises(RecordSkippedError, match="token-tag layer carrying confidence"):
        span_constraints_of(layer("token-tag", (scored("t", "N", HALF, token=0),)), 1)


def test_a_token_tag_layer_leaving_a_position_untagged_is_refused() -> None:
    with pytest.raises(RecordSkippedError, match="untagged"):
        span_constraints_of(layer("token-tag", (node("t", label="N", token=0),)), 2)


def test_a_token_tag_layer_tagging_a_position_the_item_lacks_is_refused() -> None:
    with pytest.raises(RecordSkippedError, match="does not hold"):
        span_constraints_of(layer("token-tag", (node("t", label="N", token=5),)), 1)


def test_a_token_tag_layer_whose_tag_names_no_position_is_refused() -> None:
    with pytest.raises(RecordSkippedError, match="no position"):
        span_constraints_of(layer("token-tag", (node("t", label="N"),)), 1)


def reading(count: int, identifier: str | None = None) -> TokenReading:
    return TokenReading(identifier, count)


def well_formed(lift: LayerLift) -> annotation_records.AnnotationLayer:
    annotations = {
        "tree": (
            node("r", label="S", children=("l", "m")),
            node("l", label="NP", parent="r", token=0),
            node("m", label="S\\NP", parent="r", token=1),
        ),
        "relation": (arc("d", "nsubj", 1, 0),),
        "token-tag": (
            node("s", label="NP", token=0),
            node("t", label="S\\NP", token=1),
        ),
    }

    return layer(lift.kind, annotations[lift.kind], subkind=lift.subkind)


@pytest.mark.parametrize("lift", LAYER_LIFTS, ids=lambda lift: lift.kind)
def test_a_layer_lift_produces_the_arm_it_declares(lift: LayerLift) -> None:
    produced = lift.read(LayerContext(well_formed(lift), reading(2), None))

    assert type(produced) is lift.arm


@pytest.mark.parametrize("lift", LAYER_LIFTS, ids=lambda lift: lift.kind)
def test_the_dispatcher_reaches_the_lift_that_claims_a_layer(lift: LayerLift) -> None:
    produced = supervision_of(well_formed(lift), reading(2))

    assert type(produced) is lift.arm


def test_every_registered_lift_declares_an_arm_a_load_can_report() -> None:
    declared = [lift.arm for lift in LAYER_LIFTS] + [
        lift.arm for lift in RESPONSE_LIFTS
    ]

    assert set(declared) | {GRAPH_LIFT.arm, RELATION_LIFT.arm} <= set(EMITTED_ARMS)
    assert len(set(EMITTED_ARMS)) == len(EMITTED_ARMS)


def test_a_layer_of_an_unread_kind_is_refused() -> None:
    with pytest.raises(RecordSkippedError, match="kind the reader does not read"):
        supervision_of(layer("span", (node("s", label="X", token=0),)), reading(1))


def test_a_relation_layer_of_an_unread_subkind_is_refused() -> None:
    with pytest.raises(RecordSkippedError, match="subkind the reader does not read"):
        supervision_of(
            layer("relation", (arc("d", "coref", 1, 0),), subkind="coreference"),
            reading(2),
        )


def test_a_relation_layer_naming_no_subkind_at_all_is_refused() -> None:
    with pytest.raises(RecordSkippedError, match="subkind the reader does not read"):
        supervision_of(layer("relation", (arc("d", "nsubj", 1, 0),)), reading(2))


def test_a_layer_read_on_the_tokenization_it_names_is_lifted() -> None:
    tagged = supervision_of(
        layer("token-tag", (node("t", label="N", token=0),), tokenization="t-one"),
        reading(1, "t-one"),
    )

    assert isinstance(tagged, SpanConstraints)


def test_a_layer_naming_a_tokenization_the_item_does_not_hold_is_refused() -> None:
    with pytest.raises(RecordSkippedError, match="tokenization the item does not hold"):
        supervision_of(
            layer("token-tag", (node("t", label="N", token=0),), tokenization="other"),
            reading(1, "t-one"),
        )


def test_a_layer_naming_a_tokenization_of_an_unsegmented_item_is_refused() -> None:
    with pytest.raises(RecordSkippedError, match="tokenization the item does not hold"):
        supervision_of(
            layer("token-tag", (node("t", label="N", token=0),), tokenization="t-one"),
            reading(0),
        )


def test_the_absent_tokenization_reason_is_its_own_reason() -> None:
    tagged = (node("t", label="N", token=0),)
    cases = (
        layer("token-tag", tagged, tokenization="other"),
        layer("token-tag", (node("t", label="N", token=9),)),
        layer("token-tag", (node("t", label="N"),)),
        layer("span", tagged),
        layer("relation", (arc("d", "nsubj", 1, 0),)),
        layer("relation", (arc("d", None, 1, 0),), subkind="dependency"),
        layer("tree", (node("r", label=None, token=0),)),
    )
    reasons = []

    for case in cases:
        with pytest.raises(RecordSkippedError) as raised:
            supervision_of(case, reading(1, "t-one"))

        reasons.append(raised.value.reason)

    assert len(set(reasons)) == len(cases)


def test_an_ordinal_experiment_states_the_levels_a_fit_needs() -> None:
    scale = scale_of(experiment(task="ordinal-scale"))

    assert scale.kind == "ordinal"
    assert scale.levels == SEVEN
    assert scale.modulus is None


def test_a_yes_or_no_experiment_is_an_ordinal_scale_of_two_levels() -> None:
    scale = scale_of(experiment(task="binary", minimum=0, maximum=1))

    assert scale.kind == "ordinal"
    assert scale.levels == 2


def test_a_magnitude_experiment_reads_its_modulus_off_its_own_features() -> None:
    scale = scale_of(
        experiment(
            task="magnitude", minimum=0, maximum=None, features=(("modulus", "20"),)
        )
    )

    assert scale.kind == "ratio"
    assert scale.modulus == 20.0
    assert scale.maximum is None


def test_a_magnitude_experiment_whose_modulus_is_not_a_number_is_refused() -> None:
    with pytest.raises(RecordSkippedError, match="not a number"):
        scale_of(
            experiment(
                task="magnitude", minimum=0, maximum=None, features=(("modulus", "a"),)
            )
        )


def test_a_rating_experiment_stating_no_bounds_is_refused() -> None:
    with pytest.raises(RecordSkippedError, match="no bounds"):
        scale_of(experiment(task="ordinal-scale", maximum=None))


def test_a_continuous_experiment_supplies_its_bounds() -> None:
    scale = scale_of(experiment(task="continuous", minimum=0, maximum=100))

    assert scale.kind == "continuous"
    assert scale.minimum == 0.0
    assert scale.maximum == 100.0


def test_an_experiment_whose_responses_are_not_on_a_scale_is_refused() -> None:
    with pytest.raises(RecordSkippedError) as caught:
        scale_of(experiment(task="free-text"))

    assert caught.value.reason == LayersSkip.UNSCALED


def test_an_experiment_whose_bounds_describe_no_scale_is_refused() -> None:
    with pytest.raises(RecordSkippedError, match="no scale"):
        scale_of(experiment(task="ordinal-scale", minimum=7, maximum=1))


def test_a_response_arrives_on_the_scale_its_experiment_states() -> None:
    response = response_of(
        judgment(value=3), experiment(task="ordinal-scale"), AnnotatorId("a")
    )

    assert isinstance(response, Scalar)
    assert response.value == 3.0
    assert response.annotator == "a"
    assert response.scale.levels == SEVEN


def test_a_missing_rating_is_preserved_as_null() -> None:
    response = response_of(
        judgment(),
        experiment(task="ordinal-scale"),
        AnnotatorId("a"),
    )

    assert isinstance(response, Scalar)
    assert response.value is None
    assert response.features == frozenset({RESPONSE_MISSING})
    assert response.annotator == "a"


def test_a_response_the_scale_refuses_is_refused() -> None:
    with pytest.raises(RecordSkippedError, match="scale refuses"):
        response_of(judgment(value=9), experiment(task="ordinal-scale"), None)


@pytest.mark.parametrize("task", ["ordinal-scale", "binary", "magnitude"])
def test_an_integer_scale_response_rejects_the_continuous_arm(task: str) -> None:
    held = judgment_records.Judgment(
        item=defs_records.ObjectRef(recordRef=TARGET),
        continuousValue=judgment_records.DecimalValue(coefficient=73, exponent=-2),
    )

    with pytest.raises(RecordSkippedError, match="continuous arm"):
        response_of(held, experiment(task=task), None)


def test_a_continuous_response_checks_exact_decimal_bounds_before_float() -> None:
    held = judgment_records.Judgment(
        item=defs_records.ObjectRef(recordRef=TARGET),
        continuousValue=judgment_records.DecimalValue(
            coefficient=10_000_000_000_000_001,
            exponent=-16,
        ),
    )

    with pytest.raises(RecordSkippedError, match="scale refuses"):
        response_of(
            held,
            experiment(task="continuous", minimum=0, maximum=1),
            None,
        )


@pytest.mark.parametrize("task", ["ordinal-scale", "binary", "magnitude"])
def test_an_integer_scale_response_rejects_the_categorical_arm(task: str) -> None:
    with pytest.raises(RecordSkippedError, match="categorical arm"):
        response_of(
            judgment(label="yes"),
            experiment(task=task),
            None,
        )


def test_a_categorical_response_includes_its_label_inventory() -> None:
    response = response_of(
        judgment(label="yes"),
        experiment(task="categorical", labels=("yes", "no")),
        None,
    )

    assert isinstance(response, Categorical)
    assert response.label == "yes"
    assert response.label_set == frozenset({"yes", "no"})


@pytest.mark.parametrize(
    "held",
    [
        judgment(value=1),
        judgment_records.Judgment(
            item=defs_records.ObjectRef(recordRef=TARGET),
            continuousValue=judgment_records.DecimalValue(coefficient=73, exponent=-2),
        ),
    ],
)
def test_a_categorical_response_rejects_a_numeric_arm(
    held: judgment_records.Judgment,
) -> None:
    with pytest.raises(RecordSkippedError, match="numeric arm"):
        response_of(
            held,
            experiment(task="categorical", labels=("yes", "no")),
            None,
        )


def test_a_categorical_experiment_declaring_no_inventory_is_refused() -> None:
    with pytest.raises(RecordSkippedError, match="no inventory"):
        response_of(judgment(label="yes"), experiment(task="categorical"), None)


def test_a_label_outside_the_inventory_is_refused() -> None:
    with pytest.raises(RecordSkippedError, match="outside the inventory"):
        response_of(
            judgment(label="maybe"),
            experiment(task="categorical", labels=("yes", "no")),
            None,
        )


def test_a_missing_categorical_response_is_preserved_as_null() -> None:
    response = response_of(
        judgment(),
        experiment(task="categorical", labels=("yes",)),
        AnnotatorId("a"),
    )

    assert isinstance(response, Categorical)
    assert response.label is None
    assert response.features == frozenset({RESPONSE_MISSING})
    assert response.annotator == "a"


def test_an_unnamed_agent_gives_no_annotator() -> None:
    assert annotator_of(None) is None
    assert annotator_of("") is None
    assert annotator_of("a") == "a"


def test_a_response_inside_a_record_is_named_by_its_position_in_it() -> None:
    assert item_id_of(TARGET) == TARGET
    assert item_id_of(TARGET, 0) == f"{TARGET}#0"
    assert item_id_of(TARGET, 1) != item_id_of(TARGET, 0)


def placed(
    base: annotation_records.Annotation,
    *,
    anchor: defs_records.Anchor | None = None,
    type_ref: str | None = None,
    confidence: int | None = None,
) -> annotation_records.Annotation:
    return annotation_records.Annotation(
        uuid=base.uuid,
        label=base.label,
        ontologyTypeRef=type_ref,
        parentId=base.parentId,
        childIds=base.childIds,
        anchor=anchor,
        tokenIndex=None if anchor is not None else base.tokenIndex,
        confidence=confidence,
    )


def token_anchor(index: int, tokenization: str = "tok") -> defs_records.Anchor:
    return defs_records.Anchor(
        tokenRef=defs_records.TokenRef(
            tokenIndex=index, tokenizationId=uuid_of(tokenization)
        )
    )


def span_anchor(start: int, end: int) -> defs_records.Anchor:
    return defs_records.Anchor(textSpan=defs_records.Span(byteStart=start, byteEnd=end))


def chain(depth: int) -> annotation_records.AnnotationLayer:
    return layer(
        "tree",
        tuple(
            node(
                f"n{step}",
                parent=None if step == 0 else f"n{step - 1}",
                children=() if step == depth - 1 else (f"n{step + 1}",),
                token=0 if step == depth - 1 else None,
            )
            for step in range(depth)
        ),
    )


SAMPLE_TREE: Final = layer(
    "tree",
    (
        node("t0", label="S", children=("t1", "t4")),
        node("t1", label="NP", parent="t0", children=("t2", "t3")),
        node("t2", label="NP/N", parent="t1", token=0),
        node("t3", label="N", parent="t1", token=1),
        node("t4", label="S\\NP", parent="t0", token=2),
    ),
)

CATEGORIES: Final = Signature(
    "cat", ("cat",), (Operator("S", "cat"), Operator("NP", "cat"))
)

ONTOLOGY: Final = "at://did:plc:test/pub.layers.ontology.ontology/ccg"


def test_an_annotation_uuid_becomes_a_node_id_the_source_names() -> None:
    lift = lift_tree(SAMPLE_TREE)

    assert [source.uuid for source in lift.sources] == ["t0", "t1", "t2", "t3", "t4"]
    assert lift.derivation.root == 0
    assert lift.derivation.node(2).label == "NP/N"
    assert lift.derivation.node(2).span == TokenSpan(0, 1)
    assert lift.derivation.node(0).children == (1, 4)


def test_a_tree_layer_recording_one_identifier_twice_is_refused() -> None:
    with pytest.raises(RecordSkippedError, match="identifier twice"):
        lift_tree(
            layer("tree", (node("r", children=("r",)), node("r", parent="r", token=0)))
        )


def test_a_tree_layer_holding_a_node_its_root_does_not_reach_is_refused() -> None:
    with pytest.raises(RecordSkippedError, match="does not reach"):
        lift_tree(
            layer(
                "tree",
                (
                    node("r", children=("l",)),
                    node("l", parent="r", token=0),
                    node("aside", parent="l", token=1),
                ),
            )
        )


def test_a_tree_layer_naming_a_parent_it_does_not_hold_is_refused() -> None:
    with pytest.raises(RecordSkippedError, match="does not hold"):
        lift_tree(layer("tree", (node("r", parent="gone", token=0),)))


def test_a_node_two_annotations_claim_as_a_child_is_a_named_skip() -> None:
    with pytest.raises(RecordSkippedError, match="as a child twice"):
        lift_tree(
            layer(
                "tree",
                (
                    node("r", children=("m", "l")),
                    node("m", parent="r", children=("l",)),
                    node("l", parent="m", token=0),
                ),
            )
        )


def test_a_node_one_annotation_claims_as_a_child_twice_is_a_named_skip() -> None:
    with pytest.raises(RecordSkippedError, match="as a child twice"):
        lift_tree(
            layer(
                "tree",
                (
                    node("r", children=("l", "l")),
                    node("l", parent="r", token=0),
                ),
            )
        )


def test_lifting_a_tree_layer_and_lowering_it_returns_the_annotations() -> None:
    written = lower_tree(lift_tree(SAMPLE_TREE))

    assert len(written) == len(SAMPLE_TREE.annotations)
    assert written == SAMPLE_TREE.annotations


def test_the_round_trip_compares_content_and_not_cardinality() -> None:
    relabeled = layer(
        "tree",
        (
            node("t0", label="SBAR", children=("t1", "t4")),
            node("t1", label="NP", parent="t0", children=("t2", "t3")),
            node("t2", label="NP/N", parent="t1", token=0),
            node("t3", label="N", parent="t1", token=1),
            node("t4", label="S\\NP", parent="t0", token=2),
        ),
    )
    written = lower_tree(lift_tree(relabeled))

    assert len(written) == len(SAMPLE_TREE.annotations)
    assert written == relabeled.annotations
    assert written != SAMPLE_TREE.annotations


def test_a_leaf_anchored_by_a_token_reference_spans_that_position() -> None:
    anchored = layer(
        "tree",
        (
            node("r", label="S", children=("l", "m")),
            placed(node("l", label="NP", parent="r"), anchor=token_anchor(0)),
            placed(node("m", label="S\\NP", parent="r"), anchor=token_anchor(1)),
        ),
    )
    lift = lift_tree(anchored)

    assert lift.derivation.node(1).span == TokenSpan(0, 1)
    assert lift.derivation.node(0).span == TokenSpan(0, 2)
    assert lift.sources[1].tokenization == "tok"
    assert lower_tree(lift) == anchored.annotations


def test_two_co_present_anchor_variants_are_refused_rather_than_ranked() -> None:
    both = annotation_records.Annotation(
        uuid=uuid_of("l"),
        label="NP",
        tokenIndex=0,
        anchor=token_anchor(1),
    )

    with pytest.raises(RecordSkippedError, match="two anchor variants"):
        lift_tree(layer("tree", (both,)))

    crossed = defs_records.Anchor(
        tokenRef=token_anchor(0).tokenRef,
        textSpan=span_anchor(0, 3).textSpan,
    )

    with pytest.raises(RecordSkippedError, match="two anchor variants"):
        lift_tree(layer("tree", (placed(node("l", label="NP"), anchor=crossed),)))


def test_an_anchor_variant_that_is_no_token_reference_is_refused_by_name() -> None:
    with pytest.raises(RecordSkippedError, match="textSpan"):
        lift_tree(
            layer("tree", (placed(node("l", label="NP"), anchor=span_anchor(0, 3)),))
        )


def test_an_ontology_type_becomes_the_category_and_the_label_becomes_the_rule() -> None:
    typed = layer(
        "tree",
        (
            placed(node("r", label="ba", children=("l",)), type_ref=f"{ONTOLOGY}#S"),
            placed(
                node("l", label="lex", parent="r", token=0), type_ref=f"{ONTOLOGY}#NP"
            ),
        ),
    )
    lift = lift_tree(typed, SignatureLabels(CATEGORIES))

    assert lift.derivation.node(0).label == "S"
    assert lift.derivation.node(0).rule == "ba"
    assert lift.derivation.node(1).label == "NP"
    assert lift.derivation.node(1).rule == "lex"
    assert lift.sources[0].type_ref == f"{ONTOLOGY}#S"
    assert lower_tree(lift) == typed.annotations


def test_a_node_naming_no_ontology_type_takes_its_category_from_its_label() -> None:
    lift = lift_tree(SAMPLE_TREE, SignatureLabels(CATEGORIES))

    assert lift.derivation.node(0).label == "S"
    assert lift.derivation.node(0).rule is None
    assert lift.sources[0].type_ref is None


def test_an_ontology_reference_with_no_fragment_names_its_last_segment() -> None:
    lift = lift_tree(
        layer(
            "tree",
            (placed(node("r", label="lex", token=0), type_ref=f"{ONTOLOGY}/NP"),),
        ),
        SignatureLabels(CATEGORIES),
    )

    assert lift.derivation.node(0).label == "NP"


def test_an_ontology_type_the_label_inventory_does_not_admit_is_refused() -> None:
    with pytest.raises(RecordSkippedError, match="does not admit"):
        lift_tree(
            layer(
                "tree",
                (placed(node("r", label="lex", token=0), type_ref=f"{ONTOLOGY}#VP"),),
            ),
            SignatureLabels(CATEGORIES),
        )


def test_an_ontology_type_under_a_run_with_no_label_inventory_is_refused() -> None:
    with pytest.raises(RecordSkippedError, match="no label inventory"):
        lift_tree(
            layer(
                "tree",
                (placed(node("r", label="lex", token=0), type_ref=f"{ONTOLOGY}#S"),),
            )
        )


def test_an_ontology_reference_naming_no_type_is_refused() -> None:
    with pytest.raises(RecordSkippedError, match="naming no type"):
        lift_tree(
            layer(
                "tree",
                (placed(node("r", label="lex", token=0), type_ref=f"{ONTOLOGY}#"),),
            ),
            SignatureLabels(CATEGORIES),
        )


def test_a_confidence_is_a_share_of_the_ceiling_the_wire_states() -> None:
    scored_tree = layer(
        "tree",
        (
            placed(node("r", label="S", children=("l",)), confidence=HALF),
            node("l", parent="r", token=0),
        ),
    )
    lift = lift_tree(scored_tree)

    assert lift.derivation.confidence_of(0) == HALF / CONFIDENCE_CEILING
    assert lower_tree(lift) == scored_tree.annotations


def test_a_confidence_outside_the_wire_range_is_refused() -> None:
    for stated in (-1, CONFIDENCE_CEILING + 1):
        with pytest.raises(RecordSkippedError, match="outside the range"):
            lift_tree(
                layer(
                    "tree", (placed(node("r", label="S", token=0), confidence=stated),)
                )
            )


def test_the_bounds_of_the_wire_range_are_admitted() -> None:
    for stated, share in ((0, 0.0), (CONFIDENCE_CEILING, 1.0)):
        lift = lift_tree(
            layer("tree", (placed(node("r", label="S", token=0), confidence=stated),))
        )

        assert lift.derivation.confidence_of(0) == share


def test_a_deep_tree_lifts_without_a_recursion_error() -> None:
    lift = lift_tree(chain(DEEP))

    assert len(lift.derivation) == DEEP
    assert lift.derivation.node(0).span == TokenSpan(0, 1)


def test_a_deep_cyclic_parent_chain_is_a_named_skip_and_not_a_recursion_error() -> None:
    nodes = tuple(
        node(f"n{step}", parent=f"n{(step - 1) % DEEP}") for step in range(DEEP)
    )

    with pytest.raises(RecordSkippedError, match="cycle"):
        lift_tree(layer("tree", nodes))


BOX: Final = defs_records.BoundingBox(x=0, y=0, width=1, height=1)

ANCHOR_VARIANTS: Final = (
    ("boundingBox", defs_records.Anchor(boundingBox=BOX)),
    (
        "externalTarget",
        defs_records.Anchor(
            externalTarget=defs_records.ExternalTarget(source="at://did:plc:test/a/b")
        ),
    ),
    ("pageAnchor", defs_records.Anchor(pageAnchor=defs_records.PageAnchor(page=1))),
    (
        "signalSpan",
        defs_records.Anchor(
            signalSpan=defs_records.SignalSpan(scope=defs_records.MediaScope())
        ),
    ),
    (
        "spatialRegion",
        defs_records.Anchor(spatialRegion=defs_records.SpatialEntity()),
    ),
    (
        "spatioTemporalAnchor",
        defs_records.Anchor(
            spatioTemporalAnchor=defs_records.SpatioTemporalAnchor(
                temporalSpan=defs_records.TemporalSpan(start=0, ending=1)
            )
        ),
    ),
    (
        "temporalSpan",
        defs_records.Anchor(temporalSpan=defs_records.TemporalSpan(start=0, ending=1)),
    ),
    ("textSpan", span_anchor(0, 3)),
    (
        "tokenRefSequence",
        defs_records.Anchor(
            tokenRefSequence=defs_records.TokenRefSequence(
                tokenIndexes=(0,), tokenizationId=uuid_of("tok")
            )
        ),
    ),
)


def test_the_variants_enumerated_here_are_the_ones_the_wire_type_declares() -> None:
    named = {name for name, _ in ANCHOR_VARIANTS}

    assert named | {"tokenRef"} == set(defs_records.Anchor.__field_specs__)
    assert len(named) == len(ANCHOR_VARIANTS)


@pytest.mark.parametrize(("name", "anchor"), ANCHOR_VARIANTS)
def test_every_anchor_variant_that_is_no_token_reference_is_refused_by_name(
    name: str,
    anchor: defs_records.Anchor,
) -> None:
    with pytest.raises(RecordSkippedError, match=name):
        lift_tree(layer("tree", (placed(node("l", label="NP"), anchor=anchor),)))


@pytest.mark.parametrize("anchor", [anchor for _, anchor in ANCHOR_VARIANTS])
def test_every_anchor_variant_beside_a_token_index_is_refused_as_two_variants(
    anchor: defs_records.Anchor,
) -> None:
    beside = annotation_records.Annotation(
        uuid=uuid_of("l"),
        label="NP",
        tokenIndex=0,
        anchor=anchor,
    )

    with pytest.raises(RecordSkippedError, match="two anchor variants"):
        lift_tree(layer("tree", (beside,)))


def test_an_anchor_setting_no_variant_at_all_is_a_named_skip() -> None:
    with pytest.raises(RecordSkippedError, match="sets no variant"):
        lift_tree(
            layer(
                "tree",
                (placed(node("l", label="NP"), anchor=defs_records.Anchor()),),
            )
        )


def test_a_node_naming_a_parent_other_than_the_one_claiming_it_is_a_named_skip() -> (
    None
):
    disagreeing = layer(
        "tree",
        (
            node("r", label="S", children=("a", "b")),
            node("a", label="NP", parent="r", token=0),
            node("b", label="S\\NP", parent="a", token=1),
        ),
    )

    with pytest.raises(RecordSkippedError, match="references disagree"):
        lift_tree(disagreeing)


def test_children_declared_right_to_left_span_the_positions_they_cover() -> None:
    reversed_children = layer(
        "tree",
        (
            node("r", label="S", children=("b", "a")),
            node("a", label="NP", parent="r", token=0),
            node("b", label="S\\NP", parent="r", token=1),
        ),
    )
    lift = lift_tree(reversed_children)

    assert lift.derivation.node(0).children == (2, 1)
    assert lift.derivation.node(1).span == TokenSpan(0, 1)
    assert lift.derivation.node(2).span == TokenSpan(1, 2)
    assert lift.derivation.node(0).span == TokenSpan(0, 2)
    assert lower_tree(lift) == reversed_children.annotations


def featured(
    base: annotation_records.Annotation,
    entries: tuple[tuple[str, str], ...],
) -> annotation_records.Annotation:
    return annotation_records.Annotation(
        uuid=base.uuid,
        label=base.label,
        ontologyTypeRef=base.ontologyTypeRef,
        parentId=base.parentId,
        childIds=base.childIds,
        anchor=base.anchor,
        tokenIndex=base.tokenIndex,
        confidence=base.confidence,
        features=defs_records.FeatureMap(
            entries=tuple(
                defs_records.Feature(key=key, value=value) for key, value in entries
            )
        ),
    )


def test_a_node_stating_one_feature_key_twice_is_a_named_skip() -> None:
    with pytest.raises(RecordSkippedError, match="feature key twice"):
        lift_tree(
            layer(
                "tree",
                (featured(node("r", label="S", token=0), (("a", "1"), ("a", "2"))),),
            )
        )


def test_feature_entries_come_back_in_key_order_holding_what_they_stated() -> None:
    out_of_order = layer(
        "tree",
        (featured(node("r", label="S", token=0), (("b", "2"), ("a", "1"))),),
    )
    lift = lift_tree(out_of_order)
    written = lower_tree(lift)

    assert tuple(lift.derivation.node(0).features.items()) == (
        ("a", "1"),
        ("b", "2"),
    )
    assert written[0].features is not None
    assert [(entry.key, entry.value) for entry in written[0].features.entries] == [
        ("a", "1"),
        ("b", "2"),
    ]


def test_a_layer_stating_every_thing_an_annotation_can_survives_the_round_trip() -> (
    None
):
    rich = layer(
        "tree",
        (
            featured(
                placed(
                    node("r", label="ba", children=("l", "m")),
                    type_ref=f"{ONTOLOGY}#S",
                    confidence=HALF,
                ),
                (("head", "1"), ("slash", "none")),
            ),
            featured(
                placed(
                    node("l", label="lex", parent="r"),
                    anchor=token_anchor(0),
                    type_ref=f"{ONTOLOGY}#NP",
                    confidence=CONFIDENCE_CEILING,
                ),
                (("head", "0"),),
            ),
            placed(
                node("m", label="lex", parent="r", token=1),
                type_ref=f"{ONTOLOGY}#S",
                confidence=0,
            ),
        ),
    )
    lift = lift_tree(rich, SignatureLabels(CATEGORIES))

    assert lift.derivation.node(0).label == "S"
    assert lift.derivation.node(0).rule == "ba"
    assert tuple(lift.derivation.node(0).features.items()) == (
        ("head", "1"),
        ("slash", "none"),
    )
    assert lift.derivation.confidence_of(0) == HALF / CONFIDENCE_CEILING
    assert lift.derivation.confidence_of(1) == 1.0
    assert lift.derivation.confidence_of(2) == 0.0
    assert lift.sources[1].tokenization == "tok"
    assert lower_tree(lift) == rich.annotations


@given(confidence=st.integers(min_value=0, max_value=CONFIDENCE_CEILING))
def test_every_wire_confidence_the_range_admits_comes_back_as_itself(
    confidence: int,
) -> None:
    scored_layer = layer(
        "tree", (placed(node("r", label="S", token=0), confidence=confidence),)
    )

    assert lower_tree(lift_tree(scored_layer))[0].confidence == confidence


def test_a_share_falling_between_two_wire_numbers_is_written_as_the_nearest() -> None:
    nearest = 1
    below = 0.0006
    lift = TreeLift(
        Derivation(0, {0: DerivationNode(TokenSpan(0, 1), "S")}, {0: below}),
        (NodeSource("r", None, 0, None),),
    )

    assert below * CONFIDENCE_CEILING > nearest / 2
    assert lower_tree(lift)[0].confidence == nearest
