from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

import pytest
from hypothesis import strategies as st
from lairs.integrations.codecs import CorpusFragment, FragmentRecord
from lairs.records import expression as expression_records
from lairs.records import graph as graph_records
from lairs.records import segmentation as segmentation_records
from tests.conformance.codec import CodecConformance, CodecSubject

from lofi.data.corpora.layers._translate import GraphContext, graph_of
from lofi.data.corpora.pmb import (
    DRS_NAME,
    MalformedPmbDrsError,
    PmbClause,
    PmbDrs,
    PmbDrsCodec,
    PmbDrsFile,
)
from lofi.data.laws import CodecFixture, IsoFixture

if TYPE_CHECKING:
    from lofi.data import GoldGraph

type Subject = CodecSubject[PmbDrsFile, CorpusFragment, FragmentRecord]
type DrsRecord = (
    expression_records.Expression
    | graph_records.GraphEdge
    | graph_records.GraphNode
    | segmentation_records.Segmentation
)

FIXTURES: Final = Path(__file__).with_name("fixtures")
BASIC: Final = FIXTURES / "basic.clf"
SECOND: Final = """%%% It moved.
b1 REF x1 % It [0...2]
b1 REF e1 % moved [3...8]
b1 Theme e1 x1 % moved [3...8]
b1 move \"v.01\" e1 % moved [3...8]
% . [8...9]
"""
NARY: Final = """%%% One and two.
b0 CONTINUATION b1 b2 % and [4...7]
b1 REF x1 % One [0...3]
b1 one \"n.01\" x1 % One [0...3]
b2 REF x2 % two [8...11]
b2 two \"n.01\" x2 % two [8...11]
% . [11...12]
"""


def _fixture_text() -> str:
    """Return the hand-written clausal-form fixture."""
    return BASIC.read_text(encoding="utf-8")


def _records(fragment: CorpusFragment) -> tuple[FragmentRecord, ...]:
    """Return a fragment's records."""
    return fragment.records


def _model(record: FragmentRecord) -> DrsRecord:
    """Return the Layers model for one fragment record."""
    models = {
        "pub.layers.expression.expression": expression_records.Expression,
        "pub.layers.graph.graphEdge": graph_records.GraphEdge,
        "pub.layers.graph.graphNode": graph_records.GraphNode,
        "pub.layers.segmentation.segmentation": segmentation_records.Segmentation,
    }
    return models[record.nsid].model_validate_json(record.value_json)


def _properties(
    record: graph_records.GraphEdge | graph_records.GraphNode,
) -> dict[str, str]:
    """Return a graph record's properties by key."""
    assert record.properties is not None
    return {entry.key: entry.value for entry in record.properties.entries}


def _changed_property(
    fragment: CorpusFragment,
    role: str,
    key: str,
    value: str | None,
    *,
    duplicate: bool = False,
) -> CorpusFragment:
    """Return a fragment with one graph property changed.

    Parameters
    ----------
    fragment
        Source fragment.
    role
        Text selecting one record id.
    key
        Property key to change.
    value
        Replacement value, or None to remove the property.
    duplicate
        Whether to append rather than replace the property.
    """
    changed: list[FragmentRecord] = []

    for record in fragment.records:
        if role not in record.local_id:
            changed.append(record)
            continue

        payload = json.loads(record.value_json)
        entries = payload["properties"]["entries"]

        if duplicate:
            entries.append({"key": key, "value": value})
        else:
            held = any(entry["key"] == key for entry in entries)
            entries[:] = [
                {"key": key, "value": value}
                if entry["key"] == key and value is not None
                else entry
                for entry in entries
                if entry["key"] != key or value is not None
            ]

            if value is not None and not held:
                entries.append({"key": key, "value": value})

        model = (
            graph_records.GraphNode
            if record.nsid == "pub.layers.graph.graphNode"
            else graph_records.GraphEdge
        ).model_validate_json(json.dumps(payload))
        changed.append(
            FragmentRecord(
                local_id=record.local_id,
                nsid=record.nsid,
                value_json=model.model_dump_json(),
            )
        )

    return CorpusFragment(records=tuple(changed), source=fragment.source)


def _changed_endpoint(
    fragment: CorpusFragment,
    role: str,
    endpoint: Literal["source", "target"],
    record_ref: str,
) -> CorpusFragment:
    """Return a fragment with one graph endpoint changed.

    Parameters
    ----------
    fragment
        Source fragment.
    role
        Text selecting one record id.
    endpoint
        Endpoint to change.
    record_ref
        Replacement record URI.
    """
    changed: list[FragmentRecord] = []

    for record in fragment.records:
        if role not in record.local_id:
            changed.append(record)
            continue

        payload = json.loads(record.value_json)
        payload[endpoint]["recordRef"] = record_ref
        model = graph_records.GraphEdge.model_validate_json(json.dumps(payload))
        changed.append(
            FragmentRecord(
                local_id=record.local_id,
                nsid=record.nsid,
                value_json=model.model_dump_json(),
            )
        )

    return CorpusFragment(records=tuple(changed), source=fragment.source)


def _changed_json(
    fragment: CorpusFragment,
    role: str,
    value_json: str,
) -> CorpusFragment:
    """Return a fragment with one record's JSON changed.

    Parameters
    ----------
    fragment
        Source fragment.
    role
        Text selecting one record id.
    value_json
        Replacement record JSON.
    """
    return CorpusFragment(
        records=tuple(
            FragmentRecord(
                local_id=record.local_id,
                nsid=record.nsid,
                value_json=value_json,
            )
            if role in record.local_id
            else record
            for record in fragment.records
        ),
        source=fragment.source,
    )


def _gold_graph(fragment: CorpusFragment) -> GoldGraph:
    """Return the GoldGraph reached from a decoded fragment."""
    nodes: dict[str, graph_records.GraphNode] = {}
    edges: dict[str, graph_records.GraphEdge] = {}

    for record in fragment.records:
        model = _model(record)

        if isinstance(model, graph_records.GraphNode):
            nodes[record.local_id] = model
        elif isinstance(model, graph_records.GraphEdge):
            edges[record.local_id] = model

    attachment = next(edge for edge in edges.values() if edge.edgeType == "denotes")
    start = attachment.target.recordRef

    if start is None:
        msg = "the test attachment has no graph target"
        raise AssertionError(msg)

    return graph_of(GraphContext(start, nodes, edges))


class TestPmbDrsCodec(CodecConformance):
    @pytest.fixture
    def subject(self) -> Subject:
        """Return the codec conformance subject."""
        codec = PmbDrsCodec()
        sources = (_fixture_text(), SECOND, NARY, _fixture_text().encode("utf-8"))
        parsed = tuple(codec.iso.backward(codec.decode(source)) for source in sources)
        return CodecSubject(
            IsoFixture(
                codec.iso,
                st.sampled_from(parsed),
                st.sampled_from(tuple(codec.iso.forward(item) for item in parsed)),
                8,
            ),
            CodecFixture(codec, st.sampled_from(sources), _records, 8),
        )


def test_the_codec_has_its_public_name() -> None:
    assert PmbDrsCodec().name == DRS_NAME == "pmb-drs"


def test_clauses_and_token_comments_form_one_aligned_document() -> None:
    codec = PmbDrsCodec()
    document = codec.iso.backward(codec.decode(_fixture_text()))[0]

    assert document.text == "A café robot greeted Ada."
    assert tuple(token.surface for token in document.tokens) == (
        "A",
        "café",
        "robot",
        "greeted",
        "Ada",
        ".",
    )
    assert document.clauses[0].token_indices == (0, 2)
    assert document.clauses[-1].arguments == ("x2", '"ada"')


def test_character_offsets_become_utf8_byte_offsets() -> None:
    fragment = PmbDrsCodec().decode(_fixture_text())
    segmentation = next(
        model
        for record in fragment.records
        if isinstance((model := _model(record)), segmentation_records.Segmentation)
    )
    spans = tuple(
        (token.textSpan.byteStart, token.textSpan.byteEnd)
        for token in segmentation.tokenizations[0].tokens
        if token.textSpan is not None
    )

    assert spans == ((0, 1), (2, 7), (8, 13), (14, 21), (22, 25), (25, 26))


def test_every_variable_and_clause_has_reserved_grounding_properties() -> None:
    fragment = PmbDrsCodec().decode(_fixture_text())
    nodes = tuple(
        model
        for record in fragment.records
        if isinstance((model := _model(record)), graph_records.GraphNode)
    )
    edges = tuple(
        model
        for record in fragment.records
        if isinstance((model := _model(record)), graph_records.GraphEdge)
    )
    semantic = tuple(edge for edge in edges if edge.edgeType == "custom")
    attachments = tuple(edge for edge in edges if edge.edgeType == "denotes")

    assert len(nodes) == 5
    assert len(semantic) == 10
    assert len(attachments) == 1
    assert all(node.nodeType == "custom" for node in nodes)
    assert all(edge.label is not None for edge in semantic)
    assert all(
        _properties(node).get("pmb.tokenizationId") == "drs-tokenization-0"
        and any(key.startswith("pmb.tokenIndex.") for key in _properties(node))
        for node in nodes
    )
    assert all(
        _properties(edge).get("pmb.tokenizationId") == "drs-tokenization-0"
        and any(key.startswith("pmb.tokenIndex.") for key in _properties(edge))
        for edge in semantic
    )

    root = next(node for node in nodes if node.label == "b1")
    root_properties = _properties(root)
    assert root_properties["pmb.kind"] == "box"
    assert root_properties["pmb.variable"] == "b1"
    assert root_properties["pmb.tokenizationId"] == "drs-tokenization-0"
    assert {
        int(value)
        for key, value in root_properties.items()
        if key.startswith("pmb.tokenIndex.")
    } == {0, 1, 2, 3, 4}

    first = semantic[0]
    first_properties = _properties(first)
    assert first.label == "REF"
    assert first.ordinal == 0
    assert first_properties["pmb.box"] == "b1"
    assert first_properties["pmb.argumentCount"] == "1"
    assert first_properties["pmb.argument.00000000"] == "x1"
    assert {
        int(value)
        for key, value in first_properties.items()
        if key.startswith("pmb.tokenIndex.")
    } == {0, 2}
    assert (
        attachments[0].source.recordRef
        == "at://local/pub.layers.expression.expression/drs-0"
    )
    assert (
        attachments[0].target.recordRef
        == "at://local/pub.layers.graph.graphNode/drs-0-node-00000000"
    )


def test_nary_clauses_reach_every_node_through_the_gold_graph_lift() -> None:
    graph = _gold_graph(PmbDrsCodec().decode(NARY))
    semantic = tuple(edge for edge in graph.edges if edge.edge_type == "custom")
    connectivity = tuple(
        edge
        for edge in graph.edges
        if edge.properties.get("pmb.kind") == "connectivity"
    )

    assert {node.label for node in graph.nodes} == {"b0", "b1", "b2", "x1", "x2"}
    assert len(semantic) == 5
    assert len(connectivity) == 1
    continuation = next(edge for edge in semantic if edge.label == "CONTINUATION")
    assert continuation.properties["pmb.argumentCount"] == "2"
    assert continuation.properties["pmb.argument.00000000"] == "b1"
    assert continuation.properties["pmb.argument.00000001"] == "b2"


def test_encoding_returns_canonical_counter_input() -> None:
    codec = PmbDrsCodec()

    assert codec.encode(codec.decode(_fixture_text()).records) == _fixture_text()


def test_decoding_into_a_fragment_keeps_existing_records_first() -> None:
    codec = PmbDrsCodec()
    first = codec.decode(_fixture_text())
    extended = codec.decode(SECOND, into=first)

    assert extended.records[: len(first.records)] == first.records
    assert codec.iso.backward(extended) == (
        *codec.iso.backward(first),
        *codec.iso.backward(codec.decode(SECOND)),
    )


def test_multiple_documents_round_trip_in_source_order() -> None:
    codec = PmbDrsCodec()
    source = f"{_fixture_text().rstrip()}\n\n{SECOND}"
    fragment = codec.decode(source)

    assert tuple(document.text for document in codec.iso.backward(fragment)) == (
        "A café robot greeted Ada.",
        "It moved.",
    )
    assert codec.decode(codec.encode(fragment.records)) == fragment


def test_an_empty_parsed_file_is_refused() -> None:
    with pytest.raises(MalformedPmbDrsError, match="no documents"):
        PmbDrsCodec().iso.forward(())


def test_the_iso_refuses_noncanonical_token_position_order() -> None:
    codec = PmbDrsCodec()
    document = codec.iso.backward(codec.decode(_fixture_text()))[0]
    first = document.clauses[0]
    changed = PmbClause(
        first.box,
        first.operator,
        first.arguments,
        tuple(reversed(first.token_indices)),
    )
    noncanonical = PmbDrs(
        document.text,
        document.tokens,
        (changed, *document.clauses[1:]),
    )

    with pytest.raises(MalformedPmbDrsError, match="canonical order"):
        codec.iso.forward((noncanonical,))


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("b1 REF x1\n", "no token-alignment comment"),
        ("b1 REF x1 % token\n", "no token alignment"),
        ("b1 REF x1 % dog [0...2]\n", "invalid token alignment"),
        ('b1 REF "now" % now [0...3]\n', "no graph-node argument"),
        ('b1 REF "unfinished % dog [0...3]\n', "malformed quoted argument"),
        ("% dog [0...3]\n", "tokens but no clauses"),
    ],
)
def test_malformed_clausal_form_is_refused(source: str, message: str) -> None:
    """Check that malformed source text raises the named error.

    Parameters
    ----------
    source
        Malformed source text.
    message
        Expected error-message text.
    """
    with pytest.raises(MalformedPmbDrsError, match=message):
        PmbDrsCodec().decode(source)


@pytest.mark.parametrize(
    ("role", "key", "value", "duplicate_count", "message"),
    [
        ("clause-00000000", "pmb.tokenizationId", None, 0, "another tokenization"),
        (
            "clause-00000000",
            "pmb.tokenizationId",
            "other",
            0,
            "another tokenization",
        ),
        ("clause-00000000", "pmb.tokenIndex.00000000", "0", 1, "repeats"),
        ("clause-00000000", "pmb.tokenIndex.00000099", "99", 0, "outside"),
        ("clause-00000001", "pmb.tokenIndex.00000002", None, 0, "no token"),
        ("node-00000002", "pmb.tokenIndex.00000001", None, 0, "canonical"),
        ("node-00000000", "pmb.tokenIndex.00000001", None, 0, "canonical"),
        ("node-00000000", "pmb.tokenizationId", "other", 0, "canonical"),
    ],
)
def test_invalid_graph_grounding_is_refused(
    role: str,
    key: str,
    value: str | None,
    duplicate_count: int,
    message: str,
) -> None:
    """Check that invalid grounding properties are refused.

    Parameters
    ----------
    role
        Record-id text selecting the record to change.
    key
        Property key to change.
    value
        Replacement property value.
    duplicate_count
        One when the helper should duplicate the property.
    message
        Expected error-message text.
    """
    fragment = PmbDrsCodec().decode(_fixture_text())
    changed = _changed_property(
        fragment,
        role,
        key,
        value,
        duplicate=duplicate_count == 1,
    )

    with pytest.raises(MalformedPmbDrsError, match=message):
        PmbDrsCodec().iso.backward(changed)


def test_ordered_clause_argument_mutations_remain_in_order() -> None:
    codec = PmbDrsCodec()
    fragment = codec.decode(_fixture_text())
    changed = _changed_property(
        fragment,
        "clause-00000009",
        "pmb.argument.00000000",
        '"ada"',
    )
    changed = _changed_property(
        changed,
        "clause-00000009",
        "pmb.argument.00000001",
        "x2",
    )

    parsed = codec.iso.backward(changed)

    assert parsed[0].clauses[9].arguments == ('"ada"', "x2")
    assert 'b1 Name "ada" x2 % Ada [21...24]' in codec.encode(changed.records)


@pytest.mark.parametrize(
    ("endpoint", "record_ref"),
    [
        (
            "source",
            "at://local/pub.layers.graph.graphNode/drs-0-node-00000001",
        ),
        (
            "target",
            "at://local/pub.layers.graph.graphNode/drs-0-node-00000002",
        ),
    ],
)
def test_semantic_edge_endpoints_cannot_change(
    endpoint: Literal["source", "target"],
    record_ref: str,
) -> None:
    """Check that semantic-edge endpoint mutations are refused.

    Parameters
    ----------
    endpoint
        Endpoint to change.
    record_ref
        Replacement record URI.
    """
    codec = PmbDrsCodec()
    fragment = codec.decode(_fixture_text())
    changed = _changed_endpoint(
        fragment,
        "clause-00000000",
        endpoint,
        record_ref,
    )

    with pytest.raises(MalformedPmbDrsError, match="canonical"):
        codec.iso.backward(changed)


def test_attachment_target_cannot_change() -> None:
    codec = PmbDrsCodec()
    fragment = codec.decode(_fixture_text())
    changed = _changed_endpoint(
        fragment,
        "attachment",
        "target",
        "at://local/pub.layers.graph.graphNode/drs-0-node-00000001",
    )

    with pytest.raises(MalformedPmbDrsError, match="canonical"):
        codec.iso.backward(changed)


@pytest.mark.parametrize("value_json", ["{}", "{"])
def test_invalid_layers_json_is_wrapped(value_json: str) -> None:
    """Check that invalid model data raises the codec error.

    Parameters
    ----------
    value_json
        Invalid JSON or invalid graph-edge data.
    """
    codec = PmbDrsCodec()
    fragment = _changed_json(
        codec.decode(_fixture_text()),
        "clause-00000000",
        value_json,
    )

    with pytest.raises(MalformedPmbDrsError, match="invalid Layers record JSON"):
        codec.iso.backward(fragment)


def test_each_document_requires_exactly_one_attachment() -> None:
    codec = PmbDrsCodec()
    fragment = codec.decode(_fixture_text())
    without = CorpusFragment(
        records=tuple(
            record
            for record in fragment.records
            if not record.local_id.endswith("-attachment")
        ),
        source=fragment.source,
    )

    with pytest.raises(MalformedPmbDrsError, match="0 attachment records"):
        codec.iso.backward(without)


def test_non_utf8_input_is_refused() -> None:
    with pytest.raises(MalformedPmbDrsError, match="not UTF-8"):
        PmbDrsCodec().decode(b"\xff")
