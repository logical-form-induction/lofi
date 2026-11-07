import json
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pytest
from hypothesis import strategies as st
from lairs.integrations.codecs import CorpusFragment, FragmentRecord
from tests.conformance.codec import CodecConformance, CodecSubject

from lofi.data.corpora.layers import (
    NAME,
    ExportLine,
    LayersExportCodec,
    LayersExportIso,
    MalformedExportError,
)
from lofi.data.laws import CodecFixture, IsoFixture

if TYPE_CHECKING:
    from lofi._types import JsonValue
    from lofi.data import CorpusCodec

type Subject = CodecSubject[tuple[ExportLine, ...], CorpusFragment, FragmentRecord]

EXAMPLES: Final = 25

SCHEME: Final = "at://"

SAMPLE: Final = (
    Path(__file__).resolve().parents[4] / "data" / "layers_sample" / "repository"
)

SEGMENTS: Final = st.text(
    alphabet=st.characters(codec="utf-8", exclude_characters="/"),
    min_size=1,
    max_size=4,
)

KEYS: Final = st.text(max_size=4)


def values() -> st.SearchStrategy[JsonValue]:
    return st.recursive(
        st.none()
        | st.booleans()
        | st.integers()
        | st.floats(allow_nan=False, allow_infinity=False)
        | st.text(max_size=4),
        lambda children: (
            st.lists(children, max_size=3) | st.dictionaries(KEYS, children, max_size=3)
        ),
        max_leaves=4,
    )


def objects() -> st.SearchStrategy[dict[str, JsonValue]]:
    return st.dictionaries(KEYS, values(), max_size=3)


def uris() -> st.SearchStrategy[str]:
    return st.builds(
        lambda authority, collection, key: f"{SCHEME}{authority}/{collection}/{key}",
        SEGMENTS,
        SEGMENTS,
        SEGMENTS,
    )


def lines() -> st.SearchStrategy[ExportLine]:
    return st.builds(
        lambda uri, value: {"uri": uri, "value": value},
        uris(),
        objects(),
    )


def records() -> st.SearchStrategy[FragmentRecord]:
    return st.builds(
        lambda authority, collection, key, value: FragmentRecord(
            local_id=f"{SCHEME}{authority}/{collection}/{key}",
            nsid=collection,
            value_json=json.dumps(value, sort_keys=True, separators=(",", ":")),
        ),
        SEGMENTS,
        SEGMENTS,
        SEGMENTS,
        objects(),
    )


def fragments() -> st.SearchStrategy[CorpusFragment]:
    return st.lists(records(), min_size=1, max_size=3).map(
        lambda drawn: CorpusFragment(records=tuple(drawn), source=NAME)
    )


def written(drawn: list[ExportLine]) -> str:
    return "".join(f"{json.dumps(line, sort_keys=True)}\n" for line in drawn)


def exports() -> st.SearchStrategy[str]:
    return st.lists(lines(), min_size=1, max_size=3).map(written)


def sources() -> st.SearchStrategy[str | bytes]:
    return st.one_of(exports(), exports().map(lambda text: text.encode("utf-8")))


def records_of(fragment: CorpusFragment) -> tuple[FragmentRecord, ...]:
    return fragment.records


def collection_of(uri: str) -> str:
    return uri.removeprefix(SCHEME).split("/")[1]


def sample() -> dict[str, JsonValue]:
    index = json.loads((SAMPLE / "index.json").read_text(encoding="utf-8"))

    return {
        uri: json.loads((SAMPLE / "records" / f"{digest}.json").read_text("utf-8"))
        for uri, digest in index.items()
    }


def sample_export() -> str:
    return written([{"uri": uri, "value": value} for uri, value in sample().items()])


class TestLayersExport(CodecConformance):
    @pytest.fixture
    def subject(self) -> Subject:
        codec = LayersExportCodec()

        return CodecSubject(
            IsoFixture(
                codec.iso,
                st.lists(lines(), min_size=1, max_size=3).map(tuple),
                fragments(),
                EXAMPLES,
            ),
            CodecFixture(codec, sources(), records_of, EXAMPLES),
        )


def as_the_port() -> CorpusCodec[tuple[ExportLine, ...]]:
    return LayersExportCodec()


def test_the_codec_is_usable_as_the_port_a_reader_takes() -> None:
    codec = as_the_port()
    decoded = codec.decode('{"uri": "at://a/b/c", "value": {}}')

    assert codec.name == NAME
    assert codec.iso.backward(decoded) == ({"uri": "at://a/b/c", "value": {}},)


def test_every_record_of_the_committed_sample_decodes_under_its_own_collection() -> (
    None
):
    expected = sample()
    decoded = LayersExportCodec().decode(sample_export())

    assert expected
    assert len(decoded.records) == len(expected)

    for record in decoded.records:
        assert record.local_id in expected
        assert record.nsid == collection_of(record.local_id)
        assert json.loads(record.value_json) == expected[record.local_id]


def test_the_committed_sample_survives_being_written_and_read_again() -> None:
    codec = LayersExportCodec()
    decoded = codec.decode(sample_export())

    assert codec.decode(codec.encode(decoded.records)) == decoded


def test_the_committed_sample_reads_the_same_from_bytes_as_from_text() -> None:
    codec = LayersExportCodec()
    export = sample_export()

    assert codec.decode(export.encode("utf-8")) == codec.decode(export)


def test_two_spellings_of_one_record_decode_to_one_record() -> None:
    codec = LayersExportCodec()
    one = '{"uri": "at://a/pub.layers.expression/x", "value": {"a": 1, "b": 2}}'
    other = '{"value": {"b": 2, "a": 1}, "uri": "at://a/pub.layers.expression/x"}'

    assert codec.decode(one) == codec.decode(other)


def test_a_value_holding_a_line_separator_still_writes_one_line() -> None:
    codec = LayersExportCodec()
    decoded = codec.decode(
        json.dumps(
            {"uri": "at://a/pub.layers.expression/x", "value": {"t": "a\u2028b"}}
        )
    )
    export = codec.encode(decoded.records)

    assert len(export.splitlines()) == 1
    assert codec.decode(export) == decoded


def test_decoding_into_a_fragment_leaves_it_as_it_was() -> None:
    codec = LayersExportCodec()
    first = codec.decode('{"uri": "at://a/b/one", "value": {}}')
    before = first.records
    extended = codec.decode('{"uri": "at://a/b/two", "value": {}}', into=first)

    assert first.records == before
    assert extended.records == (
        *before,
        *codec.decode('{"uri": "at://a/b/two", "value": {}}').records,
    )


def test_a_blank_line_is_not_a_missing_record() -> None:
    codec = LayersExportCodec()
    packed = (
        '{"uri": "at://a/b/one", "value": {}}\n{"uri": "at://a/b/two", "value": {}}'
    )
    spaced = (
        '{"uri": "at://a/b/one", "value": {}}\n\n{"uri": "at://a/b/two", "value": {}}\n'
    )

    assert codec.decode(spaced) == codec.decode(packed)


@pytest.mark.parametrize(
    "line",
    [
        "not json at all",
        '["a record"]',
        '{"uri": "at://a/b/c"}',
        '{"uri": "at://a/b/c", "value": {}, "cid": "z"}',
        '{"uri": 3, "value": {}}',
        '{"uri": "https://a/b/c", "value": {}}',
        '{"uri": "at://a/b", "value": {}}',
        '{"uri": "at://a//c", "value": {}}',
        '{"uri": "at://a/b/c/d", "value": {}}',
        '{"uri": "at://a/b/c", "value": 3}',
    ],
)
def test_a_line_that_is_not_a_record_is_refused(line: str) -> None:
    with pytest.raises(MalformedExportError):
        LayersExportCodec().decode(line)


def test_a_record_whose_value_is_not_json_is_refused_on_the_way_out() -> None:
    broken = FragmentRecord(local_id="at://a/b/c", nsid="b", value_json="{")

    with pytest.raises(MalformedExportError, match="not JSON"):
        LayersExportCodec().encode([broken])


def test_the_codec_reports_the_name_its_fragments_are_sourced_under() -> None:
    codec = LayersExportCodec()

    assert codec.name == NAME
    assert codec.decode('{"uri": "at://a/b/c", "value": {}}').source == NAME


def test_the_translation_is_the_one_the_codec_decodes_through() -> None:
    codec = LayersExportCodec()
    line: ExportLine = {"uri": "at://a/b/c", "value": {"n": 1}}

    assert isinstance(codec.iso, LayersExportIso)
    assert codec.iso.forward((line,)) == codec.decode(json.dumps(line))
    assert codec.iso.backward(codec.iso.forward((line,))) == (line,)
