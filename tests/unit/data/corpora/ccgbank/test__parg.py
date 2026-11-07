import subprocess
import sys
from pathlib import Path
from string import ascii_letters, digits
from typing import Final

import pytest
from hypothesis import strategies as st
from lairs.integrations.codecs import CorpusFragment, FragmentRecord
from lairs.integrations.registry import get_codec
from lairs.records import annotation as annotation_records
from tests.conformance.codec import CodecConformance, CodecSubject

from lofi.data.corpora.ccgbank._parg import PARG_NAME, PargCodec, PargIso
from lofi.data.corpora.ccgbank._parg_parse import (
    MalformedPargError,
    PargDependency,
    PargFile,
    PargSentence,
)
from lofi.data.laws import CodecFixture, IsoFixture

FIXTURES: Final = Path(__file__).with_name("fixtures")
EXAMPLES: Final = 25
IDENTIFIER_ALPHABET: Final = ascii_letters + digits + "._-"
FIELD_ALPHABET: Final = ascii_letters + digits + "()[]\\/:.,*_-"

type Subject = CodecSubject[PargFile, CorpusFragment, FragmentRecord]


def dependencies(last_index: int) -> st.SearchStrategy[PargDependency]:
    """Generate valid dependencies for one header bound."""
    field = st.text(FIELD_ALPHABET, min_size=1, max_size=8)

    return st.builds(
        PargDependency,
        target_index=st.integers(min_value=0, max_value=last_index),
        head_index=st.integers(min_value=0, max_value=last_index),
        head_category=field,
        argument_slot=st.integers(min_value=1, max_value=4),
        target_word=field,
        head_word=field,
        marker=st.one_of(st.none(), st.sampled_from(("<XB>", "<XU>"))),
    )


@st.composite
def sentences(draw: st.DrawFn) -> PargSentence:
    """Generate a nonempty synthetic PARG sentence."""
    last_index = draw(st.integers(min_value=0, max_value=8))

    return PargSentence(
        identifier=draw(st.text(IDENTIFIER_ALPHABET, min_size=1, max_size=12)),
        last_index=last_index,
        dependencies=tuple(
            draw(st.lists(dependencies(last_index), min_size=1, max_size=4))
        ),
    )


def parg_files() -> st.SearchStrategy[PargFile]:
    """Generate nonempty parsed PARG files."""
    return st.lists(sentences(), min_size=1, max_size=3).map(tuple)


def written(source: PargFile) -> str:
    """Write synthetic PARG data independently of the codec."""
    lines: list[str] = []

    for sentence in source:
        lines.append(f'<s id="{sentence.identifier}"> {sentence.last_index}')

        for dependency in sentence.dependencies:
            fields = [
                str(dependency.target_index),
                str(dependency.head_index),
                dependency.head_category,
                str(dependency.argument_slot),
                dependency.target_word,
                dependency.head_word,
            ]

            if dependency.marker is not None:
                fields.append(dependency.marker)

            lines.append("\t".join(fields))

        lines.append("<\\s>")

    return "\n".join(lines) + "\n"


def sources() -> st.SearchStrategy[str | bytes]:
    """Generate nonempty PARG text and bytes."""
    text = parg_files().map(written)

    return st.one_of(text, text.map(lambda source: source.encode("utf-8")))


def records_of(fragment: CorpusFragment) -> tuple[FragmentRecord, ...]:
    """Return a fragment's records."""
    return fragment.records


class TestPargCodec(CodecConformance):
    @pytest.fixture
    def subject(self) -> Subject:
        iso = PargIso()
        parsed = parg_files()

        return CodecSubject(
            IsoFixture(iso, parsed, parsed.map(iso.forward), EXAMPLES),
            CodecFixture(PargCodec(), sources(), records_of, EXAMPLES),
        )


def test_codec_resolves_by_declared_tag() -> None:
    assert get_codec(PARG_NAME) is PargCodec


def test_codec_resolves_by_declared_tag_in_a_fresh_process() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            (
                "from lairs.integrations.registry import get_codec; "
                "codec = get_codec('ccgbank-parg'); "
                "print(f'{codec.__module__}:{codec.__qualname__}')"
            ),
        ],
        capture_output=True,
        check=False,
        encoding="utf-8",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "lofi.data.corpora.ccgbank._parg:PargCodec"


@pytest.mark.parametrize("path", sorted(FIXTURES.glob("*.parg")), ids=lambda p: p.name)
def test_a_fixture_survives_decoding_and_encoding(path: Path) -> None:
    codec = PargCodec()
    source = path.read_text(encoding="utf-8")
    fragment = codec.decode(source)

    assert fragment.source == codec.name
    assert codec.decode(codec.encode(fragment.records)) == fragment


def test_the_parsed_shape_preserves_every_wire_field() -> None:
    source = (
        '<s id="synthetic.one"> 3\n'
        "3 1 N/N 2 quickly runs <XB>\n"
        "0 2 S\\NP 1 birds fly <XU>\n"
        "<\\s>\n"
    )
    codec = PargCodec()

    assert codec.parse(source) == codec.iso.backward(codec.decode(source))
    assert codec.parse(source.encode()) == codec.parse(source)
    assert codec.parse(source) == (
        PargSentence(
            identifier="synthetic.one",
            last_index=3,
            dependencies=(
                PargDependency(
                    target_index=3,
                    head_index=1,
                    head_category="N/N",
                    argument_slot=2,
                    target_word="quickly",
                    head_word="runs",
                    marker="<XB>",
                ),
                PargDependency(
                    target_index=0,
                    head_index=2,
                    head_category="S\\NP",
                    argument_slot=1,
                    target_word="birds",
                    head_word="fly",
                    marker="<XU>",
                ),
            ),
        ),
    )


def test_the_fragment_uses_a_dependency_relation_layer() -> None:
    source = '<s id="synthetic.one"> 1\n0 1 N 1 dogs bark\n<\\s>\n'
    fragment = PargCodec().decode(source)
    layer_record = next(
        record
        for record in fragment.records
        if record.nsid == "pub.layers.annotation.annotationLayer"
    )
    layer = annotation_records.AnnotationLayer.model_validate_json(
        layer_record.value_json
    )
    dependency = layer.annotations[0]
    feature_map = dependency.features

    assert feature_map is not None

    features = {entry.key: entry.value for entry in feature_map.entries}

    assert (layer.kind, layer.subkind, layer.formalism) == (
        "relation",
        "dependency",
        "ccgbank",
    )
    assert (dependency.targetIndex, dependency.headIndex) == (0, 1)
    assert features == {
        "argumentSlot": "1",
        "headLexicalCategory": "N",
        "headWord": "bark",
        "targetWord": "dogs",
    }


def test_decoding_into_a_fragment_keeps_existing_records_first() -> None:
    codec = PargCodec()
    first = codec.decode('<s id="synthetic.one"> 0\n<\\s>\n')
    extended = codec.decode('<s id="synthetic.two"> 0\n<\\s>\n', into=first)

    assert extended.records[: len(first.records)] == first.records
    assert len(extended.records) == 2 * len(first.records)
    assert codec.decode(codec.encode(extended.records)) == extended


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("0 1 N 1 dogs bark\n", "outside a sentence"),
        ('<s id="one"> 1\n', "not closed"),
        ("<\\s>\n", "never opened"),
        ('<s id="one"> nope\n<\\s>\n', "not a PARG sentence header"),
        ('<s id="one"> 1\n0 1 N 1 dogs\n<\\s>\n', "rather than six or seven"),
        ('<s id="one"> 1\nzero 1 N 1 dogs bark\n<\\s>\n', "target index"),
        ('<s id="one"> 1\n0 1 N 0 dogs bark\n<\\s>\n', "slots start at one"),
        ('<s id="one"> 1\n2 1 N 1 dogs bark\n<\\s>\n', "outside the header"),
        ('<s id="one"> 1\n0 1 N 1 dogs bark <XX>\n<\\s>\n', "unknown marker"),
    ],
)
def test_malformed_parg_text_is_refused(source: str, message: str) -> None:
    with pytest.raises(MalformedPargError, match=message):
        PargCodec().decode(source)


def test_invalid_parsed_state_is_refused() -> None:
    invalid = (
        PargSentence(
            "synthetic.one",
            1,
            (PargDependency(0, 1, "N", 0, "dogs", "bark"),),
        ),
    )

    with pytest.raises(MalformedPargError, match="slots start at one"):
        PargCodec().iso.forward(invalid)
