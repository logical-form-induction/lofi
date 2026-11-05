import subprocess
import sys
from pathlib import Path
from typing import Final

import pytest
from lairs.integrations.registry import get_codec

from lofi.data.corpora.ccgbank._auto import NAME, CcgbankAutoCodec
from lofi.data.corpora.ccgbank._auto_parse import (
    AutoDerivation,
    AutoLeaf,
    AutoTree,
    MalformedAutoError,
)

FIXTURES: Final = Path(__file__).with_name("fixtures")


def test_codec_resolves_by_declared_tag() -> None:
    assert get_codec(NAME) is CcgbankAutoCodec


def test_codec_resolves_by_declared_tag_in_a_fresh_process() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            (
                "from lairs.integrations.registry import get_codec; "
                "codec = get_codec('ccgbank-auto'); "
                "print(f'{codec.__module__}:{codec.__qualname__}')"
            ),
        ],
        capture_output=True,
        check=False,
        encoding="utf-8",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "lofi.data.corpora.ccgbank._auto:CcgbankAutoCodec"


@pytest.mark.parametrize("path", sorted(FIXTURES.glob("*.auto")), ids=lambda p: p.name)
def test_a_fixture_survives_decoding_and_encoding(path: Path) -> None:
    codec = CcgbankAutoCodec()
    source = path.read_text(encoding="utf-8")
    fragment = codec.decode(source)

    assert fragment.source == codec.name
    assert codec.decode(codec.encode(fragment.records)) == fragment


def test_a_leaf_round_trips_through_the_fragment() -> None:
    codec = CcgbankAutoCodec()
    source = "ID=one\n(<L N NN NN dogs N>)\n"

    parsed = codec.iso.backward(codec.decode(source))

    assert parsed == (
        AutoDerivation(
            (("ID", "one"),),
            AutoLeaf("N", "NN", "NN", "dogs", "N"),
        ),
    )


def test_parse_returns_auto_trees_without_the_fragment_round_trip() -> None:
    codec = CcgbankAutoCodec()
    source = "ID=one\n(<L N NN NN dogs N>)\n"

    assert codec.parse(source) == (
        AutoDerivation(
            (("ID", "one"),),
            AutoLeaf("N", "NN", "NN", "dogs", "N"),
        ),
    )
    assert codec.iso.backward(codec.decode(source)) == codec.parse(source)


def test_a_tree_preserves_heads_categories_and_words() -> None:
    codec = CcgbankAutoCodec()
    source = (
        "ID=one\n"
        "(<T S 1 2> (<L NP NNP NNP Kim NP>) "
        "(<L S\\NP VBD VBD left S_1\\NP_2>) )\n"
    )

    parsed = codec.iso.backward(codec.decode(source))

    assert parsed[0].tree == AutoTree(
        "S",
        1,
        (
            AutoLeaf("NP", "NNP", "NNP", "Kim", "NP"),
            AutoLeaf("S\\NP", "VBD", "VBD", "left", "S_1\\NP_2"),
        ),
    )


def test_decoding_into_a_fragment_keeps_existing_records_first() -> None:
    codec = CcgbankAutoCodec()
    first = codec.decode("ID=one\n(<L N NN NN dogs N>)\n")
    extended = codec.decode("ID=two\n(<L N NN NN cats N>)\n", into=first)

    assert extended.records[: len(first.records)] == first.records
    assert len(extended.records) == 2 * len(first.records)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("(<L N NN NN dogs N>)\n", "under no header"),
        ("ID=one\n", "heads no derivation"),
        ("ID=one\n(<T S 0 2> (<L N NN NN dogs N>) )\n", "2 daughters"),
    ],
)
def test_malformed_derivations_are_refused(source: str, message: str) -> None:
    with pytest.raises(MalformedAutoError, match=message):
        CcgbankAutoCodec().decode(source)
