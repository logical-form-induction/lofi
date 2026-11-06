"""CCGbank corpus-reader behavior and conformance."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

import pytest
from tests.conformance.corpus_reader import CorpusReaderConformance, CorpusReaderSubject

from lofi.data.corpora.ccgbank import parse_auto, parse_parg
from lofi.data.corpora.ccgbank._auto import CcgbankAutoCodec
from lofi.data.corpora.ccgbank._parg import PargCodec
from lofi.data.corpora.ccgbank.reader import CcgbankCorpusReader
from lofi.data.laws import ReaderFixture

if TYPE_CHECKING:
    from lairs.integrations.codecs import CorpusFragment

    from lofi.data.corpora.ccgbank._auto_parse import AutoFile
    from lofi.data.corpora.ccgbank._parg_parse import PargFile

_ONE = """ID=wsj_0200.1 PARSER=GOLD NUMPARSE=1
(<T S 0 2> (<L NP NNP NNP John NP>) (<L S\\NP VBZ VBZ sleeps S\\NP>) )
"""

_REPEATED = """ID=wsj_0200.2 PARSER=GOLD NUMPARSE=1
(<T S 0 2> (<L NP NNP NNP John NP>) (<L S\\NP VBZ VBZ sleeps S\\NP>) )
"""

_UNKNOWN = """ID=wsj_0000.1 PARSER=GOLD NUMPARSE=1
(<T S 0 2> (<L NP NNP NNP Alice NP>) (<L S\\NP VBZ VBZ sleeps S\\NP>) )
"""


def _write(root: Path, section: str, name: str, text: str) -> Path:
    path = root / "AUTO" / section / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


@pytest.fixture
def ccgbank(tmp_path: Path) -> Path:
    _write(tmp_path, "02", "wsj_0200.auto", _ONE + _REPEATED)
    _write(tmp_path, "00", "wsj_0000.auto", _UNKNOWN)
    return tmp_path


class TestCcgbankCorpusReader(CorpusReaderConformance):
    @pytest.fixture
    def subject(self, ccgbank: Path) -> CorpusReaderSubject[str, None]:
        return CorpusReaderSubject(
            ReaderFixture(lambda: CcgbankCorpusReader(ccgbank, splits=("train",))),
            "lofi.data.corpora.ccgbank",
        )


def test_sections_have_conventional_splits_and_stable_source_order(
    ccgbank: Path,
) -> None:
    reader = CcgbankCorpusReader(ccgbank, splits=("validation", "train"))

    assert reader.splits() == ("validation", "train")
    assert tuple(item.item_id for item in reader.items("train")) == (
        "wsj_0200.1",
        "wsj_0200.2",
    )


def test_unseen_held_out_words_are_skipped_and_reported_by_default(
    ccgbank: Path,
) -> None:
    reader = CcgbankCorpusReader(ccgbank, splits=("validation",))

    assert tuple(reader.examples("validation")) == ()
    assert reader.report().skips_by_reason == {
        "word absent from CCGbank training sections": 1
    }


def test_a_single_auto_file_is_its_own_training_vocabulary(ccgbank: Path) -> None:
    source = ccgbank / "AUTO" / "00" / "wsj_0000.auto"
    reader = CcgbankCorpusReader(source, splits=("train",))

    assert tuple(item.item_id for item in reader.items("train")) == ("wsj_0000.1",)


def test_an_unknown_category_can_be_assigned_by_declaration(ccgbank: Path) -> None:
    reader = CcgbankCorpusReader(
        ccgbank,
        splits=("validation",),
        unknown_words="assign",
        unknown_category="NP",
    )
    example = next(iter(reader.examples("validation")))
    leaves = tuple(
        node
        for node in example.supervision.derivation.nodes.values()
        if node.is_lexical
    )

    assert leaves[0].label == "NP"
    assert reader.report().skipped == 0


_FIXTURE_ROOTS: Final = (
    Path(__file__).with_name("fixtures"),
    Path(__file__).parents[3] / "fixtures",
    Path(__file__).parents[4] / "data" / "eval" / "bisk_hockenmaier_tiny",
    Path(__file__).parents[4] / "acceptance" / "documents" / "corpora",
)


def _fixture_files(suffix: str) -> tuple[Path, ...]:
    return tuple(
        path for root in _FIXTURE_ROOTS for path in sorted(root.rglob(f"*{suffix}"))
    )


def test_the_codec_route_reads_every_auto_fixture_as_its_direct_parse() -> None:
    codec = CcgbankAutoCodec()
    files = _fixture_files(".auto")

    assert files
    for path in files:
        source = path.read_bytes()
        assert codec.iso.backward(codec.decode(source)) == parse_auto(source), path


def test_the_codec_route_reads_every_parg_fixture_as_its_direct_parse() -> None:
    codec = PargCodec()
    files = tuple(path for path in _fixture_files(".parg") if path.read_bytes().strip())

    assert files
    for path in files:
        source = path.read_bytes()
        assert codec.iso.backward(codec.decode(source)) == parse_parg(source), path


def test_the_reader_parses_its_sources_through_the_registered_codecs(
    ccgbank: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parg = ccgbank / "PARG" / "02" / "wsj_0200.parg"
    parg.parent.mkdir(parents=True)
    parg.write_text('<s id="wsj_0200.1"> 0\n<\\s>\n', encoding="utf-8", newline="\n")
    parsed: list[str] = []
    auto_parse = CcgbankAutoCodec.parse
    parg_parse = PargCodec.parse

    def counted_auto(self: CcgbankAutoCodec, src: str | bytes, /) -> AutoFile:
        parsed.append(self.name)
        return auto_parse(self, src)

    def counted_parg(self: PargCodec, src: str | bytes, /) -> PargFile:
        parsed.append(self.name)
        return parg_parse(self, src)

    def refused(*_: object, **__: object) -> CorpusFragment:
        msg = "the reader built corpus records"
        raise AssertionError(msg)

    monkeypatch.setattr(CcgbankAutoCodec, "parse", counted_auto)
    monkeypatch.setattr(PargCodec, "parse", counted_parg)
    monkeypatch.setattr(CcgbankAutoCodec, "decode", refused)
    monkeypatch.setattr(PargCodec, "decode", refused)
    reader = CcgbankCorpusReader(ccgbank)

    assert set(parsed) == {"ccgbank-auto", "ccgbank-parg"}
    assert reader.report().records_seen > 0
