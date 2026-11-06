from typing import TYPE_CHECKING, Final

import pytest
from lairs.integrations.codecs import CorpusFragment, FragmentRecord
from lairs.integrations.registry import get_codec

from lofi.data.corpora.ccgbank._auto import CcgbankAutoCodec
from lofi.data.corpora.ccgbank._parg import PargCodec
from lofi.data.corpora.fracas import FracasCodec
from lofi.data.corpora.layers import LayersExportCodec
from lofi.data.corpora.megaacceptability import MegaAcceptabilityCodec
from lofi.data.corpora.megaattitude import (
    MegaIntensionalityCodec,
    MegaNegRaisingCodec,
    MegaVeridicalityCodec,
)
from lofi.data.corpora.nli import ChaosNliCodec, MnliCodec, SnliCodec, UnliCodec
from lofi.data.corpora.pmb import PmbCcgCodec, PmbDrsCodec
from lofi.data.registration import declared_codec_tags, registers_as

if TYPE_CHECKING:
    from collections.abc import Iterable

PACKAGE: Final = "lofi."

NSID: Final = "pub.layers.expression.expression"

UNCLAIMED: Final = "registration-case"

DECLARED: Final = (
    ("ccgbank-auto", CcgbankAutoCodec),
    ("ccgbank-parg", PargCodec),
    ("chaosnli", ChaosNliCodec),
    ("fracas", FracasCodec),
    ("layers-export", LayersExportCodec),
    ("megaacceptability", MegaAcceptabilityCodec),
    ("megaintensionality", MegaIntensionalityCodec),
    ("meganegraising", MegaNegRaisingCodec),
    ("megaveridicality", MegaVeridicalityCodec),
    ("mnli", MnliCodec),
    ("pmb-ccg", PmbCcgCodec),
    ("pmb-drs", PmbDrsCodec),
    ("snli", SnliCodec),
    ("unli", UnliCodec),
)


class LineCodec:
    """A codec holding one record per nonempty line."""

    __slots__ = ("name",)

    name: str

    def __init__(self) -> None:
        self.name = UNCLAIMED

    def decode(
        self,
        src: str | bytes,
        *,
        into: CorpusFragment | None = None,
    ) -> CorpusFragment:
        """Return a fragment holding one record per nonempty line."""
        text = src.decode("utf-8") if isinstance(src, bytes) else src
        held = () if into is None else into.records

        return CorpusFragment(
            records=held
            + tuple(
                FragmentRecord(local_id=f"line-{n}", nsid=NSID, value_json=line)
                for n, line in enumerate(text.splitlines())
                if line
            ),
            source=UNCLAIMED,
        )

    def encode(self, records: Iterable[FragmentRecord]) -> str:
        """Return one line per record."""
        return "".join(f"{record.value_json}\n" for record in records)


def lofi_tags() -> tuple[str, ...]:
    """Return the declared tags whose codecs come from this distribution's packages."""
    return tuple(
        tag
        for tag in declared_codec_tags()
        if get_codec(tag).__module__.startswith(PACKAGE)
    )


def test_every_corpus_codec_declares_its_own_tag() -> None:
    assert lofi_tags() == tuple(tag for tag, _ in DECLARED)


def test_each_declared_tag_resolves_to_the_class_that_declared_it() -> None:
    assert tuple((tag, get_codec(tag)) for tag, _ in DECLARED) == DECLARED


def test_each_declaring_class_lives_in_the_package_whose_corpus_it_reads() -> None:
    assert all(
        codec.__module__.startswith("lofi.data.corpora.") for _, codec in DECLARED
    )


def test_a_second_codec_declaring_a_held_tag_fails_at_class_creation() -> None:
    with pytest.raises(TypeError, match="CcgbankAutoCodec already declared"):

        @registers_as("ccgbank-auto")
        class Rival(LineCodec):
            """A codec claiming a tag another codec holds."""


def test_a_refused_declaration_leaves_the_registry_as_it_was() -> None:
    before = declared_codec_tags()

    with pytest.raises(TypeError, match="'pmb-drs'"):

        @registers_as("pmb-drs")
        class Rival(LineCodec):
            """A codec claiming a tag another codec holds."""

    assert declared_codec_tags() == before
    assert get_codec("pmb-drs") is PmbDrsCodec


def test_an_empty_tag_is_refused() -> None:
    with pytest.raises(ValueError, match="tag is empty"):
        registers_as("")


def test_a_tag_that_is_not_a_string_is_refused() -> None:
    with pytest.raises(TypeError, match="tag is not a string"):
        registers_as(3)  # ty: ignore[invalid-argument-type]


def test_a_declaration_registers_the_class_and_returns_it_unchanged() -> None:
    declared = registers_as(UNCLAIMED)(LineCodec)

    assert declared is LineCodec
    assert get_codec(UNCLAIMED) is LineCodec
    assert UNCLAIMED in declared_codec_tags()
    assert UNCLAIMED not in lofi_tags()
