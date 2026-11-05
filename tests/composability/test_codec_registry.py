"""Every LoFI codec resolves through the shared registry, builds and holds its laws."""

import subprocess
import sys
from importlib import import_module
from importlib.metadata import entry_points
from pathlib import Path
from pkgutil import walk_packages
from typing import TYPE_CHECKING, Final, NamedTuple

import pytest
from hypothesis import strategies as st
from lairs.integrations.codecs import CorpusFragment, FragmentRecord
from lairs.integrations.ports import Codec
from lairs.integrations.registry import get_codec, register_codec
from tests.unit.data.corpora.megaattitude.test__codec import (
    INTENSIONALITY,
    NEG_RAISING,
    VERIDICALITY,
)
from tests.unit.data.corpora.megaattitude.test__codec import (
    specification as megaattitude_specification,
)

from lofi.data import corpora
from lofi.data.corpora.ccgbank._auto import CcgbankAutoCodec
from lofi.data.corpora.ccgbank._parg import PargCodec
from lofi.data.corpora.fracas import FracasCodec
from lofi.data.corpora.layers import LayersExportCodec
from lofi.data.corpora.megaacceptability import (
    MegaAcceptabilityCodec,
    MegaAcceptabilitySpec,
)
from lofi.data.corpora.megaattitude import (
    MegaIntensionalityCodec,
    MegaNegRaisingCodec,
    MegaVeridicalityCodec,
)
from lofi.data.corpora.nli import (
    ChaosNliCodec,
    MnliCodec,
    SnliCodec,
    UnliCodec,
)
from lofi.data.corpora.pmb import PmbCcgCodec, PmbDrsCodec
from lofi.data.laws import CodecFixture, check_codec_laws
from lofi.data.registration import declared_codec_tags
from lofi.spec.datasets import CorpusReleaseSpec

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from importlib.metadata import EntryPoint

type RegisteredCodec = Codec[CorpusFragment, FragmentRecord]

TESTS: Final = Path(__file__).resolve().parents[1]
TREE: Final = TESTS.parent
UNIT: Final = TESTS / "unit/data/corpora"

GROUP: Final = "lairs.codecs"
DISTRIBUTION: Final = "lofi"
PACKAGE: Final = "lofi."

EXAMPLES: Final = 8

FOREIGN_TAG: Final = "third-party-corpus"

EPRINT: Final = "at://did:plc:test/pub.layers.eprint.eprint/registry"

LAYERS_EXPORT: Final = (
    '{"uri": "at://did:plc:test/pub.layers.expression.expression/e0", '
    '"value": {"text": "Birds sing."}}\n'
    '{"uri": "at://did:plc:test/pub.layers.expression.expression/e1", '
    '"value": {"text": "Birds sings."}}\n'
)

RESOLVES_WITHOUT_LOFI: Final = """
import sys

from lairs.integrations.registry import get_codec

if any(name == "lofi" or name.startswith("lofi.") for name in sys.modules):
    raise SystemExit("a lofi module was imported before discovery ran")

for tag in sys.argv[1:]:
    print(tag, get_codec(tag).__module__)
"""


def nli_specification() -> CorpusReleaseSpec:
    """Return a synthetic NLI release declaration."""
    return CorpusReleaseSpec(
        source_id="synthetic-nli",
        release_id="v1",
        split="test",
        language="en",
        source_method="manual",
        spdx="CC0-1.0",
        eprint_refs=(EPRINT,),
    )


def fracas_specification() -> CorpusReleaseSpec:
    """Return a synthetic FraCaS release declaration."""
    return CorpusReleaseSpec(
        source_id="synthetic-fracas",
        release_id="v1",
        split="test",
        language="en",
        source_method="manual",
        spdx="CC0-1.0",
        eprint_refs=(EPRINT,),
    )


def megaacceptability_specification() -> MegaAcceptabilitySpec:
    """Return a synthetic MegaAcceptability release declaration."""
    return MegaAcceptabilitySpec(
        source_id="synthetic-megaacceptability",
        release_id="v2",
        spdx="CC0-1.0",
        scale_min=1,
        scale_max=7,
        frame_refs={
            "NP _ that S": "at://did:plc:test/pub.layers.resource.template/that-s",
            "NP _ NP to VP": "at://did:plc:test/pub.layers.resource.template/np-to-vp",
            "NP _ to VP": "at://did:plc:test/pub.layers.resource.template/to-vp",
        },
        verb_refs={
            verb: f"at://did:plc:test/pub.layers.resource.entry/{verb}"
            for verb in (
                "believe",
                "think",
                "know",
                "say",
                "want",
                "expect",
                "persuade",
                "promise",
                "try",
                "manage",
            )
        },
        eprint_refs=(EPRINT,),
    )


def fixture_text(*names: str) -> tuple[str, ...]:
    """Return the text of one corpus's sample files, in the order named."""
    return tuple((UNIT / name).read_text(encoding="utf-8") for name in names)


def records_of(fragment: CorpusFragment) -> tuple[FragmentRecord, ...]:
    """Return the records a fragment holds."""
    return fragment.records


class CodecCase(NamedTuple):
    """One registered codec, the class it resolves to, and sources it decodes.

    Parameters
    ----------
    tag : str
        The tag the codec is registered under.
    codec : type[RegisteredCodec]
        The class the registry is expected to hand back.
    build : Callable[[], RegisteredCodec]
        Construct the codec, supplying whatever release declaration it takes.
    sources : tuple[str, ...]
        Sources the codec finds records in.
    """

    tag: str
    codec: type[RegisteredCodec]
    build: Callable[[], RegisteredCodec]
    sources: tuple[str, ...]


CASES: Final = (
    CodecCase(
        "ccgbank-auto",
        CcgbankAutoCodec,
        CcgbankAutoCodec,
        fixture_text(
            "ccgbank/fixtures/features_and_unary.auto",
            "ccgbank/fixtures/punctuation.auto",
        ),
    ),
    CodecCase(
        "ccgbank-parg",
        PargCodec,
        PargCodec,
        fixture_text("ccgbank/fixtures/simple.parg", "ccgbank/fixtures/markers.parg"),
    ),
    CodecCase(
        "chaosnli",
        ChaosNliCodec,
        lambda: ChaosNliCodec(nli_specification()),
        fixture_text("nli/fixtures/chaosnli.jsonl"),
    ),
    CodecCase(
        "fracas",
        FracasCodec,
        lambda: FracasCodec(fracas_specification()),
        fixture_text("fracas/fixtures/sample.xml"),
    ),
    CodecCase("layers-export", LayersExportCodec, LayersExportCodec, (LAYERS_EXPORT,)),
    CodecCase(
        "megaacceptability",
        MegaAcceptabilityCodec,
        lambda: MegaAcceptabilityCodec(megaacceptability_specification()),
        fixture_text("megaacceptability/fixtures/sample.tsv"),
    ),
    CodecCase(
        "megaintensionality",
        MegaIntensionalityCodec,
        lambda: MegaIntensionalityCodec(megaattitude_specification("v1")),
        (INTENSIONALITY,),
    ),
    CodecCase(
        "meganegraising",
        MegaNegRaisingCodec,
        lambda: MegaNegRaisingCodec(megaattitude_specification("v1")),
        (NEG_RAISING,),
    ),
    CodecCase(
        "megaveridicality",
        MegaVeridicalityCodec,
        lambda: MegaVeridicalityCodec(megaattitude_specification("v2.1")),
        (VERIDICALITY,),
    ),
    CodecCase(
        "mnli",
        MnliCodec,
        lambda: MnliCodec(nli_specification()),
        fixture_text("nli/fixtures/mnli.jsonl"),
    ),
    CodecCase(
        "pmb-ccg", PmbCcgCodec, PmbCcgCodec, fixture_text("pmb/fixtures/all_rules.ccg")
    ),
    CodecCase(
        "pmb-drs", PmbDrsCodec, PmbDrsCodec, fixture_text("pmb/fixtures/basic.clf")
    ),
    CodecCase(
        "snli",
        SnliCodec,
        lambda: SnliCodec(nli_specification()),
        fixture_text("nli/fixtures/snli.jsonl"),
    ),
    CodecCase(
        "unli",
        UnliCodec,
        lambda: UnliCodec(nli_specification()),
        fixture_text("nli/fixtures/unli.tsv"),
    ),
)


class ForeignCodec:
    """A codec a distribution other than LoFI or lairs could ship."""

    __slots__ = ("name",)

    name: str

    def __init__(self) -> None:
        self.name = FOREIGN_TAG

    def decode(
        self,
        src: str | bytes,
        *,
        into: CorpusFragment | None = None,
    ) -> CorpusFragment:
        """Return a fragment holding one record per nonempty line."""
        text = src.decode("utf-8") if isinstance(src, bytes) else src
        records = tuple(
            FragmentRecord(
                local_id=f"line-{ordinal}",
                nsid="pub.layers.expression.expression",
                value_json=line,
            )
            for ordinal, line in enumerate(text.splitlines())
            if line
        )
        held = () if into is None else into.records

        return CorpusFragment(records=held + records, source=FOREIGN_TAG)

    def encode(self, records: Iterable[FragmentRecord]) -> str:
        """Return one line per record."""
        return "".join(f"{record.value_json}\n" for record in records)


def lofi_tags() -> tuple[str, ...]:
    """Return the declared tags whose codecs come from this distribution's packages.

    A tag is declared once the module holding it is imported, so the corpora are
    walked first: otherwise the answer is whatever an earlier test happened to
    import, and a codec with no case passes for as long as nothing imports it.
    """
    for found in walk_packages(corpora.__path__, corpora.__name__ + "."):
        import_module(found.name)

    return tuple(
        tag
        for tag in declared_codec_tags()
        if get_codec(tag).__module__.startswith(PACKAGE)
    )


def lofi_entry_points() -> tuple[EntryPoint, ...]:
    """Return this distribution's installed codec entry points, in tag order."""
    return tuple(
        sorted(
            (
                point
                for point in entry_points(group=GROUP)
                if point.dist is not None and point.dist.name == DISTRIBUTION
            ),
            key=lambda point: point.name,
        )
    )


def test_a_construction_case_covers_every_codec_lofi_declares() -> None:
    assert tuple(sorted(case.tag for case in CASES)) == lofi_tags()


@pytest.mark.parametrize("case", CASES, ids=[case.tag for case in CASES])
def test_each_declared_codec_resolves_constructs_and_holds_its_laws(
    case: CodecCase,
) -> None:
    assert get_codec(case.tag) is case.codec

    codec = case.build()

    assert codec.name == case.tag

    sources = st.sampled_from(
        (*case.sources, *(source.encode("utf-8") for source in case.sources))
    )

    check_codec_laws(CodecFixture(codec, sources, records_of, EXAMPLES))


def test_installed_lofi_entry_points_exactly_cover_declared_codecs() -> None:
    points = lofi_entry_points()
    declared = lofi_tags()

    assert tuple(point.name for point in points) == declared

    for point in points:
        assert point.module.startswith(PACKAGE)
        assert point.load() is get_codec(point.name)


def test_every_installed_codec_entry_point_resolves_without_importing_lofi_first() -> (
    None
):
    points = lofi_entry_points()
    declared = lofi_tags()

    assert tuple(point.name for point in points) == declared

    finished = subprocess.run(
        [sys.executable, "-c", RESOLVES_WITHOUT_LOFI, *declared],
        capture_output=True,
        check=True,
        cwd=TREE,
        encoding="utf-8",
    )
    resolved = tuple(
        tuple(line.split()) for line in finished.stdout.splitlines() if line
    )

    assert resolved == tuple(
        (point.name, get_codec(point.name).__module__) for point in points
    )


def test_a_third_party_codec_resolves_without_patching_lofi_or_lairs() -> None:
    register_codec(FOREIGN_TAG, ForeignCodec)

    assert get_codec(FOREIGN_TAG) is ForeignCodec
    assert FOREIGN_TAG not in declared_codec_tags()

    codec = ForeignCodec()
    sources = st.sampled_from(("one\ntwo\n", b"three\n"))

    check_codec_laws(CodecFixture(codec, sources, records_of, EXAMPLES))
