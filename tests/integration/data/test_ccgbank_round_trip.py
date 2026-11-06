from typing import TYPE_CHECKING

import pytest

from lofi.data.corpora.ccgbank import (
    RoundTripCounts,
    ccgbank_round_trip_projector,
    parse_auto,
    parse_parg,
    round_trip,
)

if TYPE_CHECKING:
    from pathlib import Path


def _section_zero(root: Path) -> tuple[tuple[Path, Path], ...]:
    autos = {
        path.stem: path for path in root.rglob("*.auto") if path.parent.name == "00"
    }
    pargs = {
        path.stem: path for path in root.rglob("*.parg") if path.parent.name == "00"
    }
    assert autos.keys() == pargs.keys(), (
        f"Section 00 files without a partner: AUTO only "
        f"{sorted(autos.keys() - pargs.keys())}, PARG only "
        f"{sorted(pargs.keys() - autos.keys())}"
    )
    return tuple((autos[stem], pargs[stem]) for stem in sorted(autos))


@pytest.mark.corpus
@pytest.mark.integration
def test_gold_section_zero_round_trips_through_the_extractor(
    ccgbank_root: Path,
) -> None:
    files = _section_zero(ccgbank_root)
    assert files, "LOFI_CCGBANK contains no Section 00 .auto and .parg files"

    projector = ccgbank_round_trip_projector()
    total = RoundTripCounts()

    for auto, parg in files:
        derivations = parse_auto(auto.read_bytes())
        sentences = parse_parg(parg.read_bytes())

        for derivation, sentence in zip(derivations, sentences, strict=True):
            total += round_trip(derivation, sentence, projector)

    report = (
        f"LP {total.labeled_precision:.4f} LR {total.labeled_recall:.4f} "
        f"({total.matched} matched, {total.predicted} predicted, {total.gold} gold, "
        f"{total.unprojected} unprojected sentences); "
        f"spurious by rule {total.spurious}; missed by rule {total.missed}; "
        f"lost to raising steps {total.raised}"
    )

    assert total.labeled_precision >= 0.99, report
    assert total.labeled_recall >= 0.99, report
