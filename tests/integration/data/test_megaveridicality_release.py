import csv
import json
import os
import shutil
from pathlib import Path

import pytest

from lofi.judgment import ReadDesign
from lofi.run.corpus_designs import DATA_ROOT_VARIABLE
from lofi.run.fits import RegisteredDesigns

RESPONSES = "mega-veridicality-v2.tsv"


MEGAVERIDICALITY_TAG = "megaveridicality"


@pytest.fixture(scope="module")
def release() -> Path:
    named = os.environ.get("LOFI_MEGAVERIDICALITY")

    if named is None:
        pytest.skip("set LOFI_MEGAVERIDICALITY to a MegaVeridicality v2.1 directory")

    responses = Path(named).expanduser() / RESPONSES

    if not responses.is_file():
        pytest.skip(f"LOFI_MEGAVERIDICALITY has no {RESPONSES}")

    return responses


@pytest.mark.corpus
@pytest.mark.integration
def test_the_released_table_reads_as_a_participant_indexed_design(
    release: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "releases" / "mega-veridicality-v2.1"
    directory.mkdir(parents=True)
    shutil.copyfile(release, directory / RESPONSES)
    (directory / "release.json").write_text(
        json.dumps(
            {
                "source_id": "megaveridicality",
                "release_id": "v2.1",
                "language": "en",
                "source_method": "elicited",
                "spdx": "CC-BY-SA-4.0",
            }
        ),
        encoding="utf-8",
        newline="\n",
    )
    (tmp_path / "designs").mkdir()
    (tmp_path / "designs" / f"{MEGAVERIDICALITY_TAG}.json").write_text(
        json.dumps(
            {
                "source": "fragment",
                "fragment": {
                    "kind": "codec",
                    "codec": MEGAVERIDICALITY_TAG,
                    "declaration": "releases/mega-veridicality-v2.1/release.json",
                    "responses": f"releases/mega-veridicality-v2.1/{RESPONSES}",
                },
                "responses": {
                    "kind": "relation",
                    "inventory": ["contradicts", "neutral", "entails"],
                },
                "factors": {
                    factor: f"item['{factor}']"
                    for factor in ("verb", "frame", "polarity", "conditional")
                },
                "readouts": None,
                "keep": None,
            }
        ),
        encoding="utf-8",
        newline="\n",
    )
    monkeypatch.setenv(DATA_ROOT_VARIABLE, str(tmp_path))

    with release.open(encoding="utf-8", newline="") as handle:
        responded = [
            row for row in csv.DictReader(handle, delimiter="\t") if row["veridicality"]
        ]

    read = RegisteredDesigns().read(MEGAVERIDICALITY_TAG)

    assert isinstance(read, ReadDesign)
    assert read.design.n_rows == len(responded)
    assert read.design.response_levels == 3
    assert sorted(read.design.participant_levels) == sorted(
        {row["participant"] for row in responded}
    )
    observed = sorted(
        read.design.participant_levels[index]
        for index in read.design.participants.tolist()
    )
    assert observed == sorted(row["participant"] for row in responded)
