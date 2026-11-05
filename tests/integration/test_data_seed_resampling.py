"""Data seeds through the production run command."""

import json
from pathlib import Path
from typing import Final

import pytest

from lofi.run import MANIFEST_NAME
from lofi.run._entrypoint import main

CONFIG: Final = (
    Path(__file__).resolve().parents[2] / "lofi/configs/run/synthetic-parser.yaml"
)


def manifest(root: Path, seed: str, data_seed: str) -> dict[str, object]:
    """Return the manifest one production run writes.

    Parameters
    ----------
    root : Path
        The directory the run directory is created under.
    seed : str
        The run seed.
    data_seed : str
        The data seed the run is declared with.
    """
    assert (
        main(
            [
                "run",
                "--config",
                str(CONFIG),
                "--set",
                f"data.datasets.train.dataset.seed={data_seed}",
                "--root",
                str(root),
                "--seed",
                seed,
                "--experiment-seed",
                "5",
            ]
        )
        == 0
    )
    (directory,) = (root / "runs").glob("[0-9a-f]*")

    return json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))


@pytest.mark.integration
def test_run_seeds_resample_a_derived_corpus_and_replicate_a_fixed_one(
    tmp_path: Path,
) -> None:
    revisions: dict[str, set[str]] = {}

    for data_seed in ("derived", "5"):
        for seed in ("37", "38"):
            recorded = manifest(tmp_path / f"{data_seed}-{seed}", seed, data_seed)
            run_spec = recorded["run_spec"]

            assert isinstance(run_spec, dict)
            assert (
                str(run_spec["data"]["datasets"]["train"]["dataset"]["seed"])
                == data_seed
            )

            revisions.setdefault(data_seed, set()).add(str(recorded["corpus_revision"]))

    assert len(revisions["derived"]) == 2
    assert len(revisions["5"]) == 1
