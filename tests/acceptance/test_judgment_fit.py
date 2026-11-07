"""A judgment fit runs from its documents over a declared corpus design.

The fit document names its dataset by tag; the design document under the data
root decodes a corpus release through its registered codec and reads the
factor the fit's response model indexes as a bead expression over the item
records the release lowers to. A fit document declaring a table ablation fits
every declared rung in every held-out fold of its comparison.
"""

import json
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import pytest
import yaml
from tests.unit.run.test_corpus_designs import ACCEPTABILITY_PATH, install, use
from tests.unit.run.test_fits import (
    ENVIRONMENT,
    FREEZE,
    SEED,
    an_experiment,
)

from lofi.judgment import LINEAR, ResponseCompiler, linear_steps
from lofi.run import (
    FIT_MANIFEST_NAME,
    ExperimentStore,
    ManifestStatus,
    RunStore,
    SweepPoint,
)
from lofi.run._manifest import read_rows
from lofi.run.fits import FitExecutor, RegisteredDesigns

if TYPE_CHECKING:
    from lofi._types import JsonValue

DOCUMENTS: Final = Path(__file__).with_name("documents") / "judgment-fit"
"""The fit document every case runs."""


def _fit(name: str = "fit.yaml") -> dict[str, JsonValue]:
    return cast(
        "dict[str, JsonValue]",
        yaml.safe_load((DOCUMENTS / name).read_text(encoding="utf-8")),
    )


def test_a_fit_document_runs_over_its_declared_design(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = install(tmp_path / "data")
    use(root, monkeypatch)
    store = RunStore(tmp_path / "store", freeze_id=FREEZE, seed=SEED)
    executor = FitExecutor(
        ExperimentStore(store.root, an_experiment(tmp_path)),
        designs=RegisteredDesigns(),
        compiler=ResponseCompiler({LINEAR: linear_steps}),
        environment=ENVIRONMENT,
        infrastructure_attempts=3,
    )

    outcome = executor(SweepPoint(_fit(), seed=0))
    manifest = json.loads(
        next(store.root.glob(f"**/{FIT_MANIFEST_NAME}")).read_text(encoding="utf-8")
    )
    document = root / "designs" / "megaacceptability.json"

    assert outcome.status is ManifestStatus.COMPLETED
    assert manifest["design"]["tag"] == "megaacceptability"
    assert manifest["design"]["corpus_digests"]["designs/megaacceptability.json"] == (
        sha256(document.read_bytes()).hexdigest()
    )


def test_a_factor_the_design_does_not_read_is_refused_before_fitting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = install(tmp_path / "data")
    use(root, monkeypatch)
    path = root / "designs" / "megaacceptability.json"
    design = json.loads(path.read_text(encoding="utf-8"))
    design["factors"] = {"frame": "item['frame']"}
    path.write_text(json.dumps(design), encoding="utf-8")
    store = RunStore(tmp_path / "store", freeze_id=FREEZE, seed=SEED)
    executor = FitExecutor(
        ExperimentStore(store.root, an_experiment(tmp_path)),
        designs=RegisteredDesigns(),
        compiler=ResponseCompiler({LINEAR: linear_steps}),
        environment=ENVIRONMENT,
        infrastructure_attempts=3,
    )

    with pytest.raises(ValueError, match="no grouping factor 'verb'"):
        executor(SweepPoint(_fit(), seed=0))


def test_a_table_ablation_document_fits_every_rung_in_every_held_out_fold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = install(tmp_path / "data")
    (root / ACCEPTABILITY_PATH).write_bytes(
        (DOCUMENTS / "megaacceptability.tsv").read_bytes()
    )
    use(root, monkeypatch)
    store = RunStore(tmp_path / "store", freeze_id=FREEZE, seed=SEED)
    executor = FitExecutor(
        ExperimentStore(store.root, an_experiment(tmp_path)),
        designs=RegisteredDesigns(),
        compiler=ResponseCompiler({LINEAR: linear_steps}),
        environment=ENVIRONMENT,
        infrastructure_attempts=3,
    )
    point = SweepPoint(_fit("table-ablation.yaml"), seed=0)

    outcome = executor(point)
    recorded = read_rows(store.directory(point))

    assert outcome.status is ManifestStatus.COMPLETED
    assert recorded is not None
    ablation = [
        row for row in recorded.rows if row["metric"] == "table-ablation-log-density"
    ]
    assert {(row["fold"], row["rung"]) for row in ablation} == {
        (fold, rung)
        for fold in range(3)
        for rung in ("memorized", "joint_type", "row_classes")
    }
    assert len(ablation) == 9
    assert all(
        isinstance(row["value"], float) and row["value"] <= 0.0 for row in ablation
    )
