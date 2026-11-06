"""Tests for public comparison-point metadata assembly."""

import json
from pathlib import Path

from tests.unit.run.conftest import EXPERIMENT_SEED, a_running_manifest

from lofi.eval import config_digest
from lofi.run import ManifestIso, SweepPoint, comparison_points
from lofi.run._run_configuration import RunConfiguration
from lofi.spec import RunSpecCodec

PACKAGED_RUNS = Path(__file__).resolve().parents[3] / "lofi/configs/run"
PACKAGED_CODEC = RunSpecCodec(search_path=(PACKAGED_RUNS, PACKAGED_RUNS.parent))


def test_comparison_points_retain_sweep_and_derived_result_seeds(
    tmp_path: Path,
) -> None:
    run = PACKAGED_CODEC(PACKAGED_RUNS / "synthetic-parser.yaml")
    point = SweepPoint(json.loads(run.model_dump_json()), seed=5)
    manifest = RunConfiguration("run", tmp_path / "unused.yaml").manifest_point(
        ManifestIso().forward(a_running_manifest(point, seed=EXPERIMENT_SEED))
    )
    points = comparison_points(
        (manifest,),
        ({"point_id": point.point_id, "point/parameter_count": 17},),
    )

    [metadata] = points[config_digest(point.config)]
    assert metadata.seed == manifest.seed.seed
    assert metadata.result_seed == manifest.seed.root
    assert metadata.result_seed != metadata.seed
    assert metadata.parameter_count == 17
