"""An experiment sweeping the packaged optimizers by the packaged schedulers."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import pytest
from tests.unit.run.conftest import write_priced_experiment

from lofi.run import (
    RandomStreams,
    SeedPolicy,
    expand_experiment,
    packaged_options,
    plan_training,
    read_experiment,
    resolve_device_policy,
)
from lofi.run._boundary.manifest import _training_checkpoint_manifest
from lofi.run.operations import CHECKPOINT_RECORDS
from lofi.spec import RunSpec
from lofi.train import DirectoryStore, ResumePosition, fit_training, prepare_training

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from lofi._types import JsonValue
    from lofi.train import PreparedTraining


class _Recorded:
    """Keeps every logged metric at the step it was logged at."""

    def __init__(self) -> None:
        self.logged: list[tuple[int, dict[str, float]]] = []

    def log_scalars(self, mapping: Mapping[str, float], step: int) -> None:
        self.logged.append((step, dict(mapping)))


def _built(prepared: PreparedTraining, /) -> tuple[str, str]:
    """Return the optimizer and schedule a prepared run holds, read off their state."""
    _, optimization = prepared.state()
    groups = cast("list[dict[str, JsonValue]]", optimization.optimizer["param_groups"])
    group = groups[0]
    scheduler = optimization.scheduler or {}

    if "momentum" in group:
        optimizer = "sgd"
    else:
        optimizer = "adamw" if group["decoupled_weight_decay"] else "adam"

    if "T_max" in scheduler:
        schedule = "cosine"
    else:
        schedule = "linear" if "end_factor" in scheduler else "constant"

    return optimizer, schedule


@pytest.mark.integration
def test_a_sweep_over_optimizer_by_scheduler_trains_every_point(
    tmp_path: Path,
) -> None:
    optimizers = packaged_options("optimizer")
    schedulers = packaged_options("scheduler")
    sweep: dict[str, JsonValue] = {
        "grid": [
            {"group": "optimizer", "values": list(optimizers)},
            {"group": "scheduler", "values": list(schedulers)},
        ],
        "seeds": [0, 1, 2],
        "expected_points": len(optimizers) * len(schedulers) * 3,
    }
    experiment = write_priced_experiment(
        tmp_path, {"defaults": ["run/synthetic-parser"]}, sweep
    )
    points = expand_experiment(read_experiment(experiment), tmp_path)
    trained: set[tuple[str, str]] = set()

    assert {"adam", "adamw", "sgd"} <= set(optimizers)
    assert {"constant", "cosine", "linear"} <= set(schedulers)

    for point in points:
        run_spec = RunSpec.model_validate(point.config)
        blueprint = plan_training(run_spec)
        streams = RandomStreams(
            SeedPolicy(point.seed, samplers=blueprint.sampler_streams)
        )
        streams.apply()
        prepared = prepare_training(blueprint, streams, resolve_device_policy(run_spec))
        template = _training_checkpoint_manifest(
            blueprint, prepared, streams, CHECKPOINT_RECORDS, step=0, data_position=0
        )
        sink = _Recorded()
        report = fit_training(
            prepared,
            DirectoryStore(tmp_path / "points" / point.point_id),
            template,
            streams,
            ResumePosition(0, 0),
            sinks=(sink,),
        )
        stepped = [
            metrics for _, metrics in sink.logged if "optimizer/stepped" in metrics
        ]

        assert report.optimizer_steps > 0
        assert stepped
        assert all(metrics["optimizer/stepped"] == 1.0 for metrics in stepped)
        assert all(math.isfinite(value) for m in stepped for value in m.values())
        assert run_spec.scheduler is not None
        declared = (str(run_spec.optimizer.kind), str(run_spec.scheduler.kind))

        assert _built(prepared) == declared

        trained.add(declared)

    assert trained == {
        (optimizer, scheduler) for optimizer in optimizers for scheduler in schedulers
    }
