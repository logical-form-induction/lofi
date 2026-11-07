"""A run document that schedules a loss term's weight."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from lofi.run import RandomStreams, SeedPolicy, plan_training, resolve_device_policy
from lofi.run._boundary.manifest import _training_checkpoint_manifest
from lofi.run.operations import CHECKPOINT_RECORDS
from lofi.spec import RunSpecCodec, config_search_path
from lofi.train import DirectoryStore, ResumePosition, fit_training, prepare_training

if TYPE_CHECKING:
    from collections.abc import Mapping

CONFIG = Path(__file__).resolve().parents[2] / "lofi/configs/run/synthetic-parser.yaml"


class _Recorded:
    """Keeps every logged metric at the step it was logged at."""

    def __init__(self) -> None:
        self.logged: list[tuple[int, dict[str, float]]] = []

    def log_scalars(self, mapping: Mapping[str, float], step: int) -> None:
        self.logged.append((step, dict(mapping)))


@pytest.mark.integration
def test_a_scheduled_loss_weight_changes_the_logged_weight(tmp_path: Path) -> None:
    document = tmp_path / "scheduled.yaml"
    document.write_text(
        """defaults:
  - synthetic-parser
loss:
  terms:
    - objective:
        kind: grammaticality
        readout: sentences
    - objective:
        kind: coherence
        readout: types
        space: categories
        warmup: null
      weight: 0.1
      schedule:
        start: 0.0
        steps: 2
""",
        encoding="utf-8",
        newline="\n",
    )
    run_spec = RunSpecCodec(search_path=(CONFIG.parent, *config_search_path(tmp_path)))(
        document,
        overrides=("trainer.epochs=4", "trainer.early_stopping.patience=4"),
    )
    blueprint = plan_training(run_spec)
    streams = RandomStreams(SeedPolicy(17, samplers=blueprint.sampler_streams))
    streams.apply()
    prepared = prepare_training(blueprint, streams, resolve_device_policy(run_spec))
    sink = _Recorded()
    fit_training(
        prepared,
        DirectoryStore(tmp_path / "checkpoints"),
        _training_checkpoint_manifest(
            blueprint, prepared, streams, CHECKPOINT_RECORDS, step=0, data_position=0
        ),
        streams,
        ResumePosition(0, 0),
        sinks=(sink,),
    )
    logged = [
        (step, metrics["loss/coherence/weight"])
        for step, metrics in sink.logged
        if "optimizer/stepped" in metrics
    ]
    validated = [
        metrics["validation/loss/coherence/weight"]
        for _, metrics in sink.logged
        if "validation/loss/coherence/weight" in metrics
    ]

    assert logged == pytest.approx([(1, 0.0), (2, 0.05), (3, 0.1), (4, 0.1)])
    assert validated == pytest.approx([0.05, 0.1, 0.1, 0.1])
    assert not any(
        "loss/grammaticality/weight" in metrics for _, metrics in sink.logged
    )
