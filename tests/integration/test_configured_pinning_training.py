"""Production composition and gradients for configured span pinning."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import lofi.train as train_module
from lofi.run import RESOLVED_NAME, read_metrics
from lofi.run._entrypoint import main
from lofi.train import ParserFittingPlan

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi.nn.policy import DevicePolicy
    from lofi.run import TrainingBlueprint
    from lofi.train import (
        CheckpointManifest,
        CheckpointStore,
        FittingReport,
        MetricLog,
        PreparedTraining,
        RestorableRandomStreams,
        ResumePosition,
        StateTree,
    )

CONFIG = (
    Path(__file__).resolve().parents[2]
    / "lofi/configs/run/synthetic-pinning-lexical.yaml"
)


@pytest.mark.integration
def test_a_fresh_process_discovers_the_pinning_source_and_loss_plugins() -> None:
    root = Path(__file__).resolve().parents[2]
    script = """
from pathlib import Path
from lofi.spec import RunSpecCodec, config_search_path

root = Path.cwd()
run = RunSpecCodec(search_path=config_search_path(root))(
    root / "lofi/configs/run/synthetic-pinning-constituent.yaml"
)
assert run.data.datasets["train"].constraint_source is not None
assert (
    run.data.datasets["train"].constraint_source.model_dump()["kind"]
    == "gold-derivation"
)
assert run.loss is not None
assert run.loss.terms[-1].objective.model_dump()["kind"] == "pinning"
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        check=False,
        capture_output=True,
        encoding="utf-8",
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.integration
@pytest.mark.parametrize(
    ("config", "mode", "penalty"),
    [
        (CONFIG, "hard", None),
        (CONFIG.with_name("synthetic-pinning-soft.yaml"), "soft", 0.25),
    ],
)
def test_production_pinning_modes_train_and_reach_parser_and_readout_gradients(
    config: Path,
    mode: str,
    penalty: float | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared: list[PreparedTraining] = []
    reports: list[FittingReport] = []
    original_prepare = train_module.prepare_training
    original_fit = train_module.fit_training

    def recording_prepare(
        blueprint: TrainingBlueprint,
        streams: RestorableRandomStreams,
        policy: DevicePolicy,
        /,
    ) -> PreparedTraining:
        training = original_prepare(blueprint, streams, policy)
        prepared.append(training)

        return training

    def recording_fit(
        training: PreparedTraining,
        store: CheckpointStore[StateTree],
        manifest: CheckpointManifest,
        streams: RestorableRandomStreams,
        position: ResumePosition,
        /,
        *,
        sinks: Sequence[MetricLog],
    ) -> FittingReport:
        report = original_fit(training, store, manifest, streams, position, sinks=sinks)
        reports.append(report)

        return report

    monkeypatch.setattr(train_module, "prepare_training", recording_prepare)
    monkeypatch.setattr(train_module, "fit_training", recording_fit)

    assert (
        main(
            [
                "run",
                "--config",
                str(config),
                "--root",
                str(tmp_path),
                "--seed",
                "37",
                "--experiment-seed",
                "5",
            ]
        )
        == 0
    )
    (directory,) = (tmp_path / "runs").glob("[0-9a-f]*")
    resolved = json.loads((directory / RESOLVED_NAME).read_text(encoding="utf-8"))
    (training,) = prepared
    (report,) = reports
    plan = training.plan

    assert isinstance(plan, ParserFittingPlan)
    assert report.constraint_selection == plan.constraint_selection
    assert report.constraint_selection is not None
    assert report.constraint_selection.mass_share == 1.0
    metrics = read_metrics(directory)
    assert metrics is not None
    assert metrics.selection is not None
    assert metrics.selection.clamp.mode == mode
    assert metrics.selection.clamp.penalty == penalty
    assert metrics.selection.clamp.gradient_scale == (
        1.0 if penalty is None else penalty
    )
    assert metrics.selection.clamp.matching == plan.grammar.grammar.matching.name
    assert metrics.selection.clamp.dropped_clamps == 0
    assert metrics.selection.clamp.skipped_items == 0
    assert metrics.fitting is not None
    assert all(
        step.metrics["loss/pinning/gradient_scale"]
        == (1.0 if penalty is None else penalty)
        for epoch in metrics.fitting.epochs
        for step in epoch.steps
    )
    assert resolved["data"]["datasets"]["train"]["constraint_source"] == {
        "clamp": {
            "matching": None,
            "meaning_matching": {"kind": "identity"},
            "mode": mode,
            "negate": False,
            "on_unsatisfiable": {"kind": "error"},
            "penalty": penalty,
            "require_constituent": True,
            "scope": "constituent",
        },
        "declared_output": None,
        "kind": "gold-lexicon",
        "policy": "syntactic",
    }
    assert (
        resolved["data"]["datasets"]["train"]["constraint_policy"]["syntactic"][
            "coverage"
        ]
        == 1.0
    )
    pinning = resolved["loss"]["terms"][-1]
    assert pinning["objective"]["kind"] == "pinning"
    assert pinning["weight"] == 1.0

    parser_gradients = tuple(
        parameter.grad
        for parameter in training.model.modules["parser"].parameters()
        if parameter.grad is not None
    )
    readout_gradients = tuple(
        parameter.grad
        for parameter in training.model.modules["identity"].parameters()
        if parameter.grad is not None
    )
    assert parser_gradients
    assert readout_gradients
    assert any(bool(gradient.abs().sum() > 0.0) for gradient in parser_gradients)
    assert any(bool(gradient.abs().sum() > 0.0) for gradient in readout_gradients)
