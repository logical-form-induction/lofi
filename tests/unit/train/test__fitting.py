"""Tests for configured fitting, validation and checkpoint selection."""

import json
from math import fsum
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
import torch

from lofi._checks import NumericalFailureError
from lofi.run import RandomStreams, SeedPolicy, plan_training, resolve_device_policy
from lofi.spec import RunSpecCodec
from lofi.train import (
    CheckpointManifest,
    ComponentRecord,
    DirectoryStore,
    ParserFittingPlan,
    PrecisionRecord,
    ResumePosition,
    RngRecord,
    StateReference,
    StepOutcome,
    TrainResult,
    fit_training,
    fitting_remains,
    prepare_training,
    resume_from_checkpoint,
)
from lofi.train._fitting import _batch_order
from lofi.train._parsing import Featurized, ParserStep, Trained
from lofi.training_plugins import TrainingUnit

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi.data.synthetic.training import SyntheticTrainingBatch
    from lofi.train import PreparedTraining
    from lofi.train._store import StateTree

CONFIG = Path(__file__).resolve().parents[3] / "lofi/configs/run/synthetic-parser.yaml"
DISTANCE_CONFIG = (
    Path(__file__).resolve().parents[3] / "lofi/configs/run/synthetic-distance.yaml"
)
PACKAGED_CODEC = RunSpecCodec(search_path=(CONFIG.parent, CONFIG.parent.parent))


class Recorder:
    """Records every logged metric mapping beside its step."""

    def __init__(self) -> None:
        self.logged: list[tuple[dict[str, float], int]] = []

    def log_scalars(self, mapping: Mapping[str, float], step: int) -> None:
        self.logged.append((dict(mapping), step))


def _assert_state_equal(expected: StateTree, found: StateTree, /) -> None:
    """Assert exact equality over nested checkpoint state."""
    torch.testing.assert_close(found, expected, rtol=0.0, atol=0.0)


def _manifest(
    prepared: PreparedTraining,
    streams: RandomStreams,
    /,
) -> CheckpointManifest:
    """Return a structurally complete template for one configured run."""
    _, optimization = prepared.state()
    components = {
        role: ComponentRecord(
            tag=role,
            spec_fingerprint=f"spec-{role}",
            signature_fingerprint="signature",
            codec_identity="codec",
            state=StateReference(key=role, digest="initial"),
        )
        for role in prepared.model.modules
    }
    policy = prepared.model.policy

    return CheckpointManifest(
        run_id="configured-fitting",
        step=0,
        data_position=0,
        code_version="test",
        action_set="actions-a1",
        components=components,
        rng=RngRecord(streams=dict(streams.snapshot())),
        precision=PrecisionRecord(
            compute_dtype=str(policy.autocast_dtype or policy.dtype).removeprefix(
                "torch."
            ),
            autocast_device_type=cast("str", policy.autocast_device_type),
            gradient_scaling=optimization.scaler is not None,
        ),
        optimizer=StateReference(key="adam-state", digest="initial"),
        scaler=(
            None
            if optimization.scaler is None
            else StateReference(key="amp-state", digest="initial")
        ),
        scheduler=(
            None
            if optimization.scheduler is None
            else StateReference(key="schedule-state", digest="initial")
        ),
    )


def test_configured_epochs_validate_and_write_distinct_selections(
    tmp_path: Path,
) -> None:
    run_spec = PACKAGED_CODEC(CONFIG)
    blueprint = plan_training(run_spec)
    streams = RandomStreams(SeedPolicy(11, samplers=blueprint.sampler_streams))
    streams.apply()
    prepared = prepare_training(blueprint, streams, resolve_device_policy(run_spec))
    before = tuple(
        parameter.detach().clone() for parameter in prepared.model.parameters
    )
    store = DirectoryStore(tmp_path / "checkpoints")
    report = fit_training(
        prepared,
        store,
        _manifest(prepared, streams),
        streams,
        ResumePosition(0, 0),
        sinks=(),
    )
    latest = store.load(
        report.latest_checkpoint,
        _manifest(prepared, streams).with_(
            step=report.optimizer_steps,
            data_position=report.data_position,
        ),
    )
    plan = cast("ParserFittingPlan", prepared.plan)

    assert len(report.epochs) == plan.epochs == 2
    assert report.optimizer_steps == plan.steps_per_epoch * plan.epochs == 2
    assert report.data_position == plan.items_per_epoch * plan.epochs == 6
    assert report.best_checkpoint != report.latest_checkpoint
    assert report.latest_checkpoint == store.list()[-1]
    assert latest.manifest.fitting is not None
    assert latest.manifest.fitting.kind == "resume-latest"
    assert latest.manifest.fitting.best_checkpoint == report.best_checkpoint
    assert latest.manifest.data_position == 6
    gradients = tuple(
        parameter.grad is not None for parameter in prepared.model.parameters
    )
    changed = tuple(
        not torch.equal(initial, final)
        for initial, final in zip(before, prepared.model.parameters, strict=True)
    )

    nonzero = tuple(
        parameter.grad is not None and bool(parameter.grad.ne(0).any())
        for parameter in prepared.model.parameters
    )

    assert any(nonzero)
    assert any(not reached for reached in gradients)
    assert all(
        reached for moved, reached in zip(changed, gradients, strict=True) if moved
    )
    assert all(moved for moved, pushed in zip(changed, nonzero, strict=True) if pushed)
    for epoch in report.epochs:
        assert epoch.validation is not None
        assert epoch.validation.items == 1
    assert all(
        result.metrics["optimizer/loss_finite"] == 1.0
        for result in report.epochs[0].results
    )


def test_fitting_refuses_a_manifest_with_incomplete_component_coverage(
    tmp_path: Path,
) -> None:
    run_spec = PACKAGED_CODEC(CONFIG)
    blueprint = plan_training(run_spec)
    streams = RandomStreams(SeedPolicy(13, samplers=blueprint.sampler_streams))
    streams.apply()
    prepared = prepare_training(blueprint, streams, resolve_device_policy(run_spec))
    template = _manifest(prepared, streams)
    components = dict(template.components)
    del components[next(iter(components))]

    with pytest.raises(ValueError, match="component roles"):
        fit_training(
            prepared,
            DirectoryStore(tmp_path / "checkpoints"),
            template.with_(components=components),
            streams,
            ResumePosition(0, 0),
            sinks=(),
        )


def test_distance_fitting_checkpoints_resume_cadence_and_resumes_bitwise(
    tmp_path: Path,
) -> None:
    run_spec = RunSpecCodec()(DISTANCE_CONFIG)
    blueprint = plan_training(run_spec)
    policy = resolve_device_policy(run_spec)

    def fresh() -> tuple[PreparedTraining, RandomStreams]:
        streams = RandomStreams(SeedPolicy(19, samplers=blueprint.sampler_streams))
        streams.apply()

        return prepare_training(blueprint, streams, policy), streams

    uninterrupted, streams = fresh()
    store = DirectoryStore(tmp_path / "distance-checkpoints")
    recorder = Recorder()
    report = fit_training(
        uninterrupted,
        store,
        _manifest(uninterrupted, streams),
        streams,
        ResumePosition(0, 0),
        sinks=(recorder,),
    )
    handles = store.list()
    template = _manifest(uninterrupted, streams)
    saved = tuple(store.load(handle, template) for handle in handles)
    halfway = next(checkpoint for checkpoint in saved if checkpoint.manifest.step == 2)

    assert report.unit is TrainingUnit.STEPS
    assert recorder.logged == [
        (dict(result.metrics), result.step) for result in report.results
    ]
    assert [step for _, step in recorder.logged] == [1, 2, 3, 4]
    assert report.best_checkpoint is None
    assert report.optimizer_steps == 4
    assert report.data_position == 8
    assert tuple(checkpoint.manifest.step for checkpoint in saved) == (2, 4)
    assert all(
        checkpoint.manifest.fitting is not None
        and checkpoint.manifest.fitting.unit == "steps"
        and checkpoint.manifest.fitting.kind == "resume-latest"
        for checkpoint in saved
    )

    resumed, resumed_streams = fresh()
    resumed_template = _manifest(resumed, resumed_streams)
    position = resume_from_checkpoint(
        halfway,
        resumed_template,
        resumed.model.prepare_restore,
        resumed.optimization.prepare_restore,
        resumed_streams,
    )
    continued = fit_training(
        resumed,
        store,
        halfway.manifest,
        resumed_streams,
        position,
        sinks=(),
    )

    assert position == ResumePosition(2, 4)
    assert continued.optimizer_steps == 4
    expected_components, expected_optimization = uninterrupted.state()
    found_components, found_optimization = resumed.state()
    _assert_state_equal(
        cast("StateTree", dict(expected_components)),
        cast("StateTree", dict(found_components)),
    )
    _assert_state_equal(
        cast("StateTree", expected_optimization.optimizer),
        cast("StateTree", found_optimization.optimizer),
    )
    _assert_state_equal(
        cast("StateTree", expected_optimization.scaler),
        cast("StateTree", found_optimization.scaler),
    )
    _assert_state_equal(
        cast("StateTree", expected_optimization.scheduler),
        cast("StateTree", found_optimization.scheduler),
    )


def test_distance_fitting_keeps_only_the_declared_number_of_checkpoints(
    tmp_path: Path,
) -> None:
    run_spec = RunSpecCodec()(
        DISTANCE_CONFIG,
        overrides=("trainer.checkpoints.interval=1", "trainer.checkpoints.keep=1"),
    )
    blueprint = plan_training(run_spec)
    streams = RandomStreams(SeedPolicy(19, samplers=blueprint.sampler_streams))
    streams.apply()
    prepared = prepare_training(blueprint, streams, resolve_device_policy(run_spec))
    store = DirectoryStore(tmp_path / "distance-checkpoints")
    report = fit_training(
        prepared,
        store,
        _manifest(prepared, streams),
        streams,
        ResumePosition(0, 0),
        sinks=(),
    )

    assert store.list() == (report.latest_checkpoint,)
    assert store.manifest(report.latest_checkpoint).step == report.optimizer_steps


def test_fitting_remains_until_the_epoch_horizon_or_patience_is_reached(
    tmp_path: Path,
) -> None:
    run_spec = PACKAGED_CODEC(CONFIG)
    blueprint = plan_training(run_spec)
    streams = RandomStreams(SeedPolicy(11, samplers=blueprint.sampler_streams))
    streams.apply()
    prepared = prepare_training(blueprint, streams, resolve_device_policy(run_spec))
    template = _manifest(prepared, streams)
    store = DirectoryStore(tmp_path / "checkpoints")
    report = fit_training(
        prepared, store, template, streams, ResumePosition(0, 0), sinks=()
    )
    resumable = [
        manifest
        for manifest in (
            store.load(handle, template).manifest for handle in store.list()
        )
        if manifest.fitting is not None and manifest.fitting.kind == "resume-latest"
    ]
    first, final = resumable
    plan = cast("ParserFittingPlan", prepared.plan)
    assert first.fitting is not None
    exhausted = first.with_(
        fitting=first.fitting.with_(failures=plan.early_stopping.patience)
    )

    def at(manifest: CheckpointManifest) -> ResumePosition:
        return ResumePosition(manifest.step, manifest.data_position)

    assert report.optimizer_steps == plan.steps_per_epoch * plan.epochs
    assert fitting_remains(prepared, template, streams, ResumePosition(0, 0))
    assert fitting_remains(prepared, first, streams, at(first))
    assert not fitting_remains(prepared, final, streams, at(final))
    assert not fitting_remains(prepared, exhausted, streams, at(exhausted))


def test_fitting_remains_validates_the_checkpoint_as_fitting_does(
    tmp_path: Path,
) -> None:
    run_spec = PACKAGED_CODEC(CONFIG)
    blueprint = plan_training(run_spec)
    streams = RandomStreams(SeedPolicy(11, samplers=blueprint.sampler_streams))
    streams.apply()
    prepared = prepare_training(blueprint, streams, resolve_device_policy(run_spec))
    template = _manifest(prepared, streams)
    store = DirectoryStore(tmp_path / "checkpoints")
    fit_training(prepared, store, template, streams, ResumePosition(0, 0), sinks=())
    final = store.load(store.list()[-1], template).manifest
    assert final.fitting is not None
    position = ResumePosition(final.step, final.data_position)
    remonitored = final.with_(fitting=final.fitting.with_(metric="loss/other"))

    with pytest.raises(ValueError, match="monitored"):
        fitting_remains(prepared, remonitored, streams, position)

    with pytest.raises(ValueError, match="differs from"):
        fitting_remains(prepared, final, streams, ResumePosition(0, 0))


def test_distance_fitting_remains_until_its_step_horizon(tmp_path: Path) -> None:
    run_spec = RunSpecCodec()(DISTANCE_CONFIG)
    blueprint = plan_training(run_spec)
    streams = RandomStreams(SeedPolicy(19, samplers=blueprint.sampler_streams))
    streams.apply()
    prepared = prepare_training(blueprint, streams, resolve_device_policy(run_spec))
    template = _manifest(prepared, streams)
    store = DirectoryStore(tmp_path / "distance-checkpoints")
    fit_training(prepared, store, template, streams, ResumePosition(0, 0), sinks=())
    halfway, final = (store.load(handle, template).manifest for handle in store.list())

    assert fitting_remains(prepared, template, streams, ResumePosition(0, 0))
    assert fitting_remains(
        prepared, halfway, streams, ResumePosition(halfway.step, halfway.data_position)
    )
    assert not fitting_remains(
        prepared, final, streams, ResumePosition(final.step, final.data_position)
    )

    with pytest.raises(ValueError, match="differs from"):
        fitting_remains(prepared, final, streams, ResumePosition(0, 0))


def test_parser_fitting_logs_each_optimizer_step_and_each_validation(
    tmp_path: Path,
) -> None:
    declared_sampler = json.dumps(
        {
            "kind": "length-bucket",
            "bucket_factor": 100,
            "reshuffle_each_epoch": True,
        }
    )
    run_spec = PACKAGED_CODEC(
        CONFIG,
        overrides=(f"trainer.batch_sampler={declared_sampler}",),
    )
    blueprint = plan_training(run_spec)
    streams = RandomStreams(SeedPolicy(11, samplers=blueprint.sampler_streams))
    streams.apply()
    prepared = prepare_training(blueprint, streams, resolve_device_policy(run_spec))
    first, second = Recorder(), Recorder()

    report = fit_training(
        prepared,
        DirectoryStore(tmp_path / "checkpoints"),
        _manifest(prepared, streams),
        streams,
        ResumePosition(0, 0),
        sinks=(first, second),
    )
    plan = cast("ParserFittingPlan", prepared.plan)

    expected: list[tuple[dict[str, float], int]] = []

    for epoch_index, epoch in enumerate(report.epochs):
        window = epoch.results
        stepped = window[-1]
        validation = epoch.validation
        assert validation is not None
        names = {name for result in window for name in result.metrics}
        expected.append(
            (
                {
                    name: (
                        stepped.metrics[name]
                        if name.startswith("optimizer/")
                        else fsum(result.metrics[name] for result in window)
                        / len(window)
                    )
                    for name in names
                }
                | {"data/batch-order": _batch_order(plan.epoch(epoch_index).training)},
                stepped.step,
            )
        )
        expected.append(
            (
                {
                    f"validation/{name}": value
                    for name, value in validation.metrics.items()
                },
                stepped.step,
            )
        )

    assert [len(epoch.results) for epoch in report.epochs] == [2, 2]
    assert first.logged == second.logged == expected
    assert [step for _, step in first.logged] == [1, 1, 2, 2]
    assert first.logged[0][0]["optimizer/stepped"] == 1.0
    assert (
        first.logged[0][0]["data/batch-order"] != first.logged[2][0]["data/batch-order"]
    )
    assert "loss/grammaticality/range" in first.logged[0][0]
    assert "validation/loss/grammaticality" in first.logged[1][0]


def test_a_step_whose_loss_was_not_finite_is_a_numerical_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_spec = PACKAGED_CODEC(CONFIG)
    blueprint = plan_training(run_spec)
    streams = RandomStreams(SeedPolicy(11, samplers=blueprint.sampler_streams))
    streams.apply()
    prepared = prepare_training(blueprint, streams, resolve_device_policy(run_spec))
    run = ParserStep.run

    def skipped(
        self: ParserStep[SyntheticTrainingBatch[str]],
        inputs: Featurized[SyntheticTrainingBatch[str]],
        /,
    ) -> StepOutcome[Trained, TrainResult]:
        outcome = run(self, inputs)
        result = outcome.result
        unfinished = TrainResult(
            result.step,
            result.items,
            metrics={**result.metrics, "optimizer/loss_finite": 0.0},
        )

        return StepOutcome(outcome.outputs, unfinished)

    monkeypatch.setattr(ParserStep, "run", skipped)

    with pytest.raises(NumericalFailureError, match="the parser loss in epoch 0"):
        fit_training(
            prepared,
            DirectoryStore(tmp_path / "checkpoints"),
            _manifest(prepared, streams),
            streams,
            ResumePosition(0, 0),
            sinks=(),
        )
