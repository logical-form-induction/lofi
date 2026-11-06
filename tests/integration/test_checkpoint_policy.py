"""Declared checkpoint cadence, retention, pinned weights and the action-set digest.

Each run is composed from the packaged parser document. The pretrained token
encoder reads its weights through a stand-in loader, a small seeded stack over
the run's own vocabulary, so no checkpoint is downloaded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest
import torch
from tests.pretrained_stand_ins import StandInLoader

from lofi.run import RandomStreams, SeedPolicy, plan_training, resolve_device_policy
from lofi.run._boundary.manifest import _training_checkpoint_manifest
from lofi.run.operations import CHECKPOINT_RECORDS
from lofi.spec import RunSpecCodec, config_search_path
from lofi.train import (
    DirectoryStore,
    FingerprintMismatchError,
    ResumePosition,
    fit_training,
    prepare_training,
    resume_from_checkpoint,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from lofi.nn.tokens.pretrained import PretrainedLoader
    from lofi.run import TrainingBlueprint
    from lofi.spec import ComponentSpec, RunSpec
    from lofi.train import (
        Checkpoint,
        CheckpointId,
        CheckpointManifest,
        PreparedTraining,
        StateTree,
    )

WORDS = tuple(f"word-{index}" for index in range(8))


def _composed(directory: Path, *overrides: str, pretrained: bool) -> RunSpec:
    """Compose the packaged parser run, over a pretrained token encoder if asked."""
    document = directory / "composed.yaml"
    selected = (
        "  - model.token_encoder: pretrained\n"
        f"artifacts:\n  root: {directory / 'artifacts'}\n"
        if pretrained
        else ""
    )
    document.write_text(
        f"defaults:\n  - run/synthetic-parser\n{selected}",
        encoding="utf-8",
        newline="\n",
    )

    return RunSpecCodec(search_path=config_search_path(directory))(
        document, overrides=overrides
    )


class _Run:
    """One seeded preparation of a composed run and its fresh manifest."""

    def __init__(self, run_spec: RunSpec) -> None:
        self.blueprint: TrainingBlueprint = plan_training(run_spec)
        self.streams = RandomStreams(
            SeedPolicy(17, samplers=self.blueprint.sampler_streams)
        )
        self.streams.apply()
        token_encoder = cast("ComponentSpec", run_spec.model.token_encoder)
        resources: dict[str, PretrainedLoader] = (
            {}
            if token_encoder.pretrained_weights() is None
            else {"token_encoder": StandInLoader(WORDS, width=6, seed=3)}
        )
        self.prepared: PreparedTraining = prepare_training(
            self.blueprint,
            self.streams,
            resolve_device_policy(run_spec),
            pretrained_resources=resources,
        )
        self.template: CheckpointManifest = _training_checkpoint_manifest(
            self.blueprint,
            self.prepared,
            self.streams,
            CHECKPOINT_RECORDS,
            step=0,
            data_position=0,
        )


class _RecordingStore:
    """A directory store that records every save and may stop the run on one."""

    def __init__(self, root: Path, *, interrupt_at: int | None = None) -> None:
        self._inner = DirectoryStore(root)
        self._interrupt_at = interrupt_at
        self.saved: list[tuple[CheckpointId, str, int]] = []

    def list(self) -> tuple[CheckpointId, ...]:
        return self._inner.list()

    def manifest(self, checkpoint: CheckpointId) -> CheckpointManifest:
        return self._inner.manifest(checkpoint)

    def remove(self, checkpoint: CheckpointId) -> None:
        self._inner.remove(checkpoint)

    def load(
        self, checkpoint: CheckpointId, expected: CheckpointManifest
    ) -> Checkpoint[StateTree]:
        return self._inner.load(checkpoint, expected)

    def save(
        self, manifest: CheckpointManifest, states: Mapping[str, StateTree]
    ) -> CheckpointId:
        handle = self._inner.save(manifest, states)
        fitting = manifest.fitting

        assert fitting is not None

        self.saved.append((handle, str(fitting.kind), manifest.step))

        if len(self.saved) == self._interrupt_at:
            raise KeyboardInterrupt

        return handle


def _kinds(store: DirectoryStore, /) -> tuple[tuple[CheckpointId, str], ...]:
    """Return every stored handle with the fitting kind its manifest records."""
    kinds: list[tuple[CheckpointId, str]] = []

    for handle in store.list():
        fitting = store.manifest(handle).fitting

        assert fitting is not None

        kinds.append((handle, str(fitting.kind)))

    return tuple(kinds)


def _assert_equal(
    expected: Mapping[str, StateTree], found: Mapping[str, StateTree], /
) -> None:
    """Assert two role-keyed component states are bitwise equal."""
    assert tuple(expected) == tuple(found)

    for role, state in expected.items():
        saved = cast("dict[str, StateTree]", state)
        live = cast("dict[str, StateTree]", found[role])

        assert tuple(saved) == tuple(live)

        for name, value in saved.items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, cast("torch.Tensor", live[name])), name
            else:
                assert value == live[name], name


RETAINED = (
    "trainer.epochs=4",
    "trainer.early_stopping.patience=4",
    "trainer.early_stopping.keep_best=1",
    "trainer.checkpoints.interval=2",
    "trainer.checkpoints.keep=1",
)


@pytest.mark.integration
def test_a_frozen_pretrained_encoder_is_checkpointed_by_identity_and_resumes_bitwise(
    tmp_path: Path,
) -> None:
    run_spec = _composed(
        tmp_path, *RETAINED, "model.token_encoder.trainable=false", pretrained=True
    )
    declared = cast("ComponentSpec", run_spec.model.token_encoder)
    control = _Run(run_spec)
    encoder = control.prepared.model.modules["token_encoder"]
    parameters = {name for name, _ in encoder.named_parameters()}
    control_store = _RecordingStore(tmp_path / "control")
    fit_training(
        control.prepared,
        control_store,
        control.template,
        control.streams,
        ResumePosition(0, 0),
        sinks=(),
    )

    assert parameters
    assert not any(parameter.requires_grad for parameter in encoder.parameters())

    for handle in control_store.list():
        recorded = control_store.manifest(handle).components["token_encoder"]
        block = torch.load(
            tmp_path / "control" / handle / "roles" / "token_encoder.pt",
            map_location="cpu",
            weights_only=True,
        )

        assert recorded.pinned == declared.pretrained_weights()
        assert parameters.isdisjoint(block)

    interrupted = _Run(run_spec)
    stopping_store = _RecordingStore(tmp_path / "interrupted", interrupt_at=2)

    with pytest.raises(KeyboardInterrupt):
        fit_training(
            interrupted.prepared,
            stopping_store,
            interrupted.template,
            interrupted.streams,
            ResumePosition(0, 0),
            sinks=(),
        )

    resumed = _Run(run_spec)
    store = DirectoryStore(tmp_path / "interrupted")
    latest = next(
        handle for handle, kind in reversed(_kinds(store)) if kind == "resume-latest"
    )
    checkpoint = store.load(latest, resumed.template)
    position = resume_from_checkpoint(
        checkpoint,
        resumed.template,
        resumed.prepared.model.prepare_restore,
        resumed.prepared.optimization.prepare_restore,
        resumed.streams,
    )
    fit_training(
        resumed.prepared,
        store,
        checkpoint.manifest,
        resumed.streams,
        position,
        sinks=(),
    )

    assert position.step > 0
    _assert_equal(control.prepared.model.state(), resumed.prepared.model.state())
    assert [
        (control_store.manifest(handle).step, kind)
        for handle, kind in _kinds(DirectoryStore(tmp_path / "control"))
    ] == [(store.manifest(handle).step, kind) for handle, kind in _kinds(store)]


@pytest.mark.integration
@pytest.mark.parametrize(("keep", "keep_best"), [(1, 1), (2, 1), (1, 2)])
def test_declared_retention_keeps_only_the_newest_declared_checkpoints(
    tmp_path: Path, keep: int, keep_best: int
) -> None:
    run_spec = _composed(
        tmp_path,
        "trainer.epochs=5",
        "trainer.early_stopping.patience=5",
        f"trainer.early_stopping.keep_best={keep_best}",
        "trainer.checkpoints.interval=2",
        f"trainer.checkpoints.keep={keep}",
        pretrained=False,
    )
    run = _Run(run_spec)
    store = _RecordingStore(tmp_path / "checkpoints")
    report = fit_training(
        run.prepared,
        store,
        run.template,
        run.streams,
        ResumePosition(0, 0),
        sinks=(),
    )
    written_latest = [
        handle for handle, kind, _ in store.saved if kind == "resume-latest"
    ]
    written_best = [
        handle for handle, kind, _ in store.saved if kind == "validation-best"
    ]
    kept = dict(_kinds(DirectoryStore(tmp_path / "checkpoints")))

    assert len(written_latest) > keep
    assert len(store.saved) > len(kept)
    assert [handle for handle in kept if kept[handle] == "resume-latest"] == (
        written_latest[-keep:]
    )
    assert [handle for handle in kept if kept[handle] == "validation-best"] == (
        written_best[-keep_best:]
    )
    assert report.latest_checkpoint == written_latest[-1]
    assert report.best_checkpoint == written_best[-1]


@pytest.mark.integration
def test_resuming_after_reordering_the_rule_list_is_refused_naming_the_digest(
    tmp_path: Path,
) -> None:
    run_spec = _composed(tmp_path, "trainer.epochs=1", pretrained=False)
    document = tmp_path / "reordered.yaml"
    document.write_text(
        "defaults:\n"
        "  - run/synthetic-parser\n"
        "data:\n"
        "  datasets:\n"
        "    train:\n"
        "      dataset:\n"
        "        grammar:\n"
        "          rules:\n"
        "            rules:\n"
        '              - {rule: application, view: "\\\\"}\n'
        "              - {rule: application, view: /}\n",
        encoding="utf-8",
        newline="\n",
    )
    reordered_spec = RunSpecCodec(search_path=config_search_path(tmp_path))(
        document, overrides=("trainer.epochs=1",)
    )
    run = _Run(run_spec)
    store = DirectoryStore(tmp_path / "checkpoints")
    report = fit_training(
        run.prepared,
        store,
        run.template,
        run.streams,
        ResumePosition(0, 0),
        sinks=(),
    )
    reordered = _Run(reordered_spec)

    assert (
        reordered.blueprint.action_set_fingerprint
        != run.blueprint.action_set_fingerprint
    )
    assert reordered.template.action_set == reordered.blueprint.action_set_fingerprint
    assert store.manifest(report.latest_checkpoint).action_set == (
        run.blueprint.action_set_fingerprint
    )

    with pytest.raises(FingerprintMismatchError) as refused:
        store.load(report.latest_checkpoint, reordered.template)

    (reason,) = (
        reason for reason in refused.value.reasons if "action set digest" in reason
    )
    assert run.blueprint.action_set_fingerprint in reason
    assert reordered.blueprint.action_set_fingerprint in reason


@pytest.mark.integration
def test_every_improving_epoch_writes_a_resumable_checkpoint_off_the_cadence(
    tmp_path: Path,
) -> None:
    run_spec = _composed(
        tmp_path,
        "trainer.epochs=3",
        "trainer.early_stopping.patience=3",
        "trainer.checkpoints.interval=10",
        pretrained=False,
    )
    run = _Run(run_spec)
    store = _RecordingStore(tmp_path / "checkpoints")
    fit_training(
        run.prepared,
        store,
        run.template,
        run.streams,
        ResumePosition(0, 0),
        sinks=(),
    )
    best = {step for _, kind, step in store.saved if kind == "validation-best"}
    latest = {step for _, kind, step in store.saved if kind == "resume-latest"}
    final = max(step for _, _, step in store.saved)

    assert len(best) > 1
    assert latest == best | {final}


@pytest.mark.integration
def test_the_epoch_that_stops_the_run_writes_a_resumable_checkpoint_off_the_cadence(
    tmp_path: Path,
) -> None:
    run_spec = _composed(
        tmp_path,
        "trainer.epochs=5",
        "trainer.early_stopping.patience=1",
        "trainer.early_stopping.minimum_delta=1000000000.0",
        "trainer.checkpoints.interval=10",
        "trainer.checkpoints.keep=5",
        pretrained=False,
    )
    run = _Run(run_spec)
    store = _RecordingStore(tmp_path / "checkpoints")
    report = fit_training(
        run.prepared,
        store,
        run.template,
        run.streams,
        ResumePosition(0, 0),
        sinks=(),
    )
    best = [step for _, kind, step in store.saved if kind == "validation-best"]
    latest = [step for _, kind, step in store.saved if kind == "resume-latest"]

    assert report.stopped_early
    assert len(best) == 1
    assert latest == [best[0], report.optimizer_steps]
    assert store.manifest(report.latest_checkpoint).step == report.optimizer_steps
