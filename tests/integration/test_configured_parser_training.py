from __future__ import annotations

from hashlib import sha256
from importlib.metadata import version
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch

from lofi.run import (
    RandomStreams,
    SeedPolicy,
    TrainingBlueprint,
    plan_training,
    resolve_device_policy,
)
from lofi.spec import PACKAGED_CONFIG_ROOT, ComponentSpec, RunSpecCodec
from lofi.train import (
    CheckpointManifest,
    ComponentRecord,
    DirectoryStore,
    ParserFittingPlan,
    PrecisionRecord,
    ResumePosition,
    RngRecord,
    StateReference,
    fit_training,
    prepare_training,
    resume_from_checkpoint,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi.train import PreparedTraining
    from lofi.train._store import StateTree

CONFIG = Path(__file__).resolve().parents[2] / "lofi/configs/run/synthetic-parser.yaml"
CODEC = RunSpecCodec(search_path=(PACKAGED_CONFIG_ROOT,))


def assert_state_equal(expected: StateTree, found: StateTree, /) -> None:
    """Assert recursively over module state that may contain tensor metadata."""
    if isinstance(expected, torch.Tensor):
        assert isinstance(found, torch.Tensor)
        assert torch.equal(expected, found)

        return

    if isinstance(expected, dict):
        assert isinstance(found, dict)
        assert tuple(expected) == tuple(found)

        for expected_value, found_value in zip(
            expected.values(), found.values(), strict=True
        ):
            assert_state_equal(expected_value, found_value)

        return

    if isinstance(expected, list):
        assert isinstance(found, list)
        assert len(expected) == len(found)

        for expected_value, found_value in zip(expected, found, strict=True):
            assert_state_equal(expected_value, found_value)

        return

    if isinstance(expected, tuple):
        assert isinstance(found, tuple)
        assert len(expected) == len(found)

        for expected_value, found_value in zip(expected, found, strict=True):
            assert_state_equal(expected_value, found_value)

        return

    assert type(expected) is type(found)
    assert expected == found


def state_digest(state: StateTree, /) -> str:
    """Return the digest of the exact weights-only serialization of state."""
    serialized = BytesIO()
    torch.save(state, serialized)

    return sha256(serialized.getvalue()).hexdigest()


def configured_declarations(
    blueprint: TrainingBlueprint, /
) -> Mapping[str, tuple[str, ComponentSpec]]:
    """Return checkpoint roles beside their structural declarations."""
    model = blueprint.run_spec.model
    optional = {
        "token_encoder": ("model.token_encoder", model.token_encoder),
        "subword_pooler": ("model.subword_pooler", model.subword_pooler),
        "controller": ("model.controller", model.controller),
        "parser": ("model.parser", model.parser),
    }
    declared = {
        role: (path, cast("ComponentSpec", declaration))
        for role, (path, declaration) in optional.items()
        if declaration is not None
    }
    declared.update(
        {
            role: (f"model.encoders.{role}", declaration)
            for role, declaration in model.encoders.items()
        }
    )
    declared.update(
        {
            role: (f"model.decoders.{role}", declaration)
            for role, declaration in model.decoders.items()
        }
    )

    return declared


def state_reference(key: str, state: StateTree, /) -> StateReference:
    """Return a reference carrying the actual serialized-state digest."""
    return StateReference(key=key, digest=state_digest(state))


def checkpoint_manifest(
    blueprint: TrainingBlueprint,
    prepared: PreparedTraining,
    streams: RandomStreams,
    /,
    *,
    step: int,
    data_position: int,
) -> CheckpointManifest:
    """Return the complete manifest for one configured parser position."""
    component_state, optimization = prepared.state()
    declarations = configured_declarations(blueprint)

    if frozenset(declarations) != frozenset(component_state):
        msg = (
            f"configured declarations name {sorted(declarations)}, but component "
            f"state names {sorted(component_state)}"
        )
        raise ValueError(msg)

    signature = prepared.model.codec.codec.index.digest
    components: dict[str, ComponentRecord] = {}

    for role, state in component_state.items():
        path, declaration = declarations[role]
        components[role] = ComponentRecord(
            tag=cast("str", declaration.model_dump()["kind"]),
            spec_fingerprint=blueprint.component_fingerprints[path],
            signature_fingerprint=signature,
            codec_identity=blueprint.grammar_fingerprint,
            state=state_reference(role, cast("StateTree", state)),
            labels=(
                None
                if role not in prepared.model.checkpoint_labels
                else dict(prepared.model.checkpoint_labels[role])
            ),
        )

    policy = prepared.model.policy
    compute_dtype = policy.autocast_dtype or policy.dtype

    return CheckpointManifest(
        run_id=blueprint.declaration_fingerprint,
        step=step,
        data_position=data_position,
        code_version=version("lofi"),
        action_set="actions-a1",
        components=components,
        rng=RngRecord(streams=dict(streams.snapshot())),
        precision=PrecisionRecord(
            compute_dtype=str(compute_dtype).removeprefix("torch."),
            autocast_device_type=(
                policy.device.type
                if policy.autocast_device_type is None
                else policy.autocast_device_type
            ),
            gradient_scaling=optimization.scaler is not None,
        ),
        optimizer=state_reference(
            "optimizer", cast("StateTree", optimization.optimizer)
        ),
        scaler=(
            None
            if optimization.scaler is None
            else state_reference("scaler", cast("StateTree", optimization.scaler))
        ),
        scheduler=(
            None
            if optimization.scheduler is None
            else state_reference("scheduler", cast("StateTree", optimization.scheduler))
        ),
    )


def checkpoint_blocks(prepared: PreparedTraining, /) -> dict[str, StateTree]:
    """Return every component and run-state block named by its manifest."""
    component_state, optimization = prepared.state()
    blocks = {role: cast("StateTree", state) for role, state in component_state.items()}
    blocks["optimizer"] = cast("StateTree", optimization.optimizer)

    if optimization.scaler is not None:
        blocks["scaler"] = cast("StateTree", optimization.scaler)

    if optimization.scheduler is not None:
        blocks["scheduler"] = cast("StateTree", optimization.scheduler)

    return blocks


def test_packaged_parser_fits_and_restores_a_final_checkpoint(tmp_path: Path) -> None:
    run_spec = CODEC(CONFIG)
    blueprint = plan_training(run_spec)
    policy = resolve_device_policy(run_spec)

    def fresh() -> tuple[PreparedTraining, RandomStreams]:
        streams = RandomStreams(SeedPolicy(17, samplers=blueprint.sampler_streams))
        streams.apply()

        return prepare_training(blueprint, streams, policy), streams

    trained, trained_streams = fresh()
    store = DirectoryStore(tmp_path / "checkpoints")
    template = checkpoint_manifest(
        blueprint,
        trained,
        trained_streams,
        step=0,
        data_position=0,
    )
    report = fit_training(
        trained,
        store,
        template,
        trained_streams,
        ResumePosition(0, 0),
        sinks=(),
    )
    saved_components, saved_optimization = trained.state()
    restored, restored_streams = fresh()
    expected_manifest = checkpoint_manifest(
        blueprint,
        restored,
        restored_streams,
        step=0,
        data_position=0,
    )
    loaded = store.load(report.latest_checkpoint, expected_manifest)
    torch.rand(())
    position = resume_from_checkpoint(
        loaded,
        expected_manifest,
        restored.model.prepare_restore,
        restored.optimization.prepare_restore,
        restored_streams,
    )
    restored_components, restored_optimization = restored.state()

    plan = cast("ParserFittingPlan", trained.plan)
    assert report.optimizer_steps == plan.steps_per_epoch * plan.epochs == 2
    assert report.best_checkpoint != report.latest_checkpoint
    assert loaded.manifest.fitting is not None
    assert loaded.manifest.fitting.kind == "resume-latest"
    assert loaded.manifest.fitting.best_checkpoint == report.best_checkpoint
    assert position == ResumePosition(report.optimizer_steps, report.data_position)
    assert all(
        any(
            trained.model.named_parameters[name].grad is not None
            for name in trained.model.groups.members(group)
        )
        for group in trained.model.groups.active
    )
    assert not any(
        name.startswith("_rules._controller.") for name in saved_components["parser"]
    )
    assert dict(restored_streams.snapshot()) == loaded.manifest.rng.streams
    assert_state_equal(
        cast("StateTree", saved_optimization.optimizer),
        cast("StateTree", restored_optimization.optimizer),
    )
    assert_state_equal(
        cast("StateTree", saved_optimization.scaler),
        cast("StateTree", restored_optimization.scaler),
    )
    assert_state_equal(
        cast("StateTree", saved_optimization.scheduler),
        cast("StateTree", restored_optimization.scheduler),
    )

    for role in saved_components:
        expected = saved_components[role]
        found = restored_components[role]

        for key in expected:
            assert_state_equal(expected[key], found[key])

    resumed, resumed_streams = fresh()
    resumed_expected = checkpoint_manifest(
        blueprint,
        resumed,
        resumed_streams,
        step=0,
        data_position=0,
    )
    first_latest = next(
        checkpoint
        for handle in store.list()
        if (checkpoint := store.load(handle, resumed_expected)).manifest.fitting
        is not None
        and checkpoint.manifest.fitting.kind == "resume-latest"
        and checkpoint.manifest.fitting.completed_epochs == 1
    )
    resumed_position = resume_from_checkpoint(
        first_latest,
        resumed_expected,
        resumed.model.prepare_restore,
        resumed.optimization.prepare_restore,
        resumed_streams,
    )
    continued = fit_training(
        resumed,
        store,
        first_latest.manifest,
        resumed_streams,
        resumed_position,
        sinks=(),
    )

    assert tuple(epoch.epoch for epoch in continued.epochs) == (1,)
    assert continued.optimizer_steps == report.optimizer_steps
    assert continued.data_position == report.data_position
