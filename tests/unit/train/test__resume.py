"""Tests for completed-step checkpoint restoration."""

import random
from copy import copy, deepcopy
from hashlib import sha256
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, cast

import pytest
import torch
from tests.conformance.checkpoint_store import EMPTY_SEED_POLICY

from lofi.nn.policy import DevicePolicy
from lofi.run import RandomStreams, SeedPolicy
from lofi.train._assembly import resume_from_checkpoint
from lofi.train._autocast import AutocastPolicy, OptimizationState, StepOptions
from lofi.train._boundary import (
    CheckpointManifest,
    ComponentRecord,
    GroundingRecord,
    PrecisionRecord,
    RngRecord,
    StateReference,
)
from lofi.train._checkpoint import (
    Checkpoint,
    CheckpointId,
    FingerprintMismatchError,
)
from lofi.train._resume import (
    ResumePosition,
    _copy_state_tree,
    _restore_checkpoint,
    _validate_compatible_state,
)
from lofi.train._store import DirectoryStore, StateTree

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from torch import Generator, Tensor, nn
    from torch.optim import Optimizer

    from lofi._types import JsonValue

ROLE: Final = "model"
HANDLE: Final = CheckpointId("000000-run-step2")


def component(role: str = ROLE) -> ComponentRecord:
    return ComponentRecord(
        tag="linear",
        spec_fingerprint="spec-a",
        signature_fingerprint="signature-a",
        codec_identity="codec-a",
        state=StateReference(key=role, digest="component-state"),
    )


def manifest(
    *,
    step: int = 2,
    data_position: int = 2,
    rng: Mapping[str, str] | None = None,
    scheduler: bool = False,
    grounding: GroundingRecord | None = None,
) -> CheckpointManifest:
    return CheckpointManifest(
        run_id="run",
        step=step,
        data_position=data_position,
        code_version="1",
        action_set="actions-a1",
        components={ROLE: component()},
        rng=RngRecord(
            streams={"python": "saved", "torch": "saved"} if rng is None else dict(rng)
        ),
        precision=PrecisionRecord(
            compute_dtype="float32",
            autocast_device_type="cpu",
            gradient_scaling=False,
        ),
        grounding=grounding,
        optimizer=StateReference(key="optimizer", digest="optimizer-state"),
        scheduler=(
            StateReference(key="scheduler", digest="scheduler-state")
            if scheduler
            else None
        ),
    )


def module_state(module: nn.Module) -> dict[str, StateTree]:
    return cast("dict[str, StateTree]", dict(module.state_dict()))


def run_state(*, scheduler: bool = False) -> dict[str, StateTree]:
    state: dict[str, StateTree] = {"optimizer": {"state": {}, "param_groups": []}}

    if scheduler:
        state["scheduler"] = {}

    return state


class ModuleComponents:
    """Prepare complete role-keyed state for simple module fixtures."""

    def __init__(
        self,
        modules: Mapping[str, nn.Module],
        events: list[str] | None = None,
    ) -> None:
        self.modules = dict(modules)
        self.events = events

    def prepare_restore(
        self, states: Mapping[str, Mapping[str, StateTree]], /
    ) -> Callable[[], None]:
        """Validate and own complete component state before returning its commit."""
        expected = frozenset(self.modules)
        found = frozenset(states)
        missing = sorted(expected - found)
        extra = sorted(found - expected)

        if missing or extra:
            msg = (
                f"resume components have missing roles {missing} and extra roles "
                f"{extra}"
            )
            raise ValueError(msg)

        prepared: dict[str, dict[str, StateTree]] = {}

        for role in sorted(self.modules):
            saved = dict(states[role])
            live = cast("dict[str, StateTree]", dict(self.modules[role].state_dict()))
            _validate_compatible_state(
                f"component role {role!r}",
                cast("StateTree", saved),
                cast("StateTree", live),
            )
            detached = cast(
                "dict[str, StateTree]",
                _copy_state_tree(cast("StateTree", saved)),
            )
            isolated = deepcopy(self.modules[role])

            try:
                isolated.load_state_dict(
                    cast(
                        "dict[str, StateTree]",
                        _copy_state_tree(cast("StateTree", detached)),
                    )
                )

            except Exception as error:
                msg = (
                    f"component role {role!r} refused checkpoint state during "
                    "preparation"
                )
                raise ValueError(msg) from error

            prepared[role] = detached

        def restore() -> None:
            for role in sorted(self.modules):
                self.modules[role].load_state_dict(prepared[role])

            if self.events is not None:
                self.events.append("components")

        return restore


def component_preparer(
    modules: Mapping[str, nn.Module], /
) -> Callable[[Mapping[str, Mapping[str, StateTree]]], Callable[[], None]]:
    """Return a component-state preparation callback for module fixtures."""
    return ModuleComponents(modules).prepare_restore


class SpyStreams:
    """Random streams that record restoration without changing global state."""

    def __init__(
        self,
        current: Mapping[str, str],
        events: list[str] | None = None,
    ) -> None:
        self.current = dict(current)
        self.restored: list[dict[str, str]] = []
        self.validations = 0
        self.events = [] if events is None else events
        self.generator = torch.Generator()
        self.retry = 0

    @property
    def seed_policy(self) -> Mapping[str, JsonValue]:
        return EMPTY_SEED_POLICY

    def sampler_seed(self, name: str, /) -> int:
        del name

        return self.retry

    def loader_generator(self, name: str, /) -> Generator:
        del name

        return self.generator

    def worker_initializer(self, name: str, /) -> Callable[[int], None]:
        del name

        return self._initialize_worker

    @staticmethod
    def _initialize_worker(worker: int) -> None:
        del worker

    def snapshot(self) -> Mapping[str, str]:
        return MappingProxyType(self.current)

    def validate(self, streams: Mapping[str, str], /) -> None:
        self.validations += 1

        if set(streams) != set(self.current):
            msg = "random-stream state has incompatible keys"
            raise ValueError(msg)

    def prepare_restore(self, streams: Mapping[str, str], /) -> Callable[[], None]:
        self.validate(streams)
        prepared = dict(streams)

        def restore() -> None:
            self.events.append("streams")
            self.current = dict(prepared)
            self.restored.append(dict(prepared))

        return restore

    def restore(self, streams: Mapping[str, str], /) -> None:
        self.prepare_restore(streams)()

    def reseed(self, attempt: int, /) -> None:
        if attempt < 1:
            msg = f"a numerical retry counts from one, not {attempt}"
            raise ValueError(msg)

        self.retry = attempt
        self.generator.manual_seed(attempt)


def loaded(
    saved: CheckpointManifest,
    model_state: Mapping[str, StateTree],
    *,
    saved_run_state: Mapping[str, StateTree] | None = None,
) -> Checkpoint[StateTree]:
    states: Mapping[str, StateTree] = {ROLE: dict(model_state)}
    held_run_state: Mapping[str, StateTree] = (
        {} if saved_run_state is None else saved_run_state
    )

    return Checkpoint[StateTree](
        HANDLE,
        saved,
        states,
        run_state=held_run_state,
    )


def unchanged(module: nn.Module, before: Mapping[str, Tensor]) -> bool:
    return all(
        torch.equal(value, before[name]) for name, value in module.state_dict().items()
    )


def preparer(
    restore: Callable[[OptimizationState], None], /
) -> Callable[[OptimizationState], Callable[[], None]]:
    def prepare(state: OptimizationState) -> Callable[[], None]:
        def prepared() -> None:
            restore(state)

        return prepared

    return prepare


def test_a_resume_position_is_frozen_and_compares_by_both_counters() -> None:
    position = ResumePosition(3, 17)

    assert position.step == 3
    assert position.data_position == 17
    assert position == ResumePosition(3, 17)
    assert position != ResumePosition(4, 17)
    assert position != ResumePosition(3, 18)
    assert hash(position) == hash(ResumePosition(3, 17))
    assert repr(position) == "ResumePosition(3, 17)"

    for held in (position, copy(position), deepcopy(position)):
        assert held is position

        with pytest.raises(AttributeError, match="immutable"):
            held.__setattr__("_step", 4)

        with pytest.raises(AttributeError, match="immutable"):
            held.__delattr__("_step")

        assert (held.step, held.data_position) == (3, 17)


@pytest.mark.parametrize(("step", "data_position"), [(-1, 0), (0, -1)])
def test_a_resume_position_refuses_a_negative_counter(
    step: int,
    data_position: int,
) -> None:
    with pytest.raises(ValueError, match="negative"):
        ResumePosition(step, data_position)


def test_preflight_refuses_missing_and_extra_component_roles_before_mutation() -> None:
    saved = torch.nn.Linear(2, 1)
    checkpoint = loaded(manifest(), module_state(saved), saved_run_state=run_state())
    optimization: list[OptimizationState] = []
    streams = SpyStreams({"python": "current", "torch": "current"})

    for modules, named in (
        ({}, ROLE),
        (
            {ROLE: torch.nn.Linear(2, 1), "unwired": torch.nn.Linear(2, 1)},
            "unwired",
        ),
    ):
        before = {
            role: {name: value.clone() for name, value in module.state_dict().items()}
            for role, module in modules.items()
        }

        with pytest.raises(ValueError, match=named):
            resume_from_checkpoint(
                checkpoint,
                manifest(),
                component_preparer(modules),
                preparer(optimization.append),
                streams,
            )

        assert all(unchanged(module, before[role]) for role, module in modules.items())
        assert optimization == []
        assert streams.restored == []


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_preflight_refuses_incomplete_module_state_before_mutation(
    change: str,
) -> None:
    saved = torch.nn.Linear(2, 1)
    state = module_state(saved)

    if change == "missing":
        del state["bias"]

    else:
        state["unclaimed"] = torch.ones(1)

    live = torch.nn.Linear(2, 1)
    before = {name: value.clone() for name, value in live.state_dict().items()}
    optimization: list[OptimizationState] = []
    streams = SpyStreams({"python": "current", "torch": "current"})
    checkpoint = loaded(manifest(), state, saved_run_state=run_state())

    with pytest.raises(ValueError, match=rf"{ROLE!r}.*{change} keys"):
        resume_from_checkpoint(
            checkpoint,
            manifest(),
            component_preparer({ROLE: live}),
            preparer(optimization.append),
            streams,
        )

    assert unchanged(live, before)
    assert optimization == []
    assert streams.restored == []


def test_preflight_refuses_incompatible_module_values_before_any_mutation() -> None:
    saved = torch.nn.Linear(2, 1)
    state = module_state(saved)
    state["weight"] = torch.full_like(saved.weight, 7.0)
    state["bias"] = torch.ones(2)
    live = torch.nn.Linear(2, 1)
    before = {name: value.clone() for name, value in live.state_dict().items()}
    optimization: list[OptimizationState] = []
    streams = SpyStreams({"python": "current", "torch": "current"})
    checkpoint = loaded(manifest(), state, saved_run_state=run_state())

    with pytest.raises(ValueError, match=r"bias.*shape"):
        resume_from_checkpoint(
            checkpoint,
            manifest(),
            component_preparer({ROLE: live}),
            preparer(optimization.append),
            streams,
        )

    assert unchanged(live, before)
    assert optimization == []
    assert streams.restored == []


def test_preflight_exercises_every_module_hook_before_mutation() -> None:
    first_saved = torch.nn.Linear(2, 1)
    second_saved = torch.nn.Linear(2, 1)
    first_live = torch.nn.Linear(2, 1)
    second_live = torch.nn.Linear(2, 1)
    before = {
        "a": {name: value.clone() for name, value in first_live.state_dict().items()},
        "z": {name: value.clone() for name, value in second_live.state_dict().items()},
    }

    def reject(
        _module: nn.Module,
        _state: dict[str, Tensor],
        _prefix: str,
        _metadata: dict[str, JsonValue],
        _strict: Literal[True],
        _missing: list[str],
        _unexpected: list[str],
        _errors: list[str],
    ) -> None:
        msg = "module hook rejected state"
        raise ValueError(msg)

    second_live.register_load_state_dict_pre_hook(reject)
    saved_manifest = manifest().with_(
        components={"a": component("a"), "z": component("z")}
    )
    checkpoint = Checkpoint[StateTree](
        HANDLE,
        saved_manifest,
        {"a": module_state(first_saved), "z": module_state(second_saved)},
        run_state=run_state(),
    )
    optimization: list[OptimizationState] = []
    streams = SpyStreams({"python": "current", "torch": "current"})

    with pytest.raises(ValueError, match=r"role 'z'.*refused"):
        resume_from_checkpoint(
            checkpoint,
            saved_manifest,
            component_preparer({"a": first_live, "z": second_live}),
            preparer(optimization.append),
            streams,
        )

    assert unchanged(first_live, before["a"])
    assert unchanged(second_live, before["z"])
    assert optimization == []
    assert streams.restored == []


def test_module_preflight_owns_detached_checkpoint_state() -> None:
    saved = torch.nn.Linear(2, 1)

    with torch.no_grad():
        saved.weight.fill_(4.0)
        saved.bias.fill_(-3.0)

    live = torch.nn.Linear(2, 1)
    checkpoint = loaded(manifest(), module_state(saved), saved_run_state=run_state())
    checkpoint_state = cast("dict[str, StateTree]", checkpoint.states[ROLE])
    optimization: list[OptimizationState] = []
    streams = SpyStreams({"python": "current", "torch": "current"})

    def prepare(state: OptimizationState) -> Callable[[], None]:
        checkpoint_state["weight"] = torch.ones(3)

        return preparer(optimization.append)(state)

    resume_from_checkpoint(
        checkpoint,
        manifest(),
        component_preparer({ROLE: live}),
        prepare,
        streams,
    )

    assert all(
        torch.equal(value, saved.state_dict()[name])
        for name, value in live.state_dict().items()
    )
    assert len(optimization) == 1
    assert len(streams.restored) == 1


def test_preflight_refuses_incomplete_run_state_before_mutation() -> None:
    saved = torch.nn.Linear(2, 1)
    live = torch.nn.Linear(2, 1)
    before = {name: value.clone() for name, value in live.state_dict().items()}
    optimization: list[OptimizationState] = []
    streams = SpyStreams({"python": "current", "torch": "current"})
    checkpoint = loaded(manifest(), module_state(saved))

    with pytest.raises(ValueError, match=r"missing keys.*optimizer"):
        resume_from_checkpoint(
            checkpoint,
            manifest(),
            component_preparer({ROLE: live}),
            preparer(optimization.append),
            streams,
        )

    assert unchanged(live, before)
    assert optimization == []
    assert streams.restored == []


def test_preflight_refuses_incompatible_optimization_before_any_mutation() -> None:
    saved = torch.nn.Linear(2, 1)
    saved_policy = optimization(saved)
    saved_policy.step(lambda: saved.weight.square().sum() + saved.bias.square().sum())
    saved_optimization = saved_policy.state()
    optimizer_state = deepcopy(saved_optimization.optimizer)
    parameter_states = cast("dict[int, StateTree]", optimizer_state["state"])
    bias_state = cast("dict[str, StateTree]", parameter_states[1])
    bias_state["exp_avg"] = torch.ones(2)
    live = torch.nn.Linear(2, 1)
    live_policy = optimization(live)
    before_module = {name: value.clone() for name, value in live.state_dict().items()}
    before_optimization = deepcopy(live_policy.state().optimizer)
    streams = SpyStreams({"python": "current", "torch": "current"})
    checkpoint = loaded(
        manifest(scheduler=True),
        module_state(saved),
        saved_run_state={
            "optimizer": optimizer_state,
            "scheduler": cast("dict[str, StateTree]", saved_optimization.scheduler),
        },
    )

    with pytest.raises(ValueError, match=r"exp_avg.*shape"):
        resume_from_checkpoint(
            checkpoint,
            manifest(scheduler=True),
            component_preparer({ROLE: live}),
            live_policy.prepare_restore,
            streams,
        )

    assert unchanged(live, before_module)
    assert live_policy.state().optimizer == before_optimization
    assert streams.restored == []


def test_preflight_exercises_optimizer_hooks_before_module_mutation() -> None:
    saved = torch.nn.Linear(2, 1)
    saved_policy = optimization(saved)
    saved_policy.step(lambda: saved.weight.square().sum() + saved.bias.square().sum())
    saved_optimization = saved_policy.state()
    live = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(live.parameters(), lr=0.03)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    live_policy = AutocastPolicy(
        optimizer,
        DevicePolicy(torch.device("cpu"), torch.float32),
        StepOptions(),
        scheduler=scheduler,
    )

    def reject(_optimizer: Optimizer, _state: dict[str, StateTree]) -> None:
        msg = "optimizer hook rejected state"
        raise ValueError(msg)

    optimizer.register_load_state_dict_pre_hook(reject)
    before_module = {name: value.clone() for name, value in live.state_dict().items()}
    before_optimization = deepcopy(live_policy.state().optimizer)
    streams = SpyStreams({"python": "current", "torch": "current"})
    checkpoint = loaded(
        manifest(scheduler=True),
        module_state(saved),
        saved_run_state={
            "optimizer": saved_optimization.optimizer,
            "scheduler": cast("dict[str, StateTree]", saved_optimization.scheduler),
        },
    )

    with pytest.raises(ValueError, match="optimization policy refused"):
        resume_from_checkpoint(
            checkpoint,
            manifest(scheduler=True),
            component_preparer({ROLE: live}),
            live_policy.prepare_restore,
            streams,
        )

    assert unchanged(live, before_module)
    assert live_policy.state().optimizer == before_optimization
    assert streams.restored == []


def test_preflight_refuses_incomplete_random_state_before_mutation() -> None:
    saved_manifest = manifest(rng={"python": "saved"})
    saved = torch.nn.Linear(2, 1)
    live = torch.nn.Linear(2, 1)
    before = {name: value.clone() for name, value in live.state_dict().items()}
    optimization: list[OptimizationState] = []
    streams = SpyStreams({"python": "current", "torch": "current"})
    checkpoint = loaded(
        saved_manifest,
        module_state(saved),
        saved_run_state=run_state(),
    )

    with pytest.raises(ValueError, match=r"missing keys.*torch"):
        resume_from_checkpoint(
            checkpoint,
            saved_manifest,
            component_preparer({ROLE: live}),
            preparer(optimization.append),
            streams,
        )

    assert unchanged(live, before)
    assert optimization == []
    assert streams.restored == []


def test_preflight_refuses_malformed_random_state_before_any_mutation() -> None:
    saved = torch.nn.Linear(2, 1)
    live = torch.nn.Linear(2, 1)
    before_module = {name: value.clone() for name, value in live.state_dict().items()}
    streams = RandomStreams(SeedPolicy(17))
    streams.apply()
    before_random = dict(streams.snapshot())
    malformed = {**before_random, "python": "not base64"}
    saved_manifest = manifest(rng=malformed)
    optimization: list[OptimizationState] = []
    checkpoint = loaded(
        saved_manifest,
        module_state(saved),
        saved_run_state=run_state(),
    )

    with pytest.raises(ValueError, match=r"python.*base64"):
        resume_from_checkpoint(
            checkpoint,
            saved_manifest,
            component_preparer({ROLE: live}),
            preparer(optimization.append),
            streams,
        )

    assert unchanged(live, before_module)
    assert optimization == []
    assert dict(streams.snapshot()) == before_random


def test_each_manifest_mismatch_is_refused_before_mutation() -> None:
    saved_manifest = manifest()
    saved = torch.nn.Linear(2, 1)
    checkpoint = loaded(
        saved_manifest,
        module_state(saved),
        saved_run_state=run_state(),
    )
    apart = (
        (saved_manifest.with_(run_id="another"), "run identity"),
        (
            saved_manifest.with_(
                precision=PrecisionRecord(
                    compute_dtype="float64",
                    autocast_device_type="cpu",
                    gradient_scaling=False,
                )
            ),
            "precision",
        ),
        (
            saved_manifest.with_(
                grounding=GroundingRecord(
                    variant="expected-embedding",
                    scorer="entailment-head",
                    source="sentence",
                    gradient_path="PATHWISE",
                    prover=None,
                    prover_version=None,
                    budget_calls=None,
                    budget_wall_seconds=None,
                    budget_cpu_seconds=None,
                    calls=0,
                    questions=0,
                    questions_per_second=0.0,
                    memo_hits=0,
                    memo_hit_rate=0.0,
                    cpu_seconds=0.0,
                    wall_seconds=0.0,
                    process_starts=None,
                    label_agreement=None,
                    abstentions={},
                    degenerate_mass=0.0,
                    malformed_mass=0.0,
                    reward_mean=None,
                    variance=None,
                    verified_share=None,
                    out_of_fragment_share=0.0,
                )
            ),
            "grounding",
        ),
        (saved_manifest.with_(optimizer=None), "run-state topology"),
    )

    for expected, named in apart:
        live = torch.nn.Linear(2, 1)
        before = {name: value.clone() for name, value in live.state_dict().items()}
        optimization: list[OptimizationState] = []
        streams = SpyStreams({"python": "current", "torch": "current"})

        with pytest.raises(FingerprintMismatchError, match=named):
            resume_from_checkpoint(
                checkpoint,
                expected,
                component_preparer({ROLE: live}),
                preparer(optimization.append),
                streams,
            )

        assert unchanged(live, before)
        assert optimization == []
        assert streams.restored == []


def test_restoration_orders_components_before_optimization_before_streams() -> None:
    saved = torch.nn.Linear(2, 1)

    with torch.no_grad():
        saved.weight.fill_(3.0)
        saved.bias.fill_(-2.0)

    live = torch.nn.Linear(2, 1)
    events: list[str] = []
    components = ModuleComponents({ROLE: live}, events)
    streams = SpyStreams({"python": "current", "torch": "current"}, events)
    seen_optimization: list[OptimizationState] = []

    def restore_optimization(state: OptimizationState) -> None:
        assert all(
            torch.equal(live.state_dict()[name], value)
            for name, value in saved.state_dict().items()
        )
        seen_optimization.append(state)
        events.append("optimization")

    checkpoint = loaded(
        manifest(),
        module_state(saved),
        saved_run_state=run_state(),
    )
    position = resume_from_checkpoint(
        checkpoint,
        manifest(step=0, data_position=0),
        components.prepare_restore,
        preparer(restore_optimization),
        streams,
    )

    assert position == ResumePosition(2, 2)
    assert events == ["components", "optimization", "streams"]
    assert len(seen_optimization) == 1
    assert seen_optimization[0].optimizer == run_state()["optimizer"]
    assert streams.restored == [{"python": "saved", "torch": "saved"}]
    assert streams.validations == 1


def test_the_interior_restore_consumes_no_manifest() -> None:
    saved = torch.nn.Linear(2, 1)
    live = torch.nn.Linear(2, 1)
    optimization = OptimizationState({"state": {}, "param_groups": []})
    seen: list[OptimizationState] = []
    streams = SpyStreams({"python": "current", "torch": "current"})
    position = ResumePosition(4, 9)
    restore_components = component_preparer({ROLE: live})({ROLE: module_state(saved)})

    returned = _restore_checkpoint(
        restore_components,
        lambda: seen.append(optimization),
        streams.prepare_restore({"python": "saved", "torch": "saved"}),
        position,
    )

    assert returned is position
    assert seen == [optimization]
    assert streams.current == {"python": "saved", "torch": "saved"}
    assert streams.validations == 1


def parameter_digest(module: nn.Module) -> str:
    digest = sha256()

    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(
            bytes(
                value.detach().cpu().contiguous().view(torch.uint8).flatten().tolist()
            )
        )

    return digest.hexdigest()


def optimization(model: nn.Module) -> AutocastPolicy:
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)

    return AutocastPolicy(
        optimizer,
        DevicePolicy(torch.device("cpu"), torch.float32),
        StepOptions(),
        scheduler=scheduler,
    )


def advance(
    model: nn.Linear,
    policy: AutocastPolicy,
    streams: RandomStreams,
    data_position: int,
) -> int:
    features = torch.rand(3) + torch.rand(
        3,
        generator=streams.loader_generator("fit"),
    )
    target = torch.tensor(random.random() + data_position / 10)
    residual = model(features).squeeze() - target
    result = policy.step(residual.square)

    assert result.stepped

    return data_position + 1


def draws(streams: RandomStreams) -> tuple[float, float, float]:
    return (
        random.random(),
        torch.rand(()).item(),
        torch.rand((), generator=streams.loader_generator("fit")).item(),
    )


def exact_manifest(
    streams: RandomStreams,
    state: OptimizationState,
    *,
    step: int,
    data_position: int,
) -> CheckpointManifest:
    return CheckpointManifest(
        run_id="exact-run",
        step=step,
        data_position=data_position,
        code_version="1",
        action_set="actions-a1",
        components={ROLE: component()},
        rng=RngRecord(streams=dict(streams.snapshot())),
        precision=PrecisionRecord(
            compute_dtype="float32",
            autocast_device_type="cpu",
            gradient_scaling=False,
        ),
        optimizer=StateReference(key="optimizer", digest="optimizer-state"),
        scheduler=(
            None
            if state.scheduler is None
            else StateReference(key="scheduler", digest="scheduler-state")
        ),
    )


def exact_blocks(
    model: nn.Module,
    state: OptimizationState,
) -> dict[str, StateTree]:
    blocks: dict[str, StateTree] = {
        ROLE: cast("StateTree", dict(model.state_dict())),
        "optimizer": cast("StateTree", state.optimizer),
    }

    if state.scheduler is not None:
        blocks["scheduler"] = cast("StateTree", state.scheduler)

    return blocks


def test_completed_step_resume_matches_uninterrupted_execution(tmp_path: Path) -> None:
    seed_policy = SeedPolicy(29, loaders=("fit",))
    uninterrupted_streams = RandomStreams(seed_policy)
    uninterrupted_streams.apply()
    uninterrupted_model = torch.nn.Linear(3, 1)
    uninterrupted_policy = optimization(uninterrupted_model)
    data_position = 0

    for _ in range(2):
        data_position = advance(
            uninterrupted_model,
            uninterrupted_policy,
            uninterrupted_streams,
            data_position,
        )

    saved_state = uninterrupted_policy.state()
    saved_manifest = exact_manifest(
        uninterrupted_streams,
        saved_state,
        step=2,
        data_position=data_position,
    )
    store = DirectoryStore(tmp_path / "checkpoints")
    handle = store.save(
        saved_manifest,
        exact_blocks(uninterrupted_model, saved_state),
    )
    expected = saved_manifest.with_(
        step=0,
        data_position=0,
        rng=RngRecord(streams=dict.fromkeys(saved_manifest.rng.streams, "fresh")),
    )
    checkpoint = store.load(handle, expected)

    for _ in range(2):
        data_position = advance(
            uninterrupted_model,
            uninterrupted_policy,
            uninterrupted_streams,
            data_position,
        )

    uninterrupted_digest = parameter_digest(uninterrupted_model)
    uninterrupted_draws = draws(uninterrupted_streams)

    resumed_model = torch.nn.Linear(3, 1)
    resumed_policy = optimization(resumed_model)
    resumed_streams = RandomStreams(seed_policy)
    resumed_streams.apply()
    resumed = resume_from_checkpoint(
        checkpoint,
        expected,
        component_preparer({ROLE: resumed_model}),
        resumed_policy.prepare_restore,
        resumed_streams,
    )
    resumed_position = resumed.data_position

    for _ in range(2):
        resumed_position = advance(
            resumed_model,
            resumed_policy,
            resumed_streams,
            resumed_position,
        )

    assert resumed == ResumePosition(2, 2)
    assert resumed_position == data_position == 4
    assert parameter_digest(resumed_model) == uninterrupted_digest
    assert draws(resumed_streams) == uninterrupted_draws
