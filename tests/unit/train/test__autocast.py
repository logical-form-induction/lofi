from contextlib import contextmanager
from copy import deepcopy
from typing import TYPE_CHECKING, Final, cast, override

import pytest
import torch
from tests.conformance.optimization_policy import OptimizationPolicyConformance
from torch.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

import lofi.train._autocast as autocast_module
from lofi.nn.policy import DevicePolicy
from lofi.train import (
    ActivationMemoryExceededError,
    CheckpointManifest,
    ComponentRecord,
    DirectoryStore,
    OptimizationResult,
    PrecisionRecord,
    RngRecord,
    StateReference,
    StepMemory,
)
from lofi.train._autocast import (
    AutocastPolicy,
    GradScaling,
    OptimizationState,
    StepOptions,
)
from lofi.train.laws import OptimizationCase, OptimizationFixture

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from pathlib import Path

    from torch import Tensor

    from lofi.train import StateTree

INITIAL_SCALE: Final = 8.0
FAST_GROWTH: Final = GradScaling(
    init_scale=INITIAL_SCALE,
    growth_factor=2.0,
    backoff_factor=0.5,
    growth_interval=1,
)


def cpu_policy(autocast_dtype: torch.dtype | None) -> DevicePolicy:
    return DevicePolicy(torch.device("cpu"), torch.float32, autocast_dtype)


def weighted_sum(parameter: Tensor, weights: Sequence[float]) -> Callable[[], Tensor]:
    def forward() -> Tensor:
        return (parameter * torch.tensor(weights)).sum()

    return forward


class Calls:
    __slots__ = ("events",)

    def __init__(self) -> None:
        self.events: list[str] = []

    def add(self, event: str) -> None:
        self.events.append(event)

    def read(self) -> Sequence[str]:
        return tuple(self.events)


class RecordedOptimizer(SGD):
    def __init__(self, calls: Calls, parameters: list[Tensor], lr: float) -> None:
        super().__init__(parameters, lr=lr)

        self.calls = calls
        self.register_step_post_hook(lambda *_: calls.add("step"))

    @override
    def zero_grad(self, set_to_none: bool = True) -> None:
        self.calls.add("zero_grad")
        super().zero_grad(set_to_none=set_to_none)


class RecordedScheduler(StepLR):
    def __init__(self, calls: Calls, optimizer: SGD) -> None:
        self.recorder: Calls | None = None

        super().__init__(optimizer, step_size=1, gamma=0.5)

        self.recorder = calls

    @override
    def step(self, epoch: int | None = None) -> None:
        if self.recorder is not None:
            self.recorder.add("scheduler_step")

        super().step(epoch)


class RecordedDevicePolicy(DevicePolicy):
    __slots__ = ("calls",)

    def __init__(self, calls: Calls, autocast_dtype: torch.dtype | None) -> None:
        super().__init__(torch.device("cpu"), torch.float32, autocast_dtype)

        self.calls = calls

    @override
    @contextmanager
    def autocast(self, *, enabled: bool = True) -> Iterator[None]:
        self.calls.add("autocast_enter")

        try:
            with super().autocast(enabled=enabled):
                yield

        finally:
            self.calls.add("autocast_exit")


def recording_scaler(
    calls: Calls, devices: list[str], monkeypatch: pytest.MonkeyPatch
) -> Callable[..., GradScaler]:
    def build(
        device: str,
        *,
        init_scale: float,
        growth_factor: float,
        backoff_factor: float,
        growth_interval: int,
    ) -> GradScaler:
        devices.append(device)
        scaler = GradScaler(
            device,
            init_scale=init_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
        )
        scale, unscale_, get_scale, update = (
            scaler.scale,
            scaler.unscale_,
            scaler.get_scale,
            scaler.update,
        )

        def recorded_scale(outputs: Tensor) -> Tensor:
            calls.add("scale")

            return scale(outputs)

        def recorded_unscale_(optimizer: torch.optim.Optimizer) -> None:
            calls.add("unscale_")
            unscale_(optimizer)

        def recorded_get_scale() -> float:
            calls.add("get_scale")

            return get_scale()

        def recorded_update() -> None:
            calls.add("update")
            update()

        monkeypatch.setattr(scaler, "scale", recorded_scale)
        monkeypatch.setattr(scaler, "unscale_", recorded_unscale_)
        monkeypatch.setattr(scaler, "get_scale", recorded_get_scale)
        monkeypatch.setattr(scaler, "update", recorded_update)

        return scaler

    return build


def recorded_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[OptimizationFixture[Tensor], list[str]]:
    devices: list[str] = []

    def build(*, amp: bool, accumulation_steps: int) -> OptimizationCase[Tensor]:
        calls = Calls()
        model = torch.nn.Linear(4, 1)
        inputs = torch.randn(3, 4)
        optimizer = RecordedOptimizer(calls, list(model.parameters()), lr=0.1)
        scheduler = RecordedScheduler(calls, optimizer)
        scaling = FAST_GROWTH if amp else None
        device_policy = RecordedDevicePolicy(calls, torch.float16 if amp else None)

        def recorded_clip(parameters: list[Tensor], max_norm: float) -> Tensor:
            calls.add("clip")

            return clip_grad_norm_(parameters, max_norm)

        monkeypatch.setattr(autocast_module, "clip_grad_norm_", recorded_clip)
        monkeypatch.setattr(
            autocast_module,
            "GradScaler",
            recording_scaler(calls, devices, monkeypatch),
        )
        policy = AutocastPolicy(
            optimizer,
            device_policy,
            StepOptions(
                accumulation_steps=accumulation_steps,
                max_grad_norm=1.0,
                scaling=scaling,
            ),
            scheduler=scheduler,
        )
        calls.events.clear()

        def loss(*, finite: bool) -> Tensor:
            calls.add("forward")
            value = model(inputs).pow(2).mean()
            value.register_hook(lambda _: calls.add("backward"))

            return value if finite else value * torch.inf

        return OptimizationCase(
            policy=policy,
            finite=lambda: loss(finite=True),
            nonfinite=lambda: loss(finite=False),
            calls=calls.read,
        )

    return OptimizationFixture(build=build, scale=INITIAL_SCALE), devices


class TestAutocastPolicy(OptimizationPolicyConformance):
    @pytest.fixture
    def conformance(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> OptimizationFixture[Tensor]:
        fixture, _ = recorded_fixture(monkeypatch)

        return fixture


def test_the_scaler_is_built_for_the_device_policy_autocasts_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, devices = recorded_fixture(monkeypatch)

    fixture.build(amp=True, accumulation_steps=1)
    fixture.build(amp=False, accumulation_steps=1)

    assert devices == [cpu_policy(torch.float16).autocast_device_type]


def test_scaling_on_a_device_family_without_autocast_is_refused() -> None:
    parameter = torch.zeros(2, device="meta", requires_grad=True)
    optimizer = SGD([parameter], lr=0.1)
    device_policy = DevicePolicy(torch.device("meta"), torch.float32)

    with pytest.raises(ValueError, match="meta"):
        AutocastPolicy(optimizer, device_policy, StepOptions(scaling=FAST_GROWTH))


def test_bfloat16_computes_under_autocast_with_no_scaler() -> None:
    model = torch.nn.Linear(4, 1)
    before = model.weight.detach().clone()
    inputs = torch.randn(3, 4)
    seen: list[torch.dtype] = []
    optimizer = SGD(model.parameters(), lr=0.1)
    policy = AutocastPolicy(optimizer, cpu_policy(torch.bfloat16), StepOptions())

    def forward() -> Tensor:
        loss = model(inputs).pow(2).mean()
        seen.append(loss.dtype)

        return loss

    result = policy.step(forward)

    assert seen == [torch.bfloat16]
    assert result.scale is None
    assert result.stepped
    assert not policy.scales_gradients
    assert policy.state().scaler is None
    assert not torch.equal(model.weight.detach(), before)


def test_full_precision_computes_in_the_parameter_dtype() -> None:
    model = torch.nn.Linear(4, 1)
    inputs = torch.randn(3, 4)
    seen: list[torch.dtype] = []
    policy = AutocastPolicy(
        SGD(model.parameters(), lr=0.1), cpu_policy(None), StepOptions()
    )

    def forward() -> Tensor:
        loss = model(inputs).pow(2).mean()
        seen.append(loss.dtype)

        return loss

    policy.step(forward)

    assert seen == [torch.float32]


def test_gradients_are_clipped_after_unscaling_and_before_stepping() -> None:
    parameter = torch.tensor([3.0, 4.0], requires_grad=True)
    optimizer = SGD([parameter], lr=1.0)
    policy = AutocastPolicy(
        optimizer,
        cpu_policy(torch.float16),
        StepOptions(max_grad_norm=1.0, scaling=FAST_GROWTH),
    )
    before = parameter.detach().clone()

    result = policy.step(weighted_sum(parameter, [100.0, 0.0]))

    assert result.stepped
    assert result.scale == INITIAL_SCALE
    moved = before - parameter.detach()
    assert torch.allclose(moved, torch.tensor([1.0, 0.0]), atol=1e-4)


def test_the_reported_scale_is_the_one_the_step_was_taken_under() -> None:
    parameter = torch.tensor([1.0], requires_grad=True)
    policy = AutocastPolicy(
        SGD([parameter], lr=0.1),
        cpu_policy(torch.float16),
        StepOptions(scaling=FAST_GROWTH),
    )

    reported = [policy.step(lambda: parameter.pow(2).sum()).scale for _ in range(3)]

    assert reported == [INITIAL_SCALE, 2 * INITIAL_SCALE, 4 * INITIAL_SCALE]


def test_accumulation_averages_the_units_into_one_step() -> None:
    accumulated = torch.tensor([1.0, -2.0], requires_grad=True)
    single = accumulated.detach().clone().requires_grad_()
    units = ([3.0, 1.0], [1.0, 5.0])
    mean = [(first + second) / len(units) for first, second in zip(*units, strict=True)]
    windowed = AutocastPolicy(
        SGD([accumulated], lr=0.5),
        cpu_policy(None),
        StepOptions(accumulation_steps=len(units)),
    )
    whole = AutocastPolicy(SGD([single], lr=0.5), cpu_policy(None), StepOptions())

    first = windowed.step(weighted_sum(accumulated, units[0]))
    second = windowed.step(weighted_sum(accumulated, units[1]))
    whole.step(weighted_sum(single, mean))

    assert not first.stepped
    assert second.stepped
    assert torch.equal(accumulated.detach(), single.detach())


def test_a_nonfinite_loss_discards_the_window_and_leaves_the_scale_alone() -> None:
    parameter = torch.tensor([1.0, 1.0], requires_grad=True)
    fresh = parameter.detach().clone().requires_grad_()
    policy = AutocastPolicy(
        SGD([parameter], lr=0.5),
        cpu_policy(torch.float16),
        StepOptions(accumulation_steps=2, scaling=FAST_GROWTH),
    )
    control = AutocastPolicy(
        SGD([fresh], lr=0.5),
        cpu_policy(torch.float16),
        StepOptions(accumulation_steps=2, scaling=FAST_GROWTH),
    )
    before = parameter.detach().clone()

    policy.step(weighted_sum(parameter, [100.0, 0.0]))
    broken = policy.step(lambda: parameter.sum() * torch.inf)
    assert broken == OptimizationResult(loss_finite=False, stepped=False, scale=None)
    assert torch.equal(parameter.detach(), before)

    resumed = [policy.step(weighted_sum(parameter, [0.0, 1.0])) for _ in range(2)]

    for _ in range(2):
        control.step(weighted_sum(fresh, [0.0, 1.0]))

    assert [unit.stepped for unit in resumed] == [False, True]
    assert resumed[1].scale == INITIAL_SCALE
    assert torch.equal(parameter.detach(), fresh.detach())


def test_a_nonfinite_loss_is_reported_when_the_forward_is_not_autocast() -> None:
    parameter = torch.tensor([1.0], requires_grad=True)
    policy = AutocastPolicy(SGD([parameter], lr=0.1), cpu_policy(None), StepOptions())

    result = policy.step(lambda: parameter.sum() * torch.nan)

    assert not result.loss_finite
    assert parameter.grad is None


def fitted(
    parameter: Tensor, *, scaling: GradScaling | None, with_scheduler: bool
) -> AutocastPolicy:
    optimizer = SGD([parameter], lr=0.5, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.5) if with_scheduler else None

    return AutocastPolicy(
        optimizer,
        cpu_policy(None if scaling is None else torch.float16),
        StepOptions(scaling=scaling),
        scheduler=scheduler,
    )


def manifest(state: OptimizationState) -> CheckpointManifest:
    references = {
        "optimizer": StateReference(key="optimizer", digest="a1"),
        "scaler": StateReference(key="scaler", digest="b2"),
        "scheduler": StateReference(key="scheduler", digest="c3"),
    }

    return CheckpointManifest(
        run_id="run",
        step=3,
        data_position=0,
        code_version="0",
        action_set="actions-a1",
        components={
            "model": ComponentRecord(
                tag="linear",
                spec_fingerprint="s",
                signature_fingerprint="g",
                codec_identity="c",
                state=StateReference(key="model", digest="d4"),
            )
        },
        rng=RngRecord(streams={}),
        precision=PrecisionRecord(
            compute_dtype="float16",
            autocast_device_type="cpu",
            gradient_scaling=state.scaler is not None,
        ),
        optimizer=references["optimizer"],
        scaler=None if state.scaler is None else references["scaler"],
        scheduler=None if state.scheduler is None else references["scheduler"],
    )


def blocks(parameter: Tensor, state: OptimizationState) -> dict[str, StateTree]:
    held: dict[str, StateTree] = {
        "model": {"weight": parameter.detach().clone()},
        "optimizer": state.optimizer,
    }

    if state.scaler is not None:
        held["scaler"] = state.scaler

    if state.scheduler is not None:
        held["scheduler"] = state.scheduler

    return held


def read_back(
    store: DirectoryStore, parameter: Tensor, policy: AutocastPolicy
) -> OptimizationState:
    state = policy.state()
    saved = manifest(state)
    handle = store.save(saved, blocks(parameter, state))
    loaded = store.load(handle, saved)
    run_state = cast("dict[str, dict[str, StateTree]]", dict(loaded.run_state))

    return OptimizationState(
        run_state["optimizer"],
        scaler=run_state.get("scaler"),
        scheduler=run_state.get("scheduler"),
    )


def test_a_resumed_run_restores_the_saved_scale_rather_than_the_initial_one(
    tmp_path: Path,
) -> None:
    parameter = torch.tensor([1.0], requires_grad=True)
    policy = fitted(parameter, scaling=FAST_GROWTH, with_scheduler=True)

    for _ in range(3):
        policy.step(lambda: parameter.pow(2).sum())

    store = DirectoryStore(tmp_path / "checkpoints")
    read = read_back(store, parameter, policy)
    resumed_parameter = parameter.detach().clone().requires_grad_()
    resumed = fitted(resumed_parameter, scaling=FAST_GROWTH, with_scheduler=True)
    resumed.restore(read)

    continued = policy.step(lambda: parameter.pow(2).sum())
    after_resume = resumed.step(lambda: resumed_parameter.pow(2).sum())

    assert after_resume.scale == continued.scale == 8 * INITIAL_SCALE
    assert torch.equal(resumed_parameter.detach(), parameter.detach())


def test_restoring_carries_the_optimizer_and_scheduler_state_with_the_scale(
    tmp_path: Path,
) -> None:
    parameter = torch.tensor([2.0, -1.0], requires_grad=True)
    policy = fitted(parameter, scaling=FAST_GROWTH, with_scheduler=True)

    for _ in range(2):
        policy.step(lambda: parameter.pow(2).sum())

    read = read_back(DirectoryStore(tmp_path / "checkpoints"), parameter, policy)
    resumed_parameter = parameter.detach().clone().requires_grad_()
    resumed = fitted(resumed_parameter, scaling=FAST_GROWTH, with_scheduler=True)
    restarted_parameter = parameter.detach().clone().requires_grad_()
    restarted = fitted(restarted_parameter, scaling=FAST_GROWTH, with_scheduler=True)
    resumed.restore(read)

    policy.step(lambda: parameter.pow(2).sum())
    resumed.step(lambda: resumed_parameter.pow(2).sum())
    restarted.step(lambda: restarted_parameter.pow(2).sum())

    assert torch.equal(resumed_parameter.detach(), parameter.detach())
    assert not torch.equal(restarted_parameter.detach(), parameter.detach())


@pytest.mark.parametrize("damage", ["shape", "dtype", "state-tree"])
def test_restore_preflights_optimizer_values_before_mutation(damage: str) -> None:
    source_parameter = torch.tensor([2.0, -1.0], requires_grad=True)
    source = fitted(source_parameter, scaling=None, with_scheduler=False)
    source.step(lambda: source_parameter.pow(2).sum())
    optimizer_state = deepcopy(source.state().optimizer)
    parameter_state = cast(
        "dict[str, StateTree]",
        cast("dict[int, StateTree]", optimizer_state["state"])[0],
    )

    if damage == "shape":
        parameter_state["momentum_buffer"] = torch.zeros(3)
        named = "shape"

    elif damage == "dtype":
        parameter_state["momentum_buffer"] = torch.zeros(2, dtype=torch.float64)
        named = "dtype"

    else:
        groups = cast("list[StateTree]", optimizer_state["param_groups"])
        group = cast("dict[str, StateTree]", groups[0])
        group["lr"] = cast("StateTree", torch.device("cpu"))
        named = "unsupported"

    target_parameter = torch.tensor([4.0, 5.0], requires_grad=True)
    target = fitted(target_parameter, scaling=None, with_scheduler=False)
    before_parameter = target_parameter.detach().clone()
    before_optimization = deepcopy(target.state().optimizer)

    with pytest.raises((TypeError, ValueError), match=named):
        target.restore(OptimizationState(optimizer_state))

    assert torch.equal(target_parameter.detach(), before_parameter)
    assert target.state().optimizer == before_optimization


def test_prepared_restore_owns_detached_optimizer_state() -> None:
    source_parameter = torch.tensor([2.0, -1.0], requires_grad=True)
    source = fitted(source_parameter, scaling=None, with_scheduler=False)
    source.step(lambda: source_parameter.pow(2).sum())
    state = source.state()
    source_states = cast("dict[int, StateTree]", state.optimizer["state"])
    source_parameter_state = cast("dict[str, StateTree]", source_states[0])
    expected = cast("Tensor", source_parameter_state["momentum_buffer"]).clone()
    target_parameter = torch.tensor([4.0, 5.0], requires_grad=True)
    target = fitted(target_parameter, scaling=None, with_scheduler=False)
    prepared = target.prepare_restore(state)

    source_parameter_state["momentum_buffer"] = torch.ones(3)
    prepared()

    restored_states = cast("dict[int, StateTree]", target.state().optimizer["state"])
    restored_parameter_state = cast("dict[str, StateTree]", restored_states[0])

    assert torch.equal(
        cast("Tensor", restored_parameter_state["momentum_buffer"]), expected
    )


def test_preparation_refuses_a_nonscalar_optimizer_step() -> None:
    source_parameter = torch.tensor([2.0, -1.0], requires_grad=True)
    source = AutocastPolicy(
        torch.optim.AdamW([source_parameter], lr=0.1),
        cpu_policy(None),
        StepOptions(),
    )
    source.step(lambda: source_parameter.pow(2).sum())
    state = source.state()
    source_states = cast("dict[int, StateTree]", state.optimizer["state"])
    source_parameter_state = cast("dict[str, StateTree]", source_states[0])
    source_parameter_state["step"] = torch.ones(2)
    target_parameter = torch.tensor([4.0, 5.0], requires_grad=True)
    target = AutocastPolicy(
        torch.optim.AdamW([target_parameter], lr=0.1),
        cpu_policy(None),
        StepOptions(),
    )
    before = deepcopy(target.state().optimizer)

    with pytest.raises(ValueError, match=r"step.*scalar optimizer control"):
        target.prepare_restore(state)

    assert target.state().optimizer == before


def test_a_policy_without_a_scaler_or_scheduler_saves_neither(
    tmp_path: Path,
) -> None:
    parameter = torch.tensor([1.0], requires_grad=True)
    policy = fitted(parameter, scaling=None, with_scheduler=False)
    policy.step(lambda: parameter.pow(2).sum())

    state = policy.state()
    read = read_back(DirectoryStore(tmp_path / "checkpoints"), parameter, policy)

    assert state.scaler is None
    assert state.scheduler is None
    assert repr(state) == "OptimizationState(parts=[optimizer])"
    assert read.scaler is None
    assert read.scheduler is None


def test_restoring_refuses_state_whose_parts_are_not_the_policy_parts() -> None:
    parameter = torch.tensor([1.0], requires_grad=True)
    scaled = fitted(parameter, scaling=FAST_GROWTH, with_scheduler=True)
    plain = fitted(parameter, scaling=None, with_scheduler=False)

    with pytest.raises(ValueError, match="no scaler state"):
        scaled.restore(plain.state())

    with pytest.raises(ValueError, match="has no scaler"):
        plain.restore(scaled.state())

    with pytest.raises(ValueError, match="no scheduler state"):
        scaled.restore(
            OptimizationState(scaled.state().optimizer, scaler=scaled.state().scaler)
        )


def test_step_options_refuse_an_empty_window_and_a_degenerate_norm() -> None:
    with pytest.raises(ValueError, match="at least one unit"):
        StepOptions(accumulation_steps=0)

    for norm in (0.0, -1.0, torch.inf, torch.nan):
        with pytest.raises(ValueError, match="positive and finite"):
            StepOptions(max_grad_norm=norm)


def test_the_option_values_are_frozen_and_compare_by_content() -> None:
    scaling = GradScaling(
        init_scale=INITIAL_SCALE,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=1,
    )
    options = StepOptions(accumulation_steps=2, max_grad_norm=1.0, scaling=scaling)

    assert scaling == FAST_GROWTH
    assert hash(scaling) == hash(FAST_GROWTH)
    assert scaling != GradScaling(
        init_scale=INITIAL_SCALE,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2,
    )
    assert options == StepOptions(
        accumulation_steps=2, max_grad_norm=1.0, scaling=FAST_GROWTH
    )
    assert hash(options) == hash(
        StepOptions(accumulation_steps=2, max_grad_norm=1.0, scaling=FAST_GROWTH)
    )
    assert options != StepOptions(accumulation_steps=2, max_grad_norm=1.0)
    assert options.__eq__(scaling) is NotImplemented
    assert scaling.__eq__(options) is NotImplemented
    assert repr(scaling) == (
        "GradScaling(init_scale=8.0, growth_factor=2.0, backoff_factor=0.5, "
        "growth_interval=1)"
    )
    assert repr(options) == (
        f"StepOptions(accumulation_steps=2, max_grad_norm=1.0, scaling={scaling!r})"
    )

    with pytest.raises(AttributeError, match="immutable"):
        options.accumulation_steps = 3

    with pytest.raises(AttributeError, match="immutable"):
        scaling.init_scale = 1.0

    with pytest.raises(AttributeError, match="immutable"):
        OptimizationState({}).optimizer = {}


def test_the_policy_reports_its_optimizer_device_policy_and_options() -> None:
    parameter = torch.tensor([1.0], requires_grad=True)
    policy = AutocastPolicy(SGD([parameter], lr=0.1), cpu_policy(None), StepOptions())

    assert policy.options == StepOptions()
    assert repr(policy).startswith("AutocastPolicy(SGD, DevicePolicy(")
    assert repr(policy).endswith(f"{StepOptions()!r})")


def accelerators() -> tuple[str, ...]:
    """Return the accelerator this machine has, or nothing."""
    accelerator = torch.accelerator.current_accelerator()

    if accelerator is None or not torch.accelerator.is_available():
        return ()

    return (accelerator.type,)


@pytest.mark.parametrize("device", accelerators())
def test_a_metered_step_above_its_bound_stops_the_run(device: str) -> None:
    placement = DevicePolicy(torch.device(device), torch.float32)
    parameter = torch.ones(1 << 20, device=device, requires_grad=True)
    policy = AutocastPolicy(SGD([parameter], lr=0.1), placement, StepOptions())

    def forward() -> Tensor:
        return (parameter * parameter.exp()).sum()

    policy.step(forward)
    bounded = StepMemory(placement, 1)
    bounded.reset()

    with policy.metered(bounded), pytest.raises(ActivationMemoryExceededError):
        policy.step(forward)

    policy.step(forward)


def test_an_unmeasured_device_steps_under_any_meter() -> None:
    parameter = torch.tensor([1.0], requires_grad=True)
    placement = cpu_policy(None)
    policy = AutocastPolicy(SGD([parameter], lr=0.1), placement, StepOptions())

    with policy.metered(StepMemory(placement, 1)):
        result = policy.step(parameter.sum)

    assert result.stepped
