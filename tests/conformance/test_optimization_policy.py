from typing import TYPE_CHECKING, Final, override

import pytest
from tests.conformance.optimization_policy import (
    OPTIMIZATION_POLICY_CONFORMANCE_LAWS,
    OptimizationPolicyConformance,
)

from lofi.train import OptimizationResult
from lofi.train.laws import (
    AMP_STEP_ORDER,
    OPTIMIZATION_POLICY_LAWS,
    OptimizationCase,
    OptimizationFixture,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

INITIAL_SCALE: Final = 8.0


class Calls:
    __slots__ = ("events",)

    def __init__(self) -> None:
        self.events: list[str] = []

    def add(self, event: str) -> None:
        self.events.append(event)

    def read(self) -> Sequence[str]:
        return tuple(self.events)


class Loss:
    __slots__ = ("_calls", "finite")

    def __init__(self, calls: Calls, *, finite: bool) -> None:
        self._calls = calls
        self.finite = finite

    def backward(self) -> None:
        self._calls.add("backward")


class Optimizer:
    __slots__ = ("_calls",)

    def __init__(self, calls: Calls) -> None:
        self._calls = calls

    def zero_grad(self) -> None:
        self._calls.add("zero_grad")

    def step(self) -> None:
        self._calls.add("step")


class Scaler:
    __slots__ = ("_calls", "_scale")

    def __init__(self, calls: Calls) -> None:
        self._calls = calls
        self._scale = INITIAL_SCALE

    def scale(self, loss: Loss) -> Loss:
        self._calls.add("scale")

        return loss

    def unscale_(self, _optimizer: Optimizer) -> None:
        self._calls.add("unscale_")

    def step(self, optimizer: Optimizer) -> None:
        optimizer.step()

    def get_scale(self) -> float:
        self._calls.add("get_scale")

        return self._scale

    def update(self) -> None:
        self._calls.add("update")
        self._scale *= 2.0


class Scheduler:
    __slots__ = ("_calls",)

    def __init__(self, calls: Calls) -> None:
        self._calls = calls

    def step(self) -> None:
        self._calls.add("scheduler_step")


class ReferencePolicy:
    __slots__ = (
        "_accumulation_steps",
        "_calls",
        "_clip",
        "_optimizer",
        "_position",
        "_scaler",
        "_scheduler",
    )

    def __init__(
        self,
        calls: Calls,
        *,
        amp: bool,
        accumulation_steps: int,
    ) -> None:
        self._calls = calls
        self._optimizer = Optimizer(calls)
        self._scaler = Scaler(calls) if amp else None
        self._scheduler = Scheduler(calls)
        self._clip = lambda: calls.add("clip")
        self._accumulation_steps = accumulation_steps
        self._position = 0

    def _finish(self) -> float | None:
        scaler = self._scaler

        if scaler is None:
            self._clip()
            self._optimizer.step()
            scale = None

        else:
            scaler.unscale_(self._optimizer)
            self._clip()
            scaler.step(self._optimizer)
            scale = scaler.get_scale()
            scaler.update()

        self._scheduler.step()

        return scale

    def step(self, forward: Callable[[], Loss], /) -> OptimizationResult:
        if self._position == 0:
            self._optimizer.zero_grad()

        self._calls.add("autocast_enter")

        try:
            loss = forward()

        finally:
            self._calls.add("autocast_exit")

        if not loss.finite:
            self._position = 0

            return OptimizationResult(loss_finite=False, stepped=False, scale=None)

        scaler = self._scaler

        if scaler is None:
            loss.backward()

        else:
            scaler.scale(loss).backward()

        self._position += 1

        if self._position < self._accumulation_steps:
            return OptimizationResult(loss_finite=True, stepped=False, scale=None)

        scale = self._finish()
        self._position = 0

        return OptimizationResult(loss_finite=True, stepped=True, scale=scale)


class StepsBeforeClipping(ReferencePolicy):
    __slots__ = ()

    @override
    def _finish(self) -> float | None:
        scaler = self._scaler

        if scaler is None:
            self._optimizer.step()
            self._clip()
            scale = None

        else:
            scaler.unscale_(self._optimizer)
            scaler.step(self._optimizer)
            self._clip()
            scale = scaler.get_scale()
            scaler.update()

        self._scheduler.step()

        return scale


class UpdatesBeforeReadingScale(ReferencePolicy):
    __slots__ = ()

    @override
    def _finish(self) -> float | None:
        scaler = self._scaler

        if scaler is None:
            return super()._finish()

        scaler.unscale_(self._optimizer)
        self._clip()
        scaler.step(self._optimizer)
        scaler.update()
        scale = scaler.get_scale()
        self._scheduler.step()

        return scale


def fixture_for(
    policy_type: type[ReferencePolicy] = ReferencePolicy,
) -> OptimizationFixture[Loss]:
    def build(
        *,
        amp: bool,
        accumulation_steps: int,
    ) -> OptimizationCase[Loss]:
        calls = Calls()
        policy = policy_type(
            calls,
            amp=amp,
            accumulation_steps=accumulation_steps,
        )

        def loss(*, finite: bool) -> Loss:
            calls.add("forward")

            return Loss(calls, finite=finite)

        return OptimizationCase(
            policy=policy,
            finite=lambda: loss(finite=True),
            nonfinite=lambda: loss(finite=False),
            calls=calls.read,
        )

    return OptimizationFixture(build=build, scale=INITIAL_SCALE)


class TestReferencePolicy(OptimizationPolicyConformance):
    @pytest.fixture
    def conformance(self) -> OptimizationFixture[Loss]:
        return fixture_for()


class TestPolicyThatStepsBeforeClipping(OptimizationPolicyConformance):
    @pytest.fixture
    def conformance(self) -> OptimizationFixture[Loss]:
        return fixture_for(StepsBeforeClipping)

    @pytest.mark.xfail(strict=True, reason="step and clip are transposed")
    @override
    def test_an_amp_step_matches_the_recorded_order[LossT](
        self,
        conformance: OptimizationFixture[LossT],
    ) -> None:
        super().test_an_amp_step_matches_the_recorded_order(conformance)

    @pytest.mark.xfail(strict=True, reason="step and clip are transposed")
    @override
    def test_mixed_precision_off_skips_scaler_operations[LossT](
        self,
        conformance: OptimizationFixture[LossT],
    ) -> None:
        super().test_mixed_precision_off_skips_scaler_operations(conformance)


class TestPolicyThatUpdatesBeforeReadingScale(OptimizationPolicyConformance):
    @pytest.fixture
    def conformance(self) -> OptimizationFixture[Loss]:
        return fixture_for(UpdatesBeforeReadingScale)

    @pytest.mark.xfail(strict=True, reason="update and get_scale are transposed")
    @override
    def test_an_amp_step_matches_the_recorded_order[LossT](
        self,
        conformance: OptimizationFixture[LossT],
    ) -> None:
        super().test_an_amp_step_matches_the_recorded_order(conformance)

    @pytest.mark.xfail(strict=True, reason="update and get_scale are transposed")
    @override
    def test_the_step_scale_is_read_before_update[LossT](
        self,
        conformance: OptimizationFixture[LossT],
    ) -> None:
        super().test_the_step_scale_is_read_before_update(conformance)


def test_the_suite_binds_a_method_to_every_optimization_policy_law() -> None:
    bound = {
        method.__name__.removeprefix("test_")
        for method in OPTIMIZATION_POLICY_CONFORMANCE_LAWS
    }

    assert bound == {law.__name__ for law in OPTIMIZATION_POLICY_LAWS}


def test_the_required_amp_order_is_read_from_a_recorded_run() -> None:
    case = fixture_for().build(amp=True, accumulation_steps=1)

    case.policy.step(case.finite)

    assert tuple(case.calls()) == AMP_STEP_ORDER
