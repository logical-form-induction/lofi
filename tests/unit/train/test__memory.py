from typing import TYPE_CHECKING, cast

import pytest
import torch

from lofi.nn.policy import DeviceKind, DevicePolicy
from lofi.train import ActivationMemoryExceededError, StepMemory

if TYPE_CHECKING:
    from collections.abc import Callable

MEBIBYTE = 1 << 20


class CountingDevice:
    """Stands in for an accelerator allocator's peak count."""

    def __init__(self) -> None:
        self.held = 0
        self.peak = 0
        self.resets = 0

    def allocate(self, size: int) -> Callable[[], None]:
        self.held += size
        self.peak = max(self.peak, self.held)

        def free() -> None:
            self.held -= size

        return free

    def install(self, monkeypatch: pytest.MonkeyPatch) -> DevicePolicy:
        def reset(device: torch.device) -> None:
            del device
            self.peak = self.held
            self.resets += 1

        def peak(device: torch.device) -> int:
            del device
            return self.peak

        def synchronize(device: torch.device) -> None:
            del device

        monkeypatch.setattr(torch.accelerator, "reset_peak_memory_stats", reset)
        monkeypatch.setattr(torch.accelerator, "max_memory_allocated", peak)
        monkeypatch.setattr(torch.accelerator, "synchronize", synchronize)

        class Policy:
            @property
            def kind(self) -> DeviceKind:
                return DeviceKind.CUDA

            @property
            def device(self) -> torch.device:
                return torch.device("cuda", 1)

        return cast("DevicePolicy", Policy())


def accelerators() -> tuple[str, ...]:
    """Return the accelerator this machine has, or nothing."""
    accelerator = torch.accelerator.current_accelerator()

    if accelerator is None or not torch.accelerator.is_available():
        return ()

    return (accelerator.type,)


def policy(device: str) -> DevicePolicy:
    return DevicePolicy(torch.device(device), torch.float32)


def allocate(device: str, mebibytes: int) -> torch.Tensor:
    """Return a live tensor of the given size on a device."""
    return torch.ones(mebibytes * MEBIBYTE // 4, device=device)


def test_the_processor_keeps_no_memory_count() -> None:
    memory = StepMemory(policy("cpu"), None)
    memory.reset()

    with memory.step():
        allocate("cpu", 1)

    assert not memory.measured
    assert memory.high_water() is None


def test_the_error_is_exhausted_memory() -> None:
    error = ActivationMemoryExceededError("cuda:0", peak_bytes=2_000, max_bytes=1_000)

    assert isinstance(error, MemoryError)
    assert (error.device, error.peak_bytes, error.max_bytes) == ("cuda:0", 2_000, 1_000)
    assert str(error) == (
        "a training step held 2000 bytes on cuda:0 at its peak, above the "
        "activation_memory_bytes bound of 1000 its device policy declares"
    )


@pytest.mark.parametrize("device", accelerators())
def test_a_step_above_the_bound_stops_with_its_peak(device: str) -> None:
    memory = StepMemory(policy(device), 4 * MEBIBYTE)
    memory.reset()

    def overrun() -> None:
        with memory.step():
            held = allocate(device, 8)
            del held

    with pytest.raises(ActivationMemoryExceededError) as raised:
        overrun()

    assert raised.value.max_bytes == 4 * MEBIBYTE
    assert raised.value.peak_bytes >= 8 * MEBIBYTE
    assert raised.value.device == str(torch.device(device))


@pytest.mark.parametrize("device", accelerators())
def test_each_step_is_measured_from_its_own_start(device: str) -> None:
    memory = StepMemory(policy(device), 4 * MEBIBYTE)
    memory.reset()
    earlier = allocate(device, 8)
    del earlier

    with memory.step():
        held = allocate(device, 1)
        del held


@pytest.mark.parametrize("device", accelerators())
def test_the_high_water_spans_every_measured_step(device: str) -> None:
    memory = StepMemory(policy(device), 64 * MEBIBYTE)
    memory.reset()

    with memory.step():
        large = allocate(device, 16)
        del large

    with memory.step():
        small = allocate(device, 1)
        del small

    high_water = memory.high_water()

    assert memory.measured
    assert high_water is not None
    assert high_water >= 16 * MEBIBYTE


@pytest.mark.parametrize("device", accelerators())
def test_an_unbounded_step_is_not_measured_apart(device: str) -> None:
    memory = StepMemory(policy(device), None)
    memory.reset()

    with memory.step():
        large = allocate(device, 16)
        del large

    high_water = memory.high_water()

    assert high_water is not None
    assert high_water >= 16 * MEBIBYTE


def test_a_counted_step_is_measured_from_its_own_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counted = CountingDevice()
    memory = StepMemory(counted.install(monkeypatch), 100)
    memory.reset()
    counted.allocate(500)()

    with memory.step():
        counted.allocate(80)()

    def overrun() -> None:
        with memory.step():
            free = counted.allocate(60)
            counted.allocate(50)()
            free()

    with pytest.raises(ActivationMemoryExceededError) as raised:
        overrun()

    assert (raised.value.device, raised.value.peak_bytes) == ("cuda:1", 110)
    assert memory.high_water() == 500
    assert counted.resets == 3


def test_a_counted_point_reports_its_high_water_across_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counted = CountingDevice()
    memory = StepMemory(counted.install(monkeypatch), 100)
    memory.reset()

    with memory.step():
        counted.allocate(90)()

    counted.allocate(40)()

    with memory.step():
        counted.allocate(10)()

    assert memory.high_water() == 90
    assert repr(memory) == "StepMemory('cuda:1', max_bytes=100)"
