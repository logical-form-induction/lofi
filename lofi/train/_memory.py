"""Measures each training step's peak device memory against a declared bound.

An accelerator keeps an allocator count of the most bytes it has held since the
count was last reset. A bounded step resets that count when it starts and reads
it when it ends, so the figure is the step's own peak. The most any earlier
step or the work between steps held is folded into the meter before each
reset, so the meter still reports the most bytes the device held over the
whole point. The processor and the meta device keep no such count, so nothing
is measured on them.
"""

from contextlib import contextmanager
from typing import TYPE_CHECKING, Final, final, override

import torch

from lofi.nn.policy import DeviceKind

if TYPE_CHECKING:
    from collections.abc import Iterator

    from lofi.nn.policy import DevicePolicy

__all__ = ["ActivationMemoryExceededError", "StepMemory"]


_COUNTED: Final = frozenset({DeviceKind.CUDA, DeviceKind.MPS, DeviceKind.XPU})
"""The device families whose allocator keeps a peak count."""


class ActivationMemoryExceededError(MemoryError):
    """A training step that held more device memory than its policy allows.

    Parameters
    ----------
    device : str
        The name of the device the step ran on.
    peak_bytes : int
        The most bytes the device held at once during the step.
    max_bytes : int
        The ``activation_memory_bytes`` bound the run's device policy declares.

    Attributes
    ----------
    device : str
        The name of the device the step ran on.
    peak_bytes : int
        The step's measured peak.
    max_bytes : int
        The declared bound.
    """

    def __init__(self, device: str, *, peak_bytes: int, max_bytes: int) -> None:
        self.device = device
        self.peak_bytes = peak_bytes
        self.max_bytes = max_bytes
        super().__init__(
            f"a training step held {peak_bytes} bytes on {self.device} at its "
            f"peak, above the activation_memory_bytes bound of {max_bytes} its "
            f"device policy declares"
        )


@final
class StepMemory:
    """Measures training steps on one device against an optional bound.

    Parameters
    ----------
    policy : DevicePolicy
        The device policy the run trains under.
    max_bytes : int | None
        The most bytes one training step may hold at its peak, or None to
        leave steps unbounded.
    """

    __slots__ = ("_device", "_folded", "_max_bytes", "_measured")

    def __init__(self, policy: DevicePolicy, max_bytes: int | None, /) -> None:
        self._device = policy.device
        self._measured = policy.kind in _COUNTED
        self._max_bytes = max_bytes
        self._folded = 0

    @property
    def measured(self) -> bool:
        """Return whether the device keeps a peak count this meter reads."""
        return self._measured

    @property
    def max_bytes(self) -> int | None:
        """Return the bound one step is held to, or None where it is unbounded."""
        return self._max_bytes

    def _peak(self) -> int:
        """Return the device's peak count once its queued work has finished."""
        torch.accelerator.synchronize(self._device)

        return int(torch.accelerator.max_memory_allocated(self._device))

    def reset(self) -> None:
        """Start the point's count afresh."""
        self._folded = 0

        if self._measured:
            torch.accelerator.synchronize(self._device)
            torch.accelerator.reset_peak_memory_stats(self._device)

    @contextmanager
    def step(self) -> Iterator[None]:
        """Measure the training step the block runs.

        Raises
        ------
        ActivationMemoryExceededError
            If the device held more bytes at once during the block than the
            bound allows.
        """
        bound = self._max_bytes

        if bound is None or not self._measured:
            yield

            return

        self._folded = max(self._folded, self._peak())
        torch.accelerator.reset_peak_memory_stats(self._device)

        yield

        peak = self._peak()
        self._folded = max(self._folded, peak)

        if peak > bound:
            raise ActivationMemoryExceededError(
                str(self._device), peak_bytes=peak, max_bytes=bound
            )

    def high_water(self) -> int | None:
        """Return the most bytes the device held at once since the last reset.

        Returns
        -------
        int | None
            The peak over every step and the work between them, or None on a
            device that keeps no count.
        """
        if not self._measured:
            return None

        return max(self._folded, self._peak())

    @override
    def __repr__(self) -> str:
        """Return the device and bound this meter reads."""
        return f"StepMemory({str(self._device)!r}, max_bytes={self._max_bytes!r})"
