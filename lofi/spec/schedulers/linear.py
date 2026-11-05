"""Linear learning-rate scheduler declaration."""

from typing import Literal, NamedTuple

import didactic.api as dx

from lofi._checks import ensure_finite, ensure_non_negative
from lofi.spec.components.trainer import (
    _DEFAULT_WARMUP_START_FACTOR,
    _DEFAULT_WARMUP_STEPS,
    SchedulerSpec,
)
from lofi.spec.schedulers._checks import warmup_start_factor

__all__ = ["LinearSchedulerSettings", "LinearSchedulerSpec"]


class LinearSchedulerSettings(NamedTuple):
    """The settings a linear schedule is built from.

    Attributes
    ----------
    warmup_steps : int
        Number of optimizer steps spent warming the learning rate.
    warmup_start_factor : float
        The fraction of the learning rate the warmup starts from.
    final_factor : float
        The fraction of the declared learning rate reached at the last step.
    """

    warmup_steps: int
    warmup_start_factor: float
    final_factor: float


class LinearSchedulerSpec(SchedulerSpec):
    """Declares linear warmup followed by linear decay to a final fraction.

    Attributes
    ----------
    kind : Literal["linear"]
        Registry tag.
    warmup_steps : int
        Number of optimizer steps spent warming the learning rate.
    warmup_start_factor : float
        The fraction of the learning rate the warmup starts from, in ``(0, 1]``.
    final_factor : float
        The fraction of the declared learning rate the decay reaches at the
        last step of the horizon, in ``[0, 1]``.
    """

    kind: Literal["linear"] = "linear"
    warmup_steps: int = _DEFAULT_WARMUP_STEPS
    warmup_start_factor: float = _DEFAULT_WARMUP_START_FACTOR
    final_factor: float = 0.0

    @dx.validates("warmup_steps")
    @classmethod
    def _warmup_is_nonnegative(cls, value: int) -> int:
        """Validate the warmup's optimizer-step count."""
        return ensure_non_negative("warmup_steps", value)

    @dx.validates("warmup_start_factor")
    @classmethod
    def _start_is_a_fraction(cls, value: float) -> float:
        """Validate the warmup's starting fraction."""
        return warmup_start_factor(value)

    @dx.validates("final_factor")
    @classmethod
    def _final_factor_is_a_fraction(cls, value: float) -> float:
        """Validate the final learning-rate fraction."""
        checked = ensure_finite("final_factor", value)

        if not 0.0 <= checked <= 1.0:
            msg = f"final_factor lies outside [0, 1]: {value!r}"

            raise ValueError(msg)

        return checked

    def settings(self) -> LinearSchedulerSettings:
        """Return the settings the plugin builds this schedule from."""
        return LinearSchedulerSettings(
            warmup_steps=self.warmup_steps,
            warmup_start_factor=self.warmup_start_factor,
            final_factor=self.final_factor,
        )
