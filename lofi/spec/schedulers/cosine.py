"""Cosine learning-rate scheduler declaration."""

from typing import Literal, NamedTuple

import didactic.api as dx

from lofi._checks import ensure_non_negative
from lofi.spec.components.trainer import (
    _DEFAULT_WARMUP_START_FACTOR,
    _DEFAULT_WARMUP_STEPS,
    SchedulerSpec,
)
from lofi.spec.schedulers._checks import warmup_start_factor

__all__ = ["CosineSchedulerSettings", "CosineSchedulerSpec"]


class CosineSchedulerSettings(NamedTuple):
    """The settings a cosine schedule is built from.

    Attributes
    ----------
    warmup_steps : int
        Number of optimizer steps spent warming the learning rate.
    warmup_start_factor : float
        The fraction of the learning rate the warmup starts from.
    """

    warmup_steps: int
    warmup_start_factor: float


class CosineSchedulerSpec(SchedulerSpec):
    """Declares linear warmup followed by cosine annealing.

    Attributes
    ----------
    kind : Literal["cosine"]
        Registry tag.
    warmup_steps : int
        Number of optimizer steps spent warming the learning rate.
    warmup_start_factor : float
        The fraction of the learning rate the warmup starts from, in $(0, 1]$.
    """

    kind: Literal["cosine"] = "cosine"
    warmup_steps: int = _DEFAULT_WARMUP_STEPS
    warmup_start_factor: float = _DEFAULT_WARMUP_START_FACTOR

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

    def settings(self) -> CosineSchedulerSettings:
        """Return the settings the plugin builds this schedule from."""
        return CosineSchedulerSettings(
            warmup_steps=self.warmup_steps,
            warmup_start_factor=self.warmup_start_factor,
        )
