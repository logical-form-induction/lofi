"""Declared resource bounds for one proof service use site."""

from typing import Self

import didactic.api as dx

from lofi._checks import ensure_finite
from lofi.lf.provers import ProofBudget

__all__ = ["ProofBudgetSpec"]


class ProofBudgetSpec(dx.Model, extra="forbid"):
    """The call, wall clock, and CPU limits for one proof service use site.

    Attributes
    ----------
    calls_per_step : int | None
        The most uncached calls dispatched during one step, or no limit.
    wall_per_step : float | None
        The wall clock limit of one step, in seconds, or no limit.
    cpu_seconds : float | None
        The cumulative CPU-time limit of this use site, or no limit.
    """

    calls_per_step: int | None = None
    wall_per_step: float | None = None
    cpu_seconds: float | None = None

    @dx.model_validator()
    def _seconds_are_finite(self) -> Self:
        """Return a declaration whose clocks are finite."""
        if self.calls_per_step is not None and self.calls_per_step < 0:
            msg = "proof calls per step are nonnegative"
            raise ValueError(msg)

        if self.wall_per_step is not None:
            wall = ensure_finite("proof wall seconds per step", self.wall_per_step)

            if wall < 0.0:
                msg = "proof wall seconds per step are nonnegative"
                raise ValueError(msg)

        if self.cpu_seconds is not None:
            cpu = ensure_finite("proof CPU seconds", self.cpu_seconds)

            if cpu < 0.0:
                msg = "proof CPU seconds are nonnegative"
                raise ValueError(msg)

        return self

    def build(self) -> ProofBudget:
        """Return the interior proof budget."""
        return ProofBudget(
            calls_per_step=self.calls_per_step,
            wall_per_step=self.wall_per_step,
            cpu_seconds=self.cpu_seconds,
        )
