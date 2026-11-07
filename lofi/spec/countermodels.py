"""Neutral finite-countermodel search declarations."""

from typing import Literal, Self

import didactic.api as dx

from lofi.lf.models import CountermodelSettings

__all__ = ["CountermodelSpec"]


class CountermodelSpec(dx.Model, extra="forbid"):
    """Declare a bounded crisp search over complete finite frames."""

    method: Literal["random", "exhaustive"]
    carriers: dict[str, int]
    structures: int
    seed: int
    full_function_space_cap: int

    @dx.model_validator()
    def _settings_are_bounded(self) -> Self:
        """Validate finite-frame resources once at the declaration boundary."""
        if not self.carriers:
            msg = "a countermodel search declares at least one atomic carrier"
            raise ValueError(msg)

        if any(not name or size < 1 for name, size in self.carriers.items()):
            msg = "every named countermodel carrier has positive size"
            raise ValueError(msg)

        if self.structures < 1:
            msg = "a countermodel search examines at least one structure"
            raise ValueError(msg)

        if self.seed < 0:
            msg = "a countermodel seed is nonnegative"
            raise ValueError(msg)

        if self.full_function_space_cap < 1:
            msg = "a countermodel full-function-space cap is positive"
            raise ValueError(msg)

        return self

    def settings(self) -> CountermodelSettings:
        """Return immutable interior search settings."""
        return CountermodelSettings(
            self.method,
            carriers=self.carriers,
            structures=self.structures,
            seed=self.seed,
            full_function_space_cap=self.full_function_space_cap,
        )
