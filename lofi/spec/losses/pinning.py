"""Span-pinning loss declaration."""

from typing import Literal, NamedTuple, override

import didactic.api as dx

from lofi._checks import ensure_name
from lofi.spec._ports import Dependency
from lofi.spec.components.loss import LossSpec

__all__ = ["PinningLossSettings", "PinningLossSpec"]


class PinningLossSettings(NamedTuple):
    """The settings a pinning objective is built from.

    Attributes
    ----------
    readout : str | None
        The term-readout role whose target sets the objective pins spans to, or None
        for negative bracket supervision directly from parser marginals.
    """

    readout: str | None


class PinningLossSpec(LossSpec):
    """Declare target pinning or negative bracket supervision."""

    kind: Literal["pinning"] = "pinning"
    readout: str | None = None

    @dx.validates("readout")
    @classmethod
    def _readout_is_named(cls, value: str | None) -> str | None:
        """Return a nonempty optional term-readout role."""
        return None if value is None else ensure_name("a pinning readout role", value)

    @override
    def dependencies(self) -> tuple[Dependency, ...]:
        """Return the target readout dependency, absent for bracket supervision."""
        if self.readout is None:
            return ()

        return (Dependency(self.readout, "constraint-readout"),)

    def settings(self) -> PinningLossSettings:
        """Return the settings the plugin builds this objective from."""
        return PinningLossSettings(readout=self.readout)

    def required_chart_outputs(self) -> frozenset[str]:
        """Return the posterior chart quantities pinning depends on."""
        return frozenset({"alpha", "mu"})

    def constraint_route(self) -> Literal["target", "negative-bracket"]:
        """Return the selected-constraint representation this objective consumes."""
        return "negative-bracket" if self.readout is None else "target"
