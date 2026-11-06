"""Logical form typing loss declaration."""

from typing import Literal, NamedTuple, override

import didactic.api as dx

from lofi._checks import ensure_name
from lofi.spec._ports import Dependency
from lofi.spec.components.loss import LossSpec

__all__ = ["FormTypingLossSettings", "FormTypingLossSpec"]


class FormTypingLossSettings(NamedTuple):
    """Settings for logical form typing supervision."""

    readout: str


class FormTypingLossSpec(LossSpec):
    """Declare binary typing supervision over one complete term readout."""

    kind: Literal["form-typing"] = "form-typing"
    readout: str

    @dx.validates("readout")
    @classmethod
    def _readout_is_named(cls, value: str) -> str:
        """Return a nonempty term-readout role."""
        return ensure_name("a form-typing readout role", value)

    @override
    def dependencies(self) -> tuple[Dependency, ...]:
        """Return the complete term readout dependency."""
        return (Dependency(self.readout, "form-typing-readout"),)

    def settings(self) -> FormTypingLossSettings:
        """Return the settings the objective is built from."""
        return FormTypingLossSettings(readout=self.readout)
