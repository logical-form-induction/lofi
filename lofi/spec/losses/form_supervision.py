"""Gold logical form supervision declaration."""

from typing import Literal, NamedTuple, override

import didactic.api as dx

from lofi._checks import ensure_name
from lofi.spec._ports import Dependency
from lofi.spec.components.data import TrainingBatchKind
from lofi.spec.components.loss import LossSpec

__all__ = ["FormSupervisionLossSettings", "FormSupervisionLossSpec"]


class FormSupervisionLossSettings(NamedTuple):
    """Settings for direct gold-form supervision."""

    readout: str


class FormSupervisionLossSpec(LossSpec):
    """Declare negative log likelihood of each retained gold form."""

    kind: Literal["form-supervision"] = "form-supervision"
    readout: str

    @dx.validates("readout")
    @classmethod
    def _readout_is_named(cls, value: str) -> str:
        """Return a nonempty complete term readout role."""
        return ensure_name("a form-supervision readout role", value)

    @override
    def dependencies(self) -> tuple[Dependency, ...]:
        """Return the complete term readout dependency."""
        return (Dependency(self.readout, "form-supervision-readout"),)

    def settings(self) -> FormSupervisionLossSettings:
        """Return the settings the objective is built from."""
        return FormSupervisionLossSettings(self.readout)

    @override
    def required_training_batches(self) -> frozenset[TrainingBatchKind]:
        """Return that direct supervision consumes certified gold forms."""
        return frozenset({TrainingBatchKind.GOLD_FORMS})
