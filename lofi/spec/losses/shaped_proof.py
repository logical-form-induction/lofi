"""Shaped proof-credit declaration for sampled logical forms."""

from typing import TYPE_CHECKING, Literal, NamedTuple, Self, override

import didactic.api as dx

from lofi._checks import ensure_finite, ensure_name
from lofi.spec._ports import Dependency
from lofi.spec.components.data import TrainingBatchKind
from lofi.spec.components.loss import LossSpec
from lofi.spec.provers import ProofBudgetSpec

if TYPE_CHECKING:
    from lofi.lf.provers import ProofBudget

__all__ = [
    "ProofShapingSettings",
    "ProofShapingSpec",
    "ShapedProofLossSettings",
    "ShapedProofLossSpec",
]


class ProofShapingSettings(NamedTuple):
    """Interior credits and length cost for shaped proof supervision."""

    ill_formed: float
    well_formed: float
    type_checks: float
    proved: float
    length: float


class ProofShapingSpec(dx.Model, extra="forbid"):
    """Declare the payment for each successively stronger form outcome."""

    ill_formed: float
    well_formed: float
    type_checks: float
    proved: float
    length: float

    @dx.model_validator()
    def _credits_are_finite_and_laddered(self) -> Self:
        """Validate finite laddered credits and a nonnegative length cost."""
        earned = (
            ensure_finite("the ill-formed credit", self.ill_formed),
            ensure_finite("the well-formed credit", self.well_formed),
            ensure_finite("the type-checking credit", self.type_checks),
            ensure_finite("the proved credit", self.proved),
        )
        length = ensure_finite("the shaped-proof length cost", self.length)

        if tuple(sorted(earned)) != earned:
            msg = "shaped-proof credits do not decrease as a form reaches more"
            raise ValueError(msg)

        if length < 0.0:
            msg = f"the shaped-proof length cost is nonnegative, not {length!r}"
            raise ValueError(msg)

        return self

    def settings(self) -> ProofShapingSettings:
        """Return the interior shaping values."""
        return ProofShapingSettings(
            self.ill_formed,
            self.well_formed,
            self.type_checks,
            self.proved,
            self.length,
        )


class ShapedProofLossSettings(NamedTuple):
    """Interior settings for one shaped-proof objective."""

    readout: str
    shaping: ProofShapingSettings
    budget: ProofBudget


class ShapedProofLossSpec(LossSpec):
    """Declare proof-shaped credit over decoded forms and their gold forms."""

    kind: Literal["shaped-proof"] = "shaped-proof"
    readout: str
    shaping: ProofShapingSpec
    budget: ProofBudgetSpec

    @dx.validates("budget")
    @classmethod
    def _budget_is_a_proof_budget(cls, value: ProofBudgetSpec) -> ProofBudgetSpec:
        """Return the proof-service budget declaration."""
        if not isinstance(value, ProofBudgetSpec):
            msg = f"a shaped-proof budget is a ProofBudgetSpec, not {value!r}"
            raise TypeError(msg)

        return value

    @dx.validates("readout")
    @classmethod
    def _readout_is_named(cls, value: str) -> str:
        """Return a nonempty decoded-form readout role."""
        return ensure_name("a shaped-proof readout role", value)

    @override
    def dependencies(self) -> tuple[Dependency, ...]:
        """Return the decoded-form readout dependency."""
        return (Dependency(self.readout, "shaped-proof-readout"),)

    def settings(self) -> ShapedProofLossSettings:
        """Return the settings the objective is built from."""
        return ShapedProofLossSettings(
            self.readout,
            self.shaping.settings(),
            self.budget.build(),
        )

    @override
    def required_training_batches(self) -> frozenset[TrainingBatchKind]:
        """Return that proof shaping consumes certified gold forms."""
        return frozenset({TrainingBatchKind.GOLD_FORMS})
