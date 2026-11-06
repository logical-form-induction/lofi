"""Certified pair scoring through the shared proof service."""

from typing import TYPE_CHECKING, Literal, NamedTuple, override

from lofi.spec.components.model import PairScorerSpec
from lofi.spec.countermodels import CountermodelSpec as _CountermodelSpec
from lofi.spec.provers import ProofBudgetSpec

if TYPE_CHECKING:
    from lofi.lf.models import CountermodelSettings
    from lofi.lf.provers import ProofBudget

__all__ = [
    "ProverScorerSettings",
    "ProverScorerSpec",
]


class ProverScorerSettings(NamedTuple):
    """Resolved use-site budget of a prover scorer."""

    budget: ProofBudget
    countermodels: CountermodelSettings | None


class ProverScorerSpec(PairScorerSpec):
    """Declare certified pair evidence from the language's prover."""

    kind: Literal["prover"] = "prover"
    budget: ProofBudgetSpec
    countermodels: _CountermodelSpec | None = None

    @override
    def checkpointed(self) -> bool:
        """Return false because the proof service carries no learned state."""
        return False

    def settings(self) -> ProverScorerSettings:
        """Return the interior proof service budget."""
        if not isinstance(self.budget, ProofBudgetSpec):
            msg = f"a prover scorer budget is a ProofBudgetSpec, not {self.budget!r}"
            raise TypeError(msg)

        return ProverScorerSettings(
            self.budget.build(),
            None if self.countermodels is None else self.countermodels.settings(),
        )
