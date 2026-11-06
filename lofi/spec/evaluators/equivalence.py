"""Held-out logical form equivalence evaluator declaration."""

from typing import TYPE_CHECKING, Final, Literal, NamedTuple, override

import didactic.api as dx

from lofi._checks import ensure_name
from lofi.spec._ports import Dependency
from lofi.spec.components.run import EvaluatorSpec
from lofi.spec.evaluators.conventions import ConventionSettings, ConventionSpec
from lofi.spec.provers import ProofBudgetSpec

if TYPE_CHECKING:
    from lofi.lf.provers import ProofBudget

__all__ = [
    "EQUIVALENCE_ROWS",
    "EquivalenceEvaluatorSettings",
    "EquivalenceEvaluatorSpec",
]

EQUIVALENCE_ROWS: Final[tuple[str, ...]] = (
    "equivalence-equivalence",
    "equivalence-well-formed",
    "equivalence-timeout",
    "equivalence-proof-time",
    "equivalence-proof-calls",
    "equivalence-memo-hits",
)
"""The rows an equivalence evaluator reports, in report order."""


class EquivalenceEvaluatorSettings(NamedTuple):
    """Settings resolved from an equivalence evaluator declaration."""

    readout: str
    budget: ProofBudget
    conventions: ConventionSettings


class EquivalenceEvaluatorSpec(EvaluatorSpec):
    """Declare prover-decided equivalence of decoded and gold logical forms.

    Each held-out item's highest-weight root form is written in the run
    language's surface syntax, read back, type-checked, and put to the
    interpretation's shared proof service against the item's gold form. The
    rows report the equivalence, well-formed and timeout rates, the mean proof
    time of decided checks, and the uncached prover calls and memo hits the
    checks spent.

    Parameters
    ----------
    readout : str
        Complete term readout role that decodes logical forms.
    budget : ProofBudgetSpec
        The call, wall clock and CPU limits of the evaluation use site.
    conventions : ConventionSpec
        Conventions the resulting rows are reported under.
    """

    kind: Literal["equivalence"] = "equivalence"
    readout: str
    budget: ProofBudgetSpec
    conventions: ConventionSpec

    @dx.validates("readout")
    @classmethod
    def _readout_is_named(cls, value: str) -> str:
        """Return a nonempty complete term readout role."""
        return ensure_name("an equivalence term-readout role", value)

    @override
    def dependencies(self) -> tuple[Dependency, ...]:
        """Return the declared complete term readout dependency."""
        return (Dependency(self.readout, "terms-readout"),)

    @override
    def metrics(self, labels: tuple[str, ...] = (), /) -> tuple[str, ...]:
        """Return every row this declaration emits, in report order."""
        del labels
        return EQUIVALENCE_ROWS

    def settings(self) -> EquivalenceEvaluatorSettings:
        """Return the settings the evaluator plugin consumes."""
        return EquivalenceEvaluatorSettings(
            self.readout, self.budget.build(), self.conventions.settings()
        )
