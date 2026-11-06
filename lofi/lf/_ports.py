"""Logical form layer protocols."""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi.lf._profile import ProverBound
    from lofi.lf._verdict import Verdict
    from lofi.terms import Term

__all__ = ["EntailmentOracle"]


@runtime_checkable
class EntailmentOracle(Protocol):
    """A decision procedure for entailment between closed logical forms.

    A verdict is a function of the two forms, the theory and the per-call
    timeout alone. Budgeting, memoization and process pooling belong to the
    service that dispatches questions to the oracle.
    """

    @property
    def bound(self) -> ProverBound:
        """Return the profiles this oracle serves."""
        ...

    def entails(
        self,
        premise: Term,
        hypothesis: Term,
        /,
        *,
        theory: Sequence[Term] = (),
        timeout: float,
    ) -> Verdict:
        """Decide whether a premise, under a theory, entails a hypothesis.

        Parameters
        ----------
        premise : Term
            The closed formula assumed.
        hypothesis : Term
            The closed formula to be decided.
        theory : Sequence[Term]
            Further closed formulas assumed alongside the premise, in addition
            to the meaning postulates the profile declares.
        timeout : float
            The wall clock in seconds one decision may take before the oracle
            abstains.
        """
        ...
