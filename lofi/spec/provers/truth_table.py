"""Finite propositional truth-table prover declaration."""

from typing import Literal, NamedTuple

import didactic.api as dx

from lofi.spec.provers._base import (
    ProverServiceSettings,
    ProverSpec,
)

__all__ = ["TruthTableProverSettings", "TruthTableProverSpec"]


class TruthTableProverSettings(NamedTuple):
    """Resolved settings of a finite truth-table prover."""

    atoms: int
    service: ProverServiceSettings


class TruthTableProverSpec(ProverSpec, extra="forbid"):
    """Declare exhaustive propositional entailment with an atom cap."""

    kind: Literal["truth-table"] = "truth-table"
    atoms: int = 12

    __axioms__ = (dx.axiom("atoms > 0", message="a truth-table atom cap is positive"),)

    def settings(self) -> TruthTableProverSettings:
        """Return the resolved atom cap."""
        return TruthTableProverSettings(
            self.atoms,
            self.service_settings(),
        )
