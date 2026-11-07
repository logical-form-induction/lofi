"""Declared prover variants and proof service budgets."""

from lofi.spec.provers._base import (
    LabelingProverSpec,
    PolymorphismPolicy,
    ProverServiceSettings,
    ProverSpec,
)
from lofi.spec.provers._budget import ProofBudgetSpec
from lofi.spec.provers.smt import SmtProverSettings, SmtProverSpec
from lofi.spec.provers.tptp import TptpProverSettings, TptpProverSpec
from lofi.spec.provers.truth_table import (
    TruthTableProverSettings,
    TruthTableProverSpec,
)

__all__ = [
    "LabelingProverSpec",
    "PolymorphismPolicy",
    "ProofBudgetSpec",
    "ProverServiceSettings",
    "ProverSpec",
    "SmtProverSettings",
    "SmtProverSpec",
    "TptpProverSettings",
    "TptpProverSpec",
    "TruthTableProverSettings",
    "TruthTableProverSpec",
]
