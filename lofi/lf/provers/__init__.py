"""Optional typed entailment oracles."""

from lofi.lf.provers._capability import (
    FIRST_ORDER_CAPABILITY,
    HIGHER_ORDER_CAPABILITY,
    QUANTIFIER_FREE_CAPABILITY,
    TranslatorCapability,
    joint_capability,
)
from lofi.lf.provers._service import (
    ProofBudget,
    ProofService,
    ProofSite,
    Questions,
    ServiceReport,
)
from lofi.lf.provers._translate import (
    TPTP_DIALECT_CAPABILITIES,
    FirstOrderTranslationError,
    PolymorphismPolicy,
    TptpDialect,
    tptp_problem,
)
from lofi.lf.provers.smt import CVC5, Z3, SmtEngine, SmtOracle
from lofi.lf.provers.tptp import EPROVER, LEO3, VAMPIRE, TptpEngine, TptpOracle

__all__ = [
    "CVC5",
    "EPROVER",
    "FIRST_ORDER_CAPABILITY",
    "HIGHER_ORDER_CAPABILITY",
    "LEO3",
    "QUANTIFIER_FREE_CAPABILITY",
    "TPTP_DIALECT_CAPABILITIES",
    "VAMPIRE",
    "Z3",
    "FirstOrderTranslationError",
    "PolymorphismPolicy",
    "ProofBudget",
    "ProofService",
    "ProofSite",
    "Questions",
    "ServiceReport",
    "SmtEngine",
    "SmtOracle",
    "TptpDialect",
    "TptpEngine",
    "TptpOracle",
    "TranslatorCapability",
    "joint_capability",
    "tptp_problem",
]
