"""Statistical tests, corrections, and result types."""

from lofi.stats._bootstrap import Bootstrap, CrossedBootstrap, paired_bootstrap
from lofi.stats._corrections import (
    BenjaminiHochberg,
    BenjaminiYekutieli,
    Bonferroni,
    Holm,
    false_discovery_correction,
)
from lofi.stats._effects import (
    ESTIMATORS,
    EffectSize,
    Estimator,
    paired_cliffs_delta,
    paired_cohens_d,
    paired_effect_size,
)
from lofi.stats._equivalence import (
    Equivalence,
    EquivalenceResult,
    MarginTest,
    Verdict,
)
from lofi.stats._errors import DependenceError
from lofi.stats._nulls import BootstrapMaxima, LabelPermutation, MatchedDraw
from lofi.stats._permutation import paired_permutation, paired_rank_inversion
from lofi.stats._ports import Correction, NullModel, Statistic
from lofi.stats._values import (
    NULL_FAMILIES,
    TAILS,
    BootstrapResult,
    Decisions,
    Dependence,
    Interval,
    NullFamily,
    NullProvenance,
    PValue,
    Tail,
    p_value_against_replicates,
)

__all__ = [
    "ESTIMATORS",
    "NULL_FAMILIES",
    "TAILS",
    "BenjaminiHochberg",
    "BenjaminiYekutieli",
    "Bonferroni",
    "Bootstrap",
    "BootstrapMaxima",
    "BootstrapResult",
    "Correction",
    "CrossedBootstrap",
    "Decisions",
    "Dependence",
    "DependenceError",
    "EffectSize",
    "Equivalence",
    "EquivalenceResult",
    "Estimator",
    "Holm",
    "Interval",
    "LabelPermutation",
    "MarginTest",
    "MatchedDraw",
    "NullFamily",
    "NullModel",
    "NullProvenance",
    "PValue",
    "Statistic",
    "Tail",
    "Verdict",
    "false_discovery_correction",
    "p_value_against_replicates",
    "paired_bootstrap",
    "paired_cliffs_delta",
    "paired_cohens_d",
    "paired_effect_size",
    "paired_permutation",
    "paired_rank_inversion",
]
