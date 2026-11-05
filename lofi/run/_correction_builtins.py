"""Built-in experiment correction plugins."""

from lofi.run._boundary.experiment import (
    BenjaminiHochbergSpec,
    BenjaminiYekutieliSpec,
    BonferroniSpec,
    HolmSpec,
)
from lofi.run._correction import CorrectionPlugin
from lofi.stats import (
    BenjaminiHochberg,
    BenjaminiYekutieli,
    Bonferroni,
    Correction,
    Holm,
)

__all__ = [
    "BENJAMINI_HOCHBERG_PLUGIN",
    "BENJAMINI_YEKUTIELI_PLUGIN",
    "BONFERRONI_PLUGIN",
    "HOLM_PLUGIN",
]


def _bonferroni(declaration: BonferroniSpec, /) -> Correction:
    del declaration
    return Bonferroni()


def _holm(declaration: HolmSpec, /) -> Correction:
    del declaration
    return Holm()


def _benjamini_hochberg(declaration: BenjaminiHochbergSpec, /) -> Correction:
    del declaration
    return BenjaminiHochberg()


def _benjamini_yekutieli(declaration: BenjaminiYekutieliSpec, /) -> Correction:
    del declaration
    return BenjaminiYekutieli()


BONFERRONI_PLUGIN = CorrectionPlugin(spec=BonferroniSpec, build=_bonferroni)
HOLM_PLUGIN = CorrectionPlugin(spec=HolmSpec, build=_holm)
BENJAMINI_HOCHBERG_PLUGIN = CorrectionPlugin(
    spec=BenjaminiHochbergSpec,
    build=_benjamini_hochberg,
)
BENJAMINI_YEKUTIELI_PLUGIN = CorrectionPlugin(
    spec=BenjaminiYekutieliSpec,
    build=_benjamini_yekutieli,
)
