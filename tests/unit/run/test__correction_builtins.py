"""Tests for the built-in multiple-comparison correction plugins."""

from lofi.run import (
    BenjaminiHochbergSpec,
    BenjaminiYekutieliSpec,
    BonferroniSpec,
    HolmSpec,
)
from lofi.run._correction_builtins import (
    BENJAMINI_HOCHBERG_PLUGIN,
    BENJAMINI_YEKUTIELI_PLUGIN,
    BONFERRONI_PLUGIN,
    HOLM_PLUGIN,
)
from lofi.stats import BenjaminiHochberg, BenjaminiYekutieli, Bonferroni, Holm


def test_each_builtin_plugin_constructs_its_declared_correction() -> None:
    assert BONFERRONI_PLUGIN.spec is BonferroniSpec
    assert isinstance(BONFERRONI_PLUGIN.construct(BonferroniSpec()), Bonferroni)
    assert HOLM_PLUGIN.spec is HolmSpec
    assert isinstance(HOLM_PLUGIN.construct(HolmSpec()), Holm)
    assert BENJAMINI_HOCHBERG_PLUGIN.spec is BenjaminiHochbergSpec
    assert isinstance(
        BENJAMINI_HOCHBERG_PLUGIN.construct(BenjaminiHochbergSpec()),
        BenjaminiHochberg,
    )
    assert BENJAMINI_YEKUTIELI_PLUGIN.spec is BenjaminiYekutieliSpec
    assert isinstance(
        BENJAMINI_YEKUTIELI_PLUGIN.construct(BenjaminiYekutieliSpec()),
        BenjaminiYekutieli,
    )
