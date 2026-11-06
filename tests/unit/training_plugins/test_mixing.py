"""Tests for installed task-mixing builders."""

import pytest

from lofi.mixing import (
    Alternation,
    ProportionalMixingSpec,
    RatioMixingSpec,
    RoundRobinMixingSpec,
    TemperatureMixingSpec,
)
from lofi.training_plugins.mixing import (
    PROPORTIONAL_PLUGIN,
    RATIO_PLUGIN,
    ROUND_ROBIN_PLUGIN,
    TEMPERATURE_PLUGIN,
    mixing,
)


def test_each_mixing_plugin_builds_its_exact_runtime_policy() -> None:
    round_robin = ROUND_ROBIN_PLUGIN.construct(RoundRobinMixingSpec())
    proportional = PROPORTIONAL_PLUGIN.construct(ProportionalMixingSpec())
    ratio = RATIO_PLUGIN.construct(RatioMixingSpec(weights=(2, 1)))
    temperature = TEMPERATURE_PLUGIN.construct(TemperatureMixingSpec(temperature=2.0))

    assert round_robin.alternation is Alternation.ROUND_ROBIN
    assert proportional.alternation is Alternation.PROPORTIONAL
    assert ratio.declared == (2, 1)
    assert temperature.temperature_value == pytest.approx(2.0)


def test_an_installed_mixing_declaration_selects_its_unique_builder() -> None:
    assert mixing(RatioMixingSpec(weights=(3, 2))).declared == (3, 2)
