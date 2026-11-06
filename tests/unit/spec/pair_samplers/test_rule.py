"""Tests for the rule-instance sampler declaration."""

import didactic.api as dx
import pytest

from lofi.spec.components.trainer import PairSamplerSpec
from lofi.spec.pair_samplers.rule import (
    RulePairSamplerSettings,
    RulePairSamplerSpec,
)


def test_the_rule_tag_reaches_its_variant_through_the_family_root() -> None:
    """Resolve the installed declaration through its tagged-union root."""
    declared = PairSamplerSpec.model_validate({"kind": "rule"})

    assert type(declared) is RulePairSamplerSpec
    assert declared == RulePairSamplerSpec()
    assert declared.settings() == RulePairSamplerSettings()
    assert PairSamplerSpec.__variants__["rule"] is RulePairSamplerSpec


def test_an_extra_rule_sampler_field_is_refused() -> None:
    """Keep every configurable choice on an actual declaration field."""
    with pytest.raises(dx.ValidationError, match="replacement"):
        PairSamplerSpec.model_validate({"kind": "rule", "replacement": False})


def test_the_rule_variant_round_trips_through_json() -> None:
    """Preserve the one current declaration exactly."""
    document = '{"kind": "rule", "declared_output": null}'
    declared = PairSamplerSpec.model_validate_json(document)

    assert declared.model_dump_json() == document
    assert PairSamplerSpec.model_validate_json(declared.model_dump_json()) == declared
