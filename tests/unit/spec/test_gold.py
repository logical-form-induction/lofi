"""Tests for repeated-response aggregation declarations."""

from lofi.spec.gold import MajorityGoldSettings, MajorityGoldSpec


def test_majority_aggregation_has_one_parameter_free_setting() -> None:
    """Publish one current parameter-free majority treatment."""
    assert MajorityGoldSpec().settings() == MajorityGoldSettings()
