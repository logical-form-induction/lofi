"""Tests for the held-out participant prediction rules."""

import pytest

from lofi.spec import HeldOutParticipantPolicy


def test_each_rule_is_named_by_its_declared_spelling() -> None:
    assert [policy.value for policy in HeldOutParticipantPolicy] == [
        "marginal",
        "plug_in",
    ]


def test_an_undeclared_rule_is_refused() -> None:
    with pytest.raises(ValueError, match="posterior"):
        HeldOutParticipantPolicy("posterior")
