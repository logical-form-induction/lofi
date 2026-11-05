"""Tests for the validation shared by scheduler declarations."""

import math

import pytest

from lofi.spec.schedulers._checks import warmup_start_factor


@pytest.mark.parametrize("value", [1e-8, 0.5, 1.0])
def test_a_fraction_in_the_half_open_unit_interval_is_returned_unchanged(
    value: float,
) -> None:
    assert warmup_start_factor(value) == value


@pytest.mark.parametrize("value", [0.0, -0.5, 1.5, math.inf, math.nan])
def test_a_fraction_outside_the_half_open_unit_interval_is_refused(
    value: float,
) -> None:
    with pytest.raises(ValueError, match="warmup_start_factor"):
        warmup_start_factor(value)
