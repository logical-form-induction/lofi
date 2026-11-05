"""Tests for the validation shared by optimizer declarations."""

import math

import pytest

from lofi.spec.optimizers._checks import moment_decays, nonnegative, positive


def test_a_positive_finite_value_is_returned_unchanged() -> None:
    assert positive("learning_rate", 0.5) == 0.5
    assert nonnegative("weight_decay", 0.0) == 0.0
    assert moment_decays((0.0, 0.99)) == (0.0, 0.99)


@pytest.mark.parametrize("value", [0.0, -1.0, math.inf, math.nan])
def test_a_nonpositive_or_nonfinite_value_is_refused(value: float) -> None:
    with pytest.raises(ValueError, match="learning_rate"):
        positive("learning_rate", value)


@pytest.mark.parametrize("value", [-1e-9, math.inf, math.nan])
def test_a_negative_or_nonfinite_value_is_refused(value: float) -> None:
    with pytest.raises(ValueError, match="weight_decay"):
        nonnegative("weight_decay", value)


@pytest.mark.parametrize(
    "values", [(0.9,), (0.9, 0.99, 0.999), (1.0, 0.9), (0.9, -0.1)]
)
def test_decays_other_than_two_rates_in_the_unit_interval_are_refused(
    values: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError, match="betas"):
        moment_decays(values)
