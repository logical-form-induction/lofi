"""Tests for the rank statistics over form frequencies."""

from math import exp, log

import pytest
from hypothesis import given
from hypothesis import strategies as st

from lofi.data.synthetic._frequency import average_ranks, log_log_slope, spearman

VALUES = st.lists(st.integers(min_value=0, max_value=5), min_size=1, max_size=12)


def test_average_ranks_are_one_based_and_shared_among_ties() -> None:
    assert average_ranks((10.0, 30.0, 20.0)) == (1.0, 3.0, 2.0)
    assert average_ranks((5.0, 5.0, 1.0)) == (2.5, 2.5, 1.0)
    assert average_ranks((2.0, 2.0, 2.0)) == (2.0, 2.0, 2.0)
    assert average_ranks(()) == ()


@given(VALUES)
def test_average_ranks_sum_to_the_triangular_number(values: list[int]) -> None:
    ranks = average_ranks([float(value) for value in values])

    assert sum(ranks) == pytest.approx(len(values) * (len(values) + 1) / 2)

    for left, right in zip(values, ranks, strict=True):
        for other_value, other_rank in zip(values, ranks, strict=True):
            assert (left < other_value) == (right < other_rank)


def test_spearman_is_one_on_a_monotone_pair_and_minus_one_when_reversed() -> None:
    xs = (1.0, 4.0, 9.0, 16.0)

    assert spearman(xs, (0.1, 0.2, 0.3, 4.0)) == pytest.approx(1.0)
    assert spearman(xs, (4.0, 3.0, 2.0, 1.0)) == pytest.approx(-1.0)
    assert spearman(xs, (2.0, 1.0, 4.0, 3.0)) == pytest.approx(0.6)


def test_spearman_treats_ties_by_their_average_rank() -> None:
    assert spearman((1.0, 2.0, 3.0, 4.0), (1.0, 1.0, 2.0, 2.0)) == pytest.approx(
        0.8944271909999159
    )


def test_spearman_is_undefined_without_variation_or_units() -> None:
    assert spearman((1.0, 2.0, 3.0), (5.0, 5.0, 5.0)) is None
    assert spearman((7.0, 7.0), (1.0, 2.0)) is None
    assert spearman((), ()) is None

    with pytest.raises(ValueError, match="3 values cannot be correlated with 2"):
        spearman((1.0, 2.0, 3.0), (1.0, 2.0))


@given(VALUES, VALUES)
def test_spearman_is_symmetric_and_bounded(xs: list[int], ys: list[int]) -> None:
    width = min(len(xs), len(ys))
    left = [float(value) for value in xs[:width]]
    right = [float(value) for value in ys[:width]]
    forward = spearman(left, right)

    assert forward == spearman(right, left)

    if forward is not None:
        assert -1.0 - 1e-9 <= forward <= 1.0 + 1e-9
        assert spearman(left, [-value for value in right]) == pytest.approx(-forward)


def test_the_log_log_slope_recovers_a_power_law_exactly() -> None:
    ranks = (1.0, 2.0, 3.0, 4.0, 5.0)

    assert log_log_slope(ranks, tuple(rank**-1.5 for rank in ranks)) == (
        pytest.approx(-1.5)
    )
    assert log_log_slope(ranks, tuple(exp(2.0 * log(rank)) for rank in ranks)) == (
        pytest.approx(2.0)
    )
    assert log_log_slope(ranks, (3.0,) * 5) == pytest.approx(0.0)


def test_the_log_log_slope_is_undefined_below_two_points_or_without_spread() -> None:
    assert log_log_slope((1.0,), (2.0,)) is None
    assert log_log_slope((), ()) is None
    assert log_log_slope((2.0, 2.0), (1.0, 3.0)) is None

    with pytest.raises(ValueError, match="2 values cannot be regressed on 1"):
        log_log_slope((1.0, 2.0), (1.0,))

    with pytest.raises(ValueError, match="regressor is not positive"):
        log_log_slope((0.0, 2.0), (1.0, 1.0))

    with pytest.raises(ValueError, match="response is not positive"):
        log_log_slope((1.0, 2.0), (1.0, -1.0))
