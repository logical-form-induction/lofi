import pytest

from lofi._stats import average_ranks, spearman


def test_distinct_values_take_the_one_based_ranks_of_their_order() -> None:
    assert average_ranks([30.0, 10.0, 20.0]) == (3.0, 1.0, 2.0)


def test_tied_values_share_the_mean_of_the_ranks_they_span() -> None:
    assert average_ranks([5.0, 5.0, 9.0]) == (1.5, 1.5, 3.0)
    assert average_ranks([7.0, 7.0, 7.0]) == (2.0, 2.0, 2.0)


def test_ranking_nothing_ranks_nothing() -> None:
    assert average_ranks([]) == ()


def test_a_monotone_relation_correlates_at_one_whatever_its_shape() -> None:
    xs = [1.0, 2.0, 3.0, 4.0]

    assert spearman(xs, [1.0, 8.0, 27.0, 64.0]) == pytest.approx(1.0)
    assert spearman(xs, [64.0, 27.0, 8.0, 1.0]) == pytest.approx(-1.0)


def test_an_unrelated_pairing_correlates_near_zero() -> None:
    assert spearman([1.0, 2.0, 3.0, 4.0], [3.0, 1.0, 4.0, 2.0]) == pytest.approx(0.0)


def test_a_measurement_without_variation_correlates_with_nothing() -> None:
    assert spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None
    assert spearman([], []) is None


def test_two_measurements_of_different_lengths_cannot_be_correlated() -> None:
    with pytest.raises(ValueError, match="cannot be correlated"):
        spearman([1.0, 2.0], [1.0])
