import pytest

from lofi.judgment.agreement import (
    fisher_interval,
    rank_correlation,
    spearman_brown,
    split_half_ceiling,
)


def test_a_monotone_pairing_correlates_perfectly_in_either_direction() -> None:
    rising = [1.0, 2.0, 3.0, 4.0, 5.0]

    assert rank_correlation(rising, [2.0, 4.0, 8.0, 16.0, 32.0]) == pytest.approx(1.0)
    assert rank_correlation(rising, list(reversed(rising))) == pytest.approx(-1.0)


def test_ties_take_their_mean_rank() -> None:
    correlation = rank_correlation([1.0, 1.0, 2.0, 3.0], [1.0, 2.0, 2.0, 3.0])

    assert correlation == pytest.approx(5.0 / 6.0)


def test_the_correlation_reads_ranks_and_not_values() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    stretched = [1.0, 2.0, 3.0, 400.0]
    other = [4.0, 1.0, 3.0, 2.0]

    assert rank_correlation(values, other) == pytest.approx(
        rank_correlation(stretched, other)
    )


def test_two_sides_of_different_lengths_are_refused() -> None:
    with pytest.raises(ValueError, match="two sides of one length"):
        rank_correlation([1.0, 2.0, 3.0], [1.0, 2.0])


def test_too_few_pairs_are_refused() -> None:
    with pytest.raises(ValueError, match="at least 3 pairs"):
        rank_correlation([1.0, 2.0], [1.0, 2.0])


def test_a_side_that_never_varies_is_refused() -> None:
    with pytest.raises(ValueError, match="never varies"):
        rank_correlation([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])


def test_lengthening_leaves_a_measurement_of_the_same_length_alone() -> None:
    assert spearman_brown(0.4, 1.0) == pytest.approx(0.4)


def test_doubling_a_half_length_correlation_raises_it() -> None:
    doubled = spearman_brown(0.5, 2.0)

    assert doubled == pytest.approx(2.0 / 3.0)
    assert doubled > 0.5


def test_lengthening_fixes_the_endpoints_of_the_scale() -> None:
    assert spearman_brown(1.0, 5.0) == pytest.approx(1.0)
    assert spearman_brown(0.0, 5.0) == pytest.approx(0.0)


def test_a_factor_that_is_not_positive_is_refused() -> None:
    with pytest.raises(ValueError, match="not positive"):
        spearman_brown(0.5, 0.0)


def test_lengthening_through_the_pole_is_refused() -> None:
    with pytest.raises(ValueError, match="undefined at a correlation"):
        spearman_brown(-0.5, 3.0)


def test_the_fisher_interval_brackets_the_correlation_asymmetrically() -> None:
    lower, upper = fisher_interval(0.5, 103)

    assert lower == pytest.approx(0.3392, abs=5e-4)
    assert upper == pytest.approx(0.6323, abs=5e-4)
    assert lower < 0.5 < upper
    assert 0.5 - lower > upper - 0.5


def test_the_fisher_interval_narrows_as_observations_accumulate() -> None:
    near_lower, near_upper = fisher_interval(0.5, 20)
    far_lower, far_upper = fisher_interval(0.5, 2000)

    assert far_upper - far_lower < near_upper - near_lower
    assert near_lower < far_lower
    assert far_upper < near_upper


def test_a_wider_coverage_gives_a_wider_interval() -> None:
    narrow_lower, narrow_upper = fisher_interval(0.5, 103, level=0.5)
    wide_lower, wide_upper = fisher_interval(0.5, 103, level=0.99)

    assert wide_upper - wide_lower > narrow_upper - narrow_lower


def test_a_correlation_at_the_edge_of_its_range_has_no_fisher_interval() -> None:
    with pytest.raises(ValueError, match="undefined at a correlation"):
        fisher_interval(1.0, 103)


def test_too_few_observations_carry_no_fisher_interval() -> None:
    with pytest.raises(ValueError, match="at least 4 observations"):
        fisher_interval(0.5, 3)


@pytest.mark.parametrize("level", [0.0, 1.0, -0.5, 2.0])
def test_a_coverage_that_is_not_a_proportion_is_refused(level: float) -> None:
    with pytest.raises(ValueError, match="not a proportion"):
        fisher_interval(0.5, 103, level=level)


def paired_items(count: int, offset: int) -> list[list[float]]:
    return [
        [
            float(index % 7),
            float((index + offset) % 7) if index % 3 == 0 else float(index % 7),
        ]
        for index in range(count)
    ]


def test_a_pair_of_responses_per_item_is_the_correlation_between_the_two() -> None:
    items = paired_items(60, 2)
    ceiling = split_half_ceiling(items)
    direct = rank_correlation([item[0] for item in items], [item[1] for item in items])

    assert ceiling.value == pytest.approx(spearman_brown(direct, 2.0))
    assert ceiling.value > direct


def test_the_ceiling_counts_the_items_and_the_responses_it_read() -> None:
    ceiling = split_half_ceiling([[1.0, 2.0, 3.0, 4.0, 5.0]] * 4 + paired_items(40, 3))

    assert ceiling.items == 44
    assert ceiling.responses == 4 * 5 + 40 * 2


def test_the_interval_brackets_the_ceiling() -> None:
    ceiling = split_half_ceiling(paired_items(80, 3))
    lower, upper = ceiling.interval

    assert lower <= ceiling.value <= upper
    assert lower < upper


def test_annotators_who_agree_more_reach_a_higher_ceiling() -> None:
    agreeing = [
        [float(index % 7), float(index % 7), float((index + 1) % 7)]
        for index in range(90)
    ]
    disagreeing = [
        [float(index % 7), float((index * 5) % 7), float((index * 3) % 7)]
        for index in range(90)
    ]

    assert split_half_ceiling(agreeing).value > split_half_ceiling(disagreeing).value


def test_the_estimate_does_not_move_between_runs() -> None:
    items = paired_items(70, 4)

    assert split_half_ceiling(items).value == split_half_ceiling(items).value


def test_averaging_over_more_splits_uses_more_of_them() -> None:
    items = [
        [float(index % 7), float((index + 1) % 5), float(index % 3), float(index % 2)]
        for index in range(90)
    ]

    assert (
        split_half_ceiling(items, splits=1).value
        != split_half_ceiling(items, splits=6).value
    )


def test_averaging_over_no_splits_is_refused() -> None:
    with pytest.raises(ValueError, match="no splits"):
        split_half_ceiling(paired_items(40, 2), splits=0)


def test_an_item_that_cannot_be_split_is_refused() -> None:
    with pytest.raises(ValueError, match="fewer than 2 responses"):
        split_half_ceiling([*paired_items(40, 2), [3.0]])


def test_annotators_who_agree_perfectly_leave_the_interval_undefined() -> None:
    with pytest.raises(ValueError, match="undefined at a correlation"):
        split_half_ceiling([[float(index)] * 4 for index in range(40)])


def test_an_item_with_many_annotators_is_split_without_enumerating_them_all() -> None:
    wide = [
        [float((index * position) % 7) for position in range(1, 31)]
        for index in range(40)
    ]
    ceiling = split_half_ceiling(wide, splits=4)

    assert ceiling.items == 40
    assert ceiling.responses == 40 * 30
    assert -1.0 <= ceiling.value <= 1.0
