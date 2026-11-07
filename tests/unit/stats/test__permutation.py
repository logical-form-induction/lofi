from math import fsum
from statistics import NormalDist
from typing import cast

import pytest

from lofi.stats import PValue, paired_permutation, paired_rank_inversion

# chi-square(9) 0.999 quantile
CHI_SQUARE_AT_ONE_IN_A_THOUSAND = 27.877

BINS = 10

STANDARD = NormalDist()


def shifted(center: float) -> NormalDist:
    return NormalDist(center, 1.0)


def monotone(count: int, *, seed: int) -> tuple[list[float], list[float]]:
    truth = sorted(STANDARD.samples(count, seed=seed))

    return (truth, [3.0 * value + 10.0 for value in truth])


def test_a_space_no_larger_than_the_budget_is_walked_whole() -> None:
    result = paired_permutation(
        [1.0, 2.0, 3.0], [0.0, 0.0, 0.0], replicates=999, seed=4, tail="greater"
    )

    assert result.provenance.exact is True
    assert result.provenance.replicates == 8
    assert result.provenance.seed is None


def test_a_space_larger_than_the_budget_is_sampled() -> None:
    result = paired_permutation(
        shifted(0.5).samples(40, seed=0),
        STANDARD.samples(40, seed=1),
        replicates=499,
        seed=4,
        tail="greater",
    )

    assert result.provenance.exact is False
    assert result.provenance.replicates == 499
    assert result.provenance.seed == 4


def test_the_budget_decides_the_branch_on_one_data_set() -> None:
    left = shifted(0.6).samples(9, seed=31)
    right = STANDARD.samples(9, seed=32)
    generous = paired_permutation(left, right, replicates=600, seed=2, tail="greater")
    thin = paired_permutation(left, right, replicates=200, seed=2, tail="greater")

    assert generous.provenance.exact is True
    assert generous.provenance.replicates == 2**9
    assert thin.provenance.exact is False
    assert thin.provenance.replicates == 200


def test_the_exact_p_value_is_the_proportion_of_reassignments() -> None:
    # one of eight reassignments reaches the observed mean
    result = paired_permutation(
        [3.0, 5.0, 9.0], [0.0, 0.0, 0.0], replicates=999, seed=1, tail="greater"
    )

    assert result.value == pytest.approx(1 / 8)


def test_a_group_receives_one_sign_and_the_statistic_stays_per_item() -> None:
    result = paired_permutation(
        [1.0, 2.0, 10.0],
        [0.0, 0.0, 0.0],
        replicates=999,
        seed=1,
        tail="greater",
        groups=["speaker-a", "speaker-b", "speaker-a"],
    )

    # the four means are -13/3, -3, 3, and 13/3
    assert result.value == pytest.approx(1 / 4)
    assert result.provenance.exact is True
    assert result.provenance.replicates == 4


def test_group_count_decides_whether_the_reassignment_space_is_walked() -> None:
    result = paired_permutation(
        [float(index) for index in range(20)],
        [0.0] * 20,
        replicates=4,
        seed=3,
        tail="greater",
        groups=["first"] * 10 + ["second"] * 10,
    )

    assert result.provenance.exact is True
    assert result.provenance.replicates == 4
    assert result.provenance.seed is None


def test_unique_group_labels_preserve_the_item_level_test() -> None:
    left = shifted(0.4).samples(20, seed=23)
    right = STANDARD.samples(20, seed=24)
    item_level = paired_permutation(left, right, replicates=300, seed=8, tail="greater")
    grouped = paired_permutation(
        left,
        right,
        replicates=300,
        seed=8,
        tail="greater",
        groups=[f"item-{index}" for index in range(20)],
    )

    assert grouped == item_level


def test_sampled_groups_follow_first_appearance_order() -> None:
    result = paired_permutation(
        [1.0, -1.0, 3.0, 2.0],
        [0.0, 0.0, 0.0, 0.0],
        replicates=5,
        seed=4,
        tail="greater",
        groups=["z", "a", "z", "m"],
    )

    # first appearance gives group sums 4, -1, 2 and one extreme draw
    assert result.value == pytest.approx(1 / 3)
    assert result.provenance.exact is False
    assert result.provenance.seed == 4


def test_a_sampled_grouped_test_repeats_under_its_seed() -> None:
    left = shifted(0.4).samples(40, seed=25)
    right = STANDARD.samples(40, seed=26)
    groups = [f"speaker-{index // 2}" for index in range(40)]
    once = paired_permutation(
        left, right, replicates=300, seed=8, tail="greater", groups=groups
    )
    again = paired_permutation(
        left, right, replicates=300, seed=8, tail="greater", groups=groups
    )
    elsewhere = paired_permutation(
        left, right, replicates=300, seed=9, tail="greater", groups=groups
    )

    assert once == again
    assert once.provenance.exact is False
    assert elsewhere != once


def test_an_identical_pair_of_conditions_is_never_significant() -> None:
    scores = [1.0, 4.0, 2.0, 8.0]
    result = paired_permutation(
        scores, scores, replicates=999, seed=1, tail="two-sided"
    )

    assert result.value == 1.0


def test_the_two_branches_agree_within_monte_carlo_error() -> None:
    left = shifted(0.9).samples(11, seed=19)
    right = STANDARD.samples(11, seed=20)
    walked = paired_permutation(left, right, replicates=1 << 11, seed=1, tail="greater")
    drawn = [
        paired_permutation(left, right, replicates=2000, seed=seed, tail="greater")
        for seed in range(6)
    ]

    assert walked.provenance.exact is True

    # 2,000 draws give approximately one percent Monte Carlo error
    for estimate in drawn:
        assert estimate.provenance.exact is False
        assert abs(estimate.value - walked.value) < 0.03


def test_a_drawn_p_value_is_uniform_under_exchangeable_data() -> None:
    replications = 500
    counts = [0] * BINS

    for replication in range(replications):
        differences = STANDARD.samples(20, seed=replication)
        result = paired_permutation(
            differences,
            [0.0] * 20,
            replicates=199,
            seed=replication,
            tail="greater",
        )
        counts[min(BINS - 1, int(result.value * BINS))] += 1

    expected = replications / BINS
    statistic = fsum((count - expected) ** 2 / expected for count in counts)

    assert sum(counts) == replications
    assert statistic < CHI_SQUARE_AT_ONE_IN_A_THOUSAND


def test_a_drawn_test_repeats_under_its_seed_and_moves_under_another() -> None:
    left = shifted(0.4).samples(30, seed=5)
    right = STANDARD.samples(30, seed=6)
    once = paired_permutation(left, right, replicates=300, seed=8, tail="greater")
    again = paired_permutation(left, right, replicates=300, seed=8, tail="greater")
    elsewhere = paired_permutation(left, right, replicates=300, seed=9, tail="greater")

    assert once == again
    assert elsewhere.provenance.seed == 9
    assert elsewhere != once


def test_the_paired_test_returns_a_p_value_by_type() -> None:
    result = paired_permutation(
        [1.0, 2.0], [0.0, 1.0], replicates=999, seed=1, tail="greater"
    )

    assert isinstance(result, PValue)
    assert result.provenance.family == "permutation"
    assert result.provenance.null == "paired sign flips"


def test_the_lower_tail_reads_the_other_direction() -> None:
    upper = paired_permutation(
        [3.0, 5.0, 9.0], [0.0, 0.0, 0.0], replicates=999, seed=1, tail="greater"
    )
    lower = paired_permutation(
        [3.0, 5.0, 9.0], [0.0, 0.0, 0.0], replicates=999, seed=1, tail="less"
    )

    assert upper.value == pytest.approx(1 / 8)
    assert lower.value == 1.0


def test_a_mismatched_pair_of_sequences_is_refused() -> None:
    with pytest.raises(ValueError, match="scores on one side"):
        paired_permutation([1.0, 2.0], [0.0], replicates=99, seed=1, tail="greater")


def test_a_test_over_no_items_is_refused() -> None:
    with pytest.raises(ValueError, match="no items"):
        paired_permutation([], [], replicates=99, seed=1, tail="greater")


def test_a_score_that_is_not_finite_is_refused() -> None:
    with pytest.raises(ValueError, match="not finite"):
        paired_permutation(
            [1.0, float("inf")], [0.0, 0.0], replicates=99, seed=1, tail="greater"
        )


def test_a_score_that_is_not_a_number_is_refused() -> None:
    scores = cast("list[float]", [1.0, "2"])

    with pytest.raises(TypeError, match="not a number"):
        paired_permutation(scores, [0.0, 0.0], replicates=99, seed=1, tail="greater")


def test_a_test_with_no_replicate_budget_is_refused() -> None:
    with pytest.raises(ValueError, match="no replicate budget"):
        paired_permutation(
            STANDARD.samples(20, seed=1),
            STANDARD.samples(20, seed=2),
            replicates=0,
            seed=1,
            tail="greater",
        )


def test_a_negative_seed_is_refused() -> None:
    with pytest.raises(ValueError, match="negative"):
        paired_permutation(
            [1.0, 2.0], [0.0, 0.0], replicates=99, seed=-1, tail="greater"
        )


def test_a_grouping_that_does_not_label_every_item_is_refused() -> None:
    with pytest.raises(ValueError, match="labels for 3 items"):
        paired_permutation(
            [1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0],
            replicates=99,
            seed=1,
            tail="greater",
            groups=["a", "b"],
        )


def test_an_empty_group_label_is_refused() -> None:
    with pytest.raises(ValueError, match="is empty"):
        paired_permutation(
            [1.0, 2.0],
            [0.0, 0.0],
            replicates=99,
            seed=1,
            tail="greater",
            groups=["a", ""],
        )


def test_a_group_label_that_is_not_a_string_is_refused() -> None:
    groups = cast("list[str]", ["a", 2])

    with pytest.raises(TypeError, match="not a string"):
        paired_permutation(
            [1.0, 2.0],
            [0.0, 0.0],
            replicates=99,
            seed=1,
            tail="greater",
            groups=groups,
        )


def test_two_conventions_that_agree_on_the_ordering_are_detected() -> None:
    left, right = monotone(9, seed=11)
    result = paired_rank_inversion(left, right, replicates=999, seed=1, tail="less")

    assert result.provenance.exact is True
    assert result.value < 0.01


def test_two_unrelated_scorings_are_not_detected() -> None:
    left = STANDARD.samples(9, seed=11)
    right = STANDARD.samples(9, seed=12)
    result = paired_rank_inversion(left, right, replicates=999, seed=1, tail="less")

    assert result.value > 0.1


def test_the_rank_test_draws_where_the_swap_space_outruns_the_budget() -> None:
    left, right = monotone(20, seed=3)
    result = paired_rank_inversion(left, right, replicates=999, seed=5, tail="less")

    assert result.provenance.exact is False
    assert result.provenance.replicates == 999
    assert result.provenance.seed == 5
    assert result.value < 0.01


def test_the_rank_test_returns_a_p_value_by_type() -> None:
    left, right = monotone(6, seed=7)
    result = paired_rank_inversion(left, right, replicates=999, seed=1, tail="less")

    assert isinstance(result, PValue)
    assert result.provenance.family == "permutation"
    assert result.provenance.null == "paired rank swaps"


def test_the_rank_statistic_counts_the_pairs_ordered_oppositely() -> None:
    # three items have eight reassignments
    result = paired_rank_inversion(
        [1.0, 2.0, 3.0], [10.0, 30.0, 20.0], replicates=999, seed=1, tail="less"
    )

    assert result.provenance.replicates == 8
    assert 0.0 < result.value <= 1.0


def test_conventions_that_scored_every_item_alike_carry_no_evidence() -> None:
    # every reassignment of identical scores reproduces the observation
    scores = [1.0, 5.0, 2.0, 9.0]
    result = paired_rank_inversion(scores, scores, replicates=999, seed=1, tail="less")

    assert result.value == 1.0


def test_the_rank_test_is_reproducible_under_one_seed() -> None:
    left, right = monotone(18, seed=13)
    once = paired_rank_inversion(left, right, replicates=200, seed=6, tail="less")
    again = paired_rank_inversion(left, right, replicates=200, seed=6, tail="less")

    assert once == again


def test_the_rank_test_refuses_a_mismatched_pair() -> None:
    with pytest.raises(ValueError, match="scores on one side"):
        paired_rank_inversion([1.0], [1.0, 2.0], replicates=99, seed=1, tail="less")


def test_the_rank_test_refuses_an_empty_comparison() -> None:
    with pytest.raises(ValueError, match="no items"):
        paired_rank_inversion([], [], replicates=99, seed=1, tail="less")


def test_the_rank_test_refuses_an_empty_replicate_budget() -> None:
    left, right = monotone(20, seed=1)

    with pytest.raises(ValueError, match="no replicate budget"):
        paired_rank_inversion(left, right, replicates=0, seed=1, tail="less")
