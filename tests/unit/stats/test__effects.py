from statistics import NormalDist, fmean, stdev
from typing import cast

import pytest

from lofi.stats import (
    Bootstrap,
    BootstrapResult,
    EffectSize,
    Estimator,
    paired_cliffs_delta,
    paired_cohens_d,
    paired_effect_size,
)

STANDARD = NormalDist()

PAIRS = 10_000

# standard deviation of the simulated differences
SPREAD = 2.0

# relative recovery tolerance
TOLERANCE = 0.05


def differences(truth: float, *, count: int, seed: int) -> list[float]:
    return NormalDist(truth * SPREAD, SPREAD).samples(count, seed=seed)


def clustered(*, people: int, each: int) -> tuple[list[float], list[float], list[str]]:
    advantage = NormalDist(0.4, 0.8).samples(people, seed=41)
    base = STANDARD.samples(people * each, seed=42)
    within = NormalDist(0.0, 0.1).samples(people * each, seed=43)
    scores = [
        one + advantage[index // each] + wobble
        for index, (one, wobble) in enumerate(zip(base, within, strict=True))
    ]

    return (scores, base, [f"p{index // each}" for index in range(people * each)])


def test_the_standardized_difference_recovers_a_known_value() -> None:
    for truth, seed in ((0.5, 99), (0.8, 101)):
        scores = differences(truth, count=PAIRS, seed=seed)

        estimate = paired_cohens_d(scores, [0.0] * PAIRS)

        assert abs(estimate - truth) <= TOLERANCE * truth


def test_the_standardized_difference_is_the_mean_over_the_spread() -> None:
    scores = [1.0, 2.0, 3.0, 4.0]
    base = [0.5, 0.5, 0.5, 0.5]
    gaps = [one - other for one, other in zip(scores, base, strict=True)]

    assert paired_cohens_d(scores, base) == pytest.approx(fmean(gaps) / stdev(gaps))


def test_the_standardized_difference_turns_around_with_the_comparison() -> None:
    scores = differences(0.6, count=40, seed=51)
    base = [0.0] * 40

    assert paired_cohens_d(scores, base) == pytest.approx(
        -paired_cohens_d(base, scores)
    )


def test_the_rank_alternative_counts_wins_less_losses_and_ignores_ties() -> None:
    # one win, two losses, and one tie
    assert paired_cliffs_delta(
        [1.0, 2.0, 3.0, 0.0], [0.0, 5.0, 3.0, 1.0]
    ) == pytest.approx(-0.25)


def test_the_rank_alternative_matches_a_direct_count() -> None:
    scores = STANDARD.samples(60, seed=53)
    base = NormalDist(0.2, 1.0).samples(60, seed=54)
    won = sum(1 for one, other in zip(scores, base, strict=True) if one > other)
    lost = sum(1 for one, other in zip(scores, base, strict=True) if one < other)

    assert paired_cliffs_delta(scores, base) == pytest.approx((won - lost) / 60)


def test_the_rank_alternative_reaches_one_when_every_item_goes_one_way() -> None:
    base = STANDARD.samples(30, seed=55)
    ahead = [score + 1e-9 for score in base]

    assert paired_cliffs_delta(ahead, base) == 1.0
    assert paired_cliffs_delta(base, ahead) == -1.0


def test_the_rank_alternative_ignores_how_large_each_win_is() -> None:
    base = STANDARD.samples(30, seed=57)
    steady = [score + 1.0 + 0.01 * (index % 3) for index, score in enumerate(base)]
    lopsided = [
        score + (20.0 if index == 0 else 0.1) for index, score in enumerate(base)
    ]

    # both comparisons win every item but differ in standardized magnitude
    assert paired_cliffs_delta(steady, base) == 1.0
    assert paired_cliffs_delta(lopsided, base) == 1.0
    assert paired_cohens_d(steady, base) > 10.0 * paired_cohens_d(lopsided, base)


def test_the_interval_is_drawn_over_the_declared_unit() -> None:
    scores, base, labels = clustered(people=8, each=6)
    over_items = paired_effect_size(
        scores,
        base,
        Bootstrap(unit="item", seed=3, replicates=199, level=0.95),
        estimator="paired-cohens-d",
    )
    over_people = paired_effect_size(
        scores,
        base,
        Bootstrap(
            unit="participant", seed=3, replicates=199, groups=labels, level=0.95
        ),
        estimator="paired-cohens-d",
    )

    assert over_items.value == over_people.value
    assert over_items.interval.units == 48
    assert over_people.interval.units == 8
    assert over_people.interval.unit == "participant"
    # participant resampling preserves within-participant dependence
    assert over_people.interval.percentile.width > over_items.interval.percentile.width


def test_the_estimator_and_the_interval_travel_together() -> None:
    scores, base, _ = clustered(people=8, each=6)
    effect = paired_effect_size(
        scores,
        base,
        Bootstrap(unit="item", seed=3, replicates=199, level=0.95),
        estimator="paired-cliffs-delta",
    )

    assert effect.estimator == "paired-cliffs-delta"
    assert isinstance(effect.interval, BootstrapResult)
    assert effect.value == pytest.approx(paired_cliffs_delta(scores, base))
    assert effect.interval.percentile.covers(effect.value)


def test_the_two_estimators_report_different_numbers_for_one_comparison() -> None:
    scores, base, _ = clustered(people=8, each=6)
    resampling = Bootstrap(unit="item", seed=3, replicates=199, level=0.95)
    standardized = paired_effect_size(
        scores, base, resampling, estimator="paired-cohens-d"
    )
    ranked = paired_effect_size(
        scores, base, resampling, estimator="paired-cliffs-delta"
    )

    assert standardized.value != ranked.value
    assert abs(ranked.value) <= 1.0
    assert standardized != ranked


def test_one_seed_gives_one_effect_size() -> None:
    scores, base, _ = clustered(people=8, each=6)
    once = paired_effect_size(
        scores,
        base,
        Bootstrap(unit="item", seed=3, replicates=199, level=0.95),
        estimator="paired-cohens-d",
    )
    again = paired_effect_size(
        scores,
        base,
        Bootstrap(unit="item", seed=3, replicates=199, level=0.95),
        estimator="paired-cohens-d",
    )
    elsewhere = paired_effect_size(
        scores,
        base,
        Bootstrap(unit="item", seed=4, replicates=199, level=0.95),
        estimator="paired-cohens-d",
    )

    assert once == again
    assert hash(once) == hash(again)
    assert once != elsewhere


def test_a_difference_that_never_varies_leaves_the_estimate_undefined() -> None:
    with pytest.raises(ValueError, match="divides by nothing"):
        paired_cohens_d([3.0] * 12, [1.0] * 12)


def test_one_item_is_too_few_to_standardize() -> None:
    with pytest.raises(ValueError, match="at least 2 items"):
        paired_cohens_d([1.0], [0.0])


def test_two_sequences_of_different_length_are_refused() -> None:
    with pytest.raises(ValueError, match="on one side"):
        paired_cliffs_delta([1.0, 2.0], [1.0])


def test_a_score_that_is_not_a_number_is_refused() -> None:
    with pytest.raises(TypeError, match="not a number"):
        paired_cohens_d([1.0, cast("float", "two")], [0.0, 0.0])


def test_an_undeclared_estimator_is_refused() -> None:
    with pytest.raises(ValueError, match="not one of"):
        paired_effect_size(
            [1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0],
            Bootstrap(unit="item", seed=1, replicates=9, level=0.95),
            estimator=cast("Estimator", "hedges-g"),
        )


def test_a_bootstrap_that_is_not_one_is_refused() -> None:
    with pytest.raises(TypeError, match="bootstrap is not one"):
        paired_effect_size(
            [1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0],
            cast("Bootstrap", "item"),
            estimator="paired-cohens-d",
        )


def test_an_effect_size_around_something_that_is_not_an_interval_is_refused() -> None:
    with pytest.raises(TypeError, match="not a bootstrap result"):
        EffectSize("paired-cohens-d", cast("BootstrapResult", 0.4))


def test_an_effect_size_naming_an_undeclared_estimator_is_refused() -> None:
    interval = paired_effect_size(
        [1.0, 2.0, 3.0],
        [0.0, 0.0, 0.0],
        Bootstrap(unit="item", seed=1, replicates=9, level=0.95),
        estimator="paired-cohens-d",
    ).interval

    with pytest.raises(ValueError, match="not one of"):
        EffectSize(cast("Estimator", "hedges-g"), interval)
