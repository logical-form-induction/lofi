from statistics import NormalDist, fmean, pvariance
from typing import TYPE_CHECKING, cast

import pytest

from lofi.stats import Bootstrap, BootstrapResult, CrossedBootstrap, paired_bootstrap

if TYPE_CHECKING:
    from collections.abc import Sequence

TRUTH = 3.0

DATASETS = 200

# acceptance band for 95 percent coverage over 200 datasets
LOWEST_ACCEPTABLE = 0.92
HIGHEST_ACCEPTABLE = 0.98


def truthful(count: int, *, seed: int) -> list[float]:
    return NormalDist(TRUTH, 1.0).samples(count, seed=seed)


def test_the_declared_budget_and_level_travel_on_the_result() -> None:
    result = Bootstrap(unit="item", seed=1, replicates=250, level=0.9).interval(
        truthful(20, seed=1), fmean
    )

    assert result.replicates == 250
    assert result.percentile.level == 0.9


def test_the_declared_unit_travels_on_the_result() -> None:
    samples = truthful(24, seed=2)
    result = Bootstrap(unit="sentence", seed=7, replicates=999, level=0.95).interval(
        samples, fmean
    )

    assert result.unit == "sentence"
    assert result.units == 24
    assert result.seed == 7
    assert result.observed == pytest.approx(fmean(samples))


def test_both_intervals_are_reported_and_bracket_the_estimate() -> None:
    samples = truthful(40, seed=3)
    result = Bootstrap(unit="item", seed=3, replicates=999, level=0.95).interval(
        samples, fmean
    )

    assert result.percentile.covers(result.observed)
    assert result.bca.covers(result.observed)
    assert result.level == 0.95
    assert result.percentile.level == result.bca.level


def test_the_nominal_interval_covers_the_truth_about_as_often_as_it_claims() -> None:
    percentile = 0
    accelerated = 0

    for dataset in range(DATASETS):
        result = Bootstrap(
            unit="item", seed=dataset, replicates=999, level=0.95
        ).interval(truthful(50, seed=dataset), fmean)
        percentile += result.percentile.covers(TRUTH)
        accelerated += result.bca.covers(TRUTH)

    assert LOWEST_ACCEPTABLE <= percentile / DATASETS <= HIGHEST_ACCEPTABLE
    assert LOWEST_ACCEPTABLE <= accelerated / DATASETS <= HIGHEST_ACCEPTABLE


def test_one_seed_gives_one_interval() -> None:
    samples = truthful(30, seed=4)
    once = Bootstrap(unit="item", seed=12, replicates=999, level=0.95).interval(
        samples, fmean
    )
    again = Bootstrap(unit="item", seed=12, replicates=999, level=0.95).interval(
        samples, fmean
    )
    elsewhere = Bootstrap(unit="item", seed=13, replicates=999, level=0.95).interval(
        samples, fmean
    )

    assert once == again
    assert once.percentile != elsewhere.percentile


def test_the_acceleration_moves_the_interval_for_a_skewed_statistic() -> None:
    # variance has a right-skewed sampling distribution
    samples = NormalDist().samples(40, seed=5)
    result = Bootstrap(unit="item", seed=3, replicates=999, level=0.95).interval(
        samples, pvariance
    )

    assert result.bca.low > result.percentile.low
    assert result.bca.high > result.percentile.high


def test_a_statistic_the_resampling_cannot_move_gives_one_point() -> None:
    # constant data make the acceleration correction undefined
    result = Bootstrap(unit="item", seed=3, replicates=999, level=0.95).interval(
        [2.0] * 12, fmean
    )

    assert result.percentile == result.bca
    assert result.percentile.low == result.percentile.high == 2.0


def test_a_higher_level_gives_a_wider_interval() -> None:
    samples = truthful(40, seed=6)
    narrow = Bootstrap(unit="item", seed=3, level=0.5, replicates=999).interval(
        samples, fmean
    )
    wide = Bootstrap(unit="item", seed=3, level=0.99, replicates=999).interval(
        samples, fmean
    )

    assert narrow.percentile.width < wide.percentile.width
    assert narrow.level == 0.5


def test_a_grouping_resamples_whole_units_and_counts_them() -> None:
    samples = truthful(40, seed=7)
    labels = [f"p{index // 8}" for index in range(40)]
    grouped = Bootstrap(
        unit="participant", seed=3, groups=labels, replicates=999, level=0.95
    ).interval(samples, fmean)
    ungrouped = Bootstrap(unit="item", seed=3, replicates=999, level=0.95).interval(
        samples, fmean
    )

    assert grouped.units == 5
    assert ungrouped.units == 40
    assert grouped.observed == pytest.approx(ungrouped.observed)
    # five participant units yield a wider interval than forty item units
    assert grouped.percentile.width > ungrouped.percentile.width


def test_stratification_preserves_each_observed_stratum_count() -> None:
    samples = ((False, 0.1), (False, 0.4), (True, 0.6), (True, 0.9))

    def separation(rows: Sequence[tuple[bool, float]]) -> float:
        negatives = [score for label, score in rows if not label]
        positives = [score for label, score in rows if label]

        if not negatives or not positives:
            return float("nan")

        return fmean(positives) - fmean(negatives)

    result = Bootstrap(
        unit="sentence",
        seed=1,
        strata=[str(label) for label, _ in samples],
        replicates=999,
        level=0.95,
    ).interval(samples, separation)

    assert result.units == len(samples)
    assert result.observed == pytest.approx(0.5)
    assert result.percentile.low <= result.observed <= result.percentile.high


def test_stratification_requires_a_label_for_every_sample() -> None:
    bootstrap = Bootstrap(
        unit="sentence",
        seed=1,
        strata=["negative", "positive"],
        replicates=999,
        level=0.95,
    )

    with pytest.raises(ValueError, match="strata for 3 samples"):
        bootstrap.interval([1.0, 2.0, 3.0], fmean)


def test_each_stratum_requires_two_units_for_the_accelerated_interval() -> None:
    bootstrap = Bootstrap(
        unit="sentence",
        seed=1,
        strata=["negative", "negative", "positive"],
        replicates=999,
        level=0.95,
    )

    with pytest.raises(ValueError, match=r"stratum 'positive'.*at least 2"):
        bootstrap.interval([1.0, 2.0, 3.0], fmean)


def test_grouping_and_stratification_cannot_name_two_resampling_units() -> None:
    with pytest.raises(ValueError, match="both groups and strata"):
        Bootstrap(
            unit="sentence",
            seed=1,
            groups=["speaker-a", "speaker-a", "speaker-b", "speaker-b"],
            strata=["negative", "positive", "negative", "positive"],
            replicates=999,
            level=0.95,
        )


def test_the_paired_variant_reports_one_quantity_at_two_units() -> None:
    samples = truthful(48, seed=8)
    labels = [f"p{index // 6}" for index in range(48)]
    sentences = Bootstrap(unit="sentence", seed=5, replicates=999, level=0.95)
    participants = Bootstrap(
        unit="participant", seed=5, groups=labels, replicates=999, level=0.95
    )
    over_sentences, over_participants = paired_bootstrap(
        samples, fmean, sentences, participants
    )

    assert isinstance(over_sentences, BootstrapResult)
    assert over_sentences.unit == "sentence"
    assert over_participants.unit == "participant"
    assert over_sentences.observed == over_participants.observed
    assert over_sentences.units == 48
    assert over_participants.units == 8


def test_the_paired_variant_refuses_one_unit_named_twice() -> None:
    with pytest.raises(ValueError, match="on both sides"):
        paired_bootstrap(
            truthful(10, seed=9),
            fmean,
            Bootstrap(unit="item", seed=1, replicates=999, level=0.95),
            Bootstrap(unit="item", seed=2, replicates=999, level=0.95),
        )


def test_the_paired_variant_refuses_two_different_coverages() -> None:
    with pytest.raises(ValueError, match="coverage on one side"):
        paired_bootstrap(
            truthful(10, seed=9),
            fmean,
            Bootstrap(unit="item", seed=1, replicates=999, level=0.95),
            Bootstrap(unit="sentence", seed=1, level=0.9, replicates=999),
        )


def test_the_paired_variant_refuses_two_different_budgets() -> None:
    with pytest.raises(ValueError, match="replicates on one"):
        paired_bootstrap(
            truthful(10, seed=9),
            fmean,
            Bootstrap(unit="item", seed=1, replicates=999, level=0.95),
            Bootstrap(unit="sentence", seed=1, replicates=99, level=0.95),
        )


def test_the_paired_variant_refuses_something_that_is_not_a_bootstrap() -> None:
    not_one = cast("Bootstrap", "item")

    with pytest.raises(TypeError, match="is not one"):
        paired_bootstrap(
            truthful(10, seed=9),
            fmean,
            Bootstrap(unit="a", seed=1, replicates=999, level=0.95),
            not_one,
        )


def test_a_bootstrap_over_one_unit_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        Bootstrap(unit="item", seed=1, replicates=999, level=0.95).interval(
            [1.0], fmean
        )


def test_a_grouping_that_does_not_label_every_sample_is_refused() -> None:
    with pytest.raises(ValueError, match="labels for 3 samples"):
        Bootstrap(
            unit="participant", seed=1, groups=["a", "b"], replicates=999, level=0.95
        ).interval([1.0, 2.0, 3.0], fmean)


def test_an_empty_group_label_is_refused() -> None:
    with pytest.raises(ValueError, match="is empty"):
        Bootstrap(
            unit="participant", seed=1, groups=["a", ""], replicates=999, level=0.95
        )


def test_an_empty_unit_name_is_refused() -> None:
    with pytest.raises(ValueError, match="is empty"):
        Bootstrap(unit="", seed=1, replicates=999, level=0.95)


def test_a_bootstrap_with_no_replicate_budget_is_refused() -> None:
    with pytest.raises(ValueError, match="no replicate budget"):
        Bootstrap(unit="item", replicates=0, seed=1, level=0.95)


def test_a_negative_seed_is_refused() -> None:
    with pytest.raises(ValueError, match="negative"):
        Bootstrap(unit="item", seed=-1, replicates=999, level=0.95)


def test_a_level_that_is_not_a_proportion_is_refused() -> None:
    with pytest.raises(ValueError, match="not a proportion"):
        Bootstrap(unit="item", seed=1, level=1.0, replicates=999)


def test_a_statistic_that_does_not_return_a_number_is_refused() -> None:
    def naming(samples: Sequence[float]) -> float:
        return cast("float", f"{len(samples)} of them")

    with pytest.raises(TypeError, match="not a number"):
        Bootstrap(unit="item", seed=1, replicates=999, level=0.95).interval(
            [1.0, 2.0, 3.0], naming
        )


def test_a_statistic_that_wanders_off_the_reals_is_refused() -> None:
    def diverging(samples: Sequence[float]) -> float:
        return float("inf") if samples else 0.0

    with pytest.raises(ValueError, match="not finite"):
        Bootstrap(unit="item", seed=1, replicates=999, level=0.95).interval(
            [1.0, 2.0, 3.0], diverging
        )


def crossed_design(
    participants: int, items: int, *, seed: int
) -> tuple[list[float], list[str], list[str]]:
    participant_effects = NormalDist(0.0, 1.0).samples(participants, seed=seed)
    item_effects = NormalDist(0.0, 1.0).samples(items, seed=seed + 1)
    noise = NormalDist(0.0, 0.5).samples(participants * items, seed=seed + 2)
    samples = [
        TRUTH + participant_effects[p] + item_effects[i] + noise[p * items + i]
        for p in range(participants)
        for i in range(items)
    ]
    rows = [f"p{p}" for p in range(participants) for _ in range(items)]
    columns = [f"i{i}" for _ in range(participants) for i in range(items)]

    return samples, rows, columns


def crossed(
    rows: Sequence[str], columns: Sequence[str], *, seed: int = 5
) -> CrossedBootstrap:
    return CrossedBootstrap(
        units=("participant", "item"),
        rows=rows,
        columns=columns,
        replicates=999,
        seed=seed,
        level=0.95,
    )


def test_a_crossed_bootstrap_names_both_units_and_counts_the_observed_cells() -> None:
    samples, rows, columns = crossed_design(6, 5, seed=11)
    result = crossed(rows, columns).interval(samples, fmean)

    assert result.unit == "participant-by-item"
    assert result.units == 30
    assert result.seed == 5
    assert result.replicates == 999
    assert result.observed == pytest.approx(fmean(samples))
    assert result.percentile.covers(result.observed)
    assert result.bca.covers(result.observed)


def test_a_crossed_resample_weights_each_sample_by_both_of_its_draws() -> None:
    samples = [1.0, 10.0, 100.0, 1000.0]
    rows = ["p0", "p0", "p1", "p1"]
    columns = ["i0", "i1", "i0", "i1"]
    seen: list[tuple[float, ...]] = []

    def record(drawn: Sequence[float]) -> float:
        seen.append(tuple(sorted(drawn)))

        return fmean(drawn)

    CrossedBootstrap(
        units=("participant", "item"),
        rows=rows,
        columns=columns,
        replicates=200,
        seed=3,
        level=0.9,
    ).interval(samples, record)
    resamples = seen[1:201]
    counts = {
        (row, column): {resample.count(sample) for resample in resamples}
        for row, column, sample in zip(rows, columns, samples, strict=True)
    }

    # each cell appears row-draws times column-draws times: 0, 1, 2 or 4
    assert all(seen_counts <= {0, 1, 2, 4} for seen_counts in counts.values())
    assert any(4 in seen_counts for seen_counts in counts.values())
    assert all(len(resample) in {1, 2, 4} for resample in resamples)


def test_one_seed_gives_one_crossed_interval() -> None:
    samples, rows, columns = crossed_design(5, 4, seed=13)
    first = crossed(rows, columns, seed=9).interval(samples, fmean)
    second = crossed(rows, columns, seed=9).interval(samples, fmean)
    other = crossed(rows, columns, seed=10).interval(samples, fmean)

    assert first.percentile == second.percentile
    assert first.bca == second.bca
    assert first.percentile != other.percentile


def test_resampling_both_factors_widens_the_interval_over_either_alone() -> None:
    samples, rows, columns = crossed_design(8, 8, seed=17)
    both = crossed(rows, columns).interval(samples, fmean)
    participants = Bootstrap(
        unit="participant", groups=rows, replicates=999, seed=5, level=0.95
    ).interval(samples, fmean)
    items = Bootstrap(
        unit="item", groups=columns, replicates=999, seed=5, level=0.95
    ).interval(samples, fmean)

    assert both.percentile.width > participants.percentile.width
    assert both.percentile.width > items.percentile.width


def test_a_crossed_bootstrap_refuses_one_unit_named_twice() -> None:
    with pytest.raises(ValueError, match="resamples 'participant' on both"):
        CrossedBootstrap(
            units=("participant", "participant"),
            rows=["a", "b"],
            columns=["c", "d"],
            replicates=10,
            seed=1,
            level=0.9,
        )


def test_a_crossed_bootstrap_refuses_labels_that_miss_a_sample() -> None:
    bootstrap = crossed(["p0", "p1"], ["i0", "i1", "i0"])

    with pytest.raises(ValueError, match="2 participant labels and 3 item labels"):
        bootstrap.interval([1.0, 2.0, 3.0], fmean)


def test_a_crossed_bootstrap_refuses_a_factor_with_one_cluster() -> None:
    bootstrap = crossed(["p0", "p0", "p0"], ["i0", "i1", "i2"])

    with pytest.raises(ValueError, match="at least 2 'participant'"):
        bootstrap.interval([1.0, 2.0, 3.0], fmean)


@pytest.mark.parametrize(
    ("changed", "match"),
    [
        ({"replicates": 0}, "no replicate budget"),
        ({"seed": -1}, "seed"),
        ({"level": 1.0}, "not a proportion"),
        ({"rows": ["", "p1"]}, "participant label"),
        ({"columns": ["i0", ""]}, "item label"),
    ],
    ids=["replicates", "seed", "level", "row-label", "column-label"],
)
def test_a_crossed_bootstrap_refuses_an_invalid_declaration(
    changed: dict[str, object], match: str
) -> None:
    declared: dict[str, object] = {
        "rows": ["p0", "p1"],
        "columns": ["i0", "i1"],
        "replicates": 10,
        "seed": 1,
        "level": 0.9,
    } | changed

    with pytest.raises(ValueError, match=match):
        CrossedBootstrap(
            units=("participant", "item"),
            rows=cast("list[str]", declared["rows"]),
            columns=cast("list[str]", declared["columns"]),
            replicates=cast("int", declared["replicates"]),
            seed=cast("int", declared["seed"]),
            level=cast("float", declared["level"]),
        )
