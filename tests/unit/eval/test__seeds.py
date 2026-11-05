from statistics import NormalDist, fmean, quantiles, stdev
from typing import Final, cast

import pytest

from lofi.eval._seeds import (
    SEED_TARGETS,
    SeedBudget,
    SeedTarget,
    multi_seed_summary,
    seed_budget,
    seeds_required,
)
from lofi.eval._values import MINIMUM_SEEDS, MetricResult, Resampling

# simulation population
TRUE_MEAN: Final = 0.80
TRUE_SD: Final = 0.012

REPLICATIONS: Final = 200
MINIMUM_COVERAGE: Final = 0.9
PILOT_SEEDS: Final = 6

PILOT: Final = tuple(NormalDist(TRUE_MEAN, TRUE_SD).samples(PILOT_SEEDS, seed=17))


def _upper_decile(sample: list[float]) -> float:
    return quantiles(sample, n=10, method="inclusive")[-1]


def _estimate(target: SeedTarget, sample: list[float]) -> tuple[float, float]:
    if target == "mean":
        return (fmean(sample), TRUE_MEAN)

    if target == "variance":
        return (stdev(sample) ** 2.0, TRUE_SD**2.0)

    decile = TRUE_MEAN + NormalDist().inv_cdf(0.9) * TRUE_SD

    return (_upper_decile(sample), decile)


def _coverage(target: SeedTarget, precision: float) -> tuple[float, int, int]:
    population = NormalDist(TRUE_MEAN, TRUE_SD)
    met = 0
    counts: list[int] = []

    for replication in range(REPLICATIONS):
        pilot = population.samples(PILOT_SEEDS, seed=1000 + replication)
        needed = seeds_required(
            pilot, target=target, precision=precision, level=0.95, minimum=MINIMUM_SEEDS
        )
        counts.append(needed)
        drawn = population.samples(needed, seed=500_000 + replication)
        estimate, truth = _estimate(target, drawn)

        if abs(estimate - truth) <= precision * abs(truth):
            met += 1

    return (met / REPLICATIONS, min(counts), max(counts))


@pytest.mark.parametrize(
    ("target", "precision"),
    [("mean", 0.005), ("variance", 0.35), ("upper-decile", 0.005)],
)
def test_the_required_count_meets_its_precision_over_simulated_replications(
    target: SeedTarget, precision: float
) -> None:
    reached, smallest, largest = _coverage(target, precision)

    assert reached >= MINIMUM_COVERAGE
    assert smallest >= MINIMUM_SEEDS
    assert largest >= smallest


def test_a_variance_target_demands_more_seeds_than_a_mean_target() -> None:
    for precision in (0.02, 0.05, 0.1, 0.2):
        mean = seeds_required(
            PILOT, target="mean", precision=precision, level=0.95, minimum=MINIMUM_SEEDS
        )
        variance = seeds_required(
            PILOT,
            target="variance",
            precision=precision,
            level=0.95,
            minimum=MINIMUM_SEEDS,
        )

        assert variance > mean


def test_an_extreme_value_target_demands_more_seeds_than_a_mean_target() -> None:
    tight = 0.002
    mean = seeds_required(
        PILOT, target="mean", precision=tight, level=0.95, minimum=MINIMUM_SEEDS
    )
    decile = seeds_required(
        PILOT, target="upper-decile", precision=tight, level=0.95, minimum=MINIMUM_SEEDS
    )

    assert decile > mean


def test_a_tighter_precision_and_a_wider_coverage_each_demand_more_seeds() -> None:
    for target in SEED_TARGETS:
        loose = seeds_required(
            PILOT, target=target, precision=0.02, level=0.95, minimum=MINIMUM_SEEDS
        )
        tight = seeds_required(
            PILOT, target=target, precision=0.005, level=0.95, minimum=MINIMUM_SEEDS
        )
        wide = seeds_required(
            PILOT, target=target, precision=0.005, level=0.99, minimum=MINIMUM_SEEDS
        )

        assert tight > loose
        assert wide > tight


def test_a_larger_pilot_is_charged_less_for_its_own_uncertainty() -> None:
    short = [TRUE_MEAN - TRUE_SD, TRUE_MEAN + TRUE_SD] * 2
    spread = TRUE_SD * (len(short) / (len(short) - 1) * 39.0 / 40.0) ** 0.5
    long = [TRUE_MEAN - spread, TRUE_MEAN + spread] * 20

    assert stdev(long) == pytest.approx(stdev(short))
    assert fmean(long) == pytest.approx(fmean(short))
    assert seeds_required(
        short, target="mean", precision=0.01, level=0.95, minimum=MINIMUM_SEEDS
    ) > seeds_required(
        long, target="mean", precision=0.01, level=0.95, minimum=MINIMUM_SEEDS
    )


def test_the_calculator_refuses_what_it_cannot_price() -> None:
    with pytest.raises(ValueError, match="not a declared one"):
        seeds_required(
            PILOT,
            target=cast("SeedTarget", "maximum"),
            precision=0.01,
            level=0.95,
            minimum=MINIMUM_SEEDS,
        )

    with pytest.raises(ValueError, match="not positive"):
        seeds_required(
            PILOT, target="mean", precision=0.0, level=0.95, minimum=MINIMUM_SEEDS
        )

    with pytest.raises(ValueError, match="pilot seeds"):
        seeds_required(
            PILOT[:1], target="mean", precision=0.01, level=0.95, minimum=MINIMUM_SEEDS
        )

    with pytest.raises(ValueError, match="not a proportion"):
        seeds_required(
            PILOT, target="mean", precision=0.01, level=1.0, minimum=MINIMUM_SEEDS
        )

    with pytest.raises(TypeError, match="not a number"):
        seeds_required(
            PILOT,
            target="mean",
            precision=cast("float", "0.01"),
            level=0.95,
            minimum=MINIMUM_SEEDS,
        )

    with pytest.raises(ValueError, match="is zero"):
        seeds_required(
            [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
            target="mean",
            precision=0.01,
            level=0.95,
            minimum=MINIMUM_SEEDS,
        )


def test_the_declared_minimum_floors_the_count_and_the_pilot() -> None:
    steady = [0.70, 0.70001, 0.69999, 0.70]

    assert (
        seeds_required(steady, target="mean", precision=0.2, level=0.95, minimum=3) == 3
    )
    assert (
        seeds_required(steady, target="mean", precision=0.2, level=0.95, minimum=9) == 9
    )
    assert (
        seeds_required(steady[:3], target="mean", precision=0.2, level=0.95, minimum=3)
        == 3
    )

    with pytest.raises(ValueError, match="at least 2 pilot seeds"):
        seeds_required(steady[:1], target="mean", precision=0.2, level=0.95, minimum=1)


def test_a_pilot_too_small_to_bound_its_own_spread_is_refused() -> None:
    with pytest.raises(ValueError, match="cannot bound their own spread"):
        seeds_required(
            PILOT[:4],
            target="mean",
            precision=0.01,
            level=0.99999,
            minimum=MINIMUM_SEEDS,
        )


def test_a_budget_reports_both_counts_and_whether_the_run_met_the_requirement() -> None:
    short = SeedBudget(target="variance", precision=0.2, required=194, achieved=6)
    enough = SeedBudget(target="variance", precision=0.2, required=4, achieved=6)

    assert short.required == 194
    assert short.achieved == 6
    assert not short.is_sufficient
    assert short.shortfall == 188
    assert enough.is_sufficient
    assert enough.shortfall == 0
    assert short.to_spec() == {
        "target": "variance",
        "precision": 0.2,
        "seeds_required": 194,
        "seeds_run": 6,
        "is_sufficient": False,
    }


def test_a_budget_refuses_a_target_or_a_precision_nobody_declared() -> None:
    with pytest.raises(ValueError, match="not a declared one"):
        SeedBudget(
            target=cast("SeedTarget", "median"),
            precision=0.2,
            required=4,
            achieved=6,
        )

    with pytest.raises(ValueError, match="not positive"):
        SeedBudget(target="mean", precision=-0.2, required=4, achieved=6)

    with pytest.raises(TypeError, match="not an integer"):
        SeedBudget(target="mean", precision=0.2, required=cast("int", 4.0), achieved=6)


def test_two_budgets_are_equal_on_the_four_fields_they_hold() -> None:
    one = SeedBudget(target="mean", precision=0.2, required=4, achieved=6)
    same = SeedBudget(target="mean", precision=0.2, required=4, achieved=6)
    other = SeedBudget(target="mean", precision=0.2, required=5, achieved=6)

    assert one == same
    assert hash(one) == hash(same)
    assert one != other
    assert one != "a budget"
    assert repr(one).startswith("SeedBudget(")


def _results() -> list[MetricResult]:
    return [
        MetricResult(value, 250, Resampling("sentence", index))
        for index, value in enumerate(PILOT)
    ]


def test_a_seed_budget_prices_the_arm_it_is_read_off() -> None:
    priced = seed_budget(_results(), target="variance", precision=0.2, level=0.95)

    assert priced.achieved == PILOT_SEEDS
    assert priced.required == seeds_required(
        PILOT, target="variance", precision=0.2, level=0.95, minimum=MINIMUM_SEEDS
    )
    assert not priced.is_sufficient


def test_a_multi_seed_row_carries_the_required_count_beside_the_count_run() -> None:
    row = multi_seed_summary(_results(), target="variance", precision=0.2, level=0.95)
    priced = seed_budget(_results(), target="variance", precision=0.2, level=0.95)

    assert row.value == pytest.approx(fmean(PILOT))
    assert row.components["seeds"] == float(PILOT_SEEDS)
    assert row.components["seeds_required"] == float(priced.required)
    assert row.components["sd"] == pytest.approx(stdev(PILOT))
    assert row.resampling.unit == "seed"


def test_a_multi_seed_row_refuses_a_table_a_seed_summary_refuses() -> None:
    with pytest.raises(ValueError, match="at least"):
        multi_seed_summary(_results()[:2], target="mean", precision=0.2, level=0.95)

    with pytest.raises(TypeError, match="not a result"):
        multi_seed_summary(
            cast("list[MetricResult]", [0.7, 0.8, 0.9, 0.75]),
            target="mean",
            precision=0.2,
            level=0.95,
        )
