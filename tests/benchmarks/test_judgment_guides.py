"""Benchmarks how a variational sweep over a crossed model grows with the design.

The claim the partially factorized family is adopted for is a cost that stays
linear in the observations and in the random effects, against the cubic solve
the exact posterior needs. Each axis is timed at three sizes spanning a factor
of four, and what is asserted is the cost per unit of the size varied: a sweep
that costs more per row at the largest design than at the smallest is growing
faster than the row count.
"""

from time import perf_counter
from typing import TYPE_CHECKING, Final

import pytest
import torch

from lofi.judgment import InferenceMethod
from lofi.judgment.crossed import CrossedScales, simulate_crossed
from lofi.judgment.guides import resolve_guide
from lofi.registry import Registry

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture

    from lofi.judgment._ports import VariationalGuide
    from lofi.judgment.crossed import CrossedModel
    from lofi.judgment.guides import CrossedState

COEFFICIENTS: Final = 3
SCALES: Final = CrossedScales(0.8, 2.0, (0.7, 0.5))
DTYPE: Final = torch.float64

#: Rows timed at a fixed number of random effects.
ROW_SIZES: Final = (8000, 16000, 32000)

#: Levels per grouping factor held while the rows vary.
HELD_LEVELS: Final = 4000

#: Levels per grouping factor timed at a fixed number of rows.
LEVEL_SIZES: Final = (4000, 8000, 16000)

#: Rows held while the random effects vary.
HELD_ROWS: Final = 4000

ROUNDS: Final = 7
SWEEPS_PER_ROUND: Final = 15

TAGS: Final = ("mean_field", "structured")


def model_of(rows: int, levels: int) -> CrossedModel:
    """Return a drawn model over two crossed factors of a stated size.

    Parameters
    ----------
    rows : int
        Number of responses.
    levels : int
        Number of levels of each of the two grouping factors.

    Returns
    -------
    CrossedModel
        The model the responses were drawn from.
    """
    generator = torch.Generator().manual_seed(rows + levels)
    fixed = torch.randn((rows, COEFFICIENTS), generator=generator, dtype=DTYPE)
    fixed[:, 0] = 1.0
    participant = torch.randint(0, levels, (rows,), generator=generator)
    item = torch.randint(0, levels, (rows,), generator=generator)
    simulated = simulate_crossed(
        fixed, (participant, item), (levels, levels), SCALES, generator=generator
    )

    return simulated.model


def per_sweep(
    guide: VariationalGuide[CrossedState], model: CrossedModel, /
) -> tuple[float, CrossedState]:
    """Return the fastest time one sweep took, and the state it reached.

    Parameters
    ----------
    guide : VariationalGuide[CrossedState]
        Guide to time.
    model : CrossedModel
        Model to sweep over.

    Returns
    -------
    seconds : float
        The fastest round's mean time per sweep.
    state : CrossedState
        The state the timed sweeps reached.
    """
    state = guide.update(model, guide.initialize(model))
    fastest = float("inf")

    for _ in range(ROUNDS):
        start = perf_counter()

        for _ in range(SWEEPS_PER_ROUND):
            state = guide.update(model, state)

        fastest = min(fastest, (perf_counter() - start) / SWEEPS_PER_ROUND)

    return fastest, state


def costs_per_unit(
    guide: VariationalGuide[CrossedState],
    models: tuple[CrossedModel, ...],
    sizes: tuple[int, ...],
) -> tuple[float, ...]:
    """Return what one sweep cost per unit of the size varied, at each size.

    Parameters
    ----------
    guide : VariationalGuide[CrossedState]
        Guide to time.
    models : tuple[CrossedModel, ...]
        Models to time, in the order their sizes are given.
    sizes : tuple[int, ...]
        The size varied, in the same order.

    Returns
    -------
    tuple[float, ...]
        Seconds per sweep per unit of size.
    """
    return tuple(
        per_sweep(guide, model)[0] / size
        for model, size in zip(models, sizes, strict=True)
    )


def worst_growth(costs: tuple[float, ...]) -> float:
    """Return the largest factor by which per-unit cost rose with size.

    Parameters
    ----------
    costs : tuple[float, ...]
        Per-unit cost at each size, in increasing order of size.

    Returns
    -------
    float
        The largest ratio of a later per-unit cost to an earlier one.
    """
    return max(
        later / earlier
        for position, earlier in enumerate(costs)
        for later in costs[position + 1 :]
    )


def test_the_timed_designs_are_the_sizes_the_benchmark_claims() -> None:
    model = model_of(ROW_SIZES[0], HELD_LEVELS)

    assert model.n_rows == ROW_SIZES[0]
    assert model.n_effects == 2 * HELD_LEVELS
    assert model.n_fixed == COEFFICIENTS
    guides = Registry[object]("guide")

    assert set(guides.available()) == set(TAGS)


@pytest.mark.parametrize("tag", TAGS)
def test_a_sweep_costs_no_more_per_row_as_the_rows_grow(
    benchmark: BenchmarkFixture, tag: str
) -> None:
    guide = resolve_guide(InferenceMethod(tag))
    models = tuple(model_of(rows, HELD_LEVELS) for rows in ROW_SIZES)
    costs = costs_per_unit(guide, models, ROW_SIZES)
    largest = models[-1]
    state = guide.update(largest, guide.initialize(largest))
    benchmark.pedantic(
        lambda: guide.update(largest, state),
        rounds=ROUNDS,
        warmup_rounds=1,
        iterations=1,
    )
    measured = benchmark.stats

    if measured is None:
        pytest.fail("the sweep ran but pytest-benchmark timed nothing")

    benchmark.extra_info["guide"] = tag
    benchmark.extra_info["varied"] = "rows"
    benchmark.extra_info["sizes"] = list(ROW_SIZES)
    benchmark.extra_info["effects"] = largest.n_effects
    benchmark.extra_info["us_per_sweep_per_row"] = [cost * 1e6 for cost in costs]
    benchmark.extra_info["median_ms_per_sweep"] = measured.stats.median * 1e3

    assert worst_growth(costs) <= 1.15


@pytest.mark.parametrize("tag", TAGS)
def test_a_sweep_costs_no_more_per_effect_as_the_effects_grow(
    benchmark: BenchmarkFixture, tag: str
) -> None:
    guide = resolve_guide(InferenceMethod(tag))
    models = tuple(model_of(HELD_ROWS, levels) for levels in LEVEL_SIZES)
    effects = tuple(model.n_effects for model in models)
    costs = costs_per_unit(guide, models, effects)
    largest = models[-1]
    state = guide.update(largest, guide.initialize(largest))
    benchmark.pedantic(
        lambda: guide.update(largest, state),
        rounds=ROUNDS,
        warmup_rounds=1,
        iterations=1,
    )
    measured = benchmark.stats

    if measured is None:
        pytest.fail("the sweep ran but pytest-benchmark timed nothing")

    benchmark.extra_info["guide"] = tag
    benchmark.extra_info["varied"] = "effects"
    benchmark.extra_info["sizes"] = list(effects)
    benchmark.extra_info["rows"] = HELD_ROWS
    benchmark.extra_info["us_per_sweep_per_effect"] = [cost * 1e6 for cost in costs]
    benchmark.extra_info["median_ms_per_sweep"] = measured.stats.median * 1e3

    assert worst_growth(costs) <= 1.15
