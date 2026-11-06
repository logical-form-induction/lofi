from statistics import quantiles
from typing import TYPE_CHECKING

import pytest

from lofi.data import BatchShape
from lofi.run import _values as values_module
from lofi.run._values import (
    COST_MODEL,
    GRID_COST_MODEL,
    SECONDS_PER_HOUR,
    ExperimentPrice,
    GridCell,
    GridTiming,
    HardwareDescriptor,
    MemoryBudget,
    StackPrice,
    StackTiming,
)
from lofi.training_plugins import TrainingLength, TrainingUnit

if TYPE_CHECKING:
    from types import ModuleType


def test_a_budget_is_exceeded_only_by_a_measured_peak_above_it() -> None:
    budget = MemoryBudget(1_000)

    assert not budget.exceeded_by(None)
    assert not budget.exceeded_by(1_000)
    assert budget.exceeded_by(1_001)


def test_a_budget_compares_and_hashes_by_its_ceiling() -> None:
    budget = MemoryBudget(4_096)

    assert budget == MemoryBudget(4_096)
    assert hash(budget) == hash(MemoryBudget(4_096))
    assert budget != MemoryBudget(4_097)


HARDWARE = HardwareDescriptor(
    device_name="NVIDIA A40",
    driver="NVRM version: 550.54",
    cuda="12.8",
    torch="2.8.0",
    host_cpu="x86_64",
)

SHAPE = BatchShape(15.0, 8.0)


def a_stack(
    stack: str = "stack-a",
    *,
    points: tuple[str, ...] = ("p-1", "p-2"),
    epochs: int = 3,
    seconds: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0),
    peak: int | None = 1_024,
    budget: int = 8_192,
    shape: BatchShape | None = SHAPE,
) -> StackPrice:
    return StackPrice(
        stack=stack,
        points=points,
        length=TrainingLength(epochs, TrainingUnit.EPOCHS),
        timing=StackTiming(
            steps_per_unit=4,
            positions=tuple(range(len(seconds))),
            seconds=seconds,
            peak_allocated_bytes=peak,
            shape=shape,
        ),
        budget=MemoryBudget(budget),
    )


def a_grid(repeats: int = 5, *, scale: float = 1.0) -> GridTiming:
    return GridTiming(
        [
            GridCell(
                length,
                batch,
                tuple(scale * length * batch / 100 + 0.01 * k for k in range(repeats)),
            )
            for length in (10, 20)
            for batch in (8, 16)
        ]
    )


def an_experiment_price(
    stacks: tuple[StackPrice, ...] | None = None,
    *,
    resamples: int = 2_000,
    bootstrap_seed: int = 7,
    grid: GridTiming | None = None,
) -> ExperimentPrice:
    held = (
        stacks
        if stacks is not None
        else (
            a_stack(),
            a_stack("stack-b", points=("p-3",), seconds=(2.0, 2.0, 2.0, 2.0, 8.0)),
        )
    )

    return ExperimentPrice(
        experiment="headline",
        freeze_id="freeze/1",
        sweep_id="0123456789abcdef",
        device="cuda:0",
        hardware=HARDWARE,
        warmup=1,
        repeats=len(held[0].seconds),
        resamples=resamples,
        bootstrap_seed=bootstrap_seed,
        stacks=held,
        grid=grid,
    )


def test_a_stack_trains_its_declared_units_times_the_steps_one_unit_holds() -> None:
    stack = a_stack(epochs=3)

    assert stack.steps == 12
    assert stack.length == TrainingLength(3, TrainingUnit.EPOCHS)


def test_a_stack_reads_its_mean_step_off_the_timed_seconds() -> None:
    stack = a_stack(seconds=(5.0, 1.0, 4.0, 2.0, 9.0))

    assert stack.mean_seconds == pytest.approx(4.2)
    assert stack.mean_hours == pytest.approx(2 * 12 * 4.2 / SECONDS_PER_HOUR)


def test_a_timing_records_one_sampled_position_per_timed_step() -> None:
    with pytest.raises(ValueError, match="one sampled position per timed step"):
        StackTiming(
            steps_per_unit=4,
            positions=(0,),
            seconds=(1.0, 2.0),
            peak_allocated_bytes=None,
            shape=None,
        )

    with pytest.raises(ValueError, match="position"):
        StackTiming(
            steps_per_unit=4,
            positions=(0, -1),
            seconds=(1.0, 2.0),
            peak_allocated_bytes=None,
            shape=None,
        )


@pytest.mark.parametrize(
    ("seconds", "message"),
    [((), "at least one timed step"), ((1.0, 0.0), "takes time")],
    ids=("untimed", "instant"),
)
def test_a_timing_without_a_timed_step_that_took_time_is_refused(
    seconds: tuple[float, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        StackTiming(
            steps_per_unit=1,
            positions=tuple(range(len(seconds))),
            seconds=seconds,
            peak_allocated_bytes=None,
            shape=None,
        )


def test_a_timing_whose_unit_holds_no_step_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one step"):
        StackTiming(
            steps_per_unit=0,
            positions=(0,),
            seconds=(1.0,),
            peak_allocated_bytes=None,
            shape=None,
        )


def test_a_stack_trained_by_no_point_or_a_repeated_point_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one point"):
        a_stack(points=())

    with pytest.raises(ValueError, match="each point once"):
        a_stack(points=("p-1", "p-1"))


def test_a_stack_declaring_no_training_length_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one unit"):
        a_stack(epochs=0)


def test_an_experiment_total_is_bracketed_by_its_bootstrap_quantiles() -> None:
    price = an_experiment_price()
    central = sum(stack.mean_hours for stack in price.stacks)

    assert price.points == 3
    assert price.p50_hours == pytest.approx(central, rel=0.05)
    assert price.p50_hours < price.p90_hours


def test_the_experiment_p90_is_not_the_sum_of_stack_step_deciles() -> None:
    price = an_experiment_price()
    stacked = sum(
        len(stack.points)
        * stack.steps
        * quantiles(stack.seconds, n=10, method="inclusive")[8]
        / SECONDS_PER_HOUR
        for stack in price.stacks
    )

    assert price.p90_hours < stacked


def test_an_experiment_price_is_reproduced_by_its_recorded_seed() -> None:
    price = an_experiment_price(bootstrap_seed=11)

    assert an_experiment_price(bootstrap_seed=11).p90_hours == price.p90_hours
    assert an_experiment_price(bootstrap_seed=12).p90_hours != price.p90_hours
    assert price.resamples == 2_000
    assert price.bootstrap_seed == 11


@pytest.mark.parametrize(
    ("resamples", "bootstrap_seed", "message"),
    [(0, 0, "at least one resample"), (10, -1, "bootstrap seed")],
    ids=("no-resample", "negative-seed"),
)
def test_an_experiment_price_needs_resamples_and_a_seed(
    resamples: int, bootstrap_seed: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        an_experiment_price(resamples=resamples, bootstrap_seed=bootstrap_seed)


def test_a_stack_is_over_budget_only_above_its_own_ceiling() -> None:
    price = an_experiment_price(
        (
            a_stack(peak=4_096, budget=2_048),
            a_stack("stack-b", points=("p-3",), peak=4_096, budget=40_000),
            a_stack("stack-c", points=("p-4",), peak=None, budget=2_048),
        )
    )

    assert [stack.over_budget for stack in price.stacks] == [True, False, False]
    assert [stack.budget.max_bytes for stack in price.stacks] == [2_048, 40_000, 2_048]


def test_an_experiment_whose_stacks_disagree_with_its_repeats_is_refused() -> None:
    with pytest.raises(ValueError, match="not the 2 the price declares"):
        ExperimentPrice(
            experiment="headline",
            freeze_id="freeze/1",
            sweep_id="0123456789abcdef",
            device="cpu",
            hardware=HARDWARE,
            warmup=0,
            repeats=2,
            resamples=10,
            bootstrap_seed=0,
            stacks=(a_stack(),),
            grid=None,
        )


def test_a_point_training_two_stacks_is_refused() -> None:
    with pytest.raises(ValueError, match="exactly one stack"):
        an_experiment_price((a_stack(), a_stack("stack-b")))


def test_an_experiment_pricing_no_stack_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one stack"):
        ExperimentPrice(
            experiment="headline",
            freeze_id="freeze/1",
            sweep_id="0123456789abcdef",
            device="cpu",
            hardware=HARDWARE,
            warmup=0,
            repeats=2,
            resamples=10,
            bootstrap_seed=0,
            stacks=(),
            grid=None,
        )


def test_an_experiment_price_compares_and_hashes_by_every_field() -> None:
    price = an_experiment_price()

    assert price == an_experiment_price()
    assert hash(price) == hash(an_experiment_price())
    assert price != an_experiment_price((a_stack(),))
    assert a_stack() == a_stack()
    assert hash(a_stack()) == hash(a_stack())


def test_an_experiment_price_without_numpy_names_the_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(name: str) -> ModuleType:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(values_module, "import_module", missing)

    with pytest.raises(RuntimeError, match=r"lofi\[nn\]"):
        _ = an_experiment_price().p90_hours


def test_a_price_carries_the_hardware_and_peak_memory() -> None:
    price = an_experiment_price()

    assert price.hardware == HARDWARE
    assert [stack.peak_allocated_bytes for stack in price.stacks] == [1_024, 1_024]


def test_a_hardware_descriptor_without_an_accelerator_names_no_driver() -> None:
    bare = HardwareDescriptor(
        device_name="cpu", driver=None, cuda=None, torch="2.8.0", host_cpu="arm"
    )

    assert bare.driver is None
    assert bare.cuda is None
    assert hash(bare) != hash(HARDWARE)
    assert bare != HARDWARE

    with pytest.raises(ValueError, match="device name"):
        HardwareDescriptor(
            device_name="", driver=None, cuda=None, torch="2.8.0", host_cpu="arm"
        )


def test_a_stack_needs_at_most_its_peak_or_unmeasured_its_budget() -> None:
    assert a_stack(peak=1_024, budget=8_192).memory_bytes == 1_024
    assert a_stack(peak=None, budget=8_192).memory_bytes == 8_192


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        (BatchShape(10.0, 8.0), (1.0, 0.0, 0.0, 0.0)),
        (BatchShape(20.0, 16.0), (0.0, 0.0, 0.0, 1.0)),
        (BatchShape(15.0, 12.0), (0.25, 0.25, 0.25, 0.25)),
        (BatchShape(30.0, 8.0), (-1.0, 0.0, 2.0, 0.0)),
    ],
    ids=("corner", "far-corner", "center", "extrapolated"),
)
def test_a_grid_weights_its_cells_bilinearly(
    shape: BatchShape, expected: tuple[float, ...]
) -> None:
    weights = a_grid().weights(shape)

    assert weights == pytest.approx(expected)
    assert sum(weights) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "cells",
    [
        [GridCell(10, 8, (1.0,)), GridCell(10, 16, (1.0,))],
        [
            GridCell(10, 8, (1.0,)),
            GridCell(10, 16, (1.0,)),
            GridCell(20, 16, (1.0,)),
            GridCell(20, 8, (1.0,)),
        ],
        [
            GridCell(10, 8, (1.0,)),
            GridCell(10, 16, (1.0, 2.0)),
            GridCell(20, 8, (1.0,)),
            GridCell(20, 16, (1.0,)),
        ],
        [
            GridCell(10, 8, (1.0,)),
            GridCell(10, 16, (0.0,)),
            GridCell(20, 8, (1.0,)),
            GridCell(20, 16, (1.0,)),
        ],
    ],
    ids=("one-length", "out-of-order", "uneven", "instant"),
)
def test_a_grid_that_is_not_every_length_by_every_batch_is_refused(
    cells: list[GridCell],
) -> None:
    with pytest.raises(ValueError, match="grid"):
        GridTiming(cells)


def test_a_grid_prices_each_stack_at_its_own_shape() -> None:
    small = an_experiment_price((a_stack(shape=BatchShape(10.0, 8.0)),), grid=a_grid())
    large = an_experiment_price((a_stack(shape=BatchShape(20.0, 16.0)),), grid=a_grid())
    slow = an_experiment_price(
        (a_stack(shape=BatchShape(10.0, 8.0)),), grid=a_grid(scale=2.0)
    )

    assert small.cost_model == GRID_COST_MODEL
    assert an_experiment_price().cost_model == COST_MODEL
    assert small.p50_hours < large.p50_hours
    assert small.p50_hours < slow.p50_hours
    assert small.p50_hours <= small.p90_hours
    assert small.p50_hours != an_experiment_price((a_stack(),)).p50_hours


def test_a_grid_cannot_price_a_stack_that_reads_no_sentences() -> None:
    with pytest.raises(ValueError, match="read no sentences"):
        an_experiment_price((a_stack(shape=None),), grid=a_grid())

    with pytest.raises(ValueError, match="timed steps per cell"):
        an_experiment_price((a_stack(),), grid=a_grid(repeats=4))


def test_a_grid_price_differs_from_its_stack_price() -> None:
    price = an_experiment_price((a_stack(),), grid=a_grid())

    assert price.grid == a_grid()
    assert price != an_experiment_price((a_stack(),))


def test_a_point_costs_its_steps_at_its_stack_bootstrapped_step() -> None:
    price = an_experiment_price()
    p90 = price.point_seconds(0.9)
    p50 = price.point_seconds(0.5)
    first, second = price.stacks

    assert set(p90) == {"p-1", "p-2", "p-3"}
    assert p90["p-1"] == p90["p-2"]
    assert p50["p-1"] == pytest.approx(first.steps * first.mean_seconds, rel=0.1)
    assert p50["p-3"] == pytest.approx(second.steps * second.mean_seconds, rel=0.1)
    assert all(p50[point] < p90[point] for point in p90)
