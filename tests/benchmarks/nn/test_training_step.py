"""Benchmarks one full training step of the real stack over twelve batch shapes.

The stack is the packaged pricing reference run document, composed at each
batch shape exactly as the length-by-batch cost model composes it, then
planned and prepared as a training run prepares it. Nothing in the step is a
stand-in.

Every shape is warmed before any is timed, and the timed runs are taken one
shape at a time round the whole series, so that a busy stretch of the machine
lands on all twelve points rather than on whichever one it coincides with.
"""

import statistics
import time
from itertools import pairwise
from typing import TYPE_CHECKING, Final, NamedTuple, cast

import pytest
import torch
import yaml
from torch.profiler import ProfilerActivity, profile

from lofi.run import ExperimentSeed, LengthBatchPricing, MemoryBudget
from lofi.run._cost_model_builtins import reference_stacks
from lofi.run._timing import ReferenceStep, prepare_reference
from lofi.spec import PACKAGED_CONFIG_ROOT, DevicePolicySpec

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pytest_benchmark.fixture import BenchmarkFixture

#: The sentence lengths the series is priced at, in tokens.
LENGTHS: Final[tuple[int, ...]] = (10, 20, 30, 40)

#: The batch sizes the series is priced at, in sentences.
SIZES: Final[tuple[int, ...]] = (8, 16, 32)

#: How many timed runs each point of the series takes.
TIMED: Final = 5

#: The point every reference stack is planned for.
POINT: Final = ExperimentSeed(0).point("pricing-reference", 0)


class Timing(NamedTuple):
    """What one timed run of one training step cost.

    Attributes
    ----------
    seconds
        Wall time from the call to the device falling idle again.
    peak_bytes
        The allocator's peak over that one run.
    """

    seconds: float
    peak_bytes: int


class StepCost(NamedTuple):
    """What one training step cost at one batch shape.

    Attributes
    ----------
    sentence_length
        Tokens in each sentence of the priced batch.
    batch_size
        Sentences in the priced batch.
    seconds
        The median of the timed runs.
    seconds_fastest
        The fastest timed run, which is the one least disturbed by other work.
    peak_bytes
        The allocator's peak over the timed runs.
    launches
        How many tensor operations one step dispatched, which is how many kernels
        the device is asked to run under eager execution.
    """

    sentence_length: int
    batch_size: int
    seconds: float
    seconds_fastest: float
    peak_bytes: int
    launches: int


class Stack(NamedTuple):
    """The reference stack prepared at every priced shape on one device.

    Attributes
    ----------
    steps
        The prepared step of each shape, by sentence length and batch size.
    device
        Where every step executes.
    width
        The parser width the reference document declares.
    """

    steps: Mapping[tuple[int, int], ReferenceStep]
    device: torch.device
    width: int


def _accelerator() -> torch.device:
    """Return the available device whose allocator the benchmark measures."""
    device = torch.accelerator.current_accelerator()

    if device is None or device.type == "cpu" or not torch.accelerator.is_available():
        pytest.skip(
            "the peak activation budget is stated for one accelerator and this "
            "machine has no accelerator, so nothing here would measure what it asserts"
        )

    return device


@pytest.fixture(scope="module")
def stack() -> Stack:
    """Return the prepared stack, or skip when the machine has no accelerator.

    Returns
    -------
    Stack
        The reference stack's step at every priced shape, on the resolved device.
    """
    device = _accelerator()

    stacks = reference_stacks(LengthBatchPricing(lengths=LENGTHS, batches=SIZES))
    parser = next(iter(stacks.values())).model.parser
    width = 0 if parser is None else cast("int", parser.model_dump()["width"])

    return Stack(
        {
            shape: prepare_reference(spec, POINT, device=str(device))
            for shape, spec in stacks.items()
        },
        device,
        width,
    )


def run(step: ReferenceStep) -> None:
    """Run one training step over its batch and wait for the device.

    Parameters
    ----------
    step : ReferenceStep
        The prepared step.
    """
    step.run()


def count_launches(step: ReferenceStep) -> int:
    """Return how many tensor operations one step dispatches.

    Parameters
    ----------
    step : ReferenceStep
        The prepared step.
    """
    with profile(activities=[ProfilerActivity.CPU]) as profiled:
        run(step)

    return sum(event.count for event in profiled.key_averages())


def shapes() -> tuple[tuple[int, int], ...]:
    """Return the twelve priced shapes, batch size outermost.

    Returns
    -------
    tuple[tuple[int, int], ...]
        The sentence length and the batch size of each point.
    """
    return tuple(
        (sentence_length, batch_size)
        for batch_size in SIZES
        for sentence_length in LENGTHS
    )


def warm(stack: Stack) -> None:
    """Run one untimed step at every priced shape.

    A device compiles and caches a kernel the first time it sees a shape, and
    the first step of all pays for the tables the allocator and the driver build
    once. Both land on whichever point runs first, which is why every shape is
    run before any is timed.

    Parameters
    ----------
    stack : Stack
        The prepared stack.
    """
    for step in stack.steps.values():
        run(step)


def time_once(step: ReferenceStep) -> Timing:
    """Run one training step, timing it and reading its peak.

    Parameters
    ----------
    step : ReferenceStep
        The prepared step, already warm at its shape.
    """
    torch.accelerator.reset_peak_memory_stats(step.device)
    start = time.perf_counter()
    run(step)
    elapsed = time.perf_counter() - start

    return Timing(elapsed, int(torch.accelerator.max_memory_allocated(step.device)))


def series(stack: Stack) -> tuple[StepCost, ...]:
    """Warm every priced shape, then price the twelve points of the series.

    Each pass round the series times one run at each shape, so a stretch of the
    machine busy with other work is spread over the twelve points rather than
    charged to the one that happened to run during it.

    Parameters
    ----------
    stack : Stack
        The prepared stack.
    """
    priced = shapes()
    warm(stack)
    timed: dict[tuple[int, int], list[Timing]] = {shape: [] for shape in priced}

    for _ in range(TIMED):
        for shape in priced:
            timed[shape].append(time_once(stack.steps[shape]))

    return tuple(
        StepCost(
            sentence_length=shape[0],
            batch_size=shape[1],
            seconds=statistics.median(timing.seconds for timing in timed[shape]),
            seconds_fastest=min(timing.seconds for timing in timed[shape]),
            peak_bytes=max(timing.peak_bytes for timing in timed[shape]),
            launches=count_launches(stack.steps[shape]),
        )
        for shape in priced
    )


def of_batch(costs: tuple[StepCost, ...], batch_size: int) -> tuple[StepCost, ...]:
    """Return the priced points of one batch size, in sentence-length order.

    Parameters
    ----------
    costs : tuple[StepCost, ...]
        The priced series.
    batch_size : int
        The batch size to select.
    """
    return tuple(cost for cost in costs if cost.batch_size == batch_size)


def rising(values: tuple[float, ...]) -> bool:
    """Return whether every value is above the one before it.

    Parameters
    ----------
    values : tuple[float, ...]
        The values in the order the series priced them.
    """
    return all(later > earlier for earlier, later in pairwise(values))


def test_the_step_series_holds_the_declared_activation_budget(
    benchmark: BenchmarkFixture, stack: Stack
) -> None:
    budget = declared_budget(stack.device)
    costs: tuple[StepCost, ...] = benchmark.pedantic(
        series, args=(stack,), rounds=1, iterations=1
    )
    benchmark.extra_info["device"] = str(stack.device)
    benchmark.extra_info["width"] = stack.width
    benchmark.extra_info["activation_budget_bytes"] = budget.max_bytes
    benchmark.extra_info["series"] = [cost._asdict() for cost in costs]

    assert not budget.exceeded_by(costs[-1].peak_bytes)


def test_one_training_step_is_timed_at_the_budget_working_point(
    benchmark: BenchmarkFixture, stack: Stack
) -> None:
    step = stack.steps[LENGTHS[-1], SIZES[-1]]
    run(step)
    benchmark.pedantic(run, args=(step,), rounds=3, iterations=1)
    measured = benchmark.stats

    if measured is None:
        pytest.fail("the step ran but pytest-benchmark timed nothing")

    benchmark.extra_info["tokens"] = LENGTHS[-1]
    benchmark.extra_info["items"] = SIZES[-1]
    benchmark.extra_info["width"] = stack.width
    benchmark.extra_info["device"] = str(stack.device)
    benchmark.extra_info["ms_per_step"] = measured.stats.median * 1e3


def declared_budget(device: torch.device) -> MemoryBudget:
    """Return the activation budget the packaged policy for a device kind declares.

    Parameters
    ----------
    device : torch.device
        The device whose kind names the packaged policy.
    """
    for path in sorted((PACKAGED_CONFIG_ROOT / "device_policy").glob("*.yaml")):
        policy = DevicePolicySpec.model_validate(
            yaml.safe_load(path.read_text(encoding="utf-8"))
        )

        if torch.device(policy.device).type == device.type:
            return MemoryBudget(policy.activation_memory_bytes)

    pytest.fail(f"no packaged device policy declares a {device.type} device")


def test_one_training_step_holds_the_declared_activation_budget(stack: Stack) -> None:
    step = stack.steps[LENGTHS[-1], SIZES[-1]]
    run(step)
    torch.accelerator.reset_peak_memory_stats(stack.device)
    run(step)
    peak = int(torch.accelerator.max_memory_allocated(stack.device))

    assert not declared_budget(stack.device).exceeded_by(peak)


def test_the_step_series_covers_every_working_point(stack: Stack) -> None:
    costs = series(stack)

    assert len(costs) == len(LENGTHS) * len(SIZES)

    for batch_size in SIZES:
        priced = of_batch(costs, batch_size)

        assert tuple(cost.sentence_length for cost in priced) == LENGTHS
        assert rising(tuple(cost.seconds_fastest for cost in priced))
        assert rising(tuple(float(cost.peak_bytes) for cost in priced))
        assert rising(tuple(float(cost.launches) for cost in priced))

    for sentence_length in LENGTHS:
        wider = tuple(cost for cost in costs if cost.sentence_length == sentence_length)

        assert rising(tuple(float(cost.peak_bytes) for cost in wider))


def test_one_training_step_allocates_accelerator_memory(stack: Stack) -> None:
    step = stack.steps[LENGTHS[-1], SIZES[-1]]
    run(step)
    torch.accelerator.reset_peak_memory_stats(stack.device)
    run(step)

    assert torch.accelerator.max_memory_allocated(stack.device) > 0


def test_accelerator_selection_refuses_a_processor_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        torch.accelerator, "current_accelerator", lambda: torch.device("cpu")
    )
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: True)

    with pytest.raises(pytest.skip.Exception, match="no accelerator"):
        _accelerator()
