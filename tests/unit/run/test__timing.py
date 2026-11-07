import platform
from typing import Final

import pytest
import torch
from tests.unit.run.conftest import PACKAGED_RUNS

from lofi.data import BatchShape
from lofi.run import (
    ExperimentSeed,
    LengthBatchPricing,
    PointSeed,
    RandomStreams,
    StepTimer,
    plan_training,
)
from lofi.run._cost_model_builtins import reference_stacks
from lofi.run._device import resolve_device_policy
from lofi.run._timing import (
    ReferenceShapeError,
    TrainingStepTimer,
    prepare_reference,
)
from lofi.spec import RunSpec, RunSpecCodec
from lofi.spec.trainers.distance import DistanceTrainerSpec
from lofi.spec.trainers.parser import ParserTrainerSpec
from lofi.train import ParserFittingPlan, prepare_training

REPEATS: Final = 3
SAMPLED: Final = 8
PACKAGED_CODEC: Final = RunSpecCodec(search_path=(PACKAGED_RUNS, PACKAGED_RUNS.parent))


def a_packaged_spec(name: str) -> RunSpec:
    return PACKAGED_CODEC(PACKAGED_RUNS / f"{name}.yaml")


def at(root: int) -> PointSeed:
    """Return one point of the experiment a root seed names."""
    return ExperimentSeed(root).point("timed-point", 0)


def epoch_plan(spec: RunSpec, point: PointSeed) -> ParserFittingPlan:
    """Return the prepared fitting plan of a parser run at one point."""
    blueprint = plan_training(spec, point=point)
    streams = RandomStreams(
        point.policy(accelerator=None, samplers=blueprint.sampler_streams)
    )
    streams.apply()
    plan = prepare_training(blueprint, streams, resolve_device_policy(spec)).plan

    assert isinstance(plan, ParserFittingPlan)

    return plan


def epoch_batches(spec: RunSpec, point: PointSeed) -> int:
    """Return the batches one prepared training epoch of a parser run holds."""
    return len(epoch_plan(spec, point).batches)


def test_the_timer_is_a_step_timer() -> None:
    assert isinstance(TrainingStepTimer(), StepTimer)


def test_a_parser_stack_is_timed_over_repeated_steps_of_its_own_epoch() -> None:
    spec = a_packaged_spec("synthetic-parser")

    timing = TrainingStepTimer().time(
        spec, at(3), device="cpu", warmup=1, repeats=REPEATS
    )

    assert len(timing.seconds) == REPEATS
    assert all(second > 0 for second in timing.seconds)
    assert timing.steps_per_unit == epoch_batches(spec, at(3))
    assert timing.peak_allocated_bytes is None


def test_a_parser_stack_records_the_mean_shape_of_its_epoch() -> None:
    spec = a_packaged_spec("synthetic-parser")
    batches = epoch_plan(spec, at(3)).batches
    lengths = [len(words) for batch in batches for words in batch.words]

    timing = TrainingStepTimer().time(spec, at(3), device="cpu", warmup=0, repeats=2)

    assert timing.shape is not None
    assert timing.shape.sentence_length == pytest.approx(sum(lengths) / len(lengths))
    assert timing.shape.batch_size == pytest.approx(len(lengths) / len(batches))


def test_a_parser_stack_is_timed_over_a_seeded_sample_of_its_epoch() -> None:
    spec = a_packaged_spec("synthetic-parser")
    trainer = spec.trainer

    assert isinstance(trainer, ParserTrainerSpec)
    single = spec.with_(trainer=trainer.with_(batch_size=1, accumulation_steps=1))
    batches = epoch_batches(single, at(3))
    timer = TrainingStepTimer()

    first = timer.time(single, at(3), device="cpu", warmup=2, repeats=SAMPLED)
    again = timer.time(single, at(3), device="cpu", warmup=2, repeats=SAMPLED)
    other = timer.time(single, at(4), device="cpu", warmup=2, repeats=SAMPLED)
    in_order = tuple((2 + step) % batches for step in range(SAMPLED))

    assert len(first.positions) == SAMPLED
    assert set(first.positions) <= set(range(batches))
    assert first.positions == again.positions
    assert first.positions != other.positions
    assert first.positions != in_order


def test_a_stack_whose_corpus_seed_is_derived_is_timed_over_its_points_epoch() -> None:
    spec = PACKAGED_CODEC(
        PACKAGED_RUNS / "synthetic-parser.yaml",
        overrides=("data.datasets.train.dataset.seed=derived",),
    )
    trainer = spec.trainer

    assert isinstance(trainer, ParserTrainerSpec)
    single = spec.with_(trainer=trainer.with_(batch_size=1, accumulation_steps=1))
    timing = TrainingStepTimer().time(single, at(3), device="cpu", warmup=0, repeats=2)

    assert timing.steps_per_unit == epoch_batches(single, at(3))


def test_a_smaller_declared_batch_holds_more_steps_in_an_epoch() -> None:
    spec = a_packaged_spec("synthetic-parser")
    trainer = spec.trainer

    assert isinstance(trainer, ParserTrainerSpec)
    single = spec.with_(trainer=trainer.with_(batch_size=1, accumulation_steps=1))
    timer = TrainingStepTimer()

    paired = timer.time(spec, at(0), device="cpu", warmup=0, repeats=2)
    unpaired = timer.time(single, at(0), device="cpu", warmup=0, repeats=2)

    assert unpaired.steps_per_unit == epoch_batches(single, at(0))
    assert unpaired.steps_per_unit > paired.steps_per_unit


def test_a_step_declared_stack_is_timed_past_the_end_of_its_horizon() -> None:
    spec = a_packaged_spec("synthetic-distance")

    trainer = spec.trainer

    assert isinstance(trainer, DistanceTrainerSpec)
    timing = TrainingStepTimer().time(
        spec, at(0), device="cpu", warmup=trainer.steps, repeats=trainer.steps
    )

    assert timing.steps_per_unit == 1
    assert timing.shape is None
    assert len(timing.seconds) == trainer.steps
    assert set(timing.positions) <= set(range(trainer.steps))
    assert timing.positions != tuple(
        (trainer.steps + step) % trainer.steps for step in range(trainer.steps)
    )


@pytest.mark.parametrize(
    ("device", "message"),
    [("meta", "meta device"), ("abacus", "names no device")],
    ids=("meta", "unknown"),
)
def test_a_device_that_computes_nothing_is_refused(device: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        TrainingStepTimer().time(
            a_packaged_spec("synthetic-distance"),
            at(0),
            device=device,
            warmup=0,
            repeats=2,
        )


@pytest.mark.parametrize(
    ("declared", "device"),
    [("mps", "cpu"), ("cpu", "mps"), ("cuda", "cpu")],
    ids=("mps-on-cpu", "cpu-on-mps", "cuda-on-cpu"),
)
def test_a_device_of_another_kind_than_the_declared_one_is_refused(
    declared: str, device: str
) -> None:
    spec = a_packaged_spec("synthetic-distance")
    moved = spec.with_(device_policy=spec.device_policy.with_(device=declared))

    with pytest.raises(ValueError, match=f"declares a {declared} device"):
        TrainingStepTimer().time(moved, at(0), device=device, warmup=0, repeats=2)


def test_the_reference_stack_is_timed_at_every_cell_of_a_grid() -> None:
    stacks = reference_stacks(LengthBatchPricing(lengths=(5, 6), batches=(1, 3)))
    grid = TrainingStepTimer().time_grid(
        stacks, at(0), device="cpu", warmup=1, repeats=2
    )

    assert grid.lengths == (5, 6)
    assert grid.batches == (1, 3)
    assert [(cell.sentence_length, cell.batch_size) for cell in grid.cells] == [
        (5, 1),
        (5, 3),
        (6, 1),
        (6, 3),
    ]
    assert all(len(cell.seconds) == 2 and min(cell.seconds) > 0 for cell in grid.cells)


def test_a_reference_step_reads_exactly_its_cells_batch() -> None:
    stacks = reference_stacks(LengthBatchPricing(lengths=(5, 6), batches=(1, 3)))
    step = prepare_reference(stacks[6, 3], at(0), device="cpu")

    assert step.shape == BatchShape(6.0, 3.0)
    assert step.device == torch.device("cpu")

    step.run()


def test_a_cell_whose_first_batch_is_short_of_the_cell_is_refused() -> None:
    stacks = reference_stacks(LengthBatchPricing(lengths=(5, 6), batches=(1, 64)))

    with pytest.raises(ReferenceShapeError, match=r"cell \(5, 64\)"):
        TrainingStepTimer().time_grid(stacks, at(0), device="cpu", warmup=0, repeats=1)


def test_the_hardware_descriptor_names_the_device_and_its_framework() -> None:
    hardware = TrainingStepTimer().hardware("cpu")

    assert hardware.device_name == "cpu"
    assert hardware.driver is None
    assert hardware.cuda == torch.version.cuda
    assert hardware.torch == torch.__version__
    assert hardware.host_cpu == (platform.processor() or platform.machine())

    with pytest.raises(ValueError, match="names no device"):
        TrainingStepTimer().hardware("abacus")
