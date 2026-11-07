"""Tests for the timed training stacks a fitting plan publishes."""

from typing import TYPE_CHECKING, cast

from tests.unit.train.test__configured import CONFIG, PACKAGED_CODEC

from lofi.data import BatchShape
from lofi.run._device import resolve_device_policy
from lofi.run._seed import RandomStreams, SeedPolicy
from lofi.run._training import plan_training
from lofi.spec import RunSpecCodec
from lofi.train import prepare_training

if TYPE_CHECKING:
    from lofi.spec import RunSpec
    from lofi.train import ParserFittingPlan, PreparedTraining, StepFittingPlan


def _prepared(run_spec: RunSpec, /) -> PreparedTraining:
    blueprint = plan_training(run_spec)
    streams = RandomStreams(SeedPolicy(7, samplers=blueprint.sampler_streams))
    streams.apply()
    return prepare_training(blueprint, streams, resolve_device_policy(run_spec))


def test_a_parser_stack_runs_one_epoch_position_and_reports_its_shape() -> None:
    plan = cast("ParserFittingPlan", _prepared(PACKAGED_CODEC(CONFIG)).plan)
    stack = plan.timed_stack()
    first = plan.batches[0].words

    assert stack.positions == stack.steps_per_unit == len(plan.batches)
    assert stack.shape_at(0) == BatchShape(
        sum(len(words) for words in first) / len(first), float(len(first))
    )
    assert isinstance(stack.shape, BatchShape)

    stack.run(0)


def test_a_step_stack_runs_one_horizon_position_without_a_sentence_shape() -> None:
    run_spec = RunSpecCodec()(CONFIG.parent / "synthetic-distance.yaml")
    plan = cast("StepFittingPlan", _prepared(run_spec).plan)
    stack = plan.timed_stack()

    assert stack.positions == plan.steps
    assert stack.steps_per_unit == 1
    assert stack.shape is None
    assert stack.shape_at(0) is None

    stack.run(0)
