"""Composition checks for configured type-distance pretraining."""

from pathlib import Path

from lofi.data.synthetic import ExhaustiveTypeSampler
from lofi.run import (
    RandomStreams,
    SeedPolicy,
    plan_training,
    resolve_device_policy,
)
from lofi.spec import RunSpecCodec
from lofi.spec.trainers import DistanceTrainerSpec
from lofi.spec.type_samplers.distribution import (
    ArityBranchingSpec,
    UniformChildDepthSpec,
)
from lofi.train import DistanceEncoderStep, StepFittingPlan, prepare_training

CONFIG = (
    Path(__file__).resolve().parents[2] / "lofi/configs/run/synthetic-distance.yaml"
)


def test_the_packaged_distance_run_composes_without_parser_only_components() -> None:
    run_spec = RunSpecCodec()(CONFIG)
    trainer = run_spec.trainer
    blueprint = plan_training(run_spec)
    streams = RandomStreams(SeedPolicy(23, samplers=blueprint.sampler_streams))
    streams.apply()
    prepared = prepare_training(blueprint, streams, resolve_device_policy(run_spec))
    plan = prepared.plan

    assert isinstance(trainer, DistanceTrainerSpec)
    assert isinstance(plan, StepFittingPlan)
    assert trainer.ball_radius is None
    assert trainer.depth_curriculum is None
    assert trainer.inverse_distance_power == 0.0
    assert trainer.embedding_penalty == 0.0
    assert trainer.child_depth == UniformChildDepthSpec()
    assert trainer.branching == ArityBranchingSpec()
    assert trainer.constructor_weights == {}
    assert blueprint.sampler_streams == ("training:train:distance-batches",)
    assert set(prepared.model.modules) == {"terms"}
    (step,) = plan.trainers
    assert isinstance(step, DistanceEncoderStep)
    assert type(step.batcher.sampler) is ExhaustiveTypeSampler
    assert {term.op.sort for term in step.batcher.sampler.inventory} == {trainer.sort}
    assert set(run_spec.component_roles()) == {
        "data.datasets.train.dataset",
        "data.type_sampler",
        "model.encoders.terms",
        "training.optimizer",
        "training.scheduler",
        "training.trainer",
    }
