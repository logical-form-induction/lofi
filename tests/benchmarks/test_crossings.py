"""The training hot path constructs no Didactic boundary models."""

from pathlib import Path
from typing import TYPE_CHECKING, Final

from tests.conftest import counted_constructions

from lofi.data.synthetic.training import SyntheticTrainingBatch
from lofi.run import RandomStreams, SeedPolicy, plan_training, resolve_device_policy
from lofi.spec import (
    ComputeDtype,
    DevicePolicySpec,
    MatmulPrecision,
    RunSpecCodec,
    config_search_path,
)
from lofi.train import Featurized, ParserFittingPlan, ParserStep, prepare_training

if TYPE_CHECKING:
    from lofi.train import PreparedTraining

ROOT: Final = Path(__file__).resolve().parents[2]
CONFIG: Final = ROOT / "lofi/configs/run/synthetic-parser.yaml"
STEPS: Final = 100


def _prepared() -> PreparedTraining:
    """Build the packaged parser stack before the crossing counter opens."""
    declared = RunSpecCodec(search_path=config_search_path(ROOT))(
        CONFIG,
        extra={"trainer.accumulation_steps": 1},
    )
    blueprint = plan_training(declared)
    streams = RandomStreams(SeedPolicy(41, samplers=blueprint.sampler_streams))
    streams.apply()

    return prepare_training(blueprint, streams, resolve_device_policy(declared))


def test_the_crossing_counter_observes_a_deliberate_boundary_model() -> None:
    with counted_constructions() as constructions:
        declared = DevicePolicySpec(
            device="cpu",
            compute_dtype=ComputeDtype.FLOAT32,
            deterministic=True,
            matmul_precision=MatmulPrecision.HIGHEST,
            activation_memory_bytes=8_000_000_000,
        )

    assert declared.device == "cpu"
    assert constructions.count == 1


def test_a_hundred_packaged_parser_steps_construct_no_didactic_model() -> None:
    prepared = _prepared()
    plan = prepared.plan

    assert isinstance(plan, ParserFittingPlan)
    assert isinstance(plan.step, ParserStep)
    assert plan.batches

    outcomes = []

    with counted_constructions() as constructions:
        for step in range(STEPS):
            batch = plan.batches[step % len(plan.batches)]
            assert isinstance(batch, SyntheticTrainingBatch)
            outcomes.append(plan.step.run(Featurized(batch, step)))

    assert constructions.count == 0
    assert len(outcomes) == STEPS
    assert outcomes[-1].result.step == STEPS
    assert all(
        outcome.result.metrics["optimizer/loss_finite"] == 1.0 for outcome in outcomes
    )
