"""Integration tests for validation through the configured parser graph."""

from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
import torch

from lofi.run import RandomStreams, SeedPolicy, plan_training, resolve_device_policy
from lofi.spec import PACKAGED_CONFIG_ROOT, RunSpecCodec
from lofi.train import (
    ChartForward,
    ParserFittingPlan,
    ParserStep,
    prepare_training,
    validate_epoch,
)

if TYPE_CHECKING:
    from lofi.data import ConstraintScope
    from lofi.data.synthetic.training import SyntheticTrainingBatch as TrainingBatch
    from lofi.nn.loss._constraint import BatchLocalConstraint
    from lofi.terms import Term
    from lofi.train._store import StateTree

CONFIG = Path(__file__).resolve().parents[2] / "lofi/configs/run/synthetic-parser.yaml"
CODEC = RunSpecCodec(search_path=(PACKAGED_CONFIG_ROOT,))


def _serialized(state: StateTree, /) -> bytes:
    """Return one state tree's weights-only serialization."""
    output = BytesIO()
    torch.save(state, output)

    return output.getvalue()


@pytest.mark.integration
def test_real_validation_is_item_weighted_and_does_not_mutate_optimization() -> None:
    run_spec = CODEC(CONFIG)
    blueprint = plan_training(run_spec)
    streams = RandomStreams(SeedPolicy(23, samplers=blueprint.sampler_streams))
    streams.apply()
    prepared = prepare_training(blueprint, streams, resolve_device_policy(run_spec))
    plan = prepared.plan

    assert isinstance(plan, ParserFittingPlan)
    assert isinstance(plan.forward, ChartForward)
    assert isinstance(plan.step, ParserStep)

    validation_batches = cast(
        "tuple[TrainingBatch[BatchLocalConstraint[Term, ConstraintScope]], ...]",
        plan.fit_batches,
    )
    component_before, optimization_before = prepared.state()
    modes = {role: module.training for role, module in prepared.model.modules.items()}
    validation = validate_epoch(
        plan.forward,
        plan.step.assembly,
        validation_batches,
        prepared.model.modules,
        prepared.model.policy,
        step=0,
    )
    component_after, optimization_after = prepared.state()

    assert validation.items == sum(map(len, validation_batches)) == 1
    assert plan.early_stopping.metric in validation.metrics
    assert all(
        torch.isfinite(torch.tensor(value)) for value in validation.metrics.values()
    )
    assert modes == {
        role: module.training for role, module in prepared.model.modules.items()
    }
    assert _serialized(cast("StateTree", dict(component_before))) == _serialized(
        cast("StateTree", dict(component_after))
    )
    assert _serialized(cast("StateTree", optimization_before.optimizer)) == _serialized(
        cast("StateTree", optimization_after.optimizer)
    )
    assert _serialized(cast("StateTree", optimization_before.scheduler)) == _serialized(
        cast("StateTree", optimization_after.scheduler)
    )
