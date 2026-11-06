"""Type-distance pretraining plugin."""

from typing import TYPE_CHECKING, Final

from lofi.spec.trainers.distance import DistanceTrainerSpec
from lofi.training_plugins._plugin import (
    TrainerBuilderTarget,
    TrainerPlugin,
    TrainingLength,
    TrainingUnit,
    training_sampler_stream,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi.data import SupervisionArm
    from lofi.spec import RunSpec

__all__ = ["PLUGIN"]


def _plan(declaration: DistanceTrainerSpec, run_spec: RunSpec, /) -> tuple[str, ...]:
    """Return the one stream type pairs are drawn from.

    The run's roles were checked against the declaration when it was composed,
    so nothing further constrains the graph.
    """
    del run_spec

    return (training_sampler_stream(declaration.dataset, "distance-batches"),)


def _present(
    declaration: DistanceTrainerSpec, run_spec: RunSpec, /
) -> Mapping[str, frozenset[SupervisionArm]]:
    """Return that the trainer featurizes no dataset."""
    del declaration, run_spec

    return {}


def _price(declaration: DistanceTrainerSpec, /) -> TrainingLength:
    """Return the declared optimizer steps."""
    return TrainingLength(declaration.steps, TrainingUnit.STEPS)


PLUGIN: Final = TrainerPlugin(
    spec=DistanceTrainerSpec,
    plan=_plan,
    present=_present,
    build=TrainerBuilderTarget("lofi.train._assembly:_build_distance_training"),
    price=_price,
)
"""The distance trainer's planning, construction and pricing plugin."""
