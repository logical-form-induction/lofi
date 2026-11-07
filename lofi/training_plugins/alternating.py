"""Alternating-training plugin."""

from typing import TYPE_CHECKING, Final

from lofi.spec.trainers.composed import AlternatingTrainerSpec
from lofi.training_plugins._plugin import (
    TrainerBuilderTarget,
    TrainerPlugin,
    TrainingLength,
    TrainingUnit,
    training_sampler_stream,
)
from lofi.training_plugins.parser import _plan as _plan_parser
from lofi.training_plugins.parser import _present as _present_parser
from lofi.training_plugins.steps import (
    merge_presentations,
    step_plugin,
    steps_presentation,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi.data import SupervisionArm
    from lofi.spec import RunSpec

__all__ = ["PLUGIN"]


def _plan(declaration: AlternatingTrainerSpec, run_spec: RunSpec, /) -> tuple[str, ...]:
    """Validate the shared parser graph and every installed step declaration."""
    for task in declaration.tasks:
        step_plugin(task)

    return tuple(
        dict.fromkeys(
            (
                *_plan_parser(
                    declaration.base,
                    run_spec,
                    composed_constraint_consumer=any(
                        task.consumes_constraints() for task in declaration.tasks
                    ),
                ),
                *(
                    training_sampler_stream(dataset, purpose)
                    for task in declaration.tasks
                    for dataset, purpose in sorted(task.sampler_purposes())
                ),
            )
        )
    )


def _present(
    declaration: AlternatingTrainerSpec, run_spec: RunSpec, /
) -> Mapping[str, frozenset[SupervisionArm]]:
    """Return what the shared parser and every task present."""
    return merge_presentations(
        (
            _present_parser(declaration.base, run_spec),
            steps_presentation(declaration.tasks),
        )
    )


def _price(declaration: AlternatingTrainerSpec, /) -> TrainingLength:
    """Return the shared parser epoch horizon."""
    return TrainingLength(declaration.base.epochs, TrainingUnit.EPOCHS)


PLUGIN: Final = TrainerPlugin(
    spec=AlternatingTrainerSpec,
    plan=_plan,
    present=_present,
    build=TrainerBuilderTarget("lofi.train._assembly:_build_alternating_training"),
    price=_price,
)
