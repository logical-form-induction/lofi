"""Staged parser-training plugin."""

from typing import TYPE_CHECKING, Final

from lofi.spec.trainers.composed import StagedTrainerSpec
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


def _plan(declaration: StagedTrainerSpec, run_spec: RunSpec, /) -> tuple[str, ...]:
    """Validate the shared parser graph and each installed stage step."""
    last = len(declaration.stages) - 1

    for position, stage in enumerate(declaration.stages):
        step = stage.step
        step_plugin(step)
        target = step.accuracy_target()

        if target is None:
            continue

        if position != last:
            msg = (
                f"stage {stage.name!r} declares an accuracy target, so it must be "
                "the final stage"
            )
            raise ValueError(msg)

        known = step.evaluated_metrics()

        if target.metric not in known:
            msg = (
                f"stage {stage.name!r} stopping metric {target.metric!r} is not "
                f"evaluated; known held-out metrics: {sorted(known)}"
            )
            raise ValueError(msg)

    return tuple(
        dict.fromkeys(
            (
                *_plan_parser(
                    declaration.base,
                    run_spec,
                    composed_constraint_consumer=any(
                        stage.step.consumes_constraints()
                        for stage in declaration.stages
                    ),
                ),
                *(
                    training_sampler_stream(dataset, purpose)
                    for stage in declaration.stages
                    for dataset, purpose in sorted(stage.step.sampler_purposes())
                ),
            )
        )
    )


def _present(
    declaration: StagedTrainerSpec, run_spec: RunSpec, /
) -> Mapping[str, frozenset[SupervisionArm]]:
    """Return what the shared parser and every stage step present."""
    return merge_presentations(
        (
            _present_parser(declaration.base, run_spec),
            steps_presentation(stage.step for stage in declaration.stages),
        )
    )


def _price(declaration: StagedTrainerSpec, /) -> TrainingLength:
    """Return the parser epoch horizon covered by the stages."""
    return TrainingLength(declaration.base.epochs, TrainingUnit.EPOCHS)


PLUGIN: Final = TrainerPlugin(
    spec=StagedTrainerSpec,
    plan=_plan,
    present=_present,
    build=TrainerBuilderTarget("lofi.train._assembly:_build_staged_training"),
    price=_price,
)
