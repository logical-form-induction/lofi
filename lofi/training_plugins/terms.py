"""Generated term-pretraining plugin."""

from typing import TYPE_CHECKING, Final

from lofi.spec.trainers.generation import terms_metrics
from lofi.spec.trainers.terms import TermsTrainerSpec
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


def _plan(declaration: TermsTrainerSpec, run_spec: RunSpec, /) -> tuple[str, ...]:
    """Validate the represented spaces generated pretraining visits."""
    selected = set(declaration.spaces)
    declared = set(run_spec.model.spaces)
    missing = sorted(selected - declared)
    unselected = sorted(declared - selected)

    if missing:
        msg = (
            f"terms trainer names undeclared spaces {missing}; declared spaces: "
            f"{sorted(run_spec.model.spaces)}"
        )
        raise ValueError(msg)

    if unselected:
        msg = (
            f"terms trainer leaves declared spaces untrained {unselected}; "
            "its spaces must cover the complete represented model"
        )
        raise ValueError(msg)

    stop = declaration.stop

    if stop is not None:
        metrics = terms_metrics(
            (role, run_spec.model.spaces[role].combinators is not None)
            for role in declaration.spaces
        )

        if stop.metric not in metrics:
            msg = (
                f"terms stopping metric {stop.metric!r} is not evaluated; "
                f"known held-out metrics: {sorted(metrics)}"
            )
            raise ValueError(msg)

    streams = [training_sampler_stream(declaration.dataset, "terms-batches")]

    if stop is not None:
        streams.append(
            training_sampler_stream(declaration.dataset, "terms-held-out-batches")
        )

    return tuple(streams)


def _present(
    declaration: TermsTrainerSpec, run_spec: RunSpec, /
) -> Mapping[str, frozenset[SupervisionArm]]:
    """Return that the trainer featurizes no dataset."""
    del declaration, run_spec

    return {}


def _price(declaration: TermsTrainerSpec, /) -> TrainingLength:
    """Return the declared optimizer-step horizon."""
    return TrainingLength(declaration.step_horizon(), TrainingUnit.STEPS)


PLUGIN: Final = TrainerPlugin(
    spec=TermsTrainerSpec,
    plan=_plan,
    present=_present,
    build=TrainerBuilderTarget("lofi.train._assembly:_build_terms_training"),
    price=_price,
)
"""The terms trainer's planning, construction and pricing plugin."""
