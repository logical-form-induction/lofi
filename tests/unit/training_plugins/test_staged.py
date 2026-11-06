"""Tests for the staged-trainer plugin."""

from pathlib import Path

import pytest

from lofi.data import Relational
from lofi.spec import RunSpec, RunSpecCodec
from lofi.spec.trainers.composed import StagedTrainerSpec, StageSpec
from lofi.spec.trainers.early_stopping import AccuracyStoppingSpec
from lofi.spec.trainers.parser import ParserTrainerSpec
from lofi.spec.trainers.steps import (
    ControllerStepSpec,
    JudgmentStepSpec,
    ParserStepSpec,
)
from lofi.training_plugins import (
    FeaturizerMismatchError,
    TrainingLength,
    TrainingUnit,
)
from lofi.training_plugins.staged import PLUGIN

CONFIG = Path(__file__).resolve().parents[3] / "lofi/configs/run/synthetic-parser.yaml"
PACKAGED_CODEC = RunSpecCodec(search_path=(CONFIG.parent, CONFIG.parent.parent))


def declaration() -> tuple[StagedTrainerSpec, RunSpec]:
    """Return a staged declaration beside its complete run declaration."""
    run_spec = PACKAGED_CODEC(CONFIG)
    trainer = run_spec.trainer
    assert isinstance(trainer, ParserTrainerSpec)
    base = trainer.with_(active_groups=None)
    declared = StagedTrainerSpec(
        base=base,
        validation_stage="parser",
        stages=(
            StageSpec(
                name="parser",
                step=ParserStepSpec(name="parser", dataset="train"),
                active_groups=("parser",),
                epochs=base.epochs,
            ),
        ),
    )
    return declared, run_spec


def test_the_staged_plugin_reaches_every_stage_sampler() -> None:
    declared, run_spec = declaration()

    assert PLUGIN.spec is StagedTrainerSpec
    assert PLUGIN.plan(declared, run_spec) == (
        "training:train:validation-split",
        "training:train:training-batches",
        "training:train:validation-batches",
    )


def test_the_staged_plugin_prices_the_complete_stage_horizon() -> None:
    declared, _ = declaration()

    assert PLUGIN.price(declared) == TrainingLength(
        declared.base.epochs, TrainingUnit.EPOCHS
    )


def targeted(metric: str) -> tuple[StagedTrainerSpec, RunSpec]:
    """Return one terminal controller-pretraining target."""
    _, run_spec = declaration()
    trainer = run_spec.trainer
    assert isinstance(trainer, ParserTrainerSpec)
    base = trainer.with_(active_groups=None)
    return (
        StagedTrainerSpec(
            base=base,
            validation_stage=None,
            stages=(
                StageSpec(
                    name="controller",
                    step=ControllerStepSpec(
                        name="controller",
                        dataset="train",
                        space="categories",
                        max_depth=2,
                        stop=AccuracyStoppingSpec(
                            metric=metric,
                            target=0.95,
                            max_steps=2,
                        ),
                    ),
                    active_groups=("controller",),
                    epochs=base.epochs,
                ),
            ),
        ),
        run_spec,
    )


def test_a_targeted_stage_plans_a_distinct_held_out_stream() -> None:
    declared, run_spec = targeted("loss/controller/accuracy")

    assert PLUGIN.plan(declared, run_spec)[-1] == (
        "training:train:controller-held-out-batches"
    )


def test_an_unknown_target_metric_is_refused_during_allocation_free_planning() -> None:
    declared, run_spec = targeted("unknown")

    with pytest.raises(ValueError, match="not evaluated"):
        PLUGIN.plan(declared, run_spec)


def test_a_stage_presenting_pairs_to_a_sentence_featurizer_is_refused() -> None:
    declared, run_spec = declaration()
    judged = declared.with_(
        stages=(
            StageSpec(
                name="parser",
                step=JudgmentStepSpec(name="parser", dataset="train"),
                active_groups=("parser",),
                epochs=declared.base.epochs,
            ),
        )
    )

    with pytest.raises(FeaturizerMismatchError) as refused:
        PLUGIN.plan(judged, run_spec)

    assert refused.value.trainer == "staged"
    assert refused.value.presented is Relational


def test_a_target_on_a_stage_before_the_last_is_refused() -> None:
    declared, run_spec = targeted("loss/controller/accuracy")
    final = declared.stages[0]
    early = final.with_(name="early", epochs=1)
    staged = declared.with_(
        base=declared.base.with_(epochs=final.epochs + 1),
        stages=(early, final.with_(name="late")),
    )

    with pytest.raises(ValueError, match="must be the final stage"):
        PLUGIN.plan(staged, run_spec)
