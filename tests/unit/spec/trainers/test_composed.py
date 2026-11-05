"""Tests for alternating and staged trainer declarations."""

import didactic.api as dx
import pytest

from lofi.mixing import RatioMixingSpec
from lofi.spec import BatchSamplerSpec, RunRole, TrainingBatchKind
from lofi.spec.batch_samplers import (
    LengthBucketBatchSamplerSpec,
    SequentialBatchSamplerSpec,
)
from lofi.spec.trainers.checkpoints import CheckpointPolicySpec
from lofi.spec.trainers.composed import (
    AlternatingTrainerSpec,
    StagedTrainerSpec,
    StageSpec,
)
from lofi.spec.trainers.early_stopping import EarlyStoppingSpec
from lofi.spec.trainers.parser import ParserTrainerSpec
from lofi.spec.trainers.steps import (
    ControllerStepSpec,
    GroundedStepSpec,
    ParserStepSpec,
    TokenWarmupStepSpec,
)


def parser(*, epochs: int, active: tuple[str, ...] | None = None) -> ParserTrainerSpec:
    """Return one fixed parser horizon shared by composed trainers."""
    return ParserTrainerSpec(
        dataset="sentences",
        epochs=epochs,
        batch_size=2,
        batch_sampler=LengthBucketBatchSamplerSpec(
            bucket_factor=100,
            reshuffle_each_epoch=False,
        ),
        active_groups=active,
        early_stopping=EarlyStoppingSpec(
            metric="loss/parser",
            mode="min",
            patience=2,
            minimum_delta=0.0,
            keep_best=1,
        ),
        checkpoints=CheckpointPolicySpec(interval=1, keep=1),
    )


@pytest.mark.parametrize(
    "sampler",
    [
        LengthBucketBatchSamplerSpec(bucket_factor=100, reshuffle_each_epoch=False),
        SequentialBatchSamplerSpec(),
    ],
)
def test_alternating_training_unifies_task_and_evaluation_datasets(
    sampler: BatchSamplerSpec,
) -> None:
    trainer = AlternatingTrainerSpec(
        base=parser(epochs=2).with_(batch_sampler=sampler),
        tasks=(
            ParserStepSpec(name="parser", dataset="sentences"),
            GroundedStepSpec(name="grounded", dataset="pairs"),
        ),
        mixing=RatioMixingSpec(weights=(2, 1)),
    )

    assert trainer.dataset_requirements() == {
        "sentences": frozenset({TrainingBatchKind.SENTENCES}),
        "pairs": frozenset({TrainingBatchKind.INFERENCE_PAIRS}),
    }
    assert trainer.evaluation_dataset() == "sentences"
    assert trainer.featurized_datasets() == {"sentences", "pairs"}


def test_staged_training_declares_one_concrete_task_per_complete_interval() -> None:
    trainer = StagedTrainerSpec(
        base=parser(epochs=2),
        validation_stage="parser",
        stages=(
            StageSpec(
                name="warm-up",
                step=TokenWarmupStepSpec(
                    name="tokens", dataset="sentences", space="categories"
                ),
                active_groups=("token_encoder",),
                epochs=1,
            ),
            StageSpec(
                name="parser",
                step=ParserStepSpec(name="parser", dataset="sentences"),
                active_groups=("parser",),
                epochs=1,
            ),
        ),
    )

    assert tuple(stage.step.kind for stage in trainer.stages) == (
        "token-warm-up",
        "parser",
    )
    assert trainer.dataset_requirements() == {
        "sentences": frozenset({TrainingBatchKind.SENTENCES})
    }


def test_staged_training_requires_exact_horizon_coverage_and_stage_owned_groups() -> (
    None
):
    stage = StageSpec(
        name="parser",
        step=ParserStepSpec(name="parser", dataset="sentences"),
        active_groups=("parser",),
        epochs=1,
    )

    with pytest.raises(dx.ValidationError, match="not its base horizon"):
        StagedTrainerSpec(
            base=parser(epochs=2), validation_stage="parser", stages=(stage,)
        )

    with pytest.raises(dx.ValidationError, match="only within its stages"):
        StagedTrainerSpec(
            base=parser(epochs=1, active=("parser",)),
            validation_stage="parser",
            stages=(stage,),
        )


def test_staged_training_assigns_validation_to_its_final_stage() -> None:
    warm_up = StageSpec(
        name="warm-up",
        step=TokenWarmupStepSpec(
            name="tokens", dataset="sentences", space="categories"
        ),
        active_groups=("token_encoder",),
        epochs=1,
    )
    parser_stage = StageSpec(
        name="parser",
        step=ParserStepSpec(name="parser", dataset="sentences"),
        active_groups=("parser",),
        epochs=1,
    )

    with pytest.raises(dx.ValidationError, match=r"validation stage.*not declared"):
        StagedTrainerSpec(
            base=parser(epochs=2),
            validation_stage="unknown",
            stages=(warm_up, parser_stage),
        )

    with pytest.raises(dx.ValidationError, match=r"validation stage.*final"):
        StagedTrainerSpec(
            base=parser(epochs=2),
            validation_stage="warm-up",
            stages=(warm_up, parser_stage),
        )


def test_a_generated_staged_task_consumes_its_declared_samplers() -> None:
    trainer = StagedTrainerSpec(
        base=parser(epochs=1),
        validation_stage="controller",
        stages=(
            StageSpec(
                name="controller",
                step=ControllerStepSpec(
                    name="controller",
                    dataset="sentences",
                    space="categories",
                    max_depth=2,
                ),
                active_groups=("controller",),
                epochs=1,
            ),
        ),
    )

    assert {RunRole.TYPE_SAMPLER, RunRole.PAIR_SAMPLER} <= trainer.required_roles()
    assert not (
        {RunRole.TYPE_SAMPLER, RunRole.PAIR_SAMPLER} & trainer.forbidden_roles()
    )


def test_composed_trainers_refuse_a_task_objective_or_metric_the_loss_lacks() -> None:
    alternating = AlternatingTrainerSpec(
        base=parser(epochs=2),
        tasks=(
            ParserStepSpec(name="parser", dataset="sentences", objectives=("parser",)),
            ParserStepSpec(name="extra", dataset="sentences", objectives=("extra",)),
        ),
        mixing=RatioMixingSpec(weights=(1, 1)),
    )
    staged = StagedTrainerSpec(
        base=parser(epochs=2),
        validation_stage=None,
        stages=(
            StageSpec(
                name="only",
                step=ParserStepSpec(
                    name="extra", dataset="sentences", objectives=("extra",)
                ),
                active_groups=("parser",),
                epochs=2,
            ),
        ),
    )

    for trainer in (alternating, staged):
        trainer.refuse_loss_terms(frozenset({"parser", "extra"}))

        with pytest.raises(ValueError, match=r"absent objectives \['extra'\]"):
            trainer.refuse_loss_terms(frozenset({"parser"}))

        with pytest.raises(ValueError, match=r"early_stopping\.metric"):
            trainer.refuse_loss_terms(frozenset({"extra"}))
