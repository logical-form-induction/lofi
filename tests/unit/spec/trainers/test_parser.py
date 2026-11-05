"""Tests for the parser-trainer declaration."""

import json
import math
from typing import TYPE_CHECKING, cast

import didactic.api as dx
import pytest

from lofi.spec.batch_samplers import LengthBucketBatchSamplerSpec
from lofi.spec.components.base import RunRole
from lofi.spec.components.data import TrainingBatchKind
from lofi.spec.components.trainer import TrainerSpec
from lofi.spec.trainers.checkpoints import CheckpointPolicySpec
from lofi.spec.trainers.early_stopping import EarlyStoppingSpec
from lofi.spec.trainers.parser import ParserTrainerSpec

if TYPE_CHECKING:
    from lofi._types import JsonValue


def early_stopping() -> EarlyStoppingSpec:
    """Return one valid stopping declaration."""
    return EarlyStoppingSpec(
        metric="loss/parser",
        mode="min",
        patience=3,
        minimum_delta=0.01,
        keep_best=2,
    )


def checkpoints() -> CheckpointPolicySpec:
    """Return one valid checkpoint policy."""
    return CheckpointPolicySpec(interval=2, keep=1)


def batch_sampler(
    *, reshuffle_each_epoch: bool = False
) -> LengthBucketBatchSamplerSpec:
    """Return one explicit length-bucket declaration."""
    return LengthBucketBatchSamplerSpec(
        bucket_factor=100,
        reshuffle_each_epoch=reshuffle_each_epoch,
    )


def batch_sampler_document(
    *, reshuffle_each_epoch: bool = False
) -> dict[str, JsonValue]:
    """Return one explicit batch-sampler document."""
    return {
        "kind": "length-bucket",
        "bucket_factor": 100,
        "reshuffle_each_epoch": reshuffle_each_epoch,
    }


def early_stopping_document() -> dict[str, JsonValue]:
    """Return the stopping declaration as JSON values."""
    return cast("dict[str, JsonValue]", json.loads(early_stopping().model_dump_json()))


def test_the_parser_tag_reaches_its_variant_through_the_family_root() -> None:
    declared = TrainerSpec.model_validate(
        {
            "kind": "parser",
            "dataset": "train",
            "epochs": 12,
            "batch_size": 8,
            "accumulation_steps": 3,
            "max_grad_norm": 1.5,
            "active_groups": ["encoder", "decoder"],
            "batch_sampler": batch_sampler_document(reshuffle_each_epoch=True),
            "early_stopping": early_stopping_document(),
            "checkpoints": {"interval": 2, "keep": 1},
        }
    )

    assert type(declared) is ParserTrainerSpec
    assert declared == ParserTrainerSpec(
        dataset="train",
        epochs=12,
        batch_size=8,
        accumulation_steps=3,
        max_grad_norm=1.5,
        active_groups=("encoder", "decoder"),
        batch_sampler=batch_sampler(reshuffle_each_epoch=True),
        early_stopping=early_stopping(),
        checkpoints=checkpoints(),
    )
    assert TrainerSpec.__variants__["parser"] is ParserTrainerSpec


def test_optimizer_step_and_optional_controls_have_canonical_defaults() -> None:
    declared = ParserTrainerSpec(
        dataset="train",
        epochs=2,
        batch_size=4,
        batch_sampler=batch_sampler(),
        early_stopping=early_stopping(),
        checkpoints=checkpoints(),
    )

    assert declared.accumulation_steps == 1
    assert declared.max_grad_norm is None
    assert declared.active_groups is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("epochs", 0),
        ("epochs", -1),
        ("epochs", True),
        ("epochs", 1.5),
        ("batch_size", 0),
        ("batch_size", -1),
        ("batch_size", False),
        ("accumulation_steps", 0),
        ("accumulation_steps", -1),
        ("accumulation_steps", True),
    ],
)
def test_nonpositive_or_noninteger_training_counts_are_refused(
    field: str, value: JsonValue
) -> None:
    payload: dict[str, JsonValue] = {
        "kind": "parser",
        "dataset": "train",
        "epochs": 4,
        "batch_size": 8,
        "accumulation_steps": 2,
        "batch_sampler": batch_sampler_document(),
        "early_stopping": early_stopping_document(),
        "checkpoints": {"interval": 2, "keep": 1},
    }
    payload[field] = value

    with pytest.raises(dx.ValidationError, match=field):
        TrainerSpec.model_validate(payload)


@pytest.mark.parametrize(
    "value",
    [0.0, -1.0, math.inf, -math.inf, math.nan, True],
)
def test_invalid_gradient_clipping_is_refused(value: JsonValue) -> None:
    with pytest.raises(dx.ValidationError, match="max_grad_norm"):
        TrainerSpec.model_validate(
            {
                "kind": "parser",
                "dataset": "train",
                "epochs": 4,
                "batch_size": 8,
                "max_grad_norm": value,
                "batch_sampler": batch_sampler_document(),
                "early_stopping": early_stopping_document(),
                "checkpoints": {"interval": 2, "keep": 1},
            }
        )


@pytest.mark.parametrize(
    "active_groups",
    [
        ["encoder", "encoder"],
        [""],
        ["encoder", ""],
        ["encoder", True],
    ],
)
def test_active_groups_are_optional_distinct_names(
    active_groups: list[JsonValue],
) -> None:
    with pytest.raises(dx.ValidationError, match=r"active_groups|active group"):
        TrainerSpec.model_validate(
            {
                "kind": "parser",
                "dataset": "train",
                "epochs": 4,
                "batch_size": 8,
                "active_groups": active_groups,
                "batch_sampler": batch_sampler_document(),
                "early_stopping": early_stopping_document(),
                "checkpoints": {"interval": 2, "keep": 1},
            }
        )


@pytest.mark.parametrize(
    ("payload", "detail"),
    [
        (
            {
                "kind": "parser",
                "dataset": "train",
                "batch_size": 8,
                "batch_sampler": batch_sampler_document(),
                "early_stopping": early_stopping_document(),
                "checkpoints": {"interval": 2, "keep": 1},
            },
            "epochs",
        ),
        (
            {
                "kind": "parser",
                "dataset": "train",
                "epochs": 4,
                "batch_sampler": batch_sampler_document(),
                "early_stopping": early_stopping_document(),
                "checkpoints": {"interval": 2, "keep": 1},
            },
            "batch_size",
        ),
        (
            {
                "kind": "parser",
                "dataset": "train",
                "epochs": 4,
                "batch_size": 8,
                "batch_sampler": batch_sampler_document(),
            },
            "early_stopping",
        ),
        (
            {
                "kind": "parser",
                "dataset": "train",
                "epochs": 4,
                "batch_size": 8,
                "batch_sampler": batch_sampler_document(),
                "early_stopping": early_stopping_document(),
            },
            "checkpoints",
        ),
        (
            {
                "kind": "parser",
                "dataset": "train",
                "epochs": 4,
                "batch_size": 8,
                "batch_sampler": batch_sampler_document(),
                "shuffle": True,
                "early_stopping": early_stopping_document(),
                "checkpoints": {"interval": 2, "keep": 1},
            },
            "shuffle",
        ),
        (
            {
                "kind": "epoch",
                "dataset": "train",
                "epochs": 4,
                "batch_size": 8,
                "batch_sampler": batch_sampler_document(),
                "early_stopping": early_stopping_document(),
                "checkpoints": {"interval": 2, "keep": 1},
            },
            "'epoch'",
        ),
        (
            {
                "kind": "parser",
                "dataset": "train",
                "epochs": 4,
                "batch_size": 8,
                "early_stopping": early_stopping_document(),
                "checkpoints": {"interval": 2, "keep": 1},
            },
            "batch_sampler",
        ),
        (
            {
                "kind": "parser",
                "dataset": "train",
                "epochs": 4,
                "batch_size": 8,
                "batch_sampler": {
                    "kind": "random",
                    "reshuffle_each_epoch": "sometimes",
                },
                "early_stopping": early_stopping_document(),
                "checkpoints": {"interval": 2, "keep": 1},
            },
            "reshuffle_each_epoch",
        ),
    ],
)
def test_missing_extra_or_unknown_trainer_fields_are_refused(
    payload: dict[str, JsonValue], detail: str
) -> None:
    with pytest.raises(dx.ValidationError, match=detail):
        TrainerSpec.model_validate(payload)


def test_the_parser_trainer_round_trips_through_json_byte_for_byte() -> None:
    document = (
        '{"kind": "parser", "dataset": "train", "epochs": 12, "batch_size": 8, '
        '"accumulation_steps": 3, "max_grad_norm": 1.5, '
        '"active_groups": ["encoder", "decoder"], "batch_sampler": '
        '{"kind": "length-bucket", "bucket_factor": 100, '
        '"reshuffle_each_epoch": true, "declared_output": null}, '
        '"early_stopping": {"metric": "loss/parser", "mode": "min", '
        '"patience": 3, "minimum_delta": 0.01, "keep_best": 2}, '
        '"checkpoints": {"interval": 2, "keep": 1}, "declared_output": null}'
    )
    declared = TrainerSpec.model_validate_json(document)

    assert declared.model_dump_json() == document
    assert TrainerSpec.model_validate_json(declared.model_dump_json()) == declared


def windowed(batch_size: int, accumulation_steps: int) -> ParserTrainerSpec:
    """Return a trainer whose epochs a stated window divides or does not."""
    return ParserTrainerSpec(
        dataset="train",
        epochs=1,
        batch_size=batch_size,
        accumulation_steps=accumulation_steps,
        batch_sampler=batch_sampler(),
        early_stopping=early_stopping(),
        checkpoints=checkpoints(),
    )


@pytest.mark.parametrize(
    ("occurrences", "batch_size", "accumulation_steps"),
    [(8, 2, 2), (9, 3, 3), (11, 2, 1), (4, 3, 1), (12, 3, 2)],
)
def test_an_epoch_of_whole_accumulation_windows_is_accepted(
    occurrences: int, batch_size: int, accumulation_steps: int
) -> None:
    windowed(batch_size, accumulation_steps).refuse_epoch(occurrences)


@pytest.mark.parametrize(
    ("occurrences", "batch_size", "accumulation_steps", "microbatches"),
    [(10, 3, 3, 4), (9, 2, 3, 5), (12, 4, 2, 3)],
)
def test_an_epoch_that_leaves_a_window_open_is_refused_naming_the_key(
    occurrences: int, batch_size: int, accumulation_steps: int, microbatches: int
) -> None:
    with pytest.raises(ValueError, match=r"^trainer\.accumulation_steps") as refused:
        windowed(batch_size, accumulation_steps).refuse_epoch(occurrences)

    assert f"{microbatches} microbatches" in str(refused.value)


def test_the_parser_trainer_declares_its_roles_and_its_sentence_batches() -> None:
    trainer = ParserTrainerSpec(
        dataset="train",
        epochs=1,
        batch_size=1,
        batch_sampler=batch_sampler(),
        early_stopping=early_stopping(),
        checkpoints=checkpoints(),
    )

    assert trainer.required_roles() == {
        RunRole.PARSER,
        RunRole.TOKENS_ENCODER,
        RunRole.SUBWORD_POOLER,
        RunRole.LOSS,
    }
    assert trainer.forbidden_roles() == {
        RunRole.PAIR_SAMPLER,
        RunRole.TYPE_SAMPLER,
    }
    assert trainer.dataset_requirements() == {
        "train": frozenset({TrainingBatchKind.SENTENCES})
    }
    assert trainer.evaluation_dataset() == "train"
    assert trainer.featurized_datasets() == {"train"}


@pytest.mark.parametrize("metric", ["loss/parser", "loss/parser/accuracy"])
def test_early_stopping_monitors_a_metric_of_a_declared_loss_term(metric: str) -> None:
    declared = windowed(2, 1).with_(
        early_stopping=early_stopping().with_(metric=metric)
    )

    declared.refuse_loss_terms(frozenset({"parser", "coherence"}))


@pytest.mark.parametrize(
    "metric", ["loss/simplicity", "loss/simplicity/weight", "validation/parser", "loss"]
)
def test_early_stopping_refuses_a_metric_validation_never_reports(metric: str) -> None:
    declared = windowed(2, 1).with_(
        early_stopping=early_stopping().with_(metric=metric)
    )

    with pytest.raises(ValueError, match=r"trainer\.early_stopping\.metric"):
        declared.refuse_loss_terms(frozenset({"parser", "coherence"}))
