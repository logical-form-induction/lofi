"""Tests for the distance-trainer declaration."""

from typing import TYPE_CHECKING

import didactic.api as dx
import pytest

from lofi.spec import RunRole, TrainerSpec, TrainingBatchKind
from lofi.spec.trainers import (
    CheckpointPolicySpec,
    DepthCurriculumSpec,
    DistanceTrainerSpec,
)
from lofi.spec.type_samplers.distribution import (
    ArityBranchingSpec,
    UniformChildDepthSpec,
)

if TYPE_CHECKING:
    from lofi._types import JsonValue


def test_the_distance_tag_reaches_its_closed_variant() -> None:
    built = TrainerSpec.model_validate(
        {
            "kind": "distance",
            "dataset": "train",
            "encoder": "terms",
            "sort": "Ty",
            "steps": 20,
            "batch_size": 8,
            "checkpoints": {"interval": 5, "keep": 2},
            "inverse_distance_power": 0.0,
            "embedding_penalty": 0.0,
        }
    )

    assert built == DistanceTrainerSpec(
        dataset="train",
        encoder="terms",
        sort="Ty",
        steps=20,
        batch_size=8,
        checkpoints=CheckpointPolicySpec(interval=5, keep=2),
        inverse_distance_power=0.0,
        embedding_penalty=0.0,
    )
    assert built.ball_radius is None
    assert built.depth_curriculum is None
    assert built.inverse_distance_power == 0.0
    assert built.embedding_penalty == 0.0
    assert built.child_depth == UniformChildDepthSpec()
    assert built.branching == ArityBranchingSpec()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("steps", 0),
        ("batch_size", 0),
        ("ball_radius", -1),
        ("inverse_distance_power", -0.1),
        ("embedding_penalty", float("inf")),
        ("child_depth", {"kind": "power", "exponent": float("inf")}),
        ("branching", {"by_arity": [{"arity": 2, "deepest": [1.0]}]}),
        ("max_grad_norm", 0.0),
    ],
)
def test_invalid_distance_training_settings_are_refused(
    field: str, value: JsonValue
) -> None:
    document: dict[str, JsonValue] = {
        "kind": "distance",
        "dataset": "train",
        "encoder": "terms",
        "sort": "Ty",
        "steps": 20,
        "batch_size": 8,
        "checkpoints": {"interval": 5, "keep": 2},
        "inverse_distance_power": 0.0,
        "embedding_penalty": 0.0,
    }
    document[field] = value

    with pytest.raises(dx.ValidationError):
        TrainerSpec.model_validate(document)


def test_the_depth_curriculum_has_a_configurable_start_and_an_off_state() -> None:
    off = DistanceTrainerSpec(
        dataset="train",
        encoder="terms",
        sort="Ty",
        steps=20,
        batch_size=8,
        checkpoints=CheckpointPolicySpec(interval=5, keep=2),
        inverse_distance_power=0.0,
        embedding_penalty=0.0,
    )
    scheduled = off.with_(depth_curriculum=DepthCurriculumSpec(start_depth=3, steps=12))

    assert off.depth_curriculum is None
    assert scheduled.depth_curriculum == DepthCurriculumSpec(start_depth=3, steps=12)

    with pytest.raises(dx.ValidationError):
        DepthCurriculumSpec(start_depth=0, steps=12)


def test_an_unknown_distance_training_setting_is_refused() -> None:
    with pytest.raises(dx.ValidationError, match="unknown field"):
        DistanceTrainerSpec.model_validate(
            {
                "dataset": "train",
                "steps": 20,
                "batch_size": 8,
                "checkpoints": {"interval": 5, "keep": 2},
                "inverse_distance_power": 0.0,
                "embedding_penalty": 0.0,
                "radius": 2,
            }
        )


@pytest.mark.parametrize("omitted", ["inverse_distance_power", "embedding_penalty"])
def test_each_objective_coefficient_is_required(omitted: str) -> None:
    document: dict[str, JsonValue] = {
        "kind": "distance",
        "dataset": "train",
        "encoder": "terms",
        "sort": "Ty",
        "steps": 20,
        "batch_size": 8,
        "checkpoints": {"interval": 5, "keep": 2},
        "inverse_distance_power": 0.0,
        "embedding_penalty": 0.0,
    }
    del document[omitted]

    with pytest.raises(dx.ValidationError, match=omitted):
        TrainerSpec.model_validate(document)


def test_the_distance_trainer_declares_its_roles_and_its_sentence_batches() -> None:
    trainer = DistanceTrainerSpec(
        dataset="train",
        encoder="terms",
        sort="Ty",
        steps=2,
        batch_size=2,
        checkpoints=CheckpointPolicySpec(interval=1, keep=1),
        inverse_distance_power=0.0,
        embedding_penalty=0.0,
    )

    assert trainer.required_roles() == {RunRole.TERM_ENCODERS, RunRole.TYPE_SAMPLER}
    assert trainer.forbidden_roles().isdisjoint(trainer.required_roles())
    assert {
        RunRole.PARSER,
        RunRole.PAIR_SAMPLER,
        RunRole.LOSS,
        RunRole.EVALUATORS,
    } <= (trainer.forbidden_roles())
    assert trainer.dataset_requirements() == {
        "train": frozenset({TrainingBatchKind.SENTENCES})
    }


def test_the_checkpoint_policy_is_required() -> None:
    with pytest.raises(dx.ValidationError, match="checkpoints"):
        DistanceTrainerSpec.model_validate(
            {
                "dataset": "train",
                "steps": 20,
                "batch_size": 8,
                "inverse_distance_power": 0.0,
                "embedding_penalty": 0.0,
            }
        )
