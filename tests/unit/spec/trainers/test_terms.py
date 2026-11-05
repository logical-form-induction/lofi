"""Tests for the generated-terms trainer declaration."""

from typing import TYPE_CHECKING

import didactic.api as dx
import pytest

from lofi.spec import RunRole, TrainerSpec, TrainingBatchKind
from lofi.spec.trainers import (
    AccuracyStoppingSpec,
    CheckpointPolicySpec,
    TermGenerationSpec,
    TermsTrainerSpec,
)

if TYPE_CHECKING:
    from lofi._types import JsonValue


def _document() -> dict[str, JsonValue]:
    """Return one complete generated-terms trainer document."""
    return {
        "kind": "terms",
        "dataset": "train",
        "spaces": ["categories", "forms"],
        "steps": 4,
        "batch_size": 8,
        "generation": {"max_depth": 5},
        "checkpoints": {"interval": 2, "keep": 1},
    }


def test_the_terms_tag_reaches_its_closed_variant() -> None:
    built = TrainerSpec.model_validate(_document())

    assert built == TermsTrainerSpec(
        dataset="train",
        spaces=("categories", "forms"),
        steps=4,
        batch_size=8,
        generation=TermGenerationSpec(max_depth=5),
        checkpoints=CheckpointPolicySpec(interval=2, keep=1),
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("spaces", []),
        ("spaces", ["categories", "categories"]),
        ("steps", 0),
        ("batch_size", 0),
        ("generation", {"max_depth": 0}),
        ("max_grad_norm", 0.0),
    ],
)
def test_invalid_terms_training_settings_are_refused(
    field: str, value: JsonValue
) -> None:
    document = _document()
    document[field] = value

    with pytest.raises(dx.ValidationError):
        TrainerSpec.model_validate(document)


def test_the_step_horizon_visits_every_declared_space() -> None:
    document = _document()
    document["steps"] = 1

    with pytest.raises(dx.ValidationError, match="do not visit all 2"):
        TrainerSpec.model_validate(document)


def test_an_accuracy_target_supplies_the_step_ceiling() -> None:
    document = _document()
    del document["steps"]
    document["stop"] = {
        "metric": "loss/categories/autoencoding/exact_match",
        "target": 0.95,
        "max_steps": 5,
    }
    trainer = TrainerSpec.model_validate(document)

    assert isinstance(trainer, TermsTrainerSpec)
    assert trainer.steps is None
    assert trainer.stop == AccuracyStoppingSpec(
        metric="loss/categories/autoencoding/exact_match",
        target=0.95,
        max_steps=5,
    )
    assert trainer.step_horizon() == 5


@pytest.mark.parametrize("removed", ["neither", "both"])
def test_terms_training_declares_one_current_stopping_treatment(removed: str) -> None:
    document = _document()

    if removed == "neither":
        del document["steps"]
    else:
        document["stop"] = {
            "metric": "loss/categories/autoencoding/exact_match",
            "target": 0.95,
            "max_steps": 5,
        }

    with pytest.raises(dx.ValidationError, match="exactly one"):
        TrainerSpec.model_validate(document)


def test_the_terms_trainer_declares_only_its_generated_term_capabilities() -> None:
    trainer = TermsTrainerSpec.model_validate(_document())

    assert trainer.required_roles() == {
        RunRole.TERM_ENCODERS,
        RunRole.TERM_DECODERS,
        RunRole.PAIR_SAMPLER,
        RunRole.TYPE_SAMPLER,
    }
    assert trainer.forbidden_roles().isdisjoint(trainer.required_roles())
    assert {
        RunRole.PARSER,
        RunRole.CONTROLLER,
        RunRole.READOUTS,
        RunRole.LOSS,
        RunRole.EVALUATORS,
    } <= trainer.forbidden_roles()
    assert trainer.dataset_requirements() == {
        "train": frozenset({TrainingBatchKind.SENTENCES})
    }
