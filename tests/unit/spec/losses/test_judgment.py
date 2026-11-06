"""Tests for ordinal-judgment inference response declarations."""

from typing import TYPE_CHECKING

import didactic.api as dx
import pytest

from lofi.lf import InferenceLabel
from lofi.spec.components.data import TrainingBatchKind
from lofi.spec.losses import InferenceLossSpec, JudgmentResponseSettings

if TYPE_CHECKING:
    from lofi._types import JsonValue


def document(entails: JsonValue = None) -> dict[str, JsonValue]:
    held_entails: JsonValue = [0.1, 0.2, 0.7] if entails is None else entails
    response: dict[str, JsonValue] = {
        "kind": "judgment",
        "probabilities": {
            "entails": held_entails,
            "contradicts": [0.7, 0.2, 0.1],
            "neutral": [0.2, 0.6, 0.2],
        },
    }
    return {
        "kind": "inference",
        "signal": "reward",
        "response": response,
    }


def test_a_judgment_response_resolves_its_instrument_and_data_capability() -> None:
    declared = InferenceLossSpec.model_validate(document())
    settings = declared.settings()

    assert settings.response.kind == "judgment"
    assert settings.response.settings == JudgmentResponseSettings(
        (
            (InferenceLabel.ENTAILS, (0.1, 0.2, 0.7)),
            (InferenceLabel.CONTRADICTS, (0.7, 0.2, 0.1)),
            (InferenceLabel.NEUTRAL, (0.2, 0.6, 0.2)),
        )
    )
    assert declared.required_training_batches() == frozenset(
        {TrainingBatchKind.JUDGMENTS}
    )


@pytest.mark.parametrize(
    "row",
    [[], [1.0], [0.5, 0.4], [-0.1, 1.1]],
)
def test_a_judgment_instrument_refuses_a_non_probability_row(
    row: JsonValue,
) -> None:
    with pytest.raises(dx.ValidationError, match="judgment instrument"):
        InferenceLossSpec.model_validate(document(row))


def test_a_judgment_instrument_refuses_an_unknown_inference_label() -> None:
    with pytest.raises(dx.ValidationError, match="unknown inference label"):
        InferenceLossSpec.model_validate(
            {
                "kind": "inference",
                "signal": "reward",
                "response": {
                    "kind": "judgment",
                    "probabilities": {
                        "entails": [0.1, 0.9],
                        "non-entails": [0.9, 0.1],
                        "other": [0.5, 0.5],
                    },
                },
            }
        )
