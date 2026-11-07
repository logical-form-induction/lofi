"""Tests for the early-stopping declaration."""

from typing import TYPE_CHECKING

import didactic.api as dx
import pytest

from lofi.spec.trainers.early_stopping import AccuracyStoppingSpec, EarlyStoppingSpec

if TYPE_CHECKING:
    from lofi._types import JsonValue


def test_the_closed_declaration_round_trips_exactly() -> None:
    document = (
        '{"metric": "loss/parser", "mode": "max", "patience": 4, '
        '"minimum_delta": 0.01, "keep_best": 2}'
    )
    declared = EarlyStoppingSpec.model_validate_json(document)

    assert declared.model_dump_json() == document
    assert EarlyStoppingSpec.model_validate_json(document) == declared


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("metric", ""),
        ("mode", "smaller"),
        ("patience", 0),
        ("patience", True),
        ("minimum_delta", -0.01),
        ("minimum_delta", float("inf")),
        ("keep_best", 0),
        ("keep_best", True),
    ],
)
def test_invalid_stopping_controls_are_refused(field: str, value: JsonValue) -> None:
    payload: dict[str, JsonValue] = {
        "metric": "loss/parser",
        "mode": "min",
        "patience": 3,
        "minimum_delta": 0.0,
        "keep_best": 1,
    }
    payload[field] = value

    with pytest.raises(dx.ValidationError, match=field):
        EarlyStoppingSpec.model_validate(payload)


def test_unknown_fields_are_refused() -> None:
    with pytest.raises(dx.ValidationError, match="relative_delta"):
        EarlyStoppingSpec.model_validate(
            {
                "metric": "loss/parser",
                "mode": "min",
                "patience": 3,
                "minimum_delta": 0.0,
                "keep_best": 1,
                "relative_delta": 0.1,
            }
        )


def test_an_accuracy_target_round_trips_exactly() -> None:
    document = (
        '{"metric": "loss/controller-supervision/accuracy", "target": 0.95, '
        '"max_steps": 40}'
    )
    declared = AccuracyStoppingSpec.model_validate_json(document)

    assert declared.model_dump_json() == document


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("metric", ""),
        ("target", -0.01),
        ("target", 1.01),
        ("target", float("inf")),
        ("max_steps", 0),
        ("max_steps", True),
    ],
)
def test_invalid_accuracy_targets_are_refused(field: str, value: JsonValue) -> None:
    payload: dict[str, JsonValue] = {
        "metric": "loss/controller-supervision/accuracy",
        "target": 0.95,
        "max_steps": 40,
    }
    payload[field] = value

    with pytest.raises(dx.ValidationError, match=field):
        AccuracyStoppingSpec.model_validate(payload)
