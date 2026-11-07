"""Tests for the checkpoint policy declaration."""

from typing import TYPE_CHECKING

import didactic.api as dx
import pytest

from lofi.spec.trainers.checkpoints import CheckpointPolicySpec

if TYPE_CHECKING:
    from lofi._types import JsonValue


def test_the_declaration_round_trips_exactly() -> None:
    document = '{"interval": 3, "keep": 2}'
    declared = CheckpointPolicySpec.model_validate_json(document)

    assert declared == CheckpointPolicySpec(interval=3, keep=2)
    assert declared.model_dump_json() == document


@pytest.mark.parametrize("field", ["interval", "keep"])
def test_each_setting_is_required(field: str) -> None:
    document: dict[str, JsonValue] = {"interval": 3, "keep": 2}
    del document[field]

    with pytest.raises(dx.ValidationError, match=field):
        CheckpointPolicySpec.model_validate(document)


@pytest.mark.parametrize(
    ("field", "value"),
    [("interval", 0), ("interval", -1), ("keep", 0), ("keep", True)],
)
def test_a_nonpositive_count_is_refused(field: str, value: JsonValue) -> None:
    document: dict[str, JsonValue] = {"interval": 3, "keep": 2}
    document[field] = value

    with pytest.raises(dx.ValidationError, match=field):
        CheckpointPolicySpec.model_validate(document)


def test_an_unknown_setting_is_refused() -> None:
    with pytest.raises(dx.ValidationError, match="keep_last"):
        CheckpointPolicySpec.model_validate({"interval": 3, "keep": 2, "keep_last": 1})
