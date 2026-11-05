"""Tests for the cosine scheduler declaration."""

from typing import TYPE_CHECKING

import didactic.api as dx
import pytest

from lofi.spec.components.trainer import SchedulerSpec
from lofi.spec.schedulers.cosine import CosineSchedulerSpec

if TYPE_CHECKING:
    from lofi._types import JsonValue


def test_the_cosine_tag_reaches_its_variant_through_the_family_root() -> None:
    declared = SchedulerSpec.model_validate({"kind": "cosine", "warmup_steps": 7})

    assert type(declared) is CosineSchedulerSpec
    assert declared == CosineSchedulerSpec(warmup_steps=7)
    assert SchedulerSpec.__variants__["cosine"] is CosineSchedulerSpec


def test_cosine_defaults_to_no_warmup() -> None:
    assert CosineSchedulerSpec().warmup_steps == 0


@pytest.mark.parametrize("value", [0.0, 1.5])
def test_a_warmup_start_outside_the_unit_interval_is_refused(value: float) -> None:
    with pytest.raises(dx.ValidationError, match="warmup_start_factor"):
        SchedulerSpec.model_validate({"kind": "cosine", "warmup_start_factor": value})


@pytest.mark.parametrize("value", [-1, True, 1.5, "3"])
def test_a_negative_or_noninteger_warmup_is_refused(value: JsonValue) -> None:
    with pytest.raises(dx.ValidationError, match="warmup_steps"):
        SchedulerSpec.model_validate({"kind": "cosine", "warmup_steps": value})


def test_an_extra_cosine_field_is_refused() -> None:
    with pytest.raises(dx.ValidationError, match="total_steps"):
        SchedulerSpec.model_validate(
            {"kind": "cosine", "warmup_steps": 0, "total_steps": 100}
        )


def test_the_cosine_variant_round_trips_through_json_byte_for_byte() -> None:
    document = (
        '{"kind": "cosine", "warmup_steps": 7, "warmup_start_factor": 0.5, '
        '"declared_output": null}'
    )
    declared = SchedulerSpec.model_validate_json(document)

    assert declared.model_dump_json() == document
    assert SchedulerSpec.model_validate_json(declared.model_dump_json()) == declared
