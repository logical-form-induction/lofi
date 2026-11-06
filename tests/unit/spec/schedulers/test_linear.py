"""Tests for the linear scheduler declaration."""

import math
from typing import TYPE_CHECKING

import didactic.api as dx
import pytest

from lofi.spec.components.trainer import SchedulerSpec
from lofi.spec.schedulers.linear import LinearSchedulerSettings, LinearSchedulerSpec

if TYPE_CHECKING:
    from lofi._types import JsonValue


def test_the_linear_tag_reaches_its_variant_through_the_family_root() -> None:
    declared = SchedulerSpec.model_validate(
        {
            "kind": "linear",
            "warmup_steps": 3,
            "warmup_start_factor": 0.25,
            "final_factor": 0.1,
        }
    )

    assert type(declared) is LinearSchedulerSpec
    assert declared.settings() == LinearSchedulerSettings(
        warmup_steps=3, warmup_start_factor=0.25, final_factor=0.1
    )
    assert LinearSchedulerSpec().settings() == LinearSchedulerSettings(0, 1e-8, 0.0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("warmup_steps", -1),
        ("warmup_start_factor", 0.0),
        ("warmup_start_factor", 1.5),
        ("final_factor", -0.1),
        ("final_factor", 1.5),
        ("final_factor", math.nan),
        ("cycles", 2),
    ],
)
def test_invalid_or_unknown_settings_are_refused(field: str, value: JsonValue) -> None:
    with pytest.raises(dx.ValidationError, match=field):
        SchedulerSpec.model_validate({"kind": "linear", field: value})
