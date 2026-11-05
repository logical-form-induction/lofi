"""Tests for the constant scheduler declaration."""

import didactic.api as dx
import pytest

from lofi.spec.components.trainer import SchedulerSpec
from lofi.spec.schedulers.constant import (
    ConstantSchedulerSettings,
    ConstantSchedulerSpec,
)


def test_the_constant_tag_reaches_its_variant_through_the_family_root() -> None:
    declared = SchedulerSpec.model_validate({"kind": "constant", "warmup_steps": 4})

    assert type(declared) is ConstantSchedulerSpec
    assert declared.settings() == ConstantSchedulerSettings(
        warmup_steps=4, warmup_start_factor=1e-8
    )
    assert ConstantSchedulerSpec().settings() == ConstantSchedulerSettings(0, 1e-8)


@pytest.mark.parametrize(("field", "value"), [("warmup_steps", -1), ("factor", 0.5)])
def test_invalid_or_unknown_settings_are_refused(field: str, value: int) -> None:
    with pytest.raises(dx.ValidationError, match=field):
        SchedulerSpec.model_validate({"kind": "constant", field: value})
