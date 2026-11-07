"""Tests for the Adam optimizer declaration."""

import math
from typing import TYPE_CHECKING

import didactic.api as dx
import pytest

from lofi.spec.components.trainer import OptimizerSpec
from lofi.spec.optimizers.adam import AdamOptimizerSettings, AdamOptimizerSpec

if TYPE_CHECKING:
    from lofi._types import JsonValue


def test_the_adam_tag_reaches_its_variant_through_the_family_root() -> None:
    declared = OptimizerSpec.model_validate(
        {"kind": "adam", "learning_rate": 0.02, "betas": [0.8, 0.95]}
    )

    assert type(declared) is AdamOptimizerSpec
    assert declared == AdamOptimizerSpec(learning_rate=0.02, betas=(0.8, 0.95))
    assert OptimizerSpec.__variants__["adam"] is AdamOptimizerSpec


def test_the_settings_restate_every_field() -> None:
    declared = AdamOptimizerSpec(epsilon=1e-6, weight_decay=0.1)

    assert declared.settings() == AdamOptimizerSettings(
        learning_rate=1e-3, betas=(0.9, 0.999), epsilon=1e-6, weight_decay=0.1
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("learning_rate", 0.0),
        ("learning_rate", math.inf),
        ("epsilon", -1e-8),
        ("weight_decay", -0.1),
        ("betas", [0.9]),
        ("betas", [1.0, 0.9]),
        ("momentum", 0.9),
    ],
)
def test_invalid_or_unknown_settings_are_refused(field: str, value: JsonValue) -> None:
    with pytest.raises(dx.ValidationError):
        OptimizerSpec.model_validate({"kind": "adam", field: value})
