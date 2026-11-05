"""Tests for the AdamW optimizer declaration."""

import math
from typing import TYPE_CHECKING

import didactic.api as dx
import pytest

from lofi.spec.components.trainer import OptimizerSpec
from lofi.spec.optimizers.adamw import AdamWOptimizerSpec

if TYPE_CHECKING:
    from lofi._types import JsonValue


def test_the_adamw_tag_reaches_its_variant_through_the_family_root() -> None:
    declared = OptimizerSpec.model_validate(
        {
            "kind": "adamw",
            "learning_rate": 0.02,
            "betas": [0.8, 0.95],
            "epsilon": 1e-6,
            "weight_decay": 0.04,
        }
    )

    assert type(declared) is AdamWOptimizerSpec
    assert declared == AdamWOptimizerSpec(
        learning_rate=0.02,
        betas=(0.8, 0.95),
        epsilon=1e-6,
        weight_decay=0.04,
    )
    assert OptimizerSpec.__variants__["adamw"] is AdamWOptimizerSpec


def test_adamw_defaults_match_the_runtime_optimizer_defaults() -> None:
    declared = AdamWOptimizerSpec()

    assert declared.learning_rate == 1e-3
    assert declared.betas == (0.9, 0.999)
    assert declared.epsilon == 1e-8
    assert declared.weight_decay == 0.01


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("learning_rate", 0.0),
        ("learning_rate", -0.1),
        ("learning_rate", math.inf),
        ("learning_rate", -math.inf),
        ("learning_rate", math.nan),
        ("learning_rate", True),
        ("epsilon", 0.0),
        ("epsilon", -1e-8),
        ("epsilon", math.inf),
        ("epsilon", -math.inf),
        ("epsilon", math.nan),
        ("epsilon", False),
        ("weight_decay", -0.01),
        ("weight_decay", math.inf),
        ("weight_decay", -math.inf),
        ("weight_decay", math.nan),
        ("weight_decay", True),
    ],
)
def test_invalid_scalar_hyperparameters_are_refused(
    field: str, value: JsonValue
) -> None:
    payload: dict[str, JsonValue] = {"kind": "adamw"}
    payload[field] = value

    with pytest.raises(dx.ValidationError, match=field):
        OptimizerSpec.model_validate(payload)


@pytest.mark.parametrize(
    "betas",
    [
        [],
        [0.9],
        [0.9, 0.99, 0.999],
        [-0.1, 0.9],
        [1.0, 0.9],
        [0.9, -0.1],
        [0.9, 1.0],
        [math.inf, 0.9],
        [0.9, -math.inf],
        [math.nan, 0.9],
        [0.9, math.nan],
        [True, 0.9],
        [0.9, False],
    ],
)
def test_adamw_requires_two_finite_beta_rates_below_one(
    betas: list[JsonValue],
) -> None:
    with pytest.raises(dx.ValidationError, match="betas"):
        OptimizerSpec.model_validate({"kind": "adamw", "betas": betas})


def test_an_extra_adamw_field_is_refused() -> None:
    with pytest.raises(dx.ValidationError, match="amsgrad"):
        OptimizerSpec.model_validate({"kind": "adamw", "amsgrad": True})


def test_the_adamw_variant_round_trips_through_json_byte_for_byte() -> None:
    document = (
        '{"kind": "adamw", "learning_rate": 0.02, "betas": [0.8, 0.95], '
        '"epsilon": 1e-06, "weight_decay": 0.04, "declared_output": null}'
    )
    declared = OptimizerSpec.model_validate_json(document)

    assert declared.model_dump_json() == document
    assert OptimizerSpec.model_validate_json(declared.model_dump_json()) == declared
