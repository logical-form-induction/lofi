"""Tests for the stochastic gradient descent declaration."""

import math
from typing import TYPE_CHECKING

import didactic.api as dx
import pytest

from lofi.spec.components.trainer import OptimizerSpec
from lofi.spec.optimizers.sgd import SGDOptimizerSettings, SGDOptimizerSpec

if TYPE_CHECKING:
    from lofi._types import JsonValue


def test_the_sgd_tag_reaches_its_variant_through_the_family_root() -> None:
    declared = OptimizerSpec.model_validate(
        {"kind": "sgd", "learning_rate": 0.1, "momentum": 0.9, "nesterov": True}
    )

    assert type(declared) is SGDOptimizerSpec
    assert declared == SGDOptimizerSpec(learning_rate=0.1, momentum=0.9, nesterov=True)
    assert OptimizerSpec.__variants__["sgd"] is SGDOptimizerSpec


def test_the_settings_restate_every_field() -> None:
    declared = SGDOptimizerSpec(momentum=0.5, dampening=0.1, weight_decay=0.01)

    assert declared.settings() == SGDOptimizerSettings(
        learning_rate=1e-2,
        momentum=0.5,
        dampening=0.1,
        weight_decay=0.01,
        nesterov=False,
    )


@pytest.mark.parametrize(
    ("document", "detail"),
    [
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"learning_rate": math.nan}, "learning_rate"),
        ({"momentum": -0.1}, "momentum"),
        ({"dampening": math.inf}, "dampening"),
        ({"weight_decay": -1.0}, "weight_decay"),
        ({"nesterov": True}, "nesterov"),
        ({"nesterov": True, "momentum": 0.9, "dampening": 0.1}, "nesterov"),
        ({"betas": [0.9, 0.99]}, "betas"),
    ],
)
def test_invalid_or_unknown_settings_are_refused(
    document: dict[str, JsonValue], detail: str
) -> None:
    with pytest.raises(dx.ValidationError, match=detail):
        OptimizerSpec.model_validate({"kind": "sgd", **document})
