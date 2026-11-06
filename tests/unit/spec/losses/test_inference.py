import math
from typing import Literal

import didactic.api as dx
import pytest

from lofi.spec.components.loss import LossSpec
from lofi.spec.losses.inference import (
    InferenceLossSpec,
    LabelResponseSpec,
)


def test_the_inference_variant_is_reached_through_the_loss_family() -> None:
    declared = LossSpec.model_validate(
        {
            "kind": "inference",
            "signal": "likelihood",
            "response": {"kind": "label"},
        }
    )

    assert declared == InferenceLossSpec(
        signal="likelihood",
        response=LabelResponseSpec(),
    )


@pytest.mark.parametrize("signal", ["likelihood", "reward"])
def test_each_signal_round_trips_without_a_default(
    signal: Literal["likelihood", "reward"],
) -> None:
    declared = InferenceLossSpec(signal=signal, response=LabelResponseSpec())

    assert LossSpec.model_validate_json(declared.model_dump_json()) == declared
    assert declared.settings().signal == signal


def test_signal_and_response_are_required() -> None:
    with pytest.raises(dx.ValidationError, match="signal"):
        LossSpec.model_validate({"kind": "inference", "response": {"kind": "label"}})

    with pytest.raises(dx.ValidationError, match="response"):
        LossSpec.model_validate({"kind": "inference", "signal": "likelihood"})


@pytest.mark.parametrize("weight", [-0.1, math.inf, -math.inf, math.nan, True])
def test_auxiliary_weight_is_finite_and_nonnegative(weight: float) -> None:
    with pytest.raises(dx.ValidationError, match=r"auxiliary weight|expected float"):
        InferenceLossSpec(
            signal="likelihood",
            response=LabelResponseSpec(),
            auxiliary_weight=weight,
        )


def test_no_superseded_reward_or_grounding_fields_are_accepted() -> None:
    for field in ("grounding", "prover", "budget", "timeout"):
        with pytest.raises(dx.ValidationError, match=field):
            LossSpec.model_validate(
                {
                    "kind": "inference",
                    "signal": "likelihood",
                    "response": {"kind": "label"},
                    field: 1,
                }
            )
