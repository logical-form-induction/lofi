import math
from typing import TYPE_CHECKING

import didactic.api as dx
import pytest

from lofi.spec.components.loss import LossSpec
from lofi.spec.losses.chart_scale import ChartScaleLossSpec, ScalePenaltyName

if TYPE_CHECKING:
    from lofi._types import JsonValue


def test_the_tag_reaches_the_chart_scale_variant_through_the_family_root() -> None:
    declared = LossSpec.model_validate({"kind": "chart-scale", "target": 2.0})

    assert type(declared) is ChartScaleLossSpec
    assert declared == ChartScaleLossSpec(
        target=2.0, penalty=ScalePenaltyName.SQUARED_RATIO
    )
    assert LossSpec.__variants__["chart-scale"] is ChartScaleLossSpec


def test_the_target_defaults_to_the_unit_norm_and_the_loss_declares_no_edge() -> None:
    declared = ChartScaleLossSpec()

    assert declared.target == 1.0
    assert declared.penalty is ScalePenaltyName.SQUARED_RATIO
    assert declared.dependencies() == ()
    assert declared.declared_output is None
    assert declared.required_chart_outputs() == {"cells"}


def test_the_penalty_form_is_declared() -> None:
    declared = LossSpec.model_validate(
        {"kind": "chart-scale", "target": 2.0, "penalty": "absolute-distance"}
    )

    assert declared.penalty is ScalePenaltyName.ABSOLUTE_DISTANCE


@pytest.mark.parametrize(
    ("payload", "detail"),
    [
        ({"kind": "chart-scale", "target": 0.0}, "positive length"),
        ({"kind": "chart-scale", "target": -1.0}, "positive length"),
        ({"kind": "chart-scale", "target": math.inf}, "not finite"),
        ({"kind": "chart-scale", "target": math.nan}, "not finite"),
        ({"kind": "chart-scale", "target": True}, "target"),
        ({"kind": "chart-scale", "over": "width"}, "over"),
        ({"kind": "chart-scale", "penalty": "hinge"}, "penalty"),
    ],
)
def test_a_nonpositive_nonfinite_wrong_typed_or_extra_field_is_refused(
    payload: dict[str, JsonValue], detail: str
) -> None:
    with pytest.raises(dx.ValidationError, match=detail):
        LossSpec.model_validate(payload)


def test_the_variant_round_trips_through_its_json_form_byte_for_byte() -> None:
    document = (
        '{"kind": "chart-scale", "target": 2.0, '
        '"penalty": "squared-ratio", "declared_output": null}'
    )
    declared = LossSpec.model_validate_json(document)

    assert declared.model_dump_json() == document
    assert LossSpec.model_validate_json(declared.model_dump_json()) == declared
