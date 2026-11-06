from typing import TYPE_CHECKING

import didactic.api as dx
import pytest

from lofi.spec.components.loss import LossSpec
from lofi.spec.losses.outside_reconstruction import OutsideReconstructionLossSpec

if TYPE_CHECKING:
    from lofi._types import JsonValue
    from lofi.spec.components.model import ModelSection


def test_the_tag_reaches_the_outside_reconstruction_variant() -> None:
    declared = LossSpec.model_validate(
        {"kind": "outside-reconstruction", "decoder": "reconstruction"}
    )

    assert type(declared) is OutsideReconstructionLossSpec
    assert declared == OutsideReconstructionLossSpec(decoder="reconstruction")
    assert (
        LossSpec.__variants__["outside-reconstruction"] is OutsideReconstructionLossSpec
    )


def test_the_declaration_names_no_component_dependency_or_output_edge() -> None:
    declared = OutsideReconstructionLossSpec(decoder="reconstruction")

    assert declared.dependencies() == ()
    assert declared.declared_output is None
    assert declared.settings().decoder == "reconstruction"
    assert declared.required_chart_outputs() == {"outside_cells"}


@pytest.mark.parametrize(
    ("payload", "detail"),
    [
        ({"kind": "outside-reconstruction"}, "decoder"),
        ({"kind": "outside-reconstruction", "decoder": ""}, "span-decoder role"),
        ({"kind": "outside-reconstruction", "decoder": 3}, "decoder"),
        (
            {
                "kind": "outside-reconstruction",
                "decoder": "reconstruction",
                "weighting": "uniform",
            },
            "weighting",
        ),
    ],
)
def test_a_missing_empty_wrong_typed_or_extra_field_is_refused(
    payload: dict[str, JsonValue], detail: str
) -> None:
    with pytest.raises(dx.ValidationError, match=detail):
        LossSpec.model_validate(payload)


def test_the_variant_round_trips_through_its_json_form_byte_for_byte() -> None:
    document = (
        '{"kind": "outside-reconstruction", "decoder": "reconstruction", '
        '"declared_output": null}'
    )
    declared = LossSpec.model_validate_json(document)

    assert declared.model_dump_json() == document
    assert LossSpec.model_validate_json(declared.model_dump_json()) == declared


def test_the_loss_refuses_a_span_decoder_the_model_does_not_declare(
    packaged_model: ModelSection,
) -> None:
    OutsideReconstructionLossSpec(decoder="reconstruction").refuse_model(
        packaged_model, "loss.terms.words"
    )

    with pytest.raises(
        ValueError,
        match=r"loss\.terms\.words\.decoder: the objective names span decoder "
        r"'words', which is not declared among \['reconstruction'\]",
    ):
        OutsideReconstructionLossSpec(decoder="words").refuse_model(
            packaged_model, "loss.terms.words"
        )
