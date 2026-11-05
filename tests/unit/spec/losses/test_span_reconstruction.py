from typing import TYPE_CHECKING

import didactic.api as dx
import pytest

from lofi.spec.components.loss import LossSpec
from lofi.spec.losses.span_reconstruction import (
    SpanNormalizationName,
    SpanReconstructionLossSpec,
    SpanWeightingName,
)

if TYPE_CHECKING:
    from lofi._types import JsonValue
    from lofi.spec.components.model import ModelSection


def test_the_tag_reaches_the_reconstruction_variant_through_the_family_root() -> None:
    declared = LossSpec.model_validate(
        {
            "kind": "span-reconstruction",
            "decoder": "reconstruction",
            "weighting": "marginal",
        }
    )

    assert type(declared) is SpanReconstructionLossSpec
    assert declared == SpanReconstructionLossSpec(
        decoder="reconstruction", weighting=SpanWeightingName.MARGINAL
    )
    assert LossSpec.__variants__["span-reconstruction"] is SpanReconstructionLossSpec
    assert declared.required_chart_outputs() == {"cells", "alpha", "mu"}


def test_spans_are_weighted_uniformly_by_default_and_declare_no_edge() -> None:
    declared = SpanReconstructionLossSpec(decoder="reconstruction")

    assert declared.weighting is SpanWeightingName.UNIFORM
    assert declared.max_span_length is None
    assert declared.normalization is SpanNormalizationName.SPAN
    assert declared.dependencies() == ()
    assert declared.declared_output is None
    assert declared.required_chart_outputs() == {"cells"}


@pytest.mark.parametrize(
    ("payload", "detail"),
    [
        ({"kind": "span-reconstruction"}, "decoder"),
        ({"kind": "span-reconstruction", "decoder": ""}, "span-decoder role"),
        ({"kind": "span-reconstruction", "decoder": 3}, "decoder"),
        (
            {
                "kind": "span-reconstruction",
                "decoder": "reconstruction",
                "weighting": "inside",
            },
            "weighting",
        ),
        (
            {"kind": "span-reconstruction", "decoder": "reconstruction", "end": "."},
            "end",
        ),
        (
            {
                "kind": "span-reconstruction",
                "decoder": "reconstruction",
                "max_span_length": 0,
            },
            "positive",
        ),
        (
            {
                "kind": "span-reconstruction",
                "decoder": "reconstruction",
                "normalization": "sentence",
            },
            "normalization",
        ),
    ],
)
def test_a_missing_empty_wrong_typed_unknown_or_extra_field_is_refused(
    payload: dict[str, JsonValue], detail: str
) -> None:
    with pytest.raises(dx.ValidationError, match=detail):
        LossSpec.model_validate(payload)


def test_the_variant_round_trips_through_its_json_form_byte_for_byte() -> None:
    document = (
        '{"kind": "span-reconstruction", "decoder": "reconstruction", '
        '"weighting": "uniform", "max_span_length": null, '
        '"normalization": "span", "declared_output": null}'
    )
    declared = LossSpec.model_validate_json(document)

    assert declared.model_dump_json() == document
    assert LossSpec.model_validate_json(declared.model_dump_json()) == declared


def test_span_range_and_normalization_are_declared() -> None:
    declared = SpanReconstructionLossSpec(
        decoder="reconstruction",
        max_span_length=2,
        normalization=SpanNormalizationName.POSITION,
    )

    assert declared.max_span_length == 2
    assert declared.normalization is SpanNormalizationName.POSITION


def test_the_loss_refuses_a_span_decoder_the_model_does_not_declare(
    packaged_model: ModelSection,
) -> None:
    SpanReconstructionLossSpec(decoder="reconstruction").refuse_model(
        packaged_model, "loss.terms.words"
    )

    with pytest.raises(
        ValueError,
        match=r"loss\.terms\.words\.decoder: the objective names span decoder "
        r"'words', which is not declared among \['reconstruction'\]",
    ):
        SpanReconstructionLossSpec(decoder="words").refuse_model(
            packaged_model, "loss.terms.words"
        )
