from typing import TYPE_CHECKING, Final, Literal

import didactic.api as dx
import pytest

from lofi.spec.components.model import DecoderCapability, TermDecoderSpec
from lofi.spec.decoders.transformer import TransformerTermDecoderSpec

if TYPE_CHECKING:
    from lofi._types import JsonValue

_MINIMAL = {"width": 16, "heads": 2, "layers": 1, "feedforward_width": 32}


def test_the_tag_reaches_the_attending_variant_through_the_family_root() -> None:
    declared = TermDecoderSpec.model_validate(
        {
            "kind": "transformer",
            **_MINIMAL,
            "structure": "positions",
        }
    )

    assert type(declared) is TransformerTermDecoderSpec
    assert declared == TransformerTermDecoderSpec(
        width=16,
        heads=2,
        layers=1,
        feedforward_width=32,
        structure="positions",
    )
    assert declared.dependencies() == ()
    assert declared.settings().structure == "positions"
    assert TermDecoderSpec.__variants__["transformer"] is TransformerTermDecoderSpec


def test_the_attention_structure_is_required() -> None:
    with pytest.raises(dx.ValidationError, match="structure"):
        TermDecoderSpec.model_validate({"kind": "transformer", **_MINIMAL})


def test_the_decoder_declares_its_three_choice_capabilities() -> None:
    assert TransformerTermDecoderSpec.capabilities == frozenset(
        {
            DecoderCapability.FRONTIER,
            DecoderCapability.TEACHER_FORCED,
            DecoderCapability.MARGINAL,
        }
    )


@pytest.mark.parametrize(
    ("payload", "detail"),
    [
        ({"width": 0}, "a state of no coordinates holds nothing"),
        ({"heads": 0}, "an attention layer has a head"),
        ({"layers": 0}, "a stack has a layer"),
        ({"feedforward_width": 0}, "a block has a hidden coordinate"),
        (
            {"width": 18, "heads": 4},
            "the heads divide the width they split between them",
        ),
        ({"structure": ""}, "structure"),
    ],
)
def test_an_out_of_range_dimension_is_refused_at_the_boundary(
    payload: dict[str, JsonValue], detail: str
) -> None:
    with pytest.raises(dx.ValidationError, match=detail):
        TermDecoderSpec.model_validate(
            {
                "kind": "transformer",
                **_MINIMAL,
                "structure": "parent-recurrence",
                **payload,
            }
        )


def test_the_variant_round_trips_through_its_json_form_byte_for_byte() -> None:
    document = (
        '{"kind": "transformer", "width": 16, "heads": 2, '
        '"layers": 1, "feedforward_width": 32, '
        '"structure": "ancestor-and-sibling-bias", "choice_heads": "factored", '
        '"activation": "gelu", "declared_output": null}'
    )
    declared = TermDecoderSpec.model_validate_json(document)

    assert declared.model_dump_json() == document
    assert TermDecoderSpec.model_validate_json(declared.model_dump_json()) == declared


def test_the_structure_names_an_installed_plugin_rather_than_a_closed_list() -> None:
    declared = TermDecoderSpec.model_validate(
        {"kind": "transformer", **_MINIMAL, "structure": "my-structure"}
    )

    assert isinstance(declared, TransformerTermDecoderSpec)
    assert declared.settings().structure == "my-structure"


_HEADS_DOCUMENT: Final = {
    "width": 16,
    "heads": 2,
    "layers": 1,
    "feedforward_width": 32,
    "structure": "positions",
}


@pytest.mark.parametrize("choice_heads", ["flat", "factored"])
def test_the_choice_heads_are_declared(
    choice_heads: Literal["flat", "factored"],
) -> None:
    declared = TransformerTermDecoderSpec.model_validate(
        {**_HEADS_DOCUMENT, "choice_heads": choice_heads}
    )

    assert declared.settings().choice_heads == choice_heads
    assert (
        TransformerTermDecoderSpec.model_validate(_HEADS_DOCUMENT)
        .settings()
        .choice_heads
        == "flat"
    )
