from typing import TYPE_CHECKING

import didactic.api as dx
import pytest

from lofi.spec.components.model import TermEncoderSpec
from lofi.spec.encoders.transformer import TransformerTermEncoderSpec

if TYPE_CHECKING:
    from lofi._types import JsonValue

_MINIMAL = {"width": 16, "heads": 2, "layers": 1, "feedforward_width": 32}


def test_the_tag_reaches_the_attending_variant_through_the_family_root() -> None:
    declared = TermEncoderSpec.model_validate(
        {"kind": "transformer", **_MINIMAL, "structure": "positions"}
    )

    assert type(declared) is TransformerTermEncoderSpec
    assert declared == TransformerTermEncoderSpec(
        width=16, heads=2, layers=1, feedforward_width=32, structure="positions"
    )
    assert TermEncoderSpec.__variants__["transformer"] is TransformerTermEncoderSpec


def test_the_attention_reads_the_ancestor_mask_unless_told_otherwise() -> None:
    declared = TransformerTermEncoderSpec(
        width=16, heads=2, layers=1, feedforward_width=32
    )

    assert declared.structure == "ancestor-mask"
    assert declared.norm_cap is None
    assert declared.dependencies() == ()


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
    ],
)
def test_an_out_of_range_or_unknown_setting_is_refused_at_the_boundary(
    payload: dict[str, JsonValue], detail: str
) -> None:
    with pytest.raises(dx.ValidationError, match=detail):
        TermEncoderSpec.model_validate({"kind": "transformer", **_MINIMAL, **payload})


def test_a_structure_name_is_open_to_installed_plugins() -> None:
    declared = TransformerTermEncoderSpec(
        width=16,
        heads=2,
        layers=1,
        feedforward_width=32,
        structure="sibling-mask",
    )

    assert declared.structure == "sibling-mask"


def test_the_variant_round_trips_through_its_json_form_byte_for_byte() -> None:
    document = (
        '{"kind": "transformer", "width": 16, "heads": 2, '
        '"layers": 1, "feedforward_width": 32, "structure": "ancestor-mask", '
        '"activation": "gelu", "norm_cap": null, "declared_output": null}'
    )
    declared = TermEncoderSpec.model_validate_json(document)

    assert declared.model_dump_json() == document
    assert TermEncoderSpec.model_validate_json(declared.model_dump_json()) == declared


def test_the_feedforward_activation_defaults_to_gelu_and_is_declared() -> None:
    default = TransformerTermEncoderSpec(
        width=16, heads=2, layers=1, feedforward_width=32
    )
    declared = TransformerTermEncoderSpec(
        width=16, heads=2, layers=1, feedforward_width=32, activation="silu"
    )

    assert default.settings().activation == "gelu"
    assert declared.settings().activation == "silu"
