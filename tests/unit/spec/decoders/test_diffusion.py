from typing import TYPE_CHECKING, Final, Literal

import didactic.api as dx
import pytest

from lofi.spec.components.model import DecoderCapability, TermDecoderSpec
from lofi.spec.decoders.diffusion import DiffusionTermDecoderSpec

if TYPE_CHECKING:
    from lofi._types import JsonValue

_MINIMAL = {
    "width": 16,
    "layers": 1,
    "feedforward_width": 32,
    "noise_steps": 4,
    "first_masking": 0.2,
}


def test_the_tag_reaches_the_diffusion_variant_through_the_family_root() -> None:
    declared = TermDecoderSpec.model_validate({"kind": "diffusion", **_MINIMAL})
    assert type(declared) is DiffusionTermDecoderSpec
    assert declared == DiffusionTermDecoderSpec(
        width=16, layers=1, feedforward_width=32, noise_steps=4, first_masking=0.2
    )
    assert declared.dependencies() == ()
    assert TermDecoderSpec.__variants__["diffusion"] is DiffusionTermDecoderSpec


def test_the_decoder_declares_only_frontier_and_teacher_forcing() -> None:
    assert DiffusionTermDecoderSpec.capabilities == frozenset(
        {DecoderCapability.FRONTIER, DecoderCapability.TEACHER_FORCED}
    )


@pytest.mark.parametrize(
    ("payload", "detail"),
    [
        ({"width": 0}, "a state of no coordinates holds nothing"),
        ({"layers": 0}, "a stack has a layer"),
        ({"feedforward_width": 0}, "a block has a hidden coordinate"),
        ({"noise_steps": 0}, "a schedule has a step"),
        ({"first_masking": 0.0}, r"a masking fraction lies in \(0, 1\]"),
        ({"first_masking": 1.5}, r"a masking fraction lies in \(0, 1\]"),
    ],
)
def test_an_out_of_range_dimension_is_refused_at_the_boundary(
    payload: dict[str, JsonValue], detail: str
) -> None:
    with pytest.raises(dx.ValidationError, match=detail):
        TermDecoderSpec.model_validate({"kind": "diffusion", **_MINIMAL, **payload})


def test_a_schedule_of_one_step_masks_every_slot_on_that_step() -> None:
    declared = DiffusionTermDecoderSpec(
        width=16, layers=1, feedforward_width=32, noise_steps=1, first_masking=1.0
    )

    assert declared.noise_steps == 1
    assert declared.first_masking == 1.0


def test_the_variant_round_trips_through_its_json_form_byte_for_byte() -> None:
    document = (
        '{"kind": "diffusion", "width": 16, "layers": 1, '
        '"feedforward_width": 32, "noise_steps": 4, "first_masking": 0.2, '
        '"choice_heads": "factored", "activation": "gelu", "declared_output": null}'
    )
    declared = TermDecoderSpec.model_validate_json(document)

    assert declared.model_dump_json() == document
    assert TermDecoderSpec.model_validate_json(declared.model_dump_json()) == declared


_HEADS_DOCUMENT: Final = {
    "width": 8,
    "layers": 1,
    "feedforward_width": 16,
    "noise_steps": 2,
    "first_masking": 0.5,
}


@pytest.mark.parametrize("choice_heads", ["flat", "factored"])
def test_the_choice_heads_are_declared(
    choice_heads: Literal["flat", "factored"],
) -> None:
    declared = DiffusionTermDecoderSpec.model_validate(
        {**_HEADS_DOCUMENT, "choice_heads": choice_heads}
    )

    assert declared.settings().choice_heads == choice_heads
    assert (
        DiffusionTermDecoderSpec.model_validate(_HEADS_DOCUMENT).settings().choice_heads
        == "flat"
    )
