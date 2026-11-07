from typing import TYPE_CHECKING

import didactic.api as dx
import pytest

from lofi.spec.components.model import TermEncoderSpec
from lofi.spec.encoders.hrr import HrrTermEncoderSpec

if TYPE_CHECKING:
    from lofi._types import JsonValue


def test_the_tag_reaches_the_holographic_variant_through_the_family_root() -> None:
    declared = TermEncoderSpec.model_validate(
        {"kind": "hrr", "width": 16, "magnitude": 2.0, "floor": 0.5}
    )

    assert type(declared) is HrrTermEncoderSpec
    assert declared == HrrTermEncoderSpec(width=16, magnitude=2.0, floor=0.5)
    assert TermEncoderSpec.__variants__["hrr"] is HrrTermEncoderSpec


def test_an_atom_starts_at_unit_magnitude_and_is_read_above_a_small_floor() -> None:
    declared = HrrTermEncoderSpec(width=16)

    assert declared.magnitude == 1.0
    assert declared.floor < declared.magnitude
    assert declared.dependencies() == ()


def test_the_encoder_lends_its_codebook_to_the_components_it_supplies() -> None:
    assert HrrTermEncoderSpec(width=16).publications() == {"codebook"}


@pytest.mark.parametrize(
    ("payload", "detail"),
    [
        ({"width": 0}, "an atom of no coordinates holds nothing"),
        ({"width": -4}, "an atom of no coordinates holds nothing"),
        ({"width": 16, "magnitude": 0.0}, "an atom starts away from the origin"),
        ({"width": 16, "floor": 0.0}, "a floor is above zero"),
        (
            {"width": 16, "magnitude": 0.5, "floor": 0.75},
            "a floor is no larger than the magnitude it bounds",
        ),
    ],
)
def test_an_out_of_range_dimension_is_refused_at_the_boundary(
    payload: dict[str, JsonValue], detail: str
) -> None:
    with pytest.raises(dx.ValidationError, match=detail):
        TermEncoderSpec.model_validate({"kind": "hrr", **payload})


def test_the_variant_round_trips_through_its_json_form_byte_for_byte() -> None:
    document = (
        '{"kind": "hrr", "width": 16, "magnitude": 1.0, '
        '"floor": 0.001, "declared_output": null}'
    )
    declared = TermEncoderSpec.model_validate_json(document)

    assert declared.model_dump_json() == document
    assert TermEncoderSpec.model_validate_json(declared.model_dump_json()) == declared
