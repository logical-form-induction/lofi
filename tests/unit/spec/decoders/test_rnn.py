from typing import Final, Literal

import didactic.api as dx
import pytest

from lofi.spec.components.model import DecoderCapability, TermDecoderSpec
from lofi.spec.decoders.rnn import (
    RnnTermDecoderSpec,
    TypeDirectedSettings,
    TypeDirectedSpec,
)


def test_the_tag_reaches_the_recursive_variant_through_the_family_root() -> None:
    declared = TermDecoderSpec.model_validate({"kind": "tree-rnn", "width": 16})

    assert type(declared) is RnnTermDecoderSpec
    assert declared == RnnTermDecoderSpec(width=16)
    assert TermDecoderSpec.__variants__["tree-rnn"] is RnnTermDecoderSpec


def test_the_decoder_leaves_its_encoder_binding_to_the_represented_space() -> None:
    assert RnnTermDecoderSpec(width=16).dependencies() == ()


def test_the_decoder_holds_no_submodule_of_the_encoder_it_reads() -> None:
    assert RnnTermDecoderSpec(width=16).sharing() == {}


def test_the_recursive_decoder_declares_its_three_protocols() -> None:
    assert RnnTermDecoderSpec.capabilities == frozenset(
        {
            DecoderCapability.FRONTIER,
            DecoderCapability.TEACHER_FORCED,
            DecoderCapability.MARGINAL,
        }
    )


def test_type_direction_is_disabled_until_a_document_declares_it() -> None:
    declared = RnnTermDecoderSpec(width=16)

    assert declared.settings().type_directed == TypeDirectedSettings("none", "none")


@pytest.mark.parametrize("width", [0, -4])
def test_a_state_of_no_coordinates_is_refused_at_the_boundary(width: int) -> None:
    with pytest.raises(dx.ValidationError, match="a state of no coordinates"):
        TermDecoderSpec.model_validate({"kind": "tree-rnn", "width": width})


def test_the_variant_round_trips_through_its_json_form_byte_for_byte() -> None:
    document = (
        '{"kind": "tree-rnn", "width": 16, '
        '"type_directed": {"prefix": "type-grammar", "mask": "known"}, '
        '"choice_heads": "factored", "declared_output": null}'
    )
    declared = TermDecoderSpec.model_validate_json(document)

    assert declared.model_dump_json() == document
    assert TermDecoderSpec.model_validate_json(declared.model_dump_json()) == declared


def test_type_direction_is_declared_and_resolves_to_one_frozen_policy() -> None:
    declared = RnnTermDecoderSpec(
        width=16,
        type_directed=TypeDirectedSpec(prefix="none", mask="none"),
    )

    assert declared.settings().type_directed == TypeDirectedSettings("none", "none")


_HEADS_DOCUMENT: Final = {"width": 8}


@pytest.mark.parametrize("choice_heads", ["flat", "factored"])
def test_the_choice_heads_are_declared(
    choice_heads: Literal["flat", "factored"],
) -> None:
    declared = RnnTermDecoderSpec.model_validate(
        {**_HEADS_DOCUMENT, "choice_heads": choice_heads}
    )

    assert declared.settings().choice_heads == choice_heads
    assert (
        RnnTermDecoderSpec.model_validate(_HEADS_DOCUMENT).settings().choice_heads
        == "flat"
    )
