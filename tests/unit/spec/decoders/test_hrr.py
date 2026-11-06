from typing import Final, Literal

import didactic.api as dx
import pytest

from lofi._types import ModuleSharing, SubmoduleDeclaration
from lofi.spec.components.model import DecoderCapability, TermDecoderSpec
from lofi.spec.decoders.hrr import HrrTermDecoderSpec
from lofi.terms import (
    CompleteTree,
    Operator,
    OperatorIndex,
    PreorderArityLayout,
    Signature,
    TermCodec,
)


def test_the_tag_reaches_the_holographic_variant_through_the_family_root() -> None:
    declared = TermDecoderSpec.model_validate({"kind": "hrr"})

    assert type(declared) is HrrTermDecoderSpec
    assert declared == HrrTermDecoderSpec()
    assert TermDecoderSpec.__variants__["hrr"] is HrrTermDecoderSpec


def test_the_decoder_leaves_its_encoder_binding_to_the_represented_space() -> None:
    assert HrrTermDecoderSpec().dependencies() == ()


def test_the_decoder_declares_that_it_trains_and_saves_the_encoder_codebook() -> None:
    assert HrrTermDecoderSpec().sharing() == {
        "encoder": SubmoduleDeclaration(
            "codebook", ModuleSharing(in_parameters=True, in_state_dict=True)
        )
    }


def test_the_decoder_declares_teacher_forcing_for_every_layout() -> None:
    assert HrrTermDecoderSpec.capabilities == frozenset(
        {DecoderCapability.TEACHER_FORCED}
    )


def test_the_decoder_adds_path_dependent_capabilities_for_a_fixed_layout() -> None:
    signature = Signature(
        "hrr-capabilities",
        ("T",),
        (
            Operator("atom", "T"),
            Operator("pair", "T", arg_sorts=("T", "T")),
        ),
    )
    index = OperatorIndex(signature)
    fixed = TermCodec(
        index,
        CompleteTree(signature, signature.operators, order="df", depth=2),
    )
    preorder = TermCodec(index, PreorderArityLayout(3))
    declared = HrrTermDecoderSpec()

    assert declared.capabilities_for(preorder) == frozenset(
        {DecoderCapability.TEACHER_FORCED}
    )
    assert declared.capabilities_for(fixed) == frozenset(
        {
            DecoderCapability.FRONTIER,
            DecoderCapability.TEACHER_FORCED,
            DecoderCapability.MARGINAL,
        }
    )


def test_a_width_is_not_declared_here_because_the_codebook_fixes_it() -> None:
    with pytest.raises(dx.ValidationError, match="width"):
        TermDecoderSpec.model_validate({"kind": "hrr", "width": 16})


def test_the_variant_round_trips_through_its_json_form_byte_for_byte() -> None:
    document = '{"kind": "hrr", "choice_heads": "factored", "declared_output": null}'
    declared = TermDecoderSpec.model_validate_json(document)

    assert declared.model_dump_json() == document
    assert TermDecoderSpec.model_validate_json(declared.model_dump_json()) == declared


_HEADS_DOCUMENT: Final = {}


@pytest.mark.parametrize("choice_heads", ["flat", "factored"])
def test_the_choice_heads_are_declared(
    choice_heads: Literal["flat", "factored"],
) -> None:
    declared = HrrTermDecoderSpec.model_validate(
        {**_HEADS_DOCUMENT, "choice_heads": choice_heads}
    )

    assert declared.settings().choice_heads == choice_heads
    assert (
        HrrTermDecoderSpec.model_validate(_HEADS_DOCUMENT).settings().choice_heads
        == "flat"
    )
