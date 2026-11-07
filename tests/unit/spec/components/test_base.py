from typing import Literal

import didactic.api as dx
import pytest
from tests.unit.spec.components.conftest import SilentParser, SizedEncoder, minimal

from lofi.spec.components.data import ConstraintSourceSpec
from lofi.spec.components.model import (
    GroundingSpec,
    ParserSpec,
    ReadoutSpec,
    SubwordPoolerSpec,
    TermDecoderSpec,
    TermEncoderSpec,
    TokenEncoderSpec,
)
from lofi.spec.components.trainer import TypeSamplerSpec
from lofi.spec.outputs import DeclaredOutput, DtypeClass
from lofi.spec.provers import ProverSpec


def test_a_variant_is_reached_through_its_family_root() -> None:
    encoder = TermEncoderSpec.model_validate({"kind": "sized", "width": 12})

    assert encoder == SizedEncoder(width=12)


def test_a_consumer_reads_the_width_the_producer_derives() -> None:
    output = SizedEncoder(width=48).declared_output

    assert output == DeclaredOutput(width=48, rank=2, dtype_class=DtypeClass.REAL)


def test_the_declared_output_follows_the_field_it_is_derived_from() -> None:
    narrowed = SizedEncoder(width=48).with_(width=7)
    output = narrowed.declared_output

    assert output is not None
    assert output.width == 7


def test_a_component_with_no_declared_width_says_so() -> None:
    assert SilentParser().declared_output is None


@pytest.mark.parametrize(
    "family",
    [
        GroundingSpec,
        ProverSpec,
        ConstraintSourceSpec,
        TypeSamplerSpec,
        ReadoutSpec,
        TermDecoderSpec,
        TokenEncoderSpec,
        SubwordPoolerSpec,
    ],
    ids=lambda family: family.__name__,
)
def test_an_unimplemented_family_has_no_constructible_kind(
    family: type[dx.TaggedUnion],
) -> None:
    with pytest.raises(dx.ValidationError):
        family.model_validate({"kind": "unimplemented"})


def test_an_unregistered_kind_is_refused() -> None:
    with pytest.raises(dx.ValidationError):
        TermEncoderSpec.model_validate({"kind": "sized-encoder"})


def test_one_family_does_not_accept_another_family_variant() -> None:
    with pytest.raises(dx.ValidationError):
        ParserSpec.model_validate({"kind": "sized", "width": 12})


def test_an_unknown_setting_is_refused() -> None:
    with pytest.raises(dx.ValidationError):
        TermEncoderSpec.model_validate({"kind": "sized", "widht": 12})


def test_a_second_variant_claiming_a_tag_is_refused_at_class_creation() -> None:
    with pytest.raises(TypeError):

        class Rival(TermEncoderSpec):
            kind: Literal["sized"] = "sized"

    assert TermEncoderSpec.__variants__["sized"] is SizedEncoder


def test_a_component_a_run_fills_a_slot_with_declares_nothing_of_itself() -> None:
    parser = minimal().model.parser

    assert parser is not None
    assert parser.declared_output is None


def test_a_component_names_no_pretrained_weights_unless_it_says_so() -> None:
    assert SizedEncoder(width=12).pretrained_weights() is None


def test_a_component_contributes_checkpoint_state_unless_it_says_otherwise() -> None:
    assert SizedEncoder(width=12).checkpointed()
