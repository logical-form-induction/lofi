from typing import TYPE_CHECKING, Final, Literal

import didactic.api as dx
import pytest

from lofi._types import ModuleSharing, SubmoduleDeclaration
from lofi.spec.components.model import DecoderCapability, TermDecoderSpec
from lofi.spec.decoders.linear import LinearRnnDecoderSettings, LinearRnnDecoderSpec

if TYPE_CHECKING:
    from lofi._types import JsonValue


def test_the_linear_recurrence_is_one_registered_decoder_variant() -> None:
    declared = TermDecoderSpec.model_validate(
        {
            "kind": "linear-rnn",
            "width": 12,
            "cell": "elman-relu",
            "layers": 2,
            "choice_heads": "factored",
            "tie": "terms",
        }
    )

    assert type(declared) is LinearRnnDecoderSpec
    assert declared.settings() == LinearRnnDecoderSettings(
        12, "elman-relu", 2, "factored", "terms"
    )
    assert declared.sharing() == {
        "encoder": SubmoduleDeclaration(
            "choice-embedding",
            ModuleSharing(in_parameters=True, in_state_dict=True),
        )
    }


def test_the_linear_decoder_defaults_are_an_untied_flat_gru() -> None:
    declared = LinearRnnDecoderSpec(width=8)

    assert declared.settings() == LinearRnnDecoderSettings(8, "gru", 1, "flat", None)
    assert declared.sharing() == {}


def test_the_linear_decoder_declares_frontier_and_teacher_forcing() -> None:
    assert LinearRnnDecoderSpec.capabilities == frozenset(
        {DecoderCapability.FRONTIER, DecoderCapability.TEACHER_FORCED}
    )


@pytest.mark.parametrize(
    ("changed", "detail"),
    [
        ({"width": 0}, "state of no coordinates"),
        ({"layers": 0}, "recurrence has a layer"),
        ({"tie": ""}, "tie"),
        ({"cell": ""}, "recurrent cell"),
    ],
)
def test_invalid_linear_decoder_settings_are_refused(
    changed: dict[str, JsonValue], detail: str
) -> None:
    with pytest.raises(dx.ValidationError, match=detail):
        LinearRnnDecoderSpec.model_validate({"width": 8, **changed})


_HEADS_DOCUMENT: Final = {"width": 8}


@pytest.mark.parametrize("choice_heads", ["flat", "factored"])
def test_the_choice_heads_are_declared(
    choice_heads: Literal["flat", "factored"],
) -> None:
    declared = LinearRnnDecoderSpec.model_validate(
        {**_HEADS_DOCUMENT, "choice_heads": choice_heads}
    )

    assert declared.settings().choice_heads == choice_heads
    assert (
        LinearRnnDecoderSpec.model_validate(_HEADS_DOCUMENT).settings().choice_heads
        == "flat"
    )
