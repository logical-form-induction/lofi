from typing import TYPE_CHECKING

import didactic.api as dx
import pytest

from lofi.spec.artifacts import HubSourceSpec
from lofi.spec.components.model import TokenEncoderSpec
from lofi.spec.tokens.pretrained import (
    ContextWindowSpec,
    FixedLayerReadoutSpec,
    LastLayerReadoutSpec,
    PretrainedTokenEncoderSpec,
    ScalarMixReadoutSpec,
)

if TYPE_CHECKING:
    from lofi._types import JsonValue

REVISION = "0123456789abcdef" * 2 + "01234567"


def test_the_tag_reaches_the_pretrained_variant_through_the_family_root() -> None:
    declared = TokenEncoderSpec.model_validate(
        {
            "kind": "pretrained",
            "checkpoint": {
                "kind": "hub",
                "repository": "an-org/a-model",
                "revision": REVISION,
            },
            "architecture": "t5-encoder",
            "trainable": True,
            "window": {"rows": 64, "overflow": "refuse"},
        }
    )

    assert type(declared) is PretrainedTokenEncoderSpec
    assert declared == PretrainedTokenEncoderSpec(
        checkpoint=HubSourceSpec(repository="an-org/a-model", revision=REVISION),
        architecture="t5-encoder",
        trainable=True,
        window=ContextWindowSpec(rows=64, overflow="refuse"),
    )
    assert TokenEncoderSpec.__variants__["pretrained"] is PretrainedTokenEncoderSpec
    assert declared.settings().architecture == "t5-encoder"


def test_an_encoder_is_frozen_and_unbounded_unless_the_declaration_says_so() -> None:
    declared = PretrainedTokenEncoderSpec(
        checkpoint=HubSourceSpec(repository="an-org/a-model", revision=REVISION)
    )

    assert declared.architecture == "encoder-only"
    assert declared.settings().architecture == "encoder-only"
    assert declared.trainable is False
    assert declared.window == ContextWindowSpec()
    assert declared.readout == LastLayerReadoutSpec()
    assert declared.dependencies() == ()
    assert declared.pretrained_weights() == f"an-org/a-model@{REVISION}"


def test_a_window_reports_its_overflow_unless_it_declares_refusal() -> None:
    assert ContextWindowSpec(rows=8).overflow == "report"
    assert ContextWindowSpec(rows=8, overflow="refuse").overflow == "refuse"


def test_each_typed_layer_readout_is_declared() -> None:
    assert PretrainedTokenEncoderSpec(
        checkpoint=HubSourceSpec(repository="m", revision=REVISION),
        readout=FixedLayerReadoutSpec(layer=6),
    ).readout == FixedLayerReadoutSpec(layer=6)
    assert (
        PretrainedTokenEncoderSpec(
            checkpoint=HubSourceSpec(repository="m", revision=REVISION),
            readout=ScalarMixReadoutSpec(),
        ).readout
        == ScalarMixReadoutSpec()
    )


@pytest.mark.parametrize(
    ("payload", "detail"),
    [
        ({"kind": "pretrained"}, "checkpoint"),
        ({"kind": "pretrained", "revision": REVISION}, "checkpoint"),
        (
            {"kind": "pretrained", "checkpoint": {"kind": "hub", "repository": "m"}},
            "revision",
        ),
        (
            {
                "kind": "pretrained",
                "checkpoint": {
                    "kind": "url",
                    "url": "https://example.org/model.zip",
                    "sha256": "0" * 64,
                },
            },
            "a pretrained token encoder loads a hub repository",
        ),
        (
            {
                "kind": "pretrained",
                "checkpoint": {"kind": "hub", "repository": "", "revision": REVISION},
            },
            "a checkpoint",
        ),
        (
            {
                "kind": "pretrained",
                "checkpoint": {"kind": "hub", "repository": "m", "revision": "main"},
            },
            "full lowercase commit hash",
        ),
        (
            {
                "kind": "pretrained",
                "checkpoint": {"kind": "hub", "repository": "m", "revision": REVISION},
                "window": {"rows": 0},
            },
            "a window of no rows holds no word",
        ),
    ],
)
def test_a_missing_empty_or_out_of_range_setting_is_refused(
    payload: dict[str, JsonValue], detail: str
) -> None:
    with pytest.raises(dx.ValidationError, match=detail):
        TokenEncoderSpec.model_validate(payload)


def test_architecture_and_overflow_names_are_open_to_installed_plugins() -> None:
    declared = PretrainedTokenEncoderSpec(
        checkpoint=HubSourceSpec(repository="m", revision=REVISION),
        architecture="decoder-only",
        window=ContextWindowSpec(rows=1, overflow="drop"),
    )

    assert declared.architecture == "decoder-only"
    assert declared.window is not None
    assert declared.window.overflow == "drop"


def test_the_variant_round_trips_through_its_json_form_byte_for_byte() -> None:
    document = (
        '{"kind": "pretrained", "checkpoint": {"kind": "hub", '
        f'"repository": "an-org/a-model", "revision": "{REVISION}"}}, '
        '"architecture": "t5-encoder", "trainable": true, '
        '"window": {"rows": 64, "overflow": "refuse"}, '
        '"readout": {"kind": "fixed", "layer": 6}, "join": " | ", '
        '"declared_output": null}'
    )
    declared = TokenEncoderSpec.model_validate_json(document)

    assert declared.model_dump_json() == document
    assert TokenEncoderSpec.model_validate_json(declared.model_dump_json()) == declared
