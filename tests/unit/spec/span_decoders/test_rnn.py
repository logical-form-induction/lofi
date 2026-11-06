import didactic.api as dx
import pytest

from lofi.spec._ports import Dependency
from lofi.spec.components.model import ModelSection, SpanDecoderSpec
from lofi.spec.layers import FeedForwardLayer, FeedForwardLayerSpec
from lofi.spec.span_decoders.rnn import RnnSpanDecoderSettings, RnnSpanDecoderSpec


def test_the_tag_reaches_the_rnn_variant_through_the_family_root() -> None:
    declared = SpanDecoderSpec.model_validate({"kind": "rnn-span-decoder"})

    assert type(declared) is RnnSpanDecoderSpec
    assert declared == RnnSpanDecoderSpec()
    assert SpanDecoderSpec.__variants__["rnn-span-decoder"] is RnnSpanDecoderSpec
    assert declared.dependencies() == (Dependency("parser", "cells"),)


def test_an_undeclared_architecture_is_one_gru_layer_scored_by_one_map() -> None:
    assert RnnSpanDecoderSpec().settings() == RnnSpanDecoderSettings(
        cell="gru", layers=1, state_width=None, head=(), end="<eos>"
    )


def test_the_settings_follow_every_declared_field() -> None:
    declared = RnnSpanDecoderSpec.model_validate(
        {
            "kind": "rnn-span-decoder",
            "cell": "elman-relu",
            "layers": 3,
            "state_width": 7,
            "end": "<stop>",
            "head": [
                {"width": 5, "activation": "gelu", "dropout": 0.1},
                {"width": 3, "activation": "tanh"},
            ],
        }
    )

    assert declared.settings() == RnnSpanDecoderSettings(
        cell="elman-relu",
        layers=3,
        state_width=7,
        head=(
            FeedForwardLayer(5, "gelu", 0.1),
            FeedForwardLayer(3, "tanh", 0.0),
        ),
        end="<stop>",
    )


def test_a_width_is_not_declared_here_because_the_chart_fixes_it() -> None:
    with pytest.raises(dx.ValidationError, match="width"):
        SpanDecoderSpec.model_validate({"kind": "rnn-span-decoder", "width": 8})


def test_external_layer_names_remain_declared_until_plugin_resolution() -> None:
    declared = SpanDecoderSpec.model_validate(
        {
            "kind": "rnn-span-decoder",
            "cell": "test-package-cell",
            "head": [{"width": 4, "activation": "test-package-activation"}],
        }
    )

    assert declared.cell == "test-package-cell"
    assert declared.head == (
        FeedForwardLayerSpec(width=4, activation="test-package-activation"),
    )


@pytest.mark.parametrize("layers", [0, -2])
def test_a_recurrence_of_no_layer_is_refused(layers: int) -> None:
    with pytest.raises(dx.ValidationError, match="a recurrence has a layer"):
        RnnSpanDecoderSpec(layers=layers)


@pytest.mark.parametrize("width", [0, -3])
def test_a_state_of_no_coordinate_is_refused(width: int) -> None:
    with pytest.raises(dx.ValidationError, match="a declared layer width"):
        SpanDecoderSpec.model_validate(
            {"kind": "rnn-span-decoder", "state_width": width}
        )


def test_the_removed_shared_head_width_is_refused() -> None:
    with pytest.raises(dx.ValidationError, match="head_width"):
        RnnSpanDecoderSpec.model_validate({"kind": "rnn-span-decoder", "head_width": 4})


def test_span_decoder_roles_are_canonical_in_the_model_section() -> None:
    model = ModelSection.model_validate(
        {
            "language": {"from": "data"},
            "span_decoders": {
                "zeta": {"kind": "rnn-span-decoder"},
                "alpha": {"kind": "rnn-span-decoder"},
            },
        }
    )

    assert tuple(model.span_decoders) == ("alpha", "zeta")
    assert (
        ModelSection.model_validate({"language": {"from": "data"}}).span_decoders == {}
    )


def test_the_variant_round_trips_through_its_json_form_byte_for_byte() -> None:
    document = (
        '{"kind": "rnn-span-decoder", "cell": "lstm", "layers": 2, '
        '"state_width": 6, "head": [{"width": 4, "activation": "relu", '
        '"dropout": 0.1}], "end": "<stop>", '
        '"declared_output": null}'
    )
    declared = SpanDecoderSpec.model_validate_json(document)

    assert declared.model_dump_json() == document
    assert SpanDecoderSpec.model_validate_json(declared.model_dump_json()) == declared
