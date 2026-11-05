import didactic.api as dx
import pytest

from lofi.spec.span_functions import (
    BilinearSpanFunctionSettings,
    BilinearSpanFunctionSpec,
    CpBilinearSpanFunctionSettings,
    CpBilinearSpanFunctionSpec,
    GatedMlpSpanFunctionSettings,
    GatedMlpSpanFunctionSpec,
    HrrSpanFunctionSettings,
    HrrSpanFunctionSpec,
    MlpSpanFunctionSettings,
    MlpSpanFunctionSpec,
    SpanFunctionSpec,
    TransformerSpanFunctionSettings,
    TransformerSpanFunctionSpec,
)


def test_each_span_function_lowers_to_immutable_settings() -> None:
    assert MlpSpanFunctionSpec(layers=2, activation="gelu").declaration().settings == (
        MlpSpanFunctionSettings(2, "gelu")
    )
    assert BilinearSpanFunctionSpec().declaration().settings == (
        BilinearSpanFunctionSettings()
    )
    assert CpBilinearSpanFunctionSpec(rank=3).declaration().settings == (
        CpBilinearSpanFunctionSettings(3)
    )
    assert GatedMlpSpanFunctionSpec(
        layers=1, activation="tanh"
    ).declaration().settings == GatedMlpSpanFunctionSettings(1, "tanh")
    assert HrrSpanFunctionSpec(binding="circular").declaration().settings == (
        HrrSpanFunctionSettings("circular")
    )
    assert TransformerSpanFunctionSpec(
        layers=2,
        heads=2,
        feedforward_width=12,
        activation="relu",
    ).declaration().settings == TransformerSpanFunctionSettings(
        2, 2, 12, "relu", 0.0, "post"
    )
    assert TransformerSpanFunctionSpec(
        layers=1,
        heads=1,
        feedforward_width=4,
        activation="gelu",
        dropout=0.25,
        norm_placement="pre",
    ).declaration().settings == TransformerSpanFunctionSettings(
        1, 1, 4, "gelu", 0.25, "pre"
    )


def test_positive_mlp_layer_count_is_checked() -> None:
    with pytest.raises(dx.ValidationError, match="layers"):
        SpanFunctionSpec.model_validate(
            {"kind": "mlp", "layers": 0, "activation": "tanh"}
        )


def test_positive_cp_rank_is_checked() -> None:
    with pytest.raises(dx.ValidationError, match="rank"):
        SpanFunctionSpec.model_validate({"kind": "cp-bilinear", "rank": 0})


def test_transformer_dimensions_are_checked() -> None:
    with pytest.raises(dx.ValidationError, match="heads"):
        TransformerSpanFunctionSpec(
            layers=1,
            heads=0,
            feedforward_width=8,
            activation="gelu",
        )


def test_span_function_declarations_forbid_unknown_fields() -> None:
    with pytest.raises(dx.ValidationError, match="extra"):
        SpanFunctionSpec.model_validate({"kind": "bilinear", "extra": True})
