"""Tests for declared neural chart-composition families."""

from lofi.spec.chart_compositions import (
    ChartCompositionSpec,
    GatedResidualCompositionSettings,
    GatedResidualCompositionSpec,
    MeanFieldCompositionSettings,
    MeanFieldCompositionSpec,
)
from lofi.spec.span_functions import (
    MlpSpanFunctionSettings,
    SpanFunctionDeclaration,
)


def mlp() -> SpanFunctionDeclaration:
    """Return the default declared MLP function."""
    return SpanFunctionDeclaration("mlp", MlpSpanFunctionSettings(1, "tanh"))


def test_mean_field_composition_declares_leaf_activation_and_cell_function() -> None:
    declared = ChartCompositionSpec.model_validate({"kind": "mean-field"})

    assert type(declared) is MeanFieldCompositionSpec
    assert declared.declaration() == (
        "mean-field",
        MeanFieldCompositionSettings(mlp(), mlp()),
    )


def test_gated_residual_composition_declares_each_mixing_function() -> None:
    declared = ChartCompositionSpec.model_validate({"kind": "gated-residual"})

    assert type(declared) is GatedResidualCompositionSpec
    assert declared.declaration() == (
        "gated-residual",
        GatedResidualCompositionSettings(mlp(), mlp(), mlp()),
    )


def test_each_composition_round_trips_through_the_family_root() -> None:
    for kind in ("mean-field", "gated-residual"):
        declared = ChartCompositionSpec.model_validate({"kind": kind})

        assert (
            ChartCompositionSpec.model_validate_json(declared.model_dump_json())
            == declared
        )
