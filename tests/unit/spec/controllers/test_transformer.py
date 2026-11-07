"""Tests for the transformer controller declaration."""

from typing import Literal

import didactic.api as dx
import pytest

from lofi.spec.controllers import TransformerControllerSpec


def _zero(
    field: Literal["layers", "heads", "feedforward_width"],
) -> TransformerControllerSpec:
    """Construct a transformer declaration with one zero architecture count."""
    match field:
        case "layers":
            return TransformerControllerSpec(layers=0)
        case "heads":
            return TransformerControllerSpec(heads=0)
        case "feedforward_width":
            return TransformerControllerSpec(feedforward_width=0)


def test_a_transformer_controller_lowers_its_architecture() -> None:
    settings = TransformerControllerSpec(
        layers=3,
        heads=2,
        feedforward_width=17,
        activation="relu",
        dropout=0.1,
        norm_placement="pre",
    ).settings()

    assert settings.layers == 3
    assert settings.heads == 2
    assert settings.feedforward_width == 17
    assert settings.activation == "relu"
    assert settings.dropout == 0.1
    assert settings.norm_placement == "pre"
    assert TransformerControllerSpec().settings().norm_placement == "post"


@pytest.mark.parametrize("field", ["layers", "heads", "feedforward_width"])
def test_a_transformer_controller_refuses_a_zero_architecture_count(
    field: Literal["layers", "heads", "feedforward_width"],
) -> None:
    with pytest.raises(dx.ValidationError):
        _zero(field)
