"""Tests for the multilayer controller declaration."""

import didactic.api as dx
import pytest

from lofi.spec.controllers import MlpControllerSpec
from lofi.spec.layers import FeedForwardLayerSpec


def test_the_default_mlp_declares_its_current_architecture() -> None:
    declared = MlpControllerSpec()

    assert declared.input == "inside"
    assert declared.features == ("left", "right")
    assert declared.normalizer.kind == "softmax"
    assert declared.hidden == (FeedForwardLayerSpec(width=64, activation="tanh"),)


def test_an_mlp_lowers_hidden_layers_features_and_input() -> None:
    declared = MlpControllerSpec(
        input="span",
        features=("difference", "product"),
        hidden=(FeedForwardLayerSpec(width=7, activation="silu", dropout=0.1),),
    )

    settings = declared.settings()

    assert settings.input == "span"
    assert settings.features == ("difference", "product")
    assert settings.hidden == ((7, "silu", 0.1),)


def test_a_controller_refuses_repeated_pair_features() -> None:
    with pytest.raises(dx.ValidationError, match="repeats"):
        MlpControllerSpec(features=("left", "left"))
