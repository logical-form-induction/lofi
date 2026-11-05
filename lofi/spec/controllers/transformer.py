"""Transformer action-controller declaration."""

from typing import Literal, NamedTuple

import didactic.api as dx

from lofi._checks import ensure_name
from lofi.spec.components.model import ControllerSpec
from lofi.spec.controllers._base import (
    _DEFAULT_FEATURES,
    _DEFAULT_INPUT,
    _DEFAULT_NORMALIZER,
    ControllerFeature,
    ControllerInput,
    ControllerNormalizerSettings,
    ControllerNormalizerSpec,
    ControllerRaising,
    checked_features,
    common_settings,
)
from lofi.spec.layers import DEFAULT_ACTIVATION, NormPlacement, dropout_rate

__all__ = ["TransformerControllerSettings", "TransformerControllerSpec"]


class TransformerControllerSettings(NamedTuple):
    """Resolved settings for a transformer action controller."""

    input: ControllerInput
    features: tuple[ControllerFeature, ...]
    normalizer: ControllerNormalizerSettings
    raising: ControllerRaising
    layers: int
    heads: int
    feedforward_width: int
    activation: str
    dropout: float
    norm_placement: NormPlacement


class TransformerControllerSpec(ControllerSpec):
    """Score two child representations with self-attention."""

    kind: Literal["transformer"] = "transformer"
    input: ControllerInput = _DEFAULT_INPUT
    features: tuple[ControllerFeature, ...] = _DEFAULT_FEATURES
    normalizer: ControllerNormalizerSpec = _DEFAULT_NORMALIZER
    layers: int = 1
    heads: int = 1
    feedforward_width: int = 128
    activation: str = DEFAULT_ACTIVATION
    dropout: float = 0.0
    norm_placement: Literal["pre", "post"] = "post"

    __axioms__ = (
        dx.axiom("layers > 0", message="a transformer controller has a layer"),
        dx.axiom("heads > 0", message="a transformer controller has an attention head"),
        dx.axiom(
            "feedforward_width > 0",
            message="a transformer controller has a feed-forward coordinate",
        ),
    )

    @dx.validates("features")
    @classmethod
    def _features_are_nonempty_and_distinct(
        cls, features: tuple[ControllerFeature, ...]
    ) -> tuple[ControllerFeature, ...]:
        """Return a nonempty feature sequence without repetitions."""
        return checked_features(features)

    @dx.validates("activation")
    @classmethod
    def _activation_is_named(cls, value: str) -> str:
        """Return a named activation plugin."""
        return ensure_name("a transformer controller activation", value)

    @dx.validates("dropout")
    @classmethod
    def _dropout_is_a_rate(cls, value: float) -> float:
        """Return a finite dropout rate in the half-open unit interval."""
        return dropout_rate("a transformer controller dropout rate", value)

    def settings(self) -> TransformerControllerSettings:
        """Return the checked interior settings."""
        input_source, features, normalizer, raising = common_settings(self)
        return TransformerControllerSettings(
            input_source,
            features,
            normalizer,
            raising,
            self.layers,
            self.heads,
            self.feedforward_width,
            self.activation,
            self.dropout,
            self.norm_placement,
        )
