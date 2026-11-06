"""Multilayer action-controller declaration."""

from typing import TYPE_CHECKING, Literal, NamedTuple

import didactic.api as dx

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
from lofi.spec.layers import FeedForwardLayerSpec

if TYPE_CHECKING:
    from lofi.spec.layers import FeedForwardLayer

__all__ = ["MlpControllerSettings", "MlpControllerSpec"]


class MlpControllerSettings(NamedTuple):
    """Resolved settings for a multilayer action controller."""

    input: ControllerInput
    features: tuple[ControllerFeature, ...]
    normalizer: ControllerNormalizerSettings
    raising: ControllerRaising
    hidden: tuple[FeedForwardLayer, ...]


class MlpControllerSpec(ControllerSpec):
    """Score declared pair features with a feedforward network."""

    kind: Literal["mlp"] = "mlp"
    input: ControllerInput = _DEFAULT_INPUT
    features: tuple[ControllerFeature, ...] = _DEFAULT_FEATURES
    normalizer: ControllerNormalizerSpec = _DEFAULT_NORMALIZER
    hidden: tuple[FeedForwardLayerSpec, ...] = (
        FeedForwardLayerSpec(width=64, activation="tanh", dropout=0.0),
    )

    @dx.validates("features")
    @classmethod
    def _features_are_nonempty_and_distinct(
        cls, features: tuple[ControllerFeature, ...]
    ) -> tuple[ControllerFeature, ...]:
        """Return a nonempty feature sequence without repetitions."""
        return checked_features(features)

    def settings(self) -> MlpControllerSettings:
        """Return the checked interior settings."""
        input_source, features, normalizer, raising = common_settings(self)
        return MlpControllerSettings(
            input_source,
            features,
            normalizer,
            raising,
            tuple(layer.settings() for layer in self.hidden),
        )
