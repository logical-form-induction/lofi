"""Bilinear action-controller declaration."""

from typing import Literal, NamedTuple

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

__all__ = ["BilinearControllerSettings", "BilinearControllerSpec"]


class BilinearControllerSettings(NamedTuple):
    """Resolved settings for a bilinear action controller."""

    input: ControllerInput
    features: tuple[ControllerFeature, ...]
    normalizer: ControllerNormalizerSettings
    raising: ControllerRaising


class BilinearControllerSpec(ControllerSpec):
    """Score a pair bilinearly beside declared affine pair features."""

    kind: Literal["bilinear"] = "bilinear"
    input: ControllerInput = _DEFAULT_INPUT
    features: tuple[ControllerFeature, ...] = _DEFAULT_FEATURES
    normalizer: ControllerNormalizerSpec = _DEFAULT_NORMALIZER

    @dx.validates("features")
    @classmethod
    def _features_are_nonempty_and_distinct(
        cls, features: tuple[ControllerFeature, ...]
    ) -> tuple[ControllerFeature, ...]:
        """Return a nonempty feature sequence without repetitions."""
        return checked_features(features)

    def settings(self) -> BilinearControllerSettings:
        """Return the checked interior settings."""
        return BilinearControllerSettings(*common_settings(self))
