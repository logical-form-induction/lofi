"""Controller declarations."""

from lofi.spec.controllers._base import (
    ControllerFeature,
    ControllerInput,
    ControllerNormalizerSettings,
    ControllerNormalizerSpec,
    ControllerRaising,
    EntmaxControllerNormalizerSpec,
    SoftmaxControllerNormalizerSpec,
)
from lofi.spec.controllers.bilinear import (
    BilinearControllerSettings,
    BilinearControllerSpec,
)
from lofi.spec.controllers.mlp import MlpControllerSettings, MlpControllerSpec
from lofi.spec.controllers.transformer import (
    TransformerControllerSettings,
    TransformerControllerSpec,
)

__all__ = [
    "BilinearControllerSettings",
    "BilinearControllerSpec",
    "ControllerFeature",
    "ControllerInput",
    "ControllerNormalizerSettings",
    "ControllerNormalizerSpec",
    "ControllerRaising",
    "EntmaxControllerNormalizerSpec",
    "MlpControllerSettings",
    "MlpControllerSpec",
    "SoftmaxControllerNormalizerSpec",
    "TransformerControllerSettings",
    "TransformerControllerSpec",
]
