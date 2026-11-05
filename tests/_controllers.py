"""Factories for current controller variants used across test layers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lofi.nn.controllers import MlpController
from lofi.spec.controllers import ControllerInput, MlpControllerSpec
from lofi.spec.layers import FeedForwardLayerSpec

if TYPE_CHECKING:
    from lofi.judge import ActionSet
    from lofi.nn.sites import ComponentId

__all__ = ["mlp_controller"]


def mlp_controller(
    actions: ActionSet,
    *,
    width: int,
    component: ComponentId,
    hidden: int,
    input_source: ControllerInput = "inside",
) -> MlpController:
    """Return the default current MLP controller at a requested test width."""
    spec = MlpControllerSpec(
        input=input_source,
        hidden=(FeedForwardLayerSpec(width=hidden, activation="tanh", dropout=0.0),),
    )
    return MlpController(
        actions,
        width=width,
        component=component,
        settings=spec.settings(),
        raising=None,
    )
