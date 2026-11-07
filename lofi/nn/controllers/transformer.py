"""Transformer action controller."""

from typing import TYPE_CHECKING, Final, cast, override

import torch
from torch import nn

from lofi.nn._layers import activation
from lofi.nn.controllers._base import PairFeatures, _Controller
from lofi.nn.policy import place
from lofi.nn.raising import build_soft_controller_raising
from lofi.registry import ComponentPlugin
from lofi.spec.controllers.transformer import (
    TransformerControllerSettings,
    TransformerControllerSpec,
)

if TYPE_CHECKING:
    from torch import Tensor

    from lofi.judge import ActionSet
    from lofi.nn.build import BuildContext
    from lofi.nn.ports import TermEncoder
    from lofi.nn.raising import SoftControllerRaising
    from lofi.nn.sites import ComponentId

__all__ = ["TransformerController"]


class _TransformerHead(nn.Module):
    """Contextualize two children before scoring declared pair features."""

    def __init__(
        self,
        features: PairFeatures,
        *,
        width: int,
        actions: int,
        settings: TransformerControllerSettings,
    ) -> None:
        super().__init__()

        if width % settings.heads:
            msg = (
                f"a transformer controller of width {width} cannot divide its "
                f"coordinates over {settings.heads} attention heads"
            )
            raise ValueError(msg)

        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=settings.heads,
            dim_feedforward=settings.feedforward_width,
            dropout=settings.dropout,
            activation=activation(settings.activation),
            batch_first=True,
            norm_first=settings.norm_placement == "pre",
        )
        self._encoder = nn.TransformerEncoder(
            layer, num_layers=settings.layers, enable_nested_tensor=False
        )
        self._features = features
        self._score = nn.Linear(features.output_width, actions)

    @override
    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        """Return unnormalized action scores."""
        pair = self._encoder(torch.stack((left, right), dim=1))
        return self._score(self._features(pair[:, 0], pair[:, 1]))


class TransformerController(_Controller):
    """Score actions after self-attention over the two children."""

    def __init__(
        self,
        action_set: ActionSet,
        *,
        width: int,
        component: ComponentId,
        settings: TransformerControllerSettings,
        raising: SoftControllerRaising | None,
    ) -> None:
        features = PairFeatures(settings.features, width=width)
        head_actions = action_set if raising is None else raising.base_actions
        super().__init__(
            action_set,
            _TransformerHead(
                features,
                width=width,
                actions=len(head_actions.actions),
                settings=settings,
            ),
            width=width,
            component=component,
            input_source=settings.input,
            normalizer=settings.normalizer,
            raising_mode=settings.raising,
            raising=raising,
        )


def _build(
    settings: TransformerControllerSettings, context: BuildContext, /
) -> TransformerController:
    """Build and place a transformer controller."""
    encoder = cast("TermEncoder", context.producer("cells"))
    width = encoder.output_width * (2 if settings.input == "span" else 1)
    raising = build_soft_controller_raising(
        settings.raising, settings.input, context, encoder
    )
    return place(
        TransformerController(
            context.actions,
            width=width,
            component=context.component,
            settings=settings,
            raising=raising,
        ),
        context.policy,
    )


PLUGIN: Final = ComponentPlugin(spec=TransformerControllerSpec, build=_build)
"""The transformer controller plugin."""
