"""Multilayer action controller."""

from typing import TYPE_CHECKING, Final, cast, override

from torch import nn

from lofi.nn._layers import feedforward
from lofi.nn.controllers._base import PairFeatures, _Controller
from lofi.nn.policy import place
from lofi.nn.raising import build_soft_controller_raising
from lofi.registry import ComponentPlugin
from lofi.spec.controllers.mlp import MlpControllerSettings, MlpControllerSpec

if TYPE_CHECKING:
    from torch import Tensor

    from lofi.judge import ActionSet
    from lofi.nn.build import BuildContext
    from lofi.nn.ports import TermEncoder
    from lofi.nn.raising import SoftControllerRaising
    from lofi.nn.sites import ComponentId
    from lofi.spec.layers import FeedForwardLayer

__all__ = ["MlpController"]


class _MlpHead(nn.Module):
    """Score declared pair features through a feedforward network."""

    def __init__(
        self,
        features: PairFeatures,
        *,
        actions: int,
        hidden: tuple[FeedForwardLayer, ...],
    ) -> None:
        super().__init__()
        self._features = features
        self._network = feedforward(
            hidden, inputs=features.output_width, output=actions, bias=True
        )

    @override
    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        """Return unnormalized action scores."""
        return self._network(self._features(left, right))


class MlpController(_Controller):
    """Score action distributions with a declared multilayer network."""

    def __init__(
        self,
        action_set: ActionSet,
        *,
        width: int,
        component: ComponentId,
        settings: MlpControllerSettings,
        raising: SoftControllerRaising | None,
    ) -> None:
        features = PairFeatures(settings.features, width=width)
        head_actions = action_set if raising is None else raising.base_actions
        super().__init__(
            action_set,
            _MlpHead(
                features,
                actions=len(head_actions.actions),
                hidden=settings.hidden,
            ),
            width=width,
            component=component,
            input_source=settings.input,
            normalizer=settings.normalizer,
            raising_mode=settings.raising,
            raising=raising,
        )


def _build(settings: MlpControllerSettings, context: BuildContext, /) -> MlpController:
    """Build and place a multilayer controller."""
    encoder = cast("TermEncoder", context.producer("cells"))
    width = encoder.output_width * (2 if settings.input == "span" else 1)
    raising = build_soft_controller_raising(
        settings.raising, settings.input, context, encoder
    )
    return place(
        MlpController(
            context.actions,
            width=width,
            component=context.component,
            settings=settings,
            raising=raising,
        ),
        context.policy,
    )


PLUGIN: Final = ComponentPlugin(spec=MlpControllerSpec, build=_build)
"""The multilayer controller plugin."""
