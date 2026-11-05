"""Bilinear action controller."""

from typing import TYPE_CHECKING, Final, cast, override

from torch import nn

from lofi.nn.controllers._base import PairFeatures, _Controller
from lofi.nn.policy import place
from lofi.nn.raising import build_soft_controller_raising
from lofi.registry import ComponentPlugin
from lofi.spec.controllers.bilinear import (
    BilinearControllerSettings,
    BilinearControllerSpec,
)

if TYPE_CHECKING:
    from torch import Tensor

    from lofi.judge import ActionSet
    from lofi.nn.build import BuildContext
    from lofi.nn.ports import TermEncoder
    from lofi.nn.raising import SoftControllerRaising
    from lofi.nn.sites import ComponentId

__all__ = ["BilinearController"]


class _BilinearHead(nn.Module):
    """Combine a bilinear score with declared affine pair features."""

    def __init__(
        self,
        features: PairFeatures,
        *,
        width: int,
        actions: int,
    ) -> None:
        super().__init__()
        self._features = features
        self._bilinear = nn.Bilinear(width, width, actions, bias=False)
        self._affine = nn.Linear(features.output_width, actions)

    @override
    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        """Return unnormalized action scores."""
        return self._bilinear(left, right) + self._affine(self._features(left, right))


class BilinearController(_Controller):
    """Score actions with bilinear child interactions."""

    def __init__(
        self,
        action_set: ActionSet,
        *,
        width: int,
        component: ComponentId,
        settings: BilinearControllerSettings,
        raising: SoftControllerRaising | None,
    ) -> None:
        features = PairFeatures(settings.features, width=width)
        head_actions = action_set if raising is None else raising.base_actions
        super().__init__(
            action_set,
            _BilinearHead(features, width=width, actions=len(head_actions.actions)),
            width=width,
            component=component,
            input_source=settings.input,
            normalizer=settings.normalizer,
            raising_mode=settings.raising,
            raising=raising,
        )


def _build(
    settings: BilinearControllerSettings, context: BuildContext, /
) -> BilinearController:
    """Build and place a bilinear controller."""
    encoder = cast("TermEncoder", context.producer("cells"))
    width = encoder.output_width * (2 if settings.input == "span" else 1)
    raising = build_soft_controller_raising(
        settings.raising, settings.input, context, encoder
    )
    return place(
        BilinearController(
            context.actions,
            width=width,
            component=context.component,
            settings=settings,
            raising=raising,
        ),
        context.policy,
    )


PLUGIN: Final = ComponentPlugin(spec=BilinearControllerSpec, build=_build)
"""The bilinear controller plugin."""
