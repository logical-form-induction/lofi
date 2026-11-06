"""Shared action-controller mechanics."""

from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Protocol, override

import torch
from torch import nn

from lofi._types import ModuleSharing, SharedSubmodule
from lofi.nn._values import Encoded, LogDistribution
from lofi.nn.controllers._features import pair_feature
from lofi.nn.loss._mapping import Entmax
from lofi.nn.sites import (
    Access,
    ComponentKind,
    DTypeClass,
    Role,
    Site,
    SiteMap,
    SiteSpec,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import Tensor

    from lofi.judge import ActionSet
    from lofi.nn.raising import SoftControllerRaising
    from lofi.nn.sites import ComponentId
    from lofi.spec.controllers import (
        ControllerFeature,
        ControllerInput,
        ControllerNormalizerSettings,
        ControllerRaising,
    )

__all__ = ["ControllerHead", "PairFeatures", "_Controller"]

_CELL_AXES: Final[tuple[str, ...]] = ("cell", "width")
_WEIGHT_AXES: Final[tuple[str, ...]] = ("cell", "action")


class ControllerHead(Protocol):
    """Score an ordered pair of child representations."""

    def __call__(self, left: Tensor, right: Tensor, /) -> Tensor:
        """Return one unnormalized action score per pair and action."""
        ...


class PairFeatures(nn.Module):
    """Construct a declared sequence of ordered child pair features."""

    def __init__(self, features: tuple[ControllerFeature, ...], *, width: int) -> None:
        super().__init__()
        self._features = tuple(pair_feature(name) for name in features)
        self._width = width

    @property
    def output_width(self) -> int:
        """Return the width of the concatenated feature vector."""
        return len(self._features) * self._width

    @override
    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        """Return the requested features in declaration order."""
        return torch.cat([feature(left, right) for feature in self._features], dim=-1)


class _ActionNormalizer:
    """Map unnormalized action scores through one declared normalization."""

    __slots__ = ("_mapping",)

    def __init__(self, settings: ControllerNormalizerSettings) -> None:
        self._mapping = Entmax(settings.alpha, bisections=settings.bisections)

    def __call__(self, scores: Tensor, /) -> Tensor:
        """Return normalized action log weights."""
        return self._mapping.project(scores)


def _site_map(component: ComponentId) -> SiteMap:
    """Return the controller's published action-weight site."""
    return SiteMap(
        component,
        ComponentKind.CONTROLLER,
        {
            Site(component, Role.COMBINATOR_LOGITS): SiteSpec(
                _WEIGHT_AXES,
                DTypeClass.LOG_PROBABILITY,
                differentiable=True,
                access=Access.OBSERVE | Access.PATCH,
            )
        },
    )


class _Controller(nn.Module):
    """Apply one controller head and normalization to adjacent cells."""

    def __init__(
        self,
        action_set: ActionSet,
        head: nn.Module,
        *,
        width: int,
        component: ComponentId,
        input_source: ControllerInput,
        normalizer: ControllerNormalizerSettings,
        raising_mode: ControllerRaising,
        raising: SoftControllerRaising | None = None,
    ) -> None:
        super().__init__()
        self._head = head
        self._actions = action_set
        self._component = component
        self._width = width
        self._input = input_source
        self._normalizer = _ActionNormalizer(normalizer)
        self._raising = raising

        if (raising_mode == "soft") != (raising is not None):
            msg = (
                f"controller raising mode {raising_mode!r} and its built "
                "soft-raising module disagree"
            )
            raise ValueError(msg)

        if raising is not None and raising.action_set is not action_set:
            msg = "soft raising must score the controller's exact action inventory"
            raise ValueError(msg)

    @property
    def action_set(self) -> ActionSet:
        """Return the actions this controller weighs, in column order."""
        return self._actions

    @property
    def input_width(self) -> int:
        """Return the width of each child representation."""
        return self._width

    @property
    def input(self) -> ControllerInput:
        """Return which chart representation the controller reads."""
        return self._input

    @property
    def shared(self) -> Mapping[str, SharedSubmodule[nn.Module]]:
        """Return the whole policy shared with its chart."""
        return MappingProxyType(
            {
                "policy": SharedSubmodule(
                    self, ModuleSharing(in_parameters=True, in_state_dict=True)
                )
            }
        )

    def weigh(self, left: Encoded, /, *, right: Encoded) -> LogDistribution:
        """Return normalized action log weights for adjacent representations."""
        for name, cells in (("left", left), ("right", right)):
            if cells.axes != _CELL_AXES:
                msg = (
                    f"the {name} cells of a pair run over {_CELL_AXES}, "
                    f"not over {cells.axes}"
                )
                raise ValueError(msg)

            if cells.size("width") != self._width:
                msg = (
                    f"the {name} cells have width {cells.size('width')}, "
                    f"not this controller's width {self._width}"
                )
                raise ValueError(msg)

        head = self._head
        scores = (
            head(left.tensor, right.tensor)
            if self._raising is None
            else self._raising.scores(head, left, right)
        )

        return LogDistribution(
            self._normalizer(scores),
            _WEIGHT_AXES,
            normalized_over="action",
        )

    def site_map(self) -> SiteMap:
        """Return the published action-weight point."""
        return _site_map(self._component)

    @override
    def get_extra_state(self) -> str:
        """Return the action set digest recorded with a checkpoint."""
        return self._actions.digest

    @override
    def set_extra_state(self, state: str) -> None:
        """Refuse a checkpoint whose action columns name another inventory."""
        if not isinstance(state, str):
            msg = f"a checkpoint records its action set as a digest, not {state!r}"
            raise TypeError(msg)

        if state != self._actions.digest:
            msg = (
                f"the checkpoint records the action set {state} and this "
                f"controller weighs {self._actions.digest}"
            )
            raise ValueError(msg)
