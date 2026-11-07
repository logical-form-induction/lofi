"""Registered split weights for neural chart cell mixtures."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, final

from torch import nn

from lofi._frozen import FrozenValue
from lofi.nn.parsers.functions import SpanFunctionModule, build_span_function
from lofi.registry import Registry, RegistryError
from lofi.spec.parsers.split import (
    AttendSplitWeigherSettings,
    MarginalSplitWeigherSettings,
    SplitWeigherDeclaration,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

    from lofi.nn.parsers._ports import LogWeightReduction

__all__ = [
    "ATTEND_PLUGIN",
    "MARGINAL_PLUGIN",
    "AttendSplitWeigher",
    "MarginalSplitWeigher",
    "SplitWeigher",
    "SplitWeigherPlugin",
    "build_split_weigher",
]


class SplitWeigher(Protocol):
    """Normalize the alternatives used to mix one width of chart cells."""

    @property
    def reads_children(self) -> bool:
        """Return whether the weights are scored from the child vectors."""
        ...

    def weights(
        self,
        joined: Tensor,
        local: Tensor,
        left: Tensor | None,
        right: Tensor | None,
        mask: Tensor,
        reduction: LogWeightReduction,
        /,
        *,
        dim: int,
    ) -> Tensor:
        """Return normalized log weights over the selected alternative axis."""
        ...


@final
class MarginalSplitWeigher(FrozenValue):
    """Weight splits by child inside masses and their learned local score."""

    __slots__ = ()
    _noun = "marginal split weigher"

    @property
    def reads_children(self) -> bool:
        """Return false: the weights read child inside masses, not child vectors."""
        return False

    def weights(
        self,
        joined: Tensor,
        local: Tensor,
        left: Tensor | None,
        right: Tensor | None,
        mask: Tensor,
        reduction: LogWeightReduction,
        /,
        *,
        dim: int,
    ) -> Tensor:
        """Return the exact conditional split posterior."""
        del local, left, right
        return reduction.mixing_weights(joined, mask, dim=dim)


@final
class AttendSplitWeigher(nn.Module):
    """Weight splits only by the learned child pair score."""

    def __init__(self, score: SpanFunctionModule, /) -> None:
        super().__init__()

        if len(score.input_widths) != 2:
            msg = (
                "an attend split scorer reads the left and right child, "
                f"not {len(score.input_widths)} inputs"
            )
            raise ValueError(msg)

        if score.input_widths[0] != score.input_widths[1]:
            msg = (
                "an attend split scorer reads children of one width, not "
                f"{score.input_widths}"
            )
            raise ValueError(msg)

        if score.output_width != 1:
            msg = f"an attend split scorer returns one logit, not {score.output_width}"
            raise ValueError(msg)

        self._score = score

    @property
    def reads_children(self) -> bool:
        """Return true: the weights are the scores of the child-vector pairs."""
        return True

    @property
    def score(self) -> SpanFunctionModule:
        """Return the declared child pair scorer."""
        return self._score

    def weights(
        self,
        joined: Tensor,
        local: Tensor,
        left: Tensor | None,
        right: Tensor | None,
        mask: Tensor,
        reduction: LogWeightReduction,
        /,
        *,
        dim: int,
    ) -> Tensor:
        """Return normalized attention over the child pair scores."""
        del joined, local

        if left is None or right is None:
            msg = "attend split weights require the left and right child vectors"
            raise ValueError(msg)

        logits = self._score(left, right).squeeze(-1)

        broadcastable = (
            logits.ndim == mask.ndim
            and tuple(logits.shape[1:]) == tuple(mask.shape[1:])
            and int(mask.shape[0]) in {1, int(logits.shape[0])}
        )

        if not broadcastable:
            msg = (
                f"attend scored child pairs of shape {tuple(logits.shape)}, "
                f"not the split shape {tuple(mask.shape)}"
            )
            raise ValueError(msg)

        return reduction.mixing_weights(logits, mask, dim=dim)


class SplitWeigherPlugin[S](FrozenValue):
    """Build one split weigher from its resolved settings."""

    __slots__ = ("_build", "_settings")
    _noun = "split-weigher plugin"

    def __init__(
        self, *, settings: type[S], build: Callable[[S, int], SplitWeigher]
    ) -> None:
        object.__setattr__(self, "_settings", settings)
        object.__setattr__(self, "_build", build)

    def construct(self, settings: object, width: int, /) -> SplitWeigher:
        """Build a weigher after checking the registered settings type."""
        if not isinstance(settings, self._settings):
            msg = (
                f"a split-weigher plugin reads {self._settings.__name__}, "
                f"not {type(settings).__name__}"
            )
            raise TypeError(msg)

        return self._build(settings, width)


def build_split_weigher(
    declaration: SplitWeigherDeclaration, width: int, /
) -> SplitWeigher:
    """Build one split weigher through its installed plugin."""
    member = Registry[object]("split_weigher").get(declaration.kind)

    if not isinstance(member, SplitWeigherPlugin):
        msg = (
            f"the split_weigher entry {declaration.kind!r} names {member!r}, "
            "not a split-weigher plugin"
        )
        raise RegistryError(msg, family="split_weigher")

    return member.construct(declaration.settings, width)


def _marginal(
    settings: MarginalSplitWeigherSettings, width: int, /
) -> MarginalSplitWeigher:
    """Build exact conditional split posterior weighting."""
    del settings, width
    return MarginalSplitWeigher()


def _attend(settings: AttendSplitWeigherSettings, width: int, /) -> AttendSplitWeigher:
    """Build learned child pair attention weighting."""
    return AttendSplitWeigher(build_span_function(settings.score, (width, width), 1))


MARGINAL_PLUGIN = SplitWeigherPlugin(
    settings=MarginalSplitWeigherSettings,
    build=_marginal,
)
ATTEND_PLUGIN = SplitWeigherPlugin(
    settings=AttendSplitWeigherSettings,
    build=_attend,
)
