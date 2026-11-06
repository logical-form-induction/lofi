"""The neural chart scorer over the inside and outside kernels and the rule score."""

from math import inf
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, NamedTuple, Self, cast, override

import torch
from torch import nn

from lofi._types import ModuleSharing, SharedSubmodule
from lofi.deduce import SpanIndex
from lofi.nn._values import ChartActionScores, ChartReadouts, Encoded, LogDistribution
from lofi.nn.parsers._channel import SplitChannel
from lofi.nn.parsers._inside import _root_ids
from lofi.nn.parsers._marginals import log_constituent_marginals
from lofi.nn.parsers._outside import (
    OutsideKernel,
    VectorOutsideKernel,
)
from lofi.nn.parsers._ports import LogWeightReduction
from lofi.nn.parsers._reduction import LogSumExpReduction, MaximumReduction
from lofi.nn.parsers._stacked import SpanTransition, StackedInsideOutside
from lofi.nn.parsers._weigher import SplitWeigher, build_split_weigher
from lofi.nn.parsers.composition import (
    ChartCompositionModule,
    action_arities,
    build_chart_composition,
)
from lofi.nn.parsers.functions import SpanFunctionModule, build_span_function
from lofi.nn.policy import place
from lofi.nn.ports import ReceiverEdge
from lofi.nn.psi import RuleScorer, RuleScores
from lofi.nn.sites import (
    Access,
    ComponentKind,
    DTypeClass,
    ReceiverSide,
    Role,
    Site,
    SiteMap,
    SiteSpec,
)
from lofi.registry import ComponentPlugin
from lofi.spec.parsers.chart import InsideOutsideParserSettings, InsideOutsideParserSpec
from lofi.spec.parsers.reduction import ReductionResolver

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from types import TracebackType

    from torch import Tensor

    from lofi.nn.build import BuildContext
    from lofi.nn.ports import Controller, Patch, SubwordPooler
    from lofi.nn.sites import ComponentId
    from lofi.spec.parsers.chart import OutsideWeights, RootOutside

__all__ = ["DeepInsideOutside"]

_LEAF_AXES: Final[tuple[str, ...]] = ("item", "word", "width")
"""The axes pooled leaves arrive over."""

_SPAN_AXES: Final[tuple[str, ...]] = ("item", "span")
"""The axes of a scalar chart readout."""

_SPLIT_AXES: Final[tuple[str, ...]] = ("item", "span", "split")
"""The axes of a readout with one value per split of every span."""

_CELL_AXES: Final[tuple[str, ...]] = ("item", "span", "width")
"""The axes of the vector chart readout."""

_SPAN_VECTOR_AXES: Final[tuple[str, ...]] = ("item", "span", "span_width")
"""The axes of the concatenated inside and outside span representation."""

_RECEIVER_AXES: Final[tuple[str, ...]] = ("item", "edge", "width")
"""The axes of chart cell activations at the receivers that consume them."""

_ITEM_AXES: Final[tuple[str, ...]] = ("item",)
"""The axes of a readout with one value per item."""

_PACKED_AXES: Final[tuple[str, ...]] = ("span", "item", "width")
"""The packed layout the rule scorer reads cells in."""

_SITES: Final[Mapping[str, tuple[Role, tuple[str, ...], DTypeClass]]] = (
    MappingProxyType(
        {
            "inside": (Role.INSIDE_SCORE, _SPAN_AXES, DTypeClass.REAL),
            "outside": (Role.OUTSIDE_SCORE, _SPAN_AXES, DTypeClass.REAL),
            "alpha": (Role.SPLIT_DISTRIBUTION, _SPLIT_AXES, DTypeClass.LOG_PROBABILITY),
            "mu": (Role.CONSTITUENT_MARGINAL, _SPAN_AXES, DTypeClass.PROBABILITY),
            "log_z": (Role.LOG_PARTITION, _ITEM_AXES, DTypeClass.REAL),
            "identity": (Role.IDENTITY_SCORE, _SPLIT_AXES, DTypeClass.REAL),
            "cells": (Role.CHART_CELL, _CELL_AXES, DTypeClass.REAL),
            "outside_cells": (Role.OUTSIDE_CELL, _CELL_AXES, DTypeClass.REAL),
            "spans": (
                Role.SPAN_REPRESENTATION,
                _SPAN_VECTOR_AXES,
                DTypeClass.REAL,
            ),
        }
    )
)
"""The role, the axes and the numeric class of every readout, by its name."""

#: The input length of the chart whose one split is a pair of adjacent cells.
_PAIR: Final[int] = 2


def _require_widths(
    slot: str,
    function: SpanFunctionModule,
    input_widths: tuple[int, ...],
    output_width: int,
    /,
) -> None:
    """Refuse a span function whose signature does not match its chart slot."""
    if function.input_widths != input_widths:
        msg = (
            f"the {slot} span function reads widths {function.input_widths}, "
            f"not {input_widths}"
        )
        raise ValueError(msg)

    if function.output_width != output_width:
        msg = (
            f"the {slot} span function returns width {function.output_width}, "
            f"not {output_width}"
        )
        raise ValueError(msg)


class _ActionSlabs:
    """Scores the splits of each width through the rule scorer and keeps the actions.

    A split's weight is the rule scorer's split score, which sums every action's
    weight but the null action's.

    Parameters
    ----------
    rules : RuleScorer
        What weighs every action over a pair of adjacent cells.
    """

    __slots__ = ("_receiver", "_rules", "_slabs")

    def __init__(
        self, rules: RuleScorer, receiver: _ReceiverEdges | None = None
    ) -> None:
        self._rules = rules
        self._receiver = receiver
        self._slabs: dict[int, Tensor] = {}

    def __call__(self, left: Tensor, right: Tensor) -> Tensor:
        """Return the log weight of every split of one width, summed over its actions.

        Parameters
        ----------
        left : Tensor
            The left child of every split, rank 4 over (item, span, split, width).
        right : Tensor
            The right child of every split, over the same axes.

        Returns
        -------
        Tensor
            A rank-3 tensor over (item, span, split) in the cells' floating point
            dtype.
        """
        receiver = self._receiver

        if receiver is not None:
            left, right = receiver.receive(left, right)

        items, spans, splits, width = left.shape
        rows = left.reshape(-1, width)
        pair = SpanIndex(_PAIR)
        # a chart over two tokens has one split, whose children are its two leaves
        cells = torch.stack((rows, right.reshape(-1, width), torch.zeros_like(rows)))
        scored = self._rules.score(Encoded(cells, _PACKED_AXES), index=pair)
        span = pair.span_id(0, _PAIR)
        actions = scored.log_scores.tensor[:, span, 0, :]
        self._slabs[splits] = actions.reshape(items, spans, splits, -1)

        return scored.split_scores.tensor[:, span, 0].reshape(items, spans, splits)

    def assembled(self, like: Tensor, index: SpanIndex) -> RuleScores:
        """Return the action scores of every width as one record over the index.

        Parameters
        ----------
        like : Tensor
            A tensor of rank 2 over (item, token) to size the one-token rows off.
        index : SpanIndex
            The span addressing the sweep ran under.

        Returns
        -------
        RuleScores
            Rank 4 over (item, span, split, action), with negative infinity at every
            split a span does not have.
        """
        items = int(like.shape[0])
        actions = len(self._rules.action_set.actions)
        rows = [like.new_full((items, index.n, index.max_splits, actions), -inf)]

        for splits in sorted(self._slabs):
            slab = self._slabs[splits]
            padding = index.max_splits - splits
            spans = int(slab.shape[1])
            rows.append(
                torch.cat(
                    (slab, slab.new_full((items, spans, padding, actions), -inf)),
                    dim=2,
                )
            )

        real = like.new_tensor(index.split_mask, dtype=torch.bool)
        action_set = self._rules.action_set
        null = action_set.null

        return RuleScores(
            torch.cat(rows, dim=1),
            real.unsqueeze(0).expand(items, -1, -1),
            index,
            null=None if null is None else action_set.index(null.name),
        )


class _AttendedSlabs:
    """Score splits locally while deferring controller actions until outside."""

    __slots__ = ("_receiver", "_reduction", "_weigher")

    def __init__(
        self,
        weigher: SplitWeigher,
        reduction: LogWeightReduction,
        receiver: _ReceiverEdges | None = None,
    ) -> None:
        self._weigher = weigher
        self._reduction = reduction
        self._receiver = receiver

    def __call__(self, left: Tensor, right: Tensor) -> Tensor:
        """Return learned local-attention log weights over each span's splits."""
        receiver = self._receiver

        if receiver is not None:
            left, right = receiver.receive(left, right)

        shape = left.shape[:-1]
        local = left.new_zeros(shape)
        real = torch.ones(shape, dtype=torch.bool, device=left.device)

        return self._weigher.weights(
            local,
            local,
            left,
            right,
            real,
            self._reduction,
            dim=-1,
        )


def _lengths(leaves: Encoded) -> Tensor | None:
    """Return how many words each item holds, or None where every item is full.

    Parameters
    ----------
    leaves : Encoded
        The pooled leaves, masked over the item and word axes where ragged.

    Returns
    -------
    Tensor | None
        A rank-1 int64 tensor over the items, or None where no mask was given.
    """
    mask = leaves.mask

    if mask is None:
        return None

    return mask.reshape(mask.shape[0], -1).sum(dim=-1)


def _live_spans(lengths: Tensor | None, like: Tensor, index: SpanIndex) -> Tensor:
    """Return which spans lie within each item.

    Parameters
    ----------
    lengths : Tensor | None
        How many words each item holds, or None where every item is full.
    like : Tensor
        A tensor of rank 2 over (item, span) to place the mask beside.
    index : SpanIndex
        The span addressing the mask is over.

    Returns
    -------
    Tensor
        A rank-2 boolean tensor over (item, span).
    """
    if lengths is None:
        return like.new_ones(like.shape, dtype=torch.bool)

    ends = like.new_tensor(
        [end for _, end in index.spans_by_length()], dtype=torch.long
    )

    return ends.unsqueeze(0) <= lengths.to(torch.long).unsqueeze(-1)


class _Reading(NamedTuple):
    """One exact value selected from a scorer pass's returned readouts."""

    site: Site
    tensor: Tensor
    axes: tuple[str, ...]
    mask: Tensor | None = None


class _InstrumentScope:
    """One nestable observation and patch region over selected chart sites."""

    def __init__(
        self,
        sites: frozenset[Site],
        patches: Mapping[Site, Patch[_Reading]],
        order: tuple[Site, ...],
        enter: Callable[[_InstrumentScope], None],
        leave: Callable[[_InstrumentScope], None],
    ) -> None:
        self._sites = sites
        self._patches = MappingProxyType(dict(patches))
        self._order = order
        self._enter = enter
        self._leave = leave
        self._readings: dict[Site, _Reading] = {}
        self._entered = False
        self._closed = False

    @property
    def readings(self) -> Mapping[Site, _Reading]:
        """Return the latest reading taken at every selected site."""
        return MappingProxyType(
            {
                site: self._readings[site]
                for site in self._order
                if site in self._readings
            }
        )

    def selects(self, site: Site, /) -> bool:
        """Return whether this open scope selected a site."""
        return not self._closed and site in self._sites

    def affects(self, site: Site, /) -> bool:
        """Return whether this open scope observes or patches a site."""
        return not self._closed and (site in self._sites or site in self._patches)

    def replace(self, reading: _Reading, /) -> Tensor:
        """Return the value this scope passes onward at one site."""
        patch = self._patches.get(reading.site)

        if patch is None:
            return reading.tensor

        replacement = patch.stands_in_for(reading)

        if tuple(replacement.shape) != tuple(reading.tensor.shape):
            msg = (
                f"the patch at {reading.site!r} returned shape "
                f"{tuple(replacement.shape)}, not {tuple(reading.tensor.shape)}"
            )
            raise ValueError(msg)

        return replacement

    def record(self, reading: _Reading, /) -> None:
        """Keep a reading selected by this open scope."""
        if self.selects(reading.site):
            self._readings[reading.site] = reading

    def __enter__(self) -> Self:
        """Open this observation region."""
        if self._entered:
            msg = "a chart instrumentation scope cannot be entered twice"
            raise RuntimeError(msg)

        self._entered = True
        self._enter(self)

        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
        /,
    ) -> None:
        """Close this observation region after success or failure."""
        del kind, error, traceback
        self._closed = True
        self._leave(self)


class _ReceiverEdges:
    """Carries receiver-local chart activations through one instrumented pass."""

    def __init__(
        self,
        site: Site,
        edges: tuple[ReceiverEdge, ...],
        scopes: tuple[_InstrumentScope, ...],
        live: Tensor,
        like: Tensor,
    ) -> None:
        self._site = site
        self._edges = edges
        self._scopes = scopes
        self._like = like
        self._slabs: list[Tensor] = []
        self._histories = {scope: [] for scope in scopes}
        self._pair: tuple[Tensor, Tensor] | None = None
        targets = like.new_tensor([edge.target for edge in edges], dtype=torch.long)
        self._mask = live.index_select(1, targets)

    def _assembled(self, slabs: list[Tensor]) -> Tensor:
        """Return processed edge rows followed by masked zero rows."""
        items = int(self._like.shape[0])
        width = int(self._like.shape[-1])

        if slabs:
            held = torch.cat(slabs, dim=1)
        else:
            held = self._like.new_empty((items, 0, width))

        missing = len(self._edges) - int(held.shape[1])

        if missing:
            held = torch.cat((held, held.new_zeros((items, missing, width))), dim=1)

        return held

    def _reading(self, slabs: list[Tensor]) -> _Reading:
        """Return the receiver reading at the reach that consumes the last slab.

        The tensor holds every edge processed so far, but the mask covers only the
        live edges of the last slab, since those are the only rows this reach's
        value is consumed at. The reaches of one pass therefore cover disjoint
        edges, and together they cover every live edge.
        """
        tensor = self._assembled(slabs)
        covered = sum(int(slab.shape[1]) for slab in slabs)
        start = covered - int(slabs[-1].shape[1])
        positions = torch.arange(len(self._edges), device=tensor.device).unsqueeze(0)
        mask = self._mask & positions.ge(start) & positions.lt(covered)

        return _Reading(self._site, tensor, _RECEIVER_AXES, mask)

    def receive(self, left: Tensor, right: Tensor) -> tuple[Tensor, Tensor]:
        """Return the exact receiver inputs after every active scope's patch."""
        items, spans, splits, width = left.shape
        current = torch.stack((left, right), dim=3).reshape(items, -1, width)

        for scope in self._scopes:
            history = self._histories[scope]
            before = current
            reading = self._reading([*history, before])
            replacement = scope.replace(reading)
            start = sum(int(slab.shape[1]) for slab in history)
            stop = start + int(before.shape[1])
            history.append(before)
            current = replacement[:, start:stop]

        self._slabs.append(current)
        paired = current.reshape(items, spans, splits, 2, width)
        self._pair = (
            paired[:, :, :, ReceiverSide.LEFT.value],
            paired[:, :, :, ReceiverSide.RIGHT.value],
        )

        return self._pair

    def composition_pair(self, left: Tensor, right: Tensor, /) -> tuple[Tensor, Tensor]:
        """Return the same receiver inputs in the composition's packed order."""
        pair = self._pair

        if pair is None:
            msg = "receiver-local composition has no scored child pair to consume"
            raise RuntimeError(msg)

        self._pair = None
        packed = tuple(
            child.permute(1, 2, 0, 3).reshape(-1, child.shape[-1]) for child in pair
        )

        if tuple(packed[0].shape) != tuple(left.shape):
            msg = (
                "receiver-local scoring and composition disagree on the left "
                f"child shape: {tuple(packed[0].shape)} and {tuple(left.shape)}"
            )
            raise RuntimeError(msg)

        if tuple(packed[1].shape) != tuple(right.shape):
            msg = (
                "receiver-local scoring and composition disagree on the right "
                f"child shape: {tuple(packed[1].shape)} and {tuple(right.shape)}"
            )
            raise RuntimeError(msg)

        return (packed[0], packed[1])

    def finish(self) -> None:
        """Publish one complete reading at each active observation boundary."""
        if self._pair is not None:
            msg = "a receiver-local child pair was scored but not composed"
            raise RuntimeError(msg)

        if sum(int(slab.shape[1]) for slab in self._slabs) != len(self._edges):
            msg = "the chart pass did not visit every declared receiver edge"
            raise RuntimeError(msg)

        for scope in self._scopes:
            if scope.selects(self._site):
                scope.record(
                    _Reading(
                        self._site,
                        self._assembled(self._histories[scope]),
                        _RECEIVER_AXES,
                        self._mask,
                    )
                )


def _site_map(
    component: ComponentId, receiver: Site, published: frozenset[str]
) -> SiteMap:
    """Return every readout and receiver edge site of one scorer component."""
    sites = {
        Site(component, role, (name,)): SiteSpec(
            axes,
            dtype_class,
            differentiable=True,
            access=Access.OBSERVE,
        )
        for name, (role, axes, dtype_class) in _SITES.items()
        if name in published
    }
    sites[receiver] = SiteSpec(
        _RECEIVER_AXES,
        DTypeClass.REAL,
        differentiable=True,
        access=Access.OBSERVE | Access.PATCH,
    )

    return SiteMap(
        component,
        ComponentKind.PARSER,
        sites,
    )


class DeepInsideOutside(nn.Module):
    """Scores a chart over pooled leaves and publishes every readout it computes.

    Each leaf enters the chart as a projection of its pooled word vector with a
    log weight of zero. The inside sweep proceeds by width: the rule scorer weighs
    every action over the two child cells at every split, the split's log weight is
    the sum over actions, the scalar inside weight is the sum over splits, and the
    cell is the composition of the children mixed under the split posterior. The
    outside sweep runs over the same split scores from each item's root, and the
    constituent marginal is the product of the two passes over the partition.

    Parameters
    ----------
    rules : RuleScorer
        What weighs every action over a pair of adjacent cells, whose input width is
        the width of every cell.
    composition : ChartCompositionModule
        What projects pooled leaves and composes paired child cells at the chart's
        declared node width.
    component : ComponentId
        The handle the run assigned this component, which its published points are
        named under.
    reduction : LogWeightReduction
        How alternative derivations and their cell vectors are combined.
    weigher : SplitWeigher
        How the candidate splits are normalized before their vectors are mixed.
    checkpoint_from_width : int | None
        The narrowest span width whose cells are mixed under activation
        checkpointing, or None to hold every width's activations.

    Raises
    ------
    ValueError
        If the controller's input width is not its representation width, or a
        span-input controller is given a weigher that reads no child vectors.
    """

    def __init__(
        self,
        rules: RuleScorer,
        *,
        composition: ChartCompositionModule,
        split_left: SpanFunctionModule,
        split_right: SpanFunctionModule,
        outside_weights: OutsideWeights,
        root_outside: RootOutside,
        root_init_gain: float,
        component: ComponentId,
        reduction: LogWeightReduction,
        weigher: SplitWeigher,
        checkpoint_from_width: int | None,
    ) -> None:
        super().__init__()
        width = composition.output_width

        controller_width = width * (2 if rules.controller.input == "span" else 1)

        if rules.input_width != controller_width:
            msg = (
                f"the controller reads width {rules.input_width}, "
                f"not the {rules.controller.input} representation width "
                f"{controller_width}"
            )
            raise ValueError(msg)

        if rules.controller.input == "span" and not weigher.reads_children:
            msg = (
                "a span-input controller scores splits through a weigher that "
                "reads the child vectors, and this weigher reads none"
            )
            raise ValueError(msg)

        _require_widths("split_left", split_left, (width, width), width)
        _require_widths("split_right", split_right, (width, width), width)
        self._composition = composition
        self._rules = rules
        self._learned_weigher: nn.Module | None = (
            weigher if isinstance(weigher, nn.Module) else None
        )
        self._channel = SplitChannel(
            self.compose,
            weigher=weigher,
            reduction=reduction,
            checkpoint_from_width=checkpoint_from_width,
        )
        self._outside = OutsideKernel(reduction) if reduction.probabilistic else None
        self._vector_outside = (
            VectorOutsideKernel(
                split_left,
                split_right,
                width=width,
                weighting=outside_weights,
                root=root_outside,
                root_init_gain=root_init_gain,
            )
            if reduction.probabilistic
            else None
        )
        self._component = component
        self._input_width = composition.input_width
        self._receiver_site = Site(component, Role.RECEIVER_EDGE, ("receiver",))
        self._sites = _site_map(component, self._receiver_site, self.publishes)
        self._scopes: list[_InstrumentScope] = []
        self._receiver_passes: list[_ReceiverEdges] = []

    @property
    def publishes(self) -> frozenset[str]:
        """Return the readouts this scorer fills, by the name each has on the record."""
        always = frozenset({"inside", "log_z", "identity", "cells", "actions"})

        if self.reduction.probabilistic:
            outside = frozenset({"outside", "alpha", "mu", "spans"})

            if self.contextual_outside:
                outside |= frozenset({"outside_cells"})

            return always | outside

        return always

    @property
    def reduction(self) -> LogWeightReduction:
        """Return how the chart combines alternative derivations."""
        return self._channel.reduction

    @property
    def input_width(self) -> int:
        """Return the size of the feature axis of the leaves this scorer reads."""
        return self._input_width

    @property
    def output_width(self) -> int:
        """Return the width of every chart cell representation."""
        return self._composition.output_width

    @property
    def span_width(self) -> int:
        """Return the width of the concatenated inside-outside representation."""
        return 2 * self.output_width

    @property
    def contextual_outside(self) -> bool:
        """Return whether published outside vectors exclude their own span."""
        kernel = self._vector_outside

        return kernel is not None and kernel.weighting == "contextual"

    @property
    def rules(self) -> RuleScorer:
        """Return what weighs every action over a pair of adjacent cells."""
        return self._rules

    @property
    def shared(self) -> Mapping[str, SharedSubmodule[nn.Module]]:
        """Return the cell composition published to represented spaces."""
        return MappingProxyType(
            {
                "combine": SharedSubmodule(
                    self._composition,
                    ModuleSharing(in_parameters=False, in_state_dict=False),
                )
            }
        )

    @property
    def receiver_site(self) -> Site:
        """Return the observe-and-patch site holding receiver edge activations.

        A pass reaches the site once per chart width. Each reach hands a patch the
        edges processed so far, masked to the live edges that width consumes, so
        the reaches of one pass cover disjoint edges and a gradient at an edge
        belongs to the one reach that covers it. An observation records every
        edge once, masked to the live edges, after the pass.
        """
        return self._receiver_site

    def receiver_edges(self, index: SpanIndex, /) -> tuple[ReceiverEdge, ...]:
        """Return each child-to-receiver edge in its activation-axis order."""
        edges: list[ReceiverEdge] = []

        for target, (left, right, real) in enumerate(
            zip(
                index.left_child_ids,
                index.right_child_ids,
                index.split_mask,
                strict=True,
            )
        ):
            for split, exists in enumerate(real):
                if not exists:
                    continue

                edges.append(
                    ReceiverEdge(
                        len(edges), left[split], target, split, ReceiverSide.LEFT
                    )
                )
                edges.append(
                    ReceiverEdge(
                        len(edges), right[split], target, split, ReceiverSide.RIGHT
                    )
                )

        return tuple(edges)

    def compose(self, left: Tensor, right: Tensor) -> Tensor:
        """Return the cell composed from a pair of child cells.

        Parameters
        ----------
        left : Tensor
            The left child of each pair, rank 2 over (pair, width).
        right : Tensor
            The right child of each pair, in the order of the pair axis.

        Returns
        -------
        Tensor
            One vector per pair, over the same axes.
        """
        if self._receiver_passes:
            left, right = self._receiver_passes[-1].composition_pair(left, right)

        return self._composition.compose(left, right)

    def readouts(self, leaves: Encoded, /) -> ChartReadouts:
        """Return the quantities the chart computed for a batch.

        Parameters
        ----------
        leaves : Encoded
            One vector per word of each item, over the axes ``("item", "word",
            "width")``, masked over the item and word axes where the batch is ragged.

        Returns
        -------
        ChartReadouts
            Every readout filled, each masked to the spans that lie within its item,
            with negative infinity at every log space position outside them.

        Raises
        ------
        ValueError
            If the leaves are not over those three axes, or hold no words.
        """
        if leaves.axes != _LEAF_AXES:
            msg = f"pooled leaves run over {_LEAF_AXES}, not over {leaves.axes}"
            raise ValueError(msg)

        index = SpanIndex(leaves.size("word"))

        if index.n < 1:
            msg = "a chart spans at least one word, and the leaves hold none"
            raise ValueError(msg)

        words = leaves.tensor
        mask = leaves.mask

        if mask is not None:
            widened = mask.reshape(tuple(mask.shape) + (1,) * (words.ndim - mask.ndim))
            words = words.masked_fill(~widened, 0.0)

        vectors = self._composition.leaf(words)
        leaf_scores = vectors.new_zeros(vectors.shape[:2])
        lengths = _lengths(leaves)
        live = _live_spans(
            lengths,
            leaf_scores.new_empty((int(leaf_scores.shape[0]), index.num_spans)),
            index,
        )
        receiver_scopes = tuple(
            scope for scope in self._scopes if scope.affects(self._receiver_site)
        )
        receiver = (
            _ReceiverEdges(
                self._receiver_site,
                self.receiver_edges(index),
                receiver_scopes,
                live,
                vectors,
            )
            if receiver_scopes
            else None
        )
        input_source = self._rules.controller.input
        action_slabs = (
            _ActionSlabs(self._rules, receiver) if input_source == "inside" else None
        )
        scoring = (
            action_slabs
            if action_slabs is not None
            else _AttendedSlabs(self._channel.weigher, self.reduction, receiver)
        )

        if receiver is not None:
            self._receiver_passes.append(receiver)

        try:
            chart, split_scores = self._channel.sweep_scoring(
                vectors, leaf_scores, scoring, index, recompute=receiver is None
            )
        finally:
            if receiver is not None:
                popped = self._receiver_passes.pop()

                if popped is not receiver:
                    msg = "receiver-local scorer passes must close inside out"
                    raise RuntimeError(msg)

        inside = chart.inside
        log_z = inside.gather(1, _root_ids(inside, index, lengths).unsqueeze(1))
        real = inside.new_tensor(index.split_mask, dtype=torch.bool).unsqueeze(0)
        live_splits = live.unsqueeze(-1) & real
        actions: ChartActionScores | None = None

        if action_slabs is not None:
            scored_actions = action_slabs.assembled(leaf_scores, index).log_scores
            actions = ChartActionScores(
                self._rules.action_set,
                LogDistribution(
                    scored_actions.tensor.masked_fill(~live_splits.unsqueeze(-1), -inf),
                    scored_actions.axes,
                    live_splits,
                ),
            )

        outside_kernel = self._outside
        outside_readout: LogDistribution | None = None
        alpha_readout: LogDistribution | None = None
        mu_readout: Encoded | None = None
        outside_cells_readout: Encoded | None = None
        spans_readout: Encoded | None = None

        if outside_kernel is not None:
            outside = outside_kernel.outside(
                inside, split_scores, index, lengths=lengths
            )
            mu = log_constituent_marginals(
                inside, outside, index, lengths=lengths
            ).exp()
            outside_readout = LogDistribution(
                outside.masked_fill(~live, -inf), _SPAN_AXES, live
            )
            alpha_readout = LogDistribution(
                chart.alpha.masked_fill(~live_splits, -inf),
                _SPLIT_AXES,
                live_splits,
                normalized_over=_SPLIT_AXES[-1],
            )
            mu_readout = Encoded(mu.masked_fill(~live, 0.0), _SPAN_AXES, live)
            vector_outside = self._vector_outside

            if vector_outside is None:
                msg = "a probabilistic chart has no vector outside kernel"
                raise RuntimeError(msg)

            inside_cells = chart.cells.transpose(0, 1)
            outside_cells = vector_outside(
                chart.cells,
                inside,
                outside,
                split_scores,
                chart.alpha,
                index,
                lengths=lengths,
            ).transpose(0, 1)
            encoded_outside = Encoded(
                outside_cells.masked_fill(~live.unsqueeze(-1), 0.0),
                _CELL_AXES,
                live,
            )
            outside_cells_readout = encoded_outside if self.contextual_outside else None
            spans_readout = Encoded(
                torch.cat((inside_cells, outside_cells), dim=-1).masked_fill(
                    ~live.unsqueeze(-1), 0.0
                ),
                _SPAN_VECTOR_AXES,
                live,
            )

        if input_source == "span":
            if spans_readout is None:
                msg = "a span-input controller requires outside span representations"
                raise RuntimeError(msg)

            packed_spans = Encoded(
                spans_readout.tensor.transpose(0, 1),
                _PACKED_AXES,
                live.transpose(0, 1),
            )
            scored_actions = self._rules.score(packed_spans, index=index).log_scores
            actions = ChartActionScores(
                self._rules.action_set,
                LogDistribution(
                    scored_actions.tensor.masked_fill(~live_splits.unsqueeze(-1), -inf),
                    scored_actions.axes,
                    live_splits,
                ),
            )

        if actions is None:
            msg = "a chart pass did not score its controller actions"
            raise RuntimeError(msg)

        readouts = ChartReadouts(
            inside=LogDistribution(inside.masked_fill(~live, -inf), _SPAN_AXES, live),
            outside=outside_readout,
            alpha=alpha_readout,
            mu=mu_readout,
            log_z=LogDistribution(log_z.squeeze(1), _ITEM_AXES),
            actions=actions,
            cells=Encoded(chart.cells.transpose(0, 1), _CELL_AXES, live),
            outside_cells=outside_cells_readout,
            spans=spans_readout,
        )

        if receiver is not None:
            receiver.finish()

        if self._scopes:
            self._publish(readouts)

        return readouts

    def instrument(
        self,
        sites: frozenset[Site],
        /,
        *,
        patches: Mapping[Site, Patch[_Reading]],
    ) -> _InstrumentScope:
        """Return a nestable instrumentation scope over selected chart points.

        Parameters
        ----------
        sites : frozenset[Site]
            The published chart points to observe.
        patches : Mapping[Site, Patch[_Reading]]
            Replacements at receiver edge activations.

        Raises
        ------
        ValueError
            If a requested site is not published, or does not permit patching.
        """
        for site in sites:
            if site not in self._sites:
                msg = f"the chart scorer does not publish {site!r}"
                raise ValueError(msg)

        for site in patches:
            if site not in self._sites:
                msg = f"the chart scorer does not publish {site!r}"
                raise ValueError(msg)

            if Access.PATCH not in self._sites[site].access:
                msg = f"the chart scorer site {site!r} is observation-only"
                raise ValueError(msg)

        return _InstrumentScope(
            sites, patches, tuple(self._sites), self._attach, self._detach
        )

    def _publish(self, readouts: ChartReadouts, /) -> None:
        """Record selected objects from the readouts this pass will return."""
        values: dict[Site, LogDistribution | Encoded | None] = {
            Site(self._component, Role.INSIDE_SCORE, ("inside",)): readouts.inside,
            Site(self._component, Role.OUTSIDE_SCORE, ("outside",)): readouts.outside,
            Site(
                self._component,
                Role.SPLIT_DISTRIBUTION,
                ("alpha",),
            ): readouts.alpha,
            Site(
                self._component,
                Role.CONSTITUENT_MARGINAL,
                ("mu",),
            ): readouts.mu,
            Site(self._component, Role.LOG_PARTITION, ("log_z",)): readouts.log_z,
            Site(
                self._component,
                Role.IDENTITY_SCORE,
                ("identity",),
            ): readouts.identity,
            Site(self._component, Role.CHART_CELL, ("cells",)): readouts.cells,
            Site(
                self._component, Role.OUTSIDE_CELL, ("outside_cells",)
            ): readouts.outside_cells,
            Site(self._component, Role.SPAN_REPRESENTATION, ("spans",)): readouts.spans,
        }

        for site, value in values.items():
            selected = tuple(scope for scope in self._scopes if scope.selects(site))

            if not selected:
                continue

            if value is None:
                msg = f"the chart scorer declared {site!r}, but its pass left it empty"
                raise RuntimeError(msg)

            reading = _Reading(site, value.tensor, value.axes, value.mask)

            for scope in selected:
                scope.record(reading)

    def _attach(self, scope: _InstrumentScope) -> None:
        """Add one scope to the active nesting stack."""
        self._scopes.append(scope)

    def _detach(self, scope: _InstrumentScope) -> None:
        """Remove the innermost scope from the active nesting stack."""
        if not self._scopes or self._scopes[-1] is not scope:
            msg = "chart instrumentation scopes must close inside out"
            raise RuntimeError(msg)

        self._scopes.pop()

    def site_map(self) -> SiteMap:
        """Return the points an observer may name inside this scorer.

        Returns
        -------
        SiteMap
            One point per named readout and one receiver edge activation point, over
            the axes each projection is published over.
        """
        return self._sites

    @override
    def extra_repr(self) -> str:
        """Return the leaf width and the component handle."""
        return f"input_width={self._input_width}, component={self._component!r}"


def _build(
    settings: InsideOutsideParserSettings, context: BuildContext, /
) -> DeepInsideOutside | StackedInsideOutside:
    """Build and place a chart scored by an inside and an outside sweep.

    Parameters
    ----------
    settings : InsideOutsideParserSettings
        The chart's settings, which hold nothing.
    context : BuildContext
        The action inventory, the pooler wired to ``leaves``, the controller wired
        to ``rules``, the component handle and the device policy.

    Raises
    ------
    ValueError
        If the controller does not retain the exact action inventory of the run.
    """
    controller = cast("Controller", context.producer("rules"))
    pooler = cast("SubwordPooler", context.producer("leaves"))

    if controller.action_set is not context.actions:
        expected = tuple(action.name for action in context.actions.actions)
        received = tuple(action.name for action in controller.action_set.actions)
        msg = (
            f"controller action inventory {received} is not the exact supplied "
            f"inventory {expected}"
        )
        raise ValueError(msg)

    arities = action_arities(context.actions, context.rules)
    unsupported = tuple(arity for arity in arities if arity != _PAIR)

    if unsupported:
        arity = unsupported[0]
        msg = (
            f"chart composition {settings.composition.kind!r} cannot serve rule "
            f"arity {arity}; the inside-outside chart composes pairs"
        )
        raise ValueError(msg)

    width = settings.width

    controller_width = width * (2 if controller.input == "span" else 1)

    if controller.input_width != controller_width:
        msg = (
            f"the controller reads width {controller.input_width}, "
            f"not the {controller.input} representation width {controller_width}"
        )
        raise ValueError(msg)

    def build_layer(*, register_controller: bool) -> DeepInsideOutside:
        split_left = build_span_function(
            settings.split_left,
            (width, width),
            width,
        )
        split_right = (
            split_left
            if settings.share_splits
            else build_span_function(
                settings.split_right,
                (width, width),
                width,
            )
        )

        return DeepInsideOutside(
            RuleScorer(
                controller,
                width=controller_width,
                register_controller=register_controller,
            ),
            composition=build_chart_composition(
                settings.composition,
                pooler.output_width,
                width,
            ),
            split_left=split_left,
            split_right=split_right,
            outside_weights=settings.outside_weights,
            root_outside=settings.root_outside,
            root_init_gain=settings.root_init_gain,
            component=context.component,
            checkpoint_from_width=context.policy.checkpoint_from_width,
            reduction=settings.reduction.resolve(_REDUCTIONS),
            weigher=build_split_weigher(settings.attend, width),
        )

    first = build_layer(register_controller=settings.depth == 1)

    if settings.depth == 1:
        return place(first, context.policy)

    def build_transition() -> SpanTransition:
        span_width = 2 * width

        return SpanTransition(
            build_span_function(
                settings.transition,
                (span_width,),
                pooler.output_width,
            ),
            build_span_function(
                settings.transition,
                (span_width,),
                width,
            ),
        )

    if settings.layer_sharing == "shared":
        layers = (first,) * settings.depth
        bridge = build_transition()
        transitions = (bridge,) * (settings.depth - 1)
    else:
        layers = (
            first,
            *(
                build_layer(register_controller=False)
                for _ in range(settings.depth - 1)
            ),
        )
        transitions = tuple(build_transition() for _ in range(settings.depth - 1))

    return place(
        StackedInsideOutside(
            layers,
            transitions,
            controller=cast("nn.Module", controller),
        ),
        context.policy,
    )


class _ReductionBuilder(ReductionResolver[LogWeightReduction]):
    """Build the tensor reduction selected by a parser declaration."""

    @override
    def log_sum_exp(self) -> LogWeightReduction:
        """Return probability-summing reduction."""
        return LogSumExpReduction()

    @override
    def maximum(self) -> LogWeightReduction:
        """Return best-alternative reduction."""
        return MaximumReduction()


_REDUCTIONS: Final = _ReductionBuilder()
"""Runtime tensor reductions reached through their declared behavior."""


PLUGIN: Final = ComponentPlugin(spec=InsideOutsideParserSpec, build=_build)
"""The inside-outside chart's plugin."""
