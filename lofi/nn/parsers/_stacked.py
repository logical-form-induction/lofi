"""Recurrent neural inside-outside layers over completed span representations."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, cast, override

import torch
from torch import nn

from lofi.deduce import SpanIndex
from lofi.nn._values import ChartReadouts, Encoded

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import Tensor

    from lofi._types import SharedSubmodule
    from lofi.nn.parsers._scorer import DeepInsideOutside, _InstrumentScope, _Reading
    from lofi.nn.parsers.functions import SpanFunctionModule
    from lofi.nn.ports import Patch, ReceiverEdge
    from lofi.nn.psi import RuleScorer
    from lofi.nn.sites import Site, SiteMap

__all__ = ["SpanTransition", "StackedInsideOutside"]

_LEAF_AXES = ("item", "word", "width")
_SPAN_AXES = ("item", "span", "span_width")


class _SpanContextPatch:
    """Add one completed target-span input to each receiver edge."""

    __slots__ = ("_contexts",)

    def __init__(self, contexts: Tensor) -> None:
        self._contexts = contexts

    def stands_in_for(self, reading: _Reading, /) -> Tensor:
        """Return receiver activations conditioned on the layer below."""
        if tuple(reading.tensor.shape) != tuple(self._contexts.shape):
            msg = (
                f"a span transition prepared shape {tuple(self._contexts.shape)}, "
                f"not receiver shape {tuple(reading.tensor.shape)}"
            )
            raise ValueError(msg)

        contexts = self._contexts

        if reading.mask is not None:
            contexts = contexts.masked_fill(~reading.mask.unsqueeze(-1), 0.0)

        return reading.tensor + contexts


class SpanTransition(nn.Module):
    r"""Project one completed span tensor into the next chart layer.

    The one-token rows become the next layer's leaves. Every wider target span
    supplies a contextual input to both receiver edges that construct it. Both
    projections thus read only the fully completed concatenation
    $[\mathbf{h}^{\mathrm{in}};\mathbf{h}^{\mathrm{out}}]$
    returned by the preceding layer.
    """

    def __init__(
        self,
        leaves: SpanFunctionModule,
        receivers: SpanFunctionModule,
    ) -> None:
        super().__init__()

        if len(leaves.input_widths) != 1:
            msg = "a recurrent leaf transition reads one completed span vector"
            raise ValueError(msg)

        if receivers.input_widths != leaves.input_widths:
            msg = (
                f"leaf and receiver transitions read {leaves.input_widths} and "
                f"{receivers.input_widths}, not one shared span width"
            )
            raise ValueError(msg)

        self._leaves = leaves
        self._receivers = receivers

    @property
    def input_width(self) -> int:
        """Return the width of each completed span representation."""
        return self._leaves.input_widths[0]

    @property
    def leaf_width(self) -> int:
        """Return the width of each leaf supplied to the next layer."""
        return self._leaves.output_width

    @property
    def receiver_width(self) -> int:
        """Return the width added at each receiver edge."""
        return self._receivers.output_width

    def leaves(self, spans: Encoded, index: SpanIndex, /) -> Encoded:
        """Return the projected one-token rows of a completed span tensor."""
        _check_spans(spans, index, self.input_width)
        ids = spans.tensor.new_tensor(
            [index.span_id(word, word + 1) for word in range(index.n)],
            dtype=torch.long,
        )
        mask = spans.mask
        leaf_mask = None if mask is None else mask.index_select(1, ids)

        return Encoded(
            self._leaves(spans.tensor.index_select(1, ids)),
            _LEAF_AXES,
            leaf_mask,
        )

    def patch(
        self,
        spans: Encoded,
        index: SpanIndex,
        edges: tuple[ReceiverEdge, ...],
        /,
    ) -> Patch[_Reading]:
        """Return the receiver patch carrying target-span inputs upward."""
        _check_spans(spans, index, self.input_width)
        projected = self._receivers(spans.tensor)
        targets = projected.new_tensor(
            [edge.target for edge in edges], dtype=torch.long
        )
        contexts = projected.index_select(1, targets)

        return _SpanContextPatch(contexts)


def _check_spans(spans: Encoded, index: SpanIndex, width: int, /) -> None:
    """Validate one completed span tensor at its recurrent-layer boundary."""
    if spans.axes != _SPAN_AXES:
        msg = f"completed spans run over {_SPAN_AXES}, not {spans.axes}"
        raise ValueError(msg)

    expected = (spans.size("item"), index.num_spans, width)

    if tuple(spans.tensor.shape) != expected:
        msg = (
            f"a chart over {index.n} words supplies completed spans of shape "
            f"{expected}, not {tuple(spans.tensor.shape)}"
        )
        raise ValueError(msg)


class StackedInsideOutside(nn.Module):
    """Run recurrent chart layers over completed span representations."""

    def __init__(
        self,
        layers: tuple[DeepInsideOutside, ...],
        transitions: tuple[SpanTransition, ...],
        *,
        controller: nn.Module | None = None,
    ) -> None:
        super().__init__()

        if not layers:
            msg = "a recurrent neural chart holds at least one layer"
            raise ValueError(msg)

        if len(transitions) != len(layers) - 1:
            msg = (
                f"{len(layers)} chart layers require one transition between each "
                f"adjacent pair, not {len(transitions)}"
            )
            raise ValueError(msg)

        for lower, transition, upper in zip(
            layers[:-1], transitions, layers[1:], strict=True
        ):
            if not lower.reduction.probabilistic or "spans" not in lower.publishes:
                msg = "every recurrent source layer must publish probabilistic spans"
                raise ValueError(msg)

            if transition.input_width != lower.span_width:
                msg = (
                    f"a transition reads span width {transition.input_width}, "
                    f"not the lower layer's {lower.span_width}"
                )
                raise ValueError(msg)

            if transition.leaf_width != upper.input_width:
                msg = (
                    f"a transition returns leaf width {transition.leaf_width}, "
                    f"not the upper layer's {upper.input_width}"
                )
                raise ValueError(msg)

            if transition.receiver_width != upper.output_width:
                msg = (
                    f"a transition returns receiver width "
                    f"{transition.receiver_width}, not the upper layer's "
                    f"{upper.output_width}"
                )
                raise ValueError(msg)

        self._layers = nn.ModuleList(layers)
        self._transitions = nn.ModuleList(transitions)

        if controller is not None:
            self._controller = controller

    @property
    def depth(self) -> int:
        """Return how many chart layers run."""
        return len(self._layers)

    @property
    def layers(self) -> tuple[DeepInsideOutside, ...]:
        """Return the chart layers in execution order."""
        return cast("tuple[DeepInsideOutside, ...]", tuple(self._layers))

    @property
    def transitions(self) -> tuple[SpanTransition, ...]:
        """Return the completed-span transitions in execution order."""
        return cast("tuple[SpanTransition, ...]", tuple(self._transitions))

    @property
    def publishes(self) -> frozenset[str]:
        """Return the readouts supplied by the last layer."""
        return self.layers[-1].publishes

    @property
    def input_width(self) -> int:
        """Return the pooled-leaf width read by the first layer."""
        return self.layers[0].input_width

    @property
    def output_width(self) -> int:
        """Return the chart-cell width of the last layer."""
        return self.layers[-1].output_width

    @property
    def span_width(self) -> int:
        """Return the completed span width of the last layer."""
        return self.layers[-1].span_width

    @property
    def contextual_outside(self) -> bool:
        """Return whether the last layer publishes content-free outside vectors."""
        return self.layers[-1].contextual_outside

    @property
    def rules(self) -> RuleScorer:
        """Return the rule scorer read by the last chart layer."""
        return self.layers[-1].rules

    @property
    def shared(self) -> Mapping[str, SharedSubmodule[nn.Module]]:
        """Return the represented-space publications of the last layer."""
        return MappingProxyType(dict(self.layers[-1].shared))

    @property
    def receiver_site(self) -> Site:
        """Return the receiver-edge site of the last layer."""
        return self.layers[-1].receiver_site

    def receiver_edges(self, index: SpanIndex, /) -> tuple[ReceiverEdge, ...]:
        """Return receiver-edge metadata in the last layer's public order."""
        return self.layers[-1].receiver_edges(index)

    def readouts(self, leaves: Encoded, /) -> ChartReadouts:
        """Return readouts from the last recurrent chart layer."""
        return self.layer_readouts(leaves)[-1]

    def layer_readouts(self, leaves: Encoded, /) -> tuple[ChartReadouts, ...]:
        """Return every layer's readouts in execution order."""
        completed = [self.layers[0].readouts(leaves)]

        for layer, transition in zip(self.layers[1:], self.transitions, strict=True):
            completed.append(self.refine(layer, transition, completed[-1]))

        return tuple(completed)

    def refine(
        self,
        layer: DeepInsideOutside,
        transition: SpanTransition,
        lower: ChartReadouts,
        /,
        *,
        replacement: Tensor | None = None,
    ) -> ChartReadouts:
        """Run one upper layer from a completed lower-layer span tensor."""
        spans = lower.spans

        if spans is None:
            msg = "a recurrent source layer returned no completed span tensor"
            raise ValueError(msg)

        if replacement is not None:
            spans = Encoded(replacement, spans.axes, spans.mask)

        words = _word_count(spans.size("span"))
        index = SpanIndex(words)
        edges = layer.receiver_edges(index)
        patch = transition.patch(spans, index, edges)
        next_leaves = transition.leaves(spans, index)

        with layer.instrument(frozenset(), patches={layer.receiver_site: patch}):
            return layer.readouts(next_leaves)

    def instrument(
        self,
        sites: frozenset[Site],
        /,
        *,
        patches: Mapping[Site, Patch[_Reading]],
    ) -> _InstrumentScope:
        """Instrument the final layer whose readouts this stack publishes."""
        return self.layers[-1].instrument(sites, patches=patches)

    def site_map(self) -> SiteMap:
        """Return the final layer's published observation points."""
        return self.layers[-1].site_map()

    @override
    def extra_repr(self) -> str:
        """Return the declared depth and boundary widths."""
        return (
            f"depth={self.depth}, input_width={self.input_width}, "
            f"output_width={self.output_width}"
        )


def _word_count(spans: int, /) -> int:
    """Return the unique positive word count with this many contiguous spans."""
    if spans < 1:
        msg = f"a completed chart holds at least one span, not {spans}"
        raise ValueError(msg)

    words = 1

    while words * (words + 1) // 2 < spans:
        words += 1

    if words * (words + 1) // 2 != spans:
        msg = f"{spans} spans do not form a complete contiguous chart"
        raise ValueError(msg)

    return words
