"""The endpoint-only span encoder, the control for a recursive inside computation."""

from math import inf
from typing import TYPE_CHECKING, Final, cast, override

import torch
from torch import nn

from lofi.deduce import SpanIndex
from lofi.nn._values import ChartReadouts, Encoded, LogDistribution
from lofi.nn.policy import place
from lofi.nn.sites import (
    Access,
    ComponentKind,
    DTypeClass,
    Role,
    Site,
    SiteMap,
    SiteSpec,
)
from lofi.registry import ComponentPlugin
from lofi.spec.parsers.flat import (
    FlatParserSettings,
    FlatParserSpec,
    FlatProjectionInitialization,
    FlatSpanFeature,
)

if TYPE_CHECKING:
    from torch import Tensor

    from lofi.nn.build import BuildContext
    from lofi.nn.ports import SubwordPooler
    from lofi.nn.sites import ComponentId

__all__ = ["FlatSpanEncoder"]

_LEAF_AXES: Final[tuple[str, ...]] = ("item", "word", "width")
"""The axes pooled leaves arrive over."""

_CELL_AXES: Final[tuple[str, ...]] = ("item", "span", "width")
"""The axes of the flat chart-cell representation."""

_SCORE_AXES: Final[tuple[str, ...]] = ("item", "span")
"""The axes of the control's uniform log inside weights."""

_CELLS: Final[str] = "cells"
"""The name of the endpoint representation as a chart cell."""

_INSIDE: Final[str] = "inside"
"""The name of the control's uniform span weight."""


def _live_spans(
    mask: Tensor | None, starts: Tensor, stops: Tensor, items: int
) -> Tensor:
    """Return which spans of each item cover words that all hold data.

    Parameters
    ----------
    mask : Tensor | None
        True where a word holds data, over the item and word axes, or None where
        every word does.
    starts : Tensor
        The first word of each span, in span id order.
    stops : Tensor
        The word after the last of each span, in span id order.
    items : int
        How many items the batch holds.

    Returns
    -------
    Tensor
        A rank-2 boolean tensor over the item and span axes.
    """
    if mask is None:
        return starts.new_ones((items, int(starts.shape[0])), dtype=torch.bool)

    counts = mask.to(torch.long).cumsum(dim=1)
    running = torch.cat((counts.new_zeros((items, 1)), counts), dim=1)
    covered = running.index_select(1, stops) - running.index_select(1, starts)

    return covered == (stops - starts).unsqueeze(0)


class FlatSpanEncoder(nn.Module):
    """Represents every span by its two endpoint words and nothing between them.

    Each word passes through one projection. A declaration selects which of the
    projected left endpoint, projected right endpoint, and their bilinear form are
    summed into the cell. It also selects whether that projection starts as an
    identity map or with Xavier-uniform weights. No word strictly inside a span
    reaches its cell, which is what makes this encoder the control for a chart whose
    cells recurse over splits. Live spans receive the fixed log weight zero; those
    structural weights let the control enter the chart graph without pretending to
    learn a recursive score.

    Parameters
    ----------
    input_width : int
        How wide a pooled leaf vector is.
    width : int
        How wide a span cell is.
    component : ComponentId
        The handle the run assigned this component, which its published points are
        named under.
    features : tuple[FlatSpanFeature, ...]
        The projected endpoint features summed into each span cell.
    projection_initialization : FlatProjectionInitialization
        How the endpoint projection is initialized.
    """

    def __init__(
        self,
        *,
        input_width: int,
        width: int,
        component: ComponentId,
        features: tuple[FlatSpanFeature, ...],
        projection_initialization: FlatProjectionInitialization,
    ) -> None:
        super().__init__()
        cells = width
        leaf = input_width
        self._unary = nn.Linear(leaf, cells)
        self._pair = (
            nn.Bilinear(cells, cells, cells) if "bilinear" in features else None
        )
        if projection_initialization == "identity":
            nn.init.eye_(self._unary.weight)
        else:
            nn.init.xavier_uniform_(self._unary.weight)
        nn.init.zeros_(self._unary.bias)
        self._input_width = leaf
        self._width = cells
        self._features = features
        self._projection_initialization = projection_initialization
        self._component = component

    @property
    def publishes(self) -> frozenset[str]:
        """Return the readouts this encoder fills, by the name each has on record."""
        return frozenset({_CELLS, _INSIDE})

    @property
    def input_width(self) -> int:
        """Return the size of the feature axis of the leaves this encoder reads."""
        return self._input_width

    @property
    def width(self) -> int:
        """Return the size of the feature axis of the cells this encoder returns."""
        return self._width

    @property
    def output_width(self) -> int:
        """Return the width of each published chart-cell representation."""
        return self._width

    @property
    def features(self) -> tuple[FlatSpanFeature, ...]:
        """Return the projected endpoint features summed into each cell."""
        return self._features

    @property
    def projection_initialization(self) -> FlatProjectionInitialization:
        """Return how the endpoint projection was initialized."""
        return self._projection_initialization

    def cell(self, left: Tensor, right: Tensor) -> Tensor:
        """Return the cell a pair of endpoint words makes.

        Parameters
        ----------
        left : Tensor
            The projected first word of each span, over the cell width innermost.
        right : Tensor
            The projected last word of each span, over the same axes.

        Returns
        -------
        Tensor
            One vector per span, over those same axes.
        """
        result = torch.zeros_like(left)
        if "left" in self._features:
            result = result + left
        if "right" in self._features:
            result = result + right
        if self._pair is not None:
            result = result + self._pair(left, right)
        return result

    def readouts(self, leaves: Encoded, /) -> ChartReadouts:
        """Return every span's fixed inside weight and endpoint representation.

        Parameters
        ----------
        leaves : Encoded
            One vector per word of each item, over the axes ``("item", "word",
            "width")``, masked over the item and word axes where the batch is ragged.

        Returns
        -------
        ChartReadouts
            The cells alone, over the axes ``("item", "span", "width")``, masked to
            the spans whose every word holds data and zero outside them.

        Raises
        ------
        ValueError
            If the leaves are not over those three axes, hold no words, or run over a
            feature axis other than the one this encoder reads.
        """
        if leaves.axes != _LEAF_AXES:
            msg = f"pooled leaves run over {_LEAF_AXES}, not over {leaves.axes}"
            raise ValueError(msg)

        held = leaves.size("width")

        if held != self._input_width:
            msg = f"this encoder reads leaves {self._input_width} wide, not {held} wide"
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

        spans = tuple(index.spans_by_length())
        starts = words.new_tensor([start for start, _ in spans], dtype=torch.long)
        stops = words.new_tensor([stop for _, stop in spans], dtype=torch.long)
        vectors = self._unary(words)
        cells = self.cell(
            vectors.index_select(1, starts), vectors.index_select(1, stops - 1)
        )
        live = _live_spans(mask, starts, stops, int(words.shape[0]))

        return ChartReadouts(
            inside=LogDistribution(
                cells.new_zeros(live.shape).masked_fill(~live, -inf),
                _SCORE_AXES,
                live,
            ),
            cells=Encoded(
                cells.masked_fill(~live.unsqueeze(-1), 0.0), _CELL_AXES, live
            ),
        )

    def site_map(self) -> SiteMap:
        """Return the points an observer may name inside this encoder.

        Returns
        -------
        SiteMap
            The one published readout, keyed by the name it has on the record, over
            the axes it is published over.
        """
        return SiteMap(
            self._component,
            ComponentKind.PARSER,
            {
                Site(self._component, Role.CHART_CELL, (_CELLS,)): SiteSpec(
                    _CELL_AXES,
                    DTypeClass.REAL,
                    differentiable=True,
                    access=Access.OBSERVE,
                ),
            },
        )

    @override
    def extra_repr(self) -> str:
        """Return the widths, features, initialization, and component handle."""
        return (
            f"input_width={self._input_width}, width={self._width}, "
            f"features={self._features!r}, "
            f"projection_initialization={self._projection_initialization!r}, "
            f"component={self._component!r}"
        )


def _build(settings: FlatParserSettings, context: BuildContext, /) -> FlatSpanEncoder:
    """Build and place a chart whose every cell reads the endpoint words of its span.

    Parameters
    ----------
    settings : FlatParserSettings
        The width, endpoint features, and projection initialization of each cell.
    context : BuildContext
        The pooler wired to ``leaves``, the component handle and the device policy.
    """
    pooler = cast("SubwordPooler", context.producer("leaves"))

    return place(
        FlatSpanEncoder(
            input_width=pooler.output_width,
            width=settings.width,
            component=context.component,
            features=settings.features,
            projection_initialization=settings.projection_initialization,
        ),
        context.policy,
    )


PLUGIN: Final = ComponentPlugin(spec=FlatParserSpec, build=_build)
"""The flat span encoder's plugin."""
