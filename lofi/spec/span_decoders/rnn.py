"""Recurrent span-decoder declaration."""

from typing import TYPE_CHECKING, Final, Literal, NamedTuple, override

import didactic.api as dx

from lofi._checks import ensure_name, ensure_positive
from lofi.spec._ports import Dependency
from lofi.spec.components.model import SpanDecoderSpec
from lofi.spec.layers import DEFAULT_RECURRENT_CELL, FeedForwardLayerSpec

if TYPE_CHECKING:
    from lofi.spec.layers import FeedForwardLayer

__all__ = ["RnnSpanDecoderSettings", "RnnSpanDecoderSpec"]

_DEPENDENCIES: Final[tuple[Dependency, ...]] = (Dependency("parser", "cells"),)
"""The chart whose cell width the decoder reads."""


class RnnSpanDecoderSettings(NamedTuple):
    """The settings a recurrent span decoder is built from.

    Attributes
    ----------
    cell : str
        The cell each recurrent layer steps its state with.
    layers : int
        How many recurrent layers are stacked.
    state_width : int | None
        The width of every recurrent layer's state, or None for the width of the
        span vectors.
    head : tuple[FeedForwardLayer, ...]
        The declared hidden layers of the scoring head.
    """

    cell: str
    layers: int
    state_width: int | None
    head: tuple[FeedForwardLayer, ...]
    end: str


class RnnSpanDecoderSpec(SpanDecoderSpec):
    """Declares a recurrent decoder of the words a chart span covers.

    The decoder reads span vectors at the chart-cell width and scores the run's
    vocabulary, so the declaration names neither.

    Attributes
    ----------
    kind : Literal["rnn-span-decoder"]
        Registry tag.
    cell : str
        The cell each recurrent layer steps its state with.
    layers : int
        How many recurrent layers are stacked.
    state_width : int | None
        The width of every recurrent layer's state. None runs the state at the
        width of the span vectors.
    head : tuple[FeedForwardLayerSpec, ...]
        The declared hidden layers of the scoring head. Empty scores the state with
        one affine map.
    """

    kind: Literal["rnn-span-decoder"] = "rnn-span-decoder"
    cell: str = DEFAULT_RECURRENT_CELL
    layers: int = 1
    state_width: int | None = None
    head: tuple[FeedForwardLayerSpec, ...] = ()
    end: str = "<eos>"

    __axioms__ = (dx.axiom("layers > 0", message="a recurrence has a layer"),)

    @dx.validates("cell")
    @classmethod
    def _cell_is_named(cls, value: str) -> str:
        """Return a named recurrent-cell plugin."""
        return ensure_name("a recurrent cell", value)

    @dx.validates("end")
    @classmethod
    def _end_is_named(cls, value: str) -> str:
        """Return the declared span terminator."""
        return ensure_name("a span-decoder end token", value)

    @dx.validates("state_width")
    @classmethod
    def _width_is_positive_or_absent(cls, value: int | None) -> int | None:
        """Return a positive width, or none."""
        if value is None:
            return None

        return ensure_positive("a declared layer width", value)

    @override
    def dependencies(self) -> tuple[Dependency, ...]:
        """Return the chart whose output width the span vectors run at."""
        return _DEPENDENCIES

    def settings(self) -> RnnSpanDecoderSettings:
        """Return the settings the plugin builds this component from."""
        return RnnSpanDecoderSettings(
            cell=self.cell,
            layers=self.layers,
            state_width=self.state_width,
            head=tuple(FeedForwardLayerSpec.settings(layer) for layer in self.head),
            end=self.end,
        )
