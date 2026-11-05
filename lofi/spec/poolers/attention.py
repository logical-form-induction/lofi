"""Attention subword pooler declaration."""

from typing import TYPE_CHECKING, Literal, NamedTuple, override

import didactic.api as dx

from lofi._checks import ensure_name, ensure_positive
from lofi.spec.components.model import SubwordPoolerSpec
from lofi.spec.layers import DEFAULT_RECURRENT_CELL, FeedForwardLayerSpec
from lofi.spec.poolers._dependencies import ENCODER_DEPENDENCY

if TYPE_CHECKING:
    from lofi.spec._ports import Dependency
    from lofi.spec.layers import FeedForwardLayer

__all__ = [
    "AttentionSubwordPoolerSettings",
    "AttentionSubwordPoolerSpec",
    "SubwordContextSettings",
    "SubwordContextSpec",
]


class SubwordContextSettings(NamedTuple):
    """The settings of the recurrence a scorer reads a word's rows with.

    Attributes
    ----------
    cell : str
        The cell each recurrent layer steps its state with.
    layers : int
        How many recurrent layers are stacked.
    width : int | None
        The width of each direction's state, or None for the width of the rows.
    bidirectional : bool
        Whether the rows are read in both directions rather than left to right.
    """

    cell: str
    layers: int
    width: int | None
    bidirectional: bool


class SubwordContextSpec(dx.Model, extra="forbid"):
    """Declares the recurrence a scorer reads a word's rows with.

    Attributes
    ----------
    cell : str
        The cell each recurrent layer steps its state with.
    layers : int
        How many recurrent layers are stacked.
    width : int | None
        The width of each direction's state. None runs the state at the width of
        the rows.
    bidirectional : bool
        Whether the rows are read in both directions rather than left to right.
    """

    cell: str = DEFAULT_RECURRENT_CELL
    layers: int = 1
    width: int | None = None
    bidirectional: bool = True

    __axioms__ = (dx.axiom("layers > 0", message="a recurrence has a layer"),)

    @dx.validates("cell")
    @classmethod
    def _cell_is_named(cls, value: str) -> str:
        """Return a named recurrent-cell plugin."""
        return ensure_name("a recurrent cell", value)

    @dx.validates("width")
    @classmethod
    def _width_is_positive_or_absent(cls, value: int | None) -> int | None:
        """Return a positive state width, or none."""
        if value is None:
            return None

        return ensure_positive("a recurrent state's width", value)

    def settings(self) -> SubwordContextSettings:
        """Return the settings the pooler builds its recurrence from."""
        return SubwordContextSettings(
            cell=self.cell,
            layers=self.layers,
            width=self.width,
            bidirectional=self.bidirectional,
        )


class AttentionSubwordPoolerSettings(NamedTuple):
    """The settings an attention pooler is built from.

    Attributes
    ----------
    context : SubwordContextSettings | None
        The recurrence the scorer reads a word's rows with, or None to score the
        rows as they are.
    scorer : tuple[FeedForwardLayer, ...]
        The declared hidden layers of the scorer.
    """

    context: SubwordContextSettings | None
    scorer: tuple[FeedForwardLayer, ...]


class AttentionSubwordPoolerSpec(SubwordPoolerSpec):
    """Declares a pooler that weighs each word's rows with a learned scorer.

    Each word's vector is the sum of its rows under the softmax of their scores.

    Attributes
    ----------
    kind : Literal["attention"]
        Registry tag.
    context : SubwordContextSpec | None
        The recurrence the scorer reads a word's rows with before scoring them, or
        None to score the rows as they are.
    scorer : tuple[FeedForwardLayerSpec, ...]
        The declared hidden layers of the scorer. Empty scores each row with one
        linear map.
    """

    kind: Literal["attention"] = "attention"
    context: SubwordContextSpec | None = None
    scorer: tuple[FeedForwardLayerSpec, ...] = ()

    @override
    def dependencies(self) -> tuple[Dependency, ...]:
        """Return the token encoder this pooler gathers rows from."""
        return ENCODER_DEPENDENCY

    def settings(self) -> AttentionSubwordPoolerSettings:
        """Return the settings the plugin builds this component from."""
        return AttentionSubwordPoolerSettings(
            context=None if self.context is None else self.context.settings(),
            scorer=tuple(FeedForwardLayerSpec.settings(layer) for layer in self.scorer),
        )
