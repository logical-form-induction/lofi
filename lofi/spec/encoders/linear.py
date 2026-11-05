"""Linear-recurrent term encoder declaration."""

from typing import Literal, NamedTuple, Self, override

import didactic.api as dx

from lofi._checks import ensure_name
from lofi.spec.components.model import TermEncoderSpec
from lofi.spec.encoders.rnn import DEFAULT_RECURRENT_LAYERS
from lofi.spec.layers import DEFAULT_RECURRENT_CELL

__all__ = ["DirectionMerge", "LinearRnnEncoderSettings", "LinearRnnEncoderSpec"]


type DirectionMerge = Literal["mean", "sum", "concatenation"]
"""How a bidirectional recurrence combines its forward and backward states."""


class LinearRnnEncoderSettings(NamedTuple):
    """The resolved linear-recurrent encoder settings.

    ``merge`` is how a bidirectional recurrence combines its two directions, and
    None for a forward one.
    """

    width: int
    cell: str
    layers: int
    direction: Literal["forward", "bidirectional"]
    variables: Literal["index", "level"]
    merge: DirectionMerge | None


class LinearRnnEncoderSpec(TermEncoderSpec):
    """Declare recurrence over a codec's linearization order."""

    kind: Literal["linear-rnn"] = "linear-rnn"
    width: int
    cell: str = DEFAULT_RECURRENT_CELL
    layers: int = DEFAULT_RECURRENT_LAYERS
    direction: Literal["forward", "bidirectional"] = "forward"
    variables: Literal["index", "level"] = "index"
    merge: Literal["mean", "sum", "concatenation"] | None = None

    __axioms__ = (
        dx.axiom("width > 0", message="a state of no coordinates holds nothing"),
        dx.axiom("layers > 0", message="a recurrence has a layer"),
    )

    @dx.validates("cell")
    @classmethod
    def _cell_is_named(cls, value: str) -> str:
        """Return a named recurrent-cell plugin."""
        return ensure_name("a recurrent cell", value)

    @dx.model_validator()
    def _merge_matches_direction(self) -> Self:
        """Require a merge exactly when the recurrence runs both directions."""
        if self.direction == "bidirectional" and self.merge is None:
            msg = "a bidirectional recurrence declares how it merges its directions"
            raise ValueError(msg)

        if self.direction == "forward" and self.merge is not None:
            msg = "a forward recurrence has no second direction to merge"
            raise ValueError(msg)

        return self

    @override
    def publications(self) -> frozenset[str]:
        """Return the payload-aware choice embedding available for declared tying."""
        return frozenset({"choice-embedding"})

    def settings(self) -> LinearRnnEncoderSettings:
        """Return the frozen settings consumed by the encoder plugin."""
        return LinearRnnEncoderSettings(
            self.width,
            self.cell,
            self.layers,
            self.direction,
            self.variables,
            self.merge,
        )
