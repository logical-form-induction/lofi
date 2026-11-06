"""Registered names and resolved settings for neural layers."""

from typing import Final, Literal, NamedTuple

import didactic.api as dx

from lofi._checks import ensure_finite, ensure_name, ensure_positive

__all__ = [
    "BUILTIN_ACTIVATIONS",
    "BUILTIN_RECURRENT_CELLS",
    "DEFAULT_ACTIVATION",
    "DEFAULT_CHOICE_HEADS",
    "DEFAULT_RECURRENT_CELL",
    "Activation",
    "FeedForwardLayer",
    "FeedForwardLayerSpec",
    "HeadKind",
    "NormPlacement",
    "RecurrentCell",
    "TransformerStructure",
    "dropout_rate",
]

type Activation = str
"""The installed name of an elementwise activation."""

type RecurrentCell = str
"""The installed name of a recurrent-cell implementation."""

type HeadKind = Literal["factored", "flat"]
"""Whether a choice head is one categorical or a family choice then a member."""

type NormPlacement = Literal["pre", "post"]
"""Whether a residual sublayer normalizes its input or its sum."""

type TransformerStructure = str
"""The installed name of the structural relations a Transformer attends with."""

BUILTIN_ACTIVATIONS: Final[tuple[Activation, ...]] = (
    "tanh",
    "relu",
    "gelu",
    "silu",
)
"""Activation plugins shipped with LoFI, in documentation order."""

BUILTIN_RECURRENT_CELLS: Final[tuple[RecurrentCell, ...]] = (
    "gru",
    "lstm",
    "elman-tanh",
    "elman-relu",
)
"""Recurrent-cell plugins shipped with LoFI, in documentation order."""

DEFAULT_ACTIVATION: Final[Activation] = "gelu"
"""The activation a feedforward block applies when its declaration names none."""

DEFAULT_CHOICE_HEADS: Final[HeadKind] = "flat"
"""The choice heads a term decoder emits when its declaration names none."""

DEFAULT_RECURRENT_CELL: Final[RecurrentCell] = "gru"
"""The recurrent cell a component steps with when its declaration names none."""


def dropout_rate(what: str, value: float, /) -> float:
    """Return a finite dropout rate in the half-open unit interval.

    Raises
    ------
    ValueError
        If the rate is not finite or lies outside $[0, 1)$.
    """
    rate = ensure_finite(what, value)

    if not 0.0 <= rate < 1.0:
        msg = f"{what} lies in [0, 1), not {value}"
        raise ValueError(msg)

    return rate


class FeedForwardLayer(NamedTuple):
    """Resolved settings for one hidden feedforward layer."""

    width: int
    activation: Activation
    dropout: float

    @classmethod
    def checked(
        cls, width: int, activation: Activation, dropout: float, /
    ) -> FeedForwardLayer:
        """Return validated settings for one hidden layer."""
        rate = ensure_finite("a hidden layer's dropout", dropout)

        if not 0.0 <= rate < 1.0:
            msg = f"a hidden layer's dropout lies in [0, 1), not {dropout}"
            raise ValueError(msg)

        return cls(
            ensure_positive("a hidden layer's width", width),
            ensure_name("activation", activation),
            rate,
        )


class FeedForwardLayerSpec(dx.Model, extra="forbid"):
    """Declare one hidden feedforward layer."""

    width: int
    activation: str
    dropout: float = 0.0

    @dx.validates("width")
    @classmethod
    def _width_is_positive(cls, value: int) -> int:
        """Return a positive layer width."""
        return ensure_positive("a hidden layer's width", value)

    @dx.validates("activation")
    @classmethod
    def _activation_is_named(cls, value: str) -> str:
        """Return a named activation plugin."""
        return ensure_name("activation", value)

    @dx.validates("dropout")
    @classmethod
    def _dropout_is_a_rate(cls, value: float) -> float:
        """Return a finite dropout rate below one."""
        rate = ensure_finite("a hidden layer's dropout", value)

        if not 0.0 <= rate < 1.0:
            msg = f"a hidden layer's dropout lies in [0, 1), not {value}"
            raise ValueError(msg)

        return rate

    def settings(self) -> FeedForwardLayer:
        """Return the immutable settings this layer declares."""
        return FeedForwardLayer(self.width, self.activation, self.dropout)
