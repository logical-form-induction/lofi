"""A learned map between two type-embedding spaces.

The map sends the vector a type has in one embedding space, a syntactic
inventory's say, to a vector in another, a semantic inventory's, and comes in
two forms. The linear form is one linear layer. The multilayer form is a
feed-forward network with a declared depth.
"""

from enum import StrEnum
from typing import override

from torch import nn

from lofi._checks import ensure_finite
from lofi._frozen import FrozenValue
from lofi.nn._guards import checked_state, positive
from lofi.nn._values import Encoded

__all__ = ["MapForm", "MultilayerOptions", "TypeSpaceMap"]


class MapForm(StrEnum):
    """Names the form a map between two type-embedding spaces takes."""

    LINEAR = "linear"

    MULTILAYER = "multilayer"


def _checked_dropout(value: float) -> float:
    """Return a dropout rate.

    Parameters
    ----------
    value : float
        The fraction of hidden units to drop.

    Raises
    ------
    TypeError
        If it is not a number.
    ValueError
        If it is below zero or at least one.
    """
    rate = ensure_finite("a dropout rate", value)

    if not 0.0 <= rate < 1.0:
        msg = f"a dropout rate lies in [0, 1), not at {rate}"
        raise ValueError(msg)

    return rate


class MultilayerOptions(FrozenValue):
    """Configures the multilayer form of a map.

    Parameters
    ----------
    hidden : int
        The width of each hidden layer.
    depth : int
        How many linear layers there are. Each but the last is followed by a
        hyperbolic tangent and dropout.
    dropout : float
        The fraction of each hidden layer's units dropped in training.

    Raises
    ------
    TypeError
        If a width or a depth is not an integer, or the rate is not a number.
    ValueError
        If a width or a depth is not positive, or the rate is outside $[0, 1)$.
    """

    _noun = "space-map layers"

    __slots__ = ("_depth", "_dropout", "_hidden")

    def __init__(self, *, hidden: int, depth: int, dropout: float) -> None:
        object.__setattr__(self, "_hidden", positive("the hidden width", hidden))
        object.__setattr__(self, "_depth", positive("the depth", depth))
        object.__setattr__(self, "_dropout", _checked_dropout(dropout))

    @property
    def hidden(self) -> int:
        """Return the width of each hidden layer."""
        return self._hidden

    @property
    def depth(self) -> int:
        """Return how many linear layers there are."""
        return self._depth

    @property
    def dropout(self) -> float:
        """Return the fraction of each hidden layer's units dropped in training."""
        return self._dropout

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value configures the same network."""
        if type(other) is not MultilayerOptions:
            return NotImplemented

        return (self._hidden, self._depth, self._dropout) == (
            other._hidden,
            other._depth,
            other._dropout,
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the three settings."""
        return hash((MultilayerOptions, self._hidden, self._depth, self._dropout))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds these options."""
        return (
            f"MultilayerOptions(hidden={self._hidden}, depth={self._depth}, "
            f"dropout={self._dropout})"
        )


class TypeSpaceMap(nn.Module):
    """Maps vectors of one type-embedding space into another.

    Parameters
    ----------
    source_width : int
        How wide the vectors of the space the map reads are.
    target_width : int
        How wide the vectors of the space it returns are.
    form : MapForm
        Which form the map takes.
    layers : MultilayerOptions | None
        The layers of a multilayer map, or None for the default ones. The linear
        form is one linear layer and takes none.

    Raises
    ------
    TypeError
        If the form is not one, or a width is not an integer.
    ValueError
        If a width is not positive, or if layers are named for the linear form.
    """

    def __init__(
        self,
        *,
        source_width: int,
        target_width: int,
        form: MapForm,
        layers: MultilayerOptions | None,
    ) -> None:
        super().__init__()

        if not isinstance(form, MapForm):
            msg = f"a map takes a MapForm, not {form!r}"
            raise TypeError(msg)

        source = positive("the source width", source_width)
        target = positive("the target width", target_width)

        if form is MapForm.LINEAR:
            if layers is not None:
                msg = "the linear form is one linear layer and takes no layers"
                raise ValueError(msg)

            self._map: nn.Module = nn.Linear(source, target)
            self._layers = None
        else:
            if layers is None:
                msg = "the multilayer form requires declared layers"
                raise ValueError(msg)

            self._layers = layers
            self._map = _multilayer(source, target, self._layers)

        self._source = source
        self._target = target
        self._form = form

    @property
    def form(self) -> MapForm:
        """Return which form this map takes."""
        return self._form

    @property
    def layers(self) -> MultilayerOptions | None:
        """Return the layers of a multilayer map, or None for the linear form."""
        return self._layers

    @property
    def source_width(self) -> int:
        """Return the size of the feature axis of the vectors this map reads."""
        return self._source

    @property
    def target_width(self) -> int:
        """Return the size of the feature axis of the vectors this map returns."""
        return self._target

    def apply_to(self, vectors: Encoded, /) -> Encoded:
        """Return the vectors mapped into the target space.

        Parameters
        ----------
        vectors : Encoded
            Source-space vectors over the feature axis innermost, with any leading
            axes and any mask.

        Returns
        -------
        Encoded
            Target-space vectors over the same axes, the feature axis now of the
            target width, with the same mask.

        Raises
        ------
        ValueError
            If the vectors do not run over the feature axis innermost, or run over
            it at another width than this map reads.
        """
        checked = checked_state(vectors, self._source, "map")

        return Encoded(self._map(checked.tensor), checked.axes, checked.mask)

    @override
    def extra_repr(self) -> str:
        """Return the form and the two widths."""
        return f"{self._form.name.lower()}, {self._source} -> {self._target}"


def _multilayer(source: int, target: int, options: MultilayerOptions) -> nn.Sequential:
    """Return a feed-forward network from the source width to the target width.

    Parameters
    ----------
    source : int
        The width of the vectors the network reads.
    target : int
        The width of the vectors it returns.
    options : MultilayerOptions
        The hidden width, the depth and the dropout rate.
    """
    layers: list[nn.Module] = []
    width = source

    for _ in range(options.depth - 1):
        layers.extend(
            (nn.Linear(width, options.hidden), nn.Tanh(), nn.Dropout(options.dropout))
        )
        width = options.hidden

    layers.append(nn.Linear(width, target))

    return nn.Sequential(*layers)
