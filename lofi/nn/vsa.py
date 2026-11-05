"""Hadamard binding over real vectors.

``bind`` is a coordinatewise product, so ``bind(a, b)`` and ``bind(b, a)`` are the
same tensor. A tree code that binds one fixed role per structural position and composes
those roles along a path is therefore not injective past depth one: the argument of the
result and the result of the argument reach the same vector. ``protect`` and
``unprotect`` reorder a vector's coordinates by a fixed permutation raised to a depth.
The permutation does not commute with binding, so protecting each level's role by its
depth separates paths that differ only in the order their roles were composed. This
lets a tree code reuse one role per structural position rather than draw a distinct
atom per tree address.
"""

from functools import cache
from math import sqrt
from typing import TYPE_CHECKING, override

import torch
from torch import nn
from torch.nn import functional

from lofi._checks import CHECKS, ensure_finite, ensure_non_negative
from lofi.nn._guards import positive

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor

__all__ = [
    "AtomTable",
    "bind",
    "bundle",
    "bundle_arity",
    "bundle_norm",
    "bundle_scale",
    "protect",
    "superpose",
    "unbind",
    "unprotect",
]


def bind(left: Tensor, right: Tensor, /) -> Tensor:
    """Return the binding of two vectors.

    Parameters
    ----------
    left : Tensor
        One operand, over the coordinate axis innermost.
    right : Tensor
        The other operand, broadcastable against the first.

    Returns
    -------
    Tensor
        Their coordinatewise product, of the broadcast shape.
    """
    return left * right


def unbind(bound: Tensor, key: Tensor, /) -> Tensor:
    """Return the vector a key was bound to.

    Binding multiplies coordinatewise, so unbinding divides coordinatewise and a key
    with a zero coordinate recovers nothing at it. ``AtomTable`` holds its atoms above
    a floor for that reason.

    Parameters
    ----------
    bound : Tensor
        The binding to take apart, over the coordinate axis innermost.
    key : Tensor
        The operand to divide out, broadcastable against the binding.

    Returns
    -------
    Tensor
        The other operand of the binding, of the broadcast shape.

    Raises
    ------
    ValueError
        If the key holds a zero coordinate, when ``LOFI_CHECK`` is set.
    """
    if CHECKS and not bool(key.ne(0.0).all()):
        msg = "a binding key holds no zero coordinate, since unbinding divides by it"
        raise ValueError(msg)

    return bound / key


@cache
def _permutation(width: int, /) -> Tensor:
    """Return the coordinate order one protection applies, for a coordinate count.

    The order is drawn once per coordinate count from a fixed seed, so vectors of a
    given width are always protected the same way.

    Parameters
    ----------
    width : int
        How many coordinates the vectors carry.

    Returns
    -------
    Tensor
        A permutation of the coordinate positions, at integer precision.
    """
    return torch.randperm(width, generator=torch.Generator().manual_seed(0))


def _protection(vector: Tensor, depth: int, /) -> Tensor:
    """Return the coordinate order a protection at a depth reads a vector in.

    Parameters
    ----------
    vector : Tensor
        The vector to protect, over the coordinate axis innermost.
    depth : int
        How many times the permutation is applied.

    Returns
    -------
    Tensor
        The permutation composed with itself that many times, at integer precision.

    Raises
    ------
    ValueError
        If the vector has no coordinate axis.
    """
    if not vector.ndim:
        msg = "a protection runs over a coordinate axis, not over a scalar"
        raise ValueError(msg)

    width = int(vector.shape[-1])
    step = _permutation(width)
    order = torch.arange(width)

    for _ in range(depth):
        order = step[order]

    return order


def protect(vector: Tensor, depth: int, /) -> Tensor:
    """Return a vector with its coordinates permuted by a depth.

    Binding commutes, so a role reused at two depths of one path binds the same
    whichever order the path visits them. Permuting a role by the depth it sits at
    leaves the roles of a path and of its reverse distinct, since the permutation does
    not commute with binding.

    Parameters
    ----------
    vector : Tensor
        The vector to protect, over the coordinate axis innermost.
    depth : int
        How many times to apply the permutation. A depth of zero returns the vector
        unchanged, and a deeper protection composes the permutation with itself.

    Returns
    -------
    Tensor
        The vector with its coordinates reordered, of its own shape.

    Raises
    ------
    TypeError
        If the depth is not an integer.
    ValueError
        If the depth is negative, or the vector has no coordinate axis.
    """
    order = _protection(vector, ensure_non_negative("a protection's depth", depth))

    return vector.index_select(-1, order.to(vector.device))


def unprotect(vector: Tensor, depth: int, /) -> Tensor:
    """Return a protected vector with its coordinates put back.

    This inverts ``protect`` exactly at the same depth, since a permutation loses
    nothing.

    Parameters
    ----------
    vector : Tensor
        The vector to unprotect, over the coordinate axis innermost.
    depth : int
        The depth it was protected at.

    Returns
    -------
    Tensor
        The vector with its coordinates in their original order, of its own shape.

    Raises
    ------
    TypeError
        If the depth is not an integer.
    ValueError
        If the depth is negative, or the vector has no coordinate axis.
    """
    order = _protection(vector, ensure_non_negative("a protection's depth", depth))
    restored = torch.empty_like(order)
    restored[order] = torch.arange(order.numel())

    return vector.index_select(-1, restored.to(vector.device))


def superpose(components: Sequence[Tensor], /) -> Tensor:
    """Return the superposition of several vectors.

    Parameters
    ----------
    components : Sequence[Tensor]
        The vectors to superpose, all of one shape.

    Returns
    -------
    Tensor
        Their sum, of the shape they share.

    Raises
    ------
    ValueError
        If nothing is given, since the sum of no vector has no shape.
    """
    if not components:
        msg = "a superposition holds at least one vector"
        raise ValueError(msg)

    return torch.stack(tuple(components)).sum(dim=0)


def bundle_scale(arity: int, /) -> float:
    """Return the divisor holding a superposition at the magnitude of one component.

    Superposing independent vectors grows the result as the square root of how many
    were superposed, so dividing by that root leaves the magnitude where it started.

    Parameters
    ----------
    arity : int
        How many vectors are superposed.

    Raises
    ------
    TypeError
        If the arity is not an integer.
    ValueError
        If the arity is not positive.
    """
    return sqrt(positive("a bundle's arity", arity))


def bundle(components: Sequence[Tensor], /) -> Tensor:
    """Return the superposition of several vectors at the magnitude of one of them.

    Parameters
    ----------
    components : Sequence[Tensor]
        The vectors to bundle, all of one shape.

    Returns
    -------
    Tensor
        Their sum divided by the scale their number calls for, of the shape they share.

    Raises
    ------
    ValueError
        If nothing is given.
    """
    return superpose(components) / bundle_scale(len(components))


def bundle_norm(*, magnitude: float, arity: int, dimension: int) -> float:
    """Return the Euclidean norm a superposition of bound pairs sits near.

    A coordinate of a pair bound from two atoms of one magnitude sits at that magnitude
    squared, and superposing independent pairs adds their variances, so the norm over
    the coordinates rises as the root of the arity times the dimension. The
    superposition meant here is the plain sum, with no bundle scale applied.

    Parameters
    ----------
    magnitude : float
        The magnitude each coordinate of an atom sits at.
    arity : int
        How many bound pairs were superposed.
    dimension : int
        How many coordinates an atom has.

    Raises
    ------
    TypeError
        If the magnitude is not a number, or the arity or dimension is not an integer.
    ValueError
        If the magnitude is not finite, or the arity or dimension is not positive.
    """
    scale = ensure_finite("an atom's magnitude", magnitude)
    pairs = positive("a bundle's arity", arity)
    width = positive("an atom's dimension", dimension)

    return scale * scale * sqrt(pairs * width)


def bundle_arity(vector: Tensor, /, *, magnitude: float) -> Tensor:
    """Return how many bound pairs a superposition holds, read off its norm.

    This inverts ``bundle_norm`` at a fixed magnitude and dimension. The result is a
    real number rather than a count, since the norm of any one superposition scatters
    about the norm its arity predicts.

    Parameters
    ----------
    vector : Tensor
        The superposition to read, over the coordinate axis innermost.
    magnitude : float
        The magnitude each coordinate of an atom sits at.

    Returns
    -------
    Tensor
        The estimate at each position of the axes outside the coordinate axis.

    Raises
    ------
    TypeError
        If the magnitude is not a number.
    ValueError
        If the magnitude is not finite, or the vector has no coordinate axis.
    """
    if vector.ndim == 0:
        msg = "a superposition runs over a coordinate axis, so it is not a scalar"
        raise ValueError(msg)

    unit = bundle_norm(magnitude=magnitude, arity=1, dimension=int(vector.shape[-1]))

    return torch.linalg.vector_norm(vector, dim=-1).div(unit).square()


class AtomTable(nn.Module):
    """A table of atom vectors held away from zero.

    Each coordinate has a fixed stored sign and a learned magnitude clamped to the
    configured floor. The floor keeps coordinatewise unbinding defined.

    A coordinate is drawn around the configured magnitude with a spread of that
    magnitude over the root of the dimension. The spread scales with the magnitude so
    that a table built small is drawn as far above the floor, in its own units, as one
    built at the default.

    Parameters
    ----------
    count : int
        How many atoms the table holds.
    dimension : int
        How many coordinates each atom has.
    magnitude : float
        Where each coordinate's magnitude is centered, and the scale of its spread.
    floor : float
        The smallest magnitude any coordinate is read at.

    Raises
    ------
    TypeError
        If the count or the dimension is not an integer, or the magnitude or the floor
        is not a number.
    ValueError
        If the count or the dimension is not positive, if the magnitude or the floor is
        not finite, or if the floor is not above zero and at most the magnitude.
    """

    _signs: Tensor

    def __init__(
        self,
        *,
        count: int,
        dimension: int,
        magnitude: float = 1.0,
        floor: float = 1e-3,
    ) -> None:
        super().__init__()
        atoms = positive("an atom table's size", count)
        width = positive("an atom's dimension", dimension)
        center = ensure_finite("an atom's magnitude", magnitude)
        least = ensure_finite("an atom's magnitude floor", floor)

        if not 0.0 < least <= center:
            msg = (
                f"an atom's magnitude floor lies above zero and no higher than "
                f"the magnitude {center}, not at {least}"
            )
            raise ValueError(msg)

        self.register_buffer(
            "_signs",
            torch.randint(0, 2, (atoms, width)).mul(2).sub(1).float(),
        )
        self._magnitudes = nn.Parameter(
            torch.randn(atoms, width).mul(center / sqrt(width)).add(center)
        )
        self._magnitude = center
        self._floor = least

    @property
    def count(self) -> int:
        """Return how many atoms the table holds."""
        return int(self._signs.shape[0])

    @property
    def dimension(self) -> int:
        """Return how many coordinates each atom has."""
        return int(self._signs.shape[1])

    @property
    def magnitude(self) -> float:
        """Return where each coordinate's magnitude is centered."""
        return self._magnitude

    @property
    def floor(self) -> float:
        """Return the smallest magnitude any coordinate is read at."""
        return self._floor

    @property
    def atoms(self) -> Tensor:
        """Return every atom, over the atom axis and the coordinate axis.

        Returns
        -------
        Tensor
            The table, at the parameters' precision, with no coordinate nearer zero
            than the floor.
        """
        return self._signs * self._magnitudes.clamp_min(self._floor)

    @override
    def forward(self, indices: Tensor, /) -> Tensor:
        """Return the atom named by each index.

        Parameters
        ----------
        indices : Tensor
            The atoms to read, at integer precision.

        Returns
        -------
        Tensor
            One atom per index, over the index's own axes and then the coordinate axis.
        """
        return functional.embedding(indices, self.atoms)

    @override
    def extra_repr(self) -> str:
        """Return the size, the dimension, the magnitude and the floor."""
        return (
            f"count={self.count}, dimension={self.dimension}, "
            f"magnitude={self._magnitude}, floor={self._floor}"
        )
