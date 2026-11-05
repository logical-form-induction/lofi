"""The mapping a readout puts its scores through before it publishes them."""

from typing import TYPE_CHECKING, Final, override

import torch

from lofi._checks import ensure_finite
from lofi.nn.loss._divergence import (
    CrossEntropy,
    FenchelYoung,
    KullbackLeibler,
    resolved,
)
from lofi.nn.loss._errors import SparseDivergenceError

if TYPE_CHECKING:
    from torch import Tensor

    from lofi.nn.loss._divergence import Divergence

__all__ = ["SOFTMAX", "Entmax", "ensure_finite_pairing"]


def _thresholded(scaled: Tensor, threshold: Tensor, exponent: float) -> Tensor:
    """Return the unnormalized entmax weights above one threshold.

    Parameters
    ----------
    scaled : Tensor
        The scores times alpha minus one, the symbol axis innermost.
    threshold : Tensor
        The threshold subtracted from them, broadcasting over that axis.
    exponent : float
        The reciprocal of alpha minus one.

    Returns
    -------
    Tensor
        The positive part raised to the exponent, exactly zero where the score does
        not clear the threshold, with a finite gradient everywhere.
    """
    above = scaled > threshold
    safe = torch.where(above, scaled - threshold, scaled.new_ones(()))

    return torch.where(above, safe.pow(exponent), scaled.new_zeros(()))


def _solved(scores: Tensor, alpha: float, bisections: int) -> Tensor:
    """Return the alpha-entmax of scores over their innermost axis.

    The threshold that normalizes the weights is found by bisection, which is the
    only route open at an arbitrary alpha. The bracket holds because the largest
    score alone clears the lower end and no score clears the upper end by more than
    its share.

    Parameters
    ----------
    scores : Tensor
        Real-valued scores, the symbol axis innermost, at least one of them finite,
        negative infinity where a symbol is unavailable.
    alpha : float
        Which entropy the mapping is conjugate to, above one.
    bisections : int
        How many halvings the threshold search takes, which fixes its precision.

    Returns
    -------
    Tensor
        Probabilities summing to one along the innermost axis, exactly zero at every
        symbol below the threshold.
    """
    exponent = 1.0 / (alpha - 1.0)
    scaled = scores * (alpha - 1.0)
    highest = scaled.max(dim=-1, keepdim=True).values
    low = highest - 1.0
    high = highest - (1.0 / scores.shape[-1]) ** (alpha - 1.0)

    for _ in range(bisections):
        middle = (low + high) / 2.0
        total = _thresholded(scaled, middle, exponent).sum(dim=-1, keepdim=True)
        crowded = total > 1.0
        low = torch.where(crowded, middle, low)
        high = torch.where(crowded, high, middle)

    weights = _thresholded(scaled, high, exponent)

    return weights / weights.sum(dim=-1, keepdim=True)


def _entmax(scores: Tensor, alpha: float, bisections: int) -> Tensor:
    """Return the alpha-entmax of scores, carrying the mapping's own gradient.

    The threshold is searched for rather than solved for, and the search itself has
    no meaningful derivative. The values therefore come from the search with the
    graph detached, and the mapping's Jacobian is applied to a displacement that is
    exactly zero: diagonal in the probabilities raised to two minus alpha, less the
    rank-one term the threshold contributes, and zero off the support, where the
    mapping is locally constant.

    Parameters
    ----------
    scores : Tensor
        Real-valued scores, the symbol axis innermost, negative infinity where a
        symbol is unavailable.
    alpha : float
        Which entropy the mapping is conjugate to, above one.
    bisections : int
        How many halvings the threshold search takes.

    Returns
    -------
    Tensor
        Probabilities summing to one along the innermost axis, exactly zero at every
        symbol below the threshold.
    """
    with torch.no_grad():
        probabilities = _solved(scores, alpha, bisections)

    support = probabilities > 0.0
    safe = torch.where(support, probabilities, probabilities.new_ones(()))
    slope = torch.where(support, safe.pow(2.0 - alpha), probabilities.new_zeros(()))
    divisor = slope.sum(dim=-1, keepdim=True)
    live = torch.where(support, scores, scores.new_zeros(()))
    moved = live - live.detach()
    shared = (slope * moved).sum(dim=-1, keepdim=True) / divisor

    return probabilities + slope * (moved - shared)


class Entmax:
    """Maps a readout's scores to a distribution over symbols.

    At alpha one the mapping is the softmax, and every symbol keeps some mass. Above
    one it is sparse: most symbols get exactly zero mass, which makes the sum over
    types a finite exact one and makes any comparison that divides by the mapped
    distribution infinite wherever the other side has mass.

    Parameters
    ----------
    alpha : float
        Which entropy the mapping is conjugate to.
    bisections : int
        How many halvings the threshold search takes above alpha one, which fixes
        its precision; zero at alpha one, where no threshold is searched for.

    Raises
    ------
    TypeError
        If alpha is not a number.
    ValueError
        If alpha is not finite or is below one, which leaves the family the mapping
        is defined over, or if the bisection count is negative, or zero above alpha
        one.
    """

    __slots__ = ("_alpha", "_bisections")

    def __init__(self, alpha: float, /, *, bisections: int) -> None:
        checked = ensure_finite("alpha", alpha)

        if checked < 1.0:
            msg = f"alpha indexes the entropy family from one upward, not {alpha}"
            raise ValueError(msg)

        if bisections < 0 or (checked > 1.0 and bisections == 0):
            msg = (
                f"a threshold search above alpha one takes a positive bisection "
                f"count, not {bisections}"
            )
            raise ValueError(msg)

        self._alpha = checked
        self._bisections = bisections

    @property
    def alpha(self) -> float:
        """Return which entropy the mapping is conjugate to."""
        return self._alpha

    @property
    def sparse(self) -> bool:
        """Return whether the mapping leaves symbols at exactly zero mass."""
        return self._alpha > 1.0

    def project(self, scores: Tensor, /) -> Tensor:
        """Return the log of the distribution this mapping reads from scores.

        The mapping is invariant to a shift shared by a row, so scores that have
        already been through a softmax map to the same distribution as the scores
        behind them. A row with no available symbol has no distribution to read, and
        every symbol of it comes back at negative infinity. A score that is not a
        number or is positive infinity is carried into the result, so a diverged
        score reads as a numerical failure rather than as an absent row.

        Parameters
        ----------
        scores : Tensor
            Real-valued scores, the symbol axis innermost, negative infinity where a
            symbol is unavailable.

        Returns
        -------
        Tensor
            Log probabilities summing to one along the innermost axis, negative
            infinity at every symbol the mapping leaves at zero.
        """
        live = scores.isneginf().logical_not().any(dim=-1, keepdim=True)
        guarded = torch.where(live, scores, scores.new_zeros(()))
        absent = scores.new_full((), -torch.inf)

        if not self.sparse:
            return torch.where(live, guarded.log_softmax(dim=-1), absent)

        probabilities = _entmax(guarded, self._alpha, self._bisections)
        positive = live & (probabilities > 0.0).logical_or(probabilities.isnan())
        safe = torch.where(positive, probabilities, probabilities.new_ones(()))

        return torch.where(positive, safe.log(), absent)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether the other value is this mapping at the same alpha.

        Parameters
        ----------
        other : object
            The value to compare against.
        """
        if type(other) is not Entmax:
            return NotImplemented

        return (self._alpha, self._bisections) == (other._alpha, other._bisections)

    @override
    def __hash__(self) -> int:
        """Return the hash of the mapping and its alpha."""
        return hash((Entmax, self._alpha, self._bisections))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this mapping."""
        return f"Entmax({self._alpha!r}, bisections={self._bisections!r})"


SOFTMAX: Final[Entmax] = Entmax(1.0, bisections=0)
"""The dense mapping, which leaves no symbol at exactly zero mass."""


def ensure_finite_pairing(mapping: Entmax, divergence: Divergence, /) -> Divergence:
    """Return the divergence, refusing one that is infinite under a sparse mapping.

    The refusal is on the comparison a specification resolves to rather than on how it
    is spelled, so a Fenchel-Young loss at alpha one is refused as the KL divergence it
    computes.

    Parameters
    ----------
    mapping : Entmax
        The mapping the readout publishes through.
    divergence : Divergence
        The comparison made against what it publishes.

    Returns
    -------
    Divergence
        The divergence, unchanged.

    Raises
    ------
    SparseDivergenceError
        If the mapping is sparse and the divergence divides by a mass the mapping
        sets to exactly zero.
    """
    match resolved(divergence):
        case KullbackLeibler() | CrossEntropy() if mapping.sparse:
            raise SparseDivergenceError(
                mapping=repr(mapping),
                divergence=repr(divergence),
                replacement=repr(FenchelYoung(mapping.alpha)),
            )

        case _:
            return divergence
