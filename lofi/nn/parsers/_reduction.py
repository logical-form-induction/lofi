"""Differentiable reductions over tensor-valued log weights."""

from math import inf
from typing import TYPE_CHECKING

from lofi._frozen import FrozenValue

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["LogSumExpReduction", "MaximumReduction"]


def _masked(weights: Tensor, mask: Tensor, /) -> Tensor:
    """Return log weights with every masked position at negative infinity."""
    return weights.masked_fill(~mask, -inf)


class LogSumExpReduction(FrozenValue):
    """Sum alternative probabilities while retaining log-space tensors."""

    __slots__ = ()

    _noun = "log-sum-exp reduction"

    @property
    def probabilistic(self) -> bool:
        """Return that log-sum-exp defines posterior marginals."""
        return True

    def reduce(self, weights: Tensor, mask: Tensor, /, *, dim: int) -> Tensor:
        """Return the log of the summed probability of the real alternatives."""
        filled = _masked(weights, mask)
        peak = filled.amax(dim=dim, keepdim=True)
        shift = peak.masked_fill(peak == -inf, 0.0)
        total = (filled - shift).exp().sum(dim=dim)
        empty = total == 0.0
        combined = total.masked_fill(empty, 1.0).log() + shift.squeeze(dim)

        return combined.masked_fill(empty, -inf)

    def mixing_weights(self, weights: Tensor, mask: Tensor, /, *, dim: int) -> Tensor:
        """Return normalized log probabilities over the real alternatives."""
        filled = _masked(weights, mask)
        normalizer = self.reduce(weights, mask, dim=dim).unsqueeze(dim)
        empty = normalizer == -inf

        return (filled - normalizer.masked_fill(empty, 0.0)).masked_fill(empty, -inf)

    def validate_expected_counts(self) -> None:
        """Admit expected counts from the gradient of a log partition."""


class MaximumReduction(FrozenValue):
    """Keep the largest real alternative log weight."""

    __slots__ = ()

    _noun = "maximum reduction"

    @property
    def probabilistic(self) -> bool:
        """Return that maximum does not define posterior marginals."""
        return False

    def reduce(self, weights: Tensor, mask: Tensor, /, *, dim: int) -> Tensor:
        """Return the largest log weight among the real alternatives."""
        return _masked(weights, mask).amax(dim=dim)

    def mixing_weights(self, weights: Tensor, mask: Tensor, /, *, dim: int) -> Tensor:
        """Return a hard selection of the first largest real alternative."""
        filled = _masked(weights, mask)
        best = filled.argmax(dim=dim, keepdim=True)
        selected = filled.new_full(filled.shape, -inf).scatter(dim, best, 0.0)
        empty = filled.amax(dim=dim, keepdim=True) == -inf

        return selected.masked_fill(empty, -inf)

    def validate_expected_counts(self) -> None:
        """Refuse expected counts from a maximum instead of a log partition.

        Raises
        ------
        ValueError
            Always.
        """
        msg = "an expected count requires log-sum-exp reduction"
        raise ValueError(msg)
