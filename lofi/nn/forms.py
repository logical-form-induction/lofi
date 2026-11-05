"""Weighted term sets and scored logical forms."""

from enum import Enum
from typing import TYPE_CHECKING, Never, Self, final, override

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor

    from lofi.terms import Term

__all__ = ["Estimate", "WeightedTerms"]


class Estimate(Enum):
    """Classify what the weights of a decoded set estimate."""

    EXACT = "exact"
    TRUNCATED = "truncated"
    SAMPLED = "sampled"


@final
class WeightedTerms:
    """A finite set of decoded terms and the probability mass outside it.

    The log weights remain attached to the decoder graph. The residual is a scalar
    log probability on no listed term, and ``estimate`` states how the set was
    obtained rather than asking consumers to infer that fact from its shape.
    """

    __slots__ = ("_estimate", "_log_residual", "_log_weights", "_terms")

    def __init__(
        self,
        terms: Sequence[Term],
        log_weights: Tensor,
        log_residual: Tensor,
        estimate: Estimate,
    ) -> None:
        held = tuple(terms)
        weights = _checked_weights(held, log_weights)

        if log_residual.ndim != 0:
            msg = (
                "the residual mass is a scalar, not of shape "
                f"{tuple(log_residual.shape)}"
            )
            raise ValueError(msg)

        if bool(log_residual.isnan()) or bool(log_residual > 0.0):
            msg = "the residual is the log of a probability at most one"
            raise ValueError(msg)

        if not isinstance(estimate, Estimate):
            msg = f"an estimate is an Estimate, not {estimate!r}"
            raise TypeError(msg)

        object.__setattr__(self, "_terms", held)
        object.__setattr__(self, "_log_weights", weights)
        object.__setattr__(self, "_log_residual", log_residual)
        object.__setattr__(self, "_estimate", estimate)

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        """Refuse field reassignment."""
        del name, value
        msg = "weighted terms are immutable"
        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Refuse field deletion."""
        del name
        msg = "weighted terms are immutable"
        raise AttributeError(msg)

    def __copy__(self) -> Self:
        """Return this immutable value."""
        return self

    def __deepcopy__(self, memo: dict[int, Self]) -> Self:
        """Return this immutable value."""
        del memo
        return self

    @property
    def terms(self) -> tuple[Term, ...]:
        """Return the terms in strategy order."""
        return self._terms

    @property
    def log_weights(self) -> Tensor:
        """Return one graph-connected log weight per term."""
        return self._log_weights

    @property
    def log_residual(self) -> Tensor:
        """Return the log probability mass on no listed term."""
        return self._log_residual

    @property
    def estimate(self) -> Estimate:
        """Return how the weighted set was obtained."""
        return self._estimate

    def __len__(self) -> int:
        """Return the number of listed terms."""
        return len(self._terms)


def _checked_weights(terms: Sequence[Term], log_weights: Tensor) -> Tensor:
    """Return one log weight per term.

    Parameters
    ----------
    terms : Sequence[Term]
        The terms the weights are aligned with.
    log_weights : Tensor
        The log weight of each term, in the order of the terms.

    Returns
    -------
    Tensor
        The weights, unchanged.

    Raises
    ------
    TypeError
        If the weights are not at floating precision.
    ValueError
        If the weights are not of rank 1, or there is not one per term.
    """
    if not log_weights.is_floating_point():
        msg = f"a log weight is floating point, not {log_weights.dtype}"
        raise TypeError(msg)

    if log_weights.ndim != 1 or log_weights.shape[0] != len(terms):
        msg = (
            f"a beam holds one log weight per form, so over {len(terms)} forms the "
            f"weights are of shape ({len(terms)},), not {tuple(log_weights.shape)}"
        )
        raise ValueError(msg)

    return log_weights
