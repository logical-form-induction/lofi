"""Normalization of masked log weights."""

from math import inf
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["masked_log_softmax"]


def masked_log_softmax(weights: Tensor, mask: Tensor, /) -> Tensor:
    """Return log weights normalized over their innermost axis.

    Parameters
    ----------
    weights : Tensor
        Log weights whose innermost axis contains the alternatives.
    mask : Tensor
        A Boolean mask broadcastable to the weights.

    Returns
    -------
    Tensor
        Normalized log weights, with negative infinity at masked positions.
        A fully masked or all-negative-infinity row has zero probability mass.
    """
    filled = weights.masked_fill(~mask, -inf)
    peak = filled.amax(dim=-1, keepdim=True)
    shift = peak.masked_fill(peak == -inf, 0.0)
    total = (filled - shift).exp().sum(dim=-1, keepdim=True)

    return filled - (total.masked_fill(total == 0.0, 1.0).log() + shift)
