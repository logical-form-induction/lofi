"""Negative log-likelihood over the rows of a batch of terms."""

from typing import TYPE_CHECKING, Final

import torch

if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch import Tensor

    from lofi.nn._values import Encoded, LogDistribution

__all__ = ["live_rows", "negative_log_likelihood", "rows_of"]

_TERM_AXES: Final[tuple[str, ...]] = ("term",)
"""The axes a decoder weighs a batch of terms over."""


def negative_log_likelihood(
    weights: LogDistribution, *, terms: int, what: str
) -> tuple[Tensor, int]:
    """Return the negative log weight summed over the live rows, and their number.

    Parameters
    ----------
    weights : LogDistribution
        The log weight a decoder gave each term, masked where a row holds no term.
    terms : int
        How many terms the decoder was handed.
    what : str
        What the decoder scored, for the message.

    Returns
    -------
    tuple[Tensor, int]
        The sum over the live rows of their negative log weight, attached to the
        weights' graph, and how many rows are live.

    Raises
    ------
    ValueError
        If the weights are not one per term over the term axis alone.
    """
    if weights.axes != _TERM_AXES or weights.size(_TERM_AXES[0]) != terms:
        msg = (
            f"the decoder weighed {what} over {weights.axes} of shape "
            f"{tuple(weights.tensor.shape)}, and there are {terms} terms to weigh "
            f"over {_TERM_AXES}"
        )
        raise ValueError(msg)

    scores = weights.tensor
    live = weights.mask

    if live is None:
        return -scores.sum(), int(scores.shape[0])

    kept = torch.where(live, scores, scores.new_zeros(()))

    return -kept.sum(), int(live.sum())


def rows_of(encoded: Encoded, index: Tensor) -> Encoded:
    """Return some rows of a batch of encoded terms, in the order given.

    Parameters
    ----------
    encoded : Encoded
        One vector per term, the term axis outermost.
    index : Tensor
        Which rows to take, at integer precision.

    Returns
    -------
    Encoded
        The rows named, over the same axes, masked as they were.
    """
    mask = encoded.mask

    return type(encoded)(
        encoded.tensor[index],
        encoded.axes,
        None if mask is None else mask[index],
    )


def live_rows(masks: Iterable[Tensor | None]) -> Tensor | None:
    """Return which rows every mask keeps, or None where none excludes a row.

    Parameters
    ----------
    masks : Iterable[Tensor | None]
        One mask per value over the same rows, or None for a value keeping every row.

    Returns
    -------
    Tensor | None
        A boolean tensor over the rows, or None.
    """
    given = [mask for mask in masks if mask is not None]

    if not given:
        return None

    live = given[0]

    for mask in given[1:]:
        live = live & mask

    return live
