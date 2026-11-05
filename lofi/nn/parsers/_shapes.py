"""Parser tensor-shape validation."""

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from torch import Tensor

    from lofi.deduce import SpanIndex

__all__ = ["checked_leaf_scores", "checked_scalar_chart", "checked_split_scores"]

#: The rank of a tensor running over item and one further axis.
PAIRED_RANK: Final[int] = 2

#: The rank of a split-indexed tensor: item, then span, then split.
SPLIT_RANK: Final[int] = 3


def checked_leaf_scores(leaf_scores: Tensor, index: SpanIndex, /) -> int:
    """Return the batch size of a leaf score tensor.

    Parameters
    ----------
    leaf_scores : Tensor
        The log weight of each token as a one-token span, of rank 2.
    index : SpanIndex
        The span addressing the scores are laid out under.

    Returns
    -------
    int
        How many items the tensor covers.

    Raises
    ------
    TypeError
        If the tensor is not of a floating point dtype.
    ValueError
        If it has the wrong rank, or an extent the index does not call for.
    """
    if not leaf_scores.is_floating_point():
        msg = f"leaf scores are log weights, not {leaf_scores.dtype}"
        raise TypeError(msg)

    if leaf_scores.ndim != PAIRED_RANK:
        msg = (
            f"leaf scores run over item and token, so they have rank "
            f"{PAIRED_RANK}, not {leaf_scores.ndim}"
        )
        raise ValueError(msg)

    batch = int(leaf_scores.shape[0])
    shape = (batch, index.n)

    if tuple(leaf_scores.shape) != shape:
        msg = (
            f"an input of length {index.n} needs leaf scores of shape "
            f"{shape}, not {tuple(leaf_scores.shape)}"
        )
        raise ValueError(msg)

    return batch


def checked_scalar_chart(name: str, chart: Tensor, index: SpanIndex, /) -> int:
    """Return the batch size of a scalar chart.

    Parameters
    ----------
    name : str
        What to call the chart in a message.
    chart : Tensor
        The log weight of every span, of rank 2.
    index : SpanIndex
        The span addressing the chart is laid out under.

    Returns
    -------
    int
        How many items the chart covers.

    Raises
    ------
    TypeError
        If the chart is not of a floating point dtype.
    ValueError
        If it has the wrong rank, or an extent the index does not call for.
    """
    if not chart.is_floating_point():
        msg = f"{name} holds log weights, not {chart.dtype}"
        raise TypeError(msg)

    if chart.ndim != PAIRED_RANK:
        msg = (
            f"{name} runs over item and span, so it has rank "
            f"{PAIRED_RANK}, not {chart.ndim}"
        )
        raise ValueError(msg)

    batch = int(chart.shape[0])
    shape = (batch, index.num_spans)

    if tuple(chart.shape) != shape:
        msg = (
            f"an input of length {index.n} needs {name} of shape "
            f"{shape}, not {tuple(chart.shape)}"
        )
        raise ValueError(msg)

    return batch


def checked_split_scores(split_scores: Tensor, index: SpanIndex, batch: int, /) -> None:
    """Raise if split scores have an invalid dtype or shape.

    Parameters
    ----------
    split_scores : Tensor
        The log weight of each split of each span, of rank 3.
    index : SpanIndex
        The span addressing the scores are laid out under.
    batch : int
        How many items the tensor the scores accompany covers.

    Raises
    ------
    TypeError
        If the tensor is not of a floating point dtype.
    ValueError
        If it has the wrong rank, or an extent the index and the batch do not call for.
    """
    if not split_scores.is_floating_point():
        msg = f"split scores are log weights, not {split_scores.dtype}"
        raise TypeError(msg)

    if split_scores.ndim != SPLIT_RANK:
        msg = (
            f"split scores run over item, span and split, so they have rank "
            f"{SPLIT_RANK}, not {split_scores.ndim}"
        )
        raise ValueError(msg)

    shape = (batch, index.num_spans, index.max_splits)

    if tuple(split_scores.shape) != shape:
        msg = (
            f"an input of length {index.n} needs split scores of shape "
            f"{shape}, not {tuple(split_scores.shape)}"
        )
        raise ValueError(msg)
