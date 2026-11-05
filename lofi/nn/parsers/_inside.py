"""Inside chart computation."""

from typing import TYPE_CHECKING

import torch

from lofi._checks import CHECKS
from lofi.nn.parsers._chart import TensorChart
from lofi.nn.parsers._shapes import checked_leaf_scores, checked_split_scores

if TYPE_CHECKING:
    from torch import Tensor

    from lofi.deduce import SpanIndex
    from lofi.nn.parsers._ports import LogWeightReduction

__all__ = ["ChildGathers", "InsideKernel"]


def _checked_scores(leaf_scores: Tensor, split_scores: Tensor, index: SpanIndex) -> int:
    """Return the batch size of a pair of score tensors.

    Parameters
    ----------
    leaf_scores : Tensor
        The log weight of each token as a one-token span, of rank 2.
    split_scores : Tensor
        The log weight of each split of each span, of rank 3.
    index : SpanIndex
        The span addressing the scores are laid out under.

    Returns
    -------
    int
        How many items the two tensors agree on.

    Raises
    ------
    TypeError
        If either tensor is not of a floating point dtype.
    ValueError
        If the index spans an input of length zero, if either tensor has the wrong rank,
        or if their shapes do not match the index and each other.
    """
    if index.n < 1:
        msg = f"an inside sweep needs an input of length at least one, not {index.n}"
        raise ValueError(msg)

    batch = checked_leaf_scores(leaf_scores, index)
    checked_split_scores(split_scores, index, batch)

    return batch


def _root_ids(chart: Tensor, index: SpanIndex, lengths: Tensor | None) -> Tensor:
    """Return the span id holding each item's goal, as an index into the chart.

    Parameters
    ----------
    chart : Tensor
        The inside chart, used to place the result and to size it.
    index : SpanIndex
        The span addressing the chart is laid out under.
    lengths : Tensor | None
        How many tokens each item has, or None where every item fills the index.

    Returns
    -------
    Tensor
        A rank-1 int64 tensor of shape (item,) on the chart's device, holding the span
        id of ``(0, length)`` for each item.

    Raises
    ------
    TypeError
        If the lengths are not of an integer dtype.
    ValueError
        If the lengths are not one per item, or, under the debug checks, if a length is
        outside the range the index covers.
    """
    batch = chart.shape[0]

    if lengths is None:
        return chart.new_full((batch,), index.span_id(0, index.n), dtype=torch.long)

    if lengths.dtype.is_floating_point or lengths.dtype.is_complex:
        msg = f"a length is a whole number of tokens, not {lengths.dtype}"
        raise TypeError(msg)

    if lengths.dtype is torch.bool:
        msg = "a length is a whole number of tokens, not a boolean"
        raise TypeError(msg)

    if lengths.ndim != 1 or lengths.shape[0] != batch:
        msg = (
            f"an input of {batch} items needs lengths of shape {(batch,)}, "
            f"not {tuple(lengths.shape)}"
        )
        raise ValueError(msg)

    if CHECKS and not bool(((lengths >= 1) & (lengths <= index.n)).all()):
        msg = f"a length is outside range(1, {index.n + 1}): {lengths.tolist()}"
        raise ValueError(msg)

    roots = chart.new_tensor(
        [index.span_id(0, length) for length in range(1, index.n + 1)],
        dtype=torch.long,
    )

    return roots.index_select(0, lengths.to(torch.long) - 1)


class ChildGathers:
    """Stores an index's child ids and split mask as tensors beside a chart.

    Parameters
    ----------
    index : SpanIndex
        The span addressing the ids come from.
    like : Tensor
        A tensor to place the ids beside.
    """

    __slots__ = ("_index", "_left", "_real", "_right")

    def __init__(self, index: SpanIndex, like: Tensor, /) -> None:
        self._index = index
        self._left = like.new_tensor(index.left_child_ids, dtype=torch.long)
        self._right = like.new_tensor(index.right_child_ids, dtype=torch.long)
        self._real = like.new_tensor(index.split_mask, dtype=torch.bool)

    @property
    def index(self) -> SpanIndex:
        """Return the span addressing the ids come from."""
        return self._index

    def bounds(self, width: int) -> tuple[int, int]:
        """Return the span id range of one width.

        Parameters
        ----------
        width : int
            How many tokens the spans cover.

        Returns
        -------
        tuple[int, int]
            The first id of that width and the id after its last.
        """
        start = self._index.span_id(0, width)

        return (start, start + self._index.n - width + 1)

    def left(self, width: int) -> Tensor:
        """Return the left child id of every split of every span of one width.

        Parameters
        ----------
        width : int
            How many tokens the spans cover.

        Returns
        -------
        Tensor
            A rank-2 int64 tensor of shape (span, split) with a column per split the
            widest span has.
        """
        start, stop = self.bounds(width)

        return self._left[start:stop]

    def right(self, width: int) -> Tensor:
        """Return the right child id of every split of every span of one width.

        Parameters
        ----------
        width : int
            How many tokens the spans cover.

        Returns
        -------
        Tensor
            A rank-2 int64 tensor of shape (span, split) with a column per split the
            widest span has.
        """
        start, stop = self.bounds(width)

        return self._right[start:stop]

    def real(self, width: int) -> Tensor:
        """Return which split columns the spans of one width have.

        Parameters
        ----------
        width : int
            How many tokens the spans cover.

        Returns
        -------
        Tensor
            A rank-3 boolean tensor of shape (1, span, split), true at the first
            ``width - 1`` columns of every row.
        """
        start, stop = self.bounds(width)

        return self._real[start:stop].unsqueeze(0)


class InsideKernel:
    """Compute inside values under a selected log-weight reduction.

    Parameters
    ----------
    reduction : LogWeightReduction
        The operation that combines alternative derivations.
    """

    __slots__ = ("_reduction",)

    def __init__(self, reduction: LogWeightReduction, /) -> None:
        self._reduction = reduction

    @property
    def reduction(self) -> LogWeightReduction:
        """Return the operation that combines alternative derivations."""
        return self._reduction

    def inside(
        self, leaf_scores: Tensor, split_scores: Tensor, index: SpanIndex, /
    ) -> Tensor:
        """Return the inside weight of every span.

        Parameters
        ----------
        leaf_scores : Tensor
            A rank-2 floating point tensor of shape (item, token) holding the log weight
            of each token as a one-token span.
        split_scores : Tensor
            A rank-3 floating point tensor of shape (item, span, split) holding the log
            weight of building a span from its two children at each of its splits.
        index : SpanIndex
            The span addressing both tensors are laid out under.

        Returns
        -------
        Tensor
            A rank-2 tensor of shape (item, span) in the scores' floating point dtype,
            holding each span's inside weight under this kernel's algebra.

        Raises
        ------
        TypeError
            If either score tensor is not of a floating point dtype.
        ValueError
            If the index spans an input of length zero, or if the score tensors do not
            have the shapes the index calls for.
        """
        batch = _checked_scores(leaf_scores, split_scores, index)
        gathers = ChildGathers(index, leaf_scores)
        chart = TensorChart(index, leaf_scores, batch=batch, width=1)

        with chart as storage:
            storage.write_width(1, leaf_scores.transpose(0, 1).unsqueeze(-1))

            for width in range(2, index.n + 1):
                start, stop = gathers.bounds(width)
                joined = self.join(chart, split_scores[:, start:stop, :], width=width)
                slab = self._reduction.reduce(joined, gathers.real(width), dim=-1)
                storage.write_width(width, slab.transpose(0, 1).unsqueeze(-1))

            return storage.values().squeeze(-1).transpose(0, 1)

    def join(
        self,
        chart: TensorChart,
        split_scores: Tensor,
        /,
        *,
        width: int,
    ) -> Tensor:
        """Return the weight of every split of every span of one width.

        Parameters
        ----------
        chart : TensorChart
            The packed scalar chart holding every span narrower than the width.
        split_scores : Tensor
            A rank-3 floating point tensor of shape (item, span, split) holding the log
            weight of building each span of the width from its two children at each
            of its splits.
        width : int
            How many tokens the spans cover, at least two.

        Returns
        -------
        Tensor
            A rank-3 tensor of shape (item, span, split) in the chart's floating point
            dtype, holding the product of the two children's weights and the split
            score under this kernel's algebra, with the padding columns unmasked.
        """
        with chart as storage:
            left, right = storage.read_width_children(width)

        return (
            left.squeeze(-1).permute(2, 0, 1)
            + right.squeeze(-1).permute(2, 0, 1)
            + split_scores
        )

    def log_partition(
        self,
        leaf_scores: Tensor,
        split_scores: Tensor,
        index: SpanIndex,
        /,
        *,
        lengths: Tensor | None = None,
    ) -> Tensor:
        """Return each item's goal weight, which is its log partition.

        Parameters
        ----------
        leaf_scores : Tensor
            A rank-2 floating point tensor of shape (item, token) holding the log weight
            of each token as a one-token span.
        split_scores : Tensor
            A rank-3 floating point tensor of shape (item, span, split) holding the log
            weight of building a span from its two children at each of its splits.
        index : SpanIndex
            The span addressing both tensors are laid out under.
        lengths : Tensor | None
            A rank-1 integer tensor of shape (item,) holding how many tokens each item
            has, or None where every item runs the full length.

        Returns
        -------
        Tensor
            A rank-1 tensor of shape (item,) in the scores' floating point dtype,
            holding each item's goal weight.

        Raises
        ------
        TypeError
            If a score tensor is not of a floating point dtype, or if the lengths are
            not of an integer dtype.
        ValueError
            If the index is empty, score or length shapes disagree with it, or a checked
            length is outside its range.
        """
        chart = self.inside(leaf_scores, split_scores, index)
        roots = _root_ids(chart, index, lengths)

        return chart.gather(1, roots.unsqueeze(1)).squeeze(1)
