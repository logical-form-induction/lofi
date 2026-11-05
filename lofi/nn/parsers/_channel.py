"""Vector-valued chart cells and split posteriors."""

from math import inf
from typing import TYPE_CHECKING, Final

import torch
from torch.utils.checkpoint import checkpoint

from lofi.nn.parsers._chart import TensorChart
from lofi.nn.parsers._inside import ChildGathers, InsideKernel
from lofi.nn.parsers._reduction import LogSumExpReduction
from lofi.nn.parsers._shapes import (
    checked_leaf_scores,
    checked_scalar_chart,
    checked_split_scores,
)
from lofi.nn.parsers._values import ChanneledChart, ScoredChart
from lofi.nn.parsers._weigher import MarginalSplitWeigher, SplitWeigher

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

    from lofi.deduce import SpanIndex
    from lofi.nn.parsers._chart import _TensorChartAccess
    from lofi.nn.parsers._ports import LogWeightReduction

__all__ = ["Composition", "SplitChannel", "SplitScoring", "log_split_posterior"]

type Composition = Callable[[Tensor, Tensor], Tensor]
"""Defines the child-composition callable."""

type SplitScoring = Callable[[Tensor, Tensor], Tensor]
"""Scores the splits of one width from the child vectors on each side.

Both arguments are rank 4 over (item, span, split, width), with a column per split the
spans of that width have, and the result is rank 3 over (item, span, split) in log
space.
"""

#: The rank of a batch of leaf vectors: item, then token, then width.
_LEAF_RANK: Final[int] = 3

_DEFAULT_REDUCTION: Final = LogSumExpReduction()
"""The reduction used by a directly constructed probability chart."""


def _checked_leaf_vectors(leaf_vectors: Tensor, index: SpanIndex) -> None:
    """Raise if the leaf vectors are not one floating point vector per token.

    Parameters
    ----------
    leaf_vectors : Tensor
        The vector each token enters the chart with.
    index : SpanIndex
        The span addressing the vectors are laid out under.

    Raises
    ------
    TypeError
        If the vectors are not of a floating point dtype.
    ValueError
        If they are not of rank 3, or hold another number of tokens than the index.
    """
    if not leaf_vectors.is_floating_point():
        msg = f"a cell holds a vector at floating precision, not {leaf_vectors.dtype}"
        raise TypeError(msg)

    if leaf_vectors.ndim != _LEAF_RANK:
        msg = (
            f"leaf vectors run over item, token and width, so they have "
            f"rank {_LEAF_RANK}, not {leaf_vectors.ndim}"
        )
        raise ValueError(msg)

    if int(leaf_vectors.shape[1]) != index.n:
        msg = (
            f"an input of length {index.n} has that many leaf vectors, "
            f"not {int(leaf_vectors.shape[1])}"
        )
        raise ValueError(msg)


def _checked_items(leaf_vectors: Tensor, batch: int) -> None:
    """Raise if the leaf vectors cover another number of items than the scores.

    Parameters
    ----------
    leaf_vectors : Tensor
        The vector each token enters the chart with.
    batch : int
        How many items the scores cover.

    Raises
    ------
    ValueError
        If the two disagree.
    """
    if int(leaf_vectors.shape[0]) != batch:
        msg = (
            f"leaf vectors for {int(leaf_vectors.shape[0])} items came "
            f"with scores for {batch}"
        )
        raise ValueError(msg)


def log_split_posterior(
    inside: Tensor, split_scores: Tensor, index: SpanIndex, /
) -> Tensor:
    """Return the log posterior over the splits of every span.

    Parameters
    ----------
    inside : Tensor
        A rank-2 floating point tensor of shape (item, span) holding each span's log
        inside score, as ``InsideKernel.inside`` returns it under log-sum-exp
        reduction.
    split_scores : Tensor
        A rank-3 floating point tensor of shape (item, span, split) holding the log
        weight of building a span from its two children at each of its splits.
    index : SpanIndex
        The span addressing both tensors are laid out under.

    Returns
    -------
    Tensor
        A rank-3 tensor of shape (item, span, split) in the scores' floating point
        dtype, holding the log posterior over each span's splits.

    Raises
    ------
    TypeError
        If either tensor is not of a floating point dtype.
    ValueError
        If either tensor has the wrong rank, or if their shapes do not match the index
        and each other.
    """
    return _split_mixing(
        inside,
        split_scores,
        index,
        reduction=LogSumExpReduction(),
        weigher=MarginalSplitWeigher(),
    )


def _split_mixing(
    inside: Tensor,
    split_scores: Tensor,
    index: SpanIndex,
    /,
    *,
    reduction: LogWeightReduction,
    weigher: SplitWeigher,
) -> Tensor:
    """Return the reduction's log mixing weight over every span's splits."""
    batch = checked_scalar_chart("an inside chart", inside, index)
    checked_split_scores(split_scores, index, batch)
    shape = (batch, index.num_spans, index.max_splits)

    if index.max_splits == 0:
        return split_scores.clone()

    left_ids = inside.new_tensor(index.left_child_ids, dtype=torch.long).reshape(-1)
    right_ids = inside.new_tensor(index.right_child_ids, dtype=torch.long).reshape(-1)
    real = inside.new_tensor(index.split_mask, dtype=torch.bool).unsqueeze(0)
    joined = (
        inside.index_select(1, left_ids).reshape(shape)
        + inside.index_select(1, right_ids).reshape(shape)
        + split_scores
    )

    return weigher.weights(joined, split_scores, None, None, real, reduction, dim=-1)


class SplitChannel:
    """Computes scalar and vector charts in one inside sweep.

    Parameters
    ----------
    compose : Composition
        What builds a parent's vector from a pair of child vectors.
    weigher : SplitWeigher
        What normalizes the candidate splits mixed into a parent vector.
    checkpoint_from_width : int | None
        The narrowest span width whose cells are mixed under activation
        checkpointing: the composed pairs of such a span are not held for the
        backward pass but recomputed during it, so the activations held for a
        long input grow more slowly at the cost of a second composition. None
        holds every width's composed pairs.

    Raises
    ------
    TypeError
        If what composes a pair of cells cannot be called.
    ValueError
        If the checkpointing width is below two.
    """

    __slots__ = ("_checkpoint_from_width", "_compose", "_kernel", "_weigher")

    def __init__(
        self,
        compose: Composition,
        /,
        *,
        weigher: SplitWeigher,
        reduction: LogWeightReduction = _DEFAULT_REDUCTION,
        checkpoint_from_width: int | None = None,
    ) -> None:
        if not callable(compose):
            msg = f"a pair of cells is composed by a callable, not by {compose!r}"
            raise TypeError(msg)

        if checkpoint_from_width is not None and checkpoint_from_width < 2:
            msg = (
                f"cells are composed from width two, so checkpointing starts at a "
                f"width of at least two, not {checkpoint_from_width}"
            )
            raise ValueError(msg)

        self._compose = compose
        self._weigher = weigher
        self._checkpoint_from_width = checkpoint_from_width
        self._kernel = InsideKernel(reduction)

    @property
    def reduction(self) -> LogWeightReduction:
        """Return the operation that combines and mixes alternatives."""
        return self._kernel.reduction

    @property
    def checkpoint_from_width(self) -> int | None:
        """Return the narrowest width mixed under checkpointing, or None for none."""
        return self._checkpoint_from_width

    @property
    def compose(self) -> Composition:
        """Return what builds a parent's vector from a pair of child vectors."""
        return self._compose

    @property
    def weigher(self) -> SplitWeigher:
        """Return what normalizes the alternatives mixed into each chart cell."""
        return self._weigher

    def sweep(
        self,
        leaf_vectors: Tensor,
        leaf_scores: Tensor,
        split_scores: Tensor,
        index: SpanIndex,
        /,
    ) -> ChanneledChart:
        """Return the scalar chart, the split posterior and the mixed vectors.

        Parameters
        ----------
        leaf_vectors : Tensor
            A rank-3 floating point tensor of shape (item, token, width) holding the
            vector each token enters the chart with.
        leaf_scores : Tensor
            A rank-2 floating point tensor of shape (item, token) holding the log weight
            of each token as a one-token span.
        split_scores : Tensor
            A rank-3 floating point tensor of shape (item, span, split) holding the log
            weight of building a span from its two children at each of its splits.
        index : SpanIndex
            The span addressing the three tensors are laid out under.

        Returns
        -------
        ChanneledChart
            The inside chart over (item, span), the log split posterior over (item,
            span, split), and the vector chart over (span, item, width) in the leaf
            vectors' floating point dtype.

        Raises
        ------
        TypeError
            If a tensor is not of a floating point dtype.
        ValueError
            If the index is empty, tensor shapes or item counts are invalid, or
            composition does not return one vector per pair.
        """
        _checked_leaf_vectors(leaf_vectors, index)
        inside = self._kernel.inside(leaf_scores, split_scores, index)
        _checked_items(leaf_vectors, int(inside.shape[0]))
        gathers = ChildGathers(index, leaf_vectors)
        chart = TensorChart(
            index,
            leaf_vectors,
            batch=int(leaf_vectors.shape[0]),
            width=int(leaf_vectors.shape[2]),
        )

        with chart as cells:
            cells.write_width(1, leaf_vectors.transpose(0, 1))
            alphas = [
                split_scores.new_full(
                    (int(leaf_vectors.shape[0]), index.n, index.max_splits), -inf
                )
            ]

            for width in range(2, index.n + 1):
                start, stop = gathers.bounds(width)
                splits = width - 1
                left, right = cells.read_width_children(width)
                left = left[:, :splits]
                right = right[:, :splits]
                spans = stop - start
                batch = int(leaf_vectors.shape[0])
                shape = (batch, spans, splits)
                left_ids = gathers.left(width)[:, :splits].reshape(-1)
                right_ids = gathers.right(width)[:, :splits].reshape(-1)
                joined = (
                    inside.index_select(1, left_ids).reshape(shape)
                    + inside.index_select(1, right_ids).reshape(shape)
                    + split_scores[:, start:stop, :splits]
                )
                share = self._weigher.weights(
                    joined,
                    split_scores[:, start:stop, :splits],
                    left.permute(2, 0, 1, 3),
                    right.permute(2, 0, 1, 3),
                    gathers.real(width)[:, :, :splits],
                    self._kernel.reduction,
                    dim=-1,
                )
                alphas.append(
                    torch.cat(
                        (
                            share,
                            share.new_full(
                                (batch, spans, index.max_splits - splits), -inf
                            ),
                        ),
                        dim=-1,
                    )
                )
                cells.write_width(
                    width,
                    self._held_or_recomputed(
                        left,
                        right,
                        share,
                        width=width,
                    ),
                )

            return ChanneledChart(
                inside, torch.cat(alphas, dim=1), cells.values(), index
            )

    def sweep_scoring(
        self,
        leaf_vectors: Tensor,
        leaf_scores: Tensor,
        scoring: SplitScoring,
        index: SpanIndex,
        /,
        *,
        recompute: bool,
    ) -> ScoredChart:
        """Return the sweep whose split scores are read off the cells it builds.

        The sweep proceeds by width. For each width it hands the scoring the vectors
        of the two children at every split of every span, takes back the log weight of
        each split, reduces those to the scalar inside weight and the split posterior
        of the width, and mixes the composed children under that posterior into the
        cells the next width reads.

        Parameters
        ----------
        leaf_vectors : Tensor
            A rank-3 floating point tensor of shape (item, token, width) holding the
            vector each token enters the chart with.
        leaf_scores : Tensor
            A rank-2 floating point tensor of shape (item, token) holding the log weight
            of each token as a one-token span.
        scoring : SplitScoring
            What scores the splits of one width from the child vectors on each side.
        index : SpanIndex
            The span addressing the tensors are laid out under.
        recompute : bool
            Whether the scoring may run again during the backward pass, which
            checkpointing a width does. A scoring that records what it is handed
            may not, and its widths are never checkpointed.

        Returns
        -------
        ScoredChart
            The channeled chart and the split scores the scoring returned, over (item,
            span, split) with negative infinity at every column a span does not have.

        Raises
        ------
        TypeError
            If a tensor is not of a floating point dtype.
        ValueError
            If the index is empty, tensor shapes or item counts are invalid, if the
            scoring returns something other than one log weight per split, or if
            composition does not return one vector per pair.
        """
        _checked_leaf_vectors(leaf_vectors, index)

        if index.n < 1:
            msg = f"a sweep needs an input of length at least one, not {index.n}"
            raise ValueError(msg)

        batch = checked_leaf_scores(leaf_scores, index)
        _checked_items(leaf_vectors, batch)
        gathers = ChildGathers(index, leaf_vectors)
        cell_chart = TensorChart(
            index,
            leaf_vectors,
            batch=batch,
            width=int(leaf_vectors.shape[2]),
        )
        inside_chart = TensorChart(index, leaf_scores, batch=batch, width=1)
        rows = [
            leaf_scores.new_full((batch, index.n, index.max_splits), -inf),
        ]
        alphas = [rows[0]]

        with cell_chart as cells, inside_chart as inside:
            cells.write_width(1, leaf_vectors.transpose(0, 1))
            inside.write_width(1, leaf_scores.transpose(0, 1).unsqueeze(-1))

            for span_width in range(2, index.n + 1):
                padded, slab, share, mixed = self._scored_width(
                    scoring,
                    cells,
                    inside,
                    gathers,
                    width=span_width,
                    recompute=recompute,
                )
                inside.write_width(span_width, slab.transpose(0, 1).unsqueeze(-1))
                rows.append(padded)
                alphas.append(share)
                cells.write_width(span_width, mixed)

            alpha = torch.cat(alphas, dim=1)

            return ScoredChart(
                ChanneledChart(
                    inside.values().squeeze(-1).transpose(0, 1),
                    alpha,
                    cells.values(),
                    index,
                ),
                torch.cat(rows, dim=1),
            )

    def _scored_width(
        self,
        scoring: SplitScoring,
        cells: _TensorChartAccess,
        inside: _TensorChartAccess,
        gathers: ChildGathers,
        /,
        *,
        width: int,
        recompute: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return one width's split scores, inside slab, split posterior and cells.

        Under checkpointing, which applies from the declared width while
        gradients are recorded and the scoring may run again, the width's
        intermediate activations are recomputed during the backward pass rather
        than held from the forward pass. The returned values are the same either
        way.

        Parameters
        ----------
        scoring : SplitScoring
            What scores the splits of the width from the child vectors.
        cells : _TensorChartAccess
            The vector chart holding every narrower span.
        inside : _TensorChartAccess
            The scalar chart holding every narrower span.
        gathers : ChildGathers
            The child ids of the index the chart is laid out under.
        width : int
            How many tokens the spans cover, at least two.
        recompute : bool
            Whether the scoring may run again during the backward pass.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor, Tensor]
            The split scores padded to every split column, the inside weight of
            each span of the width, the log split posterior, and the mixed cells.
        """
        threshold = self._checkpoint_from_width

        def computed_from_prefix(
            cell_values: Tensor,
            inside_values: Tensor,
        ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
            spans = gathers.index.n - width + 1
            splits = gathers.index.max_splits
            batch = cells.batch
            left = cell_values.index_select(0, gathers.left(width).reshape(-1)).reshape(
                spans, splits, batch, cells.width
            )
            right = cell_values.index_select(
                0, gathers.right(width).reshape(-1)
            ).reshape(spans, splits, batch, cells.width)
            inside_left = inside_values.index_select(
                0, gathers.left(width).reshape(-1)
            ).reshape(spans, splits, batch, 1)
            inside_right = inside_values.index_select(
                0, gathers.right(width).reshape(-1)
            ).reshape(spans, splits, batch, 1)

            return self._width_values(
                scoring,
                left,
                right,
                inside_left + inside_right,
                gathers,
                width=width,
            )

        if (
            not recompute
            or threshold is None
            or width < threshold
            or not torch.is_grad_enabled()
        ):
            return computed_from_prefix(
                cells.view_prefix(width), inside.view_prefix(width)
            )

        return checkpoint(
            computed_from_prefix,
            cells.read_prefix(width),
            inside.read_prefix(width),
            use_reentrant=False,
        )

    def _width_values(
        self,
        scoring: SplitScoring,
        left: Tensor,
        right: Tensor,
        inside_children: Tensor,
        gathers: ChildGathers,
        /,
        *,
        width: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return one width's split scores, inside slab, split posterior and cells.

        Parameters
        ----------
        scoring : SplitScoring
            What scores the splits of the width from the child vectors.
        left : Tensor
            The left child vectors of every split of the width.
        right : Tensor
            The right child vectors over the same axes.
        inside_children : Tensor
            The summed scalar inside weights of each pair of children.
        gathers : ChildGathers
            The child ids of the index the chart is laid out under.
        width : int
            How many tokens the spans cover, at least two.
        """
        index = gathers.index
        batch = int(left.shape[2])
        splits = width - 1
        spans = index.n - width + 1
        scored = self._scored(
            scoring,
            left[:, :splits].permute(2, 0, 1, 3),
            right[:, :splits].permute(2, 0, 1, 3),
            (batch, spans, splits),
        )
        padded = torch.cat(
            (scored, scored.new_full((batch, spans, index.max_splits - splits), -inf)),
            dim=-1,
        )
        real = gathers.real(width)
        joined = inside_children.squeeze(-1).permute(2, 0, 1) + padded
        share = self._weigher.weights(
            joined,
            padded,
            left.permute(2, 0, 1, 3),
            right.permute(2, 0, 1, 3),
            real,
            self._kernel.reduction,
            dim=-1,
        )

        return (
            padded,
            self._kernel.reduction.reduce(joined, real, dim=-1),
            share,
            self._mixed(left[:, :splits], right[:, :splits], share[:, :, :splits]),
        )

    def _scored(
        self,
        scoring: SplitScoring,
        left: Tensor,
        right: Tensor,
        shape: tuple[int, int, int],
    ) -> Tensor:
        """Return the log weight of every split of one width.

        Parameters
        ----------
        scoring : SplitScoring
            What scores the splits from the child vectors.
        left : Tensor
            The left child of every split, rank 4 over (item, span, split, width).
        right : Tensor
            The right child of every split, over the same axes.
        shape : tuple[int, int, int]
            The item, span and split extents the weights are expected over.

        Returns
        -------
        Tensor
            A rank-3 tensor of that shape in log space.

        Raises
        ------
        TypeError
            If the scoring returns something other than floating point log weights.
        ValueError
            If it returns something other than one log weight per split, which would
            leave the sweep reducing a row of the wrong length.
        """
        scored = scoring(left, right)

        if not scored.is_floating_point():
            msg = f"a split's log weight is floating point, not {scored.dtype}"
            raise TypeError(msg)

        if tuple(scored.shape) != shape:
            msg = (
                f"scoring the splits of {shape} gives one log weight each, so "
                f"{shape}, not {tuple(scored.shape)}"
            )
            raise ValueError(msg)

        return scored

    def _composed(self, left: Tensor, right: Tensor) -> Tensor:
        """Return one composed vector per pair.

        Parameters
        ----------
        left : Tensor
            The left cell of each pair, rank 2 of shape (pair, width).
        right : Tensor
            The right cell of each pair, in the order of the pair axis.

        Returns
        -------
        Tensor
            The composed vectors, rank 2 of shape (pair, width) in the cells' floating
            point dtype.

        Raises
        ------
        ValueError
            If the composition returns something other than one vector per pair, which
            would leave the sweep writing a cell of the wrong size.
        """
        composed = self._compose(left, right)

        if tuple(composed.shape) != tuple(left.shape):
            msg = (
                f"composing {tuple(left.shape)} pairs of cells gives one vector "
                f"each, so {tuple(left.shape)}, not {tuple(composed.shape)}"
            )
            raise ValueError(msg)

        return composed

    def _held_or_recomputed(
        self, left: Tensor, right: Tensor, share: Tensor, /, *, width: int
    ) -> Tensor:
        """Return the mixed cells of one width, checkpointed from the declared width.

        The cells are the same either way. Under checkpointing, which applies only
        while gradients are recorded, the composed pairs are recomputed during the
        backward pass instead of being held from the forward pass.

        Parameters
        ----------
        left : Tensor
            The left child vectors of the width's real splits.
        right : Tensor
            The right child vectors over the same axes.
        share : Tensor
            The log posterior over each span's splits, rank 3 of shape (item, span,
            split) with exactly the columns the spans of the width have.
        width : int
            How many tokens the spans cover, at least two.
        """
        threshold = self._checkpoint_from_width

        if threshold is None or width < threshold or not torch.is_grad_enabled():
            return self._mixed(left, right, share)

        def mixed(left_values: Tensor, right_values: Tensor, weights: Tensor) -> Tensor:
            return self._mixed(left_values, right_values, weights)

        return checkpoint(mixed, left, right, share, use_reentrant=False)

    def _mixed(self, left: Tensor, right: Tensor, share: Tensor, /) -> Tensor:
        """Return the vector of every span of one width, mixed over its splits.

        Parameters
        ----------
        left : Tensor
            The left child vectors of the width's real splits.
        right : Tensor
            The right child vectors over the same axes.
        share : Tensor
            The log posterior over each span's splits, rank 3 of shape (item, span,
            split) with exactly the columns the spans of the width have.

        Returns
        -------
        Tensor
            A rank-3 tensor of shape (span, item, width) in the cells' floating point
            dtype, holding the vector of every span of the width.
        """
        shape = tuple(left.shape)
        features = int(left.shape[-1])
        pairs = self._composed(
            left.reshape(-1, features), right.reshape(-1, features)
        ).reshape(shape)
        weights = share.to(left.dtype).exp().permute(1, 2, 0).unsqueeze(-1)

        return (weights * pairs).sum(dim=1)
