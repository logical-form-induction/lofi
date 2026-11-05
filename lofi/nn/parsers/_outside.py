"""Outside chart computation."""

from __future__ import annotations

from math import inf, sqrt
from typing import TYPE_CHECKING, override

import torch
from torch import nn

from lofi.nn.parsers._chart import TensorChart
from lofi.nn.parsers._inside import _root_ids
from lofi.nn.parsers._normalize import masked_log_softmax
from lofi.nn.parsers._shapes import checked_scalar_chart, checked_split_scores

if TYPE_CHECKING:
    from torch import Tensor

    from lofi.deduce import SpanIndex
    from lofi.nn.parsers._ports import LogWeightReduction
    from lofi.nn.parsers.functions import SpanFunctionModule
    from lofi.spec.parsers.chart import OutsideWeights, RootOutside

__all__ = [
    "OutsideKernel",
    "ParentContexts",
    "VectorOutsideKernel",
    "log_parent_posterior",
    "parent_contexts",
]


class ParentContexts:
    """Stores each span's parent, sibling, and split positions.

    Parameters
    ----------
    parents : tuple[tuple[int, ...], ...]
        The ids of the spans that have each span as a child, per span, in ascending span
        id order.
    siblings : tuple[tuple[int, ...], ...]
        The id of the other child of each of those, in the same order.
    columns : tuple[tuple[int, ...], ...]
        The column each pair occupies in its parent's split row, in the same order.
    """

    __slots__ = ("_columns", "_parents", "_siblings")

    def __init__(
        self,
        parents: tuple[tuple[int, ...], ...],
        siblings: tuple[tuple[int, ...], ...],
        columns: tuple[tuple[int, ...], ...],
        /,
    ) -> None:
        self._parents = parents
        self._siblings = siblings
        self._columns = columns

    @property
    def parents(self) -> tuple[tuple[int, ...], ...]:
        """Return the spans that have each span as a child, per span."""
        return self._parents

    @property
    def siblings(self) -> tuple[tuple[int, ...], ...]:
        """Return the other child of each of those, per span."""
        return self._siblings

    @property
    def columns(self) -> tuple[tuple[int, ...], ...]:
        """Return the column each pair occupies in its parent's split row, per span."""
        return self._columns

    @override
    def __repr__(self) -> str:
        """Return the number of indexed spans."""
        return f"ParentContexts(spans={len(self._parents)})"


def parent_contexts(index: SpanIndex, /) -> ParentContexts:
    """Return every context each span appears in, in parent posterior order.

    Parameters
    ----------
    index : SpanIndex
        The span addressing to transpose.

    Returns
    -------
    ParentContexts
        The parent, the sibling and the split column of every context of every span, in
        ascending parent span id order, which is by parent width and then by parent
        start.
    """
    parents: list[list[int]] = [[] for _ in range(index.num_spans)]
    siblings: list[list[int]] = [[] for _ in range(index.num_spans)]
    columns: list[list[int]] = [[] for _ in range(index.num_spans)]
    left_rows = index.left_child_ids
    right_rows = index.right_child_ids

    for parent, real in enumerate(index.split_mask):
        for column, is_real in enumerate(real):
            if not is_real:
                continue

            left = left_rows[parent][column]
            right = right_rows[parent][column]

            for child, sibling in ((left, right), (right, left)):
                parents[child].append(parent)
                siblings[child].append(sibling)
                columns[child].append(column)

    return ParentContexts(
        tuple(tuple(row) for row in parents),
        tuple(tuple(row) for row in siblings),
        tuple(tuple(row) for row in columns),
    )


def _parent_slabs(
    index: SpanIndex, like: Tensor
) -> list[tuple[Tensor, Tensor, Tensor]]:
    """Return one gather table per width, on a tensor's device, narrowest first.

    Parameters
    ----------
    index : SpanIndex
        The span addressing to transpose.
    like : Tensor
        A tensor to place the tables beside.

    Returns
    -------
    list[tuple[Tensor, Tensor, Tensor]]
        One triple of rank-2 int64 parent IDs, sibling IDs, and split columns per span
        width.

    Raises
    ------
    ValueError
        If the index spans an input of length zero, which has no spans to gather.
    """
    if index.n < 1:
        msg = f"a span index of length at least one is needed, not {index.n}"
        raise ValueError(msg)

    contexts = parent_contexts(index)
    slabs = []

    for width in range(1, index.n + 1):
        start = index.span_id(0, width)
        stop = start + index.n - width + 1
        slabs.append(
            (
                like.new_tensor(contexts.parents[start:stop], dtype=torch.long),
                like.new_tensor(contexts.siblings[start:stop], dtype=torch.long),
                like.new_tensor(contexts.columns[start:stop], dtype=torch.long),
            )
        )

    return slabs


def log_parent_posterior(
    inside: Tensor, outside: Tensor, split_scores: Tensor, index: SpanIndex, /
) -> Tensor:
    """Return the log posterior over the contexts every span appears in.

    Parameters
    ----------
    inside : Tensor
        A rank-2 floating point tensor of shape (item, span) holding each span's log
        inside weight, as ``InsideKernel.inside`` returns it under log-sum-exp
        reduction.
    outside : Tensor
        A rank-2 floating point tensor of shape (item, span) holding each span's log
        outside weight, as ``OutsideKernel.outside`` returns it under log-sum-exp
        reduction.
    split_scores : Tensor
        A rank-3 floating point tensor of shape (item, span, split) holding the log
        weight of building a span from its two children at each of its splits.
    index : SpanIndex
        The span addressing the three tensors are laid out under.

    Returns
    -------
    Tensor
        A rank-3 tensor of shape (item, span, parent) in the scores' floating point
        dtype, holding the log posterior over the contexts of each span.

    Raises
    ------
    TypeError
        If a tensor is not of a floating point dtype.
    ValueError
        If the index spans an input of length zero, if a tensor has the wrong rank, or
        if the shapes do not match the index and each other.
    """
    batch = checked_scalar_chart("an inside chart", inside, index)
    surrounding = checked_scalar_chart("an outside chart", outside, index)
    checked_split_scores(split_scores, index, batch)

    if surrounding != batch:
        msg = (
            f"an inside chart of {batch} items needs an outside chart of the "
            f"same, not {surrounding}"
        )
        raise ValueError(msg)

    rules = split_scores.reshape(batch, -1)
    real = inside.new_ones((1, 1, 1), dtype=torch.bool)
    rows = []

    for width, (held, beside, column) in enumerate(
        _parent_slabs(index, inside), start=1
    ):
        shape = (batch, index.n - width + 1, index.n - width)
        # width-w spans have n - w contexts, and remaining columns are padding
        empty = outside.new_full((batch, shape[1], width - 1), -inf)

        if shape[-1] == 0:
            rows.append(empty)
            continue

        joined = (
            outside.index_select(1, held.reshape(-1)).reshape(shape)
            + rules.index_select(
                1, (held * index.max_splits + column).reshape(-1)
            ).reshape(shape)
            + inside.index_select(1, beside.reshape(-1)).reshape(shape)
        )
        rows.append(torch.cat((masked_log_softmax(joined, real), empty), dim=-1))

    return torch.cat(rows, dim=1)


class OutsideKernel:
    """Compute outside values under a selected log-weight reduction.

    Parameters
    ----------
    reduction : LogWeightReduction
        The operation that combines alternative derivations, matching the reduction
        used to compute the inside chart handed to ``outside``.
    """

    __slots__ = ("_reduction",)

    def __init__(self, reduction: LogWeightReduction, /) -> None:
        self._reduction = reduction

    @property
    def reduction(self) -> LogWeightReduction:
        """Return the operation that combines alternative derivations."""
        return self._reduction

    def outside(
        self,
        inside: Tensor,
        split_scores: Tensor,
        index: SpanIndex,
        /,
        *,
        lengths: Tensor | None = None,
    ) -> Tensor:
        """Return the outside weight of every span.

        Parameters
        ----------
        inside : Tensor
            A rank-2 floating point tensor of shape (item, span) holding each span's log
            inside weight, as ``InsideKernel.inside`` returns it under this kernel's
            algebra.
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
            A rank-2 tensor of shape (item, span) in the scores' floating point dtype,
            holding each span's outside weight under this kernel's algebra.

        Raises
        ------
        TypeError
            If a tensor is not of a floating point dtype, or if the lengths are not of
            an integer dtype.
        ValueError
            If the index is empty, tensor or length shapes disagree with it, or a
            checked length is outside its range.
        """
        batch = checked_scalar_chart("an inside chart", inside, index)
        checked_split_scores(split_scores, index, batch)
        slabs = _parent_slabs(index, inside)
        roots = _root_ids(inside, index, lengths)
        span_ids = inside.new_tensor(range(index.num_spans), dtype=torch.long)
        # seed each item's goal span with log(1)
        seeds = inside.new_full((batch, index.num_spans), -inf).masked_fill(
            span_ids.unsqueeze(0) == roots.unsqueeze(1), 0.0
        )
        rules = split_scores.reshape(batch, -1)
        real = inside.new_ones((1, 1, 1), dtype=torch.bool)
        chart = TensorChart(index, inside, batch=batch, width=1)

        with chart as storage:
            for width, (held, beside, column) in zip(
                range(index.n, 0, -1), reversed(slabs), strict=True
            ):
                start = index.span_id(0, width)
                stop = start + index.n - width + 1
                shape = (batch, stop - start, index.n - width)
                context = (
                    storage.read_span_ids(held.reshape(-1))
                    .squeeze(-1)
                    .transpose(0, 1)
                    .reshape(shape)
                    + rules.index_select(
                        1, (held * index.max_splits + column).reshape(-1)
                    ).reshape(shape)
                    + inside.index_select(1, beside.reshape(-1)).reshape(shape)
                )
                terms = torch.cat((context, seeds[:, start:stop].unsqueeze(-1)), dim=-1)
                slab = self._reduction.reduce(terms, real, dim=-1)
                storage.write_width(width, slab.transpose(0, 1).unsqueeze(-1))

            return storage.values().squeeze(-1).transpose(0, 1)


def _vector_context_tables(
    index: SpanIndex, width: int, like: Tensor, /
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return parent, sibling, column, and child-side tables for one width."""
    contexts = parent_contexts(index)
    start = index.span_id(0, width)
    stop = start + index.n - width + 1
    parent_rows = contexts.parents[start:stop]
    sibling_rows = contexts.siblings[start:stop]
    column_rows = contexts.columns[start:stop]
    child_is_right = tuple(
        tuple(
            index.right_child_ids[parent][column] == child
            for parent, column in zip(parents, columns, strict=True)
        )
        for child, (parents, columns) in enumerate(
            zip(parent_rows, column_rows, strict=True), start=start
        )
    )

    return (
        like.new_tensor(parent_rows, dtype=torch.long),
        like.new_tensor(sibling_rows, dtype=torch.long),
        like.new_tensor(column_rows, dtype=torch.long),
        like.new_tensor(child_is_right, dtype=torch.bool),
    )


class VectorOutsideKernel(nn.Module):
    """Compute one contextual vector for every span from its parent contexts."""

    def __init__(
        self,
        split_left: SpanFunctionModule,
        split_right: SpanFunctionModule,
        *,
        width: int,
        weighting: OutsideWeights,
        root: RootOutside,
        root_init_gain: float,
    ) -> None:
        super().__init__()

        if split_left.input_widths != (width, width):
            msg = (
                f"split_left reads widths {split_left.input_widths}, "
                f"not {(width, width)}"
            )
            raise ValueError(msg)

        if split_right.input_widths != (width, width):
            msg = (
                f"split_right reads widths {split_right.input_widths}, "
                f"not {(width, width)}"
            )
            raise ValueError(msg)

        if split_left.output_width != width or split_right.output_width != width:
            msg = f"both split functions return the chart width {width}"
            raise ValueError(msg)

        self._split_left = split_left
        self._split_right = split_right
        self._width = width
        self._weighting = weighting

        if root == "learned":
            self._root = nn.Parameter(
                torch.empty(width).normal_(std=root_init_gain / sqrt(width))
            )
        else:
            self.register_buffer("_root", torch.zeros(width))

    @property
    def width(self) -> int:
        """Return the width of every inside and outside vector."""
        return self._width

    @property
    def weighting(self) -> OutsideWeights:
        """Return how contexts are weighted in the recursion."""
        return self._weighting

    def outside(
        self,
        cells: Tensor,
        inside: Tensor,
        outside: Tensor,
        split_scores: Tensor,
        alpha: Tensor,
        index: SpanIndex,
        /,
        *,
        lengths: Tensor | None = None,
    ) -> Tensor:
        """Return vector outside values over ``(span, item, width)``."""
        batch = checked_scalar_chart("an inside chart", inside, index)
        surrounding = checked_scalar_chart("an outside chart", outside, index)
        checked_split_scores(split_scores, index, batch)
        checked_split_scores(alpha, index, batch)

        expected = (index.num_spans, batch, self._width)

        if tuple(cells.shape) != expected:
            msg = (
                f"an inside-vector chart has shape {expected}, not {tuple(cells.shape)}"
            )
            raise ValueError(msg)

        if not cells.is_floating_point():
            msg = f"inside vectors use floating point, not {cells.dtype}"
            raise TypeError(msg)

        if surrounding != batch:
            msg = (
                f"an inside chart of {batch} items needs an outside chart of the "
                f"same, not {surrounding}"
            )
            raise ValueError(msg)

        roots = _root_ids(inside, index, lengths)
        span_ids = inside.new_tensor(range(index.num_spans), dtype=torch.long)
        root_rows = span_ids.unsqueeze(0) == roots.unsqueeze(1)
        root = self._root.to(dtype=cells.dtype, device=cells.device)
        values = [
            root.unsqueeze(0).expand(batch, -1) * root_rows[:, span].unsqueeze(-1)
            for span in range(index.num_spans)
        ]
        inside_vectors = cells.transpose(0, 1)
        contextual = (
            log_parent_posterior(inside, outside, split_scores, index)
            if self._weighting == "contextual"
            else None
        )
        flattened_alpha = alpha.reshape(batch, -1)

        for width in range(index.n - 1, 0, -1):
            start = index.span_id(0, width)
            stop = start + index.n - width + 1
            parents, siblings, columns, child_is_right = _vector_context_tables(
                index, width, inside
            )
            contexts = index.n - width
            table = torch.stack(values, dim=1)
            parent_vectors = table.index_select(1, parents.reshape(-1)).reshape(
                batch, stop - start, contexts, self._width
            )
            sibling_vectors = inside_vectors.index_select(
                1, siblings.reshape(-1)
            ).reshape(batch, stop - start, contexts, self._width)
            from_left = self._split_left(parent_vectors, sibling_vectors)
            from_right = self._split_right(parent_vectors, sibling_vectors)
            candidates = torch.where(
                child_is_right.unsqueeze(0).unsqueeze(-1), from_left, from_right
            )

            if contextual is None:
                offsets = parents * index.max_splits + columns
                log_weights = flattened_alpha.index_select(
                    1, offsets.reshape(-1)
                ).reshape(batch, stop - start, contexts)
            else:
                log_weights = contextual[:, start:stop, :contexts]

            mixed = (log_weights.exp().unsqueeze(-1) * candidates).sum(dim=2)
            targets = span_ids[start:stop]
            live = (
                torch.ones(
                    (batch, stop - start), dtype=torch.bool, device=inside.device
                )
                if lengths is None
                else inside.new_tensor(
                    [end for _, end in index.spans_by_length()][start:stop],
                    dtype=torch.long,
                ).unsqueeze(0)
                <= lengths.to(torch.long).unsqueeze(1)
            )
            seeded = root_rows.index_select(1, targets)
            mixed = torch.where(seeded.unsqueeze(-1), root, mixed)
            mixed = mixed.masked_fill(~live.unsqueeze(-1), 0.0)

            for offset, span in enumerate(range(start, stop)):
                values[span] = mixed[:, offset]

        return torch.stack(values).to(cells.dtype)

    @override
    def forward(
        self,
        cells: Tensor,
        inside: Tensor,
        outside: Tensor,
        split_scores: Tensor,
        alpha: Tensor,
        index: SpanIndex,
        /,
        *,
        lengths: Tensor | None = None,
    ) -> Tensor:
        """Return vector outside values over ``(span, item, width)``."""
        return self.outside(
            cells,
            inside,
            outside,
            split_scores,
            alpha,
            index,
            lengths=lengths,
        )
