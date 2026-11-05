"""Packed triangular tensors for differentiable charts."""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

    from lofi.deduce import SpanIndex

__all__ = ["TensorChart"]


class _TensorChartAccess:
    """Provides one sweep with vectorized access to a packed chart."""

    __slots__ = (
        "_batch",
        "_bounds",
        "_children",
        "_prefix",
        "_spans",
        "_values",
        "_view_prefix",
        "_width",
        "_write",
    )

    def __init__(
        self,
        *,
        batch: int,
        width: int,
        bounds: Callable[[int], tuple[int, int]],
        children: Callable[[int], tuple[Tensor, Tensor]],
        spans: Callable[[Tensor], Tensor],
        prefix: Callable[[int], Tensor],
        view_prefix: Callable[[int], Tensor],
        values: Callable[[], Tensor],
        write: Callable[[int, Tensor], None],
    ) -> None:
        self._batch = batch
        self._width = width
        self._bounds = bounds
        self._children = children
        self._spans = spans
        self._prefix = prefix
        self._view_prefix = view_prefix
        self._values = values
        self._write = write

    @property
    def batch(self) -> int:
        """Return the batch size."""
        return self._batch

    @property
    def width(self) -> int:
        """Return the cell-vector width."""
        return self._width

    def bounds(self, span_width: int, /) -> tuple[int, int]:
        """Return the packed bounds of one span width."""
        return self._bounds(span_width)

    def read_width_children(self, span_width: int, /) -> tuple[Tensor, Tensor]:
        """Return copied child values for every cell of one span width."""
        return self._children(span_width)

    def read_span_ids(self, span_ids: Tensor, /) -> Tensor:
        """Return copied values for packed span ids."""
        return self._spans(span_ids)

    def read_prefix(self, span_width: int, /) -> Tensor:
        """Return a copy of every cell narrower than one span width."""
        return self._prefix(span_width)

    def view_prefix(self, span_width: int, /) -> Tensor:
        """Return the interior prefix consumed before later widths are written."""
        return self._view_prefix(span_width)

    def values(self) -> Tensor:
        """Return a copy of every packed cell in span-id order."""
        return self._values()

    def write_width(self, span_width: int, values: Tensor, /) -> None:
        """Write every cell of one span width from a validated packed slab."""
        self._write(span_width, values)


class TensorChart:
    """Stores a differentiable chart for one input length.

    Parameters
    ----------
    index : SpanIndex
        The span addressing this chart uses.
    like : Tensor
        A tensor to allocate the store off.
    batch : int
        How many items a cell holds a value for.
    width : int
        How wide the value at one cell is.

    Raises
    ------
    TypeError
        If the tensor to allocate off is not of a floating point dtype.
    ValueError
        If the index spans an input of length zero, which has no cells, or if the batch
        or the width is not positive.
    """

    __slots__ = ("_batch", "_data", "_index", "_left_ids", "_right_ids", "_width")

    def __init__(
        self, index: SpanIndex, like: Tensor, /, *, batch: int, width: int
    ) -> None:
        if index.n < 1:
            msg = f"a chart spans an input of length at least one, not {index.n}"
            raise ValueError(msg)

        if not like.is_floating_point():
            msg = f"a chart accumulates in a floating point dtype, not {like.dtype}"
            raise TypeError(msg)

        if batch < 1:
            msg = f"a chart holds at least one item, not {batch}"
            raise ValueError(msg)

        if width < 1:
            msg = f"a chart cell is at least one wide, not {width}"
            raise ValueError(msg)

        self._index = index
        self._batch = batch
        self._width = width
        self._data: Tensor = like.new_zeros((index.num_spans, batch, width))
        self._left_ids: Tensor | None = None
        self._right_ids: Tensor | None = None

    @property
    def index(self) -> SpanIndex:
        """Return the span addressing this chart uses."""
        return self._index

    @property
    def batch(self) -> int:
        """Return the batch size."""
        return self._batch

    @property
    def width(self) -> int:
        """Return the cell-vector width."""
        return self._width

    @property
    def dtype(self) -> torch.dtype:
        """Return the floating point dtype the store accumulates in."""
        return self._data.dtype

    def __enter__(self) -> _TensorChartAccess:
        """Open vectorized interior access for one validated sweep."""
        return _TensorChartAccess(
            batch=self._batch,
            width=self._width,
            bounds=self._bounds,
            children=self._read_width_children,
            spans=self._read_span_ids,
            prefix=self._read_prefix,
            view_prefix=self._view_prefix,
            values=self._values,
            write=self._write_width,
        )

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: object,
        /,
    ) -> None:
        """Close vectorized interior access after success or failure."""
        del kind, error, traceback

    def _cell(self, i: int, j: int) -> int:
        """Return the span id of a cell.

        Parameters
        ----------
        i : int
            The start of the span, inclusive.
        j : int
            The end of the span, exclusive.

        Returns
        -------
        int
            The span's id.

        Raises
        ------
        IndexError
            If the span is empty, inverted, or runs past the input.
        """
        if not 0 <= i < j <= self._index.n:
            msg = f"({i}, {j}) is not a span of an input of length {self._index.n}"
            raise IndexError(msg)

        return self._index.span_id(i, j)

    def _children(self) -> tuple[Tensor, Tensor]:
        """Return the left and right child id vectors, building them once.

        Returns
        -------
        tuple[Tensor, Tensor]
            Two rank-2 int64 tensors of shape (num_spans, max_splits), on the store's
            device: per span id, the ids of the left children over all splits, and the
            ids of the right children.
        """
        left = self._left_ids
        right = self._right_ids

        if left is None or right is None:
            left = self._data.new_tensor(self._index.left_child_ids, dtype=torch.long)
            right = self._data.new_tensor(self._index.right_child_ids, dtype=torch.long)
            self._left_ids = left
            self._right_ids = right

        return (left, right)

    def _bounds(self, span_width: int, /) -> tuple[int, int]:
        """Return the packed bounds of one span width."""
        start = self._index.span_id(0, span_width)

        return (start, start + self._index.n - span_width + 1)

    def _read_width_children(self, span_width: int, /) -> tuple[Tensor, Tensor]:
        """Return copied child values for every cell of one span width."""
        start, stop = self._bounds(span_width)
        spans = stop - start
        splits = self._index.max_splits
        shape = (spans, splits, self._batch, self._width)
        left, right = self._children()

        return (
            self._data.index_select(0, left[start:stop].reshape(-1)).reshape(shape),
            self._data.index_select(0, right[start:stop].reshape(-1)).reshape(shape),
        )

    def _read_span_ids(self, span_ids: Tensor, /) -> Tensor:
        """Return copied values for packed span ids."""
        return self._data.index_select(0, span_ids)

    def _read_prefix(self, span_width: int, /) -> Tensor:
        """Return a copy of every cell narrower than one span width."""
        start, _ = self._bounds(span_width)

        return self._data[:start].clone()

    def _view_prefix(self, span_width: int, /) -> Tensor:
        """Return the prefix a sweep has finished writing."""
        start, _ = self._bounds(span_width)

        return self._data[:start]

    def _values(self) -> Tensor:
        """Return a copy of every packed cell in span-id order."""
        return self._data.clone()

    def _write_width(self, span_width: int, values: Tensor, /) -> None:
        """Write every cell of one span width from a packed slab."""
        start, stop = self._bounds(span_width)
        self._data[start:stop] = values

    def read(self, i: int, j: int) -> Tensor:
        """Return a copy of the value at one cell.

        Parameters
        ----------
        i : int
            The start of the span, inclusive.
        j : int
            The end of the span, exclusive.

        Returns
        -------
        Tensor
            A rank-2 tensor of shape (batch, width) in the store's floating point dtype,
            holding the cell's value.

        Raises
        ------
        IndexError
            If the span is not one of the input's.
        """
        return self._data[self._cell(i, j)].clone()

    def read_children(self, i: int, j: int) -> tuple[Tensor, Tensor]:
        """Return the values of a cell's children over all of its splits.

        Parameters
        ----------
        i : int
            The start of the span, inclusive.
        j : int
            The end of the span, exclusive.

        Returns
        -------
        tuple[Tensor, Tensor]
            Two rank-3 tensors of shape (max_splits, batch, width) in the store's
            floating point dtype: the left children's values in split order, and the
            right children's.

        Raises
        ------
        IndexError
            If the span is not one of the input's.
        """
        cell = self._cell(i, j)
        left, right = self._children()

        return (
            self._data.index_select(0, left[cell]),
            self._data.index_select(0, right[cell]),
        )

    def write(self, i: int, j: int, value: Tensor, /) -> None:
        """Write the value of one cell in place.

        Parameters
        ----------
        i : int
            The start of the span, inclusive.
        j : int
            The end of the span, exclusive.
        value : Tensor
            A rank-2 tensor of shape (batch, width).

        Raises
        ------
        IndexError
            If the span is not one of the input's.
        ValueError
            If the value does not have the shape of a cell.
        """
        cell = self._cell(i, j)
        wanted = (self._batch, self._width)

        if tuple(value.shape) != wanted:
            msg = f"a cell holds a tensor of shape {wanted}, not {tuple(value.shape)}"
            raise ValueError(msg)

        self._data[cell] = value
