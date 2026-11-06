"""Dense indexing for half-open spans."""

from bisect import bisect_right
from typing import TYPE_CHECKING, override

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["Cell", "SpanIndex"]

type Cell = tuple[int, int]
"""Half-open chart span stored as start and end."""


def _cuts(i: int, j: int, parts: int) -> Iterator[tuple[Cell, ...]]:
    """Yield the ways of cutting one span into a fixed number of nonempty spans.

    Parameters
    ----------
    i : int
        The start of the span, inclusive.
    j : int
        The end of the span, exclusive.
    parts : int
        Positive number of subspans.

    Yields
    ------
    tuple of tuple of int
        Contiguous sub-spans in left-to-right order, covering the whole span.
    """
    if parts == 1:
        yield ((i, j),)
        return

    tail = parts - 1

    # specialize binary splits because this is the chart's inner loop.
    if tail == 1:
        for k in range(i + 1, j):
            yield ((i, k), (k, j))

        return

    # reserve one token for each remaining part.
    for k in range(i + 1, j - parts + 2):
        head = (i, k)

        for rest in _cuts(k, j, tail):
            yield (head, *rest)


class SpanIndex:
    """A dense index over half-open spans.

    Parameters
    ----------
    n : int
        Nonnegative input length.

    Raises
    ------
    ValueError
        If n is negative.
    """

    __slots__ = (
        "_left_child_ids",
        "_n",
        "_offsets",
        "_right_child_ids",
        "_split_mask",
    )

    def __init__(self, n: int) -> None:
        if n < 0:
            msg = f"an input length is non-negative, not {n}"
            raise ValueError(msg)

        offsets = [0] * (n + 2)
        total = 0

        for width in range(1, n + 1):
            offsets[width] = total
            total += n - width + 1

        offsets[n + 1] = total

        self._n = n
        self._offsets: tuple[int, ...] = tuple(offsets)
        self._left_child_ids: tuple[tuple[int, ...], ...] | None = None
        self._right_child_ids: tuple[tuple[int, ...], ...] | None = None
        self._split_mask: tuple[tuple[bool, ...], ...] | None = None

    @property
    def n(self) -> int:
        """Return the input length this index spans."""
        return self._n

    @property
    def num_spans(self) -> int:
        """Return how many spans there are, which is the size of the id range."""
        return self._offsets[-1]

    @property
    def max_splits(self) -> int:
        """Return the widest span's split count, which is the gather row width."""
        return max(self._n - 1, 0)

    @property
    def fanout(self) -> int:
        """Return how many contiguous regions of the input one span covers."""
        return 1

    def span_id(self, i: int, j: int) -> int:
        """Return the dense id of a span.

        Parameters
        ----------
        i : int
            The start of the span, inclusive.
        j : int
            The end of the span, exclusive.

        Returns
        -------
        int
            The id, in ``range(num_spans)``.
        """
        return self._offsets[j - i] + i

    def span_of(self, sid: int) -> Cell:
        """Return the span for a dense identifier.

        Parameters
        ----------
        sid : int
            A span id.

        Returns
        -------
        Cell
            The start, inclusive, and the end, exclusive.

        Raises
        ------
        IndexError
            If the id is outside the span range.
        """
        if not 0 <= sid < self.num_spans:
            msg = f"span id {sid} is outside range({self.num_spans})"
            raise IndexError(msg)

        width = bisect_right(self._offsets, sid, 1, self._n + 1) - 1
        i = sid - self._offsets[width]

        return (i, i + width)

    def splits(self, i: int, j: int) -> range:
        """Return the split points of a span.

        Parameters
        ----------
        i : int
            The start of the span, inclusive.
        j : int
            The end of the span, exclusive.

        Returns
        -------
        range
            The split points, ascending.
        """
        return range(i + 1, j)

    def split_id(self, i: int, j: int, k: int) -> tuple[int, int]:
        """Return the address of one split of one span.

        Parameters
        ----------
        i : int
            The start of the span, inclusive.
        j : int
            The end of the span, exclusive.
        k : int
            The split point.

        Returns
        -------
        tuple[int, int]
            The span's id, and the split's offset within that span's row.
        """
        return (self.span_id(i, j), k - i - 1)

    def spans_by_length(self, *, min_length: int = 1) -> Iterator[Cell]:
        """Yield every span in order of increasing width, then increasing start.

        Parameters
        ----------
        min_length : int
            The narrowest width to yield.

        Yields
        ------
        tuple of int
            The start, inclusive, and the end, exclusive.
        """
        for width in range(max(min_length, 1), self._n + 1):
            for i in range(self._n - width + 1):
                yield (i, i + width)

    def spans_by_length_desc(self, *, min_length: int = 1) -> Iterator[Cell]:
        """Yield every span in order of decreasing width, then increasing start.

        Parameters
        ----------
        min_length : int
            The narrowest width to yield.

        Yields
        ------
        tuple of int
            The start, inclusive, and the end, exclusive.
        """
        for width in range(self._n, max(min_length, 1) - 1, -1):
            for i in range(self._n - width + 1):
                yield (i, i + width)

    def leaves(self) -> Iterator[Cell]:
        """Yield every span of width one, left to right.

        Yields
        ------
        tuple of int
            The start, inclusive, and the end, exclusive.
        """
        for i in range(self._n):
            yield (i, i + 1)

    def regions_by_size(self) -> Iterator[Cell]:
        """Return every span in order of increasing width, then increasing start.

        Returns
        -------
        Iterator[Cell]
            Every span, each as a start, inclusive, and an end, exclusive.
        """
        return self.spans_by_length()

    def decompositions(self, region: Cell, arity: int, /) -> Iterator[tuple[Cell, ...]]:
        """Yield the ways one span decomposes into a fixed number of sub-spans.

        Parameters
        ----------
        region : Cell
            The span to decompose.
        arity : int
            How many sub-spans to decompose into.

        Yields
        ------
        tuple of tuple of int
            Contiguous sub-spans in left-to-right order, covering the region.

        Raises
        ------
        ValueError
            If the arity is negative.
        """
        if arity < 0:
            msg = f"an arity is non-negative, not {arity}"
            raise ValueError(msg)

        if arity == 0:
            return

        i, j = region
        yield from _cuts(i, j, arity)

    def binary_sweep(self) -> Iterator[tuple[Cell, Cell, Cell]]:
        """Yield every span paired with each of its two-part decompositions.

        Yields
        ------
        tuple of tuple of int
            Parent, left-child, and right-child spans for each binary split.
        """
        for cell in self.spans_by_length(min_length=2):
            for parts in self.decompositions(cell, 2):
                yield (cell, parts[0], parts[1])

    @property
    def left_child_ids(self) -> tuple[tuple[int, ...], ...]:
        """Return left-child span IDs for every split of each span."""
        rows = self._left_child_ids

        if rows is None:
            rows = tuple(
                tuple(
                    self.span_id(i, i + 1 + offset) if offset < j - i - 1 else 0
                    for offset in range(self.max_splits)
                )
                for i, j in self.spans_by_length()
            )
            self._left_child_ids = rows

        return rows

    @property
    def right_child_ids(self) -> tuple[tuple[int, ...], ...]:
        """Return right-child span IDs for every split of each span."""
        rows = self._right_child_ids

        if rows is None:
            rows = tuple(
                tuple(
                    self.span_id(i + 1 + offset, j) if offset < j - i - 1 else 0
                    for offset in range(self.max_splits)
                )
                for i, j in self.spans_by_length()
            )
            self._right_child_ids = rows

        return rows

    @property
    def split_mask(self) -> tuple[tuple[bool, ...], ...]:
        """Return valid-split masks for each span ID."""
        rows = self._split_mask

        if rows is None:
            rows = tuple(
                tuple(offset < j - i - 1 for offset in range(self.max_splits))
                for i, j in self.spans_by_length()
            )
            self._split_mask = rows

        return rows

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether two indices address the same inputs."""
        if type(other) is not SpanIndex:
            return NotImplemented

        return self._n == other._n

    @override
    def __hash__(self) -> int:
        """Return a hash consistent with equality."""
        return hash((SpanIndex, self._n))

    @override
    def __repr__(self) -> str:
        """Return a reconstructing representation."""
        return f"SpanIndex({self._n})"
