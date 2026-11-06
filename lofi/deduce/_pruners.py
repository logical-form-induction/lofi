"""Per-cell chart pruners."""

from typing import TYPE_CHECKING, override

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi.deduce._spans import Cell
    from lofi.deduce._values import ItemKey

__all__ = ["TopKPruner"]


class TopKPruner[T]:
    """Keeps the heaviest items of a cell, which is the cell beam.

    Weights are log weights, ordered as floats, so the pruner ranks under the
    order of every algebra whose weight is one: greater is better under both
    Viterbi and the inside algebra. An algebra weighted otherwise brings its
    own pruner.

    Parameters
    ----------
    k : int
        The beam, which is how many items a cell may keep.

    Raises
    ------
    ValueError
        If the beam is smaller than one.
    """

    __slots__ = ("_k",)

    def __init__(self, k: int) -> None:
        if k < 1:
            msg = f"a cell beam keeps at least one item, not {k}"
            raise ValueError(msg)

        self._k = k

    @property
    def k(self) -> int:
        """Return the beam, which is how many items a cell may keep."""
        return self._k

    def keep(
        self, span: Cell, items: Sequence[tuple[ItemKey[T], float]], /
    ) -> tuple[ItemKey[T], ...]:
        """Return the beam's worth of heaviest items, ties going to the earlier.

        Parameters
        ----------
        span : Cell
            The half-open token interval the cell covers.
        items : Sequence[tuple[ItemKey[T], float]]
            Every item of the cell with its inside weight, in arrival order.

        Returns
        -------
        tuple[ItemKey[T], ...]
            The kept items, in arrival order.
        """
        del span

        if len(items) <= self._k:
            return tuple(key for key, _ in items)

        ranked = sorted(range(len(items)), key=lambda at: (-items[at][1], at))

        return tuple(items[at][0] for at in sorted(ranked[: self._k]))

    @override
    def __repr__(self) -> str:
        """Return a rendering naming the beam."""
        return f"TopKPruner({self._k})"
