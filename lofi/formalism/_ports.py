"""Protocols for indexing input regions."""

from typing import TYPE_CHECKING, Protocol, override

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["Region", "RegionIndex"]


class Region(Protocol):
    """One item's coverage of the input."""

    @override
    def __hash__(self) -> int:
        """Return a hash over the positions this region covers."""


class RegionIndex[R: Region](Protocol):
    """Indexes input regions with arbitrary fan-out."""

    @property
    def fanout(self) -> int:
        """Return how many contiguous regions of the input one region covers."""

    def leaves(self) -> Iterator[R]:
        """Enumerate the regions a lexical axiom may cover.

        Returns
        -------
        Iterator[R]
            Regions that each cover one token.
        """

    def regions_by_size(self) -> Iterator[R]:
        """Enumerate every region, smallest first.

        Returns
        -------
        Iterator[R]
            Regions ordered after all regions they decompose into.
        """

    def decompositions(self, region: R, arity: int, /) -> Iterator[tuple[R, ...]]:
        """Enumerate ways to divide a region into a fixed number of parts.

        Parameters
        ----------
        region : R
            Region to divide.
        arity : int
            Number of parts in each decomposition.

        Returns
        -------
        Iterator[tuple[R, ...]]
            Region parts in rule-premise order.
        """
