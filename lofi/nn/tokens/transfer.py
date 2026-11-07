"""Token-embedding initialization from encoded type vectors."""

from enum import Enum
from typing import TYPE_CHECKING, Final, override

import torch

from lofi._checks import ensure_name, ensure_non_negative

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from torch import Tensor

    from lofi.nn.tokens.embedder import TokenVocabulary

__all__ = ["AmbiguityPolicy", "TransferReport", "initialize_from_types"]

_TABLE_RANK: Final[int] = 2
"""The embedding-table rank."""

_VECTOR_RANK: Final[int] = 1
"""The encoded-type-vector rank."""


class AmbiguityPolicy(Enum):
    """Resolves multiple type vectors assigned to one token key."""

    #: The row takes the average of the key's vectors.
    MEAN = "mean"

    #: The row takes the first vector in input order.
    FIRST = "first"

    #: The row is left as it was, and the key is counted as skipped.
    SKIP = "skip"


class TransferReport:
    """Counts offered, written, skipped, and missing token rows.

    Parameters
    ----------
    rows_considered : int
        How many keys the transfer was offered.
    rows_initialized : int
        How many of them reached a row and were written into it.
    rows_skipped : int
        How many of them had a row and were left alone by the policy.
    rows_unmatched : int
        How many of them the vocabulary holds no row of its own for.

    Raises
    ------
    TypeError
        If a count is not an integer.
    ValueError
        If a count is negative or the written, skipped, and unmatched keys do not
        partition the considered keys.
    """

    __slots__ = (
        "_rows_considered",
        "_rows_initialized",
        "_rows_skipped",
        "_rows_unmatched",
    )

    def __init__(
        self,
        *,
        rows_considered: int,
        rows_initialized: int,
        rows_skipped: int,
        rows_unmatched: int,
    ) -> None:
        considered = ensure_non_negative("the rows considered", rows_considered)
        initialized = ensure_non_negative("the rows initialized", rows_initialized)
        skipped = ensure_non_negative("the rows skipped", rows_skipped)
        unmatched = ensure_non_negative("the rows unmatched", rows_unmatched)

        if considered != initialized + skipped + unmatched:
            msg = (
                f"{considered} rows were considered and {initialized} written "
                f"beside {skipped} skipped and {unmatched} unmatched, which do "
                "not account for them"
            )
            raise ValueError(msg)

        self._rows_considered = considered
        self._rows_initialized = initialized
        self._rows_skipped = skipped
        self._rows_unmatched = unmatched

    @property
    def rows_considered(self) -> int:
        """Return the number of offered token keys."""
        return self._rows_considered

    @property
    def rows_initialized(self) -> int:
        """Return the number of initialized embedding rows."""
        return self._rows_initialized

    @property
    def rows_skipped(self) -> int:
        """Return the number of rows skipped by the ambiguity policy."""
        return self._rows_skipped

    @property
    def rows_unmatched(self) -> int:
        """Return the number of keys absent from the vocabulary."""
        return self._rows_unmatched

    def _identity(self) -> tuple[int, int, int, int]:
        """Return the four counts a report is equal on."""
        return (
            self._rows_considered,
            self._rows_initialized,
            self._rows_skipped,
            self._rows_unmatched,
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another report holds the same four counts.

        Parameters
        ----------
        other : object
            The value to compare against.
        """
        if type(other) is not TransferReport:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the four counts."""
        return hash((TransferReport, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return an evaluable representation."""
        return (
            f"TransferReport(rows_considered={self._rows_considered}, "
            f"rows_initialized={self._rows_initialized}, "
            f"rows_skipped={self._rows_skipped}, "
            f"rows_unmatched={self._rows_unmatched})"
        )


def _checked_table(table: Tensor) -> Tensor:
    """Return an embedding table.

    Parameters
    ----------
    table : Tensor
        The rows to write into.

    Returns
    -------
    Tensor
        The table, unchanged.

    Raises
    ------
    TypeError
        If the table is not of a floating point dtype.
    ValueError
        If the table is not rank 2 or holds no columns.
    """
    if not table.is_floating_point():
        msg = f"an embedding table holds real numbers, not {table.dtype}"
        raise TypeError(msg)

    if table.ndim != _TABLE_RANK:
        msg = (
            f"an embedding table is one row per index and one column per "
            f"feature, so it is rank {_TABLE_RANK}, not {table.ndim}"
        )
        raise ValueError(msg)

    if table.shape[1] == 0:
        msg = "an embedding table holds at least one column"
        raise ValueError(msg)

    return table


def _checked_vectors(
    vectors: Mapping[str, Sequence[Tensor]], width: int
) -> tuple[tuple[str, Sequence[Tensor]], ...]:
    """Return the key and vectors of each entry.

    Parameters
    ----------
    vectors : Mapping[str, Sequence[Tensor]]
        The encoded type vectors offered for each key, in the order they are to be
        resolved.
    width : int
        How wide the table's rows are.

    Returns
    -------
    tuple[tuple[str, Sequence[Tensor]], ...]
        One pair per key, in the order the mapping gave them.

    Raises
    ------
    TypeError
        If a key is not a string, or a vector is not of a floating point dtype.
    ValueError
        If a key is empty, if a key was offered no vector, if a vector is not rank 1, or
        if a vector's width is not the table's.
    """
    entries = tuple(
        (ensure_name("a transfer key", key), offered)
        for key, offered in vectors.items()
    )

    for key, offered in entries:
        if not offered:
            msg = f"{key!r} was offered no vector to initialize its row from"
            raise ValueError(msg)

        for position, vector in enumerate(offered):
            if not vector.is_floating_point():
                msg = (
                    f"vector {position} of {key!r} holds real numbers, not "
                    f"{vector.dtype}"
                )
                raise TypeError(msg)

            if vector.ndim != _VECTOR_RANK:
                msg = (
                    f"vector {position} of {key!r} is one number per feature, "
                    f"so it is rank {_VECTOR_RANK}, not {vector.ndim}"
                )
                raise ValueError(msg)

            if vector.shape[0] != width:
                msg = (
                    f"vector {position} of {key!r} is {vector.shape[0]} wide "
                    f"where the table's rows are {width}"
                )
                raise ValueError(msg)

    return entries


def _resolved(offered: Sequence[Tensor], policy: AmbiguityPolicy) -> Tensor | None:
    """Return the vector a key's row takes, or None where the policy declines.

    Parameters
    ----------
    offered : Sequence[Tensor]
        The key's vectors, in the order they came in.
    policy : AmbiguityPolicy
        What to do where there is more than one.

    Returns
    -------
    Tensor | None
        Rank 1 over the feature axis, at the vectors' own precision, or None where an
        ambiguous key is left alone.
    """
    if len(offered) == 1:
        return offered[0]

    if policy is AmbiguityPolicy.SKIP:
        return None

    if policy is AmbiguityPolicy.FIRST:
        return offered[0]

    return torch.stack(tuple(offered)).mean(dim=0)


def initialize_from_types(
    table: Tensor,
    vectors: Mapping[str, Sequence[Tensor]],
    *,
    vocabulary: TokenVocabulary,
    policy: AmbiguityPolicy,
) -> tuple[Tensor, TransferReport]:
    """Return the initialized embedding table and transfer report.

    Parameters
    ----------
    table : Tensor
        The rows to write into, rank 2 over the axes ``("index", "width")`` at floating
        precision, with one row per index the vocabulary describes.
    vectors : Mapping[str, Sequence[Tensor]]
        The encoded type vectors offered for each key, each rank 1 over the feature axis
        and as wide as the table's rows.
    vocabulary : TokenVocabulary
        What the table's rows are indexed by.
    policy : AmbiguityPolicy
        What to do with a key several vectors were offered for.

    Returns
    -------
    tuple[Tensor, TransferReport]
        A copy of the table with the written rows replaced, rank 2 over the axes
        ``("index", "width")`` at the table's own precision, and the four counts of what
        the transfer did.

    Raises
    ------
    TypeError
        If the policy is not an ``AmbiguityPolicy``, if the table or a vector is not of
        a floating point dtype, or if a key is not a string.
    ValueError
        If table shape, vocabulary size, keys, or source vector shapes are invalid.
    """
    if not isinstance(policy, AmbiguityPolicy):
        msg = f"a transfer runs under an AmbiguityPolicy, not {policy!r}"
        raise TypeError(msg)

    checked = _checked_table(table)

    if checked.shape[0] != vocabulary.size:
        msg = (
            f"a vocabulary of {vocabulary.size} rows was given a table of "
            f"{checked.shape[0]}"
        )
        raise ValueError(msg)

    entries = _checked_vectors(vectors, int(checked.shape[1]))
    written = checked.clone()
    initialized = 0
    skipped = 0
    unmatched = 0

    for key, offered in entries:
        if not vocabulary.holds(key):
            unmatched += 1
            continue

        resolved = _resolved(offered, policy)

        if resolved is None:
            skipped += 1
            continue

        written[vocabulary.index_of(key)] = resolved
        initialized += 1

    return written, TransferReport(
        rows_considered=len(entries),
        rows_initialized=initialized,
        rows_skipped=skipped,
        rows_unmatched=unmatched,
    )
