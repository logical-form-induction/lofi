"""Token-to-row alignments shared by data and neural components."""

from typing import override

__all__ = ["Alignment"]


class Alignment:
    """A token-to-row alignment for one item.

    Parameters
    ----------
    rows : tuple[tuple[int, int] | None, ...]
        The half-open row interval for each token, or ``None`` if the token has
        no row.

    Raises
    ------
    ValueError
        If an interval is empty or begins before a preceding interval ends.
    """

    __slots__ = ("_rows",)

    def __init__(self, rows: tuple[tuple[int, int] | None, ...]) -> None:
        self._rows = tuple(rows)
        previous = 0

        for token, span in enumerate(self._rows):
            if span is None:
                continue

            start, end = span
            if start < previous:
                msg = (
                    f"the rows for token {token} begin at {start}, before the "
                    f"rows of the tokens before it end at {previous}"
                )
                raise ValueError(msg)

            if end <= start:
                msg = f"token {token} is aligned to no row: [{start}, {end})"
                raise ValueError(msg)

            previous = end

    @property
    def rows(self) -> tuple[tuple[int, int] | None, ...]:
        """Return each token's row interval, or ``None`` if it is unaligned."""
        return self._rows

    @property
    def token_count(self) -> int:
        """Return the number of input tokens."""
        return len(self._rows)

    @property
    def word_count(self) -> int:
        """Return the number of input words."""
        return len(self._rows)

    @property
    def uncovered(self) -> tuple[int, ...]:
        """Return unaligned token positions in token order."""
        return tuple(token for token, span in enumerate(self._rows) if span is None)

    @property
    def dropped(self) -> tuple[int, ...]:
        """Return the positions of words with no encoding row."""
        return self.uncovered

    @property
    def is_total(self) -> bool:
        """Return whether every input token maps to at least one row."""
        return all(span is not None for span in self._rows)

    def rows_for(self, token: int) -> tuple[int, int] | None:
        """Return a token's row interval, or ``None`` if it is unaligned.

        Parameters
        ----------
        token : int
            The token's position in the item.

        Returns
        -------
        tuple[int, int] | None
            The half-open row interval, or ``None`` if the token is unaligned.

        Raises
        ------
        IndexError
            If the item has no token at that position.
        """
        if not 0 <= token < len(self._rows):
            msg = f"an item of {len(self._rows)} tokens has none at position {token}"
            raise IndexError(msg)

        return self._rows[token]

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value has the same token-to-row mapping."""
        if type(other) is not Alignment:
            return NotImplemented

        return self._rows == other._rows

    @override
    def __hash__(self) -> int:
        """Return a hash over the token-to-row mapping."""
        return hash((Alignment, self._rows))

    @override
    def __repr__(self) -> str:
        """Return the constructor call for this alignment."""
        return f"Alignment({self._rows!r})"
