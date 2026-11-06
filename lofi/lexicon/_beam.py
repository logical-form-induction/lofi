"""Fixed and growing per-position beam sources."""

from typing import TYPE_CHECKING, override

from lofi._checks import ensure_finite, ensure_name, ensure_non_negative
from lofi.lexicon._errors import ExhaustedBeamError
from lofi.lexicon._ports import Entry

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from lofi.lexicon._ports import Clamp

__all__ = ["BeamSource", "GrowingBeamSource"]


def _ranked[T](beams: Iterable[Iterable[Entry[T]]]) -> tuple[tuple[Entry[T], ...], ...]:
    """Return beams in rank order with interned terms.

    Parameters
    ----------
    beams : Iterable[Iterable[Entry[T]]]
        Candidate entries by token position.

    Returns
    -------
    tuple[tuple[Entry[T], ...], ...]
        Candidates sorted by descending weight at each position.

    Raises
    ------
    TypeError
        If an entry form or weight has the wrong type.
    ValueError
        If a form is empty, a weight is not finite, or one beam mixes forms or
        repeats an entry.
    """
    pool: dict[T, T] = {}
    ranked: list[tuple[Entry[T], ...]] = []

    for position, beam in enumerate(beams):
        held: dict[Entry[T], Entry[T]] = {}
        form: str | None = None

        for entry in beam:
            here = ensure_name("a beam entry's form", entry.form)
            weight = ensure_finite("a beam entry's weight", entry.weight)

            if form is None:
                form = here

            elif here != form:
                msg = (
                    f"the beam at position {position} is over {form!r} and "
                    f"holds an entry for {here!r}"
                )
                raise ValueError(msg)

            meaning = entry.meaning
            candidate = Entry(
                here,
                pool.setdefault(entry.category, entry.category),
                None if meaning is None else pool.setdefault(meaning, meaning),
                weight,
            )

            if candidate in held:
                msg = f"the beam at position {position} holds {candidate!r} twice"
                raise ValueError(msg)

            held[candidate] = candidate

        ranked.append(tuple(sorted(held, key=lambda entry: -entry.weight)))

    return tuple(ranked)


def _checked_position(position: int, positions: int) -> int:
    """Return a token position covered by the beams.

    Parameters
    ----------
    position : int
        Candidate position.
    positions : int
        Number of available positions.

    Returns
    -------
    int
        Validated position.

    Raises
    ------
    IndexError
        If ``position`` is outside the available range.
    """
    if not 0 <= position < positions:
        msg = f"there is no beam at position {position} of {positions}"
        raise IndexError(msg)

    return position


def _admitted[T](
    beams: tuple[tuple[Entry[T], ...], ...],
    frontier: tuple[int, ...],
    tokens: Sequence[str],
) -> tuple[tuple[int, Entry[T], float], ...]:
    """Return live candidates whose forms match the tokens.

    Parameters
    ----------
    beams : tuple[tuple[Entry[T], ...], ...]
        Ranked candidates by position.
    frontier : tuple[int, ...]
        Active candidate count at each position.
    tokens : Sequence[str]
        Surface tokens in position order.

    Returns
    -------
    tuple[tuple[int, Entry[T], float], ...]
        Position, entry, and weight for each admitted candidate.
    """
    return tuple(
        (position, entry, entry.weight)
        for position in range(min(len(beams), len(tokens)))
        for entry in beams[position][: frontier[position]]
        if entry.form == tokens[position]
    )


class BeamSource[T]:
    """All candidates in a per-position beam.

    Parameters
    ----------
    beams : Iterable[Iterable[Entry[T]]]
        Candidate entries by token position.

    Raises
    ------
    TypeError
        If an entry form or weight has the wrong type.
    ValueError
        If an entry is invalid, a beam mixes forms, or a beam repeats an entry.
    """

    __slots__ = ("_beams", "_widths")

    def __init__(self, beams: Iterable[Iterable[Entry[T]]]) -> None:
        self._beams = _ranked(beams)
        self._widths = tuple(len(beam) for beam in self._beams)

    @property
    def is_re_callable(self) -> bool:
        """Return whether the enumeration may be read more than once, which it may."""
        return True

    @property
    def clamps(self) -> tuple[Clamp[T], ...]:
        """Return no clamps."""
        return ()

    @property
    def positions(self) -> int:
        """Return how many token positions the beams cover."""
        return len(self._beams)

    @property
    def widths(self) -> tuple[int, ...]:
        """Return how many candidates each position offers, in position order."""
        return self._widths

    @property
    def beams(self) -> tuple[tuple[Entry[T], ...], ...]:
        """Return the candidates at each position, in position order and best first."""
        return self._beams

    def candidates(self, position: int) -> tuple[Entry[T], ...]:
        """Return the candidates at one position, best first.

        Parameters
        ----------
        position : int
            Zero-based token position.

        Returns
        -------
        tuple[Entry[T], ...]
            Candidates in descending weight order.

        Raises
        ------
        IndexError
            If no beam exists at ``position``.
        """
        return self._beams[_checked_position(position, len(self._beams))]

    def axioms(
        self, tokens: Sequence[str], /
    ) -> tuple[tuple[int, Entry[T], float], ...]:
        """Return every candidate admitted by the tokens.

        Parameters
        ----------
        tokens : Sequence[str]
            Surface tokens in position order.

        Returns
        -------
        tuple[tuple[int, Entry[T], float], ...]
            Position, entry, and weight for each candidate.
        """
        return _admitted(self._beams, self._widths, tokens)

    def __len__(self) -> int:
        """Return how many candidates the beams hold between them."""
        return sum(self._widths)

    @override
    def __repr__(self) -> str:
        """Return a rendering naming the position count and the candidate count."""
        return f"BeamSource(positions={len(self._beams)}, candidates={len(self)})"


class GrowingBeamSource[T]:
    """A widenable prefix of each beam.

    Parameters
    ----------
    source : BeamSource[T]
        Complete ranked beams.
    frontier : tuple[int, ...] | None
        Active candidate count at each position, or ``None`` to seed each
        nonempty position with its best candidate.

    Raises
    ------
    TypeError
        If a frontier entry is not an integer or is a boolean.
    ValueError
        If the frontier length or an entry is invalid.
    """

    __slots__ = ("_frontier", "_source", "_widenings")

    def __init__(
        self,
        source: BeamSource[T],
        /,
        *,
        frontier: tuple[int, ...] | None = None,
    ) -> None:
        widths = source.widths
        live = (
            tuple(min(1, width) for width in widths)
            if frontier is None
            else tuple(frontier)
        )

        if len(live) != len(widths):
            msg = (
                f"a frontier holds one entry per position, so {len(widths)} of "
                f"them, not {len(live)}"
            )
            raise ValueError(msg)

        for position, (standing, width) in enumerate(zip(live, widths, strict=True)):
            ensure_non_negative("a frontier entry", standing)

            if standing > width:
                msg = (
                    f"position {position} offers {width} candidates and "
                    f"{standing} were said to stand live"
                )
                raise ValueError(msg)

            if standing == 0 and width:
                msg = (
                    f"position {position} offers {width} candidates and admits "
                    "none, where a seeded position admits its best"
                )
                raise ValueError(msg)

        self._source = source
        self._frontier = live
        self._widenings = sum(standing - 1 for standing in live if standing)

    @property
    def is_re_callable(self) -> bool:
        """Return whether the enumeration may be read more than once."""
        return self._source.is_re_callable

    @property
    def clamps(self) -> tuple[Clamp[T], ...]:
        """Return the clamps this source enforces, which are the beams' own."""
        return self._source.clamps

    @property
    def positions(self) -> int:
        """Return how many token positions the active frontier covers."""
        return len(self._frontier)

    @property
    def frontier(self) -> tuple[int, ...]:
        """Return the active candidate count at each position."""
        return self._frontier

    @property
    def widenings(self) -> int:
        """Return the number of widenings from the seeded source."""
        return self._widenings

    def candidates(self, position: int) -> tuple[Entry[T], ...]:
        """Return live candidates at one position, best first.

        Parameters
        ----------
        position : int
            Zero-based token position.

        Returns
        -------
        tuple[Entry[T], ...]
            Active candidates in descending weight order.

        Raises
        ------
        IndexError
            If no beam exists at ``position``.
        """
        live = self._frontier[_checked_position(position, len(self._frontier))]

        return self._source.beams[position][:live]

    def next_best(self, position: int) -> Entry[T] | None:
        """Return the candidate admitted by the next widening.

        Parameters
        ----------
        position : int
            Zero-based token position.

        Returns
        -------
        Entry[T] | None
            Next inactive candidate, or ``None`` when the beam is exhausted.

        Raises
        ------
        IndexError
            If no beam exists at ``position``.
        """
        live = self._frontier[_checked_position(position, len(self._frontier))]
        beam = self._source.beams[position]

        return beam[live] if live < len(beam) else None

    def widen(self, position: int) -> GrowingBeamSource[T]:
        """Return a source admitting the next candidate at one position.

        Parameters
        ----------
        position : int
            Zero-based token position.

        Returns
        -------
        GrowingBeamSource[T]
            New source with one additional active candidate.

        Raises
        ------
        IndexError
            If no beam exists at ``position``.
        ExhaustedBeamError
            If every candidate at ``position`` is already active.
        """
        checked = _checked_position(position, len(self._frontier))
        live = self._frontier[checked]
        width = self._source.widths[checked]

        if live >= width:
            raise ExhaustedBeamError(position=checked, width=width)

        advanced = (
            *self._frontier[:checked],
            live + 1,
            *self._frontier[checked + 1 :],
        )

        return GrowingBeamSource(self._source, frontier=advanced)

    def axioms(
        self, tokens: Sequence[str], /
    ) -> tuple[tuple[int, Entry[T], float], ...]:
        """Return live candidates admitted by the tokens.

        Parameters
        ----------
        tokens : Sequence[str]
            Surface tokens in position order.

        Returns
        -------
        tuple[tuple[int, Entry[T], float], ...]
            Position, entry, and weight for each live candidate.
        """
        return _admitted(self._source.beams, self._frontier, tokens)

    @override
    def __repr__(self) -> str:
        """Return the beams underneath and how many widenings reached this source."""
        return f"GrowingBeamSource({self._source!r}, widenings={self._widenings})"
