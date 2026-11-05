"""Span accessors deciding which token spans a constraint covers.

An accessor reads the surface tokens of one input and returns the half-open
spans the constraint applies to, so a constraint declares where it holds
without knowing the length of the input it will meet. A span ``(0, 3)`` covers
the positions 0, 1 and 2.
"""

from typing import TYPE_CHECKING, final, override

from lofi._checks import ensure_non_negative
from lofi._frozen import FrozenValue

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["FixedSpan", "FormPattern", "TokenPosition", "WholeInput"]

_NOUN = "span accessor"


def _refuse_overrun(end: int, tokens: Sequence[str], covered: str) -> None:
    """Refuse a span whose end lies past the input it is read against.

    Parameters
    ----------
    end : int
        The exclusive end of the covered span.
    tokens : Sequence[str]
        The surface tokens of one input, in order.
    covered : str
        How the accessor names what it covers, for the refusal message.

    Raises
    ------
    ValueError
        If the end lies past the last token.
    """
    if end > len(tokens):
        msg = f"{covered} reaches past the {len(tokens)}-token input"
        raise ValueError(msg)


@final
class TokenPosition(FrozenValue):
    """Covers the single token at a declared position.

    Parameters
    ----------
    position : int
        The zero-based position of the covered token.

    Raises
    ------
    TypeError
        If the position is not an integer.
    ValueError
        If the position is negative.
    """

    _noun = _NOUN

    __slots__ = ("_position",)

    def __init__(self, position: int) -> None:
        object.__setattr__(
            self, "_position", ensure_non_negative("a token position", position)
        )

    @property
    def position(self) -> int:
        """Return the position of the covered token."""
        return self._position

    def spans(self, tokens: Sequence[str], /) -> tuple[tuple[int, int], ...]:
        """Return the one-token span at the declared position.

        Parameters
        ----------
        tokens : Sequence[str]
            The surface tokens of one input, in order.

        Returns
        -------
        tuple[tuple[int, int], ...]
            The single half-open span covering the declared position.

        Raises
        ------
        ValueError
            If the input holds no token at that position.
        """
        _refuse_overrun(
            self._position + 1, tokens, f"the token at position {self._position}"
        )

        return ((self._position, self._position + 1),)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another accessor covers the same position.

        Parameters
        ----------
        other : object
            The value to compare.

        Returns
        -------
        bool
            Whether ``other`` is an accessor over the same position.
        """
        if type(other) is not TokenPosition:
            return NotImplemented

        return self._position == other._position

    @override
    def __hash__(self) -> int:
        """Return a hash over the covered position."""
        return hash((TokenPosition, self._position))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this accessor."""
        return f"TokenPosition({self._position!r})"


@final
class FixedSpan(FrozenValue):
    """Covers one declared half-open token span.

    Parameters
    ----------
    start : int
        The first covered position.
    end : int
        The position one past the last covered one.

    Raises
    ------
    TypeError
        If an endpoint is not an integer.
    ValueError
        If an endpoint is negative or the span covers no position.
    """

    _noun = _NOUN

    __slots__ = ("_end", "_start")

    def __init__(self, start: int, end: int) -> None:
        held_start = ensure_non_negative("a fixed span's start", start)
        held_end = ensure_non_negative("a fixed span's end", end)

        if held_end <= held_start:
            msg = f"a fixed span covers no position over [{start}, {end})"
            raise ValueError(msg)

        object.__setattr__(self, "_start", held_start)
        object.__setattr__(self, "_end", held_end)

    @property
    def start(self) -> int:
        """Return the first covered position."""
        return self._start

    @property
    def end(self) -> int:
        """Return the position one past the last covered one."""
        return self._end

    def spans(self, tokens: Sequence[str], /) -> tuple[tuple[int, int], ...]:
        """Return the declared span.

        Parameters
        ----------
        tokens : Sequence[str]
            The surface tokens of one input, in order.

        Returns
        -------
        tuple[tuple[int, int], ...]
            The single half-open span the accessor was declared over.

        Raises
        ------
        ValueError
            If the span reaches past the input.
        """
        _refuse_overrun(self._end, tokens, f"the span [{self._start}, {self._end})")

        return ((self._start, self._end),)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another accessor covers the same span.

        Parameters
        ----------
        other : object
            The value to compare.

        Returns
        -------
        bool
            Whether ``other`` is an accessor over the same endpoints.
        """
        if type(other) is not FixedSpan:
            return NotImplemented

        return self._start == other._start and self._end == other._end

    @override
    def __hash__(self) -> int:
        """Return a hash over the endpoints."""
        return hash((FixedSpan, self._start, self._end))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this accessor."""
        return f"FixedSpan({self._start!r}, {self._end!r})"


@final
class WholeInput(FrozenValue):
    """Covers every token of whatever input it is read against."""

    _noun = _NOUN

    __slots__ = ()

    def spans(self, tokens: Sequence[str], /) -> tuple[tuple[int, int], ...]:
        """Return the span covering the whole input.

        Parameters
        ----------
        tokens : Sequence[str]
            The surface tokens of one input, in order.

        Returns
        -------
        tuple[tuple[int, int], ...]
            The single half-open span from the first position to the last.

        Raises
        ------
        ValueError
            If the input holds no token.
        """
        if not tokens:
            msg = "the whole of an empty input covers no position"
            raise ValueError(msg)

        return ((0, len(tokens)),)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value also covers the whole input.

        Parameters
        ----------
        other : object
            The value to compare.

        Returns
        -------
        bool
            Whether ``other`` is an accessor over the whole input.
        """
        if type(other) is not WholeInput:
            return NotImplemented

        return True

    @override
    def __hash__(self) -> int:
        """Return the hash shared by accessors over the whole input."""
        return hash(WholeInput)

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this accessor."""
        return "WholeInput()"


@final
class FormPattern(FrozenValue):
    """Covers every span whose tokens match a declared word pattern.

    A pattern member is a surface word the token must equal, or ``None``, which
    any one token matches. The matches are returned in token order and may
    overlap, so a pattern can cover several spans of one input at once.

    Parameters
    ----------
    pattern : Sequence[str | None]
        One matcher per covered position.

    Raises
    ------
    TypeError
        If the pattern is given as a string or a member is neither text nor
        ``None``.
    ValueError
        If the pattern is empty or a member is empty or blank.
    """

    _noun = _NOUN

    __slots__ = ("_pattern",)

    def __init__(self, pattern: Sequence[str | None]) -> None:
        if isinstance(pattern, str | bytes):
            msg = (
                "a form pattern is a sequence of matchers rather than one of "
                f"them: {pattern!r}"
            )

            raise TypeError(msg)

        held = tuple(
            None if member is None else _matcher(at, member)
            for at, member in enumerate(pattern)
        )

        if not held:
            msg = "a form pattern with no matcher covers no span"
            raise ValueError(msg)

        object.__setattr__(self, "_pattern", held)

    @property
    def pattern(self) -> tuple[str | None, ...]:
        """Return the matchers, in the order they cover positions."""
        return self._pattern

    def spans(self, tokens: Sequence[str], /) -> tuple[tuple[int, int], ...]:
        """Return every span of the input whose tokens match the pattern.

        Parameters
        ----------
        tokens : Sequence[str]
            The surface tokens of one input, in order.

        Returns
        -------
        tuple[tuple[int, int], ...]
            The matching half-open spans in ascending order of their starts,
            empty where the pattern matches nowhere.
        """
        width = len(self._pattern)

        return tuple(
            (start, start + width)
            for start in range(len(tokens) - width + 1)
            if all(
                matcher is None or matcher == tokens[start + at]
                for at, matcher in enumerate(self._pattern)
            )
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another accessor matches on the same pattern.

        Parameters
        ----------
        other : object
            The value to compare.

        Returns
        -------
        bool
            Whether ``other`` is an accessor over the same matchers.
        """
        if type(other) is not FormPattern:
            return NotImplemented

        return self._pattern == other._pattern

    @override
    def __hash__(self) -> int:
        """Return a hash over the matchers."""
        return hash((FormPattern, self._pattern))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this accessor."""
        return f"FormPattern({list(self._pattern)!r})"


def _matcher(at: int, member: str) -> str:
    """Return a validated surface matcher of a form pattern.

    Parameters
    ----------
    at : int
        The position the matcher covers, for the refusal message.
    member : str
        The surface word the token at that position must equal.

    Returns
    -------
    str
        The matcher itself.

    Raises
    ------
    TypeError
        If the matcher is not text.
    ValueError
        If the matcher is empty or blank.
    """
    if not isinstance(member, str):
        msg = f"a form pattern's matcher at {at} is not a word: {member!r}"
        raise TypeError(msg)

    if not member.strip():
        msg = f"a form pattern's matcher at {at} is blank: {member!r}"
        raise ValueError(msg)

    return member
