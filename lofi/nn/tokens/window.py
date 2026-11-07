"""Context windows, and what an encoder does with an item that overflows one."""

from typing import TYPE_CHECKING, Final, Protocol, override

from lofi._checks import ensure_non_negative
from lofi._frozen import FrozenValue
from lofi.nn._values import TruncationReport
from lofi.registry import Registry, RegistryError

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "REFUSE",
    "REPORT",
    "ContextOverflowError",
    "ContextWindow",
    "OverflowPlugin",
    "overflow_policy",
]


class _Overflow(Protocol):
    """Resolve one item that does not fit a context window."""

    def __call__(
        self, words: tuple[str, ...], *, rows: int, needed: int, kept: int
    ) -> TruncationReport:
        """Return the truncation report or refuse the item."""
        ...


class OverflowPlugin(FrozenValue):
    """Resolve an item that exceeds a context window."""

    __slots__ = ("_resolve",)
    _noun = "overflow-policy plugin"

    def __init__(self, resolve: _Overflow, /) -> None:
        object.__setattr__(self, "_resolve", resolve)

    def resolve(
        self, words: tuple[str, ...], *, rows: int, needed: int, kept: int
    ) -> TruncationReport:
        """Return the truncation report or refuse the item."""
        return self._resolve(words, rows=rows, needed=needed, kept=kept)


def overflow_policy(name: str, /) -> OverflowPlugin:
    """Return the context-overflow policy installed under ``name``."""
    member = Registry[object]("overflow_policy").get(name)

    if not isinstance(member, OverflowPlugin):
        msg = (
            f"the overflow_policy entry {name!r} names {member!r}, "
            "not an overflow-policy plugin"
        )
        raise RegistryError(msg, family="overflow_policy")

    return member


class ContextOverflowError(ValueError):
    """Reports an item longer than the window of the encoder that refused it.

    Parameters
    ----------
    words : Sequence[str]
        The item's words.
    rows : int
        How many rows the window holds.
    needed : int
        How many rows the item would take.
    """

    __slots__ = ("_needed", "_rows", "_words")

    def __init__(self, words: Sequence[str], *, rows: int, needed: int) -> None:
        held = tuple(words)

        super().__init__(
            f"an item of {len(held)} words needs {needed} rows and the window "
            f"holds {rows}: {held}"
        )

        self._words = held
        self._rows = rows
        self._needed = needed

    @property
    def words(self) -> tuple[str, ...]:
        """Return the words of the item that was refused."""
        return self._words

    @property
    def rows(self) -> int:
        """Return how many rows the window holds."""
        return self._rows

    @property
    def needed(self) -> int:
        """Return how many rows the item would take."""
        return self._needed


def _refuse(
    words: tuple[str, ...], *, rows: int, needed: int, kept: int
) -> TruncationReport:
    """Refuse an item that exceeds its row bound."""
    del kept
    raise ContextOverflowError(words, rows=rows, needed=needed)


def _report(
    words: tuple[str, ...], *, rows: int, needed: int, kept: int
) -> TruncationReport:
    """Report the suffix omitted by a bounded encoding."""
    del rows, needed
    return TruncationReport(tuple(range(kept, len(words))), words[kept:])


REFUSE: Final = OverflowPlugin(_refuse)
REPORT: Final = OverflowPlugin(_report)


class ContextWindow:
    """Bounds how many rows an encoder produces for one item.

    Parameters
    ----------
    rows : int
        How many rows one item may take.
    overflow : OverflowPlugin
        What to do with an item that needs more.

    Raises
    ------
    TypeError
        If the row count is not an integer, or the overflow is not an
        ``OverflowPlugin``.
    ValueError
        If the row count is not positive.
    """

    __slots__ = ("_overflow", "_rows")

    def __init__(self, rows: int, *, overflow: OverflowPlugin) -> None:
        held = ensure_non_negative("a window's row count", rows)

        if held == 0:
            msg = "a window of no rows holds no word"
            raise ValueError(msg)

        if not isinstance(overflow, OverflowPlugin):
            msg = f"a window overflows by an OverflowPlugin, not {overflow!r}"
            raise TypeError(msg)

        self._rows = held
        self._overflow = overflow

    @property
    def rows(self) -> int:
        """Return how many rows one item may take."""
        return self._rows

    @property
    def overflow(self) -> OverflowPlugin:
        """Return what happens to an item that needs more rows than that."""
        return self._overflow

    def resolve(
        self, words: Sequence[str], /, *, needed: int, kept: int
    ) -> TruncationReport:
        """Apply the overflow policy to a known over-bound item."""
        return self._overflow.resolve(
            tuple(words), rows=self._rows, needed=needed, kept=kept
        )

    def fit(
        self, words: Sequence[str], costs: Sequence[int] | None = None
    ) -> TruncationReport:
        """Return the words the window drops from an item.

        The words that fit are a prefix of the item: the longest one whose rows sum
        to no more than the window holds. A word is kept or dropped whole, and every
        word after the first dropped one is dropped with it.

        Parameters
        ----------
        words : Sequence[str]
            The item's words, in order.
        costs : Sequence[int] | None
            How many rows each word takes, in the same order, or None where every
            word takes one.

        Returns
        -------
        TruncationReport
            The dropped suffix, empty when the whole item fits.

        Raises
        ------
        ContextOverflowError
            If the item does not fit and the window refuses overflow.
        ValueError
            If there is not one cost per word, or a cost is not positive.
        """
        held = tuple(words)
        taken = (1,) * len(held) if costs is None else tuple(costs)

        if len(taken) != len(held):
            msg = (
                f"an item of {len(held)} words was given {len(taken)} row costs, "
                f"so a word has no cost or a cost has no word"
            )
            raise ValueError(msg)

        if any(cost <= 0 for cost in taken):
            msg = f"every word takes at least one row, so the costs cannot be {taken}"
            raise ValueError(msg)

        kept = 0
        used = 0

        for cost in taken:
            if used + cost > self._rows:
                break

            used += cost
            kept += 1

        if kept == len(held):
            return TruncationReport((), ())

        return self.resolve(held, needed=sum(taken), kept=kept)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another window holds as many rows and overflows alike."""
        if type(other) is not ContextWindow:
            return NotImplemented

        return self._rows == other._rows and self._overflow is other._overflow

    @override
    def __hash__(self) -> int:
        """Return a hash over the row count and the overflow."""
        return hash((ContextWindow, self._rows, self._overflow))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this window."""
        return f"ContextWindow({self._rows}, overflow={self._overflow!r})"
