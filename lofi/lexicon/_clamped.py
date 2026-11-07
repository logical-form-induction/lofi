"""Clamped lexical sources and unsatisfiable-clamp policies."""

from typing import TYPE_CHECKING, final, override

from lofi._checks import ensure_finite, ensure_non_negative
from lofi._frozen import FrozenValue
from lofi.lexicon._errors import UnsatisfiableClampError
from lofi.lexicon._ports import ClampMode, ClampScope, UnsatisfiableAction

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from lofi.lexicon._ports import Clamp, Entry, LexicalSource, UnsatisfiablePolicy
    from lofi.terms import Matching

__all__ = [
    "ClampReport",
    "ClampedSource",
    "DropClampPolicy",
    "ErrorPolicy",
    "SkipItemPolicy",
]


@final
class ClampReport(FrozenValue):
    """Cumulative outcomes of unsatisfiable clamps seen by a source."""

    _noun = "clamp report"

    __slots__ = ("_dropped_clamps", "_skipped_items")

    def __init__(self, dropped_clamps: int, skipped_items: int) -> None:
        object.__setattr__(
            self,
            "_dropped_clamps",
            ensure_non_negative("a dropped-clamp count", dropped_clamps),
        )
        object.__setattr__(
            self,
            "_skipped_items",
            ensure_non_negative("a skipped-item count", skipped_items),
        )

    @property
    def dropped_clamps(self) -> int:
        """Return how many unsatisfiable clamps were dropped."""
        return self._dropped_clamps

    @property
    def skipped_items(self) -> int:
        """Return how many unsatisfiable items were skipped."""
        return self._skipped_items


class _Policy(FrozenValue):
    """Immutable built-in unsatisfiable-clamp policy."""

    _noun = "unsatisfiable-clamp policy"

    __slots__ = ("_action",)

    def __init__(self, action: UnsatisfiableAction) -> None:
        object.__setattr__(self, "_action", action)

    @property
    def action(self) -> UnsatisfiableAction:
        """Return the action this policy selects."""
        return self._action


@final
class ErrorPolicy(_Policy):
    """Abort when a clamp is unsatisfiable."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(UnsatisfiableAction.ERROR)


@final
class DropClampPolicy(_Policy):
    """Drop each unsatisfiable clamp from the current item."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(UnsatisfiableAction.DROP_CLAMP)


@final
class SkipItemPolicy(_Policy):
    """Emit no axioms for an item holding an unsatisfiable clamp."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(UnsatisfiableAction.SKIP_ITEM)


def _checked_clamp[T](clamp: Clamp[T]) -> Clamp[T]:
    """Validate and return a clamp.

    Parameters
    ----------
    clamp : Clamp[T]
        Clamp to validate.

    Returns
    -------
    Clamp[T]
        The validated clamp.

    Raises
    ------
    TypeError
        If a span endpoint has the wrong type.
    ValueError
        If the span is invalid or the clamp constrains neither field.
    """
    start, end = clamp.span
    ensure_non_negative("a clamped span's start", start)
    ensure_non_negative("a clamped span's end", end)

    if end <= start:
        msg = f"a clamp covers no position over the span [{start}, {end})"
        raise ValueError(msg)

    if not clamp.require_constituent and clamp.scope is ClampScope.OCCURRENCE:
        msg = "a token-scoped clamp cannot exclude a constituent"
        raise ValueError(msg)

    if not clamp.require_constituent and (clamp.category or clamp.meaning is not None):
        msg = "a negative-bracket clamp names no category or meaning"
        raise ValueError(msg)

    if clamp.require_constituent and not clamp.category and clamp.meaning is None:
        msg = f"a clamp names the category or the meaning it pins: {clamp!r}"
        raise ValueError(msg)

    return clamp


class ClampedSource[T]:
    """Restricts another lexical source with clamps.

    Parameters
    ----------
    source : LexicalSource[T, float]
        Source to restrict.
    clamps : Iterable[Clamp[T]]
        Constraints to apply after the source's own clamps.
    category_matching : Matching[T]
        Congruence used to compare entry categories with named categories.
    meaning_matching : Matching[T]
        Congruence used to compare entry meanings with a named meaning.
    on_unsatisfiable : UnsatisfiablePolicy
        Policy applied when hard clamps remove every entry at a covered token.

    Raises
    ------
    TypeError
        If a clamp span has the wrong type.
    ValueError
        If no clamp is supplied or a clamp is invalid.
    """

    __slots__ = (
        "_category_matching",
        "_clamps",
        "_dropped_clamps",
        "_meaning_matching",
        "_policy",
        "_skipped_items",
        "_source",
    )

    def __init__(
        self,
        source: LexicalSource[T, float],
        clamps: Iterable[Clamp[T]],
        category_matching: Matching[T],
        meaning_matching: Matching[T],
        on_unsatisfiable: UnsatisfiablePolicy,
    ) -> None:
        enforced = tuple(_checked_clamp(clamp) for clamp in clamps)

        if not enforced:
            msg = "a clamped source with no clamp narrows nothing"
            raise ValueError(msg)

        self._source = source
        self._clamps: tuple[Clamp[T], ...] = enforced
        self._category_matching = category_matching
        self._meaning_matching = meaning_matching
        self._policy = on_unsatisfiable
        self._dropped_clamps = 0
        self._skipped_items = 0

    @property
    def is_re_callable(self) -> bool:
        """Return whether the enumeration may be read more than once."""
        return self._source.is_re_callable

    @property
    def clamps(self) -> tuple[Clamp[T], ...]:
        """Return the clamps this source enforces, its own after the source's."""
        return (*self._source.clamps, *self._clamps)

    @property
    def enforced(self) -> tuple[Clamp[T], ...]:
        """Return the clamps this source adds, which are the ones it narrows by."""
        return self._clamps

    @property
    def report(self) -> ClampReport:
        """Return cumulative drop and skip counts."""
        return ClampReport(self._dropped_clamps, self._skipped_items)

    def axioms(
        self, tokens: Sequence[str], /
    ) -> tuple[tuple[int, Entry[T], float], ...]:
        """Return entries admitted by the tokens under every clamp.

        Parameters
        ----------
        tokens : Sequence[str]
            Surface tokens in position order.

        Returns
        -------
        tuple[tuple[int, Entry[T], float], ...]
            Admitted position, entry, and source-weight triples.

        Raises
        ------
        UnsatisfiableClampError
            If a covered position has no admitted entry.
        """
        offered = tuple(self._source.axioms(tokens))
        active = list(self._lexical_clamps())

        while True:
            admitted = tuple(
                axiom for axiom in offered if self._admits(axiom[0], axiom[1], active)
            )
            refusal = self._unsatisfiable(tokens, offered, admitted, active)

            if refusal is None:
                break

            action = self._policy.action

            if action is UnsatisfiableAction.ERROR:
                raise refusal

            if action is UnsatisfiableAction.SKIP_ITEM:
                self._skipped_items += 1
                return ()

            active.remove(refusal.clamp)
            self._dropped_clamps += 1

        return tuple(
            (
                position,
                entry,
                ensure_finite("a lexical source weight", weight)
                - self._penalty(position, entry, active),
            )
            for position, entry, weight in admitted
        )

    def _lexical_clamps(self) -> tuple[Clamp[T], ...]:
        """Return clamps whose scope can be decided from lexical entries."""
        return tuple(
            clamp
            for clamp in self._clamps
            if clamp.scope is ClampScope.OCCURRENCE
            or (clamp.span[1] - clamp.span[0] == 1 and clamp.require_constituent)
        )

    def _admits(
        self,
        position: int,
        entry: Entry[T],
        clamps: Sequence[Clamp[T]],
    ) -> bool:
        """Return whether every covering clamp admits an entry.

        Parameters
        ----------
        position : int
            Zero-based token position.
        entry : Entry[T]
            Candidate lexical entry.

        Returns
        -------
        bool
            Whether all covering clamps admit the entry.
        """
        return all(
            clamp.mode is ClampMode.SOFT
            or clamp.admits(entry, self._category_matching, self._meaning_matching)
            for clamp in clamps
            if clamp.covers(position)
        )

    def _penalty(
        self,
        position: int,
        entry: Entry[T],
        clamps: Sequence[Clamp[T]],
    ) -> float:
        """Return the summed soft clamp penalty for one offered entry."""
        return sum(
            clamp.penalty or 0.0
            for clamp in clamps
            if clamp.mode is ClampMode.SOFT
            and clamp.covers(position)
            and not clamp.admits(
                entry,
                self._category_matching,
                self._meaning_matching,
            )
        )

    def _unsatisfiable(
        self,
        tokens: Sequence[str],
        offered: tuple[tuple[int, Entry[T], float], ...],
        admitted: tuple[tuple[int, Entry[T], float], ...],
        clamps: Sequence[Clamp[T]],
    ) -> UnsatisfiableClampError[T] | None:
        """Return the first hard clamp no entry satisfies, if there is one.

        Parameters
        ----------
        tokens : Sequence[str]
            Surface tokens in position order.
        offered : tuple[tuple[int, Entry[T], float], ...]
            Entries offered before clamping.
        admitted : tuple[tuple[int, Entry[T], float], ...]
            Entries retained after clamping.

        Raises
        ------
        UnsatisfiableClampError
            If a covered position has no admitted entry.
        """
        filled = frozenset(position for position, _, _ in admitted)

        for position in self._covered(len(tokens), clamps):
            if position in filled:
                continue

            here = tuple(entry for at, entry, _ in offered if at == position)

            return UnsatisfiableClampError(
                clamp=self._culprit(position, here, clamps),
                position=position,
                form=tokens[position],
                offered=tuple(entry.category for entry in here),
            )

        return None

    def _covered(self, length: int, clamps: Sequence[Clamp[T]]) -> tuple[int, ...]:
        """Return sentence positions covered by any clamp.

        Parameters
        ----------
        length : int
            Sentence length.

        Returns
        -------
        tuple[int, ...]
            Covered positions in ascending order.
        """
        covered = {
            position
            for clamp in clamps
            if clamp.mode is ClampMode.HARD
            for position in range(clamp.span[0], min(clamp.span[1], length))
        }

        return tuple(sorted(covered))

    def _culprit(
        self,
        position: int,
        here: tuple[Entry[T], ...],
        clamps: Sequence[Clamp[T]],
    ) -> Clamp[T]:
        """Return the unsatisfied clamp for a position.

        Parameters
        ----------
        position : int
            Covered token position.
        here : tuple[Entry[T], ...]
            Entries offered at the position.

        Returns
        -------
        Clamp[T]
            First covering clamp with no admitted entry.
        """
        covering = tuple(
            clamp
            for clamp in clamps
            if clamp.mode is ClampMode.HARD and clamp.covers(position)
        )

        for clamp in covering:
            if not any(
                clamp.admits(entry, self._category_matching, self._meaning_matching)
                for entry in here
            ):
                return clamp

        return covering[0]

    @override
    def __repr__(self) -> str:
        """Return the source narrowed and how many clamps narrow it."""
        return f"ClampedSource({self._source!r}, clamps={len(self._clamps)})"
