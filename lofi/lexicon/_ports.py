"""Lexical entries, clamps, and source protocols."""

from enum import StrEnum
from typing import (
    TYPE_CHECKING,
    Protocol,
    final,
    override,
    runtime_checkable,
)

from lofi._frozen import FrozenValue

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from lofi.terms import Matching

__all__ = [
    "Clamp",
    "ClampMode",
    "ClampPolicy",
    "ClampScope",
    "Entry",
    "LexicalSource",
    "UnsatisfiableAction",
    "UnsatisfiablePolicy",
]


class ClampMode(StrEnum):
    """How a clamp changes analyses that do not satisfy it."""

    HARD = "hard"
    SOFT = "soft"


class ClampScope(StrEnum):
    """The structural unit a clamp constrains."""

    OCCURRENCE = "token"
    CONSTITUENT = "constituent"


class UnsatisfiableAction(StrEnum):
    """Actions a clamp policy may take when no analysis satisfies a clamp."""

    ERROR = "error"
    DROP_CLAMP = "drop_clamp"
    SKIP_ITEM = "skip_item"


@runtime_checkable
class UnsatisfiablePolicy(Protocol):
    """Selects the response to an unsatisfiable clamp."""

    @property
    def action(self) -> UnsatisfiableAction:
        """Return the action to take."""


@final
class ClampPolicy[T](FrozenValue):
    """Runtime policy shared by clamps emitted from one source."""

    _noun = "clamp policy"

    __slots__ = (
        "_matching",
        "_meaning_matching",
        "_mode",
        "_negate",
        "_on_unsatisfiable",
        "_penalty",
        "_require_constituent",
        "_scope",
    )

    def __init__(
        self,
        *,
        mode: ClampMode,
        penalty: float | None,
        scope: ClampScope,
        matching: Matching[T] | None,
        meaning_matching: Matching[T],
        require_constituent: bool,
        negate: bool,
        on_unsatisfiable: UnsatisfiablePolicy,
    ) -> None:
        object.__setattr__(self, "_mode", mode)
        object.__setattr__(self, "_penalty", penalty)
        object.__setattr__(self, "_scope", scope)
        object.__setattr__(self, "_matching", matching)
        object.__setattr__(self, "_meaning_matching", meaning_matching)
        object.__setattr__(self, "_require_constituent", require_constituent)
        object.__setattr__(self, "_negate", negate)
        object.__setattr__(self, "_on_unsatisfiable", on_unsatisfiable)

    @property
    def mode(self) -> ClampMode:
        """Return whether incompatible analyses are filtered or penalized."""
        return self._mode

    @property
    def penalty(self) -> float | None:
        """Return the penalty of a soft clamp, or None for a hard clamp."""
        return self._penalty

    @property
    def scope(self) -> ClampScope:
        """Return whether clamps bind tokens or constituents."""
        return self._scope

    @property
    def matching(self) -> Matching[T] | None:
        """Return the category congruence override, or None for the grammar's."""
        return self._matching

    @property
    def meaning_matching(self) -> Matching[T]:
        """Return the meaning congruence."""
        return self._meaning_matching

    @property
    def require_constituent(self) -> bool:
        """Return whether named spans must be constituents."""
        return self._require_constituent

    @property
    def negate(self) -> bool:
        """Return whether emitted clamps exclude their labels."""
        return self._negate

    @property
    def on_unsatisfiable(self) -> UnsatisfiablePolicy:
        """Return the policy for an unsatisfiable clamp."""
        return self._on_unsatisfiable


class Entry[Category, Meaning = Category](FrozenValue):
    """A weighted lexical entry.

    Parameters
    ----------
    form : str
        Surface form.
    category : Category
        Category assigned to the form.
    meaning : Meaning | None
        Logical form, if assigned.
    weight : float
        Additive lexical log weight.
    """

    _noun = "lexical value"

    __slots__ = ("_category", "_form", "_meaning", "_weight")

    def __init__(
        self,
        form: str,
        category: Category,
        meaning: Meaning | None = None,
        weight: float = 0.0,
    ) -> None:
        object.__setattr__(self, "_form", form)
        object.__setattr__(self, "_category", category)
        object.__setattr__(self, "_meaning", meaning)
        object.__setattr__(self, "_weight", weight)

    @property
    def form(self) -> str:
        """Return the surface form the entry is keyed by."""
        return self._form

    @property
    def category(self) -> Category:
        """Return the category the form may bear."""
        return self._category

    @property
    def meaning(self) -> Meaning | None:
        """Return the entry meaning, or None if none is assigned."""
        return self._meaning

    @property
    def weight(self) -> float:
        """Return the log weight the entry contributes."""
        return self._weight

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether two entries agree on form, category, and meaning.

        Parameters
        ----------
        other : object
            Value to compare.

        Returns
        -------
        bool
            Whether ``other`` has the same identity-bearing fields.
        """
        if type(other) is not Entry:
            return NotImplemented

        return (
            self._form == other._form
            and self._category == other._category
            and self._meaning == other._meaning
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the form, the category and the meaning."""
        return hash((Entry, self._form, self._category, self._meaning))

    @override
    def __repr__(self) -> str:
        """Return a rendering naming the form, the category and the weight."""
        return (
            f"Entry(form={self._form!r}, category={self._category!r}, "
            f"weight={self._weight!r})"
        )


@final
class Clamp[T](FrozenValue):
    """Restricts the analysis of a token span.

    Parameters
    ----------
    span : tuple[int, int]
        Half-open token interval.
    category : Iterable[T]
        Permitted categories, or an empty iterable to leave category free.
    meaning : T | None
        Required meaning, or ``None`` to leave meaning free.
    mode : ClampMode
        Whether the clamp filters incompatible analyses or penalizes them.
    penalty : float | None
        Positive log-weight penalty for a soft clamp; absent for a hard clamp.
    scope : ClampScope
        Whether the clamp applies to each covered token or to the named span.
    require_constituent : bool
        Whether the named span must form a constituent. False excludes that span.
    negate : bool
        Whether to forbid rather than require the named values.

    Raises
    ------
    TypeError
        If ``category`` is provided as a string or bytes.
    """

    _noun = "lexical value"

    __slots__ = (
        "_category",
        "_meaning",
        "_mode",
        "_negate",
        "_penalty",
        "_require_constituent",
        "_scope",
        "_span",
    )

    def __init__(
        self,
        span: tuple[int, int],
        category: Iterable[T] = (),
        meaning: T | None = None,
        *,
        mode: ClampMode = ClampMode.HARD,
        penalty: float | None = None,
        scope: ClampScope = ClampScope.CONSTITUENT,
        require_constituent: bool = True,
        negate: bool = False,
    ) -> None:
        if isinstance(category, str | bytes):
            msg = (
                "a clamp's categories are a collection of them rather than one "
                f"of them: {category!r}"
            )

            raise TypeError(msg)

        if not isinstance(mode, ClampMode):
            msg = f"a clamp mode is not a ClampMode: {mode!r}"
            raise TypeError(msg)

        if not isinstance(scope, ClampScope):
            msg = f"a clamp scope is not a ClampScope: {scope!r}"
            raise TypeError(msg)

        if not isinstance(require_constituent, bool):
            msg = f"require_constituent is not a boolean: {require_constituent!r}"
            raise TypeError(msg)

        if not isinstance(negate, bool):
            msg = f"negate is not a boolean: {negate!r}"
            raise TypeError(msg)

        if mode is ClampMode.HARD:
            if penalty is not None:
                msg = "a hard clamp has no penalty"
                raise ValueError(msg)
        elif not isinstance(penalty, int | float):
            msg = "a soft clamp declares a positive finite penalty"
            raise TypeError(msg)
        elif not 0.0 < float(penalty) < float("inf"):
            msg = f"a soft clamp penalty is not positive and finite: {penalty!r}"
            raise ValueError(msg)

        object.__setattr__(self, "_span", span)
        object.__setattr__(self, "_category", frozenset(category))
        object.__setattr__(self, "_meaning", meaning)
        object.__setattr__(self, "_mode", mode)
        object.__setattr__(
            self,
            "_penalty",
            None if penalty is None else float(penalty),
        )
        object.__setattr__(self, "_scope", scope)
        object.__setattr__(self, "_require_constituent", require_constituent)
        object.__setattr__(self, "_negate", negate)

    @property
    def span(self) -> tuple[int, int]:
        """Return the half-open token interval the clamp names."""
        return self._span

    @property
    def category(self) -> frozenset[T]:
        """Return the categories named, empty where the category is free."""
        return self._category

    @property
    def meaning(self) -> T | None:
        """Return the required meaning, or None if meaning is unconstrained."""
        return self._meaning

    @property
    def mode(self) -> ClampMode:
        """Return whether incompatible analyses are filtered or penalized."""
        return self._mode

    @property
    def penalty(self) -> float | None:
        """Return the soft-clamp penalty, or None for a hard clamp."""
        return self._penalty

    @property
    def scope(self) -> ClampScope:
        """Return whether this clamp constrains tokens or a constituent."""
        return self._scope

    @property
    def require_constituent(self) -> bool:
        """Return whether the named span must be a constituent."""
        return self._require_constituent

    @property
    def negate(self) -> bool:
        """Return whether the clamp forbids what it names rather than requiring it."""
        return self._negate

    def covers(self, position: int) -> bool:
        """Return whether a token position falls inside the span.

        Parameters
        ----------
        position : int
            Zero-based token position.

        Returns
        -------
        bool
            Whether the position lies in the half-open span.
        """
        start, end = self._span

        return start <= position < end

    def admits(
        self,
        entry: Entry[T],
        category_matching: Matching[T],
        meaning_matching: Matching[T],
    ) -> bool:
        """Return whether this clamp allows an entry.

        Parameters
        ----------
        entry : Entry[T]
            Entry at a position covered by the clamp.

        Returns
        -------
        bool
            Whether the entry satisfies the clamp.
        """
        named = (
            not self._category
            or any(
                category_matching.equivalent(entry.category, held)
                for held in self._category
            )
        ) and (
            self._meaning is None
            or (
                entry.meaning is not None
                and meaning_matching.equivalent(entry.meaning, self._meaning)
            )
        )

        return named is not self._negate

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether two clamps name the same span, terms, and direction.

        Parameters
        ----------
        other : object
            Value to compare.

        Returns
        -------
        bool
            Whether ``other`` has the same clamp fields.
        """
        if type(other) is not Clamp:
            return NotImplemented

        return (
            self._span == other._span
            and self._category == other._category
            and self._meaning == other._meaning
            and self._mode is other._mode
            and self._penalty == other._penalty
            and self._scope is other._scope
            and self._require_constituent is other._require_constituent
            and self._negate is other._negate
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the span, terms, mode, scope, and negation."""
        return hash(
            (
                Clamp,
                self._span,
                self._category,
                self._meaning,
                self._mode,
                self._penalty,
                self._scope,
                self._require_constituent,
                self._negate,
            )
        )

    @override
    def __repr__(self) -> str:
        """Return a rendering naming the span, both terms and the direction."""
        return (
            f"Clamp(span={self._span!r}, category={self._category!r}, "
            f"meaning={self._meaning!r}, mode={self._mode.value!r}, "
            f"penalty={self._penalty!r}, scope={self._scope.value!r}, "
            f"require_constituent={self._require_constituent!r}, "
            f"negate={self._negate!r})"
        )


@runtime_checkable
class LexicalSource[T, W](Protocol):
    """Enumerates lexical entries for token sequences."""

    @property
    def is_re_callable(self) -> bool:
        """Return whether the enumeration may be read more than once."""

    @property
    def clamps(self) -> tuple[Clamp[T], ...]:
        """Return the clamps this source enforces, empty when it enforces none."""

    def axioms(self, tokens: Sequence[str], /) -> Iterable[tuple[int, Entry[T], W]]:
        """Enumerate entries admitted by a token sequence.

        Parameters
        ----------
        tokens : Sequence[str]
            Surface tokens in position order.

        Returns
        -------
        Iterable[tuple[int, Entry[T], W]]
            Position, entry, and source weight for each admitted axiom.
        """
