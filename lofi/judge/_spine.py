"""Bounded spine matching."""

from typing import TYPE_CHECKING, Literal, final, override

from lofi._checks import ensure_non_negative
from lofi._frozen import FrozenValue
from lofi.terms import Bindings, Spine, match

if TYPE_CHECKING:
    from collections.abc import Iterable

    from lofi.terms import Term

__all__ = [
    "SpineLimit",
    "SpineMatch",
    "SpineMatches",
    "match_spine",
]


type SpineLimit = Literal["max_order", "fuel"]
"""Bound that stopped spine enumeration."""


@final
class SpineMatch(FrozenValue):
    """Peeled spine order and its bindings."""

    _noun = "spine match value"

    __slots__ = ("_bindings", "_order")

    def __init__(self, order: int, bindings: Bindings) -> None:
        object.__setattr__(self, "_order", ensure_non_negative("order", order))
        object.__setattr__(self, "_bindings", bindings)

    @property
    def order(self) -> int:
        """Return the peeled frame count."""
        return self._order

    @property
    def bindings(self) -> Bindings:
        """Return the bindings from this match."""
        return self._bindings

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another match has the same fields."""
        if not isinstance(other, SpineMatch):
            return NotImplemented

        return self._order == other._order and self._bindings == other._bindings

    @override
    def __hash__(self) -> int:
        """Return a hash over the fields."""
        return hash((SpineMatch, self._order, self._bindings))

    @override
    def __repr__(self) -> str:
        """Return the constructor call."""
        return f"SpineMatch({self._order!r}, {self._bindings!r})"


@final
class SpineMatches(FrozenValue):
    """Spine matches and exhausted bounds."""

    _noun = "spine match value"

    __slots__ = ("_exhausted", "_matches")

    def __init__(
        self,
        matches: Iterable[SpineMatch] = (),
        exhausted: Iterable[SpineLimit] = (),
    ) -> None:
        object.__setattr__(self, "_matches", tuple(matches))
        object.__setattr__(self, "_exhausted", frozenset(exhausted))

    @property
    def matches(self) -> tuple[SpineMatch, ...]:
        """Return matches in increasing order."""
        return self._matches

    @property
    def exhausted(self) -> frozenset[SpineLimit]:
        """Return the bounds that stopped enumeration."""
        return self._exhausted

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another result has the same fields."""
        if not isinstance(other, SpineMatches):
            return NotImplemented

        return self._matches == other._matches and self._exhausted == other._exhausted

    @override
    def __hash__(self) -> int:
        """Return a hash over the fields."""
        return hash((SpineMatches, self._matches, self._exhausted))

    @override
    def __repr__(self) -> str:
        """Return the constructor call."""
        return f"SpineMatches({self._matches!r}, {self._exhausted!r})"


def _crosses_max_order(pattern: Spine, term: Term) -> bool:
    """Report whether the spine extends past the order bound."""
    if pattern.max_order is None:
        return False

    current = term

    for _ in range(pattern.max_order):
        if current.op.name not in pattern.ops or pattern.slot >= len(current.args):
            return False

        current = current.args[pattern.slot]

    return current.op.name in pattern.ops and pattern.slot < len(current.args)


def _available_order(pattern: Spine, term: Term) -> int:
    """Return the finite spine order supplied by a ground term."""
    current = term
    order = 0

    while current.op.name in pattern.ops and pattern.slot < len(current.args):
        order += 1
        current = current.args[pattern.slot]

    return order


def match_spine(
    pattern: Spine,
    term: Term,
    bindings: Bindings | None = None,
    *,
    fuel: int | None = None,
) -> SpineMatches:
    """Enumerate bounded spine matches and report exhausted limits."""
    budget = None if fuel is None else ensure_non_negative("fuel", fuel)
    found: list[SpineMatch] = []
    exhausted: set[SpineLimit] = set()

    maximum = (
        _available_order(pattern, term)
        if pattern.max_order is None
        else pattern.max_order
    )

    for attempted, order in enumerate(range(pattern.min_order, maximum + 1)):
        if budget is not None and attempted >= budget:
            exhausted.add("fuel")
            break

        exact = Spine(
            pattern.name,
            pattern.core,
            pattern.ops,
            slot=pattern.slot,
            min_order=order,
            max_order=order,
        )
        held = match(exact, term, bindings)

        if held is not None:
            found.append(SpineMatch(order, held))

    if _crosses_max_order(pattern, term):
        exhausted.add("max_order")

    return SpineMatches(found, exhausted)
