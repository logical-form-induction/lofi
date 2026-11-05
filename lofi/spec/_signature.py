"""Immutable assignments to a learner signature's fixed slots."""

from __future__ import annotations

from math import factorial
from typing import TYPE_CHECKING, final, override

from lofi._checks import ensure_name
from lofi._frozen import FrozenValue

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["SignatureAssignments"]


def _signature_inventory(field: str, names: Sequence[str]) -> tuple[str, ...]:
    """Return one validated signature-family inventory."""
    if isinstance(names, str):
        msg = f"a signature's {field} inventory is text, not a sequence of names"
        raise TypeError(msg)

    held = tuple(
        ensure_name(f"a signature's {field} name at slot {slot}", name)
        for slot, name in enumerate(names)
    )

    if len(set(held)) != len(held):
        msg = f"a signature's {field} inventory repeats a name: {held!r}"
        raise ValueError(msg)

    return held


@final
class SignatureAssignments(FrozenValue):
    """Names assigned to the learner signature's fixed slots."""

    _noun = "signature assignments"

    __slots__ = ("_combinators", "_constructors", "_primitives")

    def __init__(
        self,
        primitives: Sequence[str],
        constructors: Sequence[str],
        combinators: Sequence[str],
    ) -> None:
        object.__setattr__(
            self, "_primitives", _signature_inventory("primitive", primitives)
        )
        object.__setattr__(
            self,
            "_constructors",
            _signature_inventory("constructor", constructors),
        )
        object.__setattr__(
            self,
            "_combinators",
            _signature_inventory("combinator", combinators),
        )

    @property
    def primitives(self) -> tuple[str, ...]:
        """Return primitive names in primitive-slot order."""
        return self._primitives

    @property
    def constructors(self) -> tuple[str, ...]:
        """Return constructor names in constructor-slot order."""
        return self._constructors

    @property
    def combinators(self) -> tuple[str, ...]:
        """Return grammar-rule names in controller-column order."""
        return self._combinators

    @property
    def scramble_space(self) -> int:
        """Return the exact number of distinct non-identity scrambles."""
        return (
            factorial(len(self._primitives))
            * factorial(len(self._constructors))
            * factorial(len(self._combinators))
            - 1
        )

    def _fields(
        self,
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        """Return the three assignment fields in constructor order."""
        return self._primitives, self._constructors, self._combinators

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another assignment has the same ordered fields."""
        if type(other) is not SignatureAssignments:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the ordered fields."""
        return hash((SignatureAssignments, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds these assignments."""
        return (
            f"SignatureAssignments({self._primitives!r}, {self._constructors!r}, "
            f"{self._combinators!r})"
        )
