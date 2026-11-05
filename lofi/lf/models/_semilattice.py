"""Validated finite join-semilattices for plural denotations."""

from __future__ import annotations

from typing import TYPE_CHECKING, final

from lofi._checks import ensure_name
from lofi._frozen import FrozenValue
from lofi.lf.models._frame import Element

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi.lf.models._frame import ModelValue

__all__ = ["JoinSemilattice"]


@final
class JoinSemilattice(FrozenValue):
    """The nonempty powerset lattice over one finite atomic carrier."""

    _noun = "finite join-semilattice"
    __slots__ = ("_domain", "_type_name")

    def __init__(self, type_name: str, domain: Sequence[Element], /) -> None:
        named = ensure_name("a join-semilattice type", type_name)
        held = tuple(domain)

        if not held or any(
            type(value) is not Element or value.type_name != named or value.index != at
            for at, value in enumerate(held)
        ):
            msg = (
                f"a join-semilattice for {named!r} needs its complete indexed "
                "atomic domain"
            )
            raise ValueError(msg)

        atoms = len(held) + 1

        if atoms & (atoms - 1):
            msg = (
                f"a join-semilattice for {named!r} has a nonempty powerset "
                f"domain, not {len(held)} values"
            )
            raise ValueError(msg)

        object.__setattr__(self, "_type_name", named)
        object.__setattr__(self, "_domain", held)

    @property
    def type_name(self) -> str:
        """Return the atomic type carrying plural individuals."""
        return self._type_name

    @property
    def domain(self) -> tuple[Element, ...]:
        """Return the nonempty sums in bitset order."""
        return self._domain

    def _bits(self, value: ModelValue, /) -> int:
        if type(value) is not Element or value not in self._domain:
            msg = (
                f"a value in join-semilattice {self._type_name!r} is one of "
                f"{self._domain!r}, not {value!r}"
            )
            raise ValueError(msg)

        return value.index + 1

    def join(self, left: ModelValue, right: ModelValue, /) -> Element:
        """Return the least upper bound of two nonempty sums."""
        return self._domain[(self._bits(left) | self._bits(right)) - 1]

    def part_of(self, left: ModelValue, right: ModelValue, /) -> bool:
        """Return whether the first sum is included in the second."""
        left_bits = self._bits(left)
        right_bits = self._bits(right)
        return left_bits | right_bits == right_bits

    def atom(self, value: ModelValue, /) -> bool:
        """Return whether a sum contains exactly one atomic individual."""
        bits = self._bits(value)
        return bits & (bits - 1) == 0
