"""Finite interpretations keyed by stable vocabulary identifiers."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, final

from lofi._checks import ensure_name
from lofi._frozen import FrozenValue
from lofi.lf._constants import ChoiceFunction
from lofi.lf.models._frame import Element
from lofi.lf.models._semilattice import JoinSemilattice

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi.lf._semantics import FinitePredicate
    from lofi.lf.models._frame import Frame, ModelValue

__all__ = ["Structure"]


@final
class Structure(FrozenValue):
    """One finite frame and its non-logical constant denotations."""

    _noun = "finite structure"
    __slots__ = ("_choice", "_denotations", "_frame", "_name", "_semilattices")

    def __init__(
        self,
        name: str,
        frame: Frame,
        denotations: Mapping[int, ModelValue],
        /,
        *,
        choice: Mapping[FinitePredicate, ModelValue] | None = None,
        semilattices: tuple[str, ...] = (),
    ) -> None:
        held: dict[int, ModelValue] = {}

        for identifier, value in denotations.items():
            if not isinstance(identifier, int):
                msg = f"a constant identifier is int, not {identifier!r}"
                raise TypeError(msg)

            if identifier < 0:
                msg = f"a constant identifier is nonnegative, not {identifier!r}"
                raise ValueError(msg)

            held[identifier] = value

        object.__setattr__(self, "_name", ensure_name("a structure", name))
        object.__setattr__(self, "_frame", frame)
        object.__setattr__(self, "_denotations", MappingProxyType(held))
        object.__setattr__(
            self,
            "_choice",
            ChoiceFunction({} if choice is None else choice),
        )
        resources: dict[str, JoinSemilattice] = {}

        for type_name in semilattices:
            named = ensure_name("a join-semilattice type", type_name)

            if named in resources:
                msg = f"a structure declares join-semilattice {named!r} twice"
                raise ValueError(msg)

            resources[named] = JoinSemilattice(named, frame.atomic_domain(named))

        object.__setattr__(self, "_semilattices", MappingProxyType(resources))

    @property
    def name(self) -> str:
        """Return the stable structure name."""
        return self._name

    @property
    def frame(self) -> Frame:
        """Return the finite frame interpreted by this structure."""
        return self._frame

    @property
    def denotations(self) -> Mapping[int, ModelValue]:
        """Return denotations keyed by term-vocabulary identifier."""
        return self._denotations

    def denotation(self, identifier: int, /) -> ModelValue:
        """Return one constant's denotation."""
        try:
            return self._denotations[identifier]
        except KeyError:
            msg = (
                f"structure {self._name!r} has no denotation for constant {identifier}"
            )
            raise ValueError(msg) from None

    def choose(self, predicate: FinitePredicate, /) -> ModelValue:
        """Return this structure's choice for one finite predicate."""
        return self._choice.choose(predicate)

    def _semilattice(self, *values: ModelValue) -> JoinSemilattice:
        if not values or any(type(value) is not Element for value in values):
            msg = f"a plural operation needs atomic individuals, not {values!r}"
            raise TypeError(msg)

        type_names = {value.type_name for value in values if type(value) is Element}

        if len(type_names) != 1:
            msg = (
                f"a plural operation needs one atomic type, not {sorted(type_names)!r}"
            )
            raise ValueError(msg)

        type_name = type_names.pop()

        try:
            return self._semilattices[type_name]
        except KeyError:
            msg = f"structure {self._name!r} has no join-semilattice for {type_name!r}"
            raise ValueError(msg) from None

    def join(self, left: ModelValue, right: ModelValue, /) -> ModelValue:
        """Return the declared join of two plural individuals."""
        return self._semilattice(left, right).join(left, right)

    def part_of(self, left: ModelValue, right: ModelValue, /) -> bool:
        """Return whether one individual sum is part of another."""
        return self._semilattice(left, right).part_of(left, right)

    def atom(self, value: ModelValue, /) -> bool:
        """Return whether a plural individual is atomic."""
        return self._semilattice(value).atom(value)
