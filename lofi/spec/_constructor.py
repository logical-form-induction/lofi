"""Shared constructor and surface-notation declarations."""

from typing import Literal, Protocol, cast

import didactic.api as dx

from lofi.terms import ConstructorNotation, Operator

__all__ = [
    "ConstructorNotationSpec",
    "ConstructorSpec",
    "InfixConstructorNotationSpec",
    "MixfixConstructorNotationSpec",
    "PrefixConstructorNotationSpec",
]


def _named(field: str, message: str) -> dx.Axiom:
    """Return an axiom requiring one named field."""
    return dx.axiom(f"len({field}) > 0", message=message)


class ConstructorNotationSpec(dx.TaggedUnion, discriminator="style", extra="forbid"):
    """One declared surface form for a constructor."""


class _NotationDeclaration(Protocol):
    """A notation declaration that writes constructors of some arities."""

    def build(self, constructor: str, /) -> ConstructorNotation:
        """Build the surface notation of one named constructor."""
        ...

    def refuse_arity(self, constructor: str, arity: int, /) -> None:
        """Refuse a constructor whose arity this notation cannot write."""
        ...


class PrefixConstructorNotationSpec(ConstructorNotationSpec, extra="forbid"):
    """Prefix notation for a constructor of any arity."""

    style: Literal["prefix"] = "prefix"
    spelling: str

    __axioms__ = (_named("spelling", "prefix notation needs a spelling"),)

    def build(self, constructor: str, /) -> ConstructorNotation:
        """Build the prefix notation of one named constructor."""
        return ConstructorNotation.prefix(constructor, self.spelling)

    def refuse_arity(self, constructor: str, arity: int, /) -> None:
        """Accept a constructor of any arity."""
        del constructor, arity


class InfixConstructorNotationSpec(ConstructorNotationSpec, extra="forbid"):
    """Infix notation for a binary constructor."""

    style: Literal["infix"] = "infix"
    spelling: str
    precedence: int
    associativity: Literal["left", "right", "none"]

    __axioms__ = (
        _named("spelling", "infix notation needs a spelling"),
        dx.axiom("precedence > 0", message="infix precedence is positive"),
    )

    def build(self, constructor: str, /) -> ConstructorNotation:
        """Build the infix notation of one named constructor."""
        return ConstructorNotation.infix(
            constructor,
            self.spelling,
            precedence=self.precedence,
            associativity=self.associativity,
        )

    def refuse_arity(self, constructor: str, arity: int, /) -> None:
        """Refuse a constructor that is not binary.

        Raises
        ------
        ValueError
            If the constructor is not binary.
        """
        if arity != 2:
            msg = f"infix constructor {constructor!r} is not binary"
            raise ValueError(msg)


class MixfixConstructorNotationSpec(ConstructorNotationSpec, extra="forbid"):
    """Segments surrounding a constructor's ordered arguments."""

    style: Literal["mixfix"] = "mixfix"
    parts: tuple[str, ...]
    precedence: int

    __axioms__ = (
        dx.axiom("len(parts) > 1", message="mixfix notation separates arguments"),
        dx.axiom("any(parts)", message="mixfix notation needs a surface token"),
        dx.axiom("precedence > 0", message="mixfix precedence is positive"),
    )

    @dx.validates("parts")
    @classmethod
    def _starts_with_literal(cls, parts: tuple[str, ...]) -> tuple[str, ...]:
        """Require a leading token so the parser can select the notation."""
        if not parts or not parts[0]:
            msg = "mixfix notation needs a nonempty leading part"
            raise ValueError(msg)
        return parts

    def build(self, constructor: str, /) -> ConstructorNotation:
        """Build the mixfix notation of one named constructor."""
        return ConstructorNotation.mixfix(
            constructor, self.parts, precedence=self.precedence
        )

    def refuse_arity(self, constructor: str, arity: int, /) -> None:
        """Refuse a constructor whose arguments the parts do not separate.

        Raises
        ------
        ValueError
            If the parts are not one more than the arity.
        """
        if len(self.parts) != arity + 1:
            msg = f"mixfix constructor {constructor!r} needs {arity + 1} parts"
            raise ValueError(msg)


class ConstructorSpec(dx.Model, extra="forbid"):
    """A constructor's name, complete kind and declared notation."""

    name: str
    kind: tuple[str, ...]
    notation: ConstructorNotationSpec

    __axioms__ = (
        _named("name", "a constructor needs a name"),
        dx.axiom(
            "len(kind) > 1",
            message="a constructor kind needs an argument and a result",
        ),
        dx.axiom("all(kind)", message="a constructor sort needs a name"),
    )

    @property
    def arity(self) -> int:
        """Return the constructor arity derived from its kind."""
        return len(self.kind) - 1

    def build_notation(self) -> ConstructorNotation:
        """Build the constructor's declared surface notation."""
        return cast("_NotationDeclaration", self.notation).build(self.name)

    def refuse_notation_arity(self) -> None:
        """Refuse a notation that cannot write this constructor's arity.

        Raises
        ------
        ValueError
            If the notation does not write a constructor of this arity.
        """
        cast("_NotationDeclaration", self.notation).refuse_arity(self.name, self.arity)

    def build(self) -> Operator:
        """Build the declared payload-free constructor."""
        return Operator(
            self.name,
            self.kind[-1],
            arg_sorts=self.kind[:-1],
            family="constructor",
        )
