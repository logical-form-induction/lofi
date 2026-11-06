"""Registered higher-order languages."""

from functools import cache
from typing import Final, Protocol, cast

import didactic.api as dx

from lofi._checks import ensure_name
from lofi.registry import Registry, RegistryError
from lofi.spec._constructor import ConstructorSpec, InfixConstructorNotationSpec
from lofi.spec.language._decl import (
    AtomicTypeSpec,
    BinderNotationSpec,
    BinderSpec,
    ConstantSpec,
    InfixNotationSpec,
    LanguageDecl,
    LogicSpec,
    PrefixNotationSpec,
)
from lofi.spec.language._errors import LanguageError

__all__ = ["LANGUAGE_FIELD", "registered_language", "resolve_language"]


class _LanguageBuilder(Protocol):
    """Build one registered language declaration on first use."""

    def __call__(self) -> LanguageDecl:
        """Return the registered declaration."""


def _logical_constants() -> tuple[ConstantSpec, ...]:
    """Return standard logical constants as typed declarations."""
    return (
        ConstantSpec(
            name="not",
            type="t -> t",
            classes=("sentential", "negation"),
            notation=PrefixNotationSpec(ascii="~", unicode="¬", precedence=6),
        ),
        ConstantSpec(
            name="and",
            type="t -> t -> t",
            classes=("sentential", "conjunction"),
            notation=InfixNotationSpec(
                ascii="&", unicode="∧", precedence=4, associativity="left"
            ),
        ),
        ConstantSpec(
            name="or",
            type="t -> t -> t",
            classes=("sentential", "disjunction"),
            notation=InfixNotationSpec(
                ascii="|",
                unicode="\N{LOGICAL OR}",
                precedence=3,
                associativity="left",
            ),
        ),
        ConstantSpec(
            name="if",
            type="t -> t -> t",
            classes=("sentential", "conditional"),
            notation=InfixNotationSpec(
                ascii="=>", unicode="⇒", precedence=2, associativity="right"
            ),
        ),
        ConstantSpec(
            name="iff",
            type="t -> t -> t",
            classes=("sentential", "biconditional"),
            notation=InfixNotationSpec(
                ascii="<=>", unicode="⇔", precedence=1, associativity="right"
            ),
        ),
        ConstantSpec(
            name="eq",
            type="a -> a -> t",
            quantifies=("a",),
            classes=("sentential", "equality"),
            notation=InfixNotationSpec(
                ascii="=", unicode="=", precedence=5, associativity="left"
            ),
        ),
        ConstantSpec(
            name="forall",
            type="(a -> t) -> t",
            quantifies=("a",),
            classes=("quantifier", "universal"),
            notation=BinderNotationSpec(ascii="forall", unicode="∀", bodies=1),
        ),
        ConstantSpec(
            name="exists",
            type="(a -> t) -> t",
            quantifies=("a",),
            classes=("quantifier", "existential"),
            notation=BinderNotationSpec(ascii="exists", unicode="∃", bodies=1),
        ),
    )


@cache
def _neo_davidsonian() -> LanguageDecl:
    """Build the registered typed neo-Davidsonian language."""
    return LanguageDecl(
        name="neo-davidsonian",
        atomic_types=(
            AtomicTypeSpec(name="e", kind="individual"),
            AtomicTypeSpec(name="v", kind="event"),
            AtomicTypeSpec(name="t", kind="proposition"),
        ),
        constructors=(
            ConstructorSpec(
                name="->",
                kind=("Ty", "Ty", "Ty"),
                notation=InfixConstructorNotationSpec(
                    spelling="->", precedence=1, associativity="right"
                ),
            ),
        ),
        constants=_logical_constants(),
        binders=BinderSpec(),
        logic=LogicSpec(negation="not"),
    )


@cache
def _propositional() -> LanguageDecl:
    """Build the registered quantifier-free language declaration."""
    declaration = _neo_davidsonian()
    constants = tuple(
        constant
        for constant in declaration.constants
        if not frozenset(constant.classes) & {"quantifier", "equality"}
    )
    return declaration.with_(
        name="propositional",
        constants=constants,
        logic=declaration.logic.with_(equality=False),
    )


def registered_language(name: str, /) -> LanguageDecl:
    """Return the installed language registered under a name.

    Raises
    ------
    LanguageError
        If the name is not registered, is registered more than once, or its
        entry does not hold a language declaration.
    """
    named = ensure_name("language", name)

    try:
        member = Registry[object]("lf_language").get(named)
    except RegistryError as refused:
        raise LanguageError(str(refused)) from refused

    if not callable(member):
        msg = f"the lf-language entry {name!r} names {member!r}, not a builder"
        raise LanguageError(msg)

    declaration = cast("_LanguageBuilder", member)()

    if not isinstance(declaration, LanguageDecl):
        msg = f"the lf-language entry {name!r} builds {declaration!r}, not a language"
        raise LanguageError(msg)

    return declaration


def resolve_language[V](value: V, /) -> V | LanguageDecl:
    """Resolve a registered language name, leaving a declaration mapping intact."""
    if isinstance(value, str):
        return registered_language(value)

    return value


LANGUAGE_FIELD: Final = dx.field(converter=resolve_language)
"""The one declared field treatment for a registered language or inline declaration."""
