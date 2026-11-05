"""Shared declared languages for tests of logical-form consumers."""

from typing import TYPE_CHECKING

from lofi.spec.language import (
    ConstantSpec,
    LanguageAssembly,
    LanguageDecl,
    LanguageIso,
    registered_language,
)
from lofi.typecheck import LexicalSignature, Scheme

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lofi.spec.grammar import SignatureSpec
    from lofi.terms import Term


def _declaration(
    constants: Sequence[str] = (),
    /,
    *,
    types: Mapping[str, str] | None = None,
) -> LanguageDecl:
    """Declare one standard language with additional term constants."""
    declaration = registered_language("neo-davidsonian")
    standing = {constant.name for constant in declaration.constants}
    declared_types = {} if types is None else types
    added = tuple(
        ConstantSpec(name=name, type=declared_types.get(name, "t"))
        for name in constants
        if name not in standing
    )
    return declaration.with_(constants=(*declaration.constants, *added))


def language(
    constants: Sequence[str] = (),
    /,
    *,
    types: Mapping[str, str] | None = None,
) -> LanguageAssembly:
    """Build one standard language with additional declared term constants."""
    return LanguageIso().forward(_declaration(constants, types=types))


def grammar_language(
    signature: SignatureSpec,
    constants: Sequence[str] = (),
    /,
    *,
    types: Mapping[str, str] | None = None,
) -> LanguageAssembly:
    """Build one language extended by an explicit grammar signature."""
    return LanguageIso().extend(_declaration(constants, types=types), signature.extend)


def lexicon(
    assembly: LanguageAssembly,
    overrides: Mapping[int, Scheme] | None = None,
    /,
) -> LexicalSignature:
    """Return a language lexicon with selected symbol schemes replaced."""
    changed = {} if overrides is None else overrides
    return LexicalSignature({**assembly.lexicon.schemes, **changed})


def apply(
    assembly: LanguageAssembly,
    function: Term,
    *arguments: Term,
) -> Term:
    """Apply a term with the language's declared application operator."""
    return assembly.apply(function, *arguments)


def unary(assembly: LanguageAssembly, name: str, body: Term, /) -> Term:
    """Build a unary constant application or a lambda/quantifier form."""
    if name == assembly.calculus.lam:
        return assembly.store.op(
            assembly.store.signature.by_name(assembly.calculus.lam),
            body,
        )

    if name in assembly.profile.fragment.quantifiers:
        abstraction = unary(assembly, assembly.calculus.lam, body)
        return assembly.apply(assembly.constant(name), abstraction)

    return assembly.apply(assembly.constant(name), body)


def binary(
    assembly: LanguageAssembly,
    name: str,
    left: Term,
    right: Term,
) -> Term:
    """Build a two-argument constant application."""
    return assembly.apply(assembly.constant(name), left, right)
