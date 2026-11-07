"""Archive and rebuild a grammar with its declared learner language."""

from typing import TYPE_CHECKING

import didactic.api as dx

from lofi.spec._compile import (
    GRAMMAR_SOURCE_VERSION,
    GRAMMAR_TARGET_VERSION,
    canonical_declaration_bytes,
)
from lofi.spec.grammar._grammar import GrammarDecl
from lofi.spec.grammar._lowering import GrammarIso
from lofi.spec.language import LanguageDecl

if TYPE_CHECKING:
    from lofi.spec._archive import ArchivedLanguage
    from lofi.spec.grammar._values import GrammarAssembly

__all__ = [
    "ArchiveMismatchError",
    "archive_bytes",
    "rebuild_grammar",
]


class _GrammarArchiveDecl(dx.Model, extra="forbid"):
    """A grammar declaration beside its one learner-language declaration."""

    grammar: GrammarDecl
    language: LanguageDecl


class ArchiveMismatchError(ValueError):
    """An indexed theory that its declaration does not rebuild.

    Parameters
    ----------
    archived : str
        The theory digest the indexed declaration records.
    rebuilt : str
        The theory digest lowering the declaration produced.
    """

    __slots__ = ("archived", "rebuilt")

    def __init__(self, *, archived: str, rebuilt: str) -> None:
        self.archived = archived
        self.rebuilt = rebuilt

        msg = (
            f"the archived declaration lowers to theory {rebuilt}, not to the "
            f"archived theory {archived}"
        )

        super().__init__(msg)


def archive_bytes(assembly: GrammarAssembly, /) -> bytes:
    """Return canonical bytes that rebuild an assembly's unified store."""
    declaration = _GrammarArchiveDecl(
        grammar=GrammarDecl.model_validate_json(assembly.declaration),
        language=LanguageDecl.model_validate_json(assembly.language.declaration),
    )
    return canonical_declaration_bytes(declaration)


def rebuild_grammar(language: ArchivedLanguage, /) -> GrammarAssembly:
    """Rebuild the native grammar an indexed declaration names.

    Parameters
    ----------
    language : ArchivedLanguage
        The indexed declaration with the theory it was checked to.

    Returns
    -------
    GrammarAssembly
        The compiled grammar the declaration lowers to on the indexed route.

    Raises
    ------
    didactic.api.UnsupportedLoweringRouteError
        If the indexed route is not the route the lowerer implements.
    ArchiveMismatchError
        If lowering the declaration does not reproduce the indexed theory
        version, specification and digest.
    """
    route = language.theory.route
    expected = dx.LoweringRoute(GRAMMAR_SOURCE_VERSION, GRAMMAR_TARGET_VERSION)
    if route != expected:
        msg = (
            f"unsupported lowering route {route.source_version!r} -> "
            f"{route.target_version!r}; compiler provides "
            f"{expected.source_version!r} -> {expected.target_version!r}"
        )
        raise dx.UnsupportedLoweringRouteError(msg)
    source = _GrammarArchiveDecl.model_validate_json(language.declaration)
    assembly = GrammarIso().forward(source.grammar, source.language)
    grammar = assembly.grammar
    archived = language.theory

    if (
        grammar.theory_version != archived.version
        or grammar.theory_specification != archived.specification
        or grammar.theory_digest != archived.digest
    ):
        raise ArchiveMismatchError(
            archived=archived.digest, rebuilt=grammar.theory_digest
        )

    return assembly
