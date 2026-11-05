"""Declaration envelopes prepared at validation boundaries."""

from json import dumps, loads
from typing import TYPE_CHECKING, Final, Protocol, cast, override

import didactic.api as dx

if TYPE_CHECKING:
    from didactic.types._typing import FieldValue

__all__ = [
    "GRAMMAR_SOURCE_VERSION",
    "GRAMMAR_TARGET_VERSION",
    "LANGUAGE_SOURCE_VERSION",
    "LANGUAGE_TARGET_VERSION",
    "GrammarCompileEnvelope",
    "LanguageCompileEnvelope",
    "canonical_declaration_bytes",
]

LANGUAGE_SOURCE_VERSION: Final = "lofi-language-decl"
"""The schema a language declaration is written in."""

LANGUAGE_TARGET_VERSION: Final = "lofi-language-native"
"""The schema the native compiler reads a language declaration as."""

GRAMMAR_SOURCE_VERSION: Final = "lofi-grammar-decl"
"""The schema a grammar declaration is written in."""

GRAMMAR_TARGET_VERSION: Final = "lofi-grammar-native"
"""The schema the native compiler reads a grammar declaration as."""


class _Declaration(Protocol):
    """A declaration with one public JSON projection."""

    def model_dump_json(
        self, *, indent: int | None = None, by_alias: bool = False
    ) -> str:
        """Return the declaration's public JSON projection."""
        ...


def canonical_declaration_bytes(declaration: _Declaration, /) -> bytes:
    """Return compact ASCII JSON bytes for one declaration, keys sorted."""
    encoded = dumps(
        loads(declaration.model_dump_json()),
        separators=(",", ":"),
        ensure_ascii=True,
        sort_keys=True,
    )
    return encoded.encode()


class _CompileEnvelope(dx.Model, extra="forbid"):
    """Canonical declaration bytes paired with private prepared runtime data."""

    source_version: str
    target_version: str
    canonical_declaration: bytes
    runtime: object | None = cast(
        "object | None",
        dx.field(default=None, opaque=True),
    )

    @property
    def prepared(self) -> object:
        """Return the runtime selections resolved at declaration validation."""
        prepared = self.runtime

        if prepared is None:
            msg = "a standalone compile envelope has no prepared runtime selections"
            raise RuntimeError(msg)

        return prepared

    @override
    def model_dump(
        self,
        *,
        include: set[str] | None = None,
        exclude: set[str] | None = None,
        by_alias: bool = False,
        exclude_none: bool = False,
        exclude_defaults: bool = False,
    ) -> dict[str, FieldValue]:
        """Return the public route fields, excluding private runtime identities."""
        hidden = set() if exclude is None else set(exclude)
        hidden.add("runtime")
        return super().model_dump(
            include=include,
            exclude=hidden,
            by_alias=by_alias,
            exclude_none=exclude_none,
            exclude_defaults=exclude_defaults,
        )


class LanguageCompileEnvelope(_CompileEnvelope):
    """Prepared language declaration bytes and runtime selections."""


class GrammarCompileEnvelope(_CompileEnvelope):
    """Prepared grammar declaration bytes and runtime selections."""
