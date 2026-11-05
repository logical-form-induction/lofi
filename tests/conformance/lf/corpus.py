"""Declared one-store language shared by logical-form conformance tests."""

from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from tests.language import language

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi.spec.language import LanguageAssembly

__all__ = ["CONFORMANCE_CONSTANTS", "CONFORMANCE_TYPES", "conformance_language"]

_CORPUS_SYMBOLS: Final = (
    "f",
    "p",
    "q",
    "TRUE",
    "FALSE",
    "g",
    "h",
    "r",
    "s",
    "t",
    "u",
    "i",
    "j",
    "k",
    "friend",
    "happy",
    "v",
    "w",
    "compose",
    "l",
    "m",
    "n",
    "equal",
)

CONFORMANCE_CONSTANTS: Final = tuple(
    sorted(
        {
            *_CORPUS_SYMBOLS,
            "P",
            "always",
            "book",
            "eat",
            "give",
            "jo",
            "john",
            "love",
            "mary",
            "quietly",
            "run",
            "runs",
            "sleep",
            "sleeps",
            "undeclared",
        }
    )
)
"""Every constant named by the conformance corpus, in stable order."""

CONFORMANCE_TYPES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "john": "e",
        "mary": "e",
        "book": "e",
        "sleep": "e -> t",
        "run": "e -> t",
        "love": "e -> e -> t",
        "give": "e -> e -> e -> t",
        "quietly": "t -> t",
    }
)
"""Ground declarations required by typed conformance examples."""


def conformance_language() -> LanguageAssembly:
    """Lower a fresh one-store language for the conformance corpus."""
    return language(CONFORMANCE_CONSTANTS, types=CONFORMANCE_TYPES)
