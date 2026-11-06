"""Language identities shared by cross-lingual alignment evaluation."""

from typing import TYPE_CHECKING, Final, Literal, final, override

from lofi._checks import ensure_name
from lofi._frozen import FrozenValue

if TYPE_CHECKING:
    from lofi._types import JsonValue

__all__ = [
    "SOURCE_CLASSES",
    "SourceClass",
    "TransferLanguage",
    "UnavailableLanguage",
]

type SourceClass = Literal["gold", "silver"]
"""The annotation class a language is scored against."""

SOURCE_CLASSES: Final[tuple[SourceClass, ...]] = ("gold", "silver")
"""Annotation classes in declaration order."""


def _checked_class(source: SourceClass, /) -> SourceClass:
    """Return a declared annotation class."""
    if source not in SOURCE_CLASSES:
        msg = f"a language annotation class is not declared: {source!r}"
        raise ValueError(msg)
    return source


@final
class TransferLanguage(FrozenValue):
    """A language and the annotation class used to evaluate it."""

    _noun = "transfer language"
    __slots__ = ("_code", "_source")

    def __init__(self, code: str, source: SourceClass) -> None:
        object.__setattr__(self, "_code", ensure_name("a language code", code))
        object.__setattr__(self, "_source", _checked_class(source))

    @property
    def code(self) -> str:
        """Return the language code."""
        return self._code

    @property
    def source(self) -> SourceClass:
        """Return the annotation class."""
        return self._source

    def to_spec(self) -> dict[str, JsonValue]:
        """Return the language identity as scalar columns."""
        return {"language": self._code, "source": self._source}

    @override
    def __eq__(self, other: object) -> bool:
        if type(other) is not TransferLanguage:
            return NotImplemented
        return self._code == other._code and self._source == other._source

    @override
    def __hash__(self) -> int:
        return hash((TransferLanguage, self._code, self._source))

    @override
    def __repr__(self) -> str:
        return f"TransferLanguage({self._code!r}, {self._source!r})"


@final
class UnavailableLanguage(FrozenValue):
    """A declared silver language whose converted artifact is unavailable."""

    _noun = "unavailable language"
    __slots__ = ("_language", "_reason")

    def __init__(self, language: TransferLanguage, reason: str) -> None:
        if not isinstance(language, TransferLanguage):
            msg = f"an unavailable language is not a transfer language: {language!r}"
            raise TypeError(msg)
        if language.source != "silver":
            msg = (
                f"the {language.source} language {language.code!r} is unavailable, "
                "and only a silver language may be"
            )
            raise ValueError(msg)
        object.__setattr__(self, "_language", language)
        object.__setattr__(
            self, "_reason", ensure_name("an unavailability reason", reason)
        )

    @property
    def language(self) -> TransferLanguage:
        """Return the declared language."""
        return self._language

    @property
    def reason(self) -> str:
        """Return why its converted artifact is unavailable."""
        return self._reason

    @override
    def __eq__(self, other: object) -> bool:
        if type(other) is not UnavailableLanguage:
            return NotImplemented
        return self._language == other._language and self._reason == other._reason

    @override
    def __hash__(self) -> int:
        return hash((UnavailableLanguage, self._language, self._reason))

    @override
    def __repr__(self) -> str:
        return f"UnavailableLanguage({self._language!r}, {self._reason!r})"
