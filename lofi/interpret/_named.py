"""Named derivation interpretations declared by a grammar."""

from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, final

from lofi._checks import ensure_name
from lofi._frozen import FrozenValue
from lofi.interpret._meaning import MeaningReading
from lofi.interpret._readings import CategoryReading, HeadDependencies, StringReading

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi._types import JsonValue
    from lofi.terms import Term

__all__ = [
    "BoundInterpretation",
    "CategoryInterpretation",
    "DependencyInterpretation",
    "InterpretationDeclaration",
    "MeaningInterpretation",
    "StringInterpretation",
]

type BoundInterpretation = (
    CategoryReading[Term, Term]
    | StringReading[Term, Term]
    | HeadDependencies[Term]
    | MeaningReading[Term]
)
"""One executable derivation interpretation selected by a declared name."""


class InterpretationDeclaration(Protocol):
    """A named grammar interpretation awaiting its meaning homomorphism."""

    @property
    def name(self) -> str:
        """Return the document-level selection name."""

    @property
    def settings(self) -> Mapping[str, JsonValue]:
        """Return the canonical fingerprint settings."""

    def bind(self, meaning: MeaningReading[Term], /) -> BoundInterpretation:
        """Bind the grammar declaration to a compiled meaning homomorphism."""


class _NamedInterpretation(FrozenValue):
    """Common immutable name held by a runtime interpretation declaration."""

    _noun = "interpretation declaration"
    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        object.__setattr__(self, "_name", ensure_name("an interpretation", name))

    @property
    def name(self) -> str:
        """Return the document-level selection name."""
        return self._name


@final
class CategoryInterpretation(_NamedInterpretation):
    """A named exact-category homomorphism."""

    __slots__ = ()

    @property
    def settings(self) -> Mapping[str, JsonValue]:
        """Return the canonical fingerprint settings."""
        return MappingProxyType({"name": self.name, "kind": "category"})

    def bind(self, meaning: MeaningReading[Term], /) -> BoundInterpretation:
        """Return an exact-category reading."""
        del meaning
        return CategoryReading()


@final
class StringInterpretation(_NamedInterpretation):
    """A named surface-string homomorphism."""

    __slots__ = ()

    @property
    def settings(self) -> Mapping[str, JsonValue]:
        """Return the canonical fingerprint settings."""
        return MappingProxyType({"name": self.name, "kind": "string"})

    def bind(self, meaning: MeaningReading[Term], /) -> BoundInterpretation:
        """Return a surface-string reading."""
        del meaning
        return StringReading()


@final
class DependencyInterpretation(_NamedInterpretation):
    """A named dependency reduction using the grammar's head selections."""

    __slots__ = ("_heads",)

    def __init__(self, name: str, heads: Mapping[str, int]) -> None:
        super().__init__(name)
        object.__setattr__(self, "_heads", MappingProxyType(dict(heads)))

    @property
    def settings(self) -> Mapping[str, JsonValue]:
        """Return the canonical fingerprint settings."""
        return MappingProxyType(
            {
                "name": self.name,
                "kind": "dependency",
                "heads": dict(self._heads),
            }
        )

    def bind(self, meaning: MeaningReading[Term], /) -> BoundInterpretation:
        """Return a dependency reduction over the declared rule heads."""
        del meaning
        return HeadDependencies(self._heads)


@final
class MeaningInterpretation(_NamedInterpretation):
    """A named logical form homomorphism."""

    __slots__ = ()

    @property
    def settings(self) -> Mapping[str, JsonValue]:
        """Return the canonical fingerprint settings."""
        return MappingProxyType({"name": self.name, "kind": "meaning"})

    def bind(self, meaning: MeaningReading[Term], /) -> BoundInterpretation:
        """Return the compiled logical form reading."""
        return meaning
