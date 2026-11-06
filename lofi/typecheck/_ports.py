"""Logical form typing, clamp-projection and span-accessor protocols."""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from lofi.terms import OpName, Path, Term, TermStore

if TYPE_CHECKING:
    from lofi.typecheck._values import TypingResult, TypingRule

__all__ = [
    "ClampProjection",
    "SortedLanguage",
    "SpanAccessor",
    "Substitution",
    "TypingSystem",
]


type Substitution = Mapping[int, Term]
"""Replacement for type metavariable atom indices."""


@runtime_checkable
class TypingSystem[S](Protocol):
    """A logical form type checker and inferred-type comparator."""

    def check(
        self,
        term: Term,
        signature: S,
        rules: Mapping[OpName, TypingRule],
        /,
        *,
        annotations: Mapping[Path, Term] | None = None,
        constraints: Mapping[Path, Term] | None = None,
    ) -> TypingResult:
        """Check a logical form under a signature and rule table."""
        ...

    def subsumes(
        self,
        general: Term,
        specific: Term,
        /,
    ) -> Substitution | None:
        """Match a general type against a fixed specific type."""
        ...

    def unifies(
        self,
        left: Term,
        right: Term,
        /,
    ) -> Substitution | None:
        """Return a most-general unifier for two compatible types."""
        ...


class SortedLanguage(Protocol):
    """A store whose signature holds both the types and the terms of one language."""

    @property
    def store(self) -> TermStore:
        """Return the store holding the language's types and terms."""
        ...

    @property
    def type_sort(self) -> str:
        """Return the sort of the language's types."""
        ...

    @property
    def term_sort(self) -> str:
        """Return the sort of the language's terms."""
        ...

    @property
    def function_type(self) -> str:
        """Return the name of the function type constructor."""
        ...


@runtime_checkable
class ClampProjection[T, R](Protocol):
    """Reads the category and the meaning a clamp tests off a chart reading."""

    @property
    def has_meaning(self) -> bool:
        """Return whether readings hold a meaning a clamp may pin."""

    def category(self, reading: R, /) -> T:
        """Return the category a chart item's reading assigns its span.

        Parameters
        ----------
        reading : R
            One chart item's reading.
        """

    def meaning(self, reading: R, /) -> T | None:
        """Return the meaning a reading assigns its span, or None if it holds none.

        Parameters
        ----------
        reading : R
            One chart item's reading.
        """


@runtime_checkable
class SpanAccessor(Protocol):
    """Decides which token spans of an input a constraint covers."""

    def spans(self, tokens: Sequence[str], /) -> tuple[tuple[int, int], ...]:
        """Return the half-open token spans the constraint covers.

        Parameters
        ----------
        tokens : Sequence[str]
            The surface tokens of one input, in order.
        """
