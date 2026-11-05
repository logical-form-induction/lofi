"""Installed corruption operators over static-lexicon token sequences."""

from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Protocol, override

from lofi._checks import ensure_non_negative, ensure_positive
from lofi._frozen import FrozenValue
from lofi.data.synthetic.spec.operators import (
    CorruptionOperatorFactory,
    CorruptionOperatorPlugin,
    DeletionSpec,
    InsertionSpec,
    SubstitutionSpec,
    SwapSpec,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from random import Random

    from lofi.lexicon import StaticLexicon
    from lofi.terms import Matching, Term

__all__ = [
    "DELETION_PLUGIN",
    "INSERTION_PLUGIN",
    "SUBSTITUTION_PLUGIN",
    "SWAP_PLUGIN",
    "CorruptionContext",
    "CorruptionOperator",
]


class CorruptionOperator(Protocol):
    """A named token-sequence corruption constructed by an installed plugin."""

    @property
    def name(self) -> str:
        """Return the entry-point name used in reports."""
        ...

    def apply(
        self, random: Random, forms: tuple[str, ...], /
    ) -> tuple[str, ...] | None:
        """Return a corrupted sequence, or nothing when inapplicable."""
        ...


class CorruptionContext(FrozenValue):
    """Lexical structure shared by corruption operator builders."""

    _noun = "corruption context"
    __slots__ = ("_forms", "_max_length", "_substitutes")

    def __init__(
        self,
        lexicon: StaticLexicon[Term],
        matching: Matching[Term],
        max_length: int,
    ) -> None:
        object.__setattr__(self, "_forms", lexicon.forms)
        object.__setattr__(
            self, "_max_length", ensure_non_negative("max_length", max_length)
        )
        object.__setattr__(
            self,
            "_substitutes",
            MappingProxyType(_substitutes(lexicon, matching)),
        )

    @property
    def forms(self) -> tuple[str, ...]:
        """Return lexicon forms in declaration order."""
        return self._forms

    @property
    def max_length(self) -> int:
        """Return the longest insertion result."""
        return self._max_length

    @property
    def substitutes(self) -> Mapping[str, tuple[str, ...]]:
        """Return category-disjoint substitutes by source form."""
        return self._substitutes


def _edited(
    edits: int,
    random: Random,
    forms: tuple[str, ...],
    edit: Callable[[Random, tuple[str, ...]], tuple[str, ...] | None],
) -> tuple[str, ...] | None:
    """Apply one primitive edit the declared positive number of times."""
    candidate = forms
    for _ in range(edits):
        changed = edit(random, candidate)
        if changed is None:
            return None
        candidate = changed
    return candidate


class _Swap(FrozenValue):
    _noun = "swap corruption"
    __slots__ = ("_edits",)

    def __init__(self, edits: int) -> None:
        object.__setattr__(self, "_edits", ensure_positive("edits", edits))

    @property
    def name(self) -> str:
        return "swap"

    def apply(
        self, random: Random, forms: tuple[str, ...], /
    ) -> tuple[str, ...] | None:
        return _edited(self._edits, random, forms, self._once)

    @staticmethod
    def _once(random: Random, forms: tuple[str, ...]) -> tuple[str, ...] | None:
        places = [
            place for place in range(len(forms) - 1) if forms[place] != forms[place + 1]
        ]
        if not places:
            return None
        at = random.choice(places)
        return (*forms[:at], forms[at + 1], forms[at], *forms[at + 2 :])

    @override
    def __eq__(self, other: object) -> bool:
        return type(other) is _Swap and self._edits == other._edits

    @override
    def __hash__(self) -> int:
        return hash((_Swap, self._edits))

    @override
    def __repr__(self) -> str:
        return f"_Swap(edits={self._edits!r})"


class _Deletion(FrozenValue):
    _noun = "deletion corruption"
    __slots__ = ("_edits",)

    def __init__(self, edits: int) -> None:
        object.__setattr__(self, "_edits", ensure_positive("edits", edits))

    @property
    def name(self) -> str:
        return "deletion"

    def apply(
        self, random: Random, forms: tuple[str, ...], /
    ) -> tuple[str, ...] | None:
        return _edited(self._edits, random, forms, self._once)

    @staticmethod
    def _once(random: Random, forms: tuple[str, ...]) -> tuple[str, ...] | None:
        if len(forms) < 2:
            return None
        at = random.randrange(len(forms))
        return (*forms[:at], *forms[at + 1 :])

    @override
    def __eq__(self, other: object) -> bool:
        return type(other) is _Deletion and self._edits == other._edits

    @override
    def __hash__(self) -> int:
        return hash((_Deletion, self._edits))

    @override
    def __repr__(self) -> str:
        return f"_Deletion(edits={self._edits!r})"


class _Insertion(FrozenValue):
    _noun = "insertion corruption"
    __slots__ = ("_edits", "_forms", "_max_length")

    def __init__(self, context: CorruptionContext, edits: int) -> None:
        object.__setattr__(self, "_forms", context.forms)
        object.__setattr__(self, "_max_length", context.max_length)
        object.__setattr__(self, "_edits", ensure_positive("edits", edits))

    @property
    def name(self) -> str:
        return "insertion"

    def apply(
        self, random: Random, forms: tuple[str, ...], /
    ) -> tuple[str, ...] | None:
        return _edited(self._edits, random, forms, self._once)

    def _once(self, random: Random, forms: tuple[str, ...]) -> tuple[str, ...] | None:
        if len(forms) >= self._max_length or not self._forms:
            return None
        at = random.randrange(len(forms) + 1)
        form = random.choice(self._forms)
        return (*forms[:at], form, *forms[at:])

    def _fields(self) -> tuple[tuple[str, ...], int, int]:
        return self._forms, self._max_length, self._edits

    @override
    def __eq__(self, other: object) -> bool:
        return type(other) is _Insertion and self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        return hash((_Insertion, self._fields()))

    @override
    def __repr__(self) -> str:
        return (
            f"_Insertion(forms={self._forms!r}, max_length={self._max_length!r}, "
            f"edits={self._edits!r})"
        )


class _Substitution(FrozenValue):
    _noun = "substitution corruption"
    __slots__ = ("_edits", "_substitutes")

    def __init__(self, context: CorruptionContext, edits: int) -> None:
        object.__setattr__(self, "_substitutes", context.substitutes)
        object.__setattr__(self, "_edits", ensure_positive("edits", edits))

    @property
    def name(self) -> str:
        return "substitution"

    def apply(
        self, random: Random, forms: tuple[str, ...], /
    ) -> tuple[str, ...] | None:
        return _edited(self._edits, random, forms, self._once)

    def _once(self, random: Random, forms: tuple[str, ...]) -> tuple[str, ...] | None:
        places = [
            place for place, form in enumerate(forms) if self._substitutes.get(form)
        ]
        if not places:
            return None
        at = random.choice(places)
        form = random.choice(self._substitutes[forms[at]])
        return (*forms[:at], form, *forms[at + 1 :])

    def _fields(self) -> tuple[tuple[tuple[str, tuple[str, ...]], ...], int]:
        return tuple(self._substitutes.items()), self._edits

    @override
    def __eq__(self, other: object) -> bool:
        return type(other) is _Substitution and self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        return hash((_Substitution, self._fields()))

    @override
    def __repr__(self) -> str:
        return (
            f"_Substitution(substitutes={dict(self._substitutes)!r}, "
            f"edits={self._edits!r})"
        )


class _Factory(FrozenValue):
    """Build one installed corruption after boundary validation is discarded."""

    _noun = "corruption-operator factory"
    __slots__ = ("_build", "_edits")

    def __init__(self, edits: int, build: _OperatorBuilder) -> None:
        object.__setattr__(self, "_edits", ensure_positive("edits", edits))
        object.__setattr__(self, "_build", build)

    def __call__(self, context: CorruptionContext, /) -> CorruptionOperator:
        return self._build(context, self._edits)


class _OperatorBuilder(Protocol):
    """Build one installed corruption from interior values."""

    def __call__(
        self, context: CorruptionContext, edits: int, /
    ) -> CorruptionOperator: ...


def _swap(context: CorruptionContext, edits: int) -> CorruptionOperator:
    del context
    return _Swap(edits)


def _deletion(context: CorruptionContext, edits: int) -> CorruptionOperator:
    del context
    return _Deletion(edits)


def _insertion(context: CorruptionContext, edits: int) -> CorruptionOperator:
    return _Insertion(context, edits)


def _substitution(context: CorruptionContext, edits: int) -> CorruptionOperator:
    return _Substitution(context, edits)


def _compile_swap(declaration: SwapSpec) -> CorruptionOperatorFactory:
    return _Factory(declaration.edits, _swap)


def _compile_deletion(declaration: DeletionSpec) -> CorruptionOperatorFactory:
    return _Factory(declaration.edits, _deletion)


def _compile_insertion(declaration: InsertionSpec) -> CorruptionOperatorFactory:
    return _Factory(declaration.edits, _insertion)


def _compile_substitution(declaration: SubstitutionSpec) -> CorruptionOperatorFactory:
    return _Factory(declaration.edits, _substitution)


SWAP_PLUGIN: Final = CorruptionOperatorPlugin(spec=SwapSpec, compiler=_compile_swap)
DELETION_PLUGIN: Final = CorruptionOperatorPlugin(
    spec=DeletionSpec, compiler=_compile_deletion
)
INSERTION_PLUGIN: Final = CorruptionOperatorPlugin(
    spec=InsertionSpec, compiler=_compile_insertion
)
SUBSTITUTION_PLUGIN: Final = CorruptionOperatorPlugin(
    spec=SubstitutionSpec, compiler=_compile_substitution
)


def _substitutes(
    lexicon: StaticLexicon[Term], matching: Matching[Term]
) -> dict[str, tuple[str, ...]]:
    """Return, for every form, forms sharing no category with it."""
    categories = {
        form: tuple(entry.category for entry in lexicon.entries(form))
        for form in lexicon.forms
    }

    def disjoint(left: str, right: str) -> bool:
        return not any(
            matching.equivalent(a, b)
            for a in categories[left]
            for b in categories[right]
        )

    return {
        form: tuple(
            other for other in lexicon.forms if other != form and disjoint(form, other)
        )
        for form in lexicon.forms
    }
