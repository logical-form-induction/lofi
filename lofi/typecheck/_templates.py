r"""Semantic types written as data, and the span constraints built from them.

A template writes a type as nested lists of primitive names: a string names a
primitive, and a list of two or more members is a function type nested to the
right, so ``["e", "e", "t"]`` is $e\to(e\to t)$. Resolving a template against
a declared language turns the names into interned type terms, and a name the
language does not declare is refused there rather than when the constraint is
first applied to an input.
"""

from typing import TYPE_CHECKING, final, override

from lofi._checks import ensure_name
from lofi._frozen import FrozenValue
from lofi.lexicon import Clamp

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from lofi._types import JsonValue
    from lofi.terms import Term
    from lofi.typecheck._ports import SortedLanguage, SpanAccessor

__all__ = [
    "ConstraintTemplate",
    "ResolvedConstraint",
    "TypeTemplate",
    "UnknownPrimitiveError",
    "read_templates",
    "type_template",
]

type TypeShape = str | tuple[TypeShape, ...]
"""A primitive name, or the argument and result shapes of a function type."""


@final
class UnknownPrimitiveError(ValueError):
    """A template naming a type the resolving signature does not declare.

    Attributes
    ----------
    primitive : str
        The name the template used.
    declared : tuple[str, ...]
        The primitive names the signature declares, in ascending order.

    Parameters
    ----------
    primitive : str
        The name the template used.
    declared : Iterable[str]
        The primitive names the signature declares.
    """

    __slots__ = ("declared", "primitive")

    def __init__(self, primitive: str, declared: Iterable[str]) -> None:
        self.primitive = primitive
        self.declared = tuple(sorted(declared))
        super().__init__(
            f"the template names the type {primitive!r}, and the signature "
            f"declares {list(self.declared)}"
        )


def _primitives(language: SortedLanguage) -> frozenset[str]:
    """Return the atomic types a declared language publishes.

    Parameters
    ----------
    language : SortedLanguage
        The language the template resolves against.

    Returns
    -------
    frozenset[str]
        The names in the declared type vocabulary.
    """
    return frozenset(language.store.signature.vocabulary(language.type_sort).names)


def _shape(value: JsonValue, at: str) -> TypeShape:
    """Return the shape a decoded template value writes.

    Parameters
    ----------
    value : JsonValue
        The decoded nested-list form.
    at : str
        Where the value was read, for the refusal message.

    Returns
    -------
    TypeShape
        The primitive name, or the member shapes of a function type.

    Raises
    ------
    TypeError
        If the value is neither text nor a list.
    ValueError
        If a list holds fewer than two members.
    """
    if isinstance(value, str):
        return ensure_name(f"the type at {at}", value)

    if not isinstance(value, list):
        msg = f"the type at {at} is neither a primitive nor a list: {value!r}"
        raise TypeError(msg)

    if len(value) < 2:
        msg = (
            f"the function type at {at} has {len(value)} members, and needs an "
            "argument and a result"
        )
        raise ValueError(msg)

    return tuple(
        _shape(member, f"{at}[{position}]") for position, member in enumerate(value)
    )


def type_template(value: JsonValue, /, *, at: str = "the template") -> TypeTemplate:
    """Read one type template out of its decoded nested-list form.

    Parameters
    ----------
    value : JsonValue
        The decoded form: a primitive name, or a list of two or more members.
    at : str
        Where the value was read, for the refusal messages.

    Returns
    -------
    TypeTemplate
        The template the form writes.

    Raises
    ------
    TypeError
        If a member is neither text nor a list.
    ValueError
        If a name is empty or a list holds fewer than two members.
    """
    return TypeTemplate(_shape(value, at))


def read_templates(
    declared: Mapping[str, JsonValue],
    /,
    *,
    within: str = "a template mapping",
) -> Mapping[str, TypeTemplate]:
    """Read a mapping of named type templates out of decoded data.

    Parameters
    ----------
    declared : Mapping[str, JsonValue]
        One decoded nested-list type per name.
    within : str
        What the mapping was read from, for the refusal messages.

    Returns
    -------
    Mapping[str, TypeTemplate]
        One template per declared name, in the order the data declares them.

    Raises
    ------
    TypeError
        If a member is neither text nor a list.
    ValueError
        If a name is empty or a list holds fewer than two members.
    """
    return {
        ensure_name(f"a name in {within}", name): type_template(
            written, at=f"{name!r} in {within}"
        )
        for name, written in declared.items()
    }


@final
class TypeTemplate(FrozenValue):
    """A semantic type written over primitive names rather than over terms.

    Parameters
    ----------
    shape : TypeShape
        A primitive name, or the member shapes of a right-nested function type.

    Raises
    ------
    TypeError
        If a member is neither text nor a tuple.
    ValueError
        If a name is empty or a tuple holds fewer than two members.
    """

    _noun = "type template"

    __slots__ = ("_shape",)

    def __init__(self, shape: TypeShape) -> None:
        object.__setattr__(self, "_shape", _checked(shape))

    @property
    def shape(self) -> TypeShape:
        """Return the name or the member shapes the template writes."""
        return self._shape

    def resolve(
        self,
        language: SortedLanguage,
        /,
    ) -> Term:
        """Return the interned type term the template writes.

        Parameters
        ----------
        language : SortedLanguage
            The declared language whose one store interns the resulting type.

        Returns
        -------
        Term
            The type term, with the members of a function type nested to the
            right.

        Raises
        ------
        UnknownPrimitiveError
            If the template names a primitive the signature does not declare.
        UnknownOperatorError
            If the signature declares no such function-type operator.
        """
        return _resolved(self._shape, language, _primitives(language))

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another template writes the same type.

        Parameters
        ----------
        other : object
            The value to compare.

        Returns
        -------
        bool
            Whether ``other`` is a template over the same shape.
        """
        if type(other) is not TypeTemplate:
            return NotImplemented

        return self._shape == other._shape

    @override
    def __hash__(self) -> int:
        """Return a hash over the shape."""
        return hash((TypeTemplate, self._shape))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this template."""
        return f"TypeTemplate({self._shape!r})"


def _checked(shape: TypeShape) -> TypeShape:
    """Return a validated template shape.

    Parameters
    ----------
    shape : TypeShape
        The name or member shapes to validate.

    Returns
    -------
    TypeShape
        The shape itself, with each name checked.

    Raises
    ------
    TypeError
        If a member is neither text nor a tuple.
    ValueError
        If a name is empty or a tuple holds fewer than two members.
    """
    if isinstance(shape, str):
        return ensure_name("a template's primitive", shape)

    if not isinstance(shape, tuple):
        msg = f"a template's shape is neither a name nor a tuple: {shape!r}"
        raise TypeError(msg)

    if len(shape) < 2:
        msg = (
            f"a template's function type has {len(shape)} members, and needs "
            "an argument and a result"
        )
        raise ValueError(msg)

    return tuple(_checked(member) for member in shape)


def _resolved(
    shape: TypeShape,
    language: SortedLanguage,
    declared: frozenset[str],
) -> Term:
    """Return the interned type term one shape writes.

    Parameters
    ----------
    shape : TypeShape
        The name or member shapes to build.
    language : SortedLanguage
        The declared language whose one store interns the resulting type.
    declared : frozenset[str]
        The primitive names the signature declares.

    Returns
    -------
    Term
        The type term the shape writes.

    Raises
    ------
    UnknownPrimitiveError
        If the shape names a primitive the signature does not declare.
    """
    if isinstance(shape, str):
        if shape not in declared:
            raise UnknownPrimitiveError(shape, declared)

        return language.store.const(language.type_sort, shape)

    operator = language.store.signature.by_name(language.function_type)
    built = _resolved(shape[-1], language, declared)

    for member in reversed(shape[:-1]):
        built = language.store.op(
            operator,
            _resolved(member, language, declared),
            built,
        )

    return built


@final
class ConstraintTemplate(FrozenValue):
    """Where a type constraint holds, and which types it admits there.

    Parameters
    ----------
    accessor : SpanAccessor
        Decides which spans of an input the constraint covers.
    permitted : Iterable[TypeTemplate]
        The types the covered spans may bear, written as data.
    inverted : bool
        Whether the covered spans are refused those types rather than held to
        them.

    Raises
    ------
    ValueError
        If no type is permitted.
    """

    _noun = "constraint template"

    __slots__ = ("_accessor", "_inverted", "_permitted")

    def __init__(
        self,
        accessor: SpanAccessor,
        permitted: Iterable[TypeTemplate],
        *,
        inverted: bool = False,
    ) -> None:
        held = frozenset(permitted)

        if not held:
            msg = "a constraint template permits no type"
            raise ValueError(msg)

        object.__setattr__(self, "_accessor", accessor)
        object.__setattr__(self, "_permitted", held)
        object.__setattr__(self, "_inverted", inverted)

    @property
    def accessor(self) -> SpanAccessor:
        """Return what decides the spans the constraint covers."""
        return self._accessor

    @property
    def permitted(self) -> frozenset[TypeTemplate]:
        """Return the types the covered spans may bear."""
        return self._permitted

    @property
    def inverted(self) -> bool:
        """Return whether the types are refused rather than required."""
        return self._inverted

    def resolve(
        self,
        language: SortedLanguage,
        /,
    ) -> ResolvedConstraint:
        """Return the constraint over the type terms the templates write.

        Parameters
        ----------
        language : SortedLanguage
            The declared language whose one store interns the resulting types.

        Returns
        -------
        ResolvedConstraint
            The constraint over interned types, ready to be applied to inputs.

        Raises
        ------
        UnknownPrimitiveError
            If a template names a primitive the signature does not declare.
        UnknownOperatorError
            If the signature declares no such function-type operator.
        """
        return ResolvedConstraint(
            self._accessor,
            (template.resolve(language) for template in self._permitted),
            inverted=self._inverted,
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another template constrains the same way.

        Parameters
        ----------
        other : object
            The value to compare.

        Returns
        -------
        bool
            Whether ``other`` covers the same spans with the same types.
        """
        if type(other) is not ConstraintTemplate:
            return NotImplemented

        return (
            self._accessor == other._accessor
            and self._permitted == other._permitted
            and self._inverted == other._inverted
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the accessor, the types and the direction."""
        return hash(
            (ConstraintTemplate, self._accessor, self._permitted, self._inverted)
        )

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this template."""
        return (
            f"ConstraintTemplate({self._accessor!r}, "
            f"permitted={len(self._permitted)}, inverted={self._inverted!r})"
        )


@final
class ResolvedConstraint(FrozenValue):
    """A type constraint whose types are interned and whose spans are not.

    The spans come from the accessor once an input is in hand, so the same
    constraint clamps a three-token input and a thirty-token one.

    Parameters
    ----------
    accessor : SpanAccessor
        Decides which spans of an input the constraint covers.
    permitted : Iterable[Term]
        The types the covered spans may bear.
    inverted : bool
        Whether the covered spans are refused those types rather than held to
        them.

    Raises
    ------
    ValueError
        If no type is permitted.
    """

    _noun = "resolved constraint"

    __slots__ = ("_accessor", "_inverted", "_permitted")

    def __init__(
        self,
        accessor: SpanAccessor,
        permitted: Iterable[Term],
        *,
        inverted: bool = False,
    ) -> None:
        held = frozenset(permitted)

        if not held:
            msg = "a resolved constraint permits no type"
            raise ValueError(msg)

        object.__setattr__(self, "_accessor", accessor)
        object.__setattr__(self, "_permitted", held)
        object.__setattr__(self, "_inverted", inverted)

    @property
    def accessor(self) -> SpanAccessor:
        """Return what decides the spans the constraint covers."""
        return self._accessor

    @property
    def permitted(self) -> frozenset[Term]:
        """Return the interned types the covered spans may bear."""
        return self._permitted

    @property
    def inverted(self) -> bool:
        """Return whether the types are refused rather than required."""
        return self._inverted

    def clamps(self, tokens: Sequence[str], /) -> tuple[Clamp[Term], ...]:
        """Return one clamp per span the accessor covers in an input.

        Parameters
        ----------
        tokens : Sequence[str]
            The surface tokens of one input, in order.

        Returns
        -------
        tuple[Clamp[Term], ...]
            The clamps, in the order the accessor returns their spans, empty
            where the accessor covers nothing.

        Raises
        ------
        ValueError
            If the accessor covers a span the input does not hold.
        """
        return tuple(
            Clamp(span, self._permitted, negate=self._inverted)
            for span in self._accessor.spans(tokens)
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another constraint constrains the same way.

        Parameters
        ----------
        other : object
            The value to compare.

        Returns
        -------
        bool
            Whether ``other`` covers the same spans with the same types.
        """
        if type(other) is not ResolvedConstraint:
            return NotImplemented

        return (
            self._accessor == other._accessor
            and self._permitted == other._permitted
            and self._inverted == other._inverted
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the accessor, the types and the direction."""
        return hash(
            (ResolvedConstraint, self._accessor, self._permitted, self._inverted)
        )

    @override
    def __repr__(self) -> str:
        """Return a rendering naming the accessor and how many types it admits."""
        return (
            f"ResolvedConstraint({self._accessor!r}, "
            f"permitted={len(self._permitted)}, inverted={self._inverted!r})"
        )
