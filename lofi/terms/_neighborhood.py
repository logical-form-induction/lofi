"""Finite neighborhoods and constructor-preserving term projections."""

from itertools import product
from types import MappingProxyType
from typing import TYPE_CHECKING, final

from lofi._checks import ensure_name
from lofi._frozen import FrozenValue, Unpicklable
from lofi._native import Term, TermStore

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable, Mapping, Sequence

    from lofi.terms._ports import Constructors, OpName
    from lofi.terms._values import Operator

__all__ = [
    "DeclaredConstructors",
    "NeighborhoodError",
    "ball",
    "constituents",
    "truncate",
    "typeless",
    "types",
]


type _Distance = Callable[[Term, Term], int]
"""A term distance used to filter a finite candidate collection."""

type _ConstructorParts = tuple[OpName, tuple[Term, ...]]
"""A declared constructor and its children."""


class NeighborhoodError(ValueError):
    """An error for an invalid neighborhood bound or construction."""

    __slots__ = ()


@final
class DeclaredConstructors(Unpicklable, FrozenValue):
    """An immutable view of constructor roles over one term store.

    Parameters
    ----------
    store : TermStore
        The store in which projected and enumerated terms are interned.
    roles : Mapping[OpName, Sequence[str]]
        One nonempty role name per child slot, keyed by constructor name in
        declaration order.

    Raises
    ------
    ValueError
        If an operator is nullary, carries a payload, is not an endofunctor,
        or has a role count different from its arity.
    """

    _noun = "declared constructor view"
    _because = "retains a live term store"
    __slots__ = ("_names", "_operators", "_roles", "_store")

    def __init__(
        self,
        store: TermStore,
        roles: Mapping[OpName, Sequence[str]],
        /,
    ) -> None:
        operators: dict[OpName, Operator] = {}
        declared: dict[OpName, tuple[str, ...]] = {}

        for name, child_roles in roles.items():
            constructor = ensure_name("a constructor name", name)
            operator = store.signature.by_name(constructor)
            named = tuple(
                ensure_name(f"constructor {constructor!r} role {slot}", role)
                for slot, role in enumerate(child_roles)
            )

            if operator.arity == 0:
                msg = f"constructor {constructor!r} has no child"
                raise ValueError(msg)
            if operator.has_payload:
                msg = f"constructor {constructor!r} carries a payload"
                raise ValueError(msg)
            if operator.arg_sorts != (operator.sort,) * operator.arity:
                msg = f"constructor {constructor!r} is not an endofunctor"
                raise ValueError(msg)
            if len(named) != operator.arity:
                msg = (
                    f"constructor {constructor!r} has {operator.arity} child slots "
                    f"but {len(named)} roles"
                )
                raise ValueError(msg)

            operators[constructor] = operator
            declared[constructor] = named

        object.__setattr__(self, "_store", store)
        object.__setattr__(self, "_names", tuple(declared))
        object.__setattr__(self, "_operators", MappingProxyType(operators))
        object.__setattr__(self, "_roles", MappingProxyType(declared))

    @property
    def names(self) -> tuple[OpName, ...]:
        """Return constructor names in declaration order."""
        return self._names

    def roles(self, constructor: OpName, /) -> tuple[str, ...]:
        """Return the declared role of every child slot."""
        return self._roles[constructor]

    def is_constructor(self, term: Term, /) -> bool:
        """Return whether a term is headed by a declared constructor."""
        return term.op.name in self._operators

    def constructor(self, term: Term, /) -> OpName:
        """Return a constructor term's declared operator name."""
        name = term.op.name

        if name not in self._operators:
            msg = f"term head {name!r} is not a declared constructor"
            raise NeighborhoodError(msg)

        return name

    def make(self, constructor: OpName, children: Sequence[Term], /) -> Term:
        """Return an interned constructor application."""
        return self._store.op(self._operators[constructor], *children)


def _primitive(term: Term, constructors: Constructors[Term], /) -> None:
    """Reject a term that is not a nonconstructor leaf."""
    if term.args or constructors.is_constructor(term):
        message = "a primitive must be a nonconstructor leaf"

        raise NeighborhoodError(message)


def _parts(term: Term, constructors: Constructors[Term], /) -> _ConstructorParts | None:
    """Return validated constructor parts or None for a primitive."""
    if not constructors.is_constructor(term):
        _primitive(term, constructors)

        return None

    constructor = constructors.constructor(term)
    roles = constructors.roles(constructor)

    if len(roles) != len(term.args):
        message = (
            f"constructor {constructor!r} has {len(term.args)} children but "
            f"{len(roles)} declared roles"
        )
        raise NeighborhoodError(message)

    return constructor, term.args


def _make(
    constructors: Constructors[Term],
    constructor: OpName,
    children: Sequence[Term],
    /,
) -> Term:
    """Return a validated interned constructor application."""
    try:
        term = constructors.make(constructor, children)

    except Exception as cause:
        message = "the constructor construction was rejected"

        raise NeighborhoodError(message) from cause

    parts = _parts(term, constructors)

    if parts is None:
        message = "the constructor construction returned a primitive"

        raise NeighborhoodError(message)

    built_constructor, built_children = parts

    if built_constructor != constructor or built_children != tuple(children):
        message = "the constructor construction changed its inputs"

        raise NeighborhoodError(message)

    return term


def _validate(term: Term, constructors: Constructors[Term], /) -> None:
    """Reject a term outside the declared constructor algebra."""
    parts = _parts(term, constructors)

    if parts is None:
        return

    _, children = parts

    for child in children:
        _validate(child, constructors)


def types(
    primitives: Collection[Term],
    constructors: Constructors[Term],
    *,
    max_nodes: int,
) -> frozenset[Term]:
    """Return all bounded constructor terms over finite primitives.

    Raises
    ------
    NeighborhoodError
        If ``max_nodes`` is not positive or an input violates the declared
        constructor algebra.
    """
    if max_nodes <= 0:
        message = "max_nodes must be positive"

        raise NeighborhoodError(message)

    generated = set(primitives)

    for primitive in generated:
        _primitive(primitive, constructors)

    for nodes in range(2, max_nodes + 1):
        available = tuple(term for term in generated if term.size < nodes)

        for constructor in constructors.names:
            arity = len(constructors.roles(constructor))

            for children in product(available, repeat=arity):
                if 1 + sum(child.size for child in children) == nodes:
                    generated.add(_make(constructors, constructor, children))

    return frozenset(generated)


def ball(
    center: Term,
    candidates: Iterable[Term],
    distance: _Distance,
    *,
    radius: int,
) -> tuple[Term, ...]:
    """Return candidates within a closed distance radius of a center.

    Raises
    ------
    NeighborhoodError
        If ``radius`` is negative.
    """
    if radius < 0:
        message = "radius must be nonnegative"

        raise NeighborhoodError(message)

    return tuple(
        candidate for candidate in candidates if distance(center, candidate) <= radius
    )


def _typeless(
    term: Term,
    primitive: Term,
    constructors: Constructors[Term],
) -> Term:
    """Return a validated typeless projection."""
    parts = _parts(term, constructors)

    if parts is None:
        return primitive

    constructor, children = parts

    return _make(
        constructors,
        constructor,
        tuple(_typeless(child, primitive, constructors) for child in children),
    )


def typeless(term: Term, primitive: Term, constructors: Constructors[Term], /) -> Term:
    """Return a term with every primitive replaced by one representative."""
    _primitive(primitive, constructors)

    return _typeless(term, primitive, constructors)


def _truncate(
    term: Term,
    primitive: Term,
    constructors: Constructors[Term],
    depth: int,
) -> Term:
    """Return a term cut to a positive one-based depth."""
    parts = _parts(term, constructors)

    if parts is None:
        return term

    if depth == 1:
        return primitive

    constructor, children = parts

    return _make(
        constructors,
        constructor,
        tuple(
            _truncate(child, primitive, constructors, depth - 1) for child in children
        ),
    )


def truncate(
    term: Term,
    primitive: Term,
    constructors: Constructors[Term],
    *,
    depth: int,
) -> Term:
    """Return a term truncated to a positive one-based depth.

    Raises
    ------
    NeighborhoodError
        If ``depth`` is not positive or an input violates the declared constructor
        algebra.
    """
    if depth <= 0:
        message = "depth must be positive"

        raise NeighborhoodError(message)

    _primitive(primitive, constructors)
    _validate(term, constructors)

    return _truncate(term, primitive, constructors, depth)


def constituents(
    term: Term,
    *,
    max_depth: int,
    sort_by_depth: bool = False,
) -> tuple[Term, ...]:
    """Return bounded subtree occurrences in preorder or stable depth order.

    Raises
    ------
    NeighborhoodError
        If ``max_depth`` is not positive.
    """
    if max_depth <= 0:
        message = "max_depth must be positive"

        raise NeighborhoodError(message)

    found: list[tuple[int, Term]] = []
    pending = [(1, term)]

    while pending:
        depth, current = pending.pop()
        found.append((depth, current))

        if depth == max_depth:
            continue

        pending.extend((depth + 1, child) for child in reversed(current.args))

    if sort_by_depth:
        found.sort(key=lambda occurrence: occurrence[0])

    return tuple(current for _, current in found)
