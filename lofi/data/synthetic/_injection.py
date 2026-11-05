"""Installed ill-typing operators over well-typed logical forms."""

from typing import TYPE_CHECKING, Final, Protocol, override

from lofi._checks import ensure_positive
from lofi._frozen import FrozenValue
from lofi.data.synthetic.spec.operators import (
    ArgumentSwapSpec,
    IllTypingOperatorPlugin,
    OverApplicationSpec,
    SortMismatchSpec,
)
from lofi.terms import at_path, positions, replace_at

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from random import Random

    from lofi.data.synthetic._interpretation import InterpretationFunction
    from lofi.terms import Operator, Path, Sort, Term, TermStore

__all__ = [
    "ARGUMENT_SWAP_PLUGIN",
    "OVER_APPLICATION_PLUGIN",
    "SORT_MISMATCH_PLUGIN",
    "IllTypingContext",
    "IllTypingOperator",
]


class IllTypingOperator(Protocol):
    """A named logical-form corruption constructed by an installed plugin."""

    @property
    def name(self) -> str:
        """Return the entry-point name used in reports."""
        ...

    def apply(self, random: Random, form: Term, /) -> Term | None:
        """Return an altered form, or nothing when inapplicable."""
        ...


class IllTypingContext(FrozenValue):
    """Interpretive structure shared by ill-typing operator builders."""

    _noun = "ill-typing context"
    __slots__ = ("_application", "_interpretation", "_store")

    def __init__(self, interpretation: InterpretationFunction) -> None:
        store = interpretation.store
        object.__setattr__(self, "_interpretation", interpretation)
        object.__setattr__(self, "_store", store)
        object.__setattr__(
            self,
            "_application",
            store.signature.by_name(interpretation.calculus.operators.app),
        )

    @property
    def interpretation(self) -> InterpretationFunction:
        """Return the interpretation that checks candidate forms."""
        return self._interpretation

    @property
    def store(self) -> TermStore:
        """Return the store candidates are interned in."""
        return self._store

    @property
    def application(self) -> Operator:
        """Return the calculus-declared application operator."""
        return self._application

    def applications(self, form: Term, /) -> tuple[Path, ...]:
        """Return every application path in a form."""
        application = self._application
        return positions(form, lambda node: node.op == application)


def _edited(
    edits: int,
    random: Random,
    form: Term,
    edit: Callable[[Random, Term], Term | None],
) -> Term | None:
    """Apply one primitive edit the declared positive number of times."""
    candidate = form
    for _ in range(edits):
        changed = edit(random, candidate)
        if changed is None:
            return None
        candidate = changed
    return candidate


class _ArgumentSwap(FrozenValue):
    _noun = "argument-swap ill-typing operator"
    __slots__ = ("_context", "_edits")

    def __init__(self, context: IllTypingContext, edits: int) -> None:
        object.__setattr__(self, "_context", context)
        object.__setattr__(self, "_edits", ensure_positive("edits", edits))

    @property
    def name(self) -> str:
        return "argument-swap"

    def apply(self, random: Random, form: Term, /) -> Term | None:
        return _edited(self._edits, random, form, self._once)

    def _once(self, random: Random, form: Term) -> Term | None:
        places = [
            path
            for path in self._context.applications(form)
            if at_path(form, path).args[0] is not at_path(form, path).args[1]
        ]
        if not places:
            return None
        node = at_path(form, (at := random.choice(places)))
        swapped = self._context.store.op(node.op, node.args[1], node.args[0])
        return replace_at(self._context.store, form, at, swapped)

    @override
    def __eq__(self, other: object) -> bool:
        return type(other) is _ArgumentSwap and self._edits == other._edits

    @override
    def __hash__(self) -> int:
        return hash((_ArgumentSwap, self._edits))

    @override
    def __repr__(self) -> str:
        return f"_ArgumentSwap(edits={self._edits!r})"


class _OverApplication(FrozenValue):
    _noun = "over-application ill-typing operator"
    __slots__ = ("_context", "_edits")

    def __init__(self, context: IllTypingContext, edits: int) -> None:
        object.__setattr__(self, "_context", context)
        object.__setattr__(self, "_edits", ensure_positive("edits", edits))

    @property
    def name(self) -> str:
        return "over-application"

    def apply(self, random: Random, form: Term, /) -> Term | None:
        return _edited(self._edits, random, form, self._once)

    def _once(self, random: Random, form: Term) -> Term | None:
        places = self._context.applications(form)
        if not places:
            return None
        node = at_path(form, (at := random.choice(places)))
        saturated = self._context.store.op(node.op, node, node.args[1])
        return replace_at(self._context.store, form, at, saturated)

    @override
    def __eq__(self, other: object) -> bool:
        return type(other) is _OverApplication and self._edits == other._edits

    @override
    def __hash__(self) -> int:
        return hash((_OverApplication, self._edits))

    @override
    def __repr__(self) -> str:
        return f"_OverApplication(edits={self._edits!r})"


class _SortMismatch(FrozenValue):
    _noun = "sort-mismatch ill-typing operator"
    __slots__ = ("_context", "_edits")

    def __init__(self, context: IllTypingContext, edits: int) -> None:
        object.__setattr__(self, "_context", context)
        object.__setattr__(self, "_edits", ensure_positive("edits", edits))

    @property
    def name(self) -> str:
        return "sort-mismatch"

    def apply(self, random: Random, form: Term, /) -> Term | None:
        return _edited(self._edits, random, form, self._once)

    def _once(self, random: Random, form: Term) -> Term | None:
        assignment = self._context.interpretation.check(form).assignment
        places: list[tuple[Path, Sort, int]] = []
        store = self._context.store
        for binder in positions(form, _binds):
            sort = _bound_sort(form, binder)
            under = tuple(_under(form, binder, sort))
            bound = next(
                (
                    assignment[path]
                    for path, node, index in under
                    if node is store.bvar(sort, index) and path in assignment
                ),
                None,
            )
            if bound is None:
                continue
            places.extend(
                (path, sort, index)
                for path, node, index in under
                if node is not store.bvar(sort, index)
                and assignment.get(path, bound) != bound
            )
        if not places:
            return None
        at, sort, index = random.choice(places)
        return replace_at(store, form, at, store.bvar(sort, index))

    @override
    def __eq__(self, other: object) -> bool:
        return type(other) is _SortMismatch and self._edits == other._edits

    @override
    def __hash__(self) -> int:
        return hash((_SortMismatch, self._edits))

    @override
    def __repr__(self) -> str:
        return f"_SortMismatch(edits={self._edits!r})"


def _build_argument_swap(
    declaration: ArgumentSwapSpec, context: IllTypingContext
) -> IllTypingOperator:
    return _ArgumentSwap(context, declaration.edits)


def _build_over_application(
    declaration: OverApplicationSpec, context: IllTypingContext
) -> IllTypingOperator:
    return _OverApplication(context, declaration.edits)


def _build_sort_mismatch(
    declaration: SortMismatchSpec, context: IllTypingContext
) -> IllTypingOperator:
    return _SortMismatch(context, declaration.edits)


ARGUMENT_SWAP_PLUGIN: Final = IllTypingOperatorPlugin(
    spec=ArgumentSwapSpec, build=_build_argument_swap
)
OVER_APPLICATION_PLUGIN: Final = IllTypingOperatorPlugin(
    spec=OverApplicationSpec, build=_build_over_application
)
SORT_MISMATCH_PLUGIN: Final = IllTypingOperatorPlugin(
    spec=SortMismatchSpec, build=_build_sort_mismatch
)


def _bound_sort(form: Term, binder: Path) -> Sort:
    """Return the sort of the variable a binder's first binding site binds."""
    return at_path(form, binder).op.binding[0].sort


def _under(form: Term, binder: Path, sort: Sort) -> Iterator[tuple[Path, Term, int]]:
    """Yield descendants of a binder with the index referring to it."""
    pending: list[tuple[Path, Term, int]] = [(binder, at_path(form, binder), 0)]
    while pending:
        path, node, index = pending.pop()
        pending.extend(
            (
                (*path, position),
                node.args[position],
                index + (0 if path == binder else node.op.shift(position, sort)),
            )
            for position in range(len(node.args) - 1, -1, -1)
        )
        if path != binder:
            yield path, node, index


def _binds(node: Term) -> bool:
    """Return whether a node binds a variable over its arguments."""
    return node.op.is_binder
