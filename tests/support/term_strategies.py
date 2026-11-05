"""Generate terms and registered specification variants for tests."""

from typing import TYPE_CHECKING, Protocol, cast

from hypothesis import strategies as st

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi.terms import Term, TermStore
    from lofi.terms._ports import Sort
    from lofi.terms._values import Operator

__all__ = ["spec_of", "terms"]


class _Generated(Protocol):
    """Values whose types Hypothesis infers."""


class _FieldSpec(Protocol):
    """Field metadata needed for generation."""

    @property
    def annotation(self) -> type[_Generated]:
        """Return the field annotation."""
        ...

    @property
    def usage_mode(self) -> str:
        """Return whether callers may initialize the field."""
        ...


class _FieldRegistry(Protocol):
    """Model fields independent of a model library."""

    @property
    def __field_specs__(self) -> Mapping[str, _FieldSpec]:
        """Return field metadata by name."""
        ...


class _VariantRegistry[S](Protocol):
    """Registered variants independent of their root type."""

    @property
    def __variants__(self) -> Mapping[str, type[S]]:
        """Return registered variants by tag."""
        ...


type _Scope = tuple[int, ...]
"""How many binding sites of each bound sort enclose a point, in sort order."""


class _TermStrategies:
    """Size-bounded term strategies for one store."""

    __slots__ = (
        "_bound",
        "_by_sort",
        "_can_generate",
        "_closed",
        "_max_size",
        "_store",
        "_suffixes",
    )

    def __init__(self, store: TermStore, *, max_size: int, closed: bool) -> None:
        """Index the signature by result sort."""
        signature = store.signature
        self._store = store
        self._max_size = max_size
        self._closed = closed
        self._bound = signature.bound_sorts
        self._can_generate: dict[tuple[Sort, int, _Scope], bool] = {}
        self._suffixes: dict[tuple[Operator, int, int, _Scope], bool] = {}
        self._by_sort: Mapping[Sort, tuple[Operator, ...]] = {
            declared: tuple(op for op in signature.operators if op.sort == declared)
            for declared in signature.sorts
        }

    def _depth(self, scope: _Scope, sort: Sort) -> int:
        """Return how many binding sites of a sort enclose a point."""
        return scope[self._bound.index(sort)]

    def _under(self, op: Operator, position: int, scope: _Scope) -> _Scope:
        """Return the scope inside one argument of an operator."""
        return tuple(
            depth + op.shift(position, sort)
            for sort, depth in zip(self._bound, scope, strict=True)
        )

    def _has_payload(self, op: Operator, scope: _Scope) -> bool:
        """Return whether an operator has a valid payload at this point."""
        if op.payload_sort == "constant":
            return len(self._store.signature.vocabulary(op.sort)) > 0

        return not (
            self._closed
            and op.payload_sort == "index"
            and self._depth(scope, op.sort) == 0
        )

    def can_generate(self, wanted: Sort, budget: int, scope: _Scope) -> bool:
        """Return whether a sort is inhabited within one budget."""
        key = (wanted, budget, scope)
        known = self._can_generate.get(key)

        if known is not None:
            return known

        if budget < 1:
            return False

        found = any(
            self._has_payload(op, scope)
            and (op.arity == 0 or self.suffix_is_viable(op, 0, budget - 1, scope))
            for op in self._by_sort[wanted]
        )
        self._can_generate[key] = found

        return found

    def suffix_is_viable(
        self,
        op: Operator,
        position: int,
        remaining: int,
        scope: _Scope,
    ) -> bool:
        """Return whether the remaining arguments fit a budget."""
        key = (op, position, remaining, scope)
        known = self._suffixes.get(key)

        if known is not None:
            return known

        if position == op.arity:
            return True

        positions_left = op.arity - position - 1
        upper = remaining - positions_left

        found = any(
            self.can_generate(
                op.arg_sorts[position],
                child_budget,
                self._under(op, position, scope),
            )
            and self.suffix_is_viable(
                op,
                position + 1,
                remaining - child_budget,
                scope,
            )
            for child_budget in range(1, upper + 1)
        )
        self._suffixes[key] = found

        return found

    def operators(
        self, wanted: Sort, budget: int, scope: _Scope
    ) -> tuple[Operator, ...]:
        """Return operators usable at one generation point."""
        return tuple(
            op
            for op in self._by_sort[wanted]
            if self._has_payload(op, scope)
            and (op.arity == 0 or self.suffix_is_viable(op, 0, budget - 1, scope))
        )

    def budgets(
        self,
        op: Operator,
        position: int,
        remaining: int,
        scope: _Scope,
    ) -> tuple[int, ...]:
        """Return viable budgets for one argument position."""
        positions_left = op.arity - position - 1
        child_sort = op.arg_sorts[position]
        inner = self._under(op, position, scope)

        return tuple(
            child_budget
            for child_budget in range(1, remaining - positions_left + 1)
            if self.can_generate(child_sort, child_budget, inner)
            and self.suffix_is_viable(
                op,
                position + 1,
                remaining - child_budget,
                scope,
            )
        )

    def payloads(self, op: Operator, scope: _Scope) -> st.SearchStrategy[int | None]:
        """Return valid payloads for an operator at one generation point."""
        kind = op.payload_sort

        if kind == "none":
            return st.none()

        if kind == "constant":
            size = len(self._store.signature.vocabulary(op.sort))

            return st.integers(min_value=0, max_value=size - 1)

        if kind == "index":
            depth = self._depth(scope, op.sort)
            upper = depth - 1 if self._closed else max(depth, self._max_size - 1)

            return st.integers(min_value=0, max_value=upper)

        store = self._store
        sort = op.sort

        return st.integers(min_value=0, max_value=self._max_size - 1).map(
            lambda at: store.atom(sort, f"x{at}")
        )

    def build(
        self,
        op: Operator,
        children: list[Term],
        payload: int | None,
    ) -> Term:
        """Intern one generated term."""
        return self._store.op(op, *children, payload=payload)

    def budget_vectors(
        self,
        op: Operator,
        position: int,
        remaining: int,
        scope: _Scope,
    ) -> st.SearchStrategy[tuple[int, ...]]:
        """Generate viable argument-budget vectors."""
        if position == op.arity:
            return st.just(())

        choices = self.budgets(op, position, remaining, scope)

        return st.sampled_from(choices).flatmap(
            lambda chosen: self.budget_vectors(
                op,
                position + 1,
                remaining - chosen,
                scope,
            ).map(lambda rest: (chosen, *rest))
        )

    def _applied(
        self,
        op: Operator,
        budget: int,
        scope: _Scope,
    ) -> st.SearchStrategy[Term]:
        """Generate applications of one operator."""
        return self.budget_vectors(op, 0, budget - 1, scope).flatmap(
            lambda budgets: st.tuples(
                *(
                    self.draw(child_sort, child_budget, self._under(op, at, scope))
                    for at, (child_sort, child_budget) in enumerate(
                        zip(op.arg_sorts, budgets, strict=True)
                    )
                )
            ).flatmap(
                lambda children: self.payloads(op, scope).map(
                    lambda payload: self.build(op, list(children), payload)
                )
            )
        )

    def draw(
        self,
        wanted: Sort,
        budget: int,
        scope: _Scope,
    ) -> st.SearchStrategy[Term]:
        """Generate a term under a size and binding budget."""
        return st.sampled_from(self.operators(wanted, budget, scope)).flatmap(
            lambda op: self._applied(op, budget, scope)
        )


def terms(
    store: TermStore,
    *,
    sort: Sort,
    max_size: int,
    closed: bool,
) -> st.SearchStrategy[Term]:
    """Generate well-sorted terms from a store's signature.

    A constant is drawn from its sort's vocabulary, and a free variable from
    the names ``x0``, ``x1`` and so on, one fewer than the size bound.

    Parameters
    ----------
    store : TermStore
        The store the terms are interned in.
    sort : Sort
        The sort of the generated terms.
    max_size : int
        The most nodes a generated term holds.
    closed : bool
        Whether every bound index is bound by an enclosing binding site.

    Raises
    ------
    ValueError
        If ``sort`` is undeclared or ``max_size`` is less than one.
    """
    if sort not in store.signature.sorts:
        msg = f"sort {sort!r} is not declared by {store.signature.name!r}"

        raise ValueError(msg)

    if max_size < 1:
        msg = f"max_size must be positive, not {max_size!r}"

        raise ValueError(msg)

    builder = _TermStrategies(store, max_size=max_size, closed=closed)
    scope = (0,) * len(store.signature.bound_sorts)

    if not builder.can_generate(sort, max_size, scope):
        return st.nothing()

    return builder.draw(sort, max_size, scope)


def _variant[S](variant: type[S]) -> st.SearchStrategy[S]:
    """Generate one structurally described model variant."""
    fields = cast("_FieldRegistry", variant).__field_specs__
    strategies = {
        name: st.from_type(field.annotation)
        for name, field in fields.items()
        if field.usage_mode == "readwrite"
    }

    return st.builds(variant, **strategies)


def spec_of[S](root: type[S]) -> st.SearchStrategy[S]:
    """Generate every variant registered under a tagged-union root."""
    variants = cast("_VariantRegistry[S]", root).__variants__.values()

    return st.one_of(*(_variant(variant) for variant in variants))
