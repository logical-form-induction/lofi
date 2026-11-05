"""Named presentation of locally nameless terms.

A named term gives each binding site of an operator a variable name, and
names each constant and free variable by its name rather than by its payload.
Converting a closed term to its named form chooses canonical names for the
binding sites from an inventory, and converting a named term back closes each
named variable into the bound indices of its site.
"""

from typing import TYPE_CHECKING, Never, override

from lofi._checks import ensure_name
from lofi._native import (
    ArityError,
    BindingAtomCountError,
    ForeignTermError,
    NotLocallyClosedError,
    abstract,
    open_binder,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lofi._native import Term, TermStore
    from lofi.terms._values import Operator

__all__ = ["Nominal", "render", "to_core", "to_nominal"]


class Nominal:
    """A frozen named term node.

    Parameters
    ----------
    op : Operator
        The root operator.
    args : Sequence[Nominal]
        The arguments in operator order.
    payload : str | None
        The name of a constant or free variable, and None for any other
        operator.
    binders : Sequence[str]
        One variable name per binding site of the operator, in site order.

    Raises
    ------
    ArityError
        If the argument count differs from the operator's arity.
    BindingAtomCountError
        If the binder names do not number one per binding site.
    TypeError
        If a name is not a string.
    ValueError
        If a name is empty, two sites over a shared argument share a name, or
        the payload does not suit the operator.
    """

    __slots__ = ("_args", "_binders", "_hash", "_op", "_payload")

    def __init__(
        self,
        op: Operator,
        args: Sequence[Nominal] = (),
        *,
        payload: str | None = None,
        binders: Sequence[str] = (),
    ) -> None:
        held_args = tuple(args)

        if len(held_args) != op.arity:
            raise ArityError(
                operator=op.name,
                expected=op.arity,
                received=len(held_args),
            )

        held_binders = tuple(
            ensure_name(f"binders[{at}]", name) for at, name in enumerate(binders)
        )

        if len(held_binders) != len(op.binding):
            raise BindingAtomCountError(
                operator=op.name,
                expected=len(op.binding),
                received=len(held_binders),
            )

        for position in range(op.arity):
            names = [
                name
                for site, name in zip(op.binding, held_binders, strict=True)
                if position in site.arguments
            ]

            if len(set(names)) != len(names):
                msg = f"two binding sites over argument {position} share a name"
                raise ValueError(msg)

        if op.is_bound_variable:
            msg = f"nominal operator {op.name!r} cannot hold a de Bruijn index"
            raise ValueError(msg)

        if op.payload_sort == "none" and payload is not None:
            msg = f"operator {op.name!r} takes no payload"
            raise ValueError(msg)

        if op.payload_sort != "none" and payload is None:
            msg = f"operator {op.name!r} requires a named payload"
            raise ValueError(msg)

        held_payload = None if payload is None else ensure_name("payload", payload)
        object.__setattr__(self, "_op", op)
        object.__setattr__(self, "_args", held_args)
        object.__setattr__(self, "_payload", held_payload)
        object.__setattr__(self, "_binders", held_binders)
        object.__setattr__(
            self,
            "_hash",
            hash((op, held_args, held_payload, held_binders)),
        )

    @property
    def op(self) -> Operator:
        """Return the root operator."""
        return self._op

    @property
    def args(self) -> tuple[Nominal, ...]:
        """Return arguments in operator order."""
        return self._args

    @property
    def payload(self) -> str | None:
        """Return the constant or free-variable name when present."""
        return self._payload

    @property
    def binders(self) -> tuple[str, ...]:
        """Return one variable name per binding site, in site order."""
        return self._binders

    def scoped(self, position: int, /) -> tuple[str, ...]:
        """Return the names of the variables scoping over one argument.

        Parameters
        ----------
        position : int
            The argument position.
        """
        return tuple(
            name
            for site, name in zip(self._op.binding, self._binders, strict=True)
            if position in site.arguments
        )

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        """Refuse field reassignment."""
        del name, value
        msg = "a nominal term is immutable"

        raise AttributeError(msg)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another node has the same named structure."""
        return isinstance(other, Nominal) and (
            self._op,
            self._args,
            self._payload,
            self._binders,
        ) == (
            other._op,
            other._args,
            other._payload,
            other._binders,
        )

    @override
    def __hash__(self) -> int:
        """Return the structural hash fixed at construction."""
        return self._hash

    @override
    def __repr__(self) -> str:
        """Return the generic named rendering."""
        return _render_nominal(self)


def _inventory(names: Sequence[str]) -> tuple[str, ...]:
    """Validate and freeze a variable-name inventory."""
    held = tuple(
        ensure_name(f"names[{position}]", name) for position, name in enumerate(names)
    )

    if len(set(held)) != len(held):
        msg = "name inventory repeats a variable name"
        raise ValueError(msg)

    return held


def _free_names(
    store: TermStore,
    term: Term,
    bound: frozenset[int],
) -> frozenset[str]:
    """Collect the names of the free variables of a core term outside a set."""
    found: set[str] = set()

    def visit(node: Term) -> None:
        """Add free names below one node."""
        if (
            node.op.is_free_variable
            and node.payload is not None
            and node.payload not in bound
        ):
            found.add(store.atom_name(node.payload))

        for child in node.args:
            visit(child)

    visit(term)

    return frozenset(found)


def _payload_name(
    store: TermStore,
    term: Term,
    opened: Mapping[int, str],
) -> str | None:
    """Resolve a constant or free-variable payload to its name."""
    payload = term.payload
    kind = term.op.payload_sort

    if payload is None:
        return None

    if kind == "free":
        bound = opened.get(payload)

        if bound is not None:
            return bound

        return store.atom_name(payload)

    if kind == "constant":
        return store.signature.vocabulary(term.op.sort).name(payload)

    raise NotLocallyClosedError(max_index=term.maxidx, depth=0)


def _to_nominal(
    store: TermStore,
    term: Term,
    inventory: tuple[str, ...],
    opened: Mapping[int, str],
) -> Nominal:
    """Name one core subtree under the active bindings."""
    current, atoms = open_binder(store, term) if term.op.is_binder else (term, ())
    sites = current.op.binding
    bound = frozenset((*opened, *atoms))
    chosen: list[str] = []

    for site in sites:
        unavailable = frozenset(opened.values()).union(
            *(
                _free_names(store, current.args[position], bound)
                for position in site.arguments
            ),
            (
                name
                for earlier, name in zip(sites, chosen, strict=False)
                if set(earlier.arguments) & set(site.arguments)
            ),
        )
        chosen.extend(_choose(inventory, unavailable, 1))

    arguments: list[Nominal] = []

    for position, child in enumerate(current.args):
        child_bindings = dict(opened)
        child_bindings.update(
            (atom, name)
            for site, atom, name in zip(sites, atoms, chosen, strict=True)
            if position in site.arguments
        )
        arguments.append(_to_nominal(store, child, inventory, child_bindings))

    return Nominal(
        current.op,
        arguments,
        payload=_payload_name(store, current, opened),
        binders=chosen,
    )


def _choose(
    inventory: tuple[str, ...],
    unavailable: frozenset[str],
    count: int,
) -> tuple[str, ...]:
    """Choose the first available names in inventory order."""
    chosen = tuple(name for name in inventory if name not in unavailable)[:count]

    if len(chosen) != count:
        msg = f"name inventory size {len(inventory)} bounds presentable binder nesting"
        raise ValueError(msg)

    return chosen


def to_nominal(
    store: TermStore,
    term: Term,
    names: Sequence[str],
) -> Nominal:
    """Convert a closed core term to its canonical named form.

    Parameters
    ----------
    store : TermStore
        The store the term belongs to.
    term : Term
        The closed core term.
    names : Sequence[str]
        The variable-name inventory, in order of preference.

    Raises
    ------
    ForeignTermError
        If ``term`` does not belong to ``store``.
    NotLocallyClosedError
        If ``term`` has an open bound index.
    TypeError
        If an entry in ``names`` is not a string.
    ValueError
        If ``names`` contains an empty or repeated name or cannot name every binder.
    """
    if not store.contains(term):
        raise ForeignTermError(
            operator=term.op.name,
            position=0,
            expected=store.generation,
            received=term.generation,
        )

    if not term.is_closed:
        raise NotLocallyClosedError(max_index=term.maxidx, depth=0)

    return _to_nominal(store, term, _inventory(names), {})


def _to_core(store: TermStore, nominal: Nominal) -> Term:
    """Close one named subtree into the core representation."""
    op = nominal.op
    arguments: list[Term] = []

    for position, child in enumerate(nominal.args):
        held = _to_core(store, child)
        depths: dict[str, int] = {}

        for site, name in zip(op.binding, nominal.binders, strict=True):
            if position not in site.arguments:
                continue

            depth = depths.get(site.sort, 0)
            held = abstract(store, held, store.atom(site.sort, name), depth)
            depths[site.sort] = depth + 1

        arguments.append(held)

    payload = nominal.payload

    if payload is None:
        return store.op(op, *arguments)

    if op.is_constant:
        identifier = store.signature.vocabulary(op.sort).id(payload)

        return store.op(op, *arguments, payload=identifier)

    return store.op(op, *arguments, payload=store.atom(op.sort, payload))


def to_core(store: TermStore, nominal: Nominal) -> Term:
    """Convert a named term to its interned locally nameless form.

    Parameters
    ----------
    store : TermStore
        The store the term is interned in.
    nominal : Nominal
        The named term.
    """
    return _to_core(store, nominal)


def _render_nominal(nominal: Nominal) -> str:
    """Render one named node in generic prefix notation."""
    head = (
        nominal.op.name
        if nominal.payload is None
        else f"{nominal.op.name}({nominal.payload})"
    )

    if not nominal.args:
        return head

    written: list[str] = []

    for position, child in enumerate(nominal.args):
        body = _render_nominal(child)
        names = nominal.scoped(position)

        if names:
            body = f"[{' '.join(names)}]. {body}"

        written.append(body)

    return f"({head} {' '.join(written)})"


def render(
    store: TermStore,
    term: Term,
    names: Sequence[str],
) -> str:
    """Render a core term with canonical variable names.

    Parameters
    ----------
    store : TermStore
        The store the term belongs to.
    term : Term
        The closed core term.
    names : Sequence[str]
        The variable-name inventory, in order of preference.
    """
    return _render_nominal(to_nominal(store, term, names))
