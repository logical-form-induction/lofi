"""Declarative matching and rebuilding for interned terms."""

from types import MappingProxyType
from typing import TYPE_CHECKING, final, override

from lofi._checks import ensure_name, ensure_non_negative
from lofi._frozen import FrozenValue

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi._native import Term, TermStore
    from lofi.terms._ports import OpName, Sort
    from lofi.terms._values import Operator

__all__ = ["Bindings", "MVar", "Node", "PVar", "Pattern", "Spine", "build", "match"]


@final
class MVar(FrozenValue):
    """A metavariable for one term of a declared sort."""

    _noun = "pattern value"

    __slots__ = ("_name", "_sort")

    def __init__(self, name: str, sort: Sort = "Ty") -> None:
        object.__setattr__(self, "_name", ensure_name("name", name))
        object.__setattr__(self, "_sort", ensure_name("sort", sort))

    @property
    def name(self) -> str:
        """Return the variable name."""
        return self._name

    @property
    def sort(self) -> Sort:
        """Return the required term sort."""
        return self._sort

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another metavariable has the same fields."""
        if type(other) is not MVar:
            return NotImplemented

        return (self._name, self._sort) == (other._name, other._sort)

    @override
    def __hash__(self) -> int:
        """Return a hash over the fields."""
        return hash((MVar, self._name, self._sort))

    @override
    def __repr__(self) -> str:
        """Return the constructor call."""
        return f"MVar({self._name!r}, {self._sort!r})"


@final
class PVar(FrozenValue):
    """A metavariable for an operator name or integer payload."""

    _noun = "pattern value"

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        object.__setattr__(self, "_name", ensure_name("name", name))

    @property
    def name(self) -> str:
        """Return the variable name."""
        return self._name

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another payload variable has the same name."""
        if type(other) is not PVar:
            return NotImplemented

        return self._name == other._name

    @override
    def __hash__(self) -> int:
        """Return a hash over the name."""
        return hash((PVar, self._name))

    @override
    def __repr__(self) -> str:
        """Return the constructor call."""
        return f"PVar({self._name!r})"


@final
class Spine(FrozenValue):
    """A bounded one-hole context pattern."""

    _noun = "pattern value"

    __slots__ = ("_core", "_max_order", "_min_order", "_name", "_ops", "_slot")

    def __init__(
        self,
        name: str,
        core: Pattern,
        ops: tuple[OpName, ...],
        *,
        slot: int = 0,
        min_order: int = 0,
        max_order: int | None,
    ) -> None:
        checked_ops = tuple(ensure_name(f"ops[{at}]", op) for at, op in enumerate(ops))
        checked_slot = ensure_non_negative("slot", slot)
        minimum = ensure_non_negative("min_order", min_order)
        maximum = (
            None if max_order is None else ensure_non_negative("max_order", max_order)
        )

        if maximum is not None and minimum > maximum:
            msg = f"min_order {minimum} exceeds max_order {maximum}"
            raise ValueError(msg)

        object.__setattr__(self, "_name", ensure_name("name", name))
        object.__setattr__(self, "_core", core)
        object.__setattr__(self, "_ops", checked_ops)
        object.__setattr__(self, "_slot", checked_slot)
        object.__setattr__(self, "_min_order", minimum)
        object.__setattr__(self, "_max_order", maximum)

    @property
    def name(self) -> str:
        """Return the context variable name."""
        return self._name

    @property
    def core(self) -> Pattern:
        """Return the pattern below the context."""
        return self._core

    @property
    def ops(self) -> tuple[OpName, ...]:
        """Return the operators that may form the context."""
        return self._ops

    @property
    def slot(self) -> int:
        """Return the argument position followed through each frame."""
        return self._slot

    @property
    def min_order(self) -> int:
        """Return the minimum frame count."""
        return self._min_order

    @property
    def max_order(self) -> int | None:
        """Return the maximum frame count, or no finite maximum."""
        return self._max_order

    def _identity(
        self,
    ) -> tuple[str, Pattern, tuple[OpName, ...], int, int, int | None]:
        """Return fields in comparison order."""
        return (
            self._name,
            self._core,
            self._ops,
            self._slot,
            self._min_order,
            self._max_order,
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another spine has the same fields."""
        if type(other) is not Spine:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the fields."""
        return hash((Spine, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call."""
        return (
            f"Spine({self._name!r}, {self._core!r}, {self._ops!r}, "
            f"slot={self._slot!r}, min_order={self._min_order!r}, "
            f"max_order={self._max_order!r})"
        )


@final
class Node(FrozenValue):
    """A pattern for an operator, arguments, and payload."""

    _noun = "pattern value"

    __slots__ = ("_args", "_op", "_payload")

    def __init__(
        self,
        op: OpName | PVar,
        args: tuple[Pattern, ...] = (),
        payload: int | PVar | None = None,
    ) -> None:
        checked_op = ensure_name("op", op) if isinstance(op, str) else op
        object.__setattr__(self, "_op", checked_op)
        object.__setattr__(self, "_args", tuple(args))
        object.__setattr__(self, "_payload", payload)

    @property
    def op(self) -> OpName | PVar:
        """Return the operator pattern."""
        return self._op

    @property
    def args(self) -> tuple[Pattern, ...]:
        """Return the argument patterns."""
        return self._args

    @property
    def payload(self) -> int | PVar | None:
        """Return the payload pattern."""
        return self._payload

    def _identity(self) -> tuple[OpName | PVar, tuple[Pattern, ...], int | PVar | None]:
        """Return fields in comparison order."""
        return (self._op, self._args, self._payload)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another node has the same fields."""
        if type(other) is not Node:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the fields."""
        return hash((Node, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call."""
        return f"Node({self._op!r}, {self._args!r}, payload={self._payload!r})"


type Pattern = MVar | Node | Spine
"""A term shape without executable predicates."""


@final
class _Frame(FrozenValue):
    """A functor frame around a spine hole."""

    _noun = "pattern value"

    __slots__ = ("_args", "_op", "_payload", "_slot")

    def __init__(
        self,
        op: Operator,
        args: tuple[Term, ...],
        payload: int | None,
        slot: int,
    ) -> None:
        object.__setattr__(self, "_op", op)
        object.__setattr__(self, "_args", args)
        object.__setattr__(self, "_payload", payload)
        object.__setattr__(self, "_slot", slot)

    @property
    def op(self) -> Operator:
        """Return the frame operator."""
        return self._op

    @property
    def args(self) -> tuple[Term, ...]:
        """Return the original arguments."""
        return self._args

    @property
    def payload(self) -> int | None:
        """Return the original payload."""
        return self._payload

    @property
    def slot(self) -> int:
        """Return the hole position."""
        return self._slot

    def _identity(self) -> tuple[Operator, tuple[Term, ...], int | None, int]:
        """Return fields in comparison order."""
        return (self._op, self._args, self._payload, self._slot)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another frame has the same fields."""
        if not isinstance(other, _Frame):
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the fields."""
        return hash((_Frame, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return a field rendering."""
        return (
            f"_Frame(op={self._op!r}, args={self._args!r}, "
            f"payload={self._payload!r}, slot={self._slot!r})"
        )


type _Context = tuple[_Frame, ...]


@final
class Bindings(FrozenValue):
    """Term, payload, and spine bindings."""

    _noun = "pattern value"

    __slots__ = ("_payloads", "_spines", "_terms")

    def __init__(
        self,
        terms: Mapping[str, Term] | None = None,
        payloads: Mapping[str, OpName | int] | None = None,
        spines: Mapping[str, _Context] | None = None,
    ) -> None:
        object.__setattr__(
            self,
            "_terms",
            MappingProxyType({} if terms is None else dict(terms)),
        )
        object.__setattr__(
            self,
            "_payloads",
            MappingProxyType({} if payloads is None else dict(payloads)),
        )
        object.__setattr__(
            self,
            "_spines",
            MappingProxyType({} if spines is None else dict(spines)),
        )

    @property
    def terms(self) -> Mapping[str, Term]:
        """Return term bindings."""
        return self._terms

    @property
    def payloads(self) -> Mapping[str, OpName | int]:
        """Return operator and payload bindings."""
        return self._payloads

    @property
    def spines(self) -> Mapping[str, _Context]:
        """Return spine bindings."""
        return self._spines

    def _identity(
        self,
    ) -> tuple[
        tuple[tuple[str, Term], ...],
        tuple[tuple[str, OpName | int], ...],
        tuple[tuple[str, _Context], ...],
    ]:
        """Return fields in comparison order."""
        return (
            tuple(sorted(self._terms.items())),
            tuple(sorted(self._payloads.items())),
            tuple(sorted(self._spines.items())),
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another environment has the same bindings."""
        if not isinstance(other, Bindings):
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the bindings."""
        return hash((Bindings, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call."""
        return (
            f"Bindings({dict(self._terms)!r}, {dict(self._payloads)!r}, "
            f"{dict(self._spines)!r})"
        )


def _bind_term(bindings: Bindings, name: str, term: Term) -> Bindings | None:
    """Bind a term or check its existing identity binding."""
    held = bindings.terms.get(name)

    if held is not None:
        return bindings if held is term else None

    terms = dict(bindings.terms)
    terms[name] = term

    return Bindings(terms, bindings.payloads, bindings.spines)


def _bind_payload(
    bindings: Bindings, name: str, value: OpName | int
) -> Bindings | None:
    """Bind an operator or payload or check an existing binding."""
    held = bindings.payloads.get(name)

    if held is not None:
        return bindings if held == value else None

    payloads = dict(bindings.payloads)
    payloads[name] = value

    return Bindings(bindings.terms, payloads, bindings.spines)


def _bind_spine(bindings: Bindings, name: str, context: _Context) -> Bindings | None:
    """Bind a spine or check an existing context binding."""
    held = bindings.spines.get(name)

    if held is not None:
        return bindings if held == context else None

    spines = dict(bindings.spines)
    spines[name] = context

    return Bindings(bindings.terms, bindings.payloads, spines)


def _match_operator(
    pattern: OpName | PVar, term: Term, bindings: Bindings
) -> Bindings | None:
    """Match an operator pattern."""
    if isinstance(pattern, PVar):
        return _bind_payload(bindings, pattern.name, term.op.name)

    return bindings if pattern == term.op.name else None


def _match_payload(
    pattern: int | PVar | None, term: Term, bindings: Bindings
) -> Bindings | None:
    """Match a payload pattern."""
    if isinstance(pattern, PVar):
        if term.payload is None:
            return None

        return _bind_payload(bindings, pattern.name, term.payload)

    return bindings if pattern == term.payload else None


def _match_node(pattern: Node, term: Term, bindings: Bindings) -> Bindings | None:
    """Match one node pattern."""
    held = _match_operator(pattern.op, term, bindings)

    if held is None:
        return None

    held = _match_payload(pattern.payload, term, held)

    if held is None or len(pattern.args) != len(term.args):
        return None

    for child_pattern, child in zip(pattern.args, term.args, strict=True):
        updated = _match(child_pattern, child, held)

        if updated is None:
            return None

        held = updated

    return held


def _match_spine(pattern: Spine, term: Term, bindings: Bindings) -> Bindings | None:
    """Match a spine, preferring the longest context that fits."""
    frames: list[_Frame] = []
    remainders = [term]
    current = term

    while (
        (pattern.max_order is None or len(frames) < pattern.max_order)
        and current.op.name in pattern.ops
        and pattern.slot < len(current.args)
    ):
        frames.append(_Frame(current.op, current.args, current.payload, pattern.slot))
        current = current.args[pattern.slot]
        remainders.append(current)

    for order in range(len(frames), pattern.min_order - 1, -1):
        updated = _match(pattern.core, remainders[order], bindings)

        if updated is None:
            continue

        bound = _bind_spine(updated, pattern.name, tuple(frames[:order]))

        if bound is not None:
            return bound

    return None


def _match(pattern: Pattern, term: Term, bindings: Bindings) -> Bindings | None:
    """Match a pattern with an existing environment."""
    if isinstance(pattern, MVar):
        if pattern.sort != term.op.sort:
            return None

        return _bind_term(bindings, pattern.name, term)

    if isinstance(pattern, Node):
        return _match_node(pattern, term, bindings)

    return _match_spine(pattern, term, bindings)


def match(
    pattern: Pattern,
    term: Term,
    /,
    bindings: Bindings | None = None,
) -> Bindings | None:
    """Return extended bindings when a pattern matches a term."""
    return _match(pattern, term, Bindings() if bindings is None else bindings)


def _required[T](values: Mapping[str, T], name: str, kind: str) -> T:
    """Return one binding or raise with its kind and name."""
    try:
        return values[name]
    except KeyError:
        msg = f"unbound {kind} variable {name!r}"
        raise KeyError(msg) from None


def _build_node(pattern: Node, store: TermStore, bindings: Bindings) -> Term:
    """Build one node pattern."""
    if isinstance(pattern.op, PVar):
        head = _required(bindings.payloads, pattern.op.name, "payload")

        if not isinstance(head, str):
            msg = f"operator variable {pattern.op.name!r} is bound to {head!r}"
            raise TypeError(msg)
    else:
        head = pattern.op

    payload: int | None

    if isinstance(pattern.payload, PVar):
        held = _required(bindings.payloads, pattern.payload.name, "payload")

        if isinstance(held, str):
            msg = f"payload variable {pattern.payload.name!r} is bound to {held!r}"
            raise TypeError(msg)

        payload = held
    else:
        payload = pattern.payload

    args = tuple(_build(child, store, bindings) for child in pattern.args)
    op = store.signature.by_name(head)

    return store.op(op, *args, payload=payload)


def _build_spine(pattern: Spine, store: TermStore, bindings: Bindings) -> Term:
    """Fill a bound spine with a built core."""
    context = _required(bindings.spines, pattern.name, "spine")

    if len(context) < pattern.min_order or (
        pattern.max_order is not None and len(context) > pattern.max_order
    ):
        msg = (
            f"spine {pattern.name!r} has order {len(context)}, outside "
            f"[{pattern.min_order}, {pattern.max_order}]"
        )
        raise ValueError(msg)

    built = _build(pattern.core, store, bindings)

    for frame in reversed(context):
        if frame.slot != pattern.slot or frame.op.name not in pattern.ops:
            msg = f"spine {pattern.name!r} is incompatible with its pattern"
            raise ValueError(msg)

        args = list(frame.args)
        args[frame.slot] = built
        built = store.op(frame.op, *args, payload=frame.payload)

    return built


def _build(pattern: Pattern, store: TermStore, bindings: Bindings) -> Term:
    """Build a pattern from an environment."""
    if isinstance(pattern, MVar):
        term = _required(bindings.terms, pattern.name, "term")

        if term.op.sort != pattern.sort:
            msg = (
                f"term variable {pattern.name!r} requires sort {pattern.sort!r}, "
                f"and is bound to {term.op.sort!r}"
            )
            raise ValueError(msg)

        return term

    if isinstance(pattern, Node):
        return _build_node(pattern, store, bindings)

    return _build_spine(pattern, store, bindings)


def build(pattern: Pattern, store: TermStore, bindings: Bindings, /) -> Term:
    """Build and intern a pattern from complete bindings."""
    return _build(pattern, store, bindings)
