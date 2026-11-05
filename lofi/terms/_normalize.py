"""Lambda-term normalization over a declared calculus."""

from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, final, override

from lofi._checks import ensure_name, ensure_non_negative, ensure_positive
from lofi._frozen import FrozenValue
from lofi._native import (
    NonTerminatingError,
    close_binder,
    instantiate,
    open_binder,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from lofi._native import Term, TermStore
    from lofi.terms._ports import Calculus

__all__ = [
    "ETA_CONTRACTION",
    "ETA_POLICIES",
    "NORMALIZATION_ORDERS",
    "EtaPolicy",
    "Normalization",
    "NormalizationOrder",
    "beta_normal",
    "beta_reducts",
    "eta_contract",
    "normalize",
    "weak_head",
]

APPLICATION_ARITY = 2

type NormalizationOrder = Literal["leftmost-outermost", "weak-head"]
"""The beta-redex traversal a normalization applies."""

NORMALIZATION_ORDERS: Final[tuple[NormalizationOrder, ...]] = (
    "leftmost-outermost",
    "weak-head",
)
"""The beta-redex traversals a declaration may select."""

type EtaPolicy = Literal["off", "contract"]
"""Whether normalization contracts eta redexes."""

ETA_POLICIES: Final[tuple[EtaPolicy, ...]] = ("off", "contract")
"""The eta treatments a declaration may select."""

ETA_CONTRACTION: Final[Mapping[EtaPolicy, bool]] = MappingProxyType(
    {"off": False, "contract": True}
)
"""Whether each declared eta treatment contracts eta redexes."""


@final
class Normalization(FrozenValue):
    """One compiled normalization treatment and its resource bound.

    Parameters
    ----------
    fuel : int
        Positive beta-reduction step budget.
    order : NormalizationOrder
        Beta-redex traversal.
    eta : EtaPolicy
        Whether to contract eta redexes after beta reduction.
    associative_commutative : tuple[str, ...]
        Binary operator or curried-constant names whose trees are flattened,
        sorted and rebuilt into one canonical association.
    """

    _noun = "normalization settings"

    __slots__ = (
        "_associative_commutative",
        "_contracts_eta",
        "_eta",
        "_fuel",
        "_order",
        "_traversal",
    )

    def __init__(
        self,
        *,
        fuel: int,
        order: NormalizationOrder,
        eta: EtaPolicy,
        associative_commutative: tuple[str, ...],
    ) -> None:
        held = tuple(
            ensure_name("an associative-commutative symbol", name)
            for name in associative_commutative
        )

        if len(set(held)) != len(held):
            msg = "associative-commutative symbols are distinct"
            raise ValueError(msg)

        if order not in NORMALIZATION_ORDERS:
            msg = f"a normalization order is one of {NORMALIZATION_ORDERS!r}: {order!r}"
            raise ValueError(msg)

        if eta not in ETA_POLICIES:
            msg = f"an eta policy is one of {ETA_POLICIES!r}: {eta!r}"
            raise ValueError(msg)

        object.__setattr__(self, "_fuel", ensure_positive("normalization fuel", fuel))
        object.__setattr__(self, "_order", order)
        object.__setattr__(self, "_eta", eta)
        object.__setattr__(self, "_traversal", _TRAVERSALS[order])
        object.__setattr__(self, "_contracts_eta", ETA_CONTRACTION[eta])
        object.__setattr__(self, "_associative_commutative", held)

    @property
    def fuel(self) -> int:
        """Return the beta-reduction step budget."""
        return self._fuel

    @property
    def order(self) -> NormalizationOrder:
        """Return the declared beta-redex traversal."""
        return self._order

    @property
    def eta(self) -> EtaPolicy:
        """Return the declared eta treatment."""
        return self._eta

    def beta_reduce(self, store: TermStore, term: Term, calculus: Calculus, /) -> Term:
        """Return a term's beta normal form under the declared traversal."""
        return _reduce(store, term, calculus, self._fuel, self._traversal)

    @property
    def contracts_eta(self) -> bool:
        """Return whether normalization contracts eta redexes."""
        return self._contracts_eta

    @property
    def associative_commutative(self) -> tuple[str, ...]:
        """Return the symbols canonicalized modulo association and order."""
        return self._associative_commutative

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether every declared setting agrees."""
        return (
            type(other) is Normalization
            and self._fuel == other._fuel
            and self._order == other._order
            and self._eta == other._eta
            and self._associative_commutative == other._associative_commutative
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the complete normalization declaration."""
        return hash(
            (
                Normalization,
                self._fuel,
                self._order,
                self._eta,
                self._associative_commutative,
            )
        )

    @override
    def __repr__(self) -> str:
        """Return the constructor spelling of these settings."""
        return (
            f"Normalization(fuel={self._fuel!r}, order={self._order!r}, "
            f"eta={self._eta!r}, "
            f"associative_commutative={self._associative_commutative!r})"
        )


def _rebuild(
    store: TermStore,
    original: Term,
    arguments: tuple[Term, ...],
) -> Term:
    """Rebuild a term only when an argument changed."""
    if arguments == original.args:
        return original

    return store.op(original.op, *arguments, payload=original.payload)


def _is_abstraction(term: Term, calculus: Calculus) -> bool:
    """Return whether a term abstracts one variable over one body.

    The body is the one argument the abstraction's one binding site scopes
    over. An annotation position, when present, is not the body.

    Parameters
    ----------
    term : Term
        The candidate abstraction.
    calculus : Calculus
        The abstraction and application operators.
    """
    op = term.op

    return (
        op.name == calculus.lam
        and len(op.binding) == 1
        and len(op.binding[0].arguments) == 1
    )


def _contract(store: TermStore, redex: Term) -> Term:
    """Return the contractum of a beta redex.

    Parameters
    ----------
    store : TermStore
        The store the contractum is interned in.
    redex : Term
        An application of an abstraction to a value.
    """
    abstraction, value = redex.args
    (site,) = abstraction.op.binding

    return instantiate(store, abstraction.args[site.arguments[0]], value)


def _redex(term: Term, calculus: Calculus) -> bool:
    """Return whether a term is a beta redex."""
    return (
        term.op.name == calculus.app
        and len(term.args) == APPLICATION_ARITY
        and _is_abstraction(term.args[0], calculus)
    )


def beta_reducts(
    store: TermStore,
    term: Term,
    calculus: Calculus,
    /,
) -> tuple[Term, ...]:
    """Enumerate one-step beta reducts in root-first, left-to-right order."""
    reducts: list[Term] = []

    if _redex(term, calculus):
        reducts.append(_contract(store, term))

    if term.op.is_binder:
        opened, atoms = open_binder(store, term)

        for position, argument in enumerate(opened.args):
            for reduced in beta_reducts(store, argument, calculus):
                arguments = (
                    *opened.args[:position],
                    reduced,
                    *opened.args[position + 1 :],
                )
                rebuilt = _rebuild(store, opened, arguments)
                reducts.append(close_binder(store, rebuilt, atoms))

        return tuple(dict.fromkeys(reducts))

    for position, argument in enumerate(term.args):
        for reduced in beta_reducts(store, argument, calculus):
            arguments = (
                *term.args[:position],
                reduced,
                *term.args[position + 1 :],
            )
            reducts.append(_rebuild(store, term, arguments))

    return tuple(dict.fromkeys(reducts))


def _has_redex(term: Term, calculus: Calculus) -> bool:
    """Return whether a term contains a beta redex."""
    return _redex(term, calculus) or any(
        _has_redex(argument, calculus) for argument in term.args
    )


def _has_weak_redex(term: Term, calculus: Calculus) -> bool:
    """Return whether a term has a weak-head beta step."""
    if _redex(term, calculus):
        return True

    return (
        term.op.name == calculus.app
        and len(term.args) == APPLICATION_ARITY
        and _has_weak_redex(term.args[0], calculus)
    )


def _normal_step(
    store: TermStore,
    term: Term,
    calculus: Calculus,
) -> tuple[Term, bool]:
    """Contract the leftmost outermost beta redex."""
    if _redex(term, calculus):
        return _contract(store, term), True

    if term.op.is_binder:
        opened, atoms = open_binder(store, term)

        for position, argument in enumerate(opened.args):
            reduced, changed = _normal_step(store, argument, calculus)

            if changed:
                arguments = (
                    *opened.args[:position],
                    reduced,
                    *opened.args[position + 1 :],
                )
                rebuilt = _rebuild(store, opened, arguments)

                return close_binder(store, rebuilt, atoms), True

        return term, False

    for position, argument in enumerate(term.args):
        reduced, changed = _normal_step(store, argument, calculus)

        if changed:
            arguments = (*term.args[:position], reduced, *term.args[position + 1 :])

            return _rebuild(store, term, arguments), True

    return term, False


def _weak_step(
    store: TermStore,
    term: Term,
    calculus: Calculus,
) -> tuple[Term, bool]:
    """Contract the next weak-head beta redex."""
    if _redex(term, calculus):
        return _contract(store, term), True

    if term.op.name != calculus.app or len(term.args) != APPLICATION_ARITY:
        return term, False

    function, value = term.args
    reduced, changed = _weak_step(store, function, calculus)

    if not changed:
        return term, False

    return _rebuild(store, term, (reduced, value)), True


type _Traversal = tuple[
    str,
    Callable[[TermStore, Term, Calculus], tuple[Term, bool]],
    Callable[[Term, Calculus], bool],
]
"""A beta traversal's name, one-step contraction and pending-redex test."""


def _reduce(
    store: TermStore,
    term: Term,
    calculus: Calculus,
    fuel: int,
    plan: _Traversal,
) -> Term:
    """Apply beta steps within a reduction budget."""
    operation, step, pending = plan
    budget = ensure_non_negative("fuel", fuel)
    remaining = budget
    current = term

    while remaining:
        reduced, changed = step(store, current, calculus)

        if not changed:
            return current

        current = reduced
        remaining -= 1

    if pending(current, calculus):
        raise NonTerminatingError(operation=operation, budget=budget)

    return current


_TRAVERSALS: Final[Mapping[NormalizationOrder, _Traversal]] = MappingProxyType(
    {
        "leftmost-outermost": ("beta normalization", _normal_step, _has_redex),
        "weak-head": ("weak-head beta normalization", _weak_step, _has_weak_redex),
    }
)
"""The traversal each declared normalization order selects."""


def beta_normal(
    store: TermStore,
    term: Term,
    calculus: Calculus,
    *,
    fuel: int,
) -> Term:
    """Return a term's leftmost outermost beta normal form."""
    return _reduce(store, term, calculus, fuel, _TRAVERSALS["leftmost-outermost"])


def weak_head(
    store: TermStore,
    term: Term,
    calculus: Calculus,
    *,
    fuel: int,
) -> Term:
    """Return a term's weak-head beta normal form."""
    return _reduce(store, term, calculus, fuel, _TRAVERSALS["weak-head"])


def _eta(
    store: TermStore,
    term: Term,
    calculus: Calculus,
) -> Term:
    """Contract eta redexes below one term."""
    if term.op.is_binder:
        opened, atoms = open_binder(store, term)
        arguments = tuple(_eta(store, argument, calculus) for argument in opened.args)
        rebuilt = _rebuild(store, opened, arguments)

        if (
            term.op.name == calculus.lam
            and len(atoms) == 1
            and len(term.op.binding) == 1
        ):
            (site,) = term.op.binding

            if len(site.arguments) == 1:
                body = arguments[site.arguments[0]]

                if body.op.name == calculus.app and len(body.args) == APPLICATION_ARITY:
                    function, value = body.args
                    atom = atoms[0]

                    if (
                        value.op.is_free_variable
                        and value.payload == atom
                        and not function.free & (1 << atom)
                    ):
                        return function

        return close_binder(store, rebuilt, atoms)

    arguments = tuple(_eta(store, argument, calculus) for argument in term.args)

    return _rebuild(store, term, arguments)


def eta_contract(
    store: TermStore,
    term: Term,
    calculus: Calculus,
    /,
) -> Term:
    """Contract every eta redex explicitly."""
    return _eta(store, term, calculus)


type _AssociativeHead = tuple[Literal["operator", "constant"], str, Term]


def _associative_head(
    store: TermStore,
    term: Term,
    calculus: Calculus,
    declared: frozenset[str],
) -> _AssociativeHead | None:
    """Return the declared binary head of a direct or curried application."""
    if term.op.name in declared and len(term.args) == 2:
        return "operator", term.op.name, term

    if term.op.name != calculus.app or len(term.args) != APPLICATION_ARITY:
        return None

    partial, _ = term.args

    if partial.op.name != calculus.app or len(partial.args) != APPLICATION_ARITY:
        return None

    head, _ = partial.args

    if not head.op.is_constant or head.payload is None:
        return None

    name = store.signature.vocabulary(head.op.sort).name(head.payload)

    if name not in declared:
        return None

    return "constant", name, head


def _associative_arguments(
    store: TermStore,
    term: Term,
    calculus: Calculus,
    head: _AssociativeHead,
    declared: frozenset[str],
) -> tuple[Term, ...]:
    """Flatten one associative tree under the same declared head."""
    found = _associative_head(store, term, calculus, declared)

    if found is None or found[:2] != head[:2]:
        return (term,)

    if found[0] == "operator":
        left, right = term.args
    else:
        partial, right = term.args
        _, left = partial.args

    return (
        *_associative_arguments(store, left, calculus, head, declared),
        *_associative_arguments(store, right, calculus, head, declared),
    )


def _rebuild_associative(
    store: TermStore,
    head: _AssociativeHead,
    arguments: tuple[Term, ...],
    calculus: Calculus,
) -> Term:
    """Build the canonical left association of sorted arguments."""
    current = arguments[0]

    for argument in arguments[1:]:
        if head[0] == "operator":
            current = store.op(head[2].op, current, argument, payload=head[2].payload)
        else:
            application = store.signature.by_name(calculus.app)
            current = store.op(
                application,
                store.op(application, head[2], current),
                argument,
            )

    return current


def _associative_commutative(
    store: TermStore,
    term: Term,
    calculus: Calculus,
    declared: frozenset[str],
) -> Term:
    """Canonicalize declared associative-commutative trees recursively."""
    arguments = tuple(
        _associative_commutative(store, argument, calculus, declared)
        for argument in term.args
    )
    rebuilt = _rebuild(store, term, arguments)
    head = _associative_head(store, rebuilt, calculus, declared)

    if head is None:
        return rebuilt

    flattened = _associative_arguments(store, rebuilt, calculus, head, declared)
    ordered = tuple(sorted(flattened, key=lambda argument: argument.key()))

    return _rebuild_associative(store, head, ordered, calculus)


def normalize(
    store: TermStore,
    term: Term,
    calculus: Calculus,
    /,
    *,
    normalization: Normalization,
) -> Term:
    """Return the term's declared beta, eta and AC normal form."""
    reduced = normalization.beta_reduce(store, term, calculus)

    if normalization.contracts_eta:
        reduced = eta_contract(store, reduced, calculus)

    if normalization.associative_commutative:
        reduced = _associative_commutative(
            store,
            reduced,
            calculus,
            frozenset(normalization.associative_commutative),
        )

    return reduced
