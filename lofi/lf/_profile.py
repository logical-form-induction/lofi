"""Logical-form profiles, the fragments they declare, and prover bounds."""

from hashlib import sha256
from json import dumps
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, final, override

from lofi._checks import ensure_name
from lofi._frozen import FrozenValue
from lofi.lf._relation import LabelMap

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from lofi.lf._roles import LogicalRole
    from lofi.lf._semantics import LogicalConstantSemantics
    from lofi.terms import Sort, Term, TermStore

__all__ = [
    "ORDERS",
    "FragmentOrder",
    "LogicFragment",
    "LogicProfile",
    "LogicProfileError",
    "ProverBound",
    "ProverUnsupportedError",
]


type FragmentOrder = str
"""An order of quantification: over individuals, over their relations, or anything."""

ORDERS: Final[tuple[FragmentOrder, ...]] = ("first", "second", "higher")
"""The orders a fragment may declare, from the narrowest to the widest."""

_RANK: Final = MappingProxyType({order: rank for rank, order in enumerate(ORDERS)})


class LogicProfileError(ValueError):
    """Reports an invalid logical-form profile or fragment field."""

    __slots__ = ("field",)

    def __init__(self, field: str, message: str) -> None:
        self.field = field
        super().__init__(message)


class ProverUnsupportedError(ValueError):
    """Reports a profile field unsupported by a prover."""

    __slots__ = ("field",)

    def __init__(self, field: str, message: str) -> None:
        self.field = field
        super().__init__(message)


def _flag(field: str, *, value: bool) -> bool:
    """Return a Boolean field or raise for another value."""
    if not isinstance(value, bool):
        msg = f"{field} is not a bool: {value!r}"
        raise TypeError(msg)

    return value


def _names(field: str, values: Iterable[str]) -> frozenset[str]:
    """Return a checked inventory of distinct declared constant names."""
    held: set[str] = set()

    for value in values:
        try:
            held.add(ensure_name(field, value))
        except TypeError as error:
            raise LogicProfileError(field, str(error)) from error
        except ValueError as error:
            raise LogicProfileError(field, str(error)) from error

    return frozenset(held)


type _FragmentIdentity = tuple[
    FragmentOrder, tuple[str, ...], tuple[str, ...], bool, bool
]


@final
class LogicFragment(FrozenValue):
    """The subset of the logic a run's forms are drawn from.

    A fragment fixes the order of quantification, the quantifiers that may
    bind, the connectives that may join, and whether function symbols and
    equality appear. Constant classes in the language declaration determine
    both name inventories. The empty quantifier inventory is the
    quantifier-free fragment.

    Parameters
    ----------
    order : FragmentOrder
        Whether variables range over individuals only, over their relations
        and functions too, or over anything.
    quantifiers : Iterable[str]
        The declared constant names that may quantify.
    connectives : Iterable[str]
        The connective names that may join formulas.
    functions : bool
        Whether a symbol may take arguments and return an individual.
    equality : bool
        Whether equations appear.

    Raises
    ------
    LogicProfileError
        If the open order, a quantifier, or a connective is not a name, or a
        Boolean capability is malformed.
    """

    _noun = "logic declaration"

    __slots__ = ("_connectives", "_equality", "_functions", "_order", "_quantifiers")

    def __init__(
        self,
        *,
        order: FragmentOrder,
        quantifiers: Iterable[str] = (),
        connectives: Iterable[str] = (),
        functions: bool = True,
        equality: bool = True,
    ) -> None:
        try:
            held_order = ensure_name("order", order)
        except ValueError as error:
            field = "order"
            raise LogicProfileError(field, str(error)) from error

        object.__setattr__(self, "_order", held_order)
        object.__setattr__(self, "_quantifiers", _names("a quantifier", quantifiers))
        object.__setattr__(self, "_connectives", _names("a connective", connectives))
        object.__setattr__(self, "_functions", _flag("functions", value=functions))
        object.__setattr__(self, "_equality", _flag("equality", value=equality))

    @property
    def order(self) -> FragmentOrder:
        """Return the declared order of quantification."""
        return self._order

    @property
    def quantifiers(self) -> frozenset[str]:
        """Return the binder names that may quantify."""
        return self._quantifiers

    @property
    def connectives(self) -> frozenset[str]:
        """Return the connective names that may join formulas."""
        return self._connectives

    @property
    def functions(self) -> bool:
        """Return whether function symbols appear."""
        return self._functions

    @property
    def equality(self) -> bool:
        """Return whether equations appear."""
        return self._equality

    def excess(self, other: LogicFragment, /) -> str | None:
        """Return the first field on which another fragment exceeds this one.

        Parameters
        ----------
        other : LogicFragment
            The fragment tested for inclusion in this one.

        Returns
        -------
        str | None
            The name of the first field, in declaration order, that the other
            fragment declares more of than this one, or None when this
            fragment includes it.
        """
        if other._order != self._order and (
            self._order not in _RANK
            or other._order not in _RANK
            or _RANK[other._order] > _RANK[self._order]
        ):
            return "order"

        if not other._quantifiers <= self._quantifiers:
            return "quantifiers"

        if not other._connectives <= self._connectives:
            return "connectives"

        if other._functions and not self._functions:
            return "functions"

        if other._equality and not self._equality:
            return "equality"

        return None

    def admits(self, other: LogicFragment, /) -> bool:
        """Return whether every form of another fragment is a form of this one.

        Parameters
        ----------
        other : LogicFragment
            The fragment tested for inclusion in this one.
        """
        return self.excess(other) is None

    def _identity(self) -> _FragmentIdentity:
        """Return fields in comparison order."""
        return (
            self._order,
            tuple(sorted(self._quantifiers)),
            tuple(sorted(self._connectives)),
            self._functions,
            self._equality,
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another fragment has the same five fields."""
        return type(other) is LogicFragment and self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the five fields."""
        return hash((LogicFragment, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this fragment."""
        return (
            f"LogicFragment(order={self._order!r}, "
            f"quantifiers={sorted(self._quantifiers)!r}, "
            f"connectives={sorted(self._connectives)!r}, "
            f"functions={self._functions!r}, equality={self._equality!r})"
        )


def _domains(values: Mapping[Sort, int]) -> Mapping[Sort, int]:
    """Return checked finite domain sizes as a read-only mapping."""
    held: dict[Sort, int] = {}
    field = "domains"

    for sort, size in values.items():
        try:
            name = ensure_name("domains sort", sort)
        except TypeError as error:
            raise LogicProfileError(field, str(error)) from error
        except ValueError as error:
            raise LogicProfileError(field, str(error)) from error

        if isinstance(size, bool) or not isinstance(size, int):
            msg = f"domains[{name!r}] is not an integer: {size!r}"
            raise LogicProfileError(field, msg)

        if size < 1:
            msg = f"domains[{name!r}] is below one: {size!r}"
            raise LogicProfileError(field, msg)

        held[name] = size

    return MappingProxyType(held)


type _TermIdentity = tuple[str, str | int | None, tuple[_TermIdentity, ...]]
"""A term key with atom and symbol names restored."""


def _term_identity(store: TermStore, term: Term) -> _TermIdentity:
    """Return a store-independent identity for a term."""
    payload: str | int | None = term.payload

    if payload is not None and (term.op.is_free_variable or term.op.is_constant):
        try:
            payload = store.payload_name(term)
        except ValueError as error:
            field = "theory"
            msg = f"a theory term uses an unregistered atom index {payload}"
            raise LogicProfileError(field, msg) from error

    return (
        term.op.name,
        payload,
        tuple(_term_identity(store, arg) for arg in term.args),
    )


def _theory(
    store: TermStore, formula_sort: Sort, values: Sequence[Term]
) -> tuple[tuple[Term, ...], tuple[_TermIdentity, ...]]:
    """Return closed formula terms and their stable identities."""
    held = tuple(values)
    signature = store.signature

    if held and not store.active:
        field = "theory"
        msg = "theory terms require an active store"
        raise LogicProfileError(field, msg)

    if formula_sort not in signature.sorts:
        field = "theory"
        msg = f"a theory signature must declare formula sort {formula_sort!r}"
        raise LogicProfileError(field, msg)

    for at, term in enumerate(held):
        if not store.contains(term):
            field = "theory"
            msg = f"theory[{at}] is outside the supplied store lineage"
            raise LogicProfileError(field, msg)

        if not term.is_closed or term.free:
            field = "theory"
            msg = f"theory[{at}] contains an open or free variable"
            raise LogicProfileError(field, msg)

        if term.op.sort != formula_sort:
            field = "theory"
            msg = (
                f"theory[{at}] has sort {term.op.sort!r}, "
                f"not formula sort {formula_sort!r}"
            )
            raise LogicProfileError(field, msg)

    return held, tuple(_term_identity(store, term) for term in held)


def _definitions(
    store: TermStore,
    formula_sort: Sort,
    values: Mapping[str, Term],
) -> tuple[
    Mapping[str, Term],
    tuple[tuple[str, _TermIdentity], ...],
]:
    """Return closed logical definitions and their stable identities."""
    held: dict[str, Term] = {}
    identity: list[tuple[str, _TermIdentity]] = []

    for name, term in sorted(values.items()):
        named = ensure_name("defined logical constant", name)

        if not store.contains(term):
            field = "definitions"
            msg = f"definition for {named!r} is outside the supplied store lineage"
            raise LogicProfileError(field, msg)

        if not term.is_closed or term.free:
            field = "definitions"
            msg = f"definition for {named!r} is open or contains a free variable"
            raise LogicProfileError(field, msg)

        if term.op.sort != formula_sort:
            field = "definitions"
            msg = (
                f"definition for {named!r} has sort {term.op.sort!r}, "
                f"not formula sort {formula_sort!r}"
            )
            raise LogicProfileError(field, msg)

        held[named] = term
        identity.append((named, _term_identity(store, term)))

    return MappingProxyType(held), tuple(identity)


type _ProfileIdentity = tuple[
    LogicFragment,
    tuple[tuple[str, str], ...],
    tuple[str, ...],
    str,
    str | None,
    LabelMap,
    str,
    tuple[tuple[str, tuple[str, ...]], ...],
    tuple[_TermIdentity, ...],
    tuple[tuple[str, _TermIdentity], ...],
    tuple[tuple[Sort, int], ...],
    tuple[tuple[str, str], ...],
    tuple[tuple[str, str], ...],
]


def _digest(identity: _ProfileIdentity) -> str:
    """Return a stable digest of the native declaration."""
    (
        fragment,
        type_kinds,
        variables,
        semantics_name,
        negation,
        label_map,
        application,
        classes,
        theory,
        definitions,
        domains,
        roles,
        semantics,
    ) = identity
    value = {
        "fragment": {
            "order": fragment.order,
            "quantifiers": sorted(fragment.quantifiers),
            "connectives": sorted(fragment.connectives),
            "functions": fragment.functions,
            "equality": fragment.equality,
        },
        "type_kinds": type_kinds,
        "variables": variables,
        "semantics_name": semantics_name,
        "negation": negation,
        "label_map": label_map.name,
        "application": application,
        "classes": classes,
        "theory": theory,
        "definitions": definitions,
        "domains": domains,
        "roles": roles,
        "logical_semantics": semantics,
    }
    encoded = dumps(value, separators=(",", ":"), sort_keys=True)

    return sha256(encoded.encode()).hexdigest()


@final
class LogicProfile(FrozenValue):
    """A logical-form language declaration.

    Parameters
    ----------
    store : TermStore
        The store the theory is interned in.
    fragment : LogicFragment
        The subset of the logic the language's forms are drawn from.
    type_kinds : Mapping[str, str]
        The open logical-resource kind of each declared atomic type.
    variables : Sequence[str]
        The finite nominal inventory used to present and generate binders.
    theory : Sequence[Term]
        The closed meaning postulates every decision assumes.
    domains : Mapping[Sort, int]
        Finite domain sizes by sort.
    roles : Mapping[str, LogicalRole]
        Logical roles resolved for the language's typed constants.

    Raises
    ------
    LogicProfileError
        If a logical resource is malformed, a theory term is open, free,
        foreign or not a formula, or a domain size is below one.
    """

    _noun = "logic declaration"

    __slots__ = (
        "_application",
        "_classes",
        "_definition_identity",
        "_definitions",
        "_digest",
        "_domains",
        "_fragment",
        "_label_map",
        "_negation",
        "_roles",
        "_semantics",
        "_semantics_name",
        "_theory",
        "_theory_identity",
        "_type_kinds",
        "_variables",
    )

    def __init__(
        self,
        store: TermStore,
        *,
        formula_sort: Sort,
        application: str,
        classes: Mapping[str, frozenset[str]],
        fragment: LogicFragment,
        type_kinds: Mapping[str, str],
        variables: Sequence[str],
        semantics_name: str,
        negation: str | None,
        label_map: LabelMap,
        domains: Mapping[Sort, int],
        roles: Mapping[str, LogicalRole],
        theory: Sequence[Term] = (),
        definitions: Mapping[str, Term] | None = None,
        semantics: Mapping[str, LogicalConstantSemantics] | None = None,
    ) -> None:
        held_kinds = MappingProxyType(
            {
                ensure_name("an atomic type", name): ensure_name(
                    f"atomic type {name!r} resource kind", kind
                )
                for name, kind in type_kinds.items()
            }
        )
        held_variables = tuple(
            ensure_name("a logical variable", variable) for variable in variables
        )
        if len(set(held_variables)) != len(held_variables):
            field = "variables"
            msg = "the logical variable inventory repeats a name"
            raise LogicProfileError(field, msg)
        held_theory, theory_identity = _theory(store, formula_sort, theory)
        held_definitions, definition_identity = _definitions(
            store,
            formula_sort,
            {} if definitions is None else definitions,
        )
        held_domains = _domains(domains)
        object.__setattr__(self, "_fragment", fragment)
        object.__setattr__(self, "_type_kinds", held_kinds)
        object.__setattr__(self, "_variables", held_variables)
        object.__setattr__(
            self, "_semantics_name", ensure_name("logical semantics", semantics_name)
        )
        object.__setattr__(
            self,
            "_negation",
            None if negation is None else ensure_name("negation", negation),
        )
        object.__setattr__(self, "_label_map", label_map)
        object.__setattr__(
            self,
            "_application",
            ensure_name("application", application),
        )
        object.__setattr__(
            self,
            "_classes",
            MappingProxyType(
                {
                    ensure_name("constant", name): frozenset(
                        ensure_name("constant class", held) for held in values
                    )
                    for name, values in classes.items()
                }
            ),
        )
        object.__setattr__(self, "_theory", held_theory)
        object.__setattr__(self, "_theory_identity", theory_identity)
        object.__setattr__(self, "_definitions", held_definitions)
        object.__setattr__(self, "_definition_identity", definition_identity)
        object.__setattr__(self, "_domains", held_domains)
        object.__setattr__(self, "_roles", MappingProxyType(dict(roles)))
        object.__setattr__(
            self,
            "_semantics",
            MappingProxyType({} if semantics is None else dict(semantics)),
        )
        object.__setattr__(self, "_digest", _digest(self._identity()))

    @property
    def fragment(self) -> LogicFragment:
        """Return the declared fragment."""
        return self._fragment

    @property
    def type_kinds(self) -> Mapping[str, str]:
        """Return the open resource kind of each atomic type."""
        return self._type_kinds

    @property
    def world_types(self) -> frozenset[str]:
        """Return atomic types explicitly declared as worlds."""
        return frozenset(
            name for name, kind in self._type_kinds.items() if kind == "world"
        )

    @property
    def proposition_types(self) -> frozenset[str]:
        """Return atomic types explicitly declared as propositions."""
        return frozenset(
            name for name, kind in self._type_kinds.items() if kind == "proposition"
        )

    @property
    def index_types(self) -> frozenset[str]:
        """Return atomic types explicitly declared as indices."""
        return frozenset(
            name for name, kind in self._type_kinds.items() if kind == "index"
        )

    @property
    def variables(self) -> tuple[str, ...]:
        """Return the finite nominal variable inventory."""
        return self._variables

    @property
    def semantics(self) -> str:
        """Return the declared logical semantics name."""
        return self._semantics_name

    @property
    def negation(self) -> str | None:
        """Return the designated negation constant."""
        return self._negation

    @property
    def label_map(self) -> LabelMap:
        """Return the declared map from labels to inference cells."""
        return self._label_map

    @property
    def theory(self) -> tuple[Term, ...]:
        """Return the closed meaning postulates."""
        return self._theory

    @property
    def definitions(self) -> Mapping[str, Term]:
        """Return closed definitions keyed by logical constant name."""
        return self._definitions

    @property
    def domains(self) -> Mapping[Sort, int]:
        """Return finite domain sizes by sort."""
        return self._domains

    @property
    def application(self) -> str:
        """Return the structural application operator."""
        return self._application

    @property
    def constant_classes(self) -> Mapping[str, frozenset[str]]:
        """Return logical classes keyed by typed constant name."""
        return self._classes

    @property
    def logical_roles(self) -> Mapping[str, LogicalRole]:
        """Return resolved logical roles keyed by constant name."""
        return self._roles

    @property
    def logical_semantics(self) -> Mapping[str, LogicalConstantSemantics]:
        """Return postulated semantics plugins keyed by logical constant name."""
        return self._semantics

    @property
    def logical_constants(self) -> frozenset[str]:
        """Return every role-based, defined or postulated logical constant."""
        return (
            frozenset(self._roles)
            | frozenset(self._definitions)
            | frozenset(self._semantics)
        )

    @property
    def digest(self) -> str:
        """Return the declaration digest as lowercase hexadecimal."""
        return self._digest

    def _identity(self) -> _ProfileIdentity:
        """Return fields in comparison order."""
        return (
            self._fragment,
            tuple(sorted(self._type_kinds.items())),
            self._variables,
            self._semantics_name,
            self._negation,
            self._label_map,
            self._application,
            tuple(
                (name, tuple(sorted(classes)))
                for name, classes in sorted(self._classes.items())
            ),
            self._theory_identity,
            self._definition_identity,
            tuple(sorted(self._domains.items())),
            tuple(
                (constant, role.name) for constant, role in sorted(self._roles.items())
            ),
            tuple(
                (constant, semantics.name)
                for constant, semantics in sorted(self._semantics.items())
            ),
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another profile has the same declaration."""
        return type(other) is LogicProfile and self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the complete logical profile."""
        return hash((LogicProfile, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the profile fields most useful for diagnosis."""
        return (
            f"LogicProfile(fragment={self._fragment!r}, "
            f"type_kinds={dict(self._type_kinds)!r}, variables={self._variables!r}, "
            f"semantics={self._semantics_name!r}, negation={self._negation!r}, "
            f"label_map={self._label_map.name!r}, "
            f"theory={self._theory!r}, domains={dict(self._domains)!r})"
        )


@final
class ProverBound(FrozenValue):
    """The widest fragment and world resources a prover serves.

    Parameters
    ----------
    prover : str
        The prover name used in refusals.
    fragment : LogicFragment
        The widest fragment served; a profile is served when its fragment
        lies inside this one.
    world_types : Iterable[str]
        Named world types served. ``"*"`` serves every declared world type.
    """

    _noun = "prover bound"

    __slots__ = ("_fragment", "_prover", "_world_types")

    def __init__(
        self,
        prover: str,
        fragment: LogicFragment,
        *,
        world_types: Iterable[str] = (),
    ) -> None:
        object.__setattr__(self, "_prover", ensure_name("prover", prover))
        object.__setattr__(self, "_fragment", fragment)
        object.__setattr__(self, "_world_types", _names("a world type", world_types))

    @property
    def prover(self) -> str:
        """Return the prover name used in refusals."""
        return self._prover

    @property
    def fragment(self) -> LogicFragment:
        """Return the widest fragment served."""
        return self._fragment

    @property
    def world_types(self) -> frozenset[str]:
        """Return the named world types served by the prover."""
        return self._world_types

    def check(self, profile: LogicProfile) -> None:
        """Raise for the first unsupported profile field.

        Parameters
        ----------
        profile : LogicProfile
            The profile a prover is asked to serve.

        Raises
        ------
        ProverUnsupportedError
            If the profile's fragment exceeds the served one on some field or
            the profile names a world type the prover does not serve.
        """
        fragment = profile.fragment

        excess = self._fragment.excess(fragment)

        if excess is not None:
            msg = (
                f"{self._prover} does not serve {excess} of fragment {fragment!r}; "
                f"the served fragment is {self._fragment!r}"
            )
            raise ProverUnsupportedError(excess, msg)

        if (
            "*" not in self._world_types
            and not profile.world_types <= self._world_types
        ):
            field = "world_types"
            msg = (
                f"{self._prover} does not serve world types "
                f"{sorted(profile.world_types - self._world_types)!r}"
            )
            raise ProverUnsupportedError(field, msg)

    def _identity(self) -> tuple[str, LogicFragment, tuple[str, ...]]:
        """Return fields in comparison order."""
        return (self._prover, self._fragment, tuple(sorted(self._world_types)))

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another bound has the same fields."""
        return type(other) is ProverBound and self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the bound fields."""
        return hash((ProverBound, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this bound."""
        return (
            f"ProverBound({self._prover!r}, {self._fragment!r}, "
            f"world_types={sorted(self._world_types)!r})"
        )
