"""Binding sites, operators, vocabularies and many-sorted signatures.

A signature declares its sorts, its operators and, per sort, what the kernel
derives for it. A sort that declares a vocabulary gets one constant operator
``#const[S]`` whose payload is a constant's stable identifier, its position in
the vocabulary. A sort that some binding site binds gets one bound-variable
operator ``#bvar[S]``, and it and every sort declared open get one
free variable operator ``#fvar[S]``, so a binder can be opened over free
variables of its sort. A signature declared with holes gets one ``#hole[S]``
per sort. Declared operators never carry a payload and never use a name,
family or payload the kernel derives.

Codes order the operators sort-major, then by family, then by arity, then by
declaration index, so the operators of one family within one sort hold one
contiguous block of codes.
"""

from hashlib import sha256
from itertools import pairwise
from json import dumps
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, override

from lofi._checks import ensure_name, ensure_non_negative
from lofi._frozen import FrozenValue
from lofi._native import (
    BindingShapeError,
    DuplicateNameError,
    ErasureError,
    ReservedDeclarationError,
    UndeclaredSortError,
    UnknownConstantError,
    UnknownOperatorError,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from typing import Literal

    from lofi.terms._ports import OpName, PayloadSort, Sort

__all__ = ["BindingSite", "Operator", "Signature", "Vocabulary"]

_CONSTANT_FAMILY: Final = "constant"
_VARIABLE_FAMILY: Final = "variable"
_HOLE_FAMILY: Final = "hole"
_KERNEL_FAMILIES: Final = frozenset((_CONSTANT_FAMILY, _VARIABLE_FAMILY, _HOLE_FAMILY))
_RESERVED_PREFIX: Final = "#"


def _derived_name(kind: str, sort: Sort) -> OpName:
    """Return the name of a kernel-derived operator of one sort.

    Parameters
    ----------
    kind : str
        The derived operator's kind, one of ``const``, ``bvar``, ``fvar`` and
        ``hole``.
    sort : Sort
        The operator's result sort.
    """
    return f"{_RESERVED_PREFIX}{kind}[{sort}]"


def _position(field: str, value: int) -> int:
    """Return a validated argument position.

    Parameters
    ----------
    field : str
        The field name used in a refusal.
    value : int
        The position.

    Raises
    ------
    TypeError
        If the position is not an integer.
    ValueError
        If the position is negative.
    """
    return ensure_non_negative(field, value)


class BindingSite(FrozenValue):
    """One variable an operator brings into scope.

    The variable scopes over the listed argument positions, so occurrences of
    it inside those arguments are bound variables of its sort. An annotation
    position, when present, is an argument outside that scope holding the
    variable's classifier, as a Church-style abstraction holds the type of the
    variable it binds.

    Parameters
    ----------
    arguments : Sequence[int]
        The argument positions the variable scopes over, strictly ascending.
    sort : Sort
        The sort of the bound variable.
    annotation : int | None
        The argument position holding the variable's classifier, or None.

    Raises
    ------
    TypeError
        If a position or the sort has an invalid type.
    ValueError
        If no position is given, the positions are not strictly ascending, a
        position is negative, the sort is empty, or the annotation position
        is one the variable scopes over.
    """

    _noun = "declaration value"

    __slots__ = ("_annotation", "_arguments", "_sort")

    def __init__(
        self,
        arguments: Sequence[int],
        sort: Sort,
        annotation: int | None = None,
    ) -> None:
        held = tuple(
            _position(f"arguments[{at}]", each) for at, each in enumerate(arguments)
        )

        if not held:
            msg = "a binding site scopes over at least one argument"
            raise ValueError(msg)

        if any(left >= right for left, right in pairwise(held)):
            msg = f"binding-site arguments are not strictly ascending: {held!r}"
            raise ValueError(msg)

        if annotation is not None and _position("annotation", annotation) in held:
            msg = f"annotation position {annotation} lies inside the variable's scope"
            raise ValueError(msg)

        object.__setattr__(self, "_arguments", held)
        object.__setattr__(self, "_sort", ensure_name("sort", sort))
        object.__setattr__(self, "_annotation", annotation)

    @property
    def arguments(self) -> tuple[int, ...]:
        """Return the argument positions the variable scopes over."""
        return self._arguments

    @property
    def sort(self) -> Sort:
        """Return the bound variable's sort."""
        return self._sort

    @property
    def annotation(self) -> int | None:
        """Return the position of the variable's classifier, or None."""
        return self._annotation

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether every field matches."""
        return isinstance(other, BindingSite) and (
            self._arguments,
            self._sort,
            self._annotation,
        ) == (other._arguments, other._sort, other._annotation)

    @override
    def __hash__(self) -> int:
        """Return a hash of every field."""
        return hash((self._arguments, self._sort, self._annotation))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this site."""
        return (
            f"BindingSite({self._arguments!r}, {self._sort!r}, "
            f"annotation={self._annotation!r})"
        )


class Operator(FrozenValue):
    """A term-algebra operator declaration.

    Parameters
    ----------
    name : OpName
        The operator name.
    sort : Sort
        The result sort.
    arg_sorts : Sequence[Sort]
        The required argument sorts.
    binding : Sequence[BindingSite]
        The variables the operator binds, in order.
    payload_sort : PayloadSort
        The non-term payload kind. Only a kernel-derived operator has one.
    family : str
        The decoder-head family.
    erases_to : OpName | None
        The optional erasure-target name.

    Raises
    ------
    TypeError
        If a name or binding site has an invalid type.
    ValueError
        If a name is empty.
    BindingShapeError
        If a binding site names a position outside the operator's arity.
    """

    _noun = "declaration value"

    __slots__ = (
        "_arg_sorts",
        "_binding",
        "_erases_to",
        "_family",
        "_name",
        "_payload_sort",
        "_scopes",
        "_shifts",
        "_sort",
    )

    def __init__(
        self,
        name: OpName,
        sort: Sort,
        *,
        arg_sorts: Sequence[Sort] = (),
        binding: Sequence[BindingSite] = (),
        payload_sort: PayloadSort = "none",
        family: str = "primitive",
        erases_to: OpName | None = None,
    ) -> None:
        object.__setattr__(self, "_name", ensure_name("name", name))
        object.__setattr__(self, "_sort", ensure_name("sort", sort))
        object.__setattr__(
            self,
            "_arg_sorts",
            tuple(
                ensure_name(f"arg_sorts[{at}]", each)
                for at, each in enumerate(arg_sorts)
            ),
        )
        object.__setattr__(self, "_family", ensure_name("family", family))
        object.__setattr__(self, "_payload_sort", payload_sort)
        object.__setattr__(
            self,
            "_erases_to",
            None if erases_to is None else ensure_name("erases_to", erases_to),
        )
        sites = tuple(binding)

        for site in sites:
            if not isinstance(site, BindingSite):
                msg = f"binding holds a non-site value: {site!r}"
                raise TypeError(msg)

            outside = [
                position
                for position in (*site.arguments, site.annotation)
                if position is not None and position >= len(self._arg_sorts)
            ]

            if outside:
                raise BindingShapeError(
                    operator=self._name,
                    reason=(
                        f"names position {outside[0]}, outside the arity "
                        f"{len(self._arg_sorts)}"
                    ),
                )

        object.__setattr__(self, "_binding", sites)
        scopes = tuple(
            tuple(site for site in sites if position in site.arguments)
            for position in range(len(self._arg_sorts))
        )
        object.__setattr__(self, "_scopes", scopes)
        shifts: list[Mapping[Sort, int]] = []

        for scope in scopes:
            counts: dict[Sort, int] = {}

            for site in scope:
                counts[site.sort] = counts.get(site.sort, 0) + 1

            shifts.append(MappingProxyType(counts))

        object.__setattr__(self, "_shifts", tuple(shifts))

    @property
    def name(self) -> OpName:
        """Return the operator's name, unique within a signature."""
        return self._name

    @property
    def sort(self) -> Sort:
        """Return the operator's result sort."""
        return self._sort

    @property
    def arg_sorts(self) -> tuple[Sort, ...]:
        """Return the required argument sorts."""
        return self._arg_sorts

    @property
    def binding(self) -> tuple[BindingSite, ...]:
        """Return the variables this operator binds, in order."""
        return self._binding

    @property
    def payload_sort(self) -> PayloadSort:
        """Return the non-term payload kind."""
        return self._payload_sort

    @property
    def family(self) -> str:
        """Return the operator's decoder-head family."""
        return self._family

    @property
    def erases_to(self) -> OpName | None:
        """Return the erasure target, or None for a core operator."""
        return self._erases_to

    @property
    def arity(self) -> int:
        """Return how many arguments this operator takes."""
        return len(self._arg_sorts)

    @property
    def is_binder(self) -> bool:
        """Return whether this operator brings a variable into scope."""
        return bool(self._binding)

    @property
    def is_constant(self) -> bool:
        """Return whether this is a kernel-declared constant operator."""
        return self._payload_sort == "constant"

    @property
    def is_bound_variable(self) -> bool:
        """Return whether this is a kernel-declared bound-variable operator."""
        return self._payload_sort == "index"

    @property
    def is_free_variable(self) -> bool:
        """Return whether this is a kernel-declared free variable operator."""
        return self._payload_sort == "free"

    @property
    def is_variable(self) -> bool:
        """Return whether this operator denotes a bound or free variable."""
        return self.is_bound_variable or self.is_free_variable

    @property
    def is_hole(self) -> bool:
        """Return whether this is a kernel-declared frontier hole."""
        return self._family == _HOLE_FAMILY

    @property
    def has_payload(self) -> bool:
        """Return whether occurrences of this operator carry a payload."""
        return self._payload_sort != "none"

    def scope(self, position: int, /) -> tuple[BindingSite, ...]:
        """Return the binding sites whose variables scope over an argument.

        Parameters
        ----------
        position : int
            The argument position.

        Returns
        -------
        tuple[BindingSite, ...]
            The sites in declaration order. Among the sites of one sort, the
            first is the innermost, so its variable is the bound index zero of
            that sort directly inside the argument.
        """
        return self._scopes[position]

    def shift(self, position: int, sort: Sort, /) -> int:
        """Return how many variables of a sort scope over an argument.

        Parameters
        ----------
        position : int
            The argument position.
        sort : Sort
            The bound sort counted.
        """
        return self._shifts[position].get(sort, 0)

    @property
    def _declaration(
        self,
    ) -> tuple[
        OpName,
        Sort,
        tuple[Sort, ...],
        tuple[BindingSite, ...],
        PayloadSort,
        str,
        OpName | None,
    ]:
        """Return constructor fields in declaration order."""
        return (
            self._name,
            self._sort,
            self._arg_sorts,
            self._binding,
            self._payload_sort,
            self._family,
            self._erases_to,
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether every declaration field matches."""
        return isinstance(other, Operator) and self._declaration == other._declaration

    @override
    def __hash__(self) -> int:
        """Return a hash of the name, which a signature keeps unique."""
        return hash(self._name)

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this operator."""
        return (
            f"Operator({self._name!r}, {self._sort!r}, "
            f"arg_sorts={self._arg_sorts!r}, binding={self._binding!r}, "
            f"payload_sort={self._payload_sort!r}, family={self._family!r}, "
            f"erases_to={self._erases_to!r})"
        )


class Vocabulary(FrozenValue):
    """The ordered constants of one sort.

    A constant's stable identifier is its position, so two stores over one
    signature identify a constant by the same integer, and the digest is the
    SHA-256 of the names in order.

    Parameters
    ----------
    sort : Sort
        The sort whose constants these are.
    names : Sequence[str]
        The distinct constant names, in identifier order.

    Raises
    ------
    TypeError
        If the sort or a name is not a string.
    ValueError
        If the sort or a name is empty, or a name is repeated.
    """

    _noun = "declaration value"

    __slots__ = ("_digest", "_ids", "_names", "_sort")

    def __init__(self, sort: Sort, names: Sequence[str]) -> None:
        held = tuple(ensure_name(f"names[{at}]", each) for at, each in enumerate(names))
        ids = {name: at for at, name in enumerate(held)}

        if len(ids) != len(held):
            repeated = next(name for name in held if held.count(name) > 1)
            msg = f"the vocabulary of sort {sort!r} repeats the constant {repeated!r}"
            raise ValueError(msg)

        object.__setattr__(self, "_sort", ensure_name("sort", sort))
        object.__setattr__(self, "_names", held)
        object.__setattr__(self, "_ids", MappingProxyType(ids))
        written = dumps(list(held), separators=(",", ":"), ensure_ascii=True)
        object.__setattr__(self, "_digest", sha256(written.encode()).hexdigest())

    @property
    def sort(self) -> Sort:
        """Return the sort whose constants these are."""
        return self._sort

    @property
    def names(self) -> tuple[str, ...]:
        """Return the constant names in identifier order."""
        return self._names

    @property
    def digest(self) -> str:
        """Return the SHA-256 of the names in order, as lowercase hexadecimal."""
        return self._digest

    def id(self, name: str, /) -> int:
        """Return a constant's stable identifier.

        Parameters
        ----------
        name : str
            The constant name.

        Raises
        ------
        UnknownConstantError
            If the vocabulary has no such constant.
        """
        held = self._ids.get(name)

        if held is None:
            raise UnknownConstantError(sort=self._sort, name=name)

        return held

    def name(self, identifier: int, /) -> str:
        """Return the name of the constant with an identifier.

        Parameters
        ----------
        identifier : int
            The stable identifier.

        Raises
        ------
        ValueError
            If no constant has the identifier.
        """
        if isinstance(identifier, bool) or not isinstance(identifier, int):
            msg = f"identifier is not an integer: {identifier!r}"
            raise TypeError(msg)

        if not 0 <= identifier < len(self._names):
            msg = (
                f"the vocabulary of sort {self._sort!r} has no constant with "
                f"identifier {identifier!r}"
            )
            raise ValueError(msg)

        return self._names[identifier]

    def __len__(self) -> int:
        """Return how many constants the vocabulary holds."""
        return len(self._names)

    def __contains__(self, name: str) -> bool:
        """Return whether a name is one of the constants."""
        return name in self._ids

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether the sort and the names match in order."""
        return isinstance(other, Vocabulary) and (self._sort, self._names) == (
            other._sort,
            other._names,
        )

    @override
    def __hash__(self) -> int:
        """Return a hash of the sort and the digest."""
        return hash((self._sort, self._digest))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this vocabulary."""
        return f"Vocabulary({self._sort!r}, {self._names!r})"


def _refuse_repeats(
    names: Iterable[str],
    *,
    kind: Literal["operator", "sort", "vocabulary", "open sort"],
    signature: str,
) -> None:
    """Raise on the first repeated name.

    Parameters
    ----------
    names : Iterable[str]
        The declared names.
    kind : Literal["operator", "sort", "vocabulary", "open sort"]
        The declaration kind used in the error.
    signature : str
        The signature name.

    Raises
    ------
    DuplicateNameError
        If a name appears more than once.
    """
    seen: set[str] = set()

    for each in names:
        if each in seen:
            raise DuplicateNameError(name=each, kind=kind, signature=signature)

        seen.add(each)


def _refuse_reserved(
    operators: Iterable[Operator],
    *,
    signature: str,
    kernel_families: frozenset[str] = _KERNEL_FAMILIES,
) -> None:
    """Raise on the first declared operator claiming a kernel-derived form.

    Parameters
    ----------
    operators : Iterable[Operator]
        The declared operators.
    signature : str
        The signature name.

    Raises
    ------
    ReservedDeclarationError
        If an operator's name starts with the reserved prefix, its family is a
        kernel family, or it declares a payload.
    """
    for op in operators:
        reason = None

        if op.name.startswith(_RESERVED_PREFIX):
            reason = f"takes a name starting with {_RESERVED_PREFIX!r}"
        elif op.family in kernel_families:
            reason = f"declares the family {op.family!r}"
        elif op.payload_sort != "none":
            reason = f"declares a {op.payload_sort!r} payload"

        if reason is not None:
            raise ReservedDeclarationError(
                operator=op.name, reason=reason, signature=signature
            )


def _refuse_undeclared_sorts(
    operators: Iterable[Operator],
    sorts: frozenset[Sort],
    *,
    signature: str,
) -> None:
    """Raise on the first operator that names an undeclared sort.

    Parameters
    ----------
    operators : Iterable[Operator]
        The operators to inspect.
    sorts : frozenset[Sort]
        The declared sort inventory.
    signature : str
        The signature name.

    Raises
    ------
    UndeclaredSortError
        If an operator names a sort outside the inventory, including the sort
        of a variable it binds.
    """
    for op in operators:
        named = (op.sort, *op.arg_sorts, *(site.sort for site in op.binding))

        for sort in named:
            if sort in sorts:
                continue

            raise UndeclaredSortError(sort=sort, operator=op.name, signature=signature)


def _refuse_unclassified_annotations(
    operators: Iterable[Operator],
    classifiers: Mapping[Sort, Sort],
) -> None:
    """Raise on the first annotation position whose sort is not the classifier's.

    Parameters
    ----------
    operators : Iterable[Operator]
        The operators to inspect.
    classifiers : Mapping[Sort, Sort]
        The classifier sort of each classified sort.

    Raises
    ------
    BindingShapeError
        If an annotation binds a sort with no declared classifier, or its
        position's argument sort is not that classifier sort.
    """
    for op in operators:
        for site in op.binding:
            if site.annotation is None:
                continue

            classifier = classifiers.get(site.sort)
            held = op.arg_sorts[site.annotation]

            if classifier is None:
                reason = (
                    f"annotates a variable of sort {site.sort!r}, which has no "
                    "declared classifier"
                )
                raise BindingShapeError(operator=op.name, reason=reason)

            if held != classifier:
                reason = (
                    f"annotates a variable of sort {site.sort!r} at an argument of "
                    f"sort {held!r}, not its classifier sort {classifier!r}"
                )
                raise BindingShapeError(operator=op.name, reason=reason)


def _featured_name(primitive: OpName, feature: str) -> OpName:
    """Return the name for a primitive and feature pair.

    Parameters
    ----------
    primitive : OpName
        The primitive operator name.
    feature : str
        The feature value.

    Returns
    -------
    OpName
        The bracketed featured name.
    """
    return f"{primitive}[{feature}]"


def _featured(
    features: Mapping[OpName, Sequence[str]],
    by_name: Mapping[OpName, Operator],
    *,
    signature: str,
) -> tuple[Operator, ...]:
    """Return one operator for each declared primitive-feature pair.

    Parameters
    ----------
    features : Mapping[OpName, Sequence[str]]
        The feature values declared for each primitive.
    by_name : Mapping[OpName, Operator]
        The primitive operators keyed by name.
    signature : str
        The signature name.

    Returns
    -------
    tuple[Operator, ...]
        The generated operators in declaration order.

    Raises
    ------
    TypeError
        If a primitive or feature has an invalid type.
    ValueError
        If a primitive or feature name is empty.
    UnknownOperatorError
        If a named primitive is not declared.
    """
    generated: list[Operator] = []

    for primitive, values in features.items():
        bare = by_name.get(ensure_name("features", primitive))

        if bare is None:
            raise UnknownOperatorError(operator=primitive, signature=signature)

        field = f"features[{primitive!r}]"
        generated.extend(
            Operator(
                _featured_name(primitive, ensure_name(f"{field}[{at}]", value)),
                bare.sort,
                arg_sorts=bare.arg_sorts,
                binding=bare.binding,
                family=bare.family,
                erases_to=primitive,
            )
            for at, value in enumerate(values)
        )

    return tuple(generated)


def _derived(
    sorts: Sequence[Sort],
    *,
    vocabularies: Mapping[Sort, Vocabulary],
    bound: frozenset[Sort],
    open_sorts: frozenset[Sort],
    holes: bool,
    constant_family: str,
    variable_family: str,
) -> tuple[Operator, ...]:
    """Return the operators the kernel derives for each sort.

    Parameters
    ----------
    sorts : Sequence[Sort]
        The sort inventory in declaration order.
    vocabularies : Mapping[Sort, Vocabulary]
        The vocabulary of each sort that declares one.
    bound : frozenset[Sort]
        The sorts some binding site binds.
    open_sorts : frozenset[Sort]
        The sorts declared open to free variables.
    holes : bool
        Whether every sort gets a hole.
    """
    derived: list[Operator] = []

    for sort in sorts:
        if sort in vocabularies:
            derived.append(
                Operator(
                    _derived_name("const", sort),
                    sort,
                    payload_sort="constant",
                    family=constant_family,
                )
            )

        if sort in bound:
            derived.append(
                Operator(
                    _derived_name("bvar", sort),
                    sort,
                    payload_sort="index",
                    family=variable_family,
                )
            )

        if sort in open_sorts or sort in bound:
            derived.append(
                Operator(
                    _derived_name("fvar", sort),
                    sort,
                    payload_sort="free",
                    family=variable_family,
                )
            )

        if holes:
            derived.append(
                Operator(_derived_name("hole", sort), sort, family=_HOLE_FAMILY)
            )

    return tuple(derived)


def _erasure(
    by_name: Mapping[OpName, Operator],
    *,
    signature: str,
) -> Mapping[OpName, Operator]:
    """Return each operator's validated erasure core.

    Parameters
    ----------
    by_name : Mapping[OpName, Operator]
        The operators keyed by name.
    signature : str
        The signature name.

    Returns
    -------
    Mapping[OpName, Operator]
        Erasure cores keyed by operator name.

    Raises
    ------
    UnknownOperatorError
        If an erasure target is undeclared.
    ErasureError
        If erasure is not idempotent or shape preserving.
    """
    cores: dict[OpName, Operator] = {}

    for name, op in by_name.items():
        if op.erases_to is None:
            cores[name] = op

            continue

        target = by_name.get(op.erases_to)

        if target is None:
            raise UnknownOperatorError(operator=op.erases_to, signature=signature)

        broken: Literal["idempotent", "arity-preserving", "sort-preserving"] | None = (
            None
        )

        # a self-target is already an erasure core.
        if target.erases_to not in (None, target.name):
            broken = "idempotent"
        elif target.arity != op.arity:
            broken = "arity-preserving"
        elif target.sort != op.sort:
            broken = "sort-preserving"

        if broken is not None:
            raise ErasureError(
                operator=name,
                target=target.name,
                requirement=broken,
                signature=signature,
            )

        cores[name] = target

    return cores


def _code_order(
    inventory: Sequence[Operator],
    sorts: Sequence[Sort],
    *,
    constant_family: str,
    variable_family: str,
) -> tuple[Operator, ...]:
    """Return operators sort-major, then by family, arity and declaration index.

    The kernel-derived families come first within a sort, constants before
    variables, then the declared families in order of first declaration, and
    holes last.

    Parameters
    ----------
    inventory : Sequence[Operator]
        Every operator in declaration order, derived operators last.
    sorts : Sequence[Sort]
        The sort inventory in declaration order.
    """
    sort_rank = {sort: at for at, sort in enumerate(sorts)}
    family_rank: dict[str, int] = {constant_family: 0, variable_family: 1}

    for op in inventory:
        if op.family != _HOLE_FAMILY:
            family_rank.setdefault(op.family, len(family_rank))

    family_rank[_HOLE_FAMILY] = len(family_rank)
    ordered = sorted(
        enumerate(inventory),
        key=lambda pair: (
            sort_rank[pair[1].sort],
            family_rank[pair[1].family],
            pair[1].arity,
            pair[0],
        ),
    )

    return tuple(op for _, op in ordered)


def _blocks(operators: Sequence[Operator]) -> Mapping[int, tuple[Operator, ...]]:
    """Return operators grouped by arity.

    Parameters
    ----------
    operators : Sequence[Operator]
        The operators in code order.

    Returns
    -------
    Mapping[int, tuple[Operator, ...]]
        Operators keyed by arity.
    """
    held: dict[int, list[Operator]] = {}

    for op in operators:
        held.setdefault(op.arity, []).append(op)

    return MappingProxyType({arity: tuple(each) for arity, each in held.items()})


def _digest_of(
    signature: Signature,
    *,
    codes: Mapping[OpName, int],
    cores: Mapping[OpName, Operator],
) -> str:
    """Return the declaration digest as lowercase hexadecimal.

    Parameters
    ----------
    signature : Signature
        The signature whose declaration is digested.
    codes : Mapping[OpName, int]
        The operator codes keyed by name.
    cores : Mapping[OpName, Operator]
        The erasure cores keyed by name.

    Returns
    -------
    str
        The SHA-256 digest.
    """
    rows: list[list[str | int | None]] = [
        [codes[op.name], op.name, op.arity, op.sort, cores[op.name].name]
        for op in signature.operators
    ]
    rows.extend(
        ["sort", position, sort] for position, sort in enumerate(signature.sorts)
    )
    rows.extend(
        ["argument", op.name, position, sort]
        for op in signature.operators
        for position, sort in enumerate(op.arg_sorts)
    )
    rows.extend(
        ["family", op.name, op.family, op.payload_sort] for op in signature.operators
    )
    rows.extend(
        [op.name, at, ",".join(map(str, site.arguments)), site.sort, site.annotation]
        for op in signature.operators
        for at, site in enumerate(op.binding)
    )
    rows.extend(
        ["vocabulary", vocabulary.sort, vocabulary.digest]
        for vocabulary in signature.vocabularies.values()
    )
    rows.extend(
        ["constant-core", sort, name, core]
        for sort, mappings in signature.constant_cores.items()
        for name, core in mappings.items()
    )
    rows.extend(["open", sort] for sort in signature.open_sorts)
    rows.extend(
        ["classifier", sort, classifier]
        for sort, classifier in signature.classifiers.items()
    )
    rows.extend(
        ["feature", primitive, position, feature]
        for primitive, features in signature.features.items()
        for position, feature in enumerate(features)
    )
    rows.append(["holes", int(signature.has_holes)])
    rows.append(["binding-policy", "vacuous", int(signature.allows_vacuous_binding)])
    # canonical JSON makes the digest independent of process state.
    written = dumps(rows, separators=(",", ":"))

    return sha256(written.encode()).hexdigest()


class Signature(FrozenValue):
    """A many-sorted term-algebra signature.

    Parameters
    ----------
    name : str
        The signature name.
    sorts : Sequence[Sort]
        The sort inventory in declaration order.
    operators : Sequence[Operator]
        The declared operators in declaration order.
    vocabularies : Sequence[Vocabulary]
        The vocabulary of each sort that has constants.
    constant_cores : Mapping[Sort, Mapping[str, str]] | None
        Direct feature-core constants keyed by vocabulary sort and constant
        name. An omitted constant is its own core.
    open_sorts : Sequence[Sort]
        The sorts whose terms may hold free variables though no binder binds
        them. A bound sort has free variables in any case.
    classifiers : Mapping[Sort, Sort] | None
        The sort whose terms classify the terms of each classified sort: the
        sort of a constant's classifier and of a binding site's annotation.
    allows_vacuous_binding : bool
        Whether binders may introduce unused variables.
    features : Mapping[OpName, Sequence[str]] | None
        The optional feature values for each primitive.
    holes : bool
        Whether every sort gets a hole operator.

    Raises
    ------
    TypeError
        If a declaration field has an invalid type.
    ValueError
        If a declaration name or binding policy is invalid.
    DuplicateNameError
        If a sort, operator, vocabulary sort or open sort is repeated.
    UndeclaredSortError
        If an operator, vocabulary, open sort or classifier names an
        undeclared sort.
    ReservedDeclarationError
        If a declared operator claims a kernel-derived name, family or
        payload.
    BindingShapeError
        If an annotation position does not hold its variable's classifier
        sort.
    UnknownOperatorError
        If a feature primitive or erasure target is undeclared.
    ErasureError
        If an erasure target violates an erasure invariant.
    """

    _noun = "declaration value"

    PAD: Final = 0
    """The padding code that belongs to no operator."""

    __slots__ = (
        "_allows_vacuous_binding",
        "_bound_sorts",
        "_by_arity",
        "_by_name",
        "_classifiers",
        "_codes",
        "_constant_cores",
        "_constant_family",
        "_cores",
        "_declarations",
        "_digest",
        "_features",
        "_has_binders",
        "_holes",
        "_name",
        "_open_sorts",
        "_operators",
        "_sorts",
        "_variable_family",
        "_vocabularies",
    )

    def __init__(
        self,
        name: str,
        sorts: Sequence[Sort],
        operators: Sequence[Operator],
        *,
        vocabularies: Sequence[Vocabulary] = (),
        constant_cores: Mapping[Sort, Mapping[str, str]] | None = None,
        open_sorts: Sequence[Sort] = (),
        classifiers: Mapping[Sort, Sort] | None = None,
        allows_vacuous_binding: bool = True,
        features: Mapping[OpName, Sequence[str]] | None = None,
        holes: bool = False,
        constant_family: str = _CONSTANT_FAMILY,
        variable_family: str = _VARIABLE_FAMILY,
    ) -> None:
        if not isinstance(allows_vacuous_binding, bool):
            msg = "allows_vacuous_binding must be a bool"
            raise TypeError(msg)

        if not isinstance(holes, bool):
            msg = "holes must be a bool"
            raise TypeError(msg)

        object.__setattr__(self, "_allows_vacuous_binding", allows_vacuous_binding)
        object.__setattr__(self, "_holes", holes)
        object.__setattr__(self, "_name", ensure_name("name", name))
        held_constant_family = ensure_name("constant family", constant_family)
        held_variable_family = ensure_name("variable family", variable_family)

        if len({held_constant_family, held_variable_family, _HOLE_FAMILY}) != 3:
            msg = "constant, variable and hole families are distinct"
            raise ValueError(msg)

        object.__setattr__(self, "_constant_family", held_constant_family)
        object.__setattr__(self, "_variable_family", held_variable_family)
        object.__setattr__(
            self,
            "_sorts",
            tuple(ensure_name(f"sorts[{at}]", each) for at, each in enumerate(sorts)),
        )
        _refuse_repeats(self._sorts, kind="sort", signature=self._name)
        declared_sorts = frozenset(self._sorts)
        declared = tuple(operators)
        _refuse_reserved(
            declared,
            signature=self._name,
            kernel_families=frozenset(
                (held_constant_family, held_variable_family, _HOLE_FAMILY)
            ),
        )
        object.__setattr__(self, "_declarations", declared)
        object.__setattr__(
            self,
            "_features",
            MappingProxyType(
                {}
                if features is None
                else {primitive: tuple(each) for primitive, each in features.items()}
            ),
        )
        held_vocabularies = tuple(vocabularies)
        _refuse_repeats(
            (each.sort for each in held_vocabularies),
            kind="vocabulary",
            signature=self._name,
        )
        held_open = tuple(
            ensure_name(f"open_sorts[{at}]", each) for at, each in enumerate(open_sorts)
        )
        _refuse_repeats(held_open, kind="open sort", signature=self._name)
        held_classifiers = {} if classifiers is None else dict(classifiers)

        for role, sort in (
            *(("vocabulary", each.sort) for each in held_vocabularies),
            *(("open sort", each) for each in held_open),
            *(("classified sort", each) for each in held_classifiers),
            *(("classifier", each) for each in held_classifiers.values()),
        ):
            if sort not in declared_sorts:
                raise UndeclaredSortError(
                    sort=sort, operator=role, signature=self._name
                )

        declared_and_featured = (
            *declared,
            *_featured(
                self._features,
                {op.name: op for op in declared},
                signature=self._name,
            ),
        )
        _refuse_undeclared_sorts(
            declared_and_featured, declared_sorts, signature=self._name
        )
        _refuse_unclassified_annotations(declared_and_featured, held_classifiers)
        bound = frozenset(
            site.sort for op in declared_and_featured for site in op.binding
        )
        object.__setattr__(
            self,
            "_vocabularies",
            MappingProxyType(
                {
                    sort: each
                    for sort in self._sorts
                    for each in held_vocabularies
                    if each.sort == sort
                }
            ),
        )
        received_constant_cores = {} if constant_cores is None else constant_cores
        held_constant_cores: dict[Sort, dict[str, str]] = {}

        for received_sort, received_rows in received_constant_cores.items():
            sort = ensure_name("constant core sort", received_sort)
            vocabulary = self._vocabularies.get(sort)

            if vocabulary is None:
                msg = f"constant cores name sort {sort!r}, which has no vocabulary"
                raise ValueError(msg)

            rows: dict[str, str] = {}

            for received_source, received_target in received_rows.items():
                source = ensure_name(f"constant_cores[{sort!r}] key", received_source)
                target = ensure_name(
                    f"constant_cores[{sort!r}][{source!r}]", received_target
                )

                if source not in vocabulary:
                    msg = (
                        f"constant core source {source!r} is not in vocabulary {sort!r}"
                    )
                    raise ValueError(msg)

                if target not in vocabulary:
                    msg = (
                        f"constant core target {target!r} is not in vocabulary {sort!r}"
                    )
                    raise ValueError(msg)

                if source == target:
                    msg = f"constant core source {source!r} names itself"
                    raise ValueError(msg)

                rows[source] = target

            indirect = sorted(set(rows) & set(rows.values()))

            if indirect:
                msg = (
                    "constant cores must be direct, but target "
                    f"{indirect[0]!r} is mapped"
                )
                raise ValueError(msg)

            held_constant_cores[sort] = {
                source: rows[source] for source in vocabulary.names if source in rows
            }

        object.__setattr__(
            self,
            "_constant_cores",
            MappingProxyType(
                {
                    sort: MappingProxyType(held_constant_cores[sort])
                    for sort in self._sorts
                    if sort in held_constant_cores
                }
            ),
        )
        object.__setattr__(
            self,
            "_open_sorts",
            tuple(sort for sort in self._sorts if sort in held_open),
        )
        object.__setattr__(
            self, "_bound_sorts", tuple(sort for sort in self._sorts if sort in bound)
        )
        object.__setattr__(
            self,
            "_classifiers",
            MappingProxyType(
                {
                    sort: held_classifiers[sort]
                    for sort in self._sorts
                    if sort in held_classifiers
                }
            ),
        )
        inventory = (
            *declared_and_featured,
            *_derived(
                self._sorts,
                vocabularies=self._vocabularies,
                bound=bound,
                open_sorts=frozenset(held_open),
                holes=holes,
                constant_family=held_constant_family,
                variable_family=held_variable_family,
            ),
        )
        _refuse_repeats(
            (op.name for op in inventory), kind="operator", signature=self._name
        )
        object.__setattr__(
            self,
            "_operators",
            _code_order(
                inventory,
                self._sorts,
                constant_family=held_constant_family,
                variable_family=held_variable_family,
            ),
        )
        object.__setattr__(
            self,
            "_codes",
            MappingProxyType(
                {
                    op.name: code
                    for code, op in enumerate(self._operators, start=Signature.PAD + 1)
                }
            ),
        )
        object.__setattr__(
            self,
            "_by_name",
            MappingProxyType({op.name: op for op in self._operators}),
        )
        object.__setattr__(self, "_by_arity", _blocks(self._operators))
        object.__setattr__(
            self,
            "_cores",
            MappingProxyType(dict(_erasure(self._by_name, signature=self._name))),
        )
        object.__setattr__(self, "_has_binders", bool(bound))
        object.__setattr__(
            self, "_digest", _digest_of(self, codes=self._codes, cores=self._cores)
        )

    @property
    def name(self) -> str:
        """Return the signature's name."""
        return self._name

    @property
    def sorts(self) -> tuple[Sort, ...]:
        """Return the sort inventory, in declaration order."""
        return self._sorts

    @property
    def operators(self) -> tuple[Operator, ...]:
        """Return every operator, declared, featured and derived, in code order."""
        return self._operators

    @property
    def declarations(self) -> tuple[Operator, ...]:
        """Return the declared operators, in declaration order."""
        return self._declarations

    @property
    def features(self) -> Mapping[OpName, tuple[str, ...]]:
        """Return admitted features by primitive as a read-only view."""
        return self._features

    @property
    def vocabularies(self) -> Mapping[Sort, Vocabulary]:
        """Return the vocabulary of each sort that has one, in sort order."""
        return self._vocabularies

    @property
    def constant_cores(self) -> Mapping[Sort, Mapping[str, str]]:
        """Return direct feature-core constants by vocabulary sort."""
        return self._constant_cores

    def constant_core(self, sort: Sort, name: str, /) -> str:
        """Return a vocabulary constant's declared feature core."""
        vocabulary = self.vocabulary(sort)
        held = ensure_name("constant", name)

        if held not in vocabulary:
            msg = f"constant {held!r} is not in vocabulary {sort!r}"
            raise ValueError(msg)

        return self._constant_cores.get(sort, {}).get(held, held)

    @property
    def open_sorts(self) -> tuple[Sort, ...]:
        """Return the sorts open to free variables, in sort order."""
        return self._open_sorts

    @property
    def bound_sorts(self) -> tuple[Sort, ...]:
        """Return the sorts some binding site binds, in sort order."""
        return self._bound_sorts

    @property
    def classifiers(self) -> Mapping[Sort, Sort]:
        """Return the classifier sort of each classified sort, in sort order."""
        return self._classifiers

    @property
    def has_holes(self) -> bool:
        """Return whether every sort has a hole operator."""
        return self._holes

    @property
    def allows_vacuous_binding(self) -> bool:
        """Return whether binders may leave introduced variables unused."""
        return self._allows_vacuous_binding

    @property
    def constant_family(self) -> str:
        """Return the declared decoder family for constant operators."""
        return self._constant_family

    @property
    def variable_family(self) -> str:
        """Return the declared decoder family for variable operators."""
        return self._variable_family

    @property
    def max_arity(self) -> int:
        """Return the greatest declared arity, or zero for an empty signature."""
        return max(self._by_arity, default=0)

    @property
    def has_binders(self) -> bool:
        """Return whether any operator brings a variable into scope."""
        return self._has_binders

    @property
    def has_payloads(self) -> bool:
        """Return whether any operator takes a non-term parameter."""
        return bool(self._vocabularies or self._bound_sorts or self._open_sorts)

    @property
    def digest(self) -> str:
        """Return the declaration digest as lowercase hexadecimal."""
        return self._digest

    def _declared(self, op: Operator | OpName, /) -> Operator:
        """Return the operator declared for a name or value.

        Parameters
        ----------
        op : Operator | OpName
            The operator or its name.

        Returns
        -------
        Operator
            The signature's matching declaration.

        Raises
        ------
        UnknownOperatorError
            If the operator is undeclared or conflicts with the declaration.
        """
        name = op if isinstance(op, str) else op.name
        held = self._by_name.get(name)

        if held is None or (not isinstance(op, str) and held != op):
            raise UnknownOperatorError(operator=name, signature=self._name)

        return held

    def by_name(self, name: OpName, /) -> Operator:
        """Return the operator declared under a name.

        Parameters
        ----------
        name : OpName
            The operator name.

        Returns
        -------
        Operator
            The declared operator.

        Raises
        ------
        UnknownOperatorError
            If the name is undeclared.
        """
        return self._declared(name)

    def by_arity(self, arity: int, /) -> tuple[Operator, ...]:
        """Return operators taking a given number of arguments.

        Parameters
        ----------
        arity : int
            The nonnegative argument count.

        Returns
        -------
        tuple[Operator, ...]
            The matching operators in code order.

        Raises
        ------
        TypeError
            If the arity is not an integer.
        ValueError
            If the arity is negative.
        """
        return self._by_arity.get(ensure_non_negative("arity", arity), ())

    def featured(self, primitive: OpName, feature: str, /) -> Operator:
        """Return the declared featured operator.

        Parameters
        ----------
        primitive : OpName
            The primitive operator name.
        feature : str
            The feature value.

        Returns
        -------
        Operator
            The declared featured operator.

        Raises
        ------
        TypeError
            If either name is not a string.
        ValueError
            If either name is empty.
        UnknownOperatorError
            If the pair is undeclared.
        """
        return self._declared(
            _featured_name(
                ensure_name("primitive", primitive), ensure_name("feature", feature)
            )
        )

    def vocabulary(self, sort: Sort, /) -> Vocabulary:
        """Return a sort's vocabulary.

        Parameters
        ----------
        sort : Sort
            The sort.

        Raises
        ------
        UnknownOperatorError
            If the sort declares no vocabulary, so it has no constant operator.
        """
        held = self._vocabularies.get(sort)

        if held is None:
            raise UnknownOperatorError(
                operator=_derived_name("const", sort), signature=self._name
            )

        return held

    def constant(self, sort: Sort, /) -> Operator:
        """Return a sort's constant operator.

        Parameters
        ----------
        sort : Sort
            The sort.

        Raises
        ------
        UnknownOperatorError
            If the sort declares no vocabulary.
        """
        return self._declared(_derived_name("const", sort))

    def bound(self, sort: Sort, /) -> Operator:
        """Return a sort's bound-variable operator.

        Parameters
        ----------
        sort : Sort
            The sort.

        Raises
        ------
        UnknownOperatorError
            If no binding site binds the sort.
        """
        return self._declared(_derived_name("bvar", sort))

    def free(self, sort: Sort, /) -> Operator:
        """Return a sort's free variable operator.

        Parameters
        ----------
        sort : Sort
            The sort.

        Raises
        ------
        UnknownOperatorError
            If the sort is not open.
        """
        return self._declared(_derived_name("fvar", sort))

    def hole(self, sort: Sort, /) -> Operator:
        """Return a sort's hole operator.

        Parameters
        ----------
        sort : Sort
            The sort.

        Raises
        ------
        UnknownOperatorError
            If the signature has no holes or does not declare the sort.
        """
        return self._declared(_derived_name("hole", sort))

    def is_constant(self, op: Operator, /) -> bool:
        """Return whether an operator is this signature's constant role."""
        declared = self._declared(op)
        constant = self._vocabularies.get(declared.sort)

        return constant is not None and declared is self.constant(declared.sort)

    def is_bound_variable(self, op: Operator, /) -> bool:
        """Return whether an operator is this signature's bound-variable role."""
        declared = self._declared(op)

        return declared.sort in self._bound_sorts and declared is self.bound(
            declared.sort
        )

    def is_free_variable(self, op: Operator, /) -> bool:
        """Return whether an operator is this signature's free variable role."""
        declared = self._declared(op)

        return (
            declared.sort in self._open_sorts or declared.sort in self._bound_sorts
        ) and declared is self.free(declared.sort)

    def is_variable(self, op: Operator, /) -> bool:
        """Return whether an operator has a variable role in this signature."""
        return self.is_bound_variable(op) or self.is_free_variable(op)

    def is_binder(self, op: Operator, /) -> bool:
        """Return whether a declared operator introduces a scoped variable."""
        return self._declared(op).is_binder

    def is_hole(self, op: Operator, /) -> bool:
        """Return whether an operator is this signature's frontier-hole role."""
        declared = self._declared(op)

        return self._holes and declared is self.hole(declared.sort)

    def with_holes(self) -> Signature:
        """Return this signature with a hole operator for every sort."""
        return Signature(
            self._name,
            self._sorts,
            self._declarations,
            vocabularies=tuple(self._vocabularies.values()),
            constant_cores=self._constant_cores,
            open_sorts=self._open_sorts,
            classifiers=self._classifiers,
            allows_vacuous_binding=self._allows_vacuous_binding,
            features=self._features,
            holes=True,
            constant_family=self._constant_family,
            variable_family=self._variable_family,
        )

    def with_vocabulary(self, vocabulary: Vocabulary, /) -> Signature:
        """Return this signature with one sort's vocabulary declared or replaced.

        Parameters
        ----------
        vocabulary : Vocabulary
            The vocabulary of its sort in the returned signature.
        """
        held = {**self._vocabularies, vocabulary.sort: vocabulary}

        return Signature(
            self._name,
            self._sorts,
            self._declarations,
            vocabularies=tuple(held.values()),
            constant_cores=self._constant_cores,
            open_sorts=self._open_sorts,
            classifiers=self._classifiers,
            allows_vacuous_binding=self._allows_vacuous_binding,
            features=self._features,
            holes=self._holes,
            constant_family=self._constant_family,
            variable_family=self._variable_family,
        )

    def code(self, op: Operator | OpName, /) -> int:
        """Return an operator's code.

        Parameters
        ----------
        op : Operator | OpName
            The operator or its name.

        Returns
        -------
        int
            The positive operator code.

        Raises
        ------
        UnknownOperatorError
            If the operator is undeclared or conflicts with the declaration.
        """
        return self._codes[self._declared(op).name]

    def operator(self, code: int, /) -> Operator:
        """Return the operator mapped to a code.

        Parameters
        ----------
        code : int
            The operator code.

        Returns
        -------
        Operator
            The mapped operator.

        Raises
        ------
        UnknownOperatorError
            If the code is padding or outside the alphabet.
        """
        at = code - Signature.PAD - 1

        if at < 0 or at >= len(self._operators):
            raise UnknownOperatorError(operator=code, signature=self._name)

        return self._operators[at]

    def erase(self, op: Operator | OpName, /) -> Operator:
        """Return an operator's erasure core.

        Parameters
        ----------
        op : Operator | OpName
            The operator or its name.

        Returns
        -------
        Operator
            The validated erasure core.

        Raises
        ------
        UnknownOperatorError
            If the operator is undeclared or conflicts with the declaration.
        """
        return self._cores[self._declared(op).name]

    @override
    def __repr__(self) -> str:
        """Return the signature summary."""
        return (
            f"Signature({self._name!r}, {len(self._sorts)} sorts, "
            f"{len(self._operators)} operators)"
        )
