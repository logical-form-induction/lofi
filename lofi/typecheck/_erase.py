"""Category-to-type signature morphisms."""

from types import MappingProxyType
from typing import TYPE_CHECKING, Never, Self, final, override

from lofi._checks import ensure_name
from lofi.terms import Term, TermStore

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi.terms import Operator, OpName, Signature

__all__ = ["Erasure"]

_FUNCTION_ARITY = 2


def _core_operators(signature: Signature, category_sort: str) -> tuple[Operator, ...]:
    """Return unique category-constructor cores in code order."""
    seen: set[OpName] = set()
    cores: list[Operator] = []

    for operator in signature.operators:
        if (
            operator.sort != category_sort
            or operator.has_payload
            or not operator.arg_sorts
            or set(operator.arg_sorts) != {category_sort}
        ):
            continue

        core = signature.erase(operator)

        if core.name not in seen:
            seen.add(core.name)
            cores.append(core)

    return tuple(cores)


def _coverage(
    field: str,
    received: frozenset[OpName],
    expected: frozenset[OpName],
) -> None:
    """Require exact operator-name coverage."""
    if received == expected:
        return

    missing = tuple(sorted(expected - received))
    extra = tuple(sorted(received - expected))
    msg = f"{field} coverage has missing {missing!r} and extra {extra!r}"
    raise ValueError(msg)


def _require_target_function(operator: Operator) -> None:
    """Require a pure binary target endofunctor."""
    if (
        operator.arity == _FUNCTION_ARITY
        and operator.arg_sorts == (operator.sort, operator.sort)
        and not operator.has_payload
        and not operator.is_binder
    ):
        return

    msg = f"target function type {operator.name!r} is not a pure binary endofunctor"
    raise ValueError(msg)


def _parameters(
    target: TermStore,
    sort: str,
    arity: int,
) -> tuple[Term, ...]:
    """Return the positional metavariables for one constructor image."""
    return tuple(target.fvar(sort, f"a{position}") for position in range(arity))


def _free_payloads(store: TermStore, term: Term) -> frozenset[int]:
    """Return the free variable atom payloads in a term."""
    found: set[int] = set()

    def visit(node: Term) -> None:
        """Collect payloads below one node."""
        if store.signature.is_free_variable(node.op) and node.payload is not None:
            found.add(node.payload)

        for child in node.args:
            visit(child)

    visit(term)

    return frozenset(found)


def _parameter_payloads(parameters: tuple[Term, ...]) -> tuple[int, ...]:
    """Return the atom payload of each generated metavariable."""
    held: list[int] = []

    for parameter in parameters:
        if parameter.payload is None:
            msg = f"metavariable {parameter.op.name!r} has no atom payload"
            raise ValueError(msg)

        held.append(parameter.payload)

    return tuple(held)


@final
class Erasure:
    """A frozen signature morphism from categories to semantic types."""

    __slots__ = (
        "_atoms",
        "_category_sort",
        "_constructor_parameters",
        "_constructors",
        "_function_type",
        "_store",
        "_target_sort",
    )

    def __init__(
        self,
        store: TermStore,
        atoms: Mapping[OpName, Term],
        constructors: Mapping[OpName, Term],
        *,
        category_sort: str,
        function_type: OpName,
    ) -> None:
        """Validate and retain a signature morphism declaration."""
        function_name = ensure_name("target function type", function_type)
        held_category_sort = ensure_name("category sort", category_sort)
        target_function = store.signature.by_name(function_name)
        _require_target_function(target_function)

        category_vocabulary = store.signature.vocabulary(held_category_sort)
        source_atoms = frozenset(
            store.signature.constant_core(held_category_sort, name)
            for name in category_vocabulary.names
        )
        source_constructors = _core_operators(store.signature, held_category_sort)
        held_atoms = {
            ensure_name("an atom map key", name): image for name, image in atoms.items()
        }
        held_constructors = {
            ensure_name("a constructor map key", name): image
            for name, image in constructors.items()
        }
        _coverage(
            "atom map",
            frozenset(held_atoms),
            source_atoms,
        )
        _coverage(
            "constructor map",
            frozenset(held_constructors),
            frozenset(operator.name for operator in source_constructors),
        )

        for name, image in held_atoms.items():
            if not isinstance(image, Term) or not store.contains(image):
                msg = f"atom image {name!r} is outside the language store lineage"
                raise ValueError(msg)

            if image.op.sort != target_function.sort:
                msg = (
                    f"atom image {name!r} has sort {image.op.sort!r}, not target "
                    f"sort {target_function.sort!r}"
                )
                raise ValueError(msg)

            if image.free:
                msg = f"atom image {name!r} is not closed"
                raise ValueError(msg)

        parameter_payloads: dict[OpName, tuple[int, ...]] = {}

        for operator in source_constructors:
            name = operator.name
            image = held_constructors[name]

            if not isinstance(image, Term) or not store.contains(image):
                msg = f"constructor image {name!r} is outside the target store lineage"
                raise ValueError(msg)

            if image.op.sort != target_function.sort:
                msg = (
                    f"constructor image {name!r} has sort {image.op.sort!r}, not "
                    f"target sort {target_function.sort!r}"
                )
                raise ValueError(msg)

            parameters = _parameters(store, target_function.sort, operator.arity)
            payloads = _parameter_payloads(parameters)
            allowed = frozenset(payloads)
            extra = _free_payloads(store, image) - allowed

            if extra:
                names = tuple(sorted(store.atom_name(payload) for payload in extra))
                msg = (
                    f"constructor image {name!r} has undeclared metavariables {names!r}"
                )
                raise ValueError(msg)

            parameter_payloads[name] = payloads

        atom_snapshot = MappingProxyType(held_atoms)
        constructor_snapshot = MappingProxyType(held_constructors)
        parameter_snapshot = MappingProxyType(parameter_payloads)
        object.__setattr__(self, "_store", store)
        object.__setattr__(self, "_category_sort", held_category_sort)
        object.__setattr__(self, "_atoms", atom_snapshot)
        object.__setattr__(self, "_constructors", constructor_snapshot)
        object.__setattr__(self, "_constructor_parameters", parameter_snapshot)
        object.__setattr__(self, "_function_type", function_name)
        object.__setattr__(self, "_target_sort", target_function.sort)

    @property
    def store(self) -> TermStore:
        """Return the one language store containing categories and types."""
        return self._store

    @property
    def category_sort(self) -> str:
        """Return the source category sort."""
        return self._category_sort

    @property
    def atoms(self) -> Mapping[OpName, Term]:
        """Return the read-only core-atom images."""
        return self._atoms

    @property
    def constructors(self) -> Mapping[OpName, Term]:
        """Return the read-only constructor-image map."""
        return self._constructors

    @property
    def function_type(self) -> OpName:
        """Return the target function type constructor name."""
        return self._function_type

    @property
    def target_sort(self) -> str:
        """Return the target sort of every image."""
        return self._target_sort

    def image(
        self,
        constructor: OpName,
        arguments: tuple[Term, ...],
        /,
    ) -> Term:
        """Instantiate one constructor's target-type term simultaneously."""
        name = ensure_name("constructor", constructor)

        if name not in self._constructors:
            msg = f"constructor image {name!r} is not declared"
            raise ValueError(msg)

        payloads = self._constructor_parameters[name]

        if len(arguments) != len(payloads):
            msg = (
                f"constructor image {name!r} expects {len(payloads)} arguments, "
                f"not {len(arguments)}"
            )
            raise ValueError(msg)

        for position, argument in enumerate(arguments):
            if not isinstance(argument, Term) or not self._store.contains(argument):
                msg = (
                    f"constructor image {name!r} argument {position} is outside "
                    "the target store lineage"
                )
                raise ValueError(msg)

            if argument.op.sort != self._target_sort:
                msg = (
                    f"constructor image {name!r} argument {position} has sort "
                    f"{argument.op.sort!r}, not target sort {self._target_sort!r}"
                )
                raise ValueError(msg)

        replacements = dict(zip(payloads, arguments, strict=True))

        def instantiate(node: Term) -> Term:
            """Replace only metavariables belonging to this image."""
            if (
                self._store.signature.is_free_variable(node.op)
                and node.payload in replacements
            ):
                return replacements[node.payload]

            rebuilt = tuple(instantiate(child) for child in node.args)

            if rebuilt == node.args:
                return node

            return self._store.op(node.op, *rebuilt, payload=node.payload)

        return instantiate(self._constructors[name])

    def __call__(self, category: Term, /) -> Term:
        """Erase a source category into the target type store."""
        if not isinstance(category, Term) or not self._store.contains(category):
            msg = "category is outside the language store lineage"
            raise ValueError(msg)

        if category.op.sort != self._category_sort:
            msg = (
                f"category has sort {category.op.sort!r}, not category sort "
                f"{self._category_sort!r}"
            )
            raise ValueError(msg)

        memo: dict[Term, Term] = {}

        def erase_core(node: Term) -> Term:
            """Erase one feature-free source subtree."""
            held = memo.get(node)

            if held is not None:
                return held

            if self._store.signature.is_constant(node.op):
                name = self._store.payload_name(node)
                core = self._store.signature.constant_core(self._category_sort, name)
                erased = self._atoms[core]
            else:
                arguments = tuple(erase_core(argument) for argument in node.args)
                erased = self.image(node.op.name, arguments)

            memo[node] = erased

            return erased

        return erase_core(category.core)

    def __copy__(self) -> Self:
        """Return this frozen erasure."""
        return self

    def __deepcopy__(self, memo: dict[int, Self]) -> Self:
        """Return this frozen erasure."""
        del memo

        return self

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        """Refuse field reassignment."""
        del name, value
        msg = "an erasure declaration is immutable"
        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Refuse field deletion."""
        del name
        msg = "an erasure declaration is immutable"
        raise AttributeError(msg)

    @override
    def __reduce__(self) -> Never:
        """Refuse serialization of retained live stores."""
        msg = "an erasure retains live term stores and cannot be pickled"
        raise TypeError(msg)

    @override
    def __repr__(self) -> str:
        """Return an operational declaration summary."""
        return (
            f"Erasure(store={self._store.signature.name!r}, "
            f"category_sort={self._category_sort!r}, atoms={len(self._atoms)}, "
            f"constructors={len(self._constructors)}, "
            f"function_type={self._function_type!r})"
        )
