"""Finite carriers and capped higher-order function spaces."""

from hashlib import sha256
from random import Random
from types import MappingProxyType
from typing import TYPE_CHECKING, final, override

from lofi._checks import ensure_name, ensure_non_negative, ensure_positive
from lofi._frozen import FrozenValue

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi.terms import Term, TermStore

__all__ = [
    "Domain",
    "Element",
    "Frame",
    "FrameTypeError",
    "FunctionValue",
    "ModelValue",
]

type ModelValue = bool | float | Element | FunctionValue
"""A value in a finite higher-order frame."""

type _TypeKey = tuple[str, str | int | None, tuple["_TypeKey", ...]]


class FrameTypeError(ValueError):
    """Report a type whose domain a finite frame cannot construct."""


@final
class Element(FrozenValue):
    """One indexed member of an atomic carrier."""

    _noun = "frame element"
    __slots__ = ("_index", "_type")

    def __init__(self, type_name: str, index: int) -> None:
        object.__setattr__(self, "_type", ensure_name("an atomic type", type_name))
        object.__setattr__(
            self, "_index", ensure_non_negative("a carrier index", index)
        )

    @property
    def type_name(self) -> str:
        """Return the declared atomic type."""
        return self._type

    @property
    def index(self) -> int:
        """Return the element's zero-based carrier index."""
        return self._index

    @override
    def __eq__(self, other: object) -> bool:
        return (
            type(other) is Element
            and self._type == other._type
            and self._index == other._index
        )

    @override
    def __hash__(self) -> int:
        return hash((Element, self._type, self._index))

    @override
    def __repr__(self) -> str:
        return f"Element({self._type!r}, {self._index!r})"


@final
class FunctionValue(FrozenValue):
    """A unary function with an explicit knownness mask over its finite domain."""

    _noun = "finite function"
    __slots__ = ("_domain", "_known", "_outputs")

    def __init__(
        self,
        domain: tuple[ModelValue, ...],
        outputs: tuple[ModelValue, ...],
        known: tuple[bool, ...],
    ) -> None:
        held_domain = tuple(domain)
        held_outputs = tuple(outputs)
        held_known = tuple(known)

        if not held_domain:
            msg = "a finite function has a nonempty domain"
            raise ValueError(msg)

        if len(held_domain) != len(held_outputs) or len(held_domain) != len(held_known):
            msg = (
                "a finite function has one output and known flag per domain element: "
                f"{len(held_domain)} elements, {len(held_outputs)} outputs and "
                f"{len(held_known)} flags"
            )
            raise ValueError(msg)

        if any(not isinstance(value, bool) for value in held_known):
            msg = "a finite function knownness mask contains only bools"
            raise TypeError(msg)

        if len(set(held_domain)) != len(held_domain):
            msg = "a finite function domain has distinct elements"
            raise ValueError(msg)

        object.__setattr__(self, "_domain", held_domain)
        object.__setattr__(self, "_outputs", held_outputs)
        object.__setattr__(self, "_known", held_known)

    @property
    def domain(self) -> tuple[ModelValue, ...]:
        """Return the arguments in table order."""
        return self._domain

    @property
    def outputs(self) -> tuple[ModelValue, ...]:
        """Return one result per domain member."""
        return self._outputs

    @property
    def known(self) -> tuple[bool, ...]:
        """Return whether each tabulated result is known."""
        return self._known

    def __call__(self, argument: ModelValue, /) -> ModelValue:
        """Return the result tabulated for an argument."""
        try:
            position = self._domain.index(argument)
        except ValueError:
            msg = f"{argument!r} lies outside the function domain"
            raise ValueError(msg) from None

        if not self._known[position]:
            msg = f"the function value at {argument!r} is unknown"
            raise ValueError(msg)

        return self._outputs[position]

    @override
    def __eq__(self, other: object) -> bool:
        return (
            type(other) is FunctionValue
            and self._domain == other._domain
            and self._outputs == other._outputs
            and self._known == other._known
        )

    @override
    def __hash__(self) -> int:
        return hash((FunctionValue, self._domain, self._outputs, self._known))

    @override
    def __repr__(self) -> str:
        return f"FunctionValue({self._domain!r}, {self._outputs!r}, {self._known!r})"


@final
class Domain(FrozenValue):
    """A finite type domain and whether it is the complete frame."""

    _noun = "frame domain"
    __slots__ = ("_full", "_values")

    def __init__(self, values: tuple[ModelValue, ...], /, *, full: bool) -> None:
        held = tuple(values)

        if not held:
            msg = "a frame domain is nonempty"
            raise ValueError(msg)

        if len(set(held)) != len(held):
            msg = "a frame domain contains distinct values"
            raise ValueError(msg)

        if not isinstance(full, bool):
            msg = f"a full-frame flag is bool, not {full!r}"
            raise TypeError(msg)

        object.__setattr__(self, "_values", held)
        object.__setattr__(self, "_full", full)

    @property
    def values(self) -> tuple[ModelValue, ...]:
        """Return the domain values in stable order."""
        return self._values

    @property
    def full(self) -> bool:
        """Return whether every possible value is present."""
        return self._full

    @override
    def __eq__(self, other: object) -> bool:
        return (
            type(other) is Domain
            and self._values == other._values
            and self._full == other._full
        )

    @override
    def __hash__(self) -> int:
        return hash((Domain, self._values, self._full))


def _type_key(store: TermStore, term: Term, /) -> _TypeKey:
    payload: str | int | None = term.payload

    if payload is not None and store.signature.is_constant(term.op):
        payload = store.payload_name(term)

    return (
        term.op.name,
        payload,
        tuple(_type_key(store, argument) for argument in term.args),
    )


def _table(
    index: int, size: int, outputs: tuple[ModelValue, ...]
) -> tuple[ModelValue, ...]:
    """Decode one function table from a mixed-radix integer."""
    radix = len(outputs)
    table: list[ModelValue] = []

    for _ in range(size):
        table.append(outputs[index % radix])
        index //= radix

    return tuple(table)


def _indices(total: int, count: int, seed: int) -> tuple[int, ...]:
    """Draw distinct table indices without constructing the complete range."""
    random = Random(seed)
    chosen: set[int] = set()

    while len(chosen) < count:
        chosen.add(random.randrange(total))

    return tuple(sorted(chosen))


@final
class Frame(FrozenValue):
    """Finite atomic carriers and higher-order domains under one cap."""

    _noun = "finite frame"
    __slots__ = (
        "_cap",
        "_carriers",
        "_function_type",
        "_pool",
        "_seed",
        "_store",
        "_truth_type",
    )

    def __init__(
        self,
        store: TermStore,
        /,
        *,
        function_type: str,
        truth_type: str,
        carriers: Mapping[str, int],
        full_function_space_cap: int,
        henkin_pool: int,
        seed: int,
    ) -> None:
        held_carriers = {
            ensure_name("an atomic carrier", name): ensure_positive(
                f"carrier {name!r} size", size
            )
            for name, size in carriers.items()
        }
        truth = ensure_name("the truth type", truth_type)

        if truth in held_carriers:
            msg = "the truth algebra owns its carrier"
            raise ValueError(msg)

        object.__setattr__(self, "_store", store)
        object.__setattr__(
            self, "_function_type", ensure_name("the function type", function_type)
        )
        object.__setattr__(self, "_truth_type", truth)
        object.__setattr__(self, "_carriers", MappingProxyType(held_carriers))
        object.__setattr__(
            self,
            "_cap",
            ensure_positive("the full function-space cap", full_function_space_cap),
        )
        object.__setattr__(
            self, "_pool", ensure_positive("the Henkin-pool size", henkin_pool)
        )
        object.__setattr__(self, "_seed", ensure_non_negative("a frame seed", seed))

    def domain(self, type_: Term, /) -> Domain:
        """Build the finite domain of a declared type."""
        if not self._store.contains(type_):
            msg = "a frame type belongs to the frame's term-store lineage"
            raise ValueError(msg)

        if type_.op.name == self._function_type and len(type_.args) == 2:
            argument = self.domain(type_.args[0])
            result = self.domain(type_.args[1])
            total = len(result.values) ** len(argument.values)
            full = total <= self._cap

            if full:
                indices = tuple(range(total))
            else:
                count = min(self._pool, total)
                digest = sha256(
                    repr((self._seed, _type_key(self._store, type_))).encode()
                ).digest()
                indices = _indices(
                    total,
                    count,
                    int.from_bytes(digest[:8], byteorder="big", signed=False),
                )

            values = tuple(
                FunctionValue(
                    argument.values,
                    _table(index, len(argument.values), result.values),
                    (True,) * len(argument.values),
                )
                for index in indices
            )
            return Domain(values, full=full)

        if not type_.op.is_constant or type_.payload is None:
            msg = f"a frame type is atomic or functional, not {type_!r}"
            raise FrameTypeError(msg)

        name = self._store.payload_name(type_)

        if name == self._truth_type:
            return Domain((False, True), full=True)

        size = self._carriers.get(name)

        if size is None:
            msg = f"a frame declares no carrier for atomic type {name!r}"
            raise ValueError(msg)

        return Domain(tuple(Element(name, at) for at in range(size)), full=True)

    def atomic_domain(self, name: str, /) -> tuple[Element, ...]:
        """Return one declared atomic carrier in stable index order."""
        size = self._carriers.get(ensure_name("an atomic carrier", name))

        if size is None:
            msg = f"a frame declares no carrier for atomic type {name!r}"
            raise ValueError(msg)

        return tuple(Element(name, at) for at in range(size))
