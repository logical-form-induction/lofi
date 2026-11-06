"""Shared value types."""

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Literal, NamedTuple, Never, NewType, Self, final, override

__all__ = [
    "ClassRole",
    "ComponentId",
    "CompositionRestriction",
    "FrozenJsonValue",
    "HeadPolicy",
    "HeadSelection",
    "ItemId",
    "JsonValue",
    "ModuleSharing",
    "QuantilePopulation",
    "ReadoutId",
    "ReportValue",
    "SharedSubmodule",
    "SubmoduleDeclaration",
    "freeze_json_value",
]


class QuantilePopulation(StrEnum):
    """Enumerate populations over which a frequency quantile is measured."""

    TYPES = "types"
    TOKENS = "tokens"


class ClassRole(StrEnum):
    """Enumerate the ends of a category inventory ordered by class size."""

    CLOSED = "closed"
    OPEN = "open"


type CompositionRestriction = Literal["steedman", "kuhlmann", "unrestricted"]
"""Names the direction restriction a composition rule places on its spine."""

type HeadPolicy = Literal[
    "functor",
    "pass-through",
    "non-absorbing",
    "left-conjunct",
]
"""Policy for selecting a rule instance's lexical head."""


@final
class HeadSelection:
    """A rule's semantic head policy and physical child slot."""

    __slots__ = ("_policy", "_slot")

    def __init__(self, policy: HeadPolicy, slot: int | None) -> None:
        if policy not in {
            "functor",
            "pass-through",
            "non-absorbing",
            "left-conjunct",
        }:
            msg = f"unknown head policy: {policy!r}"
            raise ValueError(msg)

        if isinstance(slot, bool) or not isinstance(slot, int | None):
            msg = "a head slot must be an integer or None"
            raise TypeError(msg)

        if slot is not None and slot < 0:
            msg = "a head slot must be nonnegative"
            raise ValueError(msg)

        object.__setattr__(self, "_policy", policy)
        object.__setattr__(self, "_slot", slot)

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        """Raise AttributeError on field reassignment."""
        del name, value
        msg = "a head selection is immutable"

        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Raise AttributeError on field deletion."""
        del name
        msg = "a head selection is immutable"

        raise AttributeError(msg)

    def __copy__(self) -> Self:
        """Return this immutable selection."""
        return self

    def __deepcopy__(self, memo: dict[int, Self]) -> Self:
        """Return this immutable selection."""
        del memo
        return self

    @override
    def __reduce__(self) -> tuple[type[HeadSelection], tuple[HeadPolicy, int | None]]:
        """Return the selection reconstruction call."""
        return HeadSelection, self._fields()

    @property
    def policy(self) -> HeadPolicy:
        """Return the semantic head policy."""
        return self._policy

    @property
    def slot(self) -> int | None:
        """Return the physical head slot."""
        return self._slot

    def _fields(self) -> tuple[HeadPolicy, int | None]:
        """Return fields in declaration order."""
        return self._policy, self._slot

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another selection has the same fields."""
        if type(other) is not HeadSelection:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the policy and slot."""
        return hash((HeadSelection, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the rebuilding constructor call."""
        return f"HeadSelection{self._fields()!r}"


type JsonValue = (
    str | int | float | bool | list[JsonValue] | dict[str, JsonValue] | None
)
"""JSON-serializable value."""

type ReportValue = int | float | str | bool | Mapping[str, int] | Mapping[str, float]
"""One named measurement a component reports: a count, a measure, a text, a
flag, or a table of counts or of measures keyed by name."""

type FrozenJsonValue = (
    str
    | int
    | float
    | bool
    | tuple[FrozenJsonValue, ...]
    | Mapping[str, FrozenJsonValue]
    | None
)
"""Recursively read-only JSON value."""


def freeze_json_value(value: JsonValue, /) -> FrozenJsonValue:
    """Return a recursively read-only copy of a JSON value.

    Parameters
    ----------
    value : JsonValue
        Value whose arrays and objects become immutable containers.
    """
    if isinstance(value, list):
        return tuple(freeze_json_value(item) for item in value)

    if isinstance(value, dict):
        return MappingProxyType(
            {name: freeze_json_value(item) for name, item in value.items()}
        )

    return value


ItemId = NewType("ItemId", str)
"""Reader-assigned item identifier."""

ComponentId = NewType("ComponentId", str)
"""Built-component identifier within a run."""

ReadoutId = NewType("ReadoutId", str)
"""A published readout identifier."""


class ModuleSharing(NamedTuple):
    """Registered-state semantics for a shared module.

    Attributes
    ----------
    in_parameters : bool
        Whether the shared parameters appear in the consumer's ``parameters()``.
    in_state_dict : bool
        Whether the shared state appears in the consumer's ``state_dict()``.
    """

    in_parameters: bool
    in_state_dict: bool


class SharedSubmodule[M](NamedTuple):
    """A published submodule and its registered-state semantics.

    Attributes
    ----------
    module : M
        Published module.
    sharing : ModuleSharing
        How a consumer registers the module.
    """

    module: M
    sharing: ModuleSharing


class SubmoduleDeclaration(NamedTuple):
    """A selected published submodule and its expected semantics.

    Attributes
    ----------
    name : str
        Key in the producer's published-submodule mapping.
    sharing : ModuleSharing
        Expected parameter and saved-state registration.
    """

    name: str
    sharing: ModuleSharing
