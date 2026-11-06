"""Open constructor cost policies for simplicity objectives."""

from math import log
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Protocol, cast

from lofi._checks import ensure_finite
from lofi._frozen import FrozenValue
from lofi.registry import resolve_plugin
from lofi.spec.losses.simplicity import (
    LogAlphabetSymbolCostSpec,
    PerConstructorSymbolCostSpec,
    SymbolCostSpec,
    UniformSymbolCostSpec,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from lofi.terms import OperatorIndex

__all__ = [
    "DescriptionCost",
    "DescriptionCostPlugin",
    "LogAlphabetDescriptionCost",
    "PerConstructorDescriptionCost",
    "UniformDescriptionCost",
    "description_cost",
]


class DescriptionCost(Protocol):
    """Supplies one immutable cost for every symbol of an operator index."""

    def costs(self, index: OperatorIndex, /) -> tuple[float, ...]:
        """Return costs in symbol-code order, including zero for padding."""
        ...


class UniformDescriptionCost(FrozenValue):
    """Assign one cost to every nonpadding constructor."""

    _noun = "description-cost policy"
    __slots__ = ("_cost",)

    def __init__(self, cost: float) -> None:
        object.__setattr__(self, "_cost", _nonnegative("a uniform symbol cost", cost))

    def costs(self, index: OperatorIndex, /) -> tuple[float, ...]:
        """Return zero for padding and the declared cost elsewhere."""
        return (0.0, *((self._cost,) * (index.width - 1)))


class LogAlphabetDescriptionCost(FrozenValue):
    """Assign the optimal uniform-code length of the operator alphabet."""

    _noun = "description-cost policy"
    __slots__ = ()

    def costs(self, index: OperatorIndex, /) -> tuple[float, ...]:
        """Return natural-log code lengths for every nonpadding symbol."""
        cost = log(index.width - 1)
        return (0.0, *((cost,) * (index.width - 1)))


class PerConstructorDescriptionCost(FrozenValue):
    """Assign each constructor the cost declared under its operator name."""

    _noun = "description-cost policy"
    __slots__ = ("_declared",)

    def __init__(self, costs: Mapping[str, float]) -> None:
        object.__setattr__(
            self,
            "_declared",
            MappingProxyType(
                {
                    name: _nonnegative(f"the cost of constructor {name!r}", value)
                    for name, value in sorted(costs.items())
                }
            ),
        )

    def costs(self, index: OperatorIndex, /) -> tuple[float, ...]:
        """Return declared costs in code order, refusing an incomplete inventory."""
        expected = tuple(operator.name for operator in index.signature.operators)
        missing = tuple(name for name in expected if name not in self._declared)
        extra = tuple(name for name in self._declared if name not in set(expected))

        if missing or extra:
            details: list[str] = []

            if missing:
                details.append(f"missing {list(missing)!r}")

            if extra:
                details.append(f"unknown {list(extra)!r}")

            msg = "per-constructor description costs are " + " and ".join(details)
            raise ValueError(msg)

        return (0.0, *(self._declared[name] for name in expected))


def _nonnegative(name: str, value: float, /) -> float:
    """Return a finite nonnegative cost."""
    held = ensure_finite(name, value)

    if held < 0.0:
        msg = f"{name} is nonnegative, not {value!r}"
        raise ValueError(msg)

    return held


class DescriptionCostPlugin[S: SymbolCostSpec](FrozenValue):
    """Pair one symbol cost declaration with its immutable policy builder."""

    _noun = "description-cost plugin"
    __slots__ = ("_build", "_spec")

    def __init__(self, *, spec: type[S], build: Callable[[S], DescriptionCost]) -> None:
        object.__setattr__(self, "_spec", spec)
        object.__setattr__(self, "_build", build)

    @property
    def spec(self) -> type[S]:
        """Return the exact declaration variant this plugin builds."""
        return self._spec

    def construct(self, declaration: S, /) -> DescriptionCost:
        """Build the policy declared by one validated variant."""
        return self._build(declaration)


def description_cost(declaration: SymbolCostSpec, /) -> DescriptionCost:
    """Build a constructor cost policy through its unique installed plugin."""
    plugin = cast(
        "DescriptionCostPlugin[SymbolCostSpec]",
        resolve_plugin(
            "description_cost",
            cast("str", declaration.kind),
            DescriptionCostPlugin,
            type(declaration),
        ),
    )

    return plugin.construct(declaration)


def _uniform(declaration: UniformSymbolCostSpec, /) -> DescriptionCost:
    """Build a uniform constructor cost policy."""
    return UniformDescriptionCost(declaration.cost)


def _log_alphabet(declaration: LogAlphabetSymbolCostSpec, /) -> DescriptionCost:
    """Build a uniform-code-length constructor cost policy."""
    del declaration
    return LogAlphabetDescriptionCost()


def _per_constructor(declaration: PerConstructorSymbolCostSpec, /) -> DescriptionCost:
    """Build an explicitly keyed constructor cost policy."""
    return PerConstructorDescriptionCost(declaration.costs)


UNIFORM_PLUGIN: Final = DescriptionCostPlugin(
    spec=UniformSymbolCostSpec,
    build=_uniform,
)
LOG_ALPHABET_PLUGIN: Final = DescriptionCostPlugin(
    spec=LogAlphabetSymbolCostSpec,
    build=_log_alphabet,
)
PER_CONSTRUCTOR_PLUGIN: Final = DescriptionCostPlugin(
    spec=PerConstructorSymbolCostSpec,
    build=_per_constructor,
)
