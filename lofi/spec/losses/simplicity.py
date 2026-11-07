"""Simplicity-prior loss declaration."""

import enum
from typing import Literal, NamedTuple, override

import didactic.api as dx

from lofi._checks import ensure_finite, ensure_name
from lofi.spec._ports import Dependency
from lofi.spec.components.loss import LossSpec

__all__ = [
    "LogAlphabetSymbolCostSpec",
    "PerConstructorSymbolCostSpec",
    "SimplicityLossSettings",
    "SimplicityLossSpec",
    "SimplicityScope",
    "SymbolCostSpec",
    "UniformSymbolCostSpec",
]


class SymbolCostSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declares how much description length each constructor contributes."""


class UniformSymbolCostSpec(SymbolCostSpec):
    """Assign one declared cost to every nonpadding constructor."""

    kind: Literal["uniform"] = "uniform"
    cost: float = 1.0

    @dx.validates("cost")
    @classmethod
    def _cost_is_nonnegative(cls, value: float) -> float:
        """Validate a finite nonnegative constructor cost."""
        return _nonnegative("a uniform symbol cost", value)


class LogAlphabetSymbolCostSpec(SymbolCostSpec):
    """Price each constructor at the natural log of the alphabet size."""

    kind: Literal["log-alphabet"] = "log-alphabet"


class PerConstructorSymbolCostSpec(SymbolCostSpec):
    """Assign each constructor its declared nonnegative cost."""

    kind: Literal["per-constructor"] = "per-constructor"
    costs: dict[str, float]

    @dx.validates("costs")
    @classmethod
    def _costs_are_complete_declarations(
        cls, values: dict[str, float]
    ) -> dict[str, float]:
        """Validate named finite nonnegative constructor costs."""
        if not values:
            msg = "per-constructor symbol costs declare at least one constructor"
            raise ValueError(msg)

        return {
            ensure_name("a constructor name", name): _nonnegative(
                f"the cost of constructor {name!r}", value
            )
            for name, value in sorted(values.items())
        }


def _nonnegative(name: str, value: float, /) -> float:
    """Return a finite nonnegative declared cost."""
    held = ensure_finite(name, value)

    if held < 0.0:
        msg = f"{name} is nonnegative, not {value!r}"
        raise ValueError(msg)

    return held


class SimplicityScope(enum.StrEnum):
    """Names the chart cells whose terms the prior prices."""

    LEXICON = "lexicon"
    ALL_CELLS = "all-cells"


class SimplicityLossSettings(NamedTuple):
    """The settings a simplicity prior is built from.

    Attributes
    ----------
    readout : str
        The term-readout role whose terms are priced.
    scope : SimplicityScope
        Which chart cells are priced.
    symbol_cost : SymbolCostSpec
        Declared constructor cost policy.
    """

    readout: str
    scope: SimplicityScope
    symbol_cost: SymbolCostSpec


class SimplicityLossSpec(LossSpec):
    """Declares the expected description length of one term readout's terms.

    Attributes
    ----------
    kind : Literal["simplicity"]
        Registry tag.
    readout : str
        Term-readout role supplied by the model graph.
    scope : SimplicityScope
        Which chart cells are priced.
    symbol_cost : SymbolCostSpec
        Constructor cost policy resolved through ``lofi.description_cost``.
    """

    kind: Literal["simplicity"] = "simplicity"
    readout: str
    scope: SimplicityScope = SimplicityScope.LEXICON
    symbol_cost: SymbolCostSpec = dx.field(default_factory=UniformSymbolCostSpec)

    @dx.validates("readout")
    @classmethod
    def _readout_is_named(cls, value: str) -> str:
        """Validate a nonempty term-readout role."""
        return ensure_name("a simplicity readout role", value)

    @override
    def dependencies(self) -> tuple[Dependency, ...]:
        """Return the term-readout dependency."""
        return (Dependency(self.readout, "simplicity-readout"),)

    def settings(self) -> SimplicityLossSettings:
        """Return the settings the plugin builds this objective from."""
        return SimplicityLossSettings(
            readout=self.readout,
            scope=self.scope,
            symbol_cost=self.symbol_cost,
        )
