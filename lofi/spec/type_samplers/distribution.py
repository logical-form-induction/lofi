"""Declarations controlling recursively sampled term shape."""

from typing import Final, Literal, NamedTuple, Self, override

import didactic.api as dx

from lofi._checks import ensure_finite, ensure_positive

__all__ = [
    "ArityBranchingSettings",
    "ArityBranchingSpec",
    "BranchingAritySettings",
    "BranchingAritySpec",
    "ChildDepthSettings",
    "ChildDepthSpec",
    "PowerChildDepthSettings",
    "PowerChildDepthSpec",
    "UniformChildDepthSettings",
    "UniformChildDepthSpec",
]


class UniformChildDepthSettings(NamedTuple):
    """Resolved settings for equal mass over feasible child depths."""


class PowerChildDepthSettings(NamedTuple):
    """Resolved settings for power-weighted feasible child depths."""

    exponent: float


type ChildDepthSettings = UniformChildDepthSettings | PowerChildDepthSettings
"""Resolved settings for any declared child-depth distribution."""


class ChildDepthSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declare relative mass over feasible child depths."""

    def settings(self) -> ChildDepthSettings:
        """Return the resolved child-depth settings."""
        msg = f"{type(self).__name__} is the child-depth union, not a distribution"
        raise TypeError(msg)


class UniformChildDepthSpec(ChildDepthSpec):
    """Give every feasible child depth equal mass."""

    kind: Literal["uniform"] = "uniform"

    @override
    def settings(self) -> UniformChildDepthSettings:
        """Return uniform child-depth settings."""
        return UniformChildDepthSettings()


class PowerChildDepthSpec(ChildDepthSpec):
    """Weight each feasible child depth by a declared power."""

    kind: Literal["power"] = "power"
    exponent: float

    @dx.validates("exponent")
    @classmethod
    def _exponent_is_finite(cls, value: float) -> float:
        """Validate the child-depth exponent."""
        return ensure_finite("a child-depth exponent", value)

    @override
    def settings(self) -> PowerChildDepthSettings:
        """Return power child-depth settings."""
        return PowerChildDepthSettings(self.exponent)


class BranchingAritySettings(NamedTuple):
    """Resolved deepest-child weights for one constructor arity."""

    arity: int
    deepest: tuple[float, ...]


class ArityBranchingSettings(NamedTuple):
    """Resolved deepest-child weights for every declared arity."""

    by_arity: tuple[BranchingAritySettings, ...]


class BranchingAritySpec(dx.Model, extra="forbid"):
    """Declare deepest-child weights for one constructor arity."""

    arity: int
    deepest: tuple[float, ...]

    @dx.validates("arity")
    @classmethod
    def _arity_branches(cls, value: int) -> int:
        """Validate an arity with at least two child positions."""
        held = ensure_positive("a branching arity", value)

        if held < 2:
            msg = f"a branching arity is less than two: {value!r}"
            raise ValueError(msg)

        return held

    @dx.validates("deepest")
    @classmethod
    def _weights_are_nonnegative(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        """Validate finite nonnegative deepest-child weights."""
        checked = tuple(
            ensure_finite(f"a deepest-child weight at position {at}", value)
            for at, value in enumerate(values)
        )

        if any(value < 0.0 for value in checked):
            msg = f"deepest-child weights are negative: {values!r}"
            raise ValueError(msg)

        if sum(checked) <= 0.0:
            msg = "deepest-child weights contain no positive mass"
            raise ValueError(msg)

        return checked

    @dx.model_validator()
    def _one_weight_per_child(self) -> Self:
        """Require one deepest-child weight per child position."""
        if len(self.deepest) != self.arity:
            msg = (
                f"branching arity {self.arity} has {len(self.deepest)} "
                "deepest-child weights"
            )
            raise ValueError(msg)

        return self

    def settings(self) -> BranchingAritySettings:
        """Return the validated arity and its deepest-child weights."""
        return BranchingAritySettings(self.arity, self.deepest)


class ArityBranchingSpec(dx.Model, extra="forbid"):
    """Declare deepest-child weights independently for each arity.

    Omitted arities receive uniform weights. Unary constructors have no choice
    of deepest child and therefore need no row.
    """

    by_arity: tuple[BranchingAritySpec, ...] = ()

    @dx.model_validator()
    def _arities_are_unique(self) -> Self:
        """Require at most one row for each constructor arity."""
        arities = tuple(row.arity for row in self.by_arity)
        repeated = sorted({arity for arity in arities if arities.count(arity) > 1})

        if repeated:
            msg = f"branching arities are declared more than once: {repeated!r}"
            raise ValueError(msg)

        return self

    def settings(self) -> ArityBranchingSettings:
        """Return every validated per-arity branching row."""
        return ArityBranchingSettings(tuple(row.settings() for row in self.by_arity))


_DEFAULT_CHILD_DEPTH: Final = dx.field(default_factory=UniformChildDepthSpec)
"""The shared default child-depth field of generated-term trainers."""

_DEFAULT_BRANCHING: Final = dx.field(default_factory=ArityBranchingSpec)
"""The shared default branching field of generated-term trainers."""
