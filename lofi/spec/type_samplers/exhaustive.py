"""Exhaustive type-sampler declaration."""

from typing import Literal, NamedTuple, override

import didactic.api as dx

from lofi._checks import ensure_finite, ensure_positive
from lofi.spec.components.trainer import TypeSamplerSpec

__all__ = [
    "ExhaustiveTypeSamplerSettings",
    "ExhaustiveTypeSamplerSpec",
    "ExhaustiveWeightingSettings",
    "ExhaustiveWeightingSpec",
    "SizeExhaustiveWeightingSettings",
    "SizeExhaustiveWeightingSpec",
    "UniformExhaustiveWeightingSettings",
    "UniformExhaustiveWeightingSpec",
]


class UniformExhaustiveWeightingSettings(NamedTuple):
    """Resolved settings for equal exhaustive-inventory mass."""


class SizeExhaustiveWeightingSettings(NamedTuple):
    """Resolved settings for size-power exhaustive-inventory mass."""

    exponent: float


type ExhaustiveWeightingSettings = (
    UniformExhaustiveWeightingSettings | SizeExhaustiveWeightingSettings
)
"""Resolved settings for an exhaustive-inventory weighting."""


class ExhaustiveWeightingSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declare relative mass over an exhaustive inventory."""

    def settings(self) -> ExhaustiveWeightingSettings:
        """Return the resolved exhaustive-weighting settings."""
        msg = f"{type(self).__name__} is the weighting union, not a weighting"
        raise TypeError(msg)


class UniformExhaustiveWeightingSpec(ExhaustiveWeightingSpec):
    """Give every exhaustive candidate equal mass."""

    kind: Literal["uniform"] = "uniform"

    @override
    def settings(self) -> UniformExhaustiveWeightingSettings:
        """Return uniform exhaustive-weighting settings."""
        return UniformExhaustiveWeightingSettings()


class SizeExhaustiveWeightingSpec(ExhaustiveWeightingSpec):
    """Weight exhaustive candidates by a power of their node count."""

    kind: Literal["size"] = "size"
    exponent: float = 1.0

    @dx.validates("exponent")
    @classmethod
    def _exponent_is_finite(cls, value: float) -> float:
        """Validate the candidate-size exponent."""
        return ensure_finite("an exhaustive size exponent", value)

    @override
    def settings(self) -> SizeExhaustiveWeightingSettings:
        """Return size-power exhaustive-weighting settings."""
        return SizeExhaustiveWeightingSettings(self.exponent)


class ExhaustiveTypeSamplerSettings(NamedTuple):
    """The settings an exhaustive type sampler is built from.

    Attributes
    ----------
    max_size : int
        Inclusive node bound for the candidate inventory.
    weighting : ExhaustiveWeightingSettings
        Resolved relative mass over the candidate inventory.
    """

    max_size: int
    weighting: ExhaustiveWeightingSettings


class ExhaustiveTypeSamplerSpec(TypeSamplerSpec):
    """Declares sampling from a finite exhaustive type inventory.

    Attributes
    ----------
    kind : Literal["exhaustive"]
        Registry tag.
    max_size : int
        Inclusive node bound for the candidate inventory.
    weighting : ExhaustiveWeightingSpec
        Tagged relative mass over the candidate inventory.
    """

    kind: Literal["exhaustive"] = "exhaustive"
    max_size: int
    weighting: ExhaustiveWeightingSpec = dx.field(
        default_factory=UniformExhaustiveWeightingSpec
    )

    @dx.validates("max_size")
    @classmethod
    def _size_is_positive(cls, value: int) -> int:
        """Validate the candidate-inventory node bound.

        Parameters
        ----------
        value : int
            Bound to validate.
        """
        return ensure_positive("max_size", value)

    def settings(self) -> ExhaustiveTypeSamplerSettings:
        """Return the settings the plugin builds this sampler from."""
        return ExhaustiveTypeSamplerSettings(
            max_size=self.max_size, weighting=self.weighting.settings()
        )
