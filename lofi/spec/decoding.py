"""Closed declarations of term-decoding strategies."""

from typing import Literal, NamedTuple, Protocol, cast

import didactic.api as dx

from lofi._checks import ensure_finite

__all__ = [
    "BeamDecodingSpec",
    "DecodingDeclaration",
    "DecodingSettings",
    "DecodingSpec",
    "ExhaustiveDecodingSpec",
    "FullSamplingSpec",
    "GreedyDecodingSpec",
    "NucleusSamplingSpec",
    "SampleDecodingSpec",
    "SamplingDeclaration",
    "SamplingSettings",
    "SamplingSpec",
    "TopKSamplingSpec",
]


class SamplingSettings(NamedTuple):
    """Resolved restriction on one sampling strategy."""

    kind: Literal["full", "top-k", "nucleus"]
    temperature: float
    k: int | None
    p: float | None


class DecodingSettings(NamedTuple):
    """Resolved settings of one term-decoding strategy."""

    strategy: Literal["greedy", "beam", "exhaustive", "sample"]
    width: int | None
    residual_mass: float | None
    cap: int | None
    sampling: SamplingSettings | None
    draws: int | None
    rescore_tolerance: float | None
    clamp: str | None
    exact_residual: float | None


class SamplingDeclaration(Protocol):
    """A declared sampling restriction."""

    def settings(self) -> SamplingSettings:
        """Return the resolved restriction."""
        ...


def _positive_temperature(_cls: type[dx.Model], value: float) -> float:
    temperature = ensure_finite("a sampling temperature", value)

    if temperature <= 0.0:
        msg = f"a sampling temperature is positive, not {value!r}"
        raise ValueError(msg)

    return temperature


class SamplingSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declares which choices one sampling step keeps."""

    temperature: float = 1.0

    _temperature_is_positive = dx.validates("temperature")(_positive_temperature)


class FullSamplingSpec(SamplingSpec):
    """Keep every legal choice."""

    kind: Literal["full"] = "full"

    def settings(self) -> SamplingSettings:
        """Return the unrestricted sampling settings."""
        return SamplingSettings(self.kind, self.temperature, None, None)


class TopKSamplingSpec(SamplingSpec):
    """Keep the highest-scoring ``k`` legal choices."""

    kind: Literal["top-k"] = "top-k"
    k: int

    __axioms__ = (dx.axiom("k > 0", message="a top-k restriction keeps a positive k"),)

    def settings(self) -> SamplingSettings:
        """Return the top-k sampling settings."""
        return SamplingSettings(self.kind, self.temperature, self.k, None)


class NucleusSamplingSpec(SamplingSpec):
    """Keep the shortest descending prefix reaching mass ``p``."""

    kind: Literal["nucleus"] = "nucleus"
    p: float

    @dx.validates("p")
    @classmethod
    def _mass_is_fractional(cls, value: float) -> float:
        mass = ensure_finite("a nucleus mass p", value)

        if not 0.0 < mass <= 1.0:
            msg = f"a nucleus mass p lies in (0, 1], not at {value!r}"
            raise ValueError(msg)

        return mass

    def settings(self) -> SamplingSettings:
        """Return the nucleus sampling settings."""
        return SamplingSettings(self.kind, self.temperature, None, self.p)


class DecodingDeclaration(Protocol):
    """A declared term-decoding strategy."""

    def settings(self) -> DecodingSettings:
        """Return the resolved strategy settings."""
        ...


class DecodingSpec(dx.TaggedUnion, discriminator="strategy", extra="forbid"):
    """Declares one shared strategy for decoding terms."""


class GreedyDecodingSpec(DecodingSpec):
    """Declare stepwise argmax decoding."""

    strategy: Literal["greedy"] = "greedy"
    clamp: str | None = None

    def settings(self) -> DecodingSettings:
        """Return greedy decoding settings."""
        return DecodingSettings(
            self.strategy, None, None, None, None, None, None, self.clamp, None
        )


class BeamDecodingSpec(DecodingSpec):
    """Declare bounded beam decoding."""

    strategy: Literal["beam"] = "beam"
    width: int
    clamp: str | None = None

    __axioms__ = (dx.axiom("width > 0", message="a beam has positive width"),)

    def settings(self) -> DecodingSettings:
        """Return beam decoding settings."""
        return DecodingSettings(
            self.strategy,
            self.width,
            None,
            None,
            None,
            None,
            None,
            self.clamp,
            None,
        )


class ExhaustiveDecodingSpec(DecodingSpec):
    """Declare enumeration under a residual bound and term cap."""

    strategy: Literal["exhaustive"] = "exhaustive"
    residual_mass: float | None = None
    cap: int
    clamp: str | None = None
    exact_residual: float = 1e-6

    __axioms__ = (dx.axiom("cap > 0", message="an exhaustive cap is positive"),)

    @dx.validates("residual_mass")
    @classmethod
    def _mass_is_a_residual_bound(cls, value: float | None) -> float | None:
        if value is None:
            return None

        mass = ensure_finite("an exhaustive residual mass", value)

        if not 0.0 <= mass < 1.0:
            msg = f"an exhaustive residual mass lies in [0, 1), not at {value!r}"
            raise ValueError(msg)

        return mass

    @dx.validates("exact_residual")
    @classmethod
    def _exactness_is_a_residual(cls, value: float) -> float:
        """Return the residual under which an enumeration counts as exact."""
        mass = ensure_finite("an exactness residual", value)

        if not 0.0 <= mass < 1.0:
            msg = f"an exactness residual lies in [0, 1), not at {value!r}"
            raise ValueError(msg)

        return mass

    def settings(self) -> DecodingSettings:
        """Return exhaustive decoding settings."""
        return DecodingSettings(
            self.strategy,
            None,
            self.residual_mass,
            self.cap,
            None,
            None,
            None,
            self.clamp,
            self.exact_residual,
        )


class SampleDecodingSpec(DecodingSpec):
    """Declare repeated sampling under one restriction."""

    strategy: Literal["sample"] = "sample"
    restriction: SamplingSpec = FullSamplingSpec()
    draws: int = 1
    rescore_tolerance: float
    clamp: str | None = None

    __axioms__ = (dx.axiom("draws > 0", message="a sample takes positive draws"),)

    @dx.validates("rescore_tolerance")
    @classmethod
    def _rescore_tolerance_is_nonnegative(cls, value: float) -> float:
        """Return a finite non-negative sample-then-rescore tolerance."""
        tolerance = ensure_finite("a rescore tolerance", value)

        if tolerance < 0.0:
            msg = f"a rescore tolerance is non-negative, not {value!r}"
            raise ValueError(msg)

        return tolerance

    def settings(self) -> DecodingSettings:
        """Return sampling settings."""
        return DecodingSettings(
            self.strategy,
            None,
            None,
            None,
            cast("SamplingDeclaration", self.restriction).settings(),
            self.draws,
            self.rescore_tolerance,
            self.clamp,
            None,
        )
