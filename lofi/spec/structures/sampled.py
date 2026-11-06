"""Boundary declaration of deterministically sampled finite structures."""

from typing import Literal, Self

import didactic.api as dx

from lofi._checks import ensure_finite, ensure_name
from lofi.lf.models import SampledSettings

__all__ = ["SampledStructureSourceSpec", "StructureSourceSpec"]


class StructureSourceSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declare where finite-model structures come from."""


class SampledStructureSourceSpec(StructureSourceSpec):
    """Declare an order-invariant seeded draw of finite structures."""

    kind: Literal["sampled"] = "sampled"
    structures: int
    carriers: dict[str, int]
    density: float
    full_function_space_cap: int
    henkin_pool: int
    seed: int
    resample: Literal["once", "per-step"]
    atomic_sampling: str
    denotation_streams: str
    function_sampling: str
    truth_sampling: str

    @dx.validates(
        "atomic_sampling", "denotation_streams", "function_sampling", "truth_sampling"
    )
    @classmethod
    def _sampling_is_named(cls, value: str) -> str:
        """Require an installed sampling capability's entry point name."""
        return ensure_name("a sampling capability", value)

    @dx.model_validator()
    def _settings_are_bounded(self) -> Self:
        """Validate counts, carriers, density and the seed once at the boundary."""
        if self.structures < 1:
            msg = "a sampled source draws at least one structure"
            raise ValueError(msg)

        if self.full_function_space_cap < 1 or self.henkin_pool < 1:
            msg = "function-space cap and Henkin pool are positive"
            raise ValueError(msg)

        if self.seed < 0:
            msg = "a structure seed is nonnegative"
            raise ValueError(msg)

        if not self.carriers:
            msg = "a sampled source declares at least one atomic carrier"
            raise ValueError(msg)

        if any(not name or size < 1 for name, size in self.carriers.items()):
            msg = "every named atomic carrier has positive size"
            raise ValueError(msg)

        density = ensure_finite("a denotation density", self.density)

        if not 0.0 <= density <= 1.0:
            msg = "a denotation density lies in the unit interval"
            raise ValueError(msg)

        return self

    def settings(self) -> SampledSettings:
        """Return the sampled source's frozen interior settings."""
        return SampledSettings(
            atomic_sampling=self.atomic_sampling,
            denotation_streams=self.denotation_streams,
            function_sampling=self.function_sampling,
            truth_sampling=self.truth_sampling,
            structures=self.structures,
            carriers=self.carriers,
            density=self.density,
            full_function_space_cap=self.full_function_space_cap,
            henkin_pool=self.henkin_pool,
            seed=self.seed,
            resample=self.resample,
        )
