"""Finite-structure source declarations."""

from lofi.spec.structures.given import GivenStructureSourceSpec
from lofi.spec.structures.learned import (
    LearnedSettings,
    LearnedStructureSourceSpec,
)
from lofi.spec.structures.sampled import (
    SampledStructureSourceSpec,
    StructureSourceSpec,
)

__all__ = [
    "GivenStructureSourceSpec",
    "LearnedSettings",
    "LearnedStructureSourceSpec",
    "SampledStructureSourceSpec",
    "StructureSourceSpec",
]
