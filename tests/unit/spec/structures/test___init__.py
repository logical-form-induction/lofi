"""Tests for the finite-structure declaration public surface."""

from lofi.spec.structures import (
    GivenStructureSourceSpec,
    LearnedSettings,
    LearnedStructureSourceSpec,
    SampledStructureSourceSpec,
    StructureSourceSpec,
)
from lofi.spec.structures.given import (
    GivenStructureSourceSpec as DefinedGivenStructureSourceSpec,
)
from lofi.spec.structures.learned import LearnedSettings as DefinedLearnedSettings
from lofi.spec.structures.learned import (
    LearnedStructureSourceSpec as DefinedLearnedStructureSourceSpec,
)
from lofi.spec.structures.sampled import (
    SampledStructureSourceSpec as DefinedSampledStructureSourceSpec,
)
from lofi.spec.structures.sampled import (
    StructureSourceSpec as DefinedStructureSourceSpec,
)


def test_the_structure_surface_publishes_the_current_declarations() -> None:
    assert GivenStructureSourceSpec is DefinedGivenStructureSourceSpec
    assert LearnedSettings is DefinedLearnedSettings
    assert LearnedStructureSourceSpec is DefinedLearnedStructureSourceSpec
    assert SampledStructureSourceSpec is DefinedSampledStructureSourceSpec
    assert StructureSourceSpec is DefinedStructureSourceSpec
