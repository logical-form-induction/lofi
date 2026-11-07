"""Tests for the dataset-declaration public surface."""

from lofi.spec.datasets import (
    AssignUnknownWordSpec,
    CcgbankDatasetSpec,
    ChaosNliDatasetSpec,
    CorpusReleaseSpec,
    FracasDatasetSpec,
    MnliDatasetSpec,
    PmbCcgDatasetSpec,
    PmbDrsDatasetSpec,
    SkipUnknownWordSpec,
    SnliDatasetSpec,
    UnknownWordSpec,
    UnliDatasetSpec,
    ccgbank_grammar,
)
from lofi.spec.datasets.ccgbank import (
    AssignUnknownWordSpec as DefinedAssignUnknownWordSpec,
)
from lofi.spec.datasets.ccgbank import CcgbankDatasetSpec as DefinedCcgbankDatasetSpec
from lofi.spec.datasets.ccgbank import (
    SkipUnknownWordSpec as DefinedSkipUnknownWordSpec,
)
from lofi.spec.datasets.ccgbank import UnknownWordSpec as DefinedUnknownWordSpec
from lofi.spec.datasets.ccgbank import ccgbank_grammar as defined_ccgbank_grammar
from lofi.spec.datasets.collected import (
    ChaosNliDatasetSpec as DefinedChaosNliDatasetSpec,
)
from lofi.spec.datasets.collected import CorpusReleaseSpec as DefinedCorpusReleaseSpec
from lofi.spec.datasets.collected import FracasDatasetSpec as DefinedFracasDatasetSpec
from lofi.spec.datasets.collected import MnliDatasetSpec as DefinedMnliDatasetSpec
from lofi.spec.datasets.collected import PmbCcgDatasetSpec as DefinedPmbCcgDatasetSpec
from lofi.spec.datasets.collected import PmbDrsDatasetSpec as DefinedPmbDrsDatasetSpec
from lofi.spec.datasets.collected import SnliDatasetSpec as DefinedSnliDatasetSpec
from lofi.spec.datasets.collected import UnliDatasetSpec as DefinedUnliDatasetSpec


def test_the_dataset_surface_publishes_the_current_ccgbank_declarations() -> None:
    assert AssignUnknownWordSpec is DefinedAssignUnknownWordSpec
    assert CcgbankDatasetSpec is DefinedCcgbankDatasetSpec
    assert ChaosNliDatasetSpec is DefinedChaosNliDatasetSpec
    assert CorpusReleaseSpec is DefinedCorpusReleaseSpec
    assert FracasDatasetSpec is DefinedFracasDatasetSpec
    assert MnliDatasetSpec is DefinedMnliDatasetSpec
    assert PmbCcgDatasetSpec is DefinedPmbCcgDatasetSpec
    assert PmbDrsDatasetSpec is DefinedPmbDrsDatasetSpec
    assert SkipUnknownWordSpec is DefinedSkipUnknownWordSpec
    assert SnliDatasetSpec is DefinedSnliDatasetSpec
    assert UnliDatasetSpec is DefinedUnliDatasetSpec
    assert UnknownWordSpec is DefinedUnknownWordSpec
    assert ccgbank_grammar is defined_ccgbank_grammar
