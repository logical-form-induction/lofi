"""Dataset declarations registered by data plugins."""

from lofi.spec.datasets.ccgbank import (
    AssignUnknownWordSpec,
    CcgbankDatasetSpec,
    SkipUnknownWordSpec,
    UnknownWordSpec,
    ccgbank_grammar,
)
from lofi.spec.datasets.collected import (
    ChaosNliDatasetSpec,
    CorpusReleaseSpec,
    CorpusSourceSpec,
    FracasDatasetSpec,
    MnliDatasetSpec,
    PmbCcgDatasetSpec,
    PmbDrsDatasetSpec,
    SnliDatasetSpec,
    UnliDatasetSpec,
)
from lofi.spec.datasets.pmb import PmbGrammarLanguage, pmb_grammar

__all__ = [
    "AssignUnknownWordSpec",
    "CcgbankDatasetSpec",
    "ChaosNliDatasetSpec",
    "CorpusReleaseSpec",
    "CorpusSourceSpec",
    "FracasDatasetSpec",
    "MnliDatasetSpec",
    "PmbCcgDatasetSpec",
    "PmbDrsDatasetSpec",
    "PmbGrammarLanguage",
    "SkipUnknownWordSpec",
    "SnliDatasetSpec",
    "UnknownWordSpec",
    "UnliDatasetSpec",
    "ccgbank_grammar",
    "pmb_grammar",
]
