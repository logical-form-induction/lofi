"""Evaluation operations whose inputs require optional corpus adapters."""

from lofi.eval._crosslingual_alignment import (
    AlignmentCoverage,
    CrosslingualAlignmentCell,
    CrosslingualAlignmentDiagonal,
    CrosslingualAlignmentMatrix,
    JaccardFloor,
    LanguageAlignment,
    UnavailableAlignmentCell,
    crosslingual_alignment_matrix,
    hypergeometric_jaccard_floor,
)
from lofi.eval._dependencies import (
    DEPENDENCY_MATCH_MODES,
    CompatibilityPattern,
    DependencyArc,
    DependencyCompatibilityFilter,
    DependencyMatchMode,
    DependencyPrediction,
    DependencyScorer,
    DependencyScoreRow,
    DependencyToken,
    parg_gold,
    projected_prediction,
)
from lofi.eval._transfer import (
    SOURCE_CLASSES,
    SourceClass,
    TransferLanguage,
    UnavailableLanguage,
)

__all__ = [
    "DEPENDENCY_MATCH_MODES",
    "SOURCE_CLASSES",
    "AlignmentCoverage",
    "CompatibilityPattern",
    "CrosslingualAlignmentCell",
    "CrosslingualAlignmentDiagonal",
    "CrosslingualAlignmentMatrix",
    "DependencyArc",
    "DependencyCompatibilityFilter",
    "DependencyMatchMode",
    "DependencyPrediction",
    "DependencyScoreRow",
    "DependencyScorer",
    "DependencyToken",
    "JaccardFloor",
    "LanguageAlignment",
    "SourceClass",
    "TransferLanguage",
    "UnavailableAlignmentCell",
    "UnavailableLanguage",
    "crosslingual_alignment_matrix",
    "hypergeometric_jaccard_floor",
    "parg_gold",
    "projected_prediction",
]
