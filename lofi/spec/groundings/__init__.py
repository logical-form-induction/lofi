"""Inference grounding declarations."""

from lofi.spec.groundings.expected_embedding import (
    ExpectedEmbeddingSettings,
    ExpectedEmbeddingSpec,
)
from lofi.spec.groundings.guided_proof import (
    GuidedProofScorerSettings,
    GuidedProofScorerSpec,
)
from lofi.spec.groundings.neural_proof import (
    NeuralProofScorerSettings,
    NeuralProofScorerSpec,
    RuleTemplate,
    SimilarityKernelSettings,
    SimilarityKernelSpec,
)
from lofi.spec.groundings.pairwise import (
    PairwiseGroundingSettings,
    PairwiseGroundingSpec,
)

__all__ = [
    "ExpectedEmbeddingSettings",
    "ExpectedEmbeddingSpec",
    "GuidedProofScorerSettings",
    "GuidedProofScorerSpec",
    "NeuralProofScorerSettings",
    "NeuralProofScorerSpec",
    "PairwiseGroundingSettings",
    "PairwiseGroundingSpec",
    "RuleTemplate",
    "SimilarityKernelSettings",
    "SimilarityKernelSpec",
]
