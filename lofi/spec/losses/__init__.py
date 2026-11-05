"""Training-objective declarations."""

from lofi.spec.losses.derivation_entropy import (
    DerivationEntropyDirection,
    DerivationEntropyLossSettings,
    DerivationEntropyLossSpec,
)
from lofi.spec.losses.form_supervision import (
    FormSupervisionLossSettings,
    FormSupervisionLossSpec,
)
from lofi.spec.losses.form_typing import (
    FormTypingLossSettings,
    FormTypingLossSpec,
)
from lofi.spec.losses.inference import (
    InferenceLossSettings,
    InferenceLossSpec,
    InferenceResponseDeclaration,
    InferenceResponseSpec,
    LabelResponseSettings,
    LabelResponseSpec,
    ResponseSettingsDeclaration,
)
from lofi.spec.losses.judgment import JudgmentResponseSettings, JudgmentResponseSpec
from lofi.spec.losses.shaped_proof import (
    ProofShapingSettings,
    ProofShapingSpec,
    ShapedProofLossSettings,
    ShapedProofLossSpec,
)

__all__ = [
    "DerivationEntropyDirection",
    "DerivationEntropyLossSettings",
    "DerivationEntropyLossSpec",
    "FormSupervisionLossSettings",
    "FormSupervisionLossSpec",
    "FormTypingLossSettings",
    "FormTypingLossSpec",
    "InferenceLossSettings",
    "InferenceLossSpec",
    "InferenceResponseDeclaration",
    "InferenceResponseSpec",
    "JudgmentResponseSettings",
    "JudgmentResponseSpec",
    "LabelResponseSettings",
    "LabelResponseSpec",
    "ProofShapingSettings",
    "ProofShapingSpec",
    "ResponseSettingsDeclaration",
    "ShapedProofLossSettings",
    "ShapedProofLossSpec",
]
