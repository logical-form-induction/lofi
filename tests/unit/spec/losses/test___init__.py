"""Tests for the training-objective declaration public surface."""

from lofi.spec.losses import (
    FormSupervisionLossSettings,
    FormSupervisionLossSpec,
    FormTypingLossSettings,
    FormTypingLossSpec,
    InferenceLossSettings,
    InferenceLossSpec,
    JudgmentResponseSettings,
    JudgmentResponseSpec,
    LabelResponseSpec,
    ProofShapingSettings,
    ProofShapingSpec,
    ShapedProofLossSettings,
    ShapedProofLossSpec,
)
from lofi.spec.losses.form_supervision import (
    FormSupervisionLossSettings as DefinedFormSupervisionLossSettings,
)
from lofi.spec.losses.form_supervision import (
    FormSupervisionLossSpec as DefinedFormSupervisionLossSpec,
)
from lofi.spec.losses.form_typing import (
    FormTypingLossSettings as DefinedFormTypingLossSettings,
)
from lofi.spec.losses.form_typing import FormTypingLossSpec as DefinedFormTypingLossSpec
from lofi.spec.losses.inference import (
    InferenceLossSettings as DefinedInferenceLossSettings,
)
from lofi.spec.losses.inference import InferenceLossSpec as DefinedInferenceLossSpec
from lofi.spec.losses.inference import LabelResponseSpec as DefinedLabelResponseSpec
from lofi.spec.losses.judgment import (
    JudgmentResponseSettings as DefinedJudgmentResponseSettings,
)
from lofi.spec.losses.judgment import (
    JudgmentResponseSpec as DefinedJudgmentResponseSpec,
)
from lofi.spec.losses.shaped_proof import (
    ProofShapingSettings as DefinedProofShapingSettings,
)
from lofi.spec.losses.shaped_proof import ProofShapingSpec as DefinedProofShapingSpec
from lofi.spec.losses.shaped_proof import (
    ShapedProofLossSettings as DefinedShapedProofLossSettings,
)
from lofi.spec.losses.shaped_proof import (
    ShapedProofLossSpec as DefinedShapedProofLossSpec,
)


def test_the_loss_surface_publishes_the_current_objective_declarations() -> None:
    assert FormSupervisionLossSettings is DefinedFormSupervisionLossSettings
    assert FormSupervisionLossSpec is DefinedFormSupervisionLossSpec
    assert FormTypingLossSettings is DefinedFormTypingLossSettings
    assert FormTypingLossSpec is DefinedFormTypingLossSpec
    assert InferenceLossSettings is DefinedInferenceLossSettings
    assert InferenceLossSpec is DefinedInferenceLossSpec
    assert LabelResponseSpec is DefinedLabelResponseSpec
    assert JudgmentResponseSettings is DefinedJudgmentResponseSettings
    assert JudgmentResponseSpec is DefinedJudgmentResponseSpec
    assert ProofShapingSettings is DefinedProofShapingSettings
    assert ProofShapingSpec is DefinedProofShapingSpec
    assert ShapedProofLossSettings is DefinedShapedProofLossSettings
    assert ShapedProofLossSpec is DefinedShapedProofLossSpec
