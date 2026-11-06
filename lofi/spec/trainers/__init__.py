"""Trainer declarations."""

from lofi.spec.trainers.checkpoints import CheckpointPolicySpec
from lofi.spec.trainers.composed import (
    AlternatingTrainerSpec,
    StagedTrainerSpec,
    StageSpec,
)
from lofi.spec.trainers.distance import DepthCurriculumSpec, DistanceTrainerSpec
from lofi.spec.trainers.early_stopping import AccuracyStoppingSpec, EarlyStoppingSpec
from lofi.spec.trainers.generation import TermGenerationSpec
from lofi.spec.trainers.parser import ParserTrainerSpec
from lofi.spec.trainers.steps import (
    AlternatingStepSpec,
    ControllerStepSpec,
    GroundedStepSpec,
    JudgmentStepSpec,
    ParserStepSpec,
    StepSpec,
    SupervisedWarmStartStepSpec,
    TermsStepSpec,
    TokenWarmupStepSpec,
)
from lofi.spec.trainers.terms import TermsTrainerSpec

__all__ = [
    "AccuracyStoppingSpec",
    "AlternatingStepSpec",
    "AlternatingTrainerSpec",
    "CheckpointPolicySpec",
    "ControllerStepSpec",
    "DepthCurriculumSpec",
    "DistanceTrainerSpec",
    "EarlyStoppingSpec",
    "GroundedStepSpec",
    "JudgmentStepSpec",
    "ParserStepSpec",
    "ParserTrainerSpec",
    "StageSpec",
    "StagedTrainerSpec",
    "StepSpec",
    "SupervisedWarmStartStepSpec",
    "TermGenerationSpec",
    "TermsStepSpec",
    "TermsTrainerSpec",
    "TokenWarmupStepSpec",
]
