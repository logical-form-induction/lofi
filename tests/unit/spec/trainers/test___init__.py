"""Trainer declaration import tests."""

import json
import subprocess
import sys
from typing import TYPE_CHECKING, cast

from lofi.spec import trainers

if TYPE_CHECKING:
    from pathlib import Path

    from lofi._types import JsonValue


def test_trainer_declarations_import_without_runtime_layers(tmp_path: Path) -> None:
    probe = """
import lofi.spec.trainers.parser
import lofi.spec.trainers.checkpoints
import lofi.spec.trainers.early_stopping
import lofi.spec.trainers.distance
import lofi.spec.trainers.composed
import json
import sys

names = (
    "lofi.nn",
    "lofi.run",
    "lofi.train",
    "torch",
    "panproto",
)
print(json.dumps({name: name in sys.modules for name in names}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        check=True,
        capture_output=True,
        encoding="utf-8",
        cwd=tmp_path,
    )
    imported = cast("dict[str, JsonValue]", json.loads(completed.stdout))

    assert not any(imported.values()), imported


def test_the_trainer_package_exports_the_current_variants() -> None:
    assert set(trainers.__all__) == {
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
    }
