"""Trainer plugin surface tests."""

import json
import subprocess
import sys
from typing import TYPE_CHECKING, cast

import lofi.training_plugins as plugins

if TYPE_CHECKING:
    from pathlib import Path

    from lofi._types import JsonValue


def test_trainer_plugins_import_without_runtime_layers(tmp_path: Path) -> None:
    probe = """
import json
import sys
import lofi.training_plugins.parser
import lofi.training_plugins.distance
import lofi.training_plugins.alternating
import lofi.training_plugins.mixing
import lofi.training_plugins.staged
import lofi.training_plugins.steps

names = ("lofi.nn", "lofi.run", "lofi.train", "torch", "panproto")
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


def test_the_plugin_package_exports_its_single_current_surface() -> None:
    assert set(plugins.__all__) == {
        "FeaturizerMismatchError",
        "LearnerAssembly",
        "StepBuildContext",
        "StepPlugin",
        "Trainer",
        "TrainerBuildContext",
        "TrainerBuilder",
        "TrainerBuilderTarget",
        "TrainerPlanner",
        "TrainerPlugin",
        "TrainerPresenter",
        "TrainerPricer",
        "TrainingBlueprint",
        "TrainingLength",
        "TrainingUnit",
        "step_plugin",
        "trainer_plugin",
        "training_sampler_stream",
    }
