import json
import subprocess
import sys
from typing import TYPE_CHECKING

from lofi.run._active_learning_configuration import ActiveLearningConfiguration
from lofi.run._boundary.experiment import (
    ActiveLearningExperimentConfig,
    JudgmentFitExperimentConfig,
)
from lofi.run._experiment_kinds import ACTIVE_LEARNING_PLUGIN, JUDGMENT_FIT_PLUGIN
from lofi.run._judgment_fit_configuration import JudgmentFitConfiguration

if TYPE_CHECKING:
    from pathlib import Path

PROBE = """\
import json
import sys

import lofi.run._experiment_kinds

print(json.dumps(sorted(
    name for name in (
        "lofi.judgment",
        "lofi.run._active_learning_configuration",
        "lofi.run._judgment_fit_configuration",
    )
    if name in sys.modules
)))
"""
"""Report which implementations importing the kind plugins loads."""


def test_importing_the_plugins_loads_no_configuration_implementation() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", PROBE],
        check=True,
        capture_output=True,
        encoding="utf-8",
    )

    assert json.loads(completed.stdout) == []


def test_the_judgment_fit_plugin_builds_its_configuration(tmp_path: Path) -> None:
    path = tmp_path / "fit.yaml"
    declaration = JudgmentFitExperimentConfig(path="fit.yaml")
    built = JUDGMENT_FIT_PLUGIN.construct(declaration, path)

    assert isinstance(built, JudgmentFitConfiguration)
    assert built.kind == "judgment-fit"
    assert built.path == path
    assert JUDGMENT_FIT_PLUGIN.spec is JudgmentFitExperimentConfig


def test_the_active_learning_plugin_builds_its_configuration(tmp_path: Path) -> None:
    path = tmp_path / "active.yaml"
    declaration = ActiveLearningExperimentConfig(path="active.yaml")
    built = ACTIVE_LEARNING_PLUGIN.construct(declaration, path)

    assert isinstance(built, ActiveLearningConfiguration)
    assert built.kind == "active-learning"
    assert built.path == path
    assert ACTIVE_LEARNING_PLUGIN.spec is ActiveLearningExperimentConfig
