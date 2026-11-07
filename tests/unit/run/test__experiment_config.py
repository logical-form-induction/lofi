"""Tests for installed experiment-configuration plugins."""

from __future__ import annotations

from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Literal

import pytest
from tests.support.installed import install

from lofi.registry import UnknownTagError
from lofi.run import ExperimentConfigPlugin, ExperimentConfigSpec, experiment_config


class UnitExperimentConfig(ExperimentConfigSpec):
    """A configuration kind local to the plugin resolver tests."""

    kind: Literal["unit-config"] = "unit-config"


def _build_config(declaration: UnitExperimentConfig, path: Path) -> str:
    return f"{declaration.kind}:{path.name}"


CONFIG_PLUGIN = ExperimentConfigPlugin(
    spec=UnitExperimentConfig,
    build=_build_config,
)
CONFIG_POINT = EntryPoint(
    name="unit-config",
    value=f"{__name__}:CONFIG_PLUGIN",
    group="lofi.experiment_config",
)


def test_an_installed_plugin_builds_the_exact_declared_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install(monkeypatch, "experiment_config", (CONFIG_POINT,))
    declaration = UnitExperimentConfig(path="run.yaml")

    assert CONFIG_PLUGIN.spec is UnitExperimentConfig
    assert experiment_config(declaration, Path("resolved/run.yaml")) == (
        "unit-config:run.yaml"
    )


def test_a_configuration_kind_requires_exactly_one_installed_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install(monkeypatch, "experiment_config", ())

    with pytest.raises(UnknownTagError, match="'unit-config'"):
        experiment_config(UnitExperimentConfig(path="run.yaml"), Path("run.yaml"))
