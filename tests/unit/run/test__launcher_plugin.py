"""Tests for installed sweep-launcher plugins."""

from __future__ import annotations

from importlib.metadata import EntryPoint, EntryPoints
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from tests.support.installed import install
from tests.unit.run.conftest import a_point, a_price, a_profile
from tests.unit.run.test__experiment import an_experiment_document, write_experiment

from lofi.registry import UnknownTagError
from lofi.run import (
    ArrayLauncher,
    ArrayLaunchSpec,
    ExperimentStore,
    ManifestStatus,
    RemoteLauncher,
    RemoteLaunchSpec,
    RunOutcome,
    SerialLauncher,
    SerialLaunchSpec,
    launcher_plugin,
    read_experiment,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi.run import (
        CommandRunner,
        ExperimentPrice,
        SiteProfile,
        SweepPoint,
    )
    from lofi.run._serial import ExecutePoint


class _Context:
    """Every launch resource, recording which ones a launcher asked for."""

    def __init__(self, tmp_path: Path) -> None:
        experiment = read_experiment(
            write_experiment(tmp_path, an_experiment_document())
        )
        self.profile = a_profile(tmp_path / "scratch", tag="plugin")
        self._store = ExperimentStore(
            Path(self.profile.storage.scratch_root), experiment
        )
        self.asked: list[str] = []
        self.runner = RecordingRunner()

    @property
    def store(self) -> ExperimentStore:
        return self._store

    def stop(self) -> bool:
        return False

    @property
    def run(self) -> CommandRunner:
        return self.runner

    def site(self) -> SiteProfile:
        self.asked.append("site")
        return self.profile

    def program(self) -> tuple[str, ...]:
        self.asked.append("program")
        return ("-m", "lofi.run", "task")

    def execute(self) -> ExecutePoint:
        self.asked.append("execute")
        return self._execute

    def price(self) -> ExperimentPrice:
        self.asked.append("price")
        return a_price((a_point(1),), seconds=(1.0, 2.0))

    def _execute(self, point: SweepPoint, /) -> RunOutcome:
        return RunOutcome(
            self._store.directory(point), ManifestStatus.COMPLETED, cached=False
        )


class RecordingRunner:
    def __init__(self) -> None:
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, argv: Sequence[str], /) -> int:
        self.commands.append(tuple(argv))
        return 0


LAUNCHER_POINTS = EntryPoints(
    (
        EntryPoint(
            name="serial",
            value="lofi.run._launcher_plugin:SERIAL_PLUGIN",
            group="lofi.launcher",
        ),
        EntryPoint(
            name="array",
            value="lofi.run._launcher_plugin:ARRAY_PLUGIN",
            group="lofi.launcher",
        ),
        EntryPoint(
            name="remote",
            value="lofi.run._launcher_plugin:REMOTE_PLUGIN",
            group="lofi.launcher",
        ),
    )
)


def test_builtin_launcher_plugins_ask_only_for_the_resources_they_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, "launcher", LAUNCHER_POINTS)
    serial, array, remote = (_Context(tmp_path / name) for name in ("s", "a", "r"))
    declared = ArrayLaunchSpec(chunk_size=4, throttle=7)

    built_serial = launcher_plugin(SerialLaunchSpec()).construct(
        SerialLaunchSpec(), serial
    )
    built_array = launcher_plugin(declared).construct(declared, array)
    built_remote = launcher_plugin(RemoteLaunchSpec()).construct(
        RemoteLaunchSpec(), remote
    )

    assert isinstance(built_serial, SerialLauncher)
    assert serial.asked == ["execute"]
    assert isinstance(built_array, ArrayLauncher)
    assert sorted(array.asked) == ["price", "program", "site"]
    assert isinstance(built_remote, RemoteLauncher)
    assert sorted(remote.asked) == ["program", "site"]
    assert not launcher_plugin(SerialLaunchSpec()).requires_site
    assert launcher_plugin(ArrayLaunchSpec()).requires_site
    assert launcher_plugin(RemoteLaunchSpec()).requires_site


def test_a_launcher_kind_requires_exactly_one_installed_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install(monkeypatch, "launcher", ())

    with pytest.raises(UnknownTagError, match="'serial'"):
        launcher_plugin(SerialLaunchSpec())


def test_only_the_array_plugin_cancels_work_it_left_with_a_scheduler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, "launcher", LAUNCHER_POINTS)
    context = _Context(tmp_path)

    assert not launcher_plugin(SerialLaunchSpec()).cancel(SerialLaunchSpec(), context)
    assert not launcher_plugin(RemoteLaunchSpec()).cancel(RemoteLaunchSpec(), context)
    assert context.runner.commands == []
    assert launcher_plugin(ArrayLaunchSpec()).cancel(ArrayLaunchSpec(), context)
    assert len(context.runner.commands) == 1
    assert context.runner.commands[0][0] == context.profile.submission.cancel_command[0]
