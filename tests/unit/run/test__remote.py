import shlex
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from tests.conformance.launcher import LauncherConformance, LauncherFixture
from tests.unit.run.conftest import (
    SCRATCH_IDENTITY,
    a_point,
    a_scratch_store,
    record_manifest,
)
from tests.unit.run.test__experiment import an_experiment_document, write_experiment

from lofi.run import (
    POINTS_NAME,
    SWEEPS_NAME,
    ExperimentIdentity,
    ExperimentStore,
    ListedPoints,
    ManifestStatus,
    RemoteLauncher,
    RunHandle,
    RunStatus,
    SweepPoint,
    read_experiment,
    read_points,
    sweep_id,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from lofi.run import SiteProfile

PROGRAM = ("-m", "lofi", "sweep-task")


class RecordingTransport:
    """Records every session command and exits with the next scripted status."""

    def __init__(self, exit_statuses: Sequence[int] = ()) -> None:
        self._exit_statuses = list(exit_statuses)
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, argv: Sequence[str]) -> int:
        self.commands.append(tuple(argv))

        if self._exit_statuses:
            return self._exit_statuses.pop(0)

        return 0


def a_remote(
    profile: SiteProfile, *, program: tuple[str, ...], run: RecordingTransport
) -> RemoteLauncher:
    return RemoteLauncher(
        profile, store=a_scratch_store(profile), program=program, run=run
    )


def three_points() -> tuple[SweepPoint, SweepPoint, SweepPoint]:
    return (a_point(1), a_point(2), a_point(3))


class TestRemoteLauncher(LauncherConformance):
    @pytest.fixture
    def launcher_fixture(
        self, make_profile: Callable[[str], SiteProfile]
    ) -> LauncherFixture:
        launcher = a_remote(
            make_profile("conformance"), program=PROGRAM, run=RecordingTransport()
        )

        def record(point: SweepPoint, status: ManifestStatus) -> None:
            record_manifest(launcher.store, point, status)

        return LauncherFixture(launcher, three_points(), record)


def test_each_pending_point_runs_in_its_own_session_in_order(
    make_profile: Callable[[str], SiteProfile],
) -> None:
    profile = make_profile("a")
    transport = RecordingTransport()
    launcher = a_remote(profile, program=PROGRAM, run=transport)
    points = three_points()

    handles = launcher.submit(points)

    listing = launcher.store.scope / SWEEPS_NAME / sweep_id(points) / POINTS_NAME
    assert read_points(listing) == ListedPoints(SCRATCH_IDENTITY, points, None)
    assert len(transport.commands) == 3

    for index, command in enumerate(transport.commands):
        *session, host, line = command
        preamble, invocation = line.rsplit(" && ", 1)

        assert tuple(session) == profile.transport.command
        assert host == profile.transport.host
        assert preamble == " && ".join(profile.environment.preamble)
        assert shlex.split(invocation) == [
            "exec",
            *profile.interpreter.command,
            *PROGRAM,
            str(listing),
            str(index),
            "1",
        ]

    submission = sweep_id(points)
    assert handles == [
        RunHandle(point.point_id, f"{submission}:{index}")
        for index, point in enumerate(points)
    ]
    assert [launcher.status(handle) for handle in handles] == [RunStatus.SUBMITTED] * 3

    record_manifest(launcher.store, points[0], ManifestStatus.COMPLETED)

    assert launcher.status(handles[0]) is RunStatus.COMPLETED


def test_a_session_that_fails_marks_its_point_and_the_next_still_runs(
    make_profile: Callable[[str], SiteProfile],
) -> None:
    transport = RecordingTransport(exit_statuses=(0, 255, 0))
    launcher = a_remote(make_profile("a"), program=PROGRAM, run=transport)
    points = three_points()

    handles = launcher.submit(points)

    for point in points:
        record_manifest(launcher.store, point, ManifestStatus.COMPLETED)

    assert len(transport.commands) == 3
    assert [launcher.status(handle) for handle in handles] == [
        RunStatus.COMPLETED,
        RunStatus.FAILED,
        RunStatus.COMPLETED,
    ]


def test_resubmission_skips_exactly_the_completed_points(
    make_profile: Callable[[str], SiteProfile],
) -> None:
    transport = RecordingTransport()
    launcher = a_remote(make_profile("a"), program=PROGRAM, run=transport)
    first, second, third = three_points()
    record_manifest(launcher.store, second, ManifestStatus.COMPLETED)
    record_manifest(launcher.store, third, ManifestStatus.RUNNING)

    handles = launcher.submit((first, second, third))

    assert [handle.point_id for handle in handles] == [
        first.point_id,
        third.point_id,
    ]
    assert len(transport.commands) == 2
    assert transport.commands[1][-1].endswith(" 1 1")

    record_manifest(launcher.store, first, ManifestStatus.COMPLETED)
    record_manifest(launcher.store, third, ManifestStatus.COMPLETED)

    assert launcher.submit((first, second, third)) == []
    assert len(transport.commands) == 2


def test_two_profiles_differ_only_in_profile_values(
    make_profile: Callable[[str], SiteProfile],
) -> None:
    points = three_points()
    first = RecordingTransport()
    second = RecordingTransport()
    a_remote(make_profile("alpha"), program=PROGRAM, run=first).submit(points)
    a_remote(make_profile("beta"), program=PROGRAM, run=second).submit(points)

    assert first.commands != second.commands
    assert [
        tuple(word.replace("alpha", "beta") for word in command)
        for command in first.commands
    ] == second.commands


def test_an_empty_program_is_refused(
    make_profile: Callable[[str], SiteProfile],
) -> None:
    with pytest.raises(ValueError, match="program is empty"):
        a_remote(make_profile("a"), program=(), run=RecordingTransport())


def test_an_experiment_launcher_names_its_experiment_and_freeze_in_the_points_file(
    make_profile: Callable[[str], SiteProfile], tmp_path: Path
) -> None:
    profile = make_profile("a")
    experiment = read_experiment(write_experiment(tmp_path, an_experiment_document()))
    store = ExperimentStore(Path(profile.storage.scratch_root), experiment)
    transport = RecordingTransport()
    points = three_points()

    RemoteLauncher(profile, store=store, program=PROGRAM, run=transport).submit(points)

    listing = store.scope / SWEEPS_NAME / sweep_id(points) / POINTS_NAME
    assert read_points(listing) == ListedPoints(
        ExperimentIdentity("headline", experiment.freeze_id), points, None
    )
