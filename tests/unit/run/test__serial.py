from typing import TYPE_CHECKING

import pytest
from tests.conformance.launcher import LauncherConformance, LauncherFixture
from tests.unit.run.conftest import EXPERIMENT_SEED, a_point, record_manifest

from lofi.run import (
    ManifestStatus,
    RunHandle,
    RunOutcome,
    RunStatus,
    RunStore,
    SerialLauncher,
    SweepPoint,
)

if TYPE_CHECKING:
    from pathlib import Path


class InProcessTraining:
    """Trains a point by opening, closing and recording its manifest."""

    def __init__(self, store: RunStore, *, outcome: ManifestStatus) -> None:
        self._store = store
        self._outcome = outcome
        self.trained: list[SweepPoint] = []

    def __call__(self, point: SweepPoint) -> RunOutcome:
        record_manifest(self._store, point, ManifestStatus.RUNNING)
        self.trained.append(point)
        record_manifest(self._store, point, self._outcome)

        return RunOutcome(self._store.directory(point), self._outcome, cached=False)


def three_points() -> tuple[SweepPoint, SweepPoint, SweepPoint]:
    return (a_point(1), a_point(2), a_point(3))


class TestSerialLauncher(LauncherConformance):
    @pytest.fixture
    def launcher_fixture(self, tmp_path: Path) -> LauncherFixture:
        store = RunStore(tmp_path, freeze_id=None, seed=EXPERIMENT_SEED)
        launcher = SerialLauncher(
            store, execute=InProcessTraining(store, outcome=ManifestStatus.COMPLETED)
        )

        def record(point: SweepPoint, status: ManifestStatus) -> None:
            record_manifest(store, point, status)

        return LauncherFixture(launcher, three_points(), record)


def test_a_three_point_sweep_trains_end_to_end_in_order(tmp_path: Path) -> None:
    store = RunStore(tmp_path, freeze_id=None, seed=EXPERIMENT_SEED)
    training = InProcessTraining(store, outcome=ManifestStatus.COMPLETED)
    launcher = SerialLauncher(store, execute=training)
    points = three_points()

    handles = launcher.submit(points)

    assert training.trained == list(points)
    assert handles == [
        RunHandle(point.point_id, str(store.directory(point))) for point in points
    ]
    assert [launcher.status(handle) for handle in handles] == [RunStatus.COMPLETED] * 3
    assert all(store.is_completed(point) for point in points)


def test_resubmission_skips_exactly_the_completed_points(tmp_path: Path) -> None:
    store = RunStore(tmp_path, freeze_id=None, seed=EXPERIMENT_SEED)
    training = InProcessTraining(store, outcome=ManifestStatus.COMPLETED)
    launcher = SerialLauncher(store, execute=training)
    first, second, third = three_points()
    record_manifest(store, second, ManifestStatus.COMPLETED)
    record_manifest(store, third, ManifestStatus.KILLED)

    handles = launcher.submit((first, second, third))

    assert training.trained == [first, third]
    assert [handle.point_id for handle in handles] == [
        first.point_id,
        third.point_id,
    ]
    assert launcher.submit((first, second, third)) == []
    assert training.trained == [first, third]


def test_a_killed_run_reports_failure(tmp_path: Path) -> None:
    store = RunStore(tmp_path, freeze_id=None, seed=EXPERIMENT_SEED)
    launcher = SerialLauncher(
        store, execute=InProcessTraining(store, outcome=ManifestStatus.KILLED)
    )

    (handle,) = launcher.submit((a_point(),))

    assert launcher.status(handle) is RunStatus.FAILED


def test_a_launcher_told_to_stop_executes_no_point_after_it(tmp_path: Path) -> None:
    store = RunStore(tmp_path, freeze_id=None, seed=EXPERIMENT_SEED)
    training = InProcessTraining(store, outcome=ManifestStatus.COMPLETED)
    launcher = SerialLauncher(
        store, execute=training, stop=lambda: len(training.trained) == 2
    )
    points = three_points()

    handles = launcher.submit(points)

    assert training.trained == list(points[:2])
    assert [handle.point_id for handle in handles] == [
        point.point_id for point in points[:2]
    ]
    assert not store.is_completed(points[2])
