"""Conformance laws for sweep launchers."""

from typing import TYPE_CHECKING, ClassVar, NamedTuple

import pytest

from lofi.run import Launcher, ManifestStatus, RunHandle, RunStatus, UnknownHandleError

if TYPE_CHECKING:
    from collections.abc import Callable

    from lofi.run import SweepPoint

__all__ = ["LauncherConformance", "LauncherFixture"]


class LauncherFixture(NamedTuple):
    """A launcher and three points used by its conformance laws.

    Attributes
    ----------
    launcher : Launcher
        The launcher under test.
    points : tuple[SweepPoint, SweepPoint, SweepPoint]
        Three distinct points in submission order.
    record_manifest : Callable[[SweepPoint, ManifestStatus], None]
        Records one status in the store the launcher reads.
    """

    launcher: Launcher
    points: tuple[SweepPoint, SweepPoint, SweepPoint]
    record_manifest: Callable[[SweepPoint, ManifestStatus], None]


class LauncherConformance:
    """Conformance laws shared by launcher implementations."""

    pytestmark: ClassVar[pytest.MarkDecorator] = pytest.mark.conformance

    def test_the_launcher_answers_the_declared_port(
        self, launcher_fixture: LauncherFixture
    ) -> None:
        assert isinstance(launcher_fixture.launcher, Launcher)

    def test_submission_returns_one_handle_per_point_in_order(
        self, launcher_fixture: LauncherFixture
    ) -> None:
        handles = launcher_fixture.launcher.submit(launcher_fixture.points)

        assert [handle.point_id for handle in handles] == [
            point.point_id for point in launcher_fixture.points
        ]
        assert len(set(handles)) == len(handles)

    def test_admission_keeps_an_ordered_subset_of_the_offered_points(
        self, launcher_fixture: LauncherFixture
    ) -> None:
        offered = launcher_fixture.points
        admitted = launcher_fixture.launcher.admitted(offered)
        order = [point.point_id for point in offered]

        assert [point.point_id for point in admitted] == [
            point_id
            for point_id in order
            if point_id in {point.point_id for point in admitted}
        ]

    def test_rehearsal_describes_the_submission_and_submits_nothing(
        self, launcher_fixture: LauncherFixture
    ) -> None:
        described = launcher_fixture.launcher.rehearse(launcher_fixture.points)
        handles = launcher_fixture.launcher.submit(launcher_fixture.points)

        assert described.strip()
        assert [handle.point_id for handle in handles] == [
            point.point_id for point in launcher_fixture.points
        ]

    def test_resubmission_skips_exactly_the_completed_points(
        self, launcher_fixture: LauncherFixture
    ) -> None:
        first, completed, third = launcher_fixture.points
        launcher_fixture.record_manifest(completed, ManifestStatus.COMPLETED)

        handles = launcher_fixture.launcher.submit((first, completed, third))

        assert [handle.point_id for handle in handles] == [
            first.point_id,
            third.point_id,
        ]

    def test_status_tracks_manifests_and_refuses_a_foreign_handle(
        self, launcher_fixture: LauncherFixture
    ) -> None:
        handles = launcher_fixture.launcher.submit(launcher_fixture.points)
        statuses = (
            ManifestStatus.RUNNING,
            ManifestStatus.COMPLETED,
            ManifestStatus.KILLED,
        )

        for point, status in zip(launcher_fixture.points, statuses, strict=True):
            launcher_fixture.record_manifest(point, status)

        assert [launcher_fixture.launcher.status(handle) for handle in handles] == [
            RunStatus.RUNNING,
            RunStatus.COMPLETED,
            RunStatus.FAILED,
        ]

        foreign = RunHandle(handles[0].point_id, f"foreign-{handles[0].backend_id}")

        with pytest.raises(UnknownHandleError):
            launcher_fixture.launcher.status(foreign)
