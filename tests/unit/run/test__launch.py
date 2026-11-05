import json
import sys
from typing import TYPE_CHECKING

import pytest
from tests.unit.run.conftest import EXPERIMENT_SEED, a_point, record_manifest

from lofi.run import (
    POINTS_NAME,
    SCRIPT_NAME,
    SWEEPS_NAME,
    ExperimentIdentity,
    FailureKind,
    ManifestStatus,
    PointFailure,
    PointLedger,
    RunHandle,
    RunStatus,
    RunStore,
    SweepPoint,
    UnknownHandleError,
    pending_points,
    read_points,
    subprocess_runner,
    sweep_directory,
    sweep_id,
    task_count,
    task_points,
    write_failure,
    write_points,
)

if TYPE_CHECKING:
    from pathlib import Path

    from lofi._types import JsonValue

INFRASTRUCTURE_ATTEMPTS = 3


def three_points() -> tuple[SweepPoint, ...]:
    return (a_point(1), a_point(2), a_point(3))


def test_a_sweep_id_is_stable_over_its_points_and_their_order() -> None:
    points = three_points()

    assert sweep_id(points) == sweep_id(three_points())
    assert len(sweep_id(points)) == 16
    assert sweep_id(points) != sweep_id(points[::-1])
    assert sweep_id(points) != sweep_id(points[:2])


def test_a_sweep_directory_lies_under_the_store_root(tmp_path: Path) -> None:
    store = RunStore(tmp_path, freeze_id=None, seed=EXPERIMENT_SEED)
    points = three_points()

    assert sweep_directory(store, points) == (tmp_path / SWEEPS_NAME / sweep_id(points))
    assert POINTS_NAME != SCRIPT_NAME


def test_a_sweep_directory_lies_under_the_freeze_its_store_is_addressed_by(
    tmp_path: Path,
) -> None:
    points = three_points()
    first = RunStore(tmp_path, freeze_id="freeze/1", seed=EXPERIMENT_SEED)
    second = RunStore(tmp_path, freeze_id="freeze/2", seed=EXPERIMENT_SEED)

    assert sweep_directory(first, points) == (
        tmp_path / "freeze" / "1" / SWEEPS_NAME / sweep_id(points)
    )
    assert sweep_directory(first, points) != sweep_directory(second, points)


def test_pending_points_drop_exactly_the_completed_ones(tmp_path: Path) -> None:
    store = RunStore(tmp_path, freeze_id=None, seed=EXPERIMENT_SEED)
    first, second, third = three_points()
    record_manifest(store, second, ManifestStatus.COMPLETED)
    record_manifest(store, third, ManifestStatus.KILLED)

    assert pending_points(store, (first, second, third)) == (first, third)
    assert pending_points(store, (third, first)) == (third, first)


@pytest.mark.parametrize(
    ("kind", "attempts", "pending"),
    [
        (FailureKind.NUMERICAL, 1, 0),
        (FailureKind.PREPARATION, 1, 0),
        (FailureKind.INFRASTRUCTURE, INFRASTRUCTURE_ATTEMPTS, 0),
        (FailureKind.INTERRUPTED, INFRASTRUCTURE_ATTEMPTS, 1),
        (FailureKind.INTERRUPTED, 1, 1),
    ],
)
def test_pending_points_drop_a_point_whose_failure_is_final(
    tmp_path: Path, kind: FailureKind, attempts: int, pending: int
) -> None:
    store = RunStore(tmp_path, freeze_id=None, seed=EXPERIMENT_SEED)
    point = a_point(1)
    record_manifest(store, point, ManifestStatus.KILLED)
    write_failure(
        store.directory(point),
        PointFailure(kind, "diagnosed", attempts, store.derivation(point)),
    )

    assert pending_points(store, (point,)) == (point,) * pending


@pytest.mark.parametrize(
    ("points", "chunk_size", "tasks"),
    [(0, 1, 0), (1, 1, 1), (3, 1, 3), (3, 2, 2), (4, 2, 2), (4300, 8, 538)],
)
def test_tasks_cover_the_points_in_chunks(
    points: int, chunk_size: int, tasks: int
) -> None:
    assert task_count(points, chunk_size) == tasks


def test_a_chunk_size_below_one_is_refused() -> None:
    with pytest.raises(ValueError, match="chunk size"):
        task_count(3, 0)


def test_task_points_partition_the_points_in_order() -> None:
    points = three_points()

    assert task_points(points, 0, 2) == points[:2]
    assert task_points(points, 1, 2) == points[2:]
    assert [task_points(points, task, 1) for task in range(3)] == [
        (point,) for point in points
    ]

    with pytest.raises(ValueError, match="task index 2"):
        task_points(points, 2, 2)

    with pytest.raises(ValueError, match="task index -1"):
        task_points(points, -1, 2)


IDENTITY = ExperimentIdentity("headline", "freeze/1")

NAMED: dict[str, JsonValue] = {"experiment": "headline", "freeze_id": "freeze/1"}


def test_written_points_read_back_in_order(tmp_path: Path) -> None:
    points = three_points()
    path = tmp_path / "nested" / POINTS_NAME
    write_points(path, points, identity=ExperimentIdentity("headline", "freeze/1"))
    listed = read_points(path)
    read = listed.points

    assert listed.identity == ExperimentIdentity("headline", "freeze/1")
    assert read == points
    assert [point.overrides for point in read] == [point.overrides for point in points]
    assert [point.point_id for point in read] == [point.point_id for point in points]


def test_a_listed_point_whose_id_disagrees_with_its_content_is_refused(
    tmp_path: Path,
) -> None:
    path = tmp_path / POINTS_NAME
    write_points(path, three_points(), identity=IDENTITY)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["points"][1]["seed"] = 9
    path.write_text(json.dumps(document), encoding="utf-8", newline="\n")

    with pytest.raises(ValueError, match="listed id"):
        read_points(path)


def test_a_malformed_points_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / POINTS_NAME
    entry = {
        "point_id": "x",
        "seed": "1",
        "overrides": [],
        "config": {},
        "provenance": None,
    }
    path.write_text(
        json.dumps({"experiment": NAMED, "points": [entry], "script": None}),
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(TypeError, match="seed"):
        read_points(path)

    path.write_text(
        json.dumps(
            {"experiment": NAMED, "points": [{"point_id": "x"}], "script": None}
        ),
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(TypeError, match="missing fields"):
        read_points(path)

    path.write_text(json.dumps([]), encoding="utf-8", newline="\n")

    with pytest.raises(TypeError, match="list points"):
        read_points(path)

    path.write_text(
        json.dumps(
            {"experiment": {"freeze_id": "freeze/1"}, "points": [], "script": None}
        ),
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(TypeError, match="no experiment and freeze"):
        read_points(path)

    path.write_text(
        json.dumps({"experiment": None, "points": [], "script": None}),
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(TypeError, match="no experiment and freeze"):
        read_points(path)

    path.write_text(
        json.dumps({"experiment": NAMED, "points": [], "script": 3}),
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(TypeError, match="submission script version"):
        read_points(path)

    path.write_text(
        json.dumps({"experiment": NAMED, "points": []}), encoding="utf-8", newline="\n"
    )

    with pytest.raises(TypeError, match="script"):
        read_points(path)


def test_a_point_file_without_the_provenance_field_is_refused(tmp_path: Path) -> None:
    point = three_points()[0]
    path = tmp_path / POINTS_NAME
    write_points(path, (point,), identity=IDENTITY)
    document = json.loads(path.read_text(encoding="utf-8"))
    del document["points"][0]["provenance"]
    path.write_text(json.dumps(document), encoding="utf-8", newline="\n")

    with pytest.raises(TypeError, match="provenance"):
        read_points(path)


def test_a_ledger_reads_status_from_the_manifest(tmp_path: Path) -> None:
    store = RunStore(tmp_path, freeze_id=None, seed=EXPERIMENT_SEED)
    ledger = PointLedger(store)
    submitted, running, completed, killed = (a_point(n) for n in range(1, 5))
    handles = {
        point: ledger.record(point, backend_id=f"job:{index}")
        for index, point in enumerate((submitted, running, completed, killed))
    }
    record_manifest(store, running, ManifestStatus.RUNNING)
    record_manifest(store, completed, ManifestStatus.COMPLETED)
    record_manifest(store, killed, ManifestStatus.KILLED)

    assert handles[submitted] == RunHandle(submitted.point_id, "job:0")
    assert ledger.status(handles[submitted]) is RunStatus.SUBMITTED
    assert ledger.status(handles[running]) is RunStatus.RUNNING
    assert ledger.status(handles[completed]) is RunStatus.COMPLETED
    assert ledger.status(handles[killed]) is RunStatus.FAILED


def test_a_failed_command_outranks_the_manifest(tmp_path: Path) -> None:
    store = RunStore(tmp_path, freeze_id=None, seed=EXPERIMENT_SEED)
    ledger = PointLedger(store)
    point = a_point()
    handle = ledger.record(point, backend_id="job:0")
    record_manifest(store, point, ManifestStatus.COMPLETED)
    ledger.fail(point)

    assert ledger.status(handle) is RunStatus.FAILED


def test_a_handle_never_recorded_is_refused(tmp_path: Path) -> None:
    ledger = PointLedger(RunStore(tmp_path, freeze_id=None, seed=EXPERIMENT_SEED))
    handle = RunHandle(a_point().point_id, "job:0")

    with pytest.raises(UnknownHandleError, match=a_point().point_id):
        ledger.status(handle)


def test_a_recorded_point_with_a_foreign_backend_id_is_refused(
    tmp_path: Path,
) -> None:
    ledger = PointLedger(RunStore(tmp_path, freeze_id=None, seed=EXPERIMENT_SEED))
    point = a_point()
    issued = ledger.record(point, backend_id="job:0")
    forged = RunHandle(point.point_id, "job:1")

    with pytest.raises(UnknownHandleError, match=point.point_id):
        ledger.status(forged)

    assert ledger.status(issued) is RunStatus.SUBMITTED


def test_the_subprocess_runner_returns_the_exit_status_of_the_command() -> None:
    assert subprocess_runner([sys.executable, "-c", "raise SystemExit(3)"]) == 3
    assert subprocess_runner([sys.executable, "-c", "pass"]) == 0
