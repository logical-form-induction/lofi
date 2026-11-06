import sys
from typing import TYPE_CHECKING

import pytest
from tests.unit.run.conftest import record_manifest
from tests.unit.run.test__experiment import an_experiment_document, write_experiment

from lofi.run import (
    Artifact,
    ExperimentStore,
    ManifestStatus,
    QuotaExceededError,
    QuotaReportError,
    RecordedRows,
    Usage,
    expand_experiment,
    read_experiment,
    reclaim,
    report_usage,
    subprocess_output,
    write_rows,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from lofi.run import SweepPoint

ARTIFACTS = (
    Artifact("fits", "fits/*.json", 6.0),
    Artifact("representations", "representations", 0.25),
)

MEGABYTE = 1_000_000


def write_bytes(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * size)


def settle(store: ExperimentStore, point: SweepPoint) -> None:
    store.claim(point)
    binding = store.binding(point)
    record_manifest(store.runs, point, ManifestStatus.COMPLETED)
    write_rows(
        store.directory(point),
        RecordedRows(binding.freeze_id, binding.seed_derivation, ()),
    )


def fixture_store(tmp_path: Path) -> tuple[ExperimentStore, tuple[SweepPoint, ...]]:
    path = write_experiment(tmp_path, an_experiment_document())
    experiment = read_experiment(path)
    points = expand_experiment(experiment, tmp_path)
    store = ExperimentStore(tmp_path / "scratch", experiment)

    for point in points[:4]:
        settle(store, point)

    store.claim(points[4])
    record_manifest(store.runs, points[4], ManifestStatus.RUNNING)

    for point in points[:5]:
        directory = store.directory(point)
        write_bytes(directory / "representations" / "layer-0.bin", 3 * MEGABYTE)
        write_bytes(directory / "representations" / "layer-1.bin", 2 * MEGABYTE)
        write_bytes(directory / "fits" / "probe.json", MEGABYTE)

    return store, points


def present(store: ExperimentStore, points: Sequence[SweepPoint], name: str) -> int:
    return sum((store.directory(point) / name).exists() for point in points)


def test_reclamation_removes_representations_before_fits_and_stays_under_quota(
    tmp_path: Path,
) -> None:
    store, points = fixture_store(tmp_path)
    used = 40 * MEGABYTE

    reclaimed = reclaim(store, points, ARTIFACTS, used=used, limit=28 * MEGABYTE)

    assert [entry.artifact for entry in reclaimed.removed] == [
        "representations",
        "representations",
        "representations",
    ]
    assert [entry.bytes for entry in reclaimed.removed] == [5 * MEGABYTE] * 3
    assert reclaimed.used_after == 25 * MEGABYTE
    assert reclaimed.used_after <= reclaimed.limit
    assert present(store, points[:5], "fits") == 5
    assert present(store, points[:5], "representations") == 2


def test_reclamation_reaches_the_costlier_class_only_once_the_cheaper_is_spent(
    tmp_path: Path,
) -> None:
    store, points = fixture_store(tmp_path)

    reclaimed = reclaim(
        store, points, ARTIFACTS, used=40 * MEGABYTE, limit=18 * MEGABYTE
    )

    assert [entry.artifact for entry in reclaimed.removed] == [
        *("representations",) * 4,
        *("fits",) * 2,
    ]
    assert reclaimed.used_after == 18 * MEGABYTE
    assert (store.directory(points[4]) / "representations").exists()
    assert (store.directory(points[4]) / "fits" / "probe.json").exists()


def test_a_root_that_cannot_be_brought_under_its_quota_is_refused(
    tmp_path: Path,
) -> None:
    store, points = fixture_store(tmp_path)

    with pytest.raises(QuotaExceededError, match="above its cap"):
        reclaim(store, points, ARTIFACTS, used=40 * MEGABYTE, limit=MEGABYTE)


def test_a_killed_point_awaiting_resubmission_keeps_every_file(tmp_path: Path) -> None:
    store, points = fixture_store(tmp_path)
    record_manifest(store.runs, points[4], ManifestStatus.KILLED)

    with pytest.raises(QuotaExceededError):
        reclaim(store, points, ARTIFACTS, used=40 * MEGABYTE, limit=MEGABYTE)

    assert store.entry(points[4])["status"] == "killed"
    assert (store.directory(points[4]) / "representations").exists()
    assert (store.directory(points[4]) / "fits" / "probe.json").exists()


def test_nothing_is_removed_while_the_root_is_under_its_quota(tmp_path: Path) -> None:
    store, points = fixture_store(tmp_path)

    reclaimed = reclaim(store, points, ARTIFACTS, used=MEGABYTE, limit=MEGABYTE)

    assert reclaimed.removed == ()
    assert present(store, points[:5], "representations") == 5


def test_the_quota_report_is_read_through_its_declared_pattern_and_unit() -> None:
    usage = Usage(
        (sys.executable, "-c", "print('scratch used 2.5 of 3 TB')"),
        r"used (?P<used>[0-9.]+)",
        10**12,
    )

    assert report_usage(usage, subprocess_output) == 25 * 10**11

    with pytest.raises(QuotaReportError, match="no amount"):
        report_usage(
            usage._replace(used_pattern=r"held (?P<used>\d+)"), subprocess_output
        )

    with pytest.raises(OSError, match="exited 3"):
        subprocess_output((sys.executable, "-c", "raise SystemExit(3)"))
