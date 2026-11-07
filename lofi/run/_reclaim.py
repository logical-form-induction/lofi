"""Storage reclamation under a site's quota, cheapest artifacts first.

A site reports how much its scratch root holds and declares the cap that root
is kept under. When the report exceeds the cap, reclamation removes the
declared artifact classes of the experiment's settled points, the class that
costs least to remake first, and counts the bytes each removal frees, since a
site's report may be cached and not move when a file goes. A point that is
running, submitted or awaiting resubmission keeps every file, as does any
file no class matches.
"""

import re
import shutil
from typing import TYPE_CHECKING, NamedTuple

from lofi.run._boundary.manifest import (
    FailureKind,
)
from lofi.run._store import PointStatus

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from lofi.run._boundary.sweep import SweepPoint
    from lofi.run._store import ExperimentStore

__all__ = [
    "Artifact",
    "QuotaExceededError",
    "QuotaReportError",
    "Reclamation",
    "Removed",
    "Usage",
    "reclaim",
    "report_usage",
]

_SETTLED = frozenset(
    {PointStatus.COMPLETED.value, *(kind.value for kind in FailureKind)}
) - {FailureKind.INTERRUPTED.value}
"""The statuses of a point no attempt will write to again."""


class QuotaReportError(ValueError):
    """Signals a quota report that states no amount held."""


class QuotaExceededError(RuntimeError):
    """Signals a scratch root that reclamation cannot bring under its cap.

    Parameters
    ----------
    used : int
        The bytes held after every reclaimable file was removed.
    limit : int
        The cap.
    """

    def __init__(self, used: int, limit: int) -> None:
        self.used = used
        self.limit = limit
        super().__init__(
            f"the scratch root holds {used} bytes after reclamation, above its "
            f"cap of {limit}; nothing is submitted"
        )


class Artifact(NamedTuple):
    """One reclaimable class of a point's files and the cost of remaking it."""

    name: str
    pattern: str
    regeneration_hours: float


class Usage(NamedTuple):
    """How a site reports the amount its scratch root holds."""

    command: tuple[str, ...]
    used_pattern: str
    unit_bytes: int


class Removed(NamedTuple):
    """The files of one class one point gave up."""

    artifact: str
    point_id: str
    files: int
    bytes: int


class Reclamation(NamedTuple):
    """What reclamation removed and where the holdings ended."""

    removed: tuple[Removed, ...]
    used_before: int
    used_after: int
    limit: int


def report_usage(usage: Usage, report: Callable[[Sequence[str]], str], /) -> int:
    """Return the bytes the site's quota report states are held.

    Parameters
    ----------
    usage : Usage
        The quota command, the pattern of its amount and the unit.
    report : Callable[[Sequence[str]], str]
        Runs the command and returns what it prints.

    Raises
    ------
    QuotaReportError
        If the report does not match the declared pattern.
    OSError
        If the command cannot be run.
    """
    printed = report(usage.command)
    match = re.search(usage.used_pattern, printed)

    if match is None:
        msg = f"the quota report states no amount the pattern matches: {printed!r}"
        raise QuotaReportError(msg)

    return round(float(match["used"]) * usage.unit_bytes)


def _footprint(path: Path, /) -> tuple[int, int]:
    """Return the files and bytes a path holds."""
    if path.is_file():
        return 1, path.stat().st_size

    files = tuple(entry for entry in path.rglob("*") if entry.is_file())

    return len(files), sum(entry.stat().st_size for entry in files)


def _remove(path: Path, /) -> None:
    """Remove a file or a directory tree."""
    if path.is_dir():
        shutil.rmtree(path)

    else:
        path.unlink()


def reclaim(
    store: ExperimentStore,
    points: Sequence[SweepPoint],
    artifacts: Sequence[Artifact],
    /,
    *,
    used: int,
    limit: int,
) -> Reclamation:
    """Remove settled points' artifacts, cheapest to remake first, until under the cap.

    Parameters
    ----------
    store : ExperimentStore
        The store whose point directories are reclaimed.
    points : Sequence[SweepPoint]
        The experiment's points, in the order their directories are visited.
    artifacts : Sequence[Artifact]
        The reclaimable classes.
    used : int
        The bytes the site reports held.
    limit : int
        The cap the holdings are brought under.

    Returns
    -------
    Reclamation
        Every removal, in the order made, and the holdings before and after.

    Raises
    ------
    QuotaExceededError
        If the holdings remain above the cap once every reclaimable file of
        every settled point is removed.
    """
    held = used
    removed: list[Removed] = []
    settled = tuple(
        point for point in points if store.entry(point)["status"] in _SETTLED
    )

    for artifact in sorted(artifacts, key=lambda entry: entry.regeneration_hours):
        for point in settled:
            if held <= limit:
                break

            directory = store.directory(point)
            matched = sorted(
                (path for path in directory.glob(artifact.pattern)),
                key=lambda path: len(path.parts),
            )
            files = freed = 0

            for path in matched:
                if not path.exists():
                    continue

                count, size = _footprint(path)
                _remove(path)
                files += count
                freed += size

            if files:
                held -= freed
                removed.append(Removed(artifact.name, point.point_id, files, freed))

    if held > limit:
        raise QuotaExceededError(held, limit)

    return Reclamation(tuple(removed), used, held, limit)
