"""Sweep artifacts and run-status bookkeeping shared by the launcher backends."""

import json
import subprocess
from hashlib import sha256
from typing import TYPE_CHECKING, Final, NamedTuple, Protocol, cast

from lofi.run._boundary.site import RunHandle, RunStatus
from lofi.run._boundary.sweep import SweepPoint, point_record, read_point

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from lofi._types import JsonValue

__all__ = [
    "POINTS_NAME",
    "SCRIPT_NAME",
    "SWEEPS_NAME",
    "CommandRunner",
    "ExperimentIdentity",
    "ListedPoints",
    "PointLedger",
    "PointStore",
    "RunDirectories",
    "UnknownHandleError",
    "pending_points",
    "read_points",
    "subprocess_output",
    "subprocess_runner",
    "sweep_directory",
    "sweep_id",
    "task_count",
    "task_points",
    "write_points",
]

type CommandRunner = Callable[[Sequence[str]], int]
"""Runs one local command given as an argument vector and returns its exit status."""

SWEEPS_NAME: Final = "sweeps"
"""The directory under a store's scope that holds one directory per submission."""

POINTS_NAME: Final = "points.json"
"""The file inside a sweep directory that lists the submitted points."""

SCRIPT_NAME: Final = "submit.sh"
"""The file inside a sweep directory that holds the submission script."""


class ExperimentIdentity(NamedTuple):
    """The experiment, by name, and the freeze its points run under."""

    experiment: str
    freeze_id: str


class ListedPoints(NamedTuple):
    """The points a submission lists, the experiment and the script it names."""

    identity: ExperimentIdentity
    points: tuple[SweepPoint, ...]
    script: str | None


class RunDirectories(Protocol):
    """The directories points run in, and what each records of its run."""

    @property
    def root(self) -> Path:
        """Return the root a task given the same freeze addresses the store by."""
        ...

    @property
    def scope(self) -> Path:
        """Return the directory the store's runs and submissions lie under."""
        ...

    def directory(self, point: SweepPoint, /) -> Path:
        """Return the directory a point is addressed by.

        Parameters
        ----------
        point : SweepPoint
            The point whose identifier names the directory.
        """
        ...

    def is_completed(self, point: SweepPoint, /) -> bool:
        """Return whether a point's run is completed.

        Parameters
        ----------
        point : SweepPoint
            The point whose directory is read.
        """
        ...

    def is_final(self, point: SweepPoint, /) -> bool:
        """Return whether a point's outcome is settled, so it is not offered again.

        Parameters
        ----------
        point : SweepPoint
            The point whose directory is read.
        """
        ...

    def status(self, point: SweepPoint, /) -> RunStatus:
        """Return the run status a point's directory records.

        Parameters
        ----------
        point : SweepPoint
            The point whose directory is read.
        """
        ...


class PointStore(RunDirectories, Protocol):
    """The directories of one experiment's points, which it submits under its name."""

    @property
    def identity(self) -> ExperimentIdentity:
        """Return the experiment and freeze the store's points are submitted for."""
        ...


def subprocess_output(argv: Sequence[str], /) -> str:
    """Run one local command and return what it prints.

    Parameters
    ----------
    argv : Sequence[str]
        The argument vector, the profile's declared quota command; never shell
        text.

    Raises
    ------
    OSError
        If the command cannot be run or exits nonzero.
    """
    finished = subprocess.run(list(argv), capture_output=True, check=False, text=True)

    if finished.returncode != 0:
        msg = (
            f"the command {list(argv)} exited {finished.returncode}: {finished.stderr}"
        )
        raise OSError(msg)

    return finished.stdout


def subprocess_runner(argv: Sequence[str], /) -> int:
    """Run one local command and return its exit status.

    Parameters
    ----------
    argv : Sequence[str]
        The argument vector, the profile's declared submit or transport command
        followed by its arguments; never shell text.

    Returns
    -------
    int
        The process's exit status.
    """
    return subprocess.run(list(argv), check=False).returncode


class UnknownHandleError(LookupError):
    """A run handle the launcher never issued."""

    def __init__(self, handle: RunHandle) -> None:
        self.handle = handle
        super().__init__(f"the launcher did not issue {handle!r}")


def sweep_id(points: Sequence[SweepPoint], /) -> str:
    """Return the identifier of a submission over its points in order.

    Parameters
    ----------
    points : Sequence[SweepPoint]
        The submitted points.

    Returns
    -------
    str
        Sixteen hexadecimal digits that differ whenever the point ids or
        their order differ.
    """
    return sha256("\n".join(point.point_id for point in points).encode()).hexdigest()[
        :16
    ]


def sweep_directory(store: RunDirectories, points: Sequence[SweepPoint], /) -> Path:
    """Return the directory a submission's artifacts are written under.

    The directory lies under the store's scope, so a store addressed by a
    freeze writes its submissions apart from every other freeze's.

    Parameters
    ----------
    store : RunDirectories
        The store whose scope the directory lies under.
    points : Sequence[SweepPoint]
        The submitted points.
    """
    return store.scope / SWEEPS_NAME / sweep_id(points)


def pending_points(
    store: RunDirectories, points: Sequence[SweepPoint], /
) -> tuple[SweepPoint, ...]:
    """Return the points whose outcome is not settled, in their given order.

    A completed point and a point whose failure is final are not pending.

    Parameters
    ----------
    store : RunDirectories
        The store whose records decide each outcome.
    points : Sequence[SweepPoint]
        The points offered for submission.
    """
    return tuple(point for point in points if not store.is_final(point))


def task_count(point_count: int, chunk_size: int, /) -> int:
    """Return the number of tasks that cover the points in chunks.

    Parameters
    ----------
    point_count : int
        The number of points.
    chunk_size : int
        The number of points one task runs, at least one.

    Raises
    ------
    ValueError
        If the chunk size is not positive.
    """
    if chunk_size < 1:
        msg = f"the chunk size is not positive: {chunk_size!r}"
        raise ValueError(msg)

    return -(-point_count // chunk_size)


def task_points(
    points: Sequence[SweepPoint], task: int, chunk_size: int, /
) -> tuple[SweepPoint, ...]:
    """Return the points one task runs.

    Parameters
    ----------
    points : Sequence[SweepPoint]
        The submitted points in order.
    task : int
        The task index, counted from zero.
    chunk_size : int
        The number of points one task runs, at least one.

    Raises
    ------
    ValueError
        If the chunk size is not positive, or the task index lies outside
        the tasks that cover the points.
    """
    tasks = task_count(len(points), chunk_size)

    if not 0 <= task < tasks:
        msg = f"the task index {task!r} lies outside the {tasks} tasks"
        raise ValueError(msg)

    return tuple(points[task * chunk_size : (task + 1) * chunk_size])


def write_points(
    path: Path,
    points: Sequence[SweepPoint],
    /,
    *,
    identity: ExperimentIdentity,
    script: str | None = None,
) -> None:
    """Write the points a submission lists, in order, with their experiment.

    Parameters
    ----------
    path : Path
        The file written, whose parent directories are created.
    points : Sequence[SweepPoint]
        The submitted points.
    identity : ExperimentIdentity
        The experiment and freeze the points are submitted for.
    script : str | None
        The version of the submission script that runs the points, or None
        for points no script submits.
    """
    listed: list[JsonValue] = [point_record(point) for point in points]
    experiment: JsonValue = {
        "experiment": identity.experiment,
        "freeze_id": identity.freeze_id,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"experiment": experiment, "points": listed, "script": script},
            allow_nan=False,
            indent=1,
        ),
        encoding="utf-8",
        newline="\n",
    )


def _field(entry: dict[str, JsonValue], name: str) -> JsonValue:
    """Return a named field of a listed point."""
    if name not in entry:
        msg = f"a listed point lacks its {name}"
        raise TypeError(msg)

    return entry[name]


def read_points(path: Path, /) -> ListedPoints:
    """Return the points a submission lists, in order, with their experiment.

    Parameters
    ----------
    path : Path
        The file a submission wrote.

    Raises
    ------
    TypeError
        If the file names no experiment and freeze, or an entry has a field of
        the wrong shape.
    ValueError
        If an entry's id is not the id of its configuration and seed.
    """
    document = cast("JsonValue", json.loads(path.read_text(encoding="utf-8")))

    if not isinstance(document, dict):
        msg = "the file does not list points"
        raise TypeError(msg)

    listed = _field(document, "points")

    if not isinstance(listed, list):
        msg = "the file does not list points"
        raise TypeError(msg)

    match _field(document, "experiment"):
        case {"experiment": str() as experiment, "freeze_id": str() as freeze_id}:
            identity = ExperimentIdentity(experiment, freeze_id)

        case named:
            msg = f"the file names no experiment and freeze: {named!r}"
            raise TypeError(msg)

    script = _field(document, "script")

    if script is not None and not isinstance(script, str):
        msg = f"the file names no submission script version: {script!r}"
        raise TypeError(msg)

    return ListedPoints(identity, tuple(read_point(entry) for entry in listed), script)


class PointLedger:
    """Points a launcher submitted and the run status each is in.

    A point's status is the status its store records. A point whose command
    the launcher saw exit with a nonzero status is failed whatever its store
    records.

    Parameters
    ----------
    store : RunDirectories
        The store whose records are read.
    """

    __slots__ = ("_failed", "_points", "_store")

    def __init__(self, store: RunDirectories) -> None:
        self._store = store
        self._points: dict[RunHandle, SweepPoint] = {}
        self._failed: set[str] = set()

    def record(self, point: SweepPoint, *, backend_id: str) -> RunHandle:
        """Record a submitted point and return its handle.

        Parameters
        ----------
        point : SweepPoint
            The submitted point.
        backend_id : str
            The identifier the backend gave the submission.
        """
        handle = RunHandle(point.point_id, backend_id)
        self._points[handle] = point

        return handle

    def fail(self, point: SweepPoint) -> None:
        """Record that a submitted point's command exited with a nonzero status.

        Parameters
        ----------
        point : SweepPoint
            The point whose command failed.
        """
        self._failed.add(point.point_id)

    def status(self, handle: RunHandle) -> RunStatus:
        """Return the run status of a submitted point.

        Parameters
        ----------
        handle : RunHandle
            A handle this ledger returned.

        Raises
        ------
        UnknownHandleError
            If the complete handle was never issued here.
        """
        point = self._points.get(handle)

        if point is None:
            raise UnknownHandleError(handle)

        if handle.point_id in self._failed:
            return RunStatus.FAILED

        return self._store.status(point)
