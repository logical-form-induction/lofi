"""The serial launcher, which executes points one after another in this process."""

from typing import TYPE_CHECKING

from lofi.run._launch import PointLedger, pending_points

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from lofi.run._boundary.site import RunHandle, RunStatus
    from lofi.run._boundary.sweep import SweepPoint
    from lofi.run._launch import RunDirectories
    from lofi.run._store import RunOutcome

__all__ = ["ExecutePoint", "SerialLauncher"]

type ExecutePoint = Callable[[SweepPoint], RunOutcome]
"""Executes one point through a store, in this process, and returns its outcome."""


class SerialLauncher:
    """Executes submitted points in order, in the calling process.

    Each point without a completed run is handed to the executor, which
    opens, executes and closes it through the store before the next point
    starts, so submission returns once every point has a terminal record or
    the stop condition holds after one of them.
    The device a point trains on is whatever its own run specification
    resolves, so a development machine trains on the accelerators it has.

    Parameters
    ----------
    store : RunDirectories
        The store whose records decide which points are pending and what
        status each is in.
    execute : ExecutePoint
        Called once per pending point, in order.
    stop : Callable[[], bool]
        Read after each executed point; once it holds, no further point is
        executed. The default never holds.
    """

    __slots__ = ("_execute", "_ledger", "_stop", "_store")

    def __init__(
        self,
        store: RunDirectories,
        *,
        execute: ExecutePoint,
        stop: Callable[[], bool] = lambda: False,
    ) -> None:
        self._store = store
        self._execute = execute
        self._stop = stop
        self._ledger = PointLedger(store)

    def admitted(self, points: Sequence[SweepPoint], /) -> tuple[SweepPoint, ...]:
        """Return every offered point, since this process executes any of them.

        Parameters
        ----------
        points : Sequence[SweepPoint]
            The points offered, in order.
        """
        return tuple(points)

    def rehearse(self, points: Sequence[SweepPoint], /) -> str:
        """Describe the points this process would execute, and execute none.

        Parameters
        ----------
        points : Sequence[SweepPoint]
            The points offered, of which those already completed are skipped.
        """
        pending = pending_points(self._store, points)

        return f"{len(pending)} points would be executed in this process"

    def submit(self, points: Sequence[SweepPoint], /) -> list[RunHandle]:
        """Execute every point without a completed run, in order, until told to stop.

        Parameters
        ----------
        points : Sequence[SweepPoint]
            The points offered, of which those already completed are skipped.

        Returns
        -------
        list[RunHandle]
            One handle per executed point, whose backend id is its
            directory. A point after the one the stop condition first holds
            after has no handle.
        """
        handles: list[RunHandle] = []

        for point in pending_points(self._store, points):
            outcome = self._execute(point)
            handles.append(
                self._ledger.record(point, backend_id=str(outcome.directory))
            )

            if self._stop():
                break

        return handles

    def status(self, handle: RunHandle, /) -> RunStatus:
        """Return the status the store records for an executed point.

        Parameters
        ----------
        handle : RunHandle
            A handle this launcher returned.

        Raises
        ------
        UnknownHandleError
            If the handle names a point this launcher never executed.
        """
        return self._ledger.status(handle)
