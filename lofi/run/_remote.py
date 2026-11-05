"""The remote-host launcher, which runs points one at a time over a transport."""

import shlex
from typing import TYPE_CHECKING

from lofi._checks import ensure_name
from lofi.run._array import require_scratch_root
from lofi.run._launch import (
    POINTS_NAME,
    PointLedger,
    pending_points,
    sweep_directory,
    sweep_id,
    write_points,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from lofi.run._boundary.site import RunHandle, RunStatus, SiteProfile
    from lofi.run._boundary.sweep import SweepPoint
    from lofi.run._launch import CommandRunner, PointStore

__all__ = ["RemoteLauncher"]


class RemoteLauncher:
    """Runs pending points in order on a host reached through a transport.

    A submission writes the points file under ``<scope>/sweeps/<id>/``, where
    the scope is the store's, naming the store's experiment and freeze exactly
    as the array launcher's does, and then opens one session per point: the
    profile's transport command, the host, and one shell line that runs the
    profile's preamble lines and then its interpreter on the program with
    three trailing arguments, the points file, the point's index in it and a
    chunk size of one. A session that exits with a nonzero status marks its
    point failed and the next point still runs. Submission returns once every
    session has exited.

    Parameters
    ----------
    profile : SiteProfile
        The resolved site profile every site value is read from.
    store : PointStore
        The store at the profile's scratch root whose records decide which
        points are pending and what status each one run is in.
    program : tuple[str, ...]
        The argument vector, after the interpreter, that runs one task.
    run : CommandRunner
        Runs each session command and returns its exit status.

    Raises
    ------
    ValueError
        If the program is empty or the store does not lie at the profile's
        scratch root.
    """

    __slots__ = ("_ledger", "_profile", "_program", "_run", "_store")

    def __init__(
        self,
        profile: SiteProfile,
        *,
        store: PointStore,
        program: tuple[str, ...],
        run: CommandRunner,
    ) -> None:
        if not program:
            msg = "the program is empty"
            raise ValueError(msg)

        require_scratch_root(profile, store)
        self._profile = profile
        self._program = tuple(ensure_name("program word", word) for word in program)
        self._run = run
        self._store = store
        self._ledger = PointLedger(store)

    @property
    def store(self) -> PointStore:
        """Return the store at the profile's scratch root."""
        return self._store

    def session(self, points: Path, index: int, /) -> tuple[str, ...]:
        """Return the command that runs one listed point on the host.

        Parameters
        ----------
        points : Path
            The points file as the host sees it.
        index : int
            The point's index in the file.
        """
        invocation = shlex.join(
            (*self._profile.interpreter.command, *self._program, str(points))
        )
        line = " && ".join(
            (*self._profile.environment.preamble, f"exec {invocation} {index} 1")
        )
        transport = self._profile.transport

        return (*transport.command, transport.host, line)

    def admitted(self, points: Sequence[SweepPoint], /) -> tuple[SweepPoint, ...]:
        """Return every offered point, since a session runs any of them.

        Parameters
        ----------
        points : Sequence[SweepPoint]
            The points offered, in order.
        """
        return tuple(points)

    def rehearse(self, points: Sequence[SweepPoint], /) -> str:
        """Describe the sessions submitting the points would open, and open none.

        Parameters
        ----------
        points : Sequence[SweepPoint]
            The points offered, of which those already completed are skipped.
        """
        pending = pending_points(self._store, points)

        return (
            f"{len(pending)} sessions would be opened on {self._profile.transport.host}"
        )

    def submit(self, points: Sequence[SweepPoint], /) -> list[RunHandle]:
        """Run every point without a completed run, one session each, in order.

        Parameters
        ----------
        points : Sequence[SweepPoint]
            The points offered, of which those already completed are skipped.

        Returns
        -------
        list[RunHandle]
            One handle per point run, whose backend id is the submission id
            and the point's index in the points file.
        """
        pending = pending_points(self._store, points)

        if not pending:
            return []

        listing = sweep_directory(self._store, pending) / POINTS_NAME
        submission = sweep_id(pending)
        write_points(listing, pending, identity=self._store.identity)
        handles: list[RunHandle] = []

        for index, point in enumerate(pending):
            handles.append(
                self._ledger.record(point, backend_id=f"{submission}:{index}")
            )

            if self._run(self.session(listing, index)):
                self._ledger.fail(point)

        return handles

    def status(self, handle: RunHandle, /) -> RunStatus:
        """Return the status of a point this launcher ran.

        Parameters
        ----------
        handle : RunHandle
            A handle this launcher returned.

        Raises
        ------
        UnknownHandleError
            If the handle names a point this launcher never ran.
        """
        return self._ledger.status(handle)
