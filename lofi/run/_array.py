"""The scheduler-array launcher, which submits one array job over point chunks."""

import json
import math
import shlex
from hashlib import blake2b, sha256
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, final

from lofi._checks import ensure_name
from lofi.run._launch import (
    POINTS_NAME,
    SCRIPT_NAME,
    PointLedger,
    pending_points,
    sweep_directory,
    sweep_id,
    task_count,
    task_points,
    write_points,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi.run._boundary.site import RunHandle, RunStatus, SiteProfile
    from lofi.run._boundary.sweep import SweepPoint
    from lofi.run._launch import CommandRunner, ExperimentIdentity, PointStore
    from lofi.run._values import ExperimentPrice

__all__ = [
    "ArrayJob",
    "ArrayLauncher",
    "ArraySubmission",
    "CancellationError",
    "ResourceRequest",
    "SubmissionError",
    "UnpricedPointError",
    "require_scratch_root",
    "resource_request",
]

_MEBIBYTE = 1 << 20
"""The bytes a scheduler's memory unit holds."""


class SubmissionError(RuntimeError):
    """A submit command that exited with a nonzero status.

    Attributes
    ----------
    command : tuple[str, ...]
        The argument vector that was run.
    exit_status : int
        The status it exited with.
    """

    def __init__(self, command: tuple[str, ...], exit_status: int) -> None:
        self.command = command
        self.exit_status = exit_status
        super().__init__(
            f"the submit command {shlex.join(command)} exited with {exit_status}"
        )


class CancellationError(RuntimeError):
    """A cancel command that exited with a nonzero status.

    Attributes
    ----------
    command : tuple[str, ...]
        The argument vector that was run.
    exit_status : int
        The status it exited with.
    """

    def __init__(self, command: tuple[str, ...], exit_status: int) -> None:
        self.command = command
        self.exit_status = exit_status
        super().__init__(
            f"the cancel command {shlex.join(command)} exited with {exit_status}"
        )


@final
class ArrayJob:
    """The scheduler job every array submission of one experiment shares.

    Every submission of an experiment under one freeze is named by that
    experiment and freeze, so canceling the name withdraws every task the
    experiment's submissions left with the scheduler, whichever launch
    submitted them. The name is ``lofi-`` and sixteen hexadecimal digits of
    a ``blake2b`` digest over the experiment's name and freeze.

    Parameters
    ----------
    profile : SiteProfile
        The resolved site profile whose submission declares how a job is
        named and canceled.
    identity : ExperimentIdentity
        The experiment and the freeze its points run under.
    run : CommandRunner
        Runs the cancel command and returns its exit status.
    """

    __slots__ = ("_name", "_profile", "_run")

    def __init__(
        self,
        profile: SiteProfile,
        identity: ExperimentIdentity,
        *,
        run: CommandRunner,
    ) -> None:
        named = json.dumps([identity.experiment, identity.freeze_id]).encode()
        self._name = (
            profile.submission.job_prefix + blake2b(named, digest_size=8).hexdigest()
        )
        self._profile = profile
        self._run = run

    @property
    def name(self) -> str:
        """Return the name every submission of the experiment is given."""
        return self._name

    def directive(self) -> str:
        """Return the directive text naming a submission, without the prefix."""
        return self._profile.submission.name_directive(name=self._name)

    def cancel(self) -> None:
        """Cancel every job of the experiment's name.

        Raises
        ------
        CancellationError
            If the cancel command exits with a nonzero status.
        """
        command = self._profile.submission.cancellation(name=self._name)
        exit_status = self._run(command)

        if exit_status:
            raise CancellationError(command, exit_status)


def require_scratch_root(profile: SiteProfile, store: PointStore, /) -> None:
    """Refuse a store that does not lie at a site profile's scratch root.

    A task addresses its store by the profile's scratch root, so a launcher
    reads the store there or reads directories no task writes.

    Parameters
    ----------
    profile : SiteProfile
        The resolved site profile.
    store : PointStore
        The store a launcher reads its points' records from.

    Raises
    ------
    ValueError
        If the store's root is not the profile's scratch root.
    """
    scratch = Path(profile.storage.scratch_root)

    if store.root != scratch:
        msg = (
            f"the store lies at {store.root}, and the site's tasks write under "
            f"its scratch root {scratch}"
        )
        raise ValueError(msg)


class UnpricedPointError(ValueError):
    """A point the experiment's price record does not price.

    Attributes
    ----------
    point_ids : tuple[str, ...]
        The points no priced stack trains.
    """

    def __init__(self, point_ids: tuple[str, ...]) -> None:
        self.point_ids = point_ids
        super().__init__(
            f"the price record prices no stack the points {list(point_ids)} train, "
            f"so no resource request can be read off it"
        )


class ResourceRequest(NamedTuple):
    """The wall-clock minutes and memory one array task is granted."""

    minutes: int
    megabytes: int


def resource_request(
    price: ExperimentPrice, points: Sequence[SweepPoint], chunk_size: int, /
) -> ResourceRequest:
    """Return what one task running a chunk of the points is granted.

    A task runs up to a chunk of points one after another, so it is granted
    the ninetieth percentile of the slowest point's bootstrapped training
    seconds times the chunk size, rounded up to whole minutes, and the most
    memory any of the points' stacks is known to need, rounded up to whole
    mebibytes.

    Parameters
    ----------
    price : ExperimentPrice
        The price record of the experiment the points belong to.
    points : Sequence[SweepPoint]
        The points the array runs, at least one.
    chunk_size : int
        The number of points one task runs.

    Raises
    ------
    UnpricedPointError
        If a point trains no stack the price record prices.
    """
    stacks = {point: stack for stack in price.stacks for point in stack.points}
    unpriced = tuple(point.point_id for point in points if point.point_id not in stacks)

    if unpriced:
        raise UnpricedPointError(unpriced)

    seconds = price.point_seconds(0.9)
    slowest = max(seconds[point.point_id] for point in points)
    peak = max(stacks[point.point_id].memory_bytes for point in points)

    return ResourceRequest(
        max(1, math.ceil(slowest * chunk_size / 60)), math.ceil(peak / _MEBIBYTE)
    )


class ArraySubmission(NamedTuple):
    """The artifacts one array submission writes and the command that submits it.

    ``refused`` names the points left out of the submission because their stack
    is predicted to exceed its activation budget.
    """

    directory: Path
    script: Path
    points: Path
    tasks: int
    command: tuple[str, ...]
    request: ResourceRequest
    refused: tuple[str, ...]


class ArrayLauncher:
    """Submits pending points as one scheduler array job.

    A submission writes two files under ``<scope>/sweeps/<id>/``, where the
    scope is the store's: the scratch root, or the directory under it that a
    experiment's freeze addresses. They are a points file listing every
    pending point in order, with the store's experiment and freeze and the
    version of the script, and a script the profile's submit command takes.
    The script consists of the profile's directive prefix before each
    placement directive, each resource directive and the array directive, the
    profile's preamble lines, and one line that runs the profile's interpreter
    on the program with three trailing arguments: the points file, the
    scheduler's task index variable and the chunk size. Task ``k`` of the
    array runs points ``k * chunk_size`` up to but excluding
    ``(k + 1) * chunk_size`` of the points file, so a program recovers them
    with ``task_points``. Rendering writes the files and runs nothing;
    submitting renders and then runs the submit command.

    Every resource a task is granted is read off the experiment's price
    record, and a pending point whose stack the record predicts above its
    activation budget is left out of the points file and never submitted.

    Parameters
    ----------
    profile : SiteProfile
        The resolved site profile every site value is read from.
    store : PointStore
        The store at the profile's scratch root whose records decide which
        points are pending and what status each submitted one is in, and
        whose experiment and freeze, if any, the points file names so a task
        refuses to run its points under another freeze.
    program : tuple[str, ...]
        The argument vector, after the interpreter, that runs one task.
    chunk_size : int
        The number of points one task runs, at least one.
    price : ExperimentPrice
        The price record of the experiment the points belong to.
    run : CommandRunner
        Runs the submit command and returns its exit status.
    throttle : int | None
        The number of tasks the scheduler may run at once, at least one, or
        None to let every task of a submission run at once.

    Raises
    ------
    ValueError
        If the program is empty, a count is not positive, or the store does
        not lie at the profile's scratch root.
    """

    __slots__ = (
        "_chunk_size",
        "_job",
        "_ledger",
        "_price",
        "_profile",
        "_program",
        "_refused",
        "_run",
        "_store",
        "_throttle",
    )

    def __init__(
        self,
        profile: SiteProfile,
        *,
        store: PointStore,
        program: tuple[str, ...],
        chunk_size: int,
        price: ExperimentPrice,
        run: CommandRunner,
        throttle: int | None = None,
    ) -> None:
        if not program:
            msg = "the program is empty"
            raise ValueError(msg)

        if chunk_size < 1:
            msg = f"the chunk size is not positive: {chunk_size!r}"
            raise ValueError(msg)

        if throttle is not None and throttle < 1:
            msg = f"the throttle is not positive: {throttle!r}"
            raise ValueError(msg)

        require_scratch_root(profile, store)
        self._profile = profile
        self._program = tuple(ensure_name("program word", word) for word in program)
        self._chunk_size = chunk_size
        self._price = price
        self._refused = frozenset(
            point
            for stack in price.stacks
            if stack.over_budget
            for point in stack.points
        )
        self._throttle = throttle
        self._run = run
        self._store = store
        self._ledger = PointLedger(store)
        self._job = ArrayJob(profile, store.identity, run=run)

    @property
    def store(self) -> PointStore:
        """Return the store at the profile's scratch root."""
        return self._store

    @property
    def job(self) -> ArrayJob:
        """Return the job every submission of the store's experiment shares."""
        return self._job

    def cancel(self) -> None:
        """Cancel every task the store's experiment left with the scheduler.

        Raises
        ------
        CancellationError
            If the cancel command exits with a nonzero status.
        """
        self._job.cancel()

    def admitted(self, points: Sequence[SweepPoint], /) -> tuple[SweepPoint, ...]:
        """Return the points the price record does not predict above their budget.

        Parameters
        ----------
        points : Sequence[SweepPoint]
            The points offered, in order.
        """
        return tuple(point for point in points if point.point_id not in self._refused)

    def _script(
        self, directory: Path, points: Path, tasks: int, request: ResourceRequest
    ) -> str:
        """Return the submission script text for a points file."""
        submission = self._profile.submission
        directives = (
            *submission.placement_directives(self._profile.placement),
            *submission.resource_directives(
                self._profile.placement,
                minutes=request.minutes,
                megabytes=request.megabytes,
                directory=str(directory),
            ),
            self._job.directive(),
            submission.array_directive(
                first=0,
                last=tasks - 1,
                throttle=tasks if self._throttle is None else self._throttle,
            ),
        )
        invocation = shlex.join(
            (*self._profile.interpreter.command, *self._program, str(points))
        )
        lines = (
            "#!/bin/sh",
            *(f"{submission.directive_prefix} {text}" for text in directives),
            "set -eu",
            *self._profile.environment.preamble,
            f'exec {invocation} "${submission.task_index_variable}" {self._chunk_size}',
        )

        return "\n".join(lines) + "\n"

    def _write(
        self, pending: Sequence[SweepPoint], refused: tuple[str, ...]
    ) -> ArraySubmission:
        """Write the submission artifacts for pending points."""
        directory = sweep_directory(self._store, pending)
        listing = directory / POINTS_NAME
        script = directory / SCRIPT_NAME
        tasks = task_count(len(pending), self._chunk_size)
        request = resource_request(self._price, pending, self._chunk_size)
        text = self._script(directory, listing, tasks, request)
        write_points(
            listing,
            pending,
            identity=self._store.identity,
            script=sha256(text.encode()).hexdigest()[:16],
        )
        script.write_text(text, encoding="utf-8", newline="\n")

        return ArraySubmission(
            directory,
            script,
            listing,
            tasks,
            (self._profile.submission.command, str(script)),
            request,
            refused,
        )

    def _pending(
        self, points: Sequence[SweepPoint], /
    ) -> tuple[tuple[SweepPoint, ...], tuple[str, ...]]:
        """Return the pending points admitted, and those refused, in order."""
        pending = pending_points(self._store, points)
        admitted = self.admitted(pending)
        kept = {point.point_id for point in admitted}

        return admitted, tuple(
            point.point_id for point in pending if point.point_id not in kept
        )

    def render(self, points: Sequence[SweepPoint], /) -> ArraySubmission | None:
        """Write the submission artifacts for the pending points and run nothing.

        Parameters
        ----------
        points : Sequence[SweepPoint]
            The points offered, of which those already completed are skipped
            and those predicted above their activation budget are left out.

        Returns
        -------
        ArraySubmission | None
            The written artifacts, or None when no point is left to submit.

        Raises
        ------
        UnpricedPointError
            If a pending point trains no stack the price record prices.
        """
        pending, refused = self._pending(points)

        if not pending:
            return None

        return self._write(pending, refused)

    def rehearse(self, points: Sequence[SweepPoint], /) -> str:
        """Write the submission artifacts and describe them, submitting nothing.

        Parameters
        ----------
        points : Sequence[SweepPoint]
            The points offered, of which those already completed are skipped
            and those predicted above their activation budget are left out.

        Raises
        ------
        UnpricedPointError
            If a pending point trains no stack the price record prices.
        """
        rendered = self.render(points)

        if rendered is None:
            return "every point is settled or refused"

        return (
            f"{rendered.tasks} tasks over {rendered.points} would be submitted "
            f"with {rendered.script}"
        )

    def submit(self, points: Sequence[SweepPoint], /) -> list[RunHandle]:
        """Submit the pending points as one array job.

        Parameters
        ----------
        points : Sequence[SweepPoint]
            The points offered, of which those already completed are skipped
            and those predicted above their activation budget are left out.

        Returns
        -------
        list[RunHandle]
            One handle per submitted point, in points-file order, whose
            backend id is the submission id and the task index that runs it.

        Raises
        ------
        SubmissionError
            If the submit command exits with a nonzero status.
        UnpricedPointError
            If a pending point trains no stack the price record prices.
        """
        pending, refused = self._pending(points)

        if not pending:
            return []

        rendered = self._write(pending, refused)
        exit_status = self._run(rendered.command)

        if exit_status:
            raise SubmissionError(rendered.command, exit_status)

        submission = sweep_id(pending)

        return [
            self._ledger.record(point, backend_id=f"{submission}:{task}")
            for task in range(rendered.tasks)
            for point in task_points(pending, task, self._chunk_size)
        ]

    def status(self, handle: RunHandle, /) -> RunStatus:
        """Return the status the store records for a submitted point.

        Parameters
        ----------
        handle : RunHandle
            A handle this launcher returned.

        Raises
        ------
        UnknownHandleError
            If the handle names a point this launcher never submitted.
        """
        return self._ledger.status(handle)
