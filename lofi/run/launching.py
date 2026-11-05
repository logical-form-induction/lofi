"""Launching an experiment's points from its freeze and polling its store.

An experiment declares a sweep over its configuration document, the root seed
its points are seeded from, the freeze it runs against and the launcher its
points are driven through. A launch runs only when the code that executes is
the tree its freeze tags: the checkout the run package is imported from must
have the freeze's tagged commit as its ``HEAD`` and no modified tracked file,
and the experiment document must lie inside that checkout. Anything else is
refused before anything is expanded, executed or written, and every manifest
records the tagged commit as its source revision. The freeze's ``FREEZE.json``
must list the experiment, record the commit the checkout is at, and record the
sweep identifier the expansion reproduces; anything else is refused before any
point is claimed. A point completed under this freeze and seed derivation, or
whose failure is final, is served from the store and executed again by nobody.
A launch splits the expanded points into the ones whose outcome the store
already holds and the ones it does not, binds the rest to the experiment as it
submits them, drives them through the declared launcher, writes the
experiment's results directory under ``results/<n>/<experiment>/`` from the
store, and writes a record that counts the hits.

The experiment document is read with the ``key=value`` overrides a launch is
given, the one override syntax every command takes, so ``launch.kind=array``
and ``launch.chunk_size=8`` select and shape the launcher. The overrides are
part of the task program a scheduler task runs, so a task reads the document a
launch read.

A launcher is an installed plugin of the ``lofi.launcher`` entry-point group.
It builds its backend from one launch context, which offers the experiment's
store, the in-process executor its configuration kind builds, the stop rules,
the resolved site profile, the task program, the price record and the command
runner; each resource is built only when the launcher asks for it. A launcher
whose submitted work outlives the launching process declares how that work is
canceled once a stop rule fires.

A launch halts on the experiment's declared stop rules. It reads them from the
store before anything is claimed, and while one fires it submits nothing. An
in-process launch reads them again after every point it executes and executes
no point after the one a rule first fires on. Each point it never reached
stays unsubmitted and is listed as halted in the record.

A pilot launch runs an experiment before any freeze exists. It reads no freeze
record and requires no tag: every point runs in this process on the serial
launcher, whatever launcher the experiment declares, and each manifest records
the commit the checkout is at. Its store lies under ``pilot/`` in the checkout
unless another root is named, its rows are reported as ``UNFROZEN``, and its
results directory is ``pilot/results/<experiment>/``. A pilot submits nothing
to a site, so it makes no dry run and admits no companion experiment.

A poll reads an experiment's store and rewrites its results directory, up to
a bound number of times, and executes nothing. It stops early once no point is
submitted or running, once a point's failure is final, or once a declared stop
rule fires, in which case the declared launcher cancels whatever work it left
with its site.
"""

from __future__ import annotations

import enum
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final, NamedTuple, Protocol, cast, final

from lofi.eval import UNFROZEN
from lofi.run._admission import admit
from lofi.run._boundary.launch import LaunchReport, LaunchRequest, PointOutcome
from lofi.run._boundary.site import resolve_site_profile
from lofi.run._environment import (
    ExecutionEnvironment,
    UnfrozenRevisionError,
    package_checkout,
    require_frozen_revision,
    source_revision,
)
from lofi.run._experiment import ExecutionContext, expand_experiment, read_experiment
from lofi.run._launch import (
    SWEEPS_NAME,
    pending_points,
    subprocess_output,
    subprocess_runner,
    sweep_id,
)
from lofi.run._launcher_plugin import launcher_plugin
from lofi.run._manifest import accelerator_runtime, installed_packages, read_price
from lofi.run._reclaim import Artifact, Usage, reclaim, report_usage
from lofi.run._serial import SerialLauncher
from lofi.run._store import ExperimentStore, FailureKind, PointStatus
from lofi.run.results import (
    PILOT_NAME,
    FreezeMismatchError,
    frozen_experiment,
    pilot_directory,
    results_directory,
    write_results,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from typing import TextIO

    from lofi._types import JsonValue
    from lofi.run._boundary.site import Launcher, SiteProfile
    from lofi.run._boundary.sweep import SweepPoint
    from lofi.run._launch import CommandRunner, PointStore, RunDirectories
    from lofi.run._ports import TrainingOperations
    from lofi.run._reclaim import Reclamation
    from lofi.run._serial import ExecutePoint
    from lofi.run._store import RunOutcome
    from lofi.run._values import ExperimentPrice

__all__ = [
    "ExperimentLaunching",
    "LaunchPlan",
    "LaunchResources",
    "PollExit",
    "StopReading",
    "StopRules",
    "UnlaunchableExperimentError",
    "UnlaunchedPointError",
    "UnpricedLaunchError",
    "launch",
    "plan_launch",
    "task_program",
]


class PollExit(enum.IntEnum):
    """The exit status a launch or a poll loop ends with."""

    CLEAN = 0
    FAILED = 1
    UNFINISHED = 2
    STOPPED = 3


_FAILED: Final = frozenset(kind.value for kind in FailureKind) - {
    FailureKind.INTERRUPTED.value
}
"""The statuses of a point whose failure is final."""

_PENDING: Final = frozenset(
    {
        PointStatus.SUBMITTED.value,
        PointStatus.RUNNING.value,
        PointStatus.AWAITING_RESPONSES.value,
    }
)
"""The statuses of a point its launcher has yet to finish."""


class UnlaunchedPointError(RuntimeError):
    """A pending point the backend returned no handle for.

    Attributes
    ----------
    point_id : str
        The identifier of the point that was submitted without a handle.
    """

    def __init__(self, point_id: str) -> None:
        self.point_id = point_id
        super().__init__(
            f"the backend submitted no handle for the pending point {point_id!r}"
        )


class UnlaunchableExperimentError(RuntimeError):
    """An experiment whose points the chosen launcher cannot execute.

    Attributes
    ----------
    experiment : str
        The experiment refused, by name.
    kind : str
        The kind of configuration document the experiment sweeps.
    launcher : str
        The launcher the experiment was submitted through.
    """

    def __init__(self, experiment: str, kind: str, launcher: str) -> None:
        self.experiment = experiment
        self.kind = kind
        self.launcher = launcher
        super().__init__(
            f"the experiment {experiment!r} sweeps a {kind} document, whose points "
            f"run in the launching process, so it cannot launch on the "
            f"{launcher} launcher"
        )


class UnpricedLaunchError(RuntimeError):
    """A launch that reads a price record its experiment's sweep does not have.

    Attributes
    ----------
    record : Path
        The price record the launch looked for.
    """

    def __init__(self, record: Path) -> None:
        self.record = record
        super().__init__(
            f"the launcher reads its resource requests off {record}, which is "
            f"absent; price the experiment with lofi price"
        )


class StopReading(Protocol):
    """The stop rules of an experiment found firing when they were last read."""

    @property
    def fired(self) -> tuple[str, ...]:
        """Return the names of the rules that fired, in declared order."""
        ...


class StopRules:
    """Reads an experiment's stop rules from its store, rewriting its results.

    Each reading writes the experiment's results directory from the store, as
    a status poll does, and keeps the names of the stop rules that fire on it.

    Parameters
    ----------
    directory : Path
        The experiment's results directory.
    points : Sequence[SweepPoint]
        The points of the experiment's expanded sweep, in expansion order.
    store : ExperimentStore
        The store the experiment's points execute under.
    """

    __slots__ = ("_directory", "_fired", "_points", "_store")

    def __init__(
        self, directory: Path, points: Sequence[SweepPoint], store: ExperimentStore
    ) -> None:
        self._directory = directory
        self._points = tuple(points)
        self._store = store
        self._fired: tuple[str, ...] = ()

    @property
    def fired(self) -> tuple[str, ...]:
        """Return the rules that fired when they were last read, in declared order."""
        return self._fired

    def __call__(self) -> bool:
        """Read the stop rules afresh and return whether any of them fires."""
        record = write_results(
            self._directory,
            points=self._points,
            store=self._store,
            today=datetime.now(tz=UTC).date(),
        )
        self._fired = tuple(cast("list[str]", record["stop_rules"]))

        return bool(self._fired)


class _Claiming:
    """Binds each point to its experiment as an in-process backend reaches it.

    A point is claimed immediately before it executes, so a launch a stop rule
    halts leaves every point it never reached unsubmitted.
    """

    __slots__ = ("_execute", "_store")

    def __init__(self, store: ExperimentStore, execute: ExecutePoint) -> None:
        self._store = store
        self._execute = execute

    def __call__(self, point: SweepPoint, /) -> RunOutcome:
        """Claim one point and execute it."""
        self._store.claim(point)

        return self._execute(point)


class LaunchPlan(NamedTuple):
    """The points whose outcome the store settles and the points a launch runs."""

    cached: tuple[SweepPoint, ...]
    pending: tuple[SweepPoint, ...]


def plan_launch(store: RunDirectories, points: Sequence[SweepPoint], /) -> LaunchPlan:
    """Split offered points by whether the store already settles their outcome.

    Parameters
    ----------
    store : RunDirectories
        The store whose records decide each outcome; an experiment's store
        refuses a point seeded under another experiment root.
    points : Sequence[SweepPoint]
        The points the sweep expanded to, in expansion order.

    Returns
    -------
    LaunchPlan
        The settled points and the pending ones, each in expansion order.

    Raises
    ------
    ReseededRunError
        If an experiment's point was seeded under another derivation.
    """
    pending = pending_points(store, points)
    outstanding = {point.point_id for point in pending}

    return LaunchPlan(
        tuple(point for point in points if point.point_id not in outstanding),
        pending,
    )


def launch(
    plan: LaunchPlan,
    request: LaunchRequest,
    /,
    *,
    store: PointStore,
    launcher: Launcher,
    stop: StopReading,
) -> LaunchReport:
    """Drive a plan's pending points through a backend and report the cache.

    The settled points are reported with the status the store records,
    without being offered to the backend. The pending ones are submitted in
    one call, unless a stop rule already fired, and each one's status is read
    back through the handle the backend returned for it, from the same store.
    A pending point the backend returned no handle for is halted once a stop
    rule has fired by the time the backend returns.

    Parameters
    ----------
    plan : LaunchPlan
        The split of the sweep's points the store decided.
    request : LaunchRequest
        The declaration the launch is named by, whose points are the plan's
        pending ones.
    store : PointStore
        The store each point's directory lies under.
    launcher : Launcher
        The backend the pending points are submitted to.
    stop : StopReading
        The experiment's stop rules as last read, which an in-process backend
        reads again after each point it executes.

    Returns
    -------
    LaunchReport
        One outcome per settled or launched point, the halted points and the
        fired stop rules, with the counts.

    Raises
    ------
    UnlaunchedPointError
        If the backend returns no handle for a pending point while no stop
        rule fired.
    """
    handles = (
        {}
        if stop.fired
        else {handle.point_id: handle for handle in launcher.submit(plan.pending)}
    )
    halted: list[str] = []
    outcomes = [
        PointOutcome(
            point_id=point.point_id,
            seed=point.seed,
            status=store.status(point),
            directory=str(store.directory(point)),
            cached=True,
        )
        for point in plan.cached
    ]

    for point in plan.pending:
        handle = handles.get(point.point_id)

        if handle is None:
            if not stop.fired:
                raise UnlaunchedPointError(point.point_id)

            halted.append(point.point_id)
            continue

        outcomes.append(
            PointOutcome(
                point_id=point.point_id,
                seed=point.seed,
                status=launcher.status(handle),
                directory=str(store.directory(point)),
                cached=False,
            )
        )

    return LaunchReport(
        launch_id=request.digest(),
        launcher=str(request.launch.kind),
        offered=len(plan.cached) + len(plan.pending),
        cache_hits=len(plan.cached),
        launched=len(plan.pending) - len(halted),
        outcomes=tuple(outcomes),
        halted=tuple(halted),
        stop_rules=stop.fired,
    )


def task_program(
    scratch_root: str, experiment: Path, overrides: Sequence[str] = (), /
) -> tuple[str, ...]:
    """Return the argument vector a scheduler task runs its points with.

    Parameters
    ----------
    scratch_root : str
        The site's scratch root, under which the task's store lies.
    experiment : Path
        The experiment document relative to the checkout, from which the task
        reads the freeze its points are reported against, the root seed they
        are seeded under and the failure policy they run under.
    overrides : Sequence[str]
        The ``key=value`` overrides the experiment was launched under, which
        the task reads the document with.

    Returns
    -------
    tuple[str, ...]
        The vector the site interpreter is given, after the interpreter
        itself.
    """
    return (
        "-m",
        "lofi.run",
        "task",
        "--root",
        scratch_root,
        "--experiment",
        experiment.as_posix(),
        *(word for override in overrides for word in ("--set", override)),
    )


@final
class LaunchResources:
    """The runtime resources a launcher plugin builds its backend from.

    Every resource is built when a launcher first asks for it, so a launcher
    that runs its points in this process never resolves a site profile and a
    launcher that submits to a site never builds an in-process executor.

    Parameters
    ----------
    store : ExperimentStore
        The store the experiment's points execute under.
    points : Sequence[SweepPoint]
        The experiment's expanded points, whose sweep names the price record.
    stop : StopRules
        The experiment's stop rules, read after each point executed here.
    run : CommandRunner
        Runs a site's submit, session and cancel commands.
    site : Callable[[], SiteProfile]
        Resolves the site profile.
    program : Callable[[SiteProfile], tuple[str, ...]]
        Returns the task program for the resolved site.
    execute : Callable[[], ExecutePoint]
        Builds the in-process executor the configuration kind declares.
    source : Path
        The checkout the price record is read under.
    """

    __slots__ = (
        "_execute",
        "_points",
        "_program",
        "_run",
        "_site",
        "_source",
        "_stop",
        "_store",
    )

    def __init__(
        self,
        *,
        store: ExperimentStore,
        points: Sequence[SweepPoint],
        stop: StopRules,
        run: CommandRunner,
        site: Callable[[], SiteProfile],
        program: Callable[[SiteProfile], tuple[str, ...]],
        execute: Callable[[], ExecutePoint],
        source: Path,
    ) -> None:
        self._store = store
        self._points = tuple(points)
        self._stop = stop
        self._run = run
        self._site = site
        self._program = program
        self._execute = execute
        self._source = source

    @property
    def store(self) -> ExperimentStore:
        """Return the store the experiment's points execute under."""
        return self._store

    @property
    def stop(self) -> StopRules:
        """Return the experiment's stop rules."""
        return self._stop

    @property
    def run(self) -> CommandRunner:
        """Return the runner of a site's commands."""
        return self._run

    def site(self) -> SiteProfile:
        """Return the resolved site profile."""
        return self._site()

    def program(self) -> tuple[str, ...]:
        """Return the argument vector, after the interpreter, one task runs."""
        return self._program(self.site())

    def execute(self) -> ExecutePoint:
        """Return the executor that claims and executes each point in this process.

        Raises
        ------
        PointExecutionError
            If a point cannot be executed; nothing is claimed.
        """
        return _Claiming(self._store, self._execute())

    def price(self) -> ExperimentPrice:
        """Return the tracked price record of the experiment's sweep.

        Raises
        ------
        UnpricedLaunchError
            If the sweep has no price record.
        """
        record = self._source / SWEEPS_NAME / sweep_id(self._points)
        price = read_price(record)

        if price is None:
            raise UnpricedLaunchError(record)

        return price


def _reclaim(
    profile: SiteProfile,
    store: ExperimentStore,
    points: Sequence[SweepPoint],
    report: Callable[[Sequence[str]], str],
) -> Reclamation:
    """Bring the scratch root under its quota before anything is submitted."""
    quota = profile.storage.quota
    usage = Usage(quota.command, quota.used_pattern, quota.unit_bytes)

    return reclaim(
        store,
        points,
        tuple(
            Artifact(artifact.name, artifact.pattern, artifact.regeneration_hours)
            for artifact in profile.storage.artifacts
        ),
        used=report_usage(usage, report),
        limit=quota.limit_bytes,
    )


def _count(record: Mapping[str, JsonValue], statuses: frozenset[str], /) -> int:
    """Return how many points a status record holds in the given statuses."""
    counts = cast("dict[str, int]", record["counts"])

    return sum(count for status, count in counts.items() if status in statuses)


def _unbuilt() -> ExecutePoint:
    """Refuse an in-process executor, which a status poll never builds."""
    msg = "a status poll executes no point"
    raise ValueError(msg)


@final
class ExperimentLaunching:
    """Launches, pilots and polls experiments against the process's site.

    Parameters
    ----------
    run : CommandRunner
        Runs a site's submit, session and cancel commands.
    quota_report : Callable[[Sequence[str]], str]
        Runs the site's quota command and returns what it prints.
    sleep : Callable[[float], None]
        Waits the interval between polls.
    checkout : Path | None
        The checkout the executed code, the experiment's configuration groups
        and its price records are read from and its results are written under;
        the checkout the run package is imported from when None.
    """

    __slots__ = ("_checkout", "_quota_report", "_run", "_sleep")

    def __init__(
        self,
        *,
        run: CommandRunner = subprocess_runner,
        quota_report: Callable[[Sequence[str]], str] = subprocess_output,
        sleep: Callable[[float], None] = time.sleep,
        checkout: Path | None = None,
    ) -> None:
        self._run = run
        self._quota_report = quota_report
        self._sleep = sleep
        self._checkout = checkout

    def _source(self) -> Path:
        """Return the checkout the experiment is read and written under."""
        return (
            package_checkout() if self._checkout is None else self._checkout
        ).resolve()

    def launch(
        self,
        document: Path,
        *,
        freeze: Path,
        root: Path | None,
        overrides: tuple[str, ...],
        dry_run: bool,
        companions: Sequence[Path],
        training: TrainingOperations,
        environ: Mapping[str, str],
        out: Path,
        stdout: TextIO,
    ) -> int:
        """Launch one experiment, write its results and record, and validate the record.

        Parameters
        ----------
        document : Path
            The experiment document.
        freeze : Path
            The ``FREEZE.json`` of the freeze the experiment is launched from.
        root : Path | None
            The store root an in-process launch runs under; the working
            directory when None. A launcher that submits to a site runs under
            the site's scratch root.
        overrides : tuple[str, ...]
            The ``key=value`` overrides the experiment document is read with.
        dry_run : bool
            Whether to write the launcher's submission and submit nothing.
        companions : Sequence[Path]
            The experiments of the same freeze admitted together with this one.
        training : TrainingOperations
            What prepares, fits and scores one planned run.
        environ : Mapping[str, str]
            The environment the site profile is resolved from.
        out : Path
            Where the launch record is written.
        stdout : TextIO
            Where the record is echoed.

        Returns
        -------
        int
            Zero, or three when a stop rule fired, once the written record has
            been read back and validated.

        Raises
        ------
        UnlaunchableExperimentError
            If an experiment whose configuration kind executes only in the
            launching process is launched on a launcher that submits to a
            site; nothing is expanded, submitted or written.
        UnfrozenRevisionError
            If the executing checkout is not the tree the experiment's freeze
            tags, or the experiment document lies outside it.
        FreezeRecordError
            If the freeze record is unreadable or does not list the experiment.
        FreezeMismatchError
            If the freeze record names another freeze or commit, or the
            experiment expands to another sweep than the one it froze.
        PointExecutionError
            If a point the launching process executes cannot be executed.
        AdmissionError
            If a launch to a site and its companions are unpriced, lack a
            headline or analysis partner, or need more P90 hours than the
            site's allocation lets one paper hold.
        QuotaExceededError
            If reclaiming the settled points' artifacts cannot bring the
            scratch root under its quota. A dry run reclaims nothing.
        """
        located = document.resolve()
        experiment = read_experiment(located, overrides=overrides)
        declaration = experiment.launch
        selected = launcher_plugin(declaration)

        if experiment.config.in_process and selected.requires_site:
            raise UnlaunchableExperimentError(
                experiment.name, experiment.config.kind, str(declaration.kind)
            )

        source = self._source()
        commit = require_frozen_revision(source, experiment.freeze_id)

        if not located.is_relative_to(source):
            msg = (
                f"the experiment {located} lies outside the executing checkout "
                f"{source}, so it is not read from the tree {experiment.freeze_id} "
                f"tags"
            )
            raise UnfrozenRevisionError(msg)

        environment = ExecutionEnvironment(
            installed_packages(), commit, accelerator_runtime()
        )
        expanded = expand_experiment(experiment, source)
        frozen = frozen_experiment(freeze, experiment, expanded)

        if frozen.commit != commit:
            msg = (
                f"the freeze record {freeze} records the commit {frozen.commit}, "
                f"and {experiment.freeze_id} tags {commit}"
            )
            raise FreezeMismatchError(msg)

        results = results_directory(source, experiment)
        profile = resolve_site_profile(environ) if selected.requires_site else None
        store = ExperimentStore(
            (Path.cwd() if root is None else root)
            if profile is None
            else Path(profile.storage.scratch_root),
            experiment,
        )
        stop = StopRules(results, expanded, store)
        relative = located.relative_to(source)

        if profile is not None:
            admit(
                experiment,
                tuple(
                    read_experiment(path.resolve(), overrides=overrides)
                    for path in companions
                ),
                admitted_hours=profile.allocation.admitted_hours,
            )

            if not dry_run:
                _reclaim(profile, store, expanded, self._quota_report)

        def site() -> SiteProfile:
            return resolve_site_profile(environ) if profile is None else profile

        def program(resolved: SiteProfile) -> tuple[str, ...]:
            return task_program(resolved.storage.scratch_root, relative, overrides)

        def execute() -> ExecutePoint:
            return experiment.config.executor(
                ExecutionContext(store, environment, experiment.failure, training),
                expanded,
            )

        resources = LaunchResources(
            store=store,
            points=expanded,
            stop=stop,
            run=self._run,
            site=site,
            program=program,
            execute=execute,
            source=source,
        )
        launcher = selected.construct(declaration, resources)

        if dry_run:
            stdout.write(f"{launcher.rehearse(expanded)}\n")

            return PollExit.CLEAN

        admitted = launcher.admitted(expanded)
        kept = {point.point_id for point in admitted}
        refused = tuple(
            point.point_id for point in expanded if point.point_id not in kept
        )
        plan = plan_launch(store, admitted)

        if not stop() and selected.requires_site:
            for point in plan.pending:
                store.claim(point)

        request = LaunchRequest(
            experiment=experiment.name,
            freeze_id=experiment.freeze_id,
            experiment_seed=experiment.seed_policy.root,
            launch=declaration,
            root=str(store.root),
            points=tuple(point.point_id for point in plan.pending),
            program=resources.program() if profile is not None else (),
            profile=profile,
        )
        report = launch(plan, request, store=store, launcher=launcher, stop=stop)

        return _record(
            report.with_(refused=refused), results, expanded, store, out, stdout
        )

    def pilot(
        self,
        document: Path,
        *,
        root: Path | None,
        overrides: tuple[str, ...],
        training: TrainingOperations,
        out: Path,
        stdout: TextIO,
    ) -> int:
        """Run one experiment as a pilot, outside any freeze, and write its results.

        Parameters
        ----------
        document : Path
            The experiment document.
        root : Path | None
            The store root; ``pilot/`` in the checkout when None.
        overrides : tuple[str, ...]
            The ``key=value`` overrides the experiment document is read with.
        training : TrainingOperations
            What prepares, fits and scores one planned run.
        out : Path
            Where the launch record is written.
        stdout : TextIO
            Where the record is echoed.

        Returns
        -------
        int
            Zero, or three when a stop rule fired.

        Raises
        ------
        PointExecutionError
            If a point cannot be executed; nothing is claimed.
        """
        experiment = read_experiment(document.resolve(), overrides=overrides)
        source = self._source()
        environment = ExecutionEnvironment(
            installed_packages(), source_revision(source), accelerator_runtime()
        )
        points = expand_experiment(experiment, source)
        store = ExperimentStore(
            source / PILOT_NAME if root is None else root, experiment, pilot=True
        )
        results = pilot_directory(source, experiment)
        stop = StopRules(results, points, store)
        execute = experiment.config.executor(
            ExecutionContext(store, environment, experiment.failure, training),
            points,
        )
        launcher = SerialLauncher(store, execute=_Claiming(store, execute), stop=stop)
        plan = plan_launch(store, points)
        stop()
        request = LaunchRequest(
            experiment=experiment.name,
            freeze_id=UNFROZEN,
            experiment_seed=experiment.seed_policy.root,
            launch=experiment.launch,
            root=str(store.root),
            points=tuple(point.point_id for point in plan.pending),
        )
        report = launch(plan, request, store=store, launcher=launcher, stop=stop)

        return _record(report, results, points, store, out, stdout)

    def poll(
        self,
        document: Path,
        *,
        root: Path | None,
        overrides: tuple[str, ...],
        interval: float,
        polls: int,
        environ: Mapping[str, str],
        stdout: TextIO,
    ) -> int:
        """Poll one experiment's store, rewriting its results directory each time.

        Each poll's status record is echoed on its own line.

        Parameters
        ----------
        document : Path
            The experiment document.
        root : Path | None
            The store root the experiment's points write under; the working
            directory when None.
        overrides : tuple[str, ...]
            The ``key=value`` overrides the experiment was launched under.
        interval : float
            The seconds waited between polls.
        polls : int
            The most polls made before the loop ends.
        environ : Mapping[str, str]
            The environment a site profile is resolved from when the declared
            launcher cancels its work.
        stdout : TextIO
            Where each poll's status record is echoed.

        Returns
        -------
        int
            Zero when no point is left submitted or running, none failed and
            no stop rule fired, one when a point's failure is final, two when
            the last poll still found points submitted or running, and three
            when a stop rule fired and the declared launcher's outstanding
            work was canceled.

        Raises
        ------
        CancellationError
            If a stop rule fired and the site's cancel command failed.
        """
        experiment = read_experiment(document.resolve(), overrides=overrides)
        source = self._source()
        directory = results_directory(source, experiment)
        points = expand_experiment(experiment, source)
        store = ExperimentStore(Path.cwd() if root is None else root, experiment)
        stop = StopRules(directory, points, store)
        relative = document.resolve().relative_to(source)
        resources = LaunchResources(
            store=store,
            points=points,
            stop=stop,
            run=self._run,
            site=lambda: resolve_site_profile(environ),
            program=lambda resolved: task_program(
                resolved.storage.scratch_root, relative, overrides
            ),
            execute=_unbuilt,
            source=source,
        )

        for poll in range(polls):
            if poll:
                self._sleep(interval)

            record = write_results(
                directory,
                points=points,
                store=store,
                today=datetime.now(tz=UTC).date(),
            )
            stdout.write(f"{json.dumps(record)}\n")

            if record["stop_rules"]:
                declaration = experiment.launch
                launcher_plugin(declaration).cancel(declaration, resources)

            if _count(record, _FAILED):
                return PollExit.FAILED

            if record["stop_rules"]:
                return PollExit.STOPPED

            if not _count(record, _PENDING):
                return PollExit.CLEAN

        return PollExit.UNFINISHED


def _record(
    report: LaunchReport,
    results: Path,
    points: Sequence[SweepPoint],
    store: ExperimentStore,
    out: Path,
    stdout: TextIO,
) -> int:
    """Write the results and the launch record, then echo the record read back."""
    write_results(
        results,
        points=points,
        store=store,
        today=datetime.now(tz=UTC).date(),
    )
    report.write(out)
    written = LaunchReport.read(out)
    stdout.write(f"{written.model_dump_json()}\n")

    return PollExit.STOPPED if written.stop_rules else PollExit.CLEAN
