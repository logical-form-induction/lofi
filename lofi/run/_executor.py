"""Production execution of one point through its run and checkpoint stores.

A point ends in one of four outcomes. It completes, and its directory holds
the report rows it emitted. It fails numerically, as a non-finite loss or
metric does, and the failure is recorded with its diagnostic and never run
again. Its configuration cannot be prepared for training, which the training
operations report as a ``PreparationError``, and that failure too is recorded
with its diagnostic and never run again. Or it fails for want of
infrastructure and is attempted again, resuming from its
latest checkpoint, until the site's declared attempts are spent across every
launch, when that failure is recorded instead and the point is not
attempted again. Each attempt is recorded in the point's failure record before
it starts, so an attempt whose process is killed outright still counts.

An infrastructure failure is one the machine signals explicitly: an exhausted
allocation (``MemoryError``, which the training operations raise for an
exhausted accelerator), an interrupted call (``InterruptedError``), a lost
connection (``ConnectionError``, which covers a broken pipe), a stall
(``TimeoutError``), or an ``OSError`` whose errno is in
``INFRASTRUCTURE_ERRNOS``. Any other error, such as a missing corpus or a bad
path, is a defect of the configuration or the code: it propagates and the
attempt it ended is not counted. A process asked to terminate by ``SIGTERM``,
as a scheduler preempting it is, records the interrupted attempt and stops, so
the point is resumed on a later launch rather than left running. Every failure
records the point's seed derivation, and a point is never re-run at another
seed or dropped.
"""

import errno
import signal
from contextlib import contextmanager
from typing import TYPE_CHECKING, Final, final

from lofi._checks import NumericalFailureError
from lofi.eval import UNFROZEN
from lofi.run._boundary.manifest import (
    FailureKind,
    ManifestStatus,
    PointFailure,
    RecordedRows,
    open_training_manifest,
)
from lofi.run._boundary.metrics import MetricsRecorder
from lofi.run._boundary.sweep import SweepPoint, resolve_point
from lofi.run._environment import ExecutionEnvironment, execution_environment
from lofi.run._manifest import (
    read_failure,
    remove_failure,
    training_record,
    write_failure,
    write_rows,
)
from lofi.run._ports import HeldOutScoring
from lofi.run._sinks import (
    PointReportSink,
    SinkContext,
    log_held_out_rows,
    write_artifacts,
)
from lofi.run._store import ReseededRunError, RunOutcome
from lofi.run._training import TrainingBlueprint, plan_training
from lofi.run._wiring import resolve_metric_sinks

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping
    from pathlib import Path
    from types import FrameType

    from lofi.nn.policy import DevicePolicy
    from lofi.run._boundary.experiment import FailurePolicySpec
    from lofi.run._ports import MetricSink, TrainingOperations
    from lofi.run._seed import PointSeed
    from lofi.run._store import RunStore
    from lofi.spec import RunSpecCodec, SinkSpec
    from lofi.train import RestorableRandomStreams

__all__ = [
    "CHECKPOINTS_NAME",
    "INFRASTRUCTURE_ERRNOS",
    "PointExecutionError",
    "PointExecutor",
    "PointTerminatedError",
    "PreparationError",
    "execute_to_outcome",
    "is_infrastructure",
]

CHECKPOINTS_NAME = "checkpoints"
"""Directory inside a run that holds completed-epoch checkpoints."""

INFRASTRUCTURE_ERRNOS: Final = frozenset(
    {
        errno.EIO,
        errno.ESTALE,
        errno.ETIMEDOUT,
        errno.ENETDOWN,
        errno.ENETUNREACH,
        errno.ECONNRESET,
    }
)
"""The errnos of an ``OSError`` that report a fault of the machine, not the point.

``EIO`` is a storage device failing a read or write, ``ESTALE`` a shared
filesystem invalidating an open handle, ``ETIMEDOUT`` a stalled filesystem or
connection, ``ENETDOWN`` and ``ENETUNREACH`` a lost network, and ``ECONNRESET``
a peer dropping a connection. None of them depends on the point's
configuration, so the same attempt may succeed on another try.
"""

_INFRASTRUCTURE_FAILURES: Final = (
    InterruptedError,
    ConnectionError,
    TimeoutError,
)
"""The exception types that report a fault of the machine whatever their errno."""

_UNFINISHED: Final = "the attempt started and reached no outcome"
"""The diagnostic of an attempt recorded before it runs."""


class PointExecutionError(RuntimeError):
    """A point lacks information required for production execution."""


class PreparationError(Exception):
    """A point's configuration cannot be prepared for training.

    The diagnostic is the refusal preparation raised, which depends on the
    configuration alone, so attempting the point again would fail the same way.
    """


@final
class _PointMetricSinks:
    """Build and retain one point's declared metric sinks on first use."""

    __slots__ = ("_context", "_sinks", "_specs")

    def __init__(self, specs: Mapping[str, SinkSpec], context: SinkContext) -> None:
        self._specs = specs
        self._context = context
        self._sinks: tuple[MetricSink, ...] | None = None

    def resolve(self) -> tuple[MetricSink, ...]:
        """Return the point's sinks, building them only for an executed point."""
        if self._sinks is None:
            self._sinks = resolve_metric_sinks(self._specs, self._context)

        return self._sinks

    def close(self) -> None:
        """Close every sink that was built for the point."""
        if self._sinks is not None:
            for sink in self._sinks:
                sink.close()


class PointTerminatedError(BaseException):
    """The process executing a point was asked to terminate.

    It derives from ``BaseException`` so that no handler for ordinary errors
    inside training absorbs it.

    Attributes
    ----------
    signal_number : int
        The signal the process received.
    """

    def __init__(self, signal_number: int) -> None:
        self.signal_number = signal_number
        super().__init__(
            f"the process received {signal.Signals(signal_number).name} while "
            f"executing a point"
        )


def is_infrastructure(error: BaseException, /) -> bool:
    """Return whether an error reports a fault of the machine, not of the point.

    Parameters
    ----------
    error : BaseException
        The error an attempt raised.
    """
    if isinstance(error, _INFRASTRUCTURE_FAILURES) or _is_out_of_memory(error):
        return True

    return isinstance(error, OSError) and error.errno in INFRASTRUCTURE_ERRNOS


def _is_out_of_memory(error: BaseException, /) -> bool:
    """Return whether an exception reports host or accelerator memory exhaustion."""
    kind = type(error)
    return isinstance(error, MemoryError) or (
        kind.__name__ == "OutOfMemoryError"
        and kind.__module__.partition(".")[0] == "torch"
    )


def _interrupt(signal_number: int, frame: FrameType | None) -> None:
    """Raise the termination a signal asks for inside the executing point."""
    del frame

    raise PointTerminatedError(signal_number)


@contextmanager
def _terminations_interrupt() -> Iterator[None]:
    """Raise ``PointTerminatedError`` on ``SIGTERM`` for the duration of a block."""
    previous = signal.signal(signal.SIGTERM, _interrupt)

    try:
        yield

    finally:
        signal.signal(signal.SIGTERM, previous)


def _restore(directory: Path, recorded: PointFailure | None, /) -> None:
    """Return a point's failure record to what it held before an attempt."""
    if recorded is None:
        remove_failure(directory)

    else:
        write_failure(directory, recorded)


def execute_to_outcome(
    attempt: Callable[[int], RunOutcome],
    directory: Path,
    /,
    *,
    numerical: tuple[type[Exception], ...],
    derivation: PointSeed,
    infrastructure_attempts: int,
    failure_policy: FailurePolicySpec,
    halve_batch: Callable[[], bool],
) -> RunOutcome:
    """Attempt a point until it returns or its failure is recorded.

    The point's failure record carries the attempts already spent. A final
    record settles the point without attempting it. Otherwise each attempt is
    recorded as interrupted before it starts. A numerical failure and a
    ``PreparationError`` are recorded at once. An infrastructure failure is
    attempted again until the declared infrastructure attempts are spent across
    every launch, or until its declared retry transformation cannot be applied,
    and then recorded. ``SIGTERM`` during an attempt records the attempt as
    interrupted and raises ``PointTerminatedError``. Any other exception is a
    defect: the record is restored to what it held and the exception propagates.

    Parameters
    ----------
    attempt : Callable[[int], RunOutcome]
        Executes the point once for the numbered attempt.
    directory : Path
        The point's directory, where a failure is recorded.
    numerical : tuple[type[Exception], ...]
        The exceptions that report a numerical failure of the point.
    derivation : PointSeed
        The seed derivation a failure records.
    infrastructure_attempts : int
        The attempts the point receives across launches before infrastructure
        failure settles it.

    Returns
    -------
    RunOutcome
        The attempt's outcome, or a killed outcome once a failure is recorded.

    Raises
    ------
    ReseededRunError
        If the recorded failure was seeded under another derivation.
    PointTerminatedError
        If the process receives ``SIGTERM`` while an attempt runs.
    """
    recorded = read_failure(directory)

    if recorded is not None and recorded.seed_derivation != derivation:
        raise ReseededRunError(directory.name)

    if recorded is not None and recorded.final:
        return RunOutcome(directory, ManifestStatus.KILLED, cached=True)

    attempts = 0 if recorded is None else recorded.attempts
    numerical_retries = 0
    diagnostic = _UNFINISHED if recorded is None else recorded.diagnostic

    with _terminations_interrupt():
        while attempts < infrastructure_attempts:
            attempts += 1
            started = PointFailure(
                FailureKind.INTERRUPTED, _UNFINISHED, attempts, derivation
            )
            write_failure(directory, started)

            try:
                outcome = attempt(numerical_retries)

            except numerical as error:
                if failure_policy.numerical == "reseed":
                    numerical_retries += 1
                    diagnostic = str(error)
                    continue

                failure = PointFailure(
                    FailureKind.NUMERICAL, str(error), attempts, derivation
                )
                write_failure(directory, failure)

                return RunOutcome(directory, ManifestStatus.KILLED, cached=False)

            except PreparationError as error:
                failure = PointFailure(
                    FailureKind.PREPARATION, str(error), attempts, derivation
                )
                write_failure(directory, failure)

                return RunOutcome(directory, ManifestStatus.KILLED, cached=False)

            except PointTerminatedError as terminated:
                write_failure(directory, started._replace(diagnostic=str(terminated)))
                raise

            except BaseException as error:
                if _is_out_of_memory(error):
                    diagnostic = str(error)
                    if (
                        failure_policy.out_of_memory == "halve-batch"
                        and not halve_batch()
                    ):
                        break
                    continue

                if isinstance(error, OSError) and is_infrastructure(error):
                    diagnostic = str(error)
                    continue

                _restore(directory, recorded)
                raise

            remove_failure(directory)

            return outcome

    failure = PointFailure(FailureKind.INFRASTRUCTURE, diagnostic, attempts, derivation)
    write_failure(directory, failure)

    return RunOutcome(directory, ManifestStatus.KILLED, cached=False)


@final
class PointExecutor:
    """Execute points through one store with production training semantics.

    Every point is seeded from the experiment root its store is bound to: a
    point's run seed is derived from that root and the point's identifier, and
    its manifest records the root and the derivation. Every evaluator a run
    declares is scored. An executor whose store is addressed by an
    experiment's freeze reports the rows against that freeze: such a store
    holds only the directories of its freeze, and the executor binds each
    point's directory to the freeze and seed derivation before executing it,
    so the rows it records name the freeze they were produced under. An
    executor whose store names no freeze reports them as ``UNFROZEN``, which
    ingestion against a freeze refuses.

    Parameters
    ----------
    store : RunStore
        The store each point's run directory lies under, whose root seeds
        every point and whose freeze, if any, every row is reported against.
    codec : RunSpecCodec
        The codec a point's configuration is validated through.
    operations : TrainingOperations
        What prepares, recovers, fits and scores one planned run.
    environment : ExecutionEnvironment | None
        The recorded execution environment, or None to read the process's own.
    infrastructure_attempts : int
        The attempts each point receives across launches before infrastructure
        failure settles it.
    """

    __slots__ = (
        "_codec",
        "_environment",
        "_failure_policy",
        "_infrastructure_attempts",
        "_operations",
        "_store",
    )

    def __init__(
        self,
        store: RunStore,
        *,
        codec: RunSpecCodec,
        operations: TrainingOperations,
        infrastructure_attempts: int,
        failure_policy: FailurePolicySpec,
        environment: ExecutionEnvironment | None = None,
    ) -> None:
        self._store = store
        self._codec = codec
        self._operations = operations
        self._infrastructure_attempts = infrastructure_attempts
        self._failure_policy = failure_policy
        self._environment = environment

    @property
    def store(self) -> RunStore:
        """Return the run store this executor is bound to."""
        return self._store

    def _train(
        self,
        point: SweepPoint,
        blueprint: TrainingBlueprint,
        derived: PointSeed,
        directory: Path,
        streams: RestorableRandomStreams,
        policy: DevicePolicy,
        metric_sinks: tuple[MetricSink, ...],
        numerical_retry: int,
        /,
    ) -> ManifestStatus:
        """Fit and score inside the store's seeded callback, then record the rows.

        The metric sinks the run declares are built once for the point, receive
        what every infrastructure attempt logs, and are closed when point
        execution ends. Each attempt records the point's measurements and
        achieved reports in the run directory's metrics record, and the
        attempt that completes logs the held-out rows it records to every sink.
        """
        freeze_id = self._store.freeze_id
        sink = PointReportSink(derived.point_id)
        scoring = (
            HeldOutScoring(
                UNFROZEN if freeze_id is None else freeze_id, derived.root, sink
            )
            if blueprint.run_spec.evaluators
            else None
        )
        if numerical_retry:
            streams.reseed(numerical_retry)

        self._operations.execute(
            blueprint,
            directory,
            streams,
            policy,
            scoring=scoring,
            sinks=metric_sinks,
            recorder=MetricsRecorder(directory, point),
        )

        write_artifacts(directory, sink.artifacts)
        write_rows(directory, RecordedRows(freeze_id, derived, sink.rows))
        log_held_out_rows(metric_sinks, sink.rows)

        return ManifestStatus.COMPLETED

    def __call__(self, point: SweepPoint, /) -> RunOutcome:
        """Execute one point, delegating caching and lifecycle to its store.

        Raises
        ------
        PointExecutionError
            If the point carries no provenance.
        ReseededRunError
            If the point's directory, or the failure it records, was seeded
            under another derivation.
        PointTerminatedError
            If the process receives ``SIGTERM`` while the point executes.
        """
        if point.provenance is None:
            msg = f"the executable point {point.point_id!r} carries no provenance"
            raise PointExecutionError(msg)

        run_spec = resolve_point(point, self._codec)
        directory = self._store.directory(point)
        derived = self._store.derivation(point)

        if self._store.freeze_id is not None:
            self._store.claim(point)

        blueprint = plan_training(run_spec, point=derived)
        environment = self._environment or execution_environment()
        metric_sinks = _PointMetricSinks(
            blueprint.run_spec.sinks, SinkContext(derived.point_id)
        )

        def halve_batch() -> bool:
            nonlocal blueprint
            contracted = blueprint.run_spec.trainer.halve_batch()

            if contracted is None:
                return False
            blueprint = plan_training(
                blueprint.run_spec.with_(trainer=contracted), point=derived
            )
            return True

        def attempt(numerical_retry: int) -> RunOutcome:
            manifest = open_training_manifest(blueprint, derived, environment)
            record = training_record(point, blueprint)
            return self._store.execute(
                point,
                manifest,
                lambda _point, run, streams, policy: self._train(
                    point,
                    blueprint,
                    derived,
                    run,
                    streams,
                    policy,
                    metric_sinks.resolve(),
                    numerical_retry,
                ),
                record=record,
            )

        try:
            return execute_to_outcome(
                attempt,
                directory,
                numerical=(NumericalFailureError,),
                derivation=derived,
                infrastructure_attempts=self._infrastructure_attempts,
                failure_policy=self._failure_policy,
                halve_batch=halve_batch,
            )

        finally:
            metric_sinks.close()
