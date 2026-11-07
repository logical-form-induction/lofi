"""Content-addressed run directories, the cache they form and their history.

A store addressed by a freeze keeps its directories under that freeze, so two
freezes never share or touch one directory: a point a new freeze offers is
executed again in a directory of its own, and what another freeze completed
stays as it was. Every store is bound to one experiment root, and a directory
recorded under another root's seed derivation is refused rather than served.
An experiment reads such a store through an ``ExperimentStore``, which binds
each point it submits to the experiment's freeze and seed derivation. A pilot
of an experiment reads a store addressed by no freeze: it binds each point it
submits as ``UNFROZEN``, so its directories and rows are never those of a
freeze.
"""

from enum import StrEnum
from typing import TYPE_CHECKING, Final, NamedTuple, final

from lofi.eval import UNFROZEN, freeze_number
from lofi.run._boundary.history import RunHistory
from lofi.run._boundary.manifest import (
    FailureKind,
    ManifestStatus,
    PointBinding,
    close_manifest,
    decode_seed_derivation,
)
from lofi.run._boundary.site import RunStatus
from lofi.run._device import accelerator_of, device_scope
from lofi.run._launch import ExperimentIdentity
from lofi.run._manifest import (
    MANIFEST_NAME,
    RESOLVED_NAME,
    SCHEMA_FINGERPRINT_NAME,
    read_binding,
    read_failure,
    read_manifest,
    read_record,
    read_rows,
    write_binding,
    write_manifest,
    write_record,
)
from lofi.run._provenance import CommittedRun
from lofi.run._seed import ExperimentSeed, PointSeed, RandomStreams

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from lofi._types import JsonValue
    from lofi.nn.policy import DevicePolicy
    from lofi.run._boundary.manifest import (
        ExperimentManifest,
        PointFailure,
    )
    from lofi.run._boundary.sweep import SweepPoint
    from lofi.run._experiment import Experiment
    from lofi.run._manifest import (
        RunRecord,
        StoredRecord,
    )
    from lofi.spec import ArchivedLanguage
    from lofi.train import RestorableRandomStreams

__all__ = [
    "MANIFEST_NAME",
    "RUNS_NAME",
    "ExperimentStore",
    "PointStatus",
    "ReseededRunError",
    "RunOutcome",
    "RunStore",
    "Training",
]

RUNS_NAME: Final = "runs"
"""The directory under a store's scope that holds one directory per point."""

type Training = Callable[
    [SweepPoint, Path, RestorableRandomStreams, DevicePolicy], ManifestStatus
]
"""Trains one point under the store's random-stream and device policies."""


class ReseededRunError(ValueError):
    """A recorded run whose seed was derived otherwise than the run offered now.

    Attributes
    ----------
    point_id : str
        The identifier of the point whose recorded run was seeded otherwise.
    """

    def __init__(self, point_id: str) -> None:
        self.point_id = point_id
        super().__init__(
            f"the run of the point {point_id!r} records a seed_derivation other "
            f"than the experiment root now given, so it is not served or resumed "
            f"under it"
        )


class PointStatus(StrEnum):
    """The status of an experiment point whose directory records no failure."""

    COMPLETED = "completed"
    RUNNING = "running"
    KILLED = "killed"
    AWAITING_RESPONSES = "awaiting-responses"
    SUBMITTED = "submitted"
    NOT_SUBMITTED = "not-submitted"


class RunOutcome(NamedTuple):
    """The result of executing one point against a store."""

    directory: Path
    status: ManifestStatus
    cached: bool


class RunStore:
    """Run directories under one scope, each named by a point identifier.

    A store addressed by no freeze has its root as its scope. A store
    addressed by a freeze ``freeze/<n>`` has ``<root>/freeze/<n>``
    as its scope, so its directories are its freeze's alone. A point's
    directory is ``<scope>/runs/<point_id>/``, so two points with the same
    resolved configuration and seed share one directory within a scope
    whichever sweeps produced them. The directory holds the experiment manifest under
    ``manifest.json`` beside the resolved configuration, its sources, the
    configuration fingerprint and the environment record. A point whose
    manifest is closed as completed is a cache hit: executing it again trains
    nothing. Any other state trains again: a manifest closed as killed, which
    records training that was interrupted or raised, and a manifest left
    running, which a process that ended without closing it leaves behind.

    The store is bound to one experiment root. A directory whose recorded run or
    final failure names a seed derivation other than the one that root derives
    for its point is neither served nor offered again: reading its outcome
    raises, so a point is never served under another root.

    Parameters
    ----------
    root : Path
        The directory the store's scope lies under.
    freeze_id : str | None
        The freeze the store's directories are addressed by, or None for a
        store outside any experiment.
    seed : ExperimentSeed
        The experiment root every point's run seed is derived from.

    Raises
    ------
    ValueError
        If the freeze identifier is not of the form ``freeze/<n>``.
    """

    __slots__ = ("_freeze_id", "_root", "_seed")

    def __init__(
        self, root: Path, *, freeze_id: str | None, seed: ExperimentSeed
    ) -> None:
        if freeze_id is not None:
            freeze_number(freeze_id)

        self._root = root
        self._freeze_id = freeze_id
        self._seed = seed

    @property
    def root(self) -> Path:
        """Return the store root."""
        return self._root

    @property
    def freeze_id(self) -> str | None:
        """Return the store's freeze, or None outside an experiment."""
        return self._freeze_id

    @property
    def seed(self) -> ExperimentSeed:
        """Return the experiment root every point's run seed is derived from."""
        return self._seed

    def derivation(self, point: SweepPoint, /) -> PointSeed:
        """Return the run seed the store's root derives for a point.

        Parameters
        ----------
        point : SweepPoint
            The point whose seed is derived.
        """
        return self._seed.point(point.point_id, point.seed)

    @property
    def scope(self) -> Path:
        """Return the directory the store's runs and submissions lie under."""
        return self._root if self._freeze_id is None else self._root / self._freeze_id

    def directory(self, point: SweepPoint, /) -> Path:
        """Return the run directory a point is addressed by.

        Parameters
        ----------
        point : SweepPoint
            The point whose identifier names the directory.
        """
        return self.scope / RUNS_NAME / point.point_id

    def manifest(self, point: SweepPoint, /) -> ExperimentManifest | None:
        """Return the manifest recorded for a point, or None when none is.

        Parameters
        ----------
        point : SweepPoint
            The point whose run directory is read.
        """
        return read_manifest(self.directory(point))

    def record(self, point: SweepPoint, /) -> StoredRecord | None:
        """Return the files recorded beside a point's manifest, or None when none are.

        Parameters
        ----------
        point : SweepPoint
            The point whose run directory is read.
        """
        if self.manifest(point) is None:
            return None

        return read_record(self.directory(point))

    def status(self, point: SweepPoint, /) -> RunStatus:
        """Return the run status a point's manifest and failure record give.

        A final failure record is a failed run, whatever the manifest a killed
        process left behind still says. Otherwise no manifest is a submitted
        run, a running manifest a running one, a completed manifest a completed
        one and a killed manifest a failed one.

        Parameters
        ----------
        point : SweepPoint
            The point whose run directory is read.
        """
        failure = self.failure(point)

        if failure is not None and failure.final:
            return RunStatus.FAILED

        manifest = self.manifest(point)

        if manifest is None:
            return RunStatus.SUBMITTED

        match manifest.status:
            case ManifestStatus.RUNNING:
                return RunStatus.RUNNING

            case ManifestStatus.COMPLETED:
                return RunStatus.COMPLETED

            case ManifestStatus.KILLED:
                return RunStatus.FAILED

    def claim(self, point: SweepPoint, /) -> None:
        """Bind a point's directory to the store's freeze and seed derivation.

        Parameters
        ----------
        point : SweepPoint
            The point submitted.

        Raises
        ------
        ValueError
            If the store is addressed by no freeze.
        ReseededRunError
            If the directory is bound under another derivation.
        """
        if self._freeze_id is None:
            msg = "a store addressed by no freeze binds no point to an experiment"
            raise ValueError(msg)

        directory = self.directory(point)
        derivation = self.derivation(point)
        recorded = read_binding(directory)

        if recorded is not None and recorded.seed_derivation != derivation:
            raise ReseededRunError(point.point_id)

        write_binding(directory, PointBinding(self._freeze_id, derivation))

    def failure(self, point: SweepPoint, /) -> PointFailure | None:
        """Return the failure a point's directory records, or None when it records none.

        Parameters
        ----------
        point : SweepPoint
            The point whose directory is read.
        """
        return read_failure(self.directory(point))

    def history(self) -> RunHistory:
        """Return the history of the runs directory, initializing it when absent."""
        return RunHistory(self.scope / RUNS_NAME)

    def committable(
        self, point: SweepPoint, languages: Mapping[str, ArchivedLanguage], /
    ) -> CommittedRun:
        """Return a point's run as it would be committed, read from its directory.

        Parameters
        ----------
        point : SweepPoint
            The point whose run directory is read.
        languages : Mapping[str, ArchivedLanguage]
            Every indexed language the run was resolved against, keyed by its
            declared name.

        Raises
        ------
        FileNotFoundError
            If the directory holds no resolved configuration or fingerprint.
        ValueError
            If the languages are not the ones the directory's fingerprint names.
        """
        directory = self.directory(point)

        return CommittedRun(
            point.point_id,
            configuration=(directory / RESOLVED_NAME).read_bytes(),
            fingerprint=(directory / SCHEMA_FINGERPRINT_NAME).read_bytes(),
            languages=languages,
        )

    def verify(self, point: SweepPoint, history: RunHistory, /) -> CommittedRun:
        """Return a point's committed run once its directory is shown to still hold it.

        The files the directory holds now are compared byte for byte, through
        their identifiers, with the bytes the history committed. Neither the
        directory nor the history is written.

        Parameters
        ----------
        point : SweepPoint
            The point whose directory is checked.
        history : RunHistory
            The history the point was committed to.

        Raises
        ------
        UncommittedRunError
            If the history holds no commit for the point.
        TamperedRecordError
            If a file no longer hashes to its committed identifier.
        FileNotFoundError
            If the directory holds no resolved configuration or fingerprint.
        """
        committed = history.replay(point.point_id)
        directory = self.directory(point)
        committed.require_persisted(
            configuration=(directory / RESOLVED_NAME).read_bytes(),
            fingerprint=(directory / SCHEMA_FINGERPRINT_NAME).read_bytes(),
        )

        return committed

    def is_completed(self, point: SweepPoint, /) -> bool:
        """Return whether a point's recorded manifest is closed as completed.

        A recorded run, in whatever status, is read only when the seed
        derivation its manifest records is the one the store's root derives
        for the point, so a run of another root is neither served nor resumed.

        Parameters
        ----------
        point : SweepPoint
            The point whose run directory is read.

        Raises
        ------
        ReseededRunError
            If the recorded manifest names a seed derivation other than the
            one the store's root derives for the point.
        """
        manifest = self.manifest(point)

        if manifest is None:
            return False

        if decode_seed_derivation(manifest.seed_derivation) != self.derivation(point):
            raise ReseededRunError(point.point_id)

        return manifest.status is ManifestStatus.COMPLETED

    def is_final(self, point: SweepPoint, /) -> bool:
        """Return whether a point's outcome is settled, so it is not executed again.

        A completed run and a final failure record are settled.

        Parameters
        ----------
        point : SweepPoint
            The point whose run directory is read.

        Raises
        ------
        ReseededRunError
            If the recorded manifest or failure names a seed derivation other
            than the one the store's root derives for the point.
        """
        failure = self.failure(point)
        derivation = self.derivation(point)

        if failure is not None and failure.seed_derivation != derivation:
            raise ReseededRunError(point.point_id)

        return self.is_completed(point) or (failure is not None and failure.final)

    def execute(
        self,
        point: SweepPoint,
        manifest: ExperimentManifest,
        train: Training,
        /,
        *,
        record: RunRecord,
    ) -> RunOutcome:
        """Train a point unless its directory already holds a completed run.

        The record and the running manifest are on disk before training is
        called. Training that returns closes the manifest with the status it
        returns. A run that completes is committed to the store's history,
        with the record's declarations, before its manifest is closed, so a
        directory closed as completed is a run its history holds. Training
        that raises, as an interrupt from a signal does, closes the manifest
        as killed and the exception propagates.

        Parameters
        ----------
        point : SweepPoint
            The point to train.
        manifest : ExperimentManifest
            The running manifest written before training starts and closed
            with the status training returns.
        train : Training
            Called once with the point, its directory, the single applied
            random-stream policy and the exact device policy yielded by the run
            scope when the point is not already completed.
        record : RunRecord
            The sources, fingerprint and accelerator runtime written beside
            the manifest, and the declarations a completed run commits.

        Returns
        -------
        RunOutcome
            The directory, the terminal status, and whether the completed run
            was served from the directory without training.

        Raises
        ------
        ReseededRunError
            If the manifest's seed derivation is not derived from the store's
            root, or the directory records a run, completed or not, whose seed
            derivation is not the manifest's; nothing is written or trained.
        ValueError
            If the manifest's seed derivation names another point, a completed
            run was seeded under another policy, the manifest is not running,
            training returns
            the running status, or the history holds the point under a
            different record. A commit that fails leaves the manifest running.
        """
        directory = self.directory(point)
        seed_policy = manifest.seed_policy.policy()
        derived = decode_seed_derivation(manifest.seed_derivation)

        if (derived.point_id, derived.seed) != (point.point_id, point.seed):
            msg = (
                f"seed_derivation names the point {derived.point_id!r} at seed "
                f"{derived.seed}, not the run point {point.point_id!r} at seed "
                f"{point.seed}"
            )
            raise ValueError(msg)

        if derived.experiment != self._seed:
            raise ReseededRunError(point.point_id)

        device = manifest.run_spec.device_policy.device
        expected_accelerator = accelerator_of(device)

        if seed_policy.accelerator != expected_accelerator:
            msg = (
                f"seed_policy accelerator {seed_policy.accelerator!r} does not "
                f"match run device {device!r}"
            )
            raise ValueError(msg)

        if self.is_completed(point):
            stored = self.manifest(point)

            if stored is not None and stored.seed_policy != manifest.seed_policy:
                msg = "seed_policy does not match the completed run"
                raise ValueError(msg)

            return RunOutcome(directory, ManifestStatus.COMPLETED, cached=True)

        if manifest.status is not ManifestStatus.RUNNING:
            msg = f"a run opens with a running manifest, not {manifest.status!r}"
            raise ValueError(msg)

        write_record(directory, manifest, record)
        write_manifest(directory, manifest)

        try:
            with device_scope(manifest.run_spec) as policy:
                streams = RandomStreams(seed_policy)
                streams.apply()
                status = train(point, directory, streams, policy)

        except BaseException:
            write_manifest(directory, close_manifest(manifest, ManifestStatus.KILLED))
            raise

        closed = close_manifest(manifest, status)

        if closed.status is ManifestStatus.COMPLETED:
            self.history().commit(self.committable(point, record.languages))

        write_manifest(directory, closed)

        return RunOutcome(directory, closed.status, cached=False)


@final
class ExperimentStore:
    """A run store read and claimed on behalf of one experiment.

    A point of the experiment is completed when its directory is bound to the
    experiment's freeze under the point's seed derivation, its manifest is closed
    as completed, and it records rows produced under that freeze and
    derivation.

    A pilot store runs the experiment outside any freeze. Its directories lie
    under the root itself, as a store addressed by no freeze has them, each
    point it submits is bound as ``UNFROZEN``, and a point is completed only
    by rows recorded outside any freeze.

    Parameters
    ----------
    root : Path
        The root of the run store, addressed by the experiment's freeze unless
        the store is a pilot's, that the point directories lie under.
    experiment : Experiment
        The experiment whose freeze and root seed bind each point.
    pilot : bool
        Whether the store runs the experiment as a pilot, outside its freeze.
    """

    __slots__ = ("_experiment", "_pilot", "_runs")

    def __init__(
        self, root: Path, experiment: Experiment, *, pilot: bool = False
    ) -> None:
        self._runs = RunStore(
            root,
            freeze_id=None if pilot else experiment.freeze_id,
            seed=experiment.seed_policy,
        )
        self._experiment = experiment
        self._pilot = pilot

    @property
    def runs(self) -> RunStore:
        """Return the store, addressed by the experiment's freeze, of its points."""
        return self._runs

    @property
    def experiment(self) -> Experiment:
        """Return the experiment the store is read for."""
        return self._experiment

    @property
    def pilot(self) -> bool:
        """Return whether the store runs the experiment outside its freeze."""
        return self._pilot

    @property
    def freeze_id(self) -> str:
        """Return the freeze every row is reported against, ``UNFROZEN`` for a pilot."""
        return UNFROZEN if self._pilot else self._experiment.freeze_id

    @property
    def root(self) -> Path:
        """Return the store root."""
        return self._runs.root

    @property
    def scope(self) -> Path:
        """Return the directory the experiment's runs and submissions lie under."""
        return self._runs.scope

    @property
    def identity(self) -> ExperimentIdentity:
        """Return the experiment, by name, and the freeze its rows report."""
        return ExperimentIdentity(self._experiment.name, self.freeze_id)

    def directory(self, point: SweepPoint, /) -> Path:
        """Return the directory a point is addressed by.

        Parameters
        ----------
        point : SweepPoint
            The point whose identifier names the directory.
        """
        return self._runs.directory(point)

    def binding(self, point: SweepPoint, /) -> PointBinding:
        """Return the freeze and seed derivation the experiment binds a point to.

        Parameters
        ----------
        point : SweepPoint
            The point bound.
        """
        return PointBinding(self.freeze_id, self._runs.derivation(point))

    def claim(self, point: SweepPoint, /) -> None:
        """Bind a point's directory to the experiment, as it is submitted.

        Parameters
        ----------
        point : SweepPoint
            The point submitted.

        Raises
        ------
        ReseededRunError
            If the directory is bound under another derivation.
        """
        directory = self.directory(point)
        binding = self.binding(point)
        recorded = read_binding(directory)

        if recorded is not None and recorded.seed_derivation != binding.seed_derivation:
            raise ReseededRunError(point.point_id)

        write_binding(directory, binding)

    def manifest(self, point: SweepPoint, /) -> bytes | None:
        """Return the bytes of the manifest a point's directory records, if any.

        Parameters
        ----------
        point : SweepPoint
            The point whose directory is read.
        """
        path = self.directory(point) / self._experiment.config.manifest_name

        return path.read_bytes() if path.is_file() else None

    def entry(self, point: SweepPoint, /) -> dict[str, JsonValue]:
        """Return the status entry of a point, as the experiment's status records it.

        A point whose directory is not bound is not submitted. A final failure
        is reported with its diagnostic and attempt count. A bound point with
        no manifest is submitted; a running manifest is running; a completed
        manifest with rows produced under this freeze is completed; a manifest
        awaiting the responses to a collected batch awaits them and is resumed
        when submitted again; any other manifest is killed and awaits
        resubmission.

        Parameters
        ----------
        point : SweepPoint
            The point whose directory is read.

        Raises
        ------
        ReseededRunError
            If the directory, its failure or its rows were seeded under
            another derivation.
        """
        directory = self.directory(point)
        expected = self.binding(point)
        bound = read_binding(directory)

        if bound is None:
            return {"status": PointStatus.NOT_SUBMITTED.value}

        derivation = expected.seed_derivation

        if bound.seed_derivation != derivation:
            raise ReseededRunError(point.point_id)

        failure = read_failure(directory)

        if failure is not None and failure.seed_derivation != derivation:
            raise ReseededRunError(point.point_id)

        if failure is not None and failure.final:
            return {
                "status": failure.kind.value,
                "diagnostic": failure.diagnostic,
                "attempts": failure.attempts,
            }

        data = self.manifest(point)

        if data is None:
            return {"status": PointStatus.SUBMITTED.value}

        recorded = self._experiment.config.manifest_point(data)
        rows = read_rows(directory)
        produced = (
            rows is not None
            and rows.freeze_id == self._runs.freeze_id
            and rows.seed_derivation == derivation
        )

        if recorded.status is ManifestStatus.COMPLETED and produced:
            return {"status": PointStatus.COMPLETED.value}

        if recorded.awaiting_responses:
            unfinished = PointStatus.AWAITING_RESPONSES
        elif recorded.status is ManifestStatus.RUNNING:
            unfinished = PointStatus.RUNNING
        else:
            unfinished = PointStatus.KILLED

        return {"status": unfinished.value}

    def is_completed(self, point: SweepPoint, /) -> bool:
        """Return whether a point is completed under the experiment's freeze.

        Parameters
        ----------
        point : SweepPoint
            The point whose directory is read.

        Raises
        ------
        ReseededRunError
            If the directory was seeded under another derivation.
        """
        return self.entry(point)["status"] == PointStatus.COMPLETED.value

    def is_final(self, point: SweepPoint, /) -> bool:
        """Return whether a point's outcome under the experiment's freeze is settled.

        A completed point and a final failure are settled, so neither is
        offered again.

        Parameters
        ----------
        point : SweepPoint
            The point whose directory is read.

        Raises
        ------
        ReseededRunError
            If the directory was seeded under another derivation.
        """
        status = self.entry(point)["status"]

        return status == PointStatus.COMPLETED.value or status in {
            kind.value for kind in FailureKind
        }

    def status(self, point: SweepPoint, /) -> RunStatus:
        """Return the run status of a point under the experiment's freeze.

        Parameters
        ----------
        point : SweepPoint
            The point whose directory is read.
        """
        match self.entry(point)["status"]:
            case PointStatus.COMPLETED.value:
                return RunStatus.COMPLETED

            case PointStatus.RUNNING.value:
                return RunStatus.RUNNING

            case (
                PointStatus.SUBMITTED.value
                | PointStatus.NOT_SUBMITTED.value
                | PointStatus.AWAITING_RESPONSES.value
            ):
                return RunStatus.SUBMITTED

            case _:
                return RunStatus.FAILED
