"""Run-layer protocols."""

from typing import TYPE_CHECKING, NamedTuple, Protocol, Self, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from lofi._types import JsonValue, ReportValue
    from lofi.eval import EvaluationArtifact, Report
    from lofi.nn.policy import DevicePolicy
    from lofi.run._boundary.active import ActiveLearningManifest
    from lofi.run._experiment import Experiment
    from lofi.run._training import TrainingBlueprint
    from lofi.run._values import RunPhase
    from lofi.train import (
        FittingReport,
        PreparedTraining,
        RestorableRandomStreams,
        ResumePosition,
    )

__all__ = [
    "ActiveLearningRun",
    "BeadPrediction",
    "CountedParameter",
    "ExperimentReporting",
    "FreezableModule",
    "FreezeMinting",
    "HeldOutScoring",
    "Ingested",
    "MetricSink",
    "ModelReleasing",
    "ModuleRegistrar",
    "PointExport",
    "PointPublication",
    "PointRecorder",
    "Published",
    "ReleaseAuthorization",
    "ReleasePublication",
    "ReportSink",
    "ReportingComponent",
    "ResultIngestion",
    "ResultSink",
    "SeedBudgeting",
    "TrainingOperations",
]


class ActiveLearningRun(Protocol):
    """Runs one active-learning document outside any experiment."""

    def __call__(
        self, document: Path, seed: int, out: Path, /
    ) -> ActiveLearningManifest:
        """Run the document's rounds under a seed, keeping them under ``out``.

        Parameters
        ----------
        document : Path
            The active-learning document.
        seed : int
            The run seed every stream derives from.
        out : Path
            The directory the rounds, manifest and rows are kept under.
        """
        ...


class BeadPrediction(Protocol):
    """Scores bead inference items with the trained point a run holds."""

    def __call__(
        self,
        run: Path,
        serving: Path,
        items: Path,
        destination: Path,
        /,
    ) -> int:
        """Write one prediction row per bead item and return the row count.

        The serving document declares the response encoding and the element
        each of the point's input roles reads.
        """
        ...


@runtime_checkable
class ModuleRegistrar[M](Protocol):
    """Registers named child modules."""

    def add_module(self, name: str, module: M) -> None:
        """Register a child module.

        Parameters
        ----------
        name : str
            Name used in module traversal and saved state.
        module : M
            Child module to register.
        """


class FreezableModule(Protocol):
    """Disables gradients for its registered parameters."""

    def requires_grad_(self, *, requires_grad: bool = True) -> Self:
        """Set whether registered parameters require gradients.

        Parameters
        ----------
        requires_grad : bool
            Whether gradients are required.

        Returns
        -------
        Self
            This module.
        """


class CountedParameter(Protocol):
    """Exposes a parameter's size and whether it takes a gradient."""

    @property
    def requires_grad(self) -> bool:
        """Return whether the parameter takes a gradient."""
        ...

    def numel(self) -> int:
        """Return the number of elements the parameter holds."""
        ...


@runtime_checkable
class ReportingComponent(Protocol):
    """A built component that reports what it measured about itself.

    A run records the report of every component role whose module provides
    one in the point's metrics record, under the role's name.
    """

    def report(self) -> Mapping[str, ReportValue]:
        """Return the component's named measurements."""
        ...


@runtime_checkable
class MetricSink(Protocol):
    """A sink for scalar metrics and tables."""

    def log_scalars(self, mapping: Mapping[str, float], step: int) -> None:
        """Log named scalar values at one step.

        Parameters
        ----------
        mapping : Mapping[str, float]
            The scalar metrics to log.
        step : int
            The training step.
        """

    def log_table(self, name: str, rows: Sequence[Mapping[str, JsonValue]]) -> None:
        """Log named table rows.

        Parameters
        ----------
        name : str
            The name used in validation or reporting.
        rows : Sequence[Mapping[str, JsonValue]]
            The table rows.
        """

    def close(self) -> None:
        """Close the sink."""


@runtime_checkable
class ReportSink(Protocol):
    """A report sink.

    Attributes
    ----------
    name : str
        Unique sink name.
    """

    @property
    def name(self) -> str:
        """Return the sink name."""

    def emit(self, report: Report) -> None:
        """Write a report.

        Parameters
        ----------
        report : Report
            Completed report to write.
        """


class ResultSink(Protocol):
    """A destination for the rows and files one experiment point reports.

    A held-out row states the computation budget it was scored under beside
    its value, so a report reaches the sink with that approximation. An
    evaluator artifact is stored under the evaluator role that emitted it.
    """

    def emit(self, report: Report, /) -> None:
        """Record a report's rows.

        Parameters
        ----------
        report : Report
            Completed report.
        """

    def emit_artifact(self, role: str, artifact: EvaluationArtifact, /) -> None:
        """Record one evaluator artifact under its declared role."""


class HeldOutScoring(NamedTuple):
    """The freeze, run seed and sink a trained point's held-out results go to.

    The freeze is ``freeze/<n>``, or ``UNFROZEN`` for a point scored
    outside any experiment.
    """

    freeze_id: str
    seed: int
    sink: ResultSink


class ExperimentReporting(Protocol):
    """Renders the tables one experiment document declares."""

    def __call__(
        self, experiment: Path, root: Path, /, *, pilot: bool
    ) -> tuple[Path, ...]:
        """Render declared tables from the experiment's stored result rows.

        Parameters
        ----------
        experiment : Path
            The experiment document.
        root : Path
            The repository root its results lie under.
        pilot : bool
            Whether the rows are the experiment's pilot results rather than
            its results under its freeze.
        """
        ...


class PointExport(Protocol):
    """Writes the offline bundle of one trained point."""

    def __call__(self, run: Path, release: Path, destination: Path, /) -> None:
        """Bundle the trained point a run directory holds.

        Parameters
        ----------
        run : Path
            The run directory of a completed point.
        release : Path
            The release declaration document.
        destination : Path
            The bundle directory to create.
        """
        ...


class ResultIngestion(Protocol):
    """Accepts an experiment's results against its freeze and writes them."""

    def __call__(self, experiment: Path, freeze: Path, root: Path, /) -> Ingested:
        """Ingest one experiment's results directory.

        Parameters
        ----------
        experiment : Path
            The experiment document.
        freeze : Path
            The ``FREEZE.json`` of the freeze it ran under.
        root : Path
            The repository root the experiment's results lie under.

        Raises
        ------
        ValueError
            If the results do not conform to the freeze.
        """
        ...


class Ingested(NamedTuple):
    """How many rows an ingestion accepted and the files it wrote."""

    rows: int
    table: Path
    audit: Path


class ReleaseAuthorization(NamedTuple):
    """The explicit legal and redistribution approval for one public write."""

    reviewer: str
    license_identifier: str
    corpus_redistribution_confirmed: bool
    artifact_redistribution_confirmed: bool


class ReleasePublication(NamedTuple):
    """What an authorized public release names beyond its local preflight.

    Attributes
    ----------
    evidence : Path
        Where the release evidence is recorded.
    authorization : ReleaseAuthorization
        The reviewer, the reviewed license and both redistribution
        confirmations.
    token_environment : str
        The environment variable the Hub token is read from.
    """

    evidence: Path
    authorization: ReleaseAuthorization
    token_environment: str


class FreezeMinting(Protocol):
    """Mints one freeze and returns the record it wrote."""

    def __call__(
        self,
        root: Path,
        /,
        *,
        number: int,
        experiments: Sequence[Path],
        readiness: Path,
    ) -> Mapping[str, JsonValue]:
        """Tag a clean ``HEAD`` as a freeze and record what it froze.

        Parameters
        ----------
        root : Path
            The repository root.
        number : int
            The freeze number.
        experiments : Sequence[Path]
            The experiment documents, relative to the root, run against it.
        readiness : Path
            The PI's readiness notes, one line per readiness condition.
        """
        ...


class SeedBudgeting(Protocol):
    """Prices an experiment's seed count from a pilot's rows and writes it."""

    def __call__(self, experiment: Experiment, pilot: Path, /) -> Path:
        """Write the experiment's seed budget beside its document.

        Parameters
        ----------
        experiment : Experiment
            The experiment whose evaluation section declares the budget.
        pilot : Path
            The results directory whose rows score the budgeted metric.
        """
        ...


class ModelReleasing(Protocol):
    """Prepares, and when authorized publishes, one model release."""

    def __call__(
        self,
        bundle: Path,
        repo_id: str,
        run: Path,
        /,
        *,
        publication: ReleasePublication | None,
        environ: Mapping[str, str],
    ) -> Mapping[str, JsonValue]:
        """Verify a release locally and publish it when a publication is given.

        Parameters
        ----------
        bundle : Path
            The offline bundle ``lofi export`` wrote.
        repo_id : str
            The ``owner/name`` repository the release is published to.
        run : Path
            The run directory whose declared release checkpoint the bundle
            reproduces.
        publication : ReleasePublication | None
            The authorization of a public release, or None for the read-only
            preflight.
        environ : Mapping[str, str]
            The environment the Hub token is read from.
        """
        ...


class Published(NamedTuple):
    """The records one evaluator role's predictions were published as.

    Attributes
    ----------
    role : str
        The evaluator role whose predictions were published.
    path : Path
        The records file written, one record per line.
    records : int
        How many items' records were written.
    unlabeled : int
        How many items of the split the role's evaluator selected no
        derivation for, and so wrote no record for.
    """

    role: str
    path: Path
    records: int
    unlabeled: int


class PointPublication(Protocol):
    """Publishes a trained point's predictions as external records."""

    def __call__(
        self, run: Path, tokenizations: Path, selection: Path, destination: Path, /
    ) -> tuple[Published, ...]:
        """Publish the predictions a selection names and report each role's file.

        Parameters
        ----------
        run : Path
            The run directory of a completed point.
        tokenizations : Path
            The document joining each item to its tokenization record.
        selection : Path
            The document naming the evaluator roles, the split and the
            formalism the records are published under.
        destination : Path
            The records directory to create, which must not exist.

        Raises
        ------
        ValueError
            If the point, the documents or the selected predictions are
            refused.
        """
        ...


class TrainingOperations(Protocol):
    """Trains one lowered blueprint and scores what it trained."""

    def execute(
        self,
        blueprint: TrainingBlueprint,
        directory: Path,
        streams: RestorableRandomStreams,
        policy: DevicePolicy,
        /,
        *,
        scoring: HeldOutScoring | None,
        sinks: Sequence[MetricSink],
        recorder: PointRecorder,
    ) -> None:
        """Prepare, recover and fit one training point, then score it.

        Parameters
        ----------
        blueprint : TrainingBlueprint
            The lowered run.
        directory : Path
            The run directory checkpoints are written under.
        streams : RestorableRandomStreams
            The run's applied random streams.
        policy : DevicePolicy
            The run's device policy.
        scoring : HeldOutScoring | None
            Where the held-out evaluators the run declares report their rows,
            or None to score no evaluator.
        sinks : Sequence[MetricSink]
            Where the metrics of fitting are logged as it goes. The caller
            built them and closes them.
        recorder : PointRecorder
            What records the point's measurements and achieved reports in its
            run directory.
        """


class PointRecorder(Protocol):
    """Records what one training point measured and achieved as it runs.

    A recorder is bound to one point's run directory. It is told what
    preparation built, how long each phase took, what fitting completed and
    what components contribute, and it writes the point's metrics record once
    preparation ends and again when the point closes.
    """

    def prepared(
        self, blueprint: TrainingBlueprint, prepared: PreparedTraining, /
    ) -> None:
        """Record what preparation built and write the prepared record.

        Parameters
        ----------
        blueprint : TrainingBlueprint
            The lowered run.
        prepared : PreparedTraining
            The model, optimization and fitting plan preparation built.
        """

    def timed(self, phase: RunPhase, seconds: float, /) -> None:
        """Add the wall time one phase of the point took.

        Parameters
        ----------
        phase : RunPhase
            The phase.
        seconds : float
            The wall time it took, in seconds.
        """

    def fitted(self, report: FittingReport, start: ResumePosition, /) -> None:
        """Record what one fitting invocation completed.

        Parameters
        ----------
        report : FittingReport
            What the invocation completed.
        start : ResumePosition
            The position the invocation started from.
        """

    def contribute(self, name: str, report: Mapping[str, ReportValue], /) -> None:
        """Record a named report a component or a stage contributes.

        Parameters
        ----------
        name : str
            The name the report is recorded under.
        report : Mapping[str, ReportValue]
            The named measurements.
        """

    def closed(self, *, peak_memory_bytes: int | None) -> None:
        """Write the closed record.

        Parameters
        ----------
        peak_memory_bytes : int | None
            The most bytes the run device held at once during the point, or
            None where the device reports no peak.
        """
