import errno
import os
import signal
from pathlib import Path
from typing import TYPE_CHECKING, Never, override

import pytest

from lofi.eval import (
    UNFROZEN,
    Approximation,
    ConventionVector,
    DeclaredBudget,
    EvaluationArtifact,
    MetricResult,
    ReportIdentity,
    ReportRow,
    Resampling,
    report,
)
from lofi.run import (
    ExecutionEnvironment,
    ExperimentSeed,
    FailureKind,
    FailurePolicySpec,
    ManifestStatus,
    MetricsRecorder,
    NullSink,
    PointExecutionError,
    PointExecutor,
    PointFailure,
    PointSeed,
    PointTerminatedError,
    PreparationError,
    ReseededRunError,
    RunStore,
    SeedPolicy,
    Selection,
    SinkContext,
    SweepPoint,
    decode_seed_derivation,
    installed_packages,
    is_infrastructure,
    read_binding,
    read_failure,
    read_rows,
    resolved_document,
    trace,
    write_failure,
)
from lofi.run import _executor as executor_module
from lofi.run._sinks import ARTIFACT_DIRECTORY, HELD_OUT_TABLE
from lofi.spec import RunSpecCodec, config_search_path
from lofi.spec.trainers.parser import ParserTrainerSpec
from lofi.train import ActivationMemoryExceededError, NumericalFailureError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from types import FrameType

    from lofi._types import JsonValue
    from lofi.nn.policy import DevicePolicy
    from lofi.run import (
        HeldOutScoring,
        MetricSink,
        PointRecorder,
        TrainingBlueprint,
    )
    from lofi.spec import SinkSpec
    from lofi.train import RestorableRandomStreams

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "lofi/configs/run/synthetic-parser.yaml"


ENVIRONMENT = ExecutionEnvironment(installed_packages(), "0123abcd", "cpu")
EXPERIMENT_SEED = ExperimentSeed(7)
FREEZE = "freeze/1"
INFRASTRUCTURE_ATTEMPTS = 3
FAILURE_POLICY = FailurePolicySpec(numerical="final", out_of_memory="retry")


class _Grammar:
    fingerprint = "grammar:executor-test"


class UnreachedTraining:
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
        del blueprint, directory, streams, policy, scoring, sinks, recorder
        msg = "the executor reached training after refusing the point"
        raise AssertionError(msg)


class RecordingTraining:
    def __init__(self) -> None:
        self.roots: list[JsonValue] = []
        self.scorings: list[HeldOutScoring | None] = []
        self.sinks: list[tuple[MetricSink, ...]] = []
        self.recorders: list[tuple[Path, PointRecorder]] = []

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
        del policy
        self.recorders.append((directory, recorder))
        self.roots.append(streams.seed_policy["root"])
        self.scorings.append(scoring)
        self.sinks.append(tuple(sinks))

        if scoring is not None:
            approximation = Approximation(DeclaredBudget(5, 8, mass=1.0), residual=None)
            row = ReportRow(
                identity=ReportIdentity(
                    blueprint.run_spec,
                    _Grammar(),
                    ConventionVector(
                        punctuation="keep",
                        unary_chains="keep",
                        trivial_spans="keep",
                        minimum_length=0,
                        aggregation="sentence",
                        coverage="zero",
                        self_arcs="allow",
                        gold_derivation="canonical",
                    ),
                    freeze_id=scoring.freeze_id,
                ),
                method="ccg",
                suite="held-out",
                task="brackets",
                dataset="synthetic-parser",
                split="dev",
                metric="unlabeled-bracketing-f1",
                result=MetricResult(0.5, 2, Resampling("sentence")),
                approximation=approximation,
                seed=scoring.seed,
            )
            scoring.sink.emit(report((row,)))
            scoring.sink.emit_artifact(
                "brackets",
                EvaluationArtifact("annotations.json", b'{"kind":"annotations"}\n'),
            )


class FailingTraining:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.attempts = 0

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
        del blueprint, directory, streams, policy, scoring, sinks, recorder
        self.attempts += 1

        raise self.error


class OutOfMemoryOnce(RecordingTraining):
    def __init__(self, error: MemoryError) -> None:
        super().__init__()
        self.error = error
        self.batches: list[int] = []

    @override
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
        trainer = blueprint.run_spec.trainer
        assert isinstance(trainer, ParserTrainerSpec)
        self.batches.append(trainer.batch_size)

        if len(self.batches) == 1:
            raise self.error

        super().execute(
            blueprint,
            directory,
            streams,
            policy,
            scoring=scoring,
            sinks=sinks,
            recorder=recorder,
        )


def a_traced_point(seed: int, *, overrides: tuple[str, ...] = ()) -> SweepPoint:
    composition = trace(
        Selection(
            config=CONFIG,
            profile=None,
            overrides=overrides,
            overlays=(),
            search_path=config_search_path(ROOT),
        )
    )

    return SweepPoint(
        resolved_document(composition.spec),
        provenance=composition.provenance,
        seed=seed,
    )


def execute(
    root: Path, point: SweepPoint, seed: ExperimentSeed
) -> tuple[PointSeed, SeedPolicy, JsonValue]:
    store = RunStore(root, freeze_id=None, seed=seed)
    training = RecordingTraining()
    PointExecutor(
        store,
        codec=RunSpecCodec(),
        operations=training,
        infrastructure_attempts=INFRASTRUCTURE_ATTEMPTS,
        failure_policy=FAILURE_POLICY,
        environment=ENVIRONMENT,
    )(point)
    manifest = store.manifest(point)

    assert manifest is not None

    policy = manifest.seed_policy.policy()
    (trained,) = training.roots

    return decode_seed_derivation(manifest.seed_derivation), policy, trained


def test_experiment_roots_drive_the_seed_each_point_trains_under(
    tmp_path: Path,
) -> None:
    points = [a_traced_point(seed) for seed in (0, 1)]

    first = [execute(tmp_path / "a", point, ExperimentSeed(7)) for point in points]
    second = [execute(tmp_path / "b", point, ExperimentSeed(8)) for point in points]

    for (seed, policy, trained), (other, _, other_trained) in zip(
        first, second, strict=True
    ):
        assert seed.root == policy.root == trained
        assert trained != other_trained
        assert seed.point_id == other.point_id

    assert first[0][2] != first[1][2]


def test_each_point_is_recorded_by_a_metrics_recorder_bound_to_its_run(
    tmp_path: Path,
) -> None:
    point = a_traced_point(0)
    store = RunStore(tmp_path, freeze_id=None, seed=ExperimentSeed(7))
    training = RecordingTraining()
    PointExecutor(
        store,
        codec=RunSpecCodec(),
        operations=training,
        infrastructure_attempts=INFRASTRUCTURE_ATTEMPTS,
        failure_policy=FAILURE_POLICY,
        environment=ENVIRONMENT,
    )(point)
    ((directory, recorder),) = training.recorders

    assert isinstance(recorder, MetricsRecorder)
    assert directory == store.directory(point)


def test_an_identical_point_reproduces_its_seed_under_the_same_root(
    tmp_path: Path,
) -> None:
    point = a_traced_point(3)

    first = execute(tmp_path / "a", point, ExperimentSeed(7))
    again = execute(tmp_path / "b", point, ExperimentSeed(7))

    assert first == again


def test_every_manifest_records_the_experiment_root_and_derivation(
    tmp_path: Path,
) -> None:
    point = a_traced_point(2)

    seed, policy, _ = execute(tmp_path, point, ExperimentSeed(11))

    assert seed == ExperimentSeed(11).point(point.point_id, point.seed)
    assert seed.experiment == ExperimentSeed(11)
    assert seed.seed == point.seed
    assert policy.root == seed.root


def test_an_executable_point_requires_traced_provenance(tmp_path: Path) -> None:
    point = SweepPoint({"not": "a resolved run"})
    executor = PointExecutor(
        RunStore(tmp_path, freeze_id=FREEZE, seed=EXPERIMENT_SEED),
        codec=RunSpecCodec(),
        operations=UnreachedTraining(),
        infrastructure_attempts=INFRASTRUCTURE_ATTEMPTS,
        failure_policy=FAILURE_POLICY,
    )

    with pytest.raises(PointExecutionError, match=point.point_id):
        executor(point)

    assert not any(tmp_path.iterdir())


def test_the_executor_is_bound_to_the_exact_run_store(tmp_path: Path) -> None:
    store = RunStore(tmp_path, freeze_id=None, seed=EXPERIMENT_SEED)

    assert (
        PointExecutor(
            store,
            codec=RunSpecCodec(),
            operations=UnreachedTraining(),
            infrastructure_attempts=INFRASTRUCTURE_ATTEMPTS,
            failure_policy=FAILURE_POLICY,
        ).store
        is store
    )


HELD_OUT = """evaluators:
  heldout:
    kind: bracketing
    interval: {replicates: 50, level: 0.95}
    punctuation: {source: none, forms: []}
    decoder: minimum-risk
    conventions: {punctuation: keep, unary_chains: keep, trivial_spans: keep,
      minimum_length: 0, aggregation: sentence, coverage: zero, self_arcs: allow,
      gold_derivation: canonical}
"""


def an_evaluated_point(tmp_path: Path) -> SweepPoint:
    document = tmp_path / "run.yaml"
    document.write_text(
        "defaults:\n  - run/synthetic-parser\n" + HELD_OUT, "utf-8", newline="\n"
    )
    composition = trace(
        Selection(
            config=document,
            profile=None,
            overrides=(),
            overlays=(),
            search_path=config_search_path(tmp_path),
        )
    )

    return SweepPoint(
        resolved_document(composition.spec), provenance=composition.provenance
    )


def test_an_executor_outside_a_freeze_scores_the_held_out_rows_unfrozen(
    tmp_path: Path,
) -> None:
    point = an_evaluated_point(tmp_path)
    store = RunStore(tmp_path / "store", freeze_id=None, seed=EXPERIMENT_SEED)
    training = RecordingTraining()
    seed = ExperimentSeed(7).point(point.point_id, point.seed)

    outcome = bound_executor(store, training)(point)
    recorded = read_rows(store.directory(point))

    assert outcome.status is ManifestStatus.COMPLETED
    (scoring,) = training.scorings
    assert scoring is not None
    assert (scoring.freeze_id, scoring.seed) == (UNFROZEN, seed.root)
    assert recorded is not None
    assert recorded.freeze_id is None
    (row,) = recorded.rows
    assert (row["freeze_id"], row["point_id"]) == ("unfrozen", point.point_id)


def test_a_run_declaring_no_evaluator_is_given_no_scoring(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "store", freeze_id=None, seed=EXPERIMENT_SEED)
    training = RecordingTraining()

    bound_executor(store, training)(a_traced_point(0))

    assert training.scorings == [None]


def test_an_executor_bound_to_a_freeze_scores_and_records_the_held_out_rows(
    tmp_path: Path,
) -> None:
    point = an_evaluated_point(tmp_path)
    store = RunStore(tmp_path / "store", freeze_id=FREEZE, seed=EXPERIMENT_SEED)
    training = RecordingTraining()
    seed = ExperimentSeed(7).point(point.point_id, point.seed)

    outcome = bound_executor(store, training)(point)
    recorded = read_rows(store.directory(point))

    assert outcome.status is ManifestStatus.COMPLETED
    (scoring,) = training.scorings
    assert scoring is not None
    assert (scoring.freeze_id, scoring.seed) == ("freeze/1", seed.root)
    assert recorded is not None
    assert recorded.freeze_id == "freeze/1"
    assert recorded.seed_derivation == seed
    (row,) = recorded.rows
    assert (row["freeze_id"], row["point_id"], row["seed"]) == (
        "freeze/1",
        point.point_id,
        seed.root,
    )
    assert (row["budget_depth"], row["budget_beam"], row["residual_measured"]) == (
        5,
        8,
        False,
    )
    artifact = (
        store.directory(point) / ARTIFACT_DIRECTORY / "brackets" / "annotations.json"
    )
    assert artifact.read_bytes() == b'{"kind":"annotations"}\n'


def bound_executor(
    store: RunStore,
    training: RecordingTraining | FailingTraining | TerminatedTraining,
) -> PointExecutor:
    return PointExecutor(
        store,
        codec=RunSpecCodec(),
        operations=training,
        infrastructure_attempts=INFRASTRUCTURE_ATTEMPTS,
        failure_policy=FAILURE_POLICY,
        environment=ENVIRONMENT,
    )


def test_a_point_declaring_no_evaluator_records_no_rows_and_is_not_scored(
    tmp_path: Path,
) -> None:
    point = a_traced_point(0)
    store = RunStore(tmp_path, freeze_id=FREEZE, seed=EXPERIMENT_SEED)
    training = RecordingTraining()

    bound_executor(store, training)(point)
    recorded = read_rows(store.directory(point))

    assert training.scorings == [None]
    assert recorded is not None
    assert recorded.rows == ()


def test_a_point_completed_under_another_freeze_is_executed_again(
    tmp_path: Path,
) -> None:
    point = a_traced_point(0)
    first = RunStore(tmp_path, freeze_id=FREEZE, seed=EXPERIMENT_SEED)
    second = RunStore(tmp_path, freeze_id="freeze/2", seed=EXPERIMENT_SEED)
    training = RecordingTraining()

    bound_executor(first, training)(point)
    cached = bound_executor(first, training)(point)
    again = bound_executor(second, training)(point)
    recorded = read_rows(second.directory(point))
    binding = read_binding(second.directory(point))
    kept = read_rows(first.directory(point))

    assert cached.cached
    assert not again.cached
    assert again.directory == second.directory(point) != first.directory(point)
    assert len(training.roots) == 2
    assert recorded is not None
    assert recorded.freeze_id == "freeze/2"
    assert binding is not None
    assert binding.freeze_id == "freeze/2"
    assert kept is not None
    assert kept.freeze_id == FREEZE
    assert first.is_completed(point)


def test_a_second_experiment_root_over_the_same_store_is_refused(
    tmp_path: Path,
) -> None:
    point = a_traced_point(0)
    store = RunStore(tmp_path, freeze_id=FREEZE, seed=EXPERIMENT_SEED)
    numerical = FailingTraining(NumericalFailureError("the loss", None))
    bound_executor(store, numerical)(point)

    reseeded = RunStore(tmp_path, freeze_id=FREEZE, seed=ExperimentSeed(8))

    with pytest.raises(ReseededRunError):
        bound_executor(reseeded, RecordingTraining())(point)

    assert numerical.attempts == 1


def test_a_failure_recorded_under_another_root_is_refused_before_it_settles(
    tmp_path: Path,
) -> None:
    point = a_traced_point(0)
    store = RunStore(tmp_path, freeze_id=None, seed=EXPERIMENT_SEED)
    bound_executor(store, FailingTraining(NumericalFailureError("the loss", None)))(
        point
    )
    training = RecordingTraining()

    reseeded = RunStore(tmp_path, freeze_id=None, seed=ExperimentSeed(8))

    with pytest.raises(ReseededRunError):
        bound_executor(reseeded, training)(point)

    assert training.roots == []


def failing_executor(store: RunStore, training: FailingTraining) -> PointExecutor:
    return bound_executor(store, training)


def derivation(point: SweepPoint) -> PointSeed:
    return EXPERIMENT_SEED.point(point.point_id, point.seed)


def test_a_numerical_failure_is_recorded_once_and_never_run_again(
    tmp_path: Path,
) -> None:
    point = a_traced_point(0)
    store = RunStore(tmp_path, freeze_id=FREEZE, seed=EXPERIMENT_SEED)
    training = FailingTraining(NumericalFailureError("the loss", None))
    executor = failing_executor(store, training)

    first = executor(point)
    again = executor(point)
    failure = read_failure(store.directory(point))

    assert (first.status, first.cached) == (ManifestStatus.KILLED, False)
    assert (again.status, again.cached) == (ManifestStatus.KILLED, True)
    assert training.attempts == 1
    assert failure is not None
    assert (failure.kind, failure.diagnostic, failure.attempts) == (
        FailureKind.NUMERICAL,
        "the loss is not finite",
        1,
    )
    assert failure.seed_derivation == derivation(point)
    assert read_rows(store.directory(point)) is None


@pytest.mark.parametrize(
    "error",
    [
        MemoryError("planted accelerator OOM"),
        ActivationMemoryExceededError("cuda:0", peak_bytes=9, max_bytes=8),
    ],
    ids=["exhausted", "step-bound"],
)
def test_a_planted_oom_halves_the_declared_batch_before_retrying(
    tmp_path: Path, error: MemoryError
) -> None:
    point = a_traced_point(
        0,
        overrides=(
            "trainer.batch_size=32",
            "trainer.accumulation_steps=1",
        ),
    )
    store = RunStore(tmp_path, freeze_id=FREEZE, seed=EXPERIMENT_SEED)
    training = OutOfMemoryOnce(error)
    executor = PointExecutor(
        store,
        codec=RunSpecCodec(),
        operations=training,
        infrastructure_attempts=INFRASTRUCTURE_ATTEMPTS,
        failure_policy=FailurePolicySpec(
            numerical="final", out_of_memory="halve-batch"
        ),
        environment=ENVIRONMENT,
    )

    outcome = executor(point)
    manifest = store.manifest(point)

    assert outcome.status is ManifestStatus.COMPLETED
    assert training.batches == [32, 16]
    assert manifest is not None
    assert isinstance(manifest.run_spec.trainer, ParserTrainerSpec)
    assert manifest.run_spec.trainer.batch_size == 16
    assert read_failure(store.directory(point)) is None


def test_an_uncontractable_oom_is_not_retried_unchanged(tmp_path: Path) -> None:
    point = a_traced_point(
        0,
        overrides=(
            "trainer.batch_size=1",
            "trainer.accumulation_steps=1",
        ),
    )
    store = RunStore(tmp_path, freeze_id=FREEZE, seed=EXPERIMENT_SEED)
    training = OutOfMemoryOnce(MemoryError("planted accelerator OOM"))
    executor = PointExecutor(
        store,
        codec=RunSpecCodec(),
        operations=training,
        infrastructure_attempts=INFRASTRUCTURE_ATTEMPTS,
        failure_policy=FailurePolicySpec(
            numerical="final", out_of_memory="halve-batch"
        ),
        environment=ENVIRONMENT,
    )

    outcome = executor(point)
    failure = read_failure(store.directory(point))

    assert outcome.status is ManifestStatus.KILLED
    assert training.batches == [1]
    assert failure is not None
    assert (failure.kind, failure.diagnostic, failure.attempts) == (
        FailureKind.INFRASTRUCTURE,
        "planted accelerator OOM",
        1,
    )


def test_a_preparation_failure_is_recorded_once_and_never_run_again(
    tmp_path: Path,
) -> None:
    point = a_traced_point(0)
    store = RunStore(tmp_path, freeze_id=FREEZE, seed=EXPERIMENT_SEED)
    training = FailingTraining(PreparationError("an epoch cannot close its windows"))
    executor = failing_executor(store, training)

    first = executor(point)
    again = executor(point)
    failure = read_failure(store.directory(point))

    assert (first.status, first.cached) == (ManifestStatus.KILLED, False)
    assert (again.status, again.cached) == (ManifestStatus.KILLED, True)
    assert training.attempts == 1
    assert failure is not None
    assert failure.final
    assert (failure.kind, failure.diagnostic, failure.attempts) == (
        FailureKind.PREPARATION,
        "an epoch cannot close its windows",
        1,
    )
    assert failure.seed_derivation == derivation(point)


def test_a_floating_point_error_of_another_kind_is_a_defect_left_unrecorded(
    tmp_path: Path,
) -> None:
    point = a_traced_point(0)
    store = RunStore(tmp_path, freeze_id=None, seed=EXPERIMENT_SEED)
    training = FailingTraining(FloatingPointError("a division by zero"))

    with pytest.raises(FloatingPointError, match="division"):
        failing_executor(store, training)(point)

    assert read_failure(store.directory(point)) is None


def test_a_point_declaring_no_sink_trains_against_the_null_sink(
    tmp_path: Path,
) -> None:
    training = RecordingTraining()

    bound_executor(RunStore(tmp_path, freeze_id=None, seed=EXPERIMENT_SEED), training)(
        a_traced_point(0)
    )

    ((sink,),) = training.sinks
    assert isinstance(sink, NullSink)


class ClosingSink:
    def __init__(self) -> None:
        self.closed = 0

    def log_scalars(self, mapping: Mapping[str, float], step: int) -> None:
        del mapping, step

    def log_table(self, name: str, rows: Sequence[Mapping[str, JsonValue]]) -> None:
        del name, rows

    def close(self) -> None:
        self.closed += 1


def test_a_point_builds_declared_sinks_once_and_closes_them_however_it_ends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[tuple[str, ClosingSink]] = []

    def resolve(
        specs: Mapping[str, SinkSpec], context: SinkContext, /
    ) -> tuple[ClosingSink]:
        del specs
        sink = ClosingSink()
        built.append((context.point_id, sink))

        return (sink,)

    monkeypatch.setattr(executor_module, "resolve_metric_sinks", resolve)
    point = a_traced_point(0)
    training = FailingTraining(TimeoutError("the filesystem stalled"))

    bound_executor(RunStore(tmp_path, freeze_id=None, seed=EXPERIMENT_SEED), training)(
        point
    )

    assert training.attempts == INFRASTRUCTURE_ATTEMPTS
    assert [point_id for point_id, _ in built] == [point.point_id]
    assert all(sink.closed == 1 for _, sink in built)


class TableSink(ClosingSink):
    def __init__(self) -> None:
        super().__init__()
        self.tables: list[tuple[str, tuple[Mapping[str, JsonValue], ...]]] = []

    @override
    def log_table(self, name: str, rows: Sequence[Mapping[str, JsonValue]]) -> None:
        self.tables.append((name, tuple(rows)))


def test_a_completed_point_logs_its_recorded_rows_to_each_declared_sink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sink = TableSink()

    def resolve(
        specs: Mapping[str, SinkSpec], context: SinkContext, /
    ) -> tuple[TableSink]:
        del specs, context
        return (sink,)

    monkeypatch.setattr(executor_module, "resolve_metric_sinks", resolve)
    point = a_traced_point(0)
    store = RunStore(tmp_path, freeze_id=FREEZE, seed=EXPERIMENT_SEED)

    bound_executor(store, RecordingTraining())(point)
    recorded = read_rows(store.directory(point))

    assert recorded is not None
    assert sink.tables == [(HELD_OUT_TABLE, recorded.rows)]


def test_a_cached_point_does_not_build_or_contact_a_declared_sink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[str] = []

    def resolve(
        specs: Mapping[str, SinkSpec], context: SinkContext, /
    ) -> tuple[ClosingSink]:
        del specs
        built.append(context.point_id)

        return (ClosingSink(),)

    point = a_traced_point(0)
    store = RunStore(tmp_path, freeze_id=None, seed=EXPERIMENT_SEED)
    executor = bound_executor(store, RecordingTraining())
    executor(point)
    monkeypatch.setattr(executor_module, "resolve_metric_sinks", resolve)

    outcome = executor(point)

    assert outcome.cached
    assert built == []


@pytest.mark.parametrize(
    "error",
    [
        MemoryError("the allocation is exhausted"),
        ActivationMemoryExceededError("cuda:0", peak_bytes=9, max_bytes=8),
        InterruptedError("the call was interrupted"),
        ConnectionError("the connection was reset"),
        BrokenPipeError("the pipe broke"),
        TimeoutError("the filesystem stalled"),
    ],
    ids=["memory", "step-bound", "interrupt", "connection", "pipe", "timeout"],
)
def test_an_infrastructure_failure_is_attempted_again_until_its_attempts_are_spent(
    tmp_path: Path, error: Exception
) -> None:
    point = a_traced_point(0)
    store = RunStore(tmp_path, freeze_id=FREEZE, seed=EXPERIMENT_SEED)
    training = FailingTraining(error)

    outcome = failing_executor(store, training)(point)
    failure = read_failure(store.directory(point))

    assert outcome.status is ManifestStatus.KILLED
    assert training.attempts == INFRASTRUCTURE_ATTEMPTS
    assert failure is not None
    assert (failure.kind, failure.diagnostic, failure.attempts) == (
        FailureKind.INFRASTRUCTURE,
        str(error),
        INFRASTRUCTURE_ATTEMPTS,
    )
    assert failure.seed_derivation == derivation(point)


def test_a_relaunch_does_not_attempt_an_exhausted_point_again(
    tmp_path: Path,
) -> None:
    point = a_traced_point(0)
    store = RunStore(tmp_path, freeze_id=FREEZE, seed=EXPERIMENT_SEED)
    failing_executor(store, FailingTraining(TimeoutError("stalled")))(point)
    training = RecordingTraining()

    outcome = bound_executor(store, training)(point)
    failure = read_failure(store.directory(point))

    assert (outcome.status, outcome.cached) == (ManifestStatus.KILLED, True)
    assert training.roots == []
    assert store.is_final(point)
    assert failure is not None
    assert (failure.kind, failure.attempts) == (
        FailureKind.INFRASTRUCTURE,
        INFRASTRUCTURE_ATTEMPTS,
    )


def test_attempts_spent_on_an_earlier_launch_count_toward_the_bound(
    tmp_path: Path,
) -> None:
    point = a_traced_point(0)
    store = RunStore(tmp_path, freeze_id=FREEZE, seed=EXPERIMENT_SEED)
    spent = PointFailure(
        FailureKind.INTERRUPTED, "the node failed", 2, derivation(point)
    )
    write_failure(store.directory(point), spent)
    training = FailingTraining(TimeoutError("stalled"))

    outcome = failing_executor(store, training)(point)
    failure = read_failure(store.directory(point))

    assert outcome.status is ManifestStatus.KILLED
    assert training.attempts == INFRASTRUCTURE_ATTEMPTS - 2
    assert failure is not None
    assert (failure.kind, failure.attempts) == (
        FailureKind.INFRASTRUCTURE,
        INFRASTRUCTURE_ATTEMPTS,
    )


def test_an_attempt_is_recorded_before_it_runs(tmp_path: Path) -> None:
    point = a_traced_point(0)
    store = RunStore(tmp_path, freeze_id=FREEZE, seed=EXPERIMENT_SEED)
    seen: list[PointFailure | None] = []

    class Observed(RecordingTraining):
        @override
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
            seen.append(read_failure(directory))
            super().execute(
                blueprint,
                directory,
                streams,
                policy,
                scoring=scoring,
                sinks=sinks,
                recorder=recorder,
            )

    outcome = bound_executor(store, Observed())(point)
    (during,) = seen

    assert outcome.status is ManifestStatus.COMPLETED
    assert during is not None
    assert (during.kind, during.attempts) == (FailureKind.INTERRUPTED, 1)
    assert read_failure(store.directory(point)) is None


def test_an_interruption_left_at_the_bound_is_recorded_infra_failed_unrun(
    tmp_path: Path,
) -> None:
    point = a_traced_point(0)
    store = RunStore(tmp_path, freeze_id=FREEZE, seed=EXPERIMENT_SEED)
    spent = PointFailure(
        FailureKind.INTERRUPTED,
        "the node failed",
        INFRASTRUCTURE_ATTEMPTS,
        derivation(point),
    )
    write_failure(store.directory(point), spent)
    training = RecordingTraining()

    outcome = bound_executor(store, training)(point)
    failure = read_failure(store.directory(point))

    assert outcome.status is ManifestStatus.KILLED
    assert training.roots == []
    assert failure is not None
    assert (failure.kind, failure.diagnostic, failure.attempts) == (
        FailureKind.INFRASTRUCTURE,
        "the node failed",
        INFRASTRUCTURE_ATTEMPTS,
    )


class TerminatedTraining:
    def __init__(self) -> None:
        self.attempts = 0

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
        del blueprint, directory, streams, policy, scoring, sinks, recorder
        self.attempts += 1
        signal.raise_signal(signal.SIGTERM)


def _unhandled(signal_number: int, frame: FrameType | None) -> Never:
    del signal_number, frame
    msg = "SIGTERM reached the handler the executor should have replaced"
    raise AssertionError(msg)


def test_a_terminated_point_is_recorded_interrupted_and_resumed_later(
    tmp_path: Path,
) -> None:
    point = a_traced_point(0)
    store = RunStore(tmp_path, freeze_id=FREEZE, seed=EXPERIMENT_SEED)
    previous = signal.signal(signal.SIGTERM, _unhandled)

    try:
        with pytest.raises(PointTerminatedError, match="SIGTERM"):
            bound_executor(store, TerminatedTraining())(point)

        restored = signal.getsignal(signal.SIGTERM)

    finally:
        signal.signal(signal.SIGTERM, previous)

    manifest = store.manifest(point)
    failure = read_failure(store.directory(point))

    assert restored is _unhandled
    assert manifest is not None
    assert manifest.status is ManifestStatus.KILLED
    assert failure is not None
    assert (failure.kind, failure.attempts) == (FailureKind.INTERRUPTED, 1)
    assert "SIGTERM" in failure.diagnostic
    assert not store.is_final(point)

    training = RecordingTraining()
    resumed = bound_executor(store, training)(point)

    assert resumed.status is ManifestStatus.COMPLETED
    assert len(training.roots) == 1
    assert read_failure(store.directory(point)) is None


def test_a_terminated_attempt_counts_toward_the_bound_on_the_next_launch(
    tmp_path: Path,
) -> None:
    point = a_traced_point(0)
    store = RunStore(tmp_path, freeze_id=FREEZE, seed=EXPERIMENT_SEED)
    previous = signal.signal(signal.SIGTERM, _unhandled)

    try:
        with pytest.raises(PointTerminatedError):
            bound_executor(store, TerminatedTraining())(point)

    finally:
        signal.signal(signal.SIGTERM, previous)

    training = FailingTraining(TimeoutError("stalled"))
    failing_executor(store, training)(point)
    failure = read_failure(store.directory(point))

    assert training.attempts == INFRASTRUCTURE_ATTEMPTS - 1
    assert failure is not None
    assert failure.attempts == INFRASTRUCTURE_ATTEMPTS


@pytest.mark.parametrize(
    "number",
    [
        errno.EIO,
        errno.ESTALE,
        errno.ETIMEDOUT,
        errno.ENETDOWN,
        errno.ENETUNREACH,
        errno.ECONNRESET,
    ],
    ids=["eio", "estale", "etimedout", "enetdown", "enetunreach", "econnreset"],
)
def test_an_os_error_with_an_infrastructure_errno_is_attempted_again(
    tmp_path: Path, number: int
) -> None:
    point = a_traced_point(0)
    store = RunStore(tmp_path, freeze_id=FREEZE, seed=EXPERIMENT_SEED)
    error = OSError(number, os.strerror(number))
    training = FailingTraining(error)

    outcome = failing_executor(store, training)(point)
    failure = read_failure(store.directory(point))

    assert is_infrastructure(error)
    assert outcome.status is ManifestStatus.KILLED
    assert training.attempts == INFRASTRUCTURE_ATTEMPTS
    assert failure is not None
    assert failure.kind is FailureKind.INFRASTRUCTURE


@pytest.mark.parametrize(
    "number",
    [errno.ENOSPC, errno.EACCES, errno.ENOENT, errno.EROFS],
    ids=["enospc", "eacces", "enoent", "erofs"],
)
def test_an_os_error_with_another_errno_propagates_unrecorded(
    tmp_path: Path, number: int
) -> None:
    point = a_traced_point(0)
    store = RunStore(tmp_path, freeze_id=FREEZE, seed=EXPERIMENT_SEED)
    error = OSError(number, os.strerror(number))
    training = FailingTraining(error)

    with pytest.raises(OSError, match=os.strerror(number)):
        failing_executor(store, training)(point)

    assert not is_infrastructure(error)
    assert training.attempts == 1
    assert read_failure(store.directory(point)) is None


def test_a_defect_leaves_the_attempts_an_earlier_launch_recorded(
    tmp_path: Path,
) -> None:
    point = a_traced_point(0)
    store = RunStore(tmp_path, freeze_id=FREEZE, seed=EXPERIMENT_SEED)
    spent = PointFailure(
        FailureKind.INTERRUPTED, "the node failed", 1, derivation(point)
    )
    write_failure(store.directory(point), spent)

    with pytest.raises(KeyError):
        failing_executor(store, FailingTraining(KeyError("a component")))(point)

    assert read_failure(store.directory(point)) == spent


@pytest.mark.parametrize(
    "error",
    [
        KeyError("a missing component"),
        FileNotFoundError("no corpus at /data/ccgbank"),
        PermissionError("the store is read-only"),
    ],
    ids=["defect", "missing-corpus", "bad-path"],
)
def test_a_defect_in_training_propagates_and_records_no_outcome(
    tmp_path: Path, error: Exception
) -> None:
    point = a_traced_point(0)
    store = RunStore(tmp_path, freeze_id=FREEZE, seed=EXPERIMENT_SEED)
    training = FailingTraining(error)

    with pytest.raises(type(error)):
        failing_executor(store, training)(point)

    assert training.attempts == 1
    assert read_failure(store.directory(point)) is None
