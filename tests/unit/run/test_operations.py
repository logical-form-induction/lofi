from __future__ import annotations

import socket
import sqlite3
import weakref
from json import loads
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
import torch

from lofi.nn.policy import DeviceKind
from lofi.run import (
    CHECKPOINTS_NAME,
    ExecutionEnvironment,
    ExperimentSeed,
    FailureKind,
    FailurePolicySpec,
    HeldOutScoring,
    ManifestStatus,
    MetricsRecorder,
    PointExecutor,
    PointReportSink,
    PreparationError,
    RandomStreams,
    RunStore,
    SeedPolicy,
    Selection,
    SerialLauncher,
    SweepPoint,
    installed_packages,
    operations,
    plan_training,
    read_failure,
    read_metrics,
    resolve_device_policy,
    resolved_document,
    trace,
)
from lofi.spec import (
    CompositionSpec,
    ConfigDocumentError,
    RunSpec,
    RunSpecCodec,
    config_search_path,
)
from lofi.spec.grammar import declare_grammar
from lofi.train import ActivationMemoryExceededError, DirectoryStore, StepMemory

INFRASTRUCTURE_ATTEMPTS = 3
FAILURE_POLICY = FailurePolicySpec(numerical="final", out_of_memory="retry")

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lofi._types import JsonValue, ReportValue
    from lofi.data.synthetic.spec.corpus import SyntheticCorpusDecl
    from lofi.nn.policy import DevicePolicy
    from lofi.run import RunPhase, TrainingBlueprint
    from lofi.train import (
        CheckpointManifest,
        CheckpointStore,
        FittingReport,
        MetricLog,
        PreparedTraining,
        RestorableRandomStreams,
        ResumePosition,
        StateTree,
    )


CONFIG = Path(__file__).resolve().parents[3] / "lofi/configs/run/synthetic-parser.yaml"


CONVENTIONS: dict[str, JsonValue] = {
    "punctuation": "keep",
    "unary_chains": "keep",
    "trivial_spans": "keep",
    "minimum_length": 0,
    "aggregation": "sentence",
    "coverage": "zero",
    "self_arcs": "allow",
    "gold_derivation": "canonical",
}


def a_blueprint(*, evaluated: bool) -> TrainingBlueprint:
    document = loads(
        RunSpecCodec(search_path=config_search_path(CONFIG.parents[3]))(
            CONFIG
        ).model_dump_json()
    )

    if evaluated:
        document["evaluators"] = {
            "brackets": {
                "kind": "bracketing",
                "interval": {"replicates": 50, "level": 0.95},
                "conventions": CONVENTIONS,
                "punctuation": {"source": "none", "forms": []},
                "decoder": "minimum-risk",
            }
        }

    return plan_training(RunSpec.model_validate(document))


def train(
    blueprint: TrainingBlueprint, directory: Path, scoring: HeldOutScoring | None
) -> None:
    streams = RandomStreams(SeedPolicy(3, samplers=blueprint.sampler_streams))
    streams.apply()
    operations.TRAINING_OPERATIONS.execute(
        blueprint,
        directory,
        streams,
        resolve_device_policy(blueprint.run_spec),
        scoring=scoring,
        sinks=(),
        recorder=MetricsRecorder(directory, SweepPoint(loads(blueprint.configuration))),
    )


def test_production_operations_score_the_held_out_evaluators_they_are_given(
    tmp_path: Path,
) -> None:
    sink = PointReportSink("point")

    train(a_blueprint(evaluated=True), tmp_path, HeldOutScoring("freeze/1", 3, sink))

    (row,) = sink.rows
    assert (row["task"], row["freeze_id"], row["seed"]) == ("brackets", "freeze/1", 3)
    assert DirectoryStore(tmp_path / CHECKPOINTS_NAME).list()


def test_production_operations_given_no_scoring_score_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unscored(*arguments: object, **keywords: object) -> None:
        del arguments, keywords
        msg = "production operations scored a run given no scoring"
        raise AssertionError(msg)

    monkeypatch.setattr(operations, "evaluate_held_out", unscored)

    train(a_blueprint(evaluated=False), tmp_path, None)

    assert DirectoryStore(tmp_path / CHECKPOINTS_NAME).list()


def test_a_fragment_appending_one_composition_rule_to_the_application_rules_trains(
    tmp_path: Path,
) -> None:
    group = tmp_path / "configs/data/datasets/train/dataset/grammar/rules"
    group.mkdir(parents=True)
    (group / "forward-composition.yaml").write_text(
        "defaults:\n"
        "  - data/datasets/train/dataset/grammar/rules/application\n"
        "name: forward-composition\n"
        "rules+:\n"
        "  - {rule: composition, direction: unrestricted, order: 1, view: /}\n",
        encoding="utf-8",
        newline="\n",
    )
    document = tmp_path / "run.yaml"
    document.write_text(
        "defaults:\n"
        "  - run/synthetic-parser\n"
        "  - data.datasets.train.dataset.grammar.rules: forward-composition\n"
        "model:\n"
        "  spaces:\n"
        "    categories:\n"
        "      combinators:\n"
        "        decoders:\n"
        '          "composition:/:1:harmonic:unrestricted": application\n',
        encoding="utf-8",
        newline="\n",
    )
    run_spec = trace(
        Selection(
            config=document,
            profile=None,
            overrides=(),
            overlays=(),
            search_path=config_search_path(tmp_path),
        )
    ).spec
    packaged = cast(
        "SyntheticCorpusDecl",
        RunSpecCodec(search_path=config_search_path(CONFIG.parents[3]))(CONFIG)
        .data.datasets["train"]
        .dataset,
    )
    declared = cast("SyntheticCorpusDecl", run_spec.data.datasets["train"].dataset)
    blueprint = plan_training(run_spec)
    rules = declare_grammar(declared.grammar).rules
    packaged_rules = declare_grammar(packaged.grammar).rules

    train(blueprint, tmp_path / "run", None)

    assert rules.rules[:-1] == packaged_rules.rules
    assert rules.rules[-1] == CompositionSpec(
        direction="unrestricted", order=1, view="/"
    )
    assert "composition:/:1:harmonic:unrestricted" in {
        action.name for action in blueprint.actions.actions
    }
    assert DirectoryStore(tmp_path / "run" / CHECKPOINTS_NAME).list()


def test_production_operations_record_the_point_as_it_runs(tmp_path: Path) -> None:
    sink = PointReportSink("point")

    train(a_blueprint(evaluated=True), tmp_path, HeldOutScoring("freeze/1", 3, sink))
    record = read_metrics(tmp_path)

    assert record is not None
    assert record.stage == "closed"
    assert set(record.phases) == {"preparation", "recovery", "fitting", "scoring"}
    assert all(seconds >= 0.0 for seconds in record.phases.values())
    assert record.peak_memory_bytes is None
    assert record.fitting is not None
    assert record.fitting.epochs
    assert record.point.parameter_count > 0
    assert "grounding" not in record.contributed


class _Recording:
    """Keeps what production operations tell a recorder, in order."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.contributed: dict[str, Mapping[str, ReportValue]] = {}
        self.peak: list[int | None] = []

    def prepared(
        self, blueprint: TrainingBlueprint, prepared: PreparedTraining, /
    ) -> None:
        del blueprint, prepared
        self.calls.append("prepared")

    def timed(self, phase: RunPhase, seconds: float, /) -> None:
        del seconds
        self.calls.append(phase.value)

    def fitted(self, report: FittingReport, start: ResumePosition, /) -> None:
        del report
        self.calls.append(f"fitted from {start.step}")

    def contribute(self, name: str, report: Mapping[str, ReportValue], /) -> None:
        self.calls.append(f"contributed {name}")
        self.contributed[name] = report

    def closed(self, *, peak_memory_bytes: int | None) -> None:
        self.calls.append("closed")
        self.peak.append(peak_memory_bytes)


def test_a_component_that_reports_on_itself_is_contributed_under_its_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepare = operations.train.prepare_training

    def reporting(
        blueprint: TrainingBlueprint,
        streams: RestorableRandomStreams,
        policy: DevicePolicy,
    ) -> PreparedTraining:
        prepared = prepare(blueprint, streams, policy)
        parser = prepared.model.modules["parser"]
        object.__setattr__(parser, "report", lambda: {"cells": 3})

        return prepared

    monkeypatch.setattr(operations.train, "prepare_training", reporting)
    blueprint = a_blueprint(evaluated=False)
    streams = RandomStreams(SeedPolicy(3, samplers=blueprint.sampler_streams))
    streams.apply()
    recording = _Recording()
    operations.TRAINING_OPERATIONS.execute(
        blueprint,
        tmp_path,
        streams,
        resolve_device_policy(blueprint.run_spec),
        scoring=None,
        sinks=(),
        recorder=recording,
    )

    assert recording.calls == [
        "preparation",
        "prepared",
        "recovery",
        "fitted from 0",
        "fitting",
        "contributed token_encoder",
        "contributed parser",
        "closed",
    ]
    assert recording.contributed == {
        "token_encoder": {
            "static_vectors/rows_read": 8.0,
            "static_vectors/rows_matched": 8.0,
            "static_vectors/rows_dropped": 0.0,
        },
        "parser": {"cells": 3},
    }
    assert recording.peak == [None]


def test_the_recorder_is_closed_with_the_peak_the_device_held_after_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recording = _Recording()

    def reset(memory: StepMemory) -> None:
        del memory
        recording.calls.append("reset")

    def peak(memory: StepMemory) -> int:
        del memory
        recording.calls.append("peak")

        return 4096

    monkeypatch.setattr(StepMemory, "reset", reset)
    monkeypatch.setattr(StepMemory, "high_water", peak)
    blueprint = a_blueprint(evaluated=False)
    streams = RandomStreams(SeedPolicy(3, samplers=blueprint.sampler_streams))
    streams.apply()
    operations.TRAINING_OPERATIONS.execute(
        blueprint,
        tmp_path,
        streams,
        resolve_device_policy(blueprint.run_spec),
        scoring=None,
        sinks=(),
        recorder=recording,
    )

    assert recording.calls.index("reset") < recording.calls.index("preparation")
    assert recording.calls[-2:] == ["peak", "closed"]
    assert recording.peak == [4096]


def test_an_accelerator_out_of_memory_is_reported_as_an_infrastructure_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def exhausted(*arguments: object, **keywords: object) -> None:
        del arguments, keywords
        msg = "CUDA out of memory. Tried to allocate 2.00 GiB"
        raise torch.OutOfMemoryError(msg)

    monkeypatch.setattr(operations.train, "fit_training", exhausted)

    with pytest.raises(MemoryError, match="Tried to allocate") as raised:
        train(a_blueprint(evaluated=False), tmp_path, None)

    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


def test_a_step_above_its_bound_is_raised_by_name_without_its_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    held: list[weakref.ref[torch.nn.Module]] = []

    def bounded(
        prepared: PreparedTraining, *arguments: object, **keywords: object
    ) -> None:
        del arguments, keywords
        held.extend(weakref.ref(module) for module in prepared.model.modules.values())
        device = "cuda:0"
        raise ActivationMemoryExceededError(device, peak_bytes=9, max_bytes=8)

    monkeypatch.setattr(operations.train, "fit_training", bounded)

    with pytest.raises(ActivationMemoryExceededError) as raised:
        train(a_blueprint(evaluated=False), tmp_path, None)

    assert (raised.value.device, raised.value.peak_bytes, raised.value.max_bytes) == (
        "cuda:0",
        9,
        8,
    )
    assert raised.value.__context__ is None
    assert held
    assert all(reference() is None for reference in held)


def test_fitting_runs_under_the_declared_step_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bounds: list[int | None] = []
    fit = operations.train.fit_training

    def metered(
        prepared: PreparedTraining,
        store: CheckpointStore[StateTree],
        manifest: CheckpointManifest,
        streams: RestorableRandomStreams,
        position: ResumePosition,
        /,
        *,
        sinks: Sequence[MetricLog],
    ) -> FittingReport:
        memory = prepared.optimization.memory
        bounds.append(None if memory is None else memory.max_bytes)

        return fit(prepared, store, manifest, streams, position, sinks=sinks)

    monkeypatch.setattr(operations.train, "fit_training", metered)
    blueprint = a_blueprint(evaluated=False)
    train(blueprint, tmp_path, None)

    assert bounds == [blueprint.run_spec.device_policy.activation_memory_bytes]


def test_an_out_of_memory_attempt_releases_its_training_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    held: list[weakref.ref[torch.nn.Module]] = []

    def exhausted(
        prepared: PreparedTraining, *arguments: object, **keywords: object
    ) -> None:
        del arguments, keywords
        held.extend(weakref.ref(module) for module in prepared.model.modules.values())
        msg = "CUDA out of memory"
        raise torch.OutOfMemoryError(msg)

    monkeypatch.setattr(operations.train, "fit_training", exhausted)

    with pytest.raises(MemoryError):
        train(a_blueprint(evaluated=False), tmp_path, None)

    assert held
    assert all(reference() is None for reference in held)


@pytest.mark.parametrize(
    ("kind", "module"), [(DeviceKind.CUDA, "cuda"), (DeviceKind.MPS, "mps")]
)
def test_an_out_of_memory_attempt_empties_its_device_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: DeviceKind, module: str
) -> None:
    emptied: list[str] = []

    class Policy:
        @property
        def kind(self) -> DeviceKind:
            return kind

    def exhausted(*arguments: object) -> None:
        del arguments
        msg = "out of memory"
        raise torch.OutOfMemoryError(msg)

    monkeypatch.setattr(operations.train, "prepare_training", exhausted)
    monkeypatch.setattr(
        getattr(torch, module), "empty_cache", lambda: emptied.append(module)
    )
    blueprint = a_blueprint(evaluated=False)
    streams = RandomStreams(SeedPolicy(3, samplers=blueprint.sampler_streams))

    with pytest.raises(MemoryError):
        operations.TRAINING_OPERATIONS.execute(
            blueprint,
            tmp_path,
            streams,
            cast("DevicePolicy", Policy()),
            scoring=None,
            sinks=(),
            recorder=MetricsRecorder(
                tmp_path, SweepPoint(loads(blueprint.configuration))
            ),
        )

    assert emptied == [module]


def test_an_out_of_memory_point_is_attempted_again_and_recorded_as_infra_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts: list[int] = []

    def exhausted(*arguments: object, **keywords: object) -> None:
        del arguments, keywords
        attempts.append(1)
        msg = "CUDA out of memory"
        raise torch.OutOfMemoryError(msg)

    monkeypatch.setattr(operations.train, "fit_training", exhausted)
    composition = trace(
        Selection(
            config=CONFIG,
            profile=None,
            overrides=(),
            overlays=(),
            search_path=config_search_path(tmp_path),
        )
    )
    point = SweepPoint(
        resolved_document(composition.spec), provenance=composition.provenance
    )
    store = RunStore(tmp_path / "store", freeze_id=None, seed=ExperimentSeed(7))

    outcome = PointExecutor(
        store,
        codec=RunSpecCodec(),
        operations=operations.TRAINING_OPERATIONS,
        infrastructure_attempts=INFRASTRUCTURE_ATTEMPTS,
        failure_policy=FAILURE_POLICY,
        environment=ExecutionEnvironment(installed_packages(), "0123abcd", "cpu"),
    )(point)

    assert outcome.status is ManifestStatus.KILLED
    assert len(attempts) == INFRASTRUCTURE_ATTEMPTS
    failure = read_failure(store.directory(point))
    assert failure is not None
    assert (failure.kind, failure.diagnostic, failure.attempts) == (
        FailureKind.INFRASTRUCTURE,
        "CUDA out of memory",
        INFRASTRUCTURE_ATTEMPTS,
    )


def offline(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Refuse and record every socket connection the test's process attempts."""
    attempts: list[object] = []

    def refuse(self: socket.socket, address: object) -> None:
        del self
        attempts.append(address)
        msg = f"a test contacted {address!r}"
        raise OSError(msg)

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setenv("MLFLOW_DISABLE_TELEMETRY", "true")
    monkeypatch.setenv("DO_NOT_TRACK", "1")

    return attempts


def mlflow_overlay(directory: Path, store: Path) -> Path:
    """Write an overlay declaring an MLflow sink over a local database file."""
    overlay = directory / "tracking.yaml"
    overlay.write_text(
        "sinks:\n"
        "  tracking:\n"
        "    kind: mlflow\n"
        "    experiment_name: lofi\n"
        f"    tracking_uri: sqlite:///{store}\n",
        encoding="utf-8",
        newline="\n",
    )

    return overlay


def histories(store: Path) -> dict[str, dict[str, list[tuple[int, float]]]]:
    """Return every metric history of an MLflow database file, by run name.

    Each history lists the step and value of every logged entry, by step.
    """
    found: dict[str, dict[str, list[tuple[int, float]]]] = {}

    if not store.exists():
        return found

    with sqlite3.connect(store) as database:
        logged = database.execute(
            "SELECT runs.name, metrics.key, metrics.step, metrics.value "
            "FROM metrics JOIN runs ON runs.run_uuid = metrics.run_uuid "
            "ORDER BY runs.name, metrics.key, metrics.step"
        ).fetchall()

    for run, key, step, value in logged:
        found.setdefault(run, {}).setdefault(key, []).append((step, value))

    return found


def assert_one_entry_per_optimizer_step(
    logged: Mapping[str, list[tuple[int, float]]],
) -> None:
    """Assert every training and validation history has one entry per step."""
    ((steps, completed),) = logged["fitting/optimizer_steps"]
    every_step = list(range(1, int(completed) + 1))

    assert steps == completed > 0
    assert {
        "loss/coherence/controller_entropy",
        "loss/grammaticality/range",
    } <= set(logged)
    assert "validation/loss/grammaticality" in logged

    for name, entries in logged.items():
        if not name.startswith("fitting/"):
            assert [step for step, _ in entries] == every_step, name


def test_the_experiment_executor_logs_each_point_to_its_own_mlflow_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    attempts = offline(monkeypatch)
    tracking = tmp_path / "mlflow.db"
    points = tuple(
        traced_point(
            (f"optimizer.learning_rate={rate}",),
            overlays=(mlflow_overlay(tmp_path, tracking),),
        )
        for rate in (0.001, 0.002)
    )
    runs = RunStore(tmp_path / "runs", freeze_id=None, seed=ExperimentSeed(5))
    execute = PointExecutor(
        runs,
        codec=RunSpecCodec(),
        operations=operations.TRAINING_OPERATIONS,
        infrastructure_attempts=1,
        failure_policy=FAILURE_POLICY,
    )
    handles = SerialLauncher(runs, execute=execute).submit(points)
    logged = histories(tracking)

    with sqlite3.connect(tracking) as database:
        artifacts = database.execute("SELECT artifact_uri FROM runs").fetchall()

    assert artifacts
    assert all(
        (Path.from_uri(uri) if uri.startswith("file:") else Path(uri)).is_relative_to(
            tmp_path
        )
        for (uri,) in artifacts
    )
    assert len(handles) == 2
    assert sorted(logged) == sorted(point.point_id for point in points)
    assert all(runs.is_completed(point) for point in points)

    for point in logged.values():
        assert_one_entry_per_optimizer_step(point)

    assert attempts == []


def test_a_refused_preparation_is_reported_as_a_preparation_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refused(*arguments: object) -> None:
        del arguments
        msg = "an epoch has 3 microbatches, which cannot close 2-unit windows"
        raise ValueError(msg)

    monkeypatch.setattr(operations.train, "prepare_training", refused)

    with pytest.raises(PreparationError, match="3 microbatches") as raised:
        train(a_blueprint(evaluated=False), tmp_path, None)

    assert isinstance(raised.value.__cause__, ValueError)


def traced_point(
    overrides: tuple[str, ...], *, overlays: tuple[Path, ...] = ()
) -> SweepPoint:
    """Return one provenance-bearing development point."""
    composition = trace(
        Selection(
            config=CONFIG,
            profile=None,
            overrides=overrides,
            overlays=overlays,
            search_path=config_search_path(CONFIG.parents[3]),
        )
    )

    return SweepPoint(
        resolved_document(composition.spec), provenance=composition.provenance
    )


def test_a_diverging_point_is_recorded_and_the_sweep_completes_the_next(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runs"
    points = tuple(
        traced_point(
            (
                f"optimizer.learning_rate={rate}",
                (
                    'loss.terms=[{"objective": {"kind": "grammaticality", '
                    '"readout": "sentences"}}]'
                ),
            )
        )
        for rate in (1.0e30, 1.0e-3)
    )
    store = RunStore(root, freeze_id=None, seed=ExperimentSeed(5))
    execute = PointExecutor(
        store,
        codec=RunSpecCodec(),
        operations=operations.TRAINING_OPERATIONS,
        infrastructure_attempts=1,
        failure_policy=FAILURE_POLICY,
    )
    handles = SerialLauncher(store, execute=execute).submit(points)
    diverging, converging = points
    first = store.directory(diverging)
    second = store.directory(converging)
    failure = read_failure(first)

    assert len(handles) == 2
    assert failure is not None
    assert (failure.kind, failure.attempts, failure.final) == (
        FailureKind.NUMERICAL,
        1,
        True,
    )
    assert "is not finite" in failure.diagnostic
    assert read_failure(Path(second)) is None
    assert loads((Path(second) / "manifest.json").read_text(encoding="utf-8"))[
        "status"
    ] == (ManifestStatus.COMPLETED.value)
    assert first.name == diverging.point_id
    assert second.name == converging.point_id


def test_composition_refuses_a_size_whose_epoch_leaves_a_window_open() -> None:
    with pytest.raises(ConfigDocumentError) as refused:
        traced_point(("data.datasets.train.dataset.size=8",))

    assert "trainer.accumulation_steps" in str(refused.value)
    assert "3 microbatches" in str(refused.value)
