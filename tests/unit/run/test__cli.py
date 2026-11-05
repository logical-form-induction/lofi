import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Final, NamedTuple

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import yaml
from tests.unit.run.conftest import (
    EXPERIMENT_SEED,
    a_document,
    a_packaged_run,
    record_manifest,
    write_priced_experiment,
)
from tests.unit.run.test__experiment import (
    NO_FACTORS,
    a_judgment_experiment_document,
    an_experiment_document,
    write_experiment,
    write_factorial_experiment,
)
from tests.unit.spec.acquisition.test_active import (
    a_document as an_active_learning_document,
)

from lofi import _checks
from lofi.data import BatchShape
from lofi.eval import UNFROZEN
from lofi.run import (
    COST_MODEL,
    GRID_COST_MODEL,
    POINTS_NAME,
    RUNS_NAME,
    SWEEPS_NAME,
    ActiveLearningManifest,
    BeadPrediction,
    ExecutionEnvironment,
    ExperimentIdentity,
    ExperimentSeed,
    FailurePolicySpec,
    GridCell,
    GridTiming,
    HardwareDescriptor,
    ManifestStatus,
    RunOutcome,
    RunStore,
    Selection,
    StackTiming,
    SweepPoint,
    TrainingOperations,
    amendment_ledger,
    build_parser,
    compose,
    composer,
    expand_experiment,
    main,
    read_experiment,
    read_price,
    read_rows,
    render_annotated,
    resolved_document,
    sweep_id,
    trace,
    write_points,
)
from lofi.run import _cli as cli_module
from lofi.run import _run_configuration as run_configuration
from lofi.run._configs import packaged_groups, packaged_options
from lofi.run._ports import Published, ReleasePublication
from lofi.run._timing import TrainingStepTimer
from lofi.run.operations import TRAINING_OPERATIONS
from lofi.run.results import SEED_BUDGET_SUFFIX, write_seed_budget
from lofi.spec import (
    ConfigSource,
    LeafOrigin,
    Provenance,
    RunSpecCodec,
    config_search_path,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lofi._types import JsonValue
    from lofi.nn.policy import DevicePolicy
    from lofi.run import (
        Experiment,
        HeldOutScoring,
        MetricSink,
        PointExport,
        PointRecorder,
        PointSeed,
        StepTimer,
        TrainingBlueprint,
    )
    from lofi.run._ports import (
        ActiveLearningRun,
        ExperimentReporting,
        PointPublication,
        SeedBudgeting,
    )
    from lofi.run.probing import PointProbe
    from lofi.spec import RunSpec
    from lofi.train import RestorableRandomStreams


class Workspace(NamedTuple):
    root: Path
    config: Path
    overlay: Path
    profile: str


class RecordingExecutor:
    """Trains a point by recording manifests in the store, like the serial tests."""

    def __init__(self, store: RunStore, *, outcome: ManifestStatus) -> None:
        self._store = store
        self._outcome = outcome
        self.trained: list[SweepPoint] = []

    def __call__(self, point: SweepPoint) -> RunOutcome:
        record_manifest(self._store, point, ManifestStatus.RUNNING)
        self.trained.append(point)
        record_manifest(self._store, point, self._outcome)

        return RunOutcome(self._store.directory(point), self._outcome, cached=False)


class Timing:
    """A step-timer double that answers one timing and records each stack asked."""

    def __init__(self, peak: int | None = None) -> None:
        self.timed: list[RunSpec] = []
        self.grids: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        self._peak = peak

    def time(
        self,
        spec: RunSpec,
        point: PointSeed,
        /,
        *,
        device: str,
        warmup: int,
        repeats: int,
    ) -> StackTiming:
        del point, device, warmup
        self.timed.append(spec)

        return StackTiming(
            steps_per_unit=2,
            positions=list(range(repeats)),
            seconds=[0.5 + position for position in range(repeats)],
            peak_allocated_bytes=self._peak,
            shape=BatchShape(12.0, 4.0),
        )

    def time_grid(
        self,
        stacks: Mapping[tuple[int, int], RunSpec],
        point: PointSeed,
        /,
        *,
        device: str,
        warmup: int,
        repeats: int,
    ) -> GridTiming:
        del point, device, warmup
        lengths = tuple(dict.fromkeys(length for length, _ in stacks))
        batches = tuple(dict.fromkeys(batch for _, batch in stacks))
        self.grids.append((lengths, batches))

        return GridTiming(
            [
                GridCell(
                    length, batch, tuple(length * batch / 1_000 for _ in range(repeats))
                )
                for length, batch in stacks
            ]
        )

    def hardware(self, device: str, /) -> HardwareDescriptor:
        return HardwareDescriptor(
            device_name=device, driver=None, cuda=None, torch="2.8.0", host_cpu="arm"
        )


class Run(NamedTuple):
    status: int
    stdout: str
    stderr: str


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
        msg = "the mocked point executor reached training"
        raise AssertionError(msg)


def run_main(
    argv: Sequence[str],
    *,
    execute: RecordingExecutor | None = None,
    training: TrainingOperations | None = None,
    timer: StepTimer | None = None,
    export: PointExport | None = None,
    bead_predict: BeadPrediction | None = None,
    probe: PointProbe | None = None,
    publish: PointPublication | None = None,
    reports: ExperimentReporting | None = None,
    active_learn: ActiveLearningRun | None = None,
    seed_budget: SeedBudgeting | None = None,
    environ: dict[str, str] | None = None,
) -> Run:
    stdout = io.StringIO()
    stderr = io.StringIO()

    with pytest.MonkeyPatch.context() as patch:
        if execute is not None:
            recording = execute

            def built(store: RunStore, **settings: object) -> RecordingExecutor:
                del store, settings
                return recording

            patch.setattr(cli_module, "PointExecutor", built)
            patch.setattr(run_configuration, "PointExecutor", built)
            training = UnreachedTraining() if training is None else training

        status = main(
            argv,
            training=training,
            timer=timer,
            export=export,
            bead_predict=bead_predict,
            probe=probe,
            publish=publish,
            reports=reports,
            active_learn=active_learn,
            seed_budget=seed_budget,
            environ=environ or {},
            stdout=stdout,
            stderr=stderr,
        )

    return Run(status, stdout.getvalue(), stderr.getvalue())


def test_results_hands_the_experiment_and_root_to_the_report_renderer(
    tmp_path: Path,
) -> None:
    calls: list[tuple[Path, Path, bool]] = []
    output = tmp_path / "results" / "1" / "headline" / "table.tex"

    def reports(experiment: Path, root: Path, /, *, pilot: bool) -> tuple[Path, ...]:
        calls.append((experiment, root, pilot))
        return (output,)

    argv = [
        "results",
        "--experiment",
        "experiments/headline.yaml",
        "--root",
        str(tmp_path),
    ]

    rendered = run_main(argv, reports=reports)
    unwired = run_main(argv)

    assert rendered == Run(0, f"{output}\n", "")
    assert calls == [(tmp_path / "experiments/headline.yaml", tmp_path, False)]
    assert unwired.status == 1
    assert "no report renderer" in unwired.stderr


def test_results_of_a_pilot_are_rendered_from_the_pilot_root(tmp_path: Path) -> None:
    calls: list[tuple[Path, Path, bool]] = []
    output = tmp_path / "pilot" / "results" / "headline" / "table.md"

    def reports(experiment: Path, root: Path, /, *, pilot: bool) -> tuple[Path, ...]:
        calls.append((experiment, root, pilot))
        return (output,)

    rendered = run_main(
        [
            "results",
            "--experiment",
            "experiments/headline.yaml",
            "--root",
            str(tmp_path),
            "--pilot",
        ],
        reports=reports,
    )

    assert rendered == Run(0, f"{output}\n", "")
    assert calls == [(tmp_path / "experiments/headline.yaml", tmp_path, True)]


def write_json(path: Path, document: JsonValue) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8", newline="\n")

    return path


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    document = a_document()
    document["defaults"] = [{"trainer": "long"}]
    del document["trainer"]
    config = write_json(tmp_path / "run.json", document)
    write_json(
        tmp_path / "trainer" / "long.json",
        {"kind": "launcher-unit-trainer", "epochs": 5},
    )
    write_json(tmp_path / "profiles" / "longer.json", {"trainer": {"epochs": 7}})
    overlay = write_json(
        tmp_path / "overlay.json", {"device_policy": {"deterministic": False}}
    )
    return Workspace(tmp_path, config, overlay, "longer")


def selection(
    workspace: Workspace,
    *,
    profile: str | None = None,
    overrides: tuple[str, ...] = (),
    overlays: tuple[Path, ...] = (),
) -> Selection:
    return Selection(
        config=workspace.config,
        profile=profile,
        overrides=overrides,
        overlays=overlays,
        search_path=config_search_path(workspace.root),
    )


def composition_arguments(workspace: Workspace) -> list[str]:
    return [
        "--config",
        str(workspace.config),
        "--profile",
        workspace.profile,
        "--extra",
        str(workspace.overlay),
        "--set",
        "trainer.epochs=3",
    ]


def leaf_lines(rendered: str) -> dict[str, str]:
    lines: dict[str, str] = {}
    trail: list[str] = []

    for line in rendered.splitlines():
        depth = (len(line) - len(line.lstrip(" "))) // 2
        key, _, rest = line.strip().partition(":")
        trail = [*trail[:depth], key]

        if rest:
            lines[".".join(trail)] = rest.strip()

    return lines


def at(document: Mapping[str, JsonValue], *keys: str) -> JsonValue:
    value: JsonValue = dict(document)

    for key in keys:
        assert isinstance(value, dict)
        value = value[key]

    return value


def test_the_parser_reads_the_shared_options_in_order() -> None:
    namespace = build_parser().parse_args(
        [
            "show-config",
            "--config",
            "run.yaml",
            "--profile",
            "fast",
            "--set",
            "a=1",
            "--extra",
            "x.yaml",
            "--set",
            "b=2",
            "--extra",
            "y.yaml",
        ]
    )

    assert namespace.command == "show-config"
    assert namespace.profile == "fast"
    assert namespace.overrides == ["a=1", "b=2"]
    assert [str(path) for path in namespace.overlays] == ["x.yaml", "y.yaml"]


def test_every_option_reaches_the_run_specification(workspace: Workspace) -> None:
    from_group = compose(selection(workspace))
    from_profile = compose(selection(workspace, profile=workspace.profile))
    from_overlay = compose(
        selection(workspace, profile=workspace.profile, overlays=(workspace.overlay,))
    )
    from_override = compose(
        selection(
            workspace,
            profile=workspace.profile,
            overlays=(workspace.overlay,),
            overrides=("trainer.epochs=3",),
        )
    )

    assert at(resolved_document(from_group), "trainer", "epochs") == 5
    assert at(resolved_document(from_profile), "trainer", "epochs") == 7
    assert at(resolved_document(from_overlay), "trainer", "epochs") == 7
    assert (
        at(resolved_document(from_overlay), "device_policy", "deterministic") is False
    )
    assert at(resolved_document(from_override), "trainer", "epochs") == 3
    assert (
        at(resolved_document(from_override), "device_policy", "deterministic") is False
    )


def test_the_options_change_the_point_the_run_command_trains(
    workspace: Workspace,
) -> None:
    store = RunStore(workspace.root, freeze_id=None, seed=EXPERIMENT_SEED)
    executor = RecordingExecutor(store, outcome=ManifestStatus.COMPLETED)

    result = run_main(
        [
            "run",
            *composition_arguments(workspace),
            "--seed",
            "4",
            "--experiment-seed",
            "7",
            "--root",
            str(workspace.root),
        ],
        execute=executor,
    )
    (point,) = executor.trained
    plain = SweepPoint(resolved_document(compose(selection(workspace))))

    assert result.status == 0
    assert point.seed == 4
    assert point.overrides == ("trainer.epochs=3",)
    assert at(point.config, "trainer", "epochs") == 3
    assert at(point.config, "device_policy", "deterministic") is False
    assert point.point_id != plain.point_id
    assert result.stdout == f"{point.point_id} completed {store.directory(point)}\n"


def test_a_completed_point_is_served_from_its_directory(workspace: Workspace) -> None:
    store = RunStore(workspace.root, freeze_id=None, seed=EXPERIMENT_SEED)
    executor = RecordingExecutor(store, outcome=ManifestStatus.COMPLETED)
    argv = ["run", *seeded(workspace)]

    first = run_main(argv, execute=executor)
    second = run_main(argv, execute=executor)

    assert (first.status, second.status) == (0, 0)
    assert len(executor.trained) == 1
    assert first.stdout == second.stdout


def test_a_failed_point_exits_with_status_one(workspace: Workspace) -> None:
    store = RunStore(workspace.root, freeze_id=None, seed=EXPERIMENT_SEED)
    executor = RecordingExecutor(store, outcome=ManifestStatus.KILLED)

    result = run_main(
        ["run", *seeded(workspace)],
        execute=executor,
    )
    (point,) = executor.trained

    assert result.status == 1
    assert result.stdout == f"{point.point_id} failed {store.directory(point)}\n"


def test_explicit_training_operations_build_the_store_bound_executor(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    created: list[RecordingExecutor] = []
    expected_operations = UnreachedTraining()

    def executor(
        store: RunStore,
        *,
        codec: RunSpecCodec,
        operations: TrainingOperations,
        infrastructure_attempts: int,
        failure_policy: FailurePolicySpec,
    ) -> RecordingExecutor:
        del codec
        assert store.seed == ExperimentSeed(7)
        assert infrastructure_attempts == FailurePolicySpec().infrastructure_attempts
        assert failure_policy == FailurePolicySpec()
        assert operations is expected_operations
        assert store.freeze_id is None
        built = RecordingExecutor(store, outcome=ManifestStatus.COMPLETED)
        created.append(built)

        return built

    monkeypatch.setattr(cli_module, "PointExecutor", executor)
    result = run_main(
        ["run", *seeded(workspace)],
        training=expected_operations,
    )

    assert result.status == 0
    assert len(created) == 1
    assert len(created[0].trained) == 1
    assert created[0].trained[0].provenance is not None


def test_a_run_gives_its_point_the_infrastructure_attempts_it_names(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    granted: list[int] = []

    def executor(
        store: RunStore,
        *,
        codec: RunSpecCodec,
        operations: TrainingOperations,
        infrastructure_attempts: int,
        failure_policy: FailurePolicySpec,
    ) -> RecordingExecutor:
        del codec, operations
        granted.append(infrastructure_attempts)
        assert failure_policy == FailurePolicySpec(infrastructure_attempts=5)

        return RecordingExecutor(store, outcome=ManifestStatus.COMPLETED)

    monkeypatch.setattr(cli_module, "PointExecutor", executor)
    result = run_main(
        ["run", *seeded(workspace), "--infrastructure-attempts", "5"],
        training=UnreachedTraining(),
    )

    assert result.status == 0
    assert granted == [5]


@pytest.mark.parametrize("command", ["run", "resume"])
def test_a_development_point_runs_under_the_failure_policy_its_options_declare(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    policies: list[FailurePolicySpec] = []

    def executor(
        store: RunStore,
        *,
        codec: RunSpecCodec,
        operations: TrainingOperations,
        infrastructure_attempts: int,
        failure_policy: FailurePolicySpec,
    ) -> RecordingExecutor:
        del codec, operations, infrastructure_attempts
        policies.append(failure_policy)

        return RecordingExecutor(store, outcome=ManifestStatus.COMPLETED)

    monkeypatch.setattr(cli_module, "PointExecutor", executor)
    point = SweepPoint(resolved_document(compose(selection(workspace))))
    record_manifest(
        RunStore(workspace.root, freeze_id=None, seed=EXPERIMENT_SEED),
        point,
        ManifestStatus.KILLED,
    )
    result = run_main(
        [
            command,
            *seeded(workspace),
            "--numerical",
            "reseed",
            "--out-of-memory",
            "halve-batch",
        ],
        training=UnreachedTraining(),
    )

    assert result.status == 0
    assert policies == [
        FailurePolicySpec(numerical="reseed", out_of_memory="halve-batch")
    ]


def test_a_run_refuses_no_infrastructure_attempt(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    store = RunStore(workspace.root, freeze_id=None, seed=EXPERIMENT_SEED)
    executor = RecordingExecutor(store, outcome=ManifestStatus.COMPLETED)

    with pytest.raises(SystemExit) as refused:
        run_main(
            ["run", *seeded(workspace), "--infrastructure-attempts", "0"],
            execute=executor,
        )

    assert refused.value.code == 2
    assert "--infrastructure-attempts" in capsys.readouterr().err
    assert executor.trained == []


@pytest.mark.parametrize("command", ["run", "resume"])
def test_a_development_command_requires_an_experiment_root(
    workspace: Workspace, command: str, capsys: pytest.CaptureFixture[str]
) -> None:
    store = RunStore(workspace.root, freeze_id=None, seed=EXPERIMENT_SEED)
    executor = RecordingExecutor(store, outcome=ManifestStatus.COMPLETED)

    with pytest.raises(SystemExit) as refused:
        run_main([command, "--config", str(workspace.config)], execute=executor)

    assert refused.value.code == 2
    assert "--experiment-seed" in capsys.readouterr().err
    assert executor.trained == []


def test_a_task_requires_its_experiment_and_takes_no_seed_or_freeze(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    listing = str(workspace.root / POINTS_NAME)

    with pytest.raises(SystemExit) as unnamed:
        run_main(["task", listing, "0", "1"])

    assert unnamed.value.code == 2
    assert "--experiment" in capsys.readouterr().err

    for option, value in (("--experiment-seed", "7"), ("--freeze-id", "freeze/1")):
        with pytest.raises(SystemExit) as refused:
            run_main(
                [
                    "task",
                    listing,
                    "0",
                    "1",
                    "--experiment",
                    "e.yaml",
                    option,
                    value,
                ]
            )

        assert refused.value.code == 2
        assert option in capsys.readouterr().err


def test_no_command_launches_a_sweep_outside_an_experiment(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as refused:
        run_main(["sweep", "--config", str(workspace.config)])

    assert refused.value.code == 2
    assert "invalid choice: 'sweep'" in capsys.readouterr().err


def test_a_run_under_another_experiment_root_is_refused_rather_than_served(
    workspace: Workspace,
) -> None:
    store = RunStore(workspace.root, freeze_id=None, seed=EXPERIMENT_SEED)
    executor = RecordingExecutor(store, outcome=ManifestStatus.COMPLETED)
    argv = ["run", *seeded(workspace)]
    first = run_main(argv, execute=executor)
    (point,) = executor.trained
    argv[argv.index("--experiment-seed") + 1] = "8"

    again = run_main(argv, execute=executor)

    assert first.status == 0
    assert again.status == 2
    assert "seed_derivation" in again.stderr
    assert point.point_id in again.stderr
    assert again.stdout == ""
    assert executor.trained == [point]
    assert RunStore(workspace.root, freeze_id=None, seed=EXPERIMENT_SEED).is_completed(
        point
    )


def test_a_negative_experiment_root_is_refused(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as refused:
        run_main(["run", "--config", str(workspace.config), "--experiment-seed", "-1"])

    assert refused.value.code == 2
    assert "--experiment-seed" in capsys.readouterr().err


def test_a_training_command_requires_training_operations(
    workspace: Workspace,
) -> None:
    result = run_main(["run", *seeded(workspace)])

    assert result.status == 1
    assert "no training operations" in result.stderr
    assert not (workspace.root / "runs").exists()


def test_resume_needs_a_run_directory_and_then_trains_again(
    workspace: Workspace,
) -> None:
    store = RunStore(workspace.root, freeze_id=None, seed=EXPERIMENT_SEED)
    executor = RecordingExecutor(store, outcome=ManifestStatus.COMPLETED)
    argv = ["resume", *seeded(workspace)]
    point = SweepPoint(resolved_document(compose(selection(workspace))))

    nothing = run_main(argv, execute=executor)
    record_manifest(store, point, ManifestStatus.KILLED)
    resumed = run_main(argv, execute=executor)

    assert nothing.status == 1
    assert point.point_id in nothing.stderr
    assert executor.trained == [point]
    assert resumed.status == 0
    assert resumed.stdout == f"{point.point_id} completed {store.directory(point)}\n"


@pytest.mark.parametrize("status", list(ManifestStatus))
def test_resume_refuses_a_run_seeded_under_another_experiment_root(
    workspace: Workspace, status: ManifestStatus
) -> None:
    store = RunStore(workspace.root, freeze_id=None, seed=EXPERIMENT_SEED)
    executor = RecordingExecutor(store, outcome=ManifestStatus.COMPLETED)
    point = SweepPoint(resolved_document(compose(selection(workspace))))
    record_manifest(store, point, status)
    argv = ["resume", *seeded(workspace)]
    argv[argv.index("--experiment-seed") + 1] = "8"

    resumed = run_main(argv, execute=executor)

    assert resumed.status == 2
    assert "seed_derivation" in resumed.stderr
    assert point.point_id in resumed.stderr
    assert executor.trained == []
    assert resumed.stdout == ""


def test_sweep_overrides_apply_after_the_shared_ones(workspace: Workspace) -> None:
    compose_point = composer(selection(workspace, overrides=("trainer.epochs=9",)))

    assert at(compose_point((), 0), "trainer", "epochs") == 9
    assert at(compose_point(("trainer.epochs=2",), 0), "trainer", "epochs") == 2


EXPERIMENT: Final = Path("experiments/headline.yaml")


def tagged(checkout: Path, freeze_id: str) -> str:
    del checkout, freeze_id

    return "0" * 40


def a_checked_out_experiment(
    workspace: Workspace,
    monkeypatch: pytest.MonkeyPatch,
    *,
    freeze_id: str = "freeze/1",
    root: int = 7,
) -> Experiment:
    """Write an experiment into the workspace and execute from it as the checkout."""
    document = an_experiment_document(freeze_id=freeze_id)
    document["seed_policy"] = {"root": root}
    written = write_experiment(workspace.root, document)
    monkeypatch.setattr(cli_module, "package_checkout", lambda: workspace.root)

    return read_experiment(written)


def task_arguments(listing: Path, *more: str) -> list[str]:
    return [
        "task",
        str(listing),
        *more,
        "--experiment",
        str(EXPERIMENT),
    ]


def test_a_task_trains_its_chunk_of_the_listed_points(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    a_checked_out_experiment(workspace, monkeypatch)
    monkeypatch.setattr(cli_module, "require_frozen_revision", tagged)
    store = RunStore(workspace.root, freeze_id="freeze/1", seed=EXPERIMENT_SEED)
    executor = RecordingExecutor(store, outcome=ManifestStatus.COMPLETED)
    compose_point = composer(selection(workspace))
    points = tuple(
        SweepPoint(compose_point((f"trainer.epochs={epochs}",), seed), seed=seed)
        for epochs in (1, 2, 3)
        for seed in (0, 1)
    )
    listing = workspace.root / POINTS_NAME
    write_points(listing, points, identity=ExperimentIdentity("headline", "freeze/1"))

    result = run_main(
        [*task_arguments(listing, "1", "2"), "--root", str(workspace.root)],
        execute=executor,
    )

    assert result.status == 0
    assert executor.trained == list(points[2:4])
    assert len(result.stdout.splitlines()) == 2


def test_a_task_seeds_its_points_from_the_root_its_experiment_declares(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = a_checked_out_experiment(workspace, monkeypatch, root=11)
    monkeypatch.setattr(cli_module, "require_frozen_revision", tagged)
    point = SweepPoint(composer(selection(workspace))((), 3), seed=3)
    listing = workspace.root / POINTS_NAME
    write_points(listing, (point,), identity=ExperimentIdentity("headline", "freeze/1"))
    stores: list[RunStore] = []

    def executor(
        store: RunStore,
        *,
        codec: RunSpecCodec,
        operations: TrainingOperations,
        infrastructure_attempts: int,
        failure_policy: FailurePolicySpec,
        environment: ExecutionEnvironment,
    ) -> RecordingExecutor:
        del codec, operations, environment
        assert infrastructure_attempts == experiment.failure.infrastructure_attempts
        assert failure_policy == experiment.failure
        stores.append(store)

        return RecordingExecutor(store, outcome=ManifestStatus.COMPLETED)

    monkeypatch.setattr(run_configuration, "PointExecutor", executor)

    result = run_main(
        [*task_arguments(listing, "0", "1"), "--root", str(workspace.root)],
        training=UnreachedTraining(),
    )
    (store,) = stores

    assert result.status == 0
    assert store.seed == experiment.seed_policy == ExperimentSeed(11)
    assert store.freeze_id == experiment.freeze_id
    assert store.derivation(point) == experiment.seed_policy.point(
        point.point_id, point.seed
    )


def seeded(workspace: Workspace) -> list[str]:
    return [
        "--config",
        str(workspace.config),
        "--experiment-seed",
        "7",
        "--root",
        str(workspace.root),
    ]


def price_arguments(experiment: Path, *, repeats: int = 2) -> list[str]:
    return [
        "price",
        "--experiment",
        str(experiment),
        "--device",
        "cpu",
        "--warmup",
        "0",
        "--repeats",
        str(repeats),
        "--root",
        str(experiment.parents[1]),
    ]


def a_learning_rate_experiment(root: Path) -> Path:
    return write_priced_experiment(
        root,
        a_packaged_run("synthetic-parser"),
        {
            "grid": [{"path": "optimizer.learning_rate", "values": [0.001, 0.0005]}],
            "seeds": [0, 1, 2],
            "expected_points": 6,
        },
    )


def test_price_times_the_experiments_own_stacks_and_writes_where_its_sweep_lives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_checks, "CHECKS", False)
    monkeypatch.chdir(tmp_path)
    experiment = a_learning_rate_experiment(tmp_path)

    result = run_main(price_arguments(experiment), timer=TrainingStepTimer())
    points = expand_experiment(read_experiment(experiment), tmp_path)
    directory = tmp_path / SWEEPS_NAME / sweep_id(points)
    price = read_price(directory)
    resamples = read_experiment(experiment).pricing.resamples
    lines = result.stdout.splitlines()

    assert result.status == 0, result.stderr
    assert price is not None
    assert price.sweep_id == sweep_id(points)
    assert price.points == len(points)
    assert len(price.stacks) == 2
    assert all(len(stack.seconds) == 2 for stack in price.stacks)
    assert lines[0].startswith(f"stack {price.stacks[0].stack} points=3 length=2 ")
    assert "peak unmeasured" in lines[0]
    assert price.hardware.torch == torch.__version__
    assert price.hardware.device_name == "cpu"
    assert lines[2:] == [
        f"device: cpu (cpu, torch {torch.__version__}, host {price.hardware.host_cpu})",
        f"points: {len(points)}",
        f"cost model: {COST_MODEL}",
        f"resamples: {resamples} (seed {EXPERIMENT_SEED.root})",
        f"p50 hours: {price.p50_hours:.6f}",
        f"p90 hours: {price.p90_hours:.6f}",
        "declared: unpriced",
        f"written: {directory / 'price.json'}",
    ]
    assert result.stderr == ""


def test_price_reports_the_estimate_a_priced_experiment_already_declares(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_checks, "CHECKS", False)
    monkeypatch.chdir(tmp_path)
    experiment = a_learning_rate_experiment(tmp_path)
    document = json.loads(experiment.read_text(encoding="utf-8"))
    document["compute"] = {
        "kind": "estimate",
        "p50_hours": 1.5,
        "p90_hours": 2.5,
        "cost_model": "v0",
    }
    experiment.write_text(json.dumps(document), encoding="utf-8", newline="\n")

    result = run_main(price_arguments(experiment), timer=Timing())

    assert result.status == 0
    assert "declared: p50 1.5 p90 2.5 (v0)" in result.stdout.splitlines()


@pytest.mark.usefixtures("open_ended_trainer")
def test_price_refuses_an_experiment_whose_trainer_declares_no_length(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_checks, "CHECKS", False)
    monkeypatch.chdir(tmp_path)
    run = a_packaged_run("synthetic-parser")
    run["trainer"] = {"kind": "open-ended-unit-trainer", "patience": 2}
    experiment = write_priced_experiment(
        tmp_path, run, {"grid": [], "seeds": [0, 1, 2], "expected_points": 3}
    )
    timer = Timing()

    result = run_main(price_arguments(experiment), timer=timer)

    assert result.status == 2
    assert "training length is undeclared" in result.stderr
    assert timer.timed == []
    assert not (tmp_path / SWEEPS_NAME).exists()


def test_price_refuses_to_time_a_step_under_the_debug_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_checks, "CHECKS", True)
    monkeypatch.chdir(tmp_path)
    timer = Timing()

    result = run_main(
        price_arguments(a_learning_rate_experiment(tmp_path)), timer=timer
    )

    assert result.status == 1
    assert "LOFI_CHECK" in result.stderr
    assert timer.timed == []


def test_price_refuses_a_single_timed_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_checks, "CHECKS", False)
    monkeypatch.chdir(tmp_path)
    timer = Timing()

    result = run_main(
        price_arguments(a_learning_rate_experiment(tmp_path), repeats=1), timer=timer
    )

    assert result.status == 2
    assert "at least two timed runs" in result.stderr
    assert timer.timed == []


def test_price_without_a_step_timer_prices_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    result = run_main(price_arguments(a_learning_rate_experiment(tmp_path)))

    assert result.status == 1
    assert "no step timer" in result.stderr


def test_price_refuses_an_experiment_document_its_loader_refuses(
    tmp_path: Path,
) -> None:
    experiment = a_learning_rate_experiment(tmp_path)
    document = json.loads(experiment.read_text(encoding="utf-8"))
    document["paper"] = "B"
    experiment.write_text(json.dumps(document), encoding="utf-8", newline="\n")

    result = run_main(price_arguments(experiment), timer=Timing())

    assert result.status == 2
    assert "paper" in result.stderr


def test_price_refuses_a_judgment_fit_experiment_naming_why(tmp_path: Path) -> None:
    experiment = write_experiment(
        tmp_path, a_judgment_experiment_document(), name="fits"
    )
    timer = Timing()

    result = run_main(price_arguments(experiment), timer=timer)

    assert result.status == 2
    assert "sweeps a judgment-fit document" in result.stderr
    assert timer.timed == []


@pytest.mark.parametrize("option", ["--device", "--warmup", "--repeats"])
def test_price_declares_its_device_and_counts_rather_than_defaulting_them(
    tmp_path: Path, option: str
) -> None:
    argv = price_arguments(tmp_path / "experiments" / "pricing.yaml")
    position = argv.index(option)
    del argv[position : position + 2]

    with pytest.raises(SystemExit):
        build_parser().parse_args(argv)


def test_show_config_annotates_every_leaf_with_its_source(
    workspace: Workspace,
) -> None:
    result = run_main(["show-config", *composition_arguments(workspace)])
    lines = leaf_lines(result.stdout)
    provenance = trace(
        selection(
            workspace,
            profile=workspace.profile,
            overlays=(workspace.overlay,),
            overrides=("trainer.epochs=3",),
        )
    ).provenance

    assert result.status == 0
    assert set(lines) == set(provenance.paths)
    assert lines["trainer.epochs"] == "3  # override trainer.epochs=3"
    assert lines["trainer.kind"] == (
        f'"launcher-unit-trainer"  # group {workspace.root / "trainer" / "long.json"}'
    )
    assert lines["device_policy.deterministic"] == "false  # extra"
    assert lines["device_policy.device"] == f'"cpu"  # file {workspace.config}'
    assert lines["scheduler"] == "null  # default"
    assert lines["probes"] == "{}  # default"


def test_show_config_reports_the_profile_where_it_sets_a_leaf(
    workspace: Workspace,
) -> None:
    result = run_main(
        [
            "show-config",
            "--config",
            str(workspace.config),
            "--profile",
            workspace.profile,
        ]
    )

    assert leaf_lines(result.stdout)["trainer.epochs"] == "7  # profile longer"


def test_rendering_names_the_expression_a_leaf_was_resolved_from() -> None:
    provenance = Provenance(
        origins={
            "a.b": LeafOrigin(
                source=ConfigSource.FILE, layer="run.yaml", expression="${a.c}"
            ),
            "a.c": LeafOrigin(source=ConfigSource.FILE, layer="run.yaml"),
            "d[0]": LeafOrigin(source=ConfigSource.EXTRA),
            "d[1]": LeafOrigin(source=ConfigSource.OVERRIDE, layer='d+=["x"]'),
        }
    )

    rendered = render_annotated({"a": {"b": 2, "c": 2}, "d": [1, "x"]}, provenance)

    assert rendered == (
        "a:\n"
        "  b: 2  # file run.yaml, from ${a.c}\n"
        "  c: 2  # file run.yaml\n"
        "d[0]: 1  # extra\n"
        'd[1]: "x"  # override d+=["x"]\n'
    )


def test_a_malformed_override_is_refused_naming_its_path(workspace: Workspace) -> None:
    result = run_main(
        ["show-config", "--config", str(workspace.config), "--set", "trainer.epochs=x"]
    )

    assert result.status == 2
    assert result.stderr.startswith("lofi show-config: trainer.epochs")
    assert result.stdout == ""


def test_a_missing_document_is_refused(workspace: Workspace) -> None:
    result = run_main(["show-config", "--config", str(workspace.root / "absent.json")])

    assert result.status == 2
    assert "absent.json" in result.stderr


def test_a_missing_overlay_is_refused(workspace: Workspace) -> None:
    result = run_main(
        [
            "show-config",
            "--config",
            str(workspace.config),
            "--extra",
            str(workspace.root / "absent.json"),
        ]
    )

    assert result.status == 2
    assert "absent.json" in result.stderr


def test_a_command_is_required() -> None:
    with pytest.raises(SystemExit) as exited:
        main([], stdout=io.StringIO(), stderr=io.StringIO())

    assert exited.value.code == 2


def test_importing_the_command_line_performs_no_side_effect(tmp_path: Path) -> None:
    source = (
        "import json, logging, os, sys\n"
        "handlers = list(logging.root.handlers)\n"
        "level = logging.root.level\n"
        "import lofi.run\n"
        "import lofi.run._cli\n"
        "print(json.dumps({\n"
        "    'files': sorted(os.listdir()),\n"
        "    'handlers_added': len(logging.root.handlers) - len(handlers),\n"
        "    'level_changed': logging.root.level != level,\n"
        "    'torch': 'torch' in sys.modules,\n"
        "}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", source],
        check=True,
        capture_output=True,
        encoding="utf-8",
        cwd=tmp_path,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    observed = json.loads(completed.stdout)

    assert observed == {
        "files": [],
        "handlers_added": 0,
        "level_changed": False,
        "torch": False,
    }


def test_a_stack_over_its_own_device_budget_is_priced_and_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_checks, "CHECKS", False)
    experiment = write_priced_experiment(
        tmp_path,
        a_packaged_run("synthetic-parser"),
        {
            "grid": [
                {
                    "path": "device_policy.activation_memory_bytes",
                    "values": [1_000, 4_000],
                }
            ],
            "seeds": [0, 1, 2],
            "expected_points": 6,
        },
    )

    result = run_main(price_arguments(experiment), timer=Timing(peak=2_000))
    (directory,) = (tmp_path / SWEEPS_NAME).iterdir()
    price = read_price(directory)

    assert result.status == 0
    assert "held 2000 bytes of the 1000 its device policy allows" in result.stderr
    assert "of the 4000" not in result.stderr
    assert "3 points are not submitted to an array" in result.stderr
    assert price is not None
    assert [stack.over_budget for stack in price.stacks] == [True, False]


def test_price_under_the_length_batch_grid_its_document_declares(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_checks, "CHECKS", False)
    monkeypatch.chdir(tmp_path)
    experiment = a_learning_rate_experiment(tmp_path)
    document = json.loads(experiment.read_text(encoding="utf-8"))
    document["pricing"] = {"kind": "length-batch", "lengths": [5, 6], "batches": [1, 2]}
    experiment.write_text(json.dumps(document), encoding="utf-8", newline="\n")

    result = run_main(price_arguments(experiment), timer=TrainingStepTimer())
    points = expand_experiment(read_experiment(experiment), tmp_path)
    price = read_price(tmp_path / SWEEPS_NAME / sweep_id(points))

    assert result.status == 0, result.stderr
    assert price is not None
    assert price.cost_model == GRID_COST_MODEL
    assert f"cost model: {GRID_COST_MODEL}" in result.stdout.splitlines()
    assert price.grid is not None
    assert [(cell.sentence_length, cell.batch_size) for cell in price.grid.cells] == [
        (5, 1),
        (5, 2),
        (6, 1),
        (6, 2),
    ]
    assert all(len(cell.seconds) == 2 for cell in price.grid.cells)
    assert all(stack.timing.shape is not None for stack in price.stacks)
    assert 0 < price.p50_hours <= price.p90_hours


def test_price_reads_and_expands_the_experiment_under_the_root_it_writes_to(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_checks, "CHECKS", False)
    root = tmp_path / "repository"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    run = a_packaged_run("synthetic-parser")
    optimizer = run.pop("optimizer")
    run["defaults"] = [{"optimizer": "project"}]
    group = root / "configs" / "optimizer"
    group.mkdir(parents=True)
    (group / "project.yaml").write_text(
        yaml.safe_dump(optimizer), encoding="utf-8", newline="\n"
    )
    experiment = write_priced_experiment(
        root,
        run,
        {
            "grid": [{"path": "optimizer.learning_rate", "values": [0.001, 0.0005]}],
            "seeds": [0, 1, 2],
            "expected_points": 6,
        },
    )
    argv = price_arguments(experiment)
    argv[argv.index("--experiment") + 1] = str(experiment.relative_to(root))

    result = run_main(argv, timer=Timing())
    points = expand_experiment(read_experiment(experiment), root)

    assert result.status == 0, result.stderr
    assert read_price(root / SWEEPS_NAME / sweep_id(points)) is not None
    assert not (elsewhere / SWEEPS_NAME).exists()


def test_the_configs_command_lists_every_packaged_group_with_its_options() -> None:
    result = run_main(["configs"])
    listed = {
        line.partition(":")[0]: line.partition(":")[2].split()
        for line in result.stdout.splitlines()
    }

    assert result.status == 0
    assert set(listed) == set(packaged_groups())
    assert all(listed[slot] == list(packaged_options(slot)) for slot in listed)


_HELD_OUT: dict[str, JsonValue] = {
    "heldout": {
        "kind": "bracketing",
        "interval": {"replicates": 50, "level": 0.95},
        "conventions": {
            "punctuation": "keep",
            "unary_chains": "keep",
            "trivial_spans": "keep",
            "minimum_length": 0,
            "aggregation": "sentence",
            "coverage": "zero",
            "self_arcs": "allow",
            "gold_derivation": "canonical",
        },
    }
}


def test_a_development_run_writes_its_held_out_rows_locally_marked_unfrozen(
    tmp_path: Path,
) -> None:
    config = tmp_path / "dev-run.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "defaults": ["run/synthetic-parser"],
                "data": {
                    "datasets": {
                        "train": {"dataset": {"retained": {"dev": 0.25, "test": 0.25}}}
                    }
                },
                "trainer": {"accumulation_steps": 1},
                "evaluators": {
                    "lexicon": {
                        "kind": "lexical-types",
                        "interval": {"replicates": 50, "level": 0.95},
                        "readout": "category-terms",
                        "alignment": {
                            "kind": "identity",
                            "fit_on": "dev",
                            "score_on": "test",
                            "random_floor": {"replicates": 50, "level": 0.95},
                            "stability": {"replicates": 50, "level": 0.95},
                        },
                        "conventions": {
                            "punctuation": "keep",
                            "unary_chains": "keep",
                            "trivial_spans": "keep",
                            "minimum_length": 0,
                            "aggregation": "sentence",
                            "coverage": "zero",
                            "self_arcs": "allow",
                            "gold_derivation": "canonical",
                        },
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
        newline="\n",
    )

    result = run_main(
        [
            "run",
            "--config",
            str(config),
            "--experiment-seed",
            "5",
            "--root",
            str(tmp_path),
        ],
        training=TRAINING_OPERATIONS,
    )
    (line,) = result.stdout.splitlines()
    point_id, status, directory = line.split()
    recorded = read_rows(Path(directory))

    assert result.status == 0, result.stderr
    assert status == "completed"
    assert Path(directory) == tmp_path / RUNS_NAME / point_id
    assert recorded is not None
    assert recorded.freeze_id is None
    assert recorded.rows
    assert {row["freeze_id"] for row in recorded.rows} == {UNFROZEN}
    assert {row["task"] for row in recorded.rows} == {"lexicon"}
    assert {row["metric"] for row in recorded.rows} == {
        "lexical-type-f1",
        "primitive-identity-accuracy",
        "primitive-random-relabeling-floor",
        "constructor-identity-accuracy",
        "constructor-random-relabeling-floor",
    }


def test_a_task_runs_only_from_a_checkout_of_its_experiments_freeze_tag(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    a_checked_out_experiment(workspace, monkeypatch, freeze_id="freeze/404")
    store = RunStore(workspace.root, freeze_id=None, seed=EXPERIMENT_SEED)
    executor = RecordingExecutor(store, outcome=ManifestStatus.COMPLETED)
    listing = workspace.root / POINTS_NAME
    write_points(listing, (), identity=ExperimentIdentity("headline", "freeze/404"))

    result = run_main(
        [*task_arguments(listing, "0", "1"), "--root", str(workspace.root)],
        execute=executor,
    )

    assert result.status == 1
    assert "has no tag freeze/404" in result.stderr
    assert executor.trained == []


def test_a_task_checks_the_executing_checkout_rather_than_its_directory(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    git = shutil.which("git") or "git"
    written = write_experiment(workspace.root, an_experiment_document())

    for arguments in (
        ("init", "--quiet", "--initial-branch", "main"),
        ("config", "user.name", "Task Test"),
        ("config", "user.email", "task@example.org"),
        ("config", "commit.gpgSign", "false"),
        ("add", "--all"),
        ("commit", "--quiet", "--message", "Gate commit"),
        ("tag", "freeze/1"),
    ):
        subprocess.run(
            (git, "-C", str(workspace.root), *arguments),
            check=True,
            capture_output=True,
        )

    monkeypatch.chdir(workspace.root)
    executor = RecordingExecutor(
        RunStore(workspace.root, freeze_id=None, seed=EXPERIMENT_SEED),
        outcome=ManifestStatus.COMPLETED,
    )
    listing = workspace.root / "listed" / POINTS_NAME
    write_points(listing, (), identity=ExperimentIdentity("headline", "freeze/1"))

    result = run_main(
        [
            "task",
            str(listing),
            "0",
            "1",
            "--experiment",
            str(written),
        ],
        execute=executor,
    )

    assert result.status == 1
    assert "executing checkout" in result.stderr


@pytest.mark.parametrize(
    "identity",
    [
        ExperimentIdentity("headline", "freeze/2"),
        ExperimentIdentity("analysis", "freeze/1"),
    ],
    ids=["another-freeze", "another-experiment"],
)
def test_a_task_refuses_points_listed_for_anything_but_its_experiment(
    workspace: Workspace,
    monkeypatch: pytest.MonkeyPatch,
    identity: ExperimentIdentity,
) -> None:
    a_checked_out_experiment(workspace, monkeypatch)
    monkeypatch.setattr(cli_module, "require_frozen_revision", tagged)
    executor = RecordingExecutor(
        RunStore(workspace.root, freeze_id="freeze/1", seed=EXPERIMENT_SEED),
        outcome=ManifestStatus.COMPLETED,
    )
    listing = workspace.root / POINTS_NAME
    write_points(
        listing,
        (SweepPoint(composer(selection(workspace))((), 0), seed=0),),
        identity=identity,
    )

    result = run_main(
        [*task_arguments(listing, "0", "1"), "--root", str(workspace.root)],
        execute=executor,
    )

    assert result.status == 1
    assert "headline" in result.stderr
    assert executor.trained == []


def test_a_task_meeting_a_point_completed_under_another_freeze_trains_it_again(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    a_checked_out_experiment(workspace, monkeypatch, freeze_id="freeze/2")
    monkeypatch.setattr(cli_module, "require_frozen_revision", tagged)
    point = SweepPoint(composer(selection(workspace))((), 0), seed=0)
    earlier = RunStore(workspace.root, freeze_id="freeze/1", seed=EXPERIMENT_SEED)
    unscoped = RunStore(workspace.root, freeze_id=None, seed=EXPERIMENT_SEED)

    for completed in (earlier, unscoped):
        record_manifest(completed, point, ManifestStatus.COMPLETED)

    store = RunStore(workspace.root, freeze_id="freeze/2", seed=EXPERIMENT_SEED)
    executor = RecordingExecutor(store, outcome=ManifestStatus.COMPLETED)
    listing = workspace.root / POINTS_NAME
    write_points(listing, (point,), identity=ExperimentIdentity("headline", "freeze/2"))

    result = run_main(
        [*task_arguments(listing, "0", "1"), "--root", str(workspace.root)],
        execute=executor,
    )

    assert result.status == 0
    assert executor.trained == [point]
    assert result.stdout.split() == [
        point.point_id,
        "completed",
        str(store.directory(point)),
    ]
    assert earlier.is_completed(point)
    assert unscoped.is_completed(point)


def test_export_hands_the_run_release_and_destination_to_its_exporter(
    tmp_path: Path,
) -> None:
    calls: list[tuple[Path, Path, Path]] = []

    def export(run: Path, release: Path, destination: Path, /) -> None:
        calls.append((run, release, destination))

    argv = [
        "export",
        "--run",
        str(tmp_path / "runs" / "point"),
        "--release",
        str(tmp_path / "release.yaml"),
        "--out",
        str(tmp_path / "bundle"),
    ]

    exported = run_main(argv, export=export)
    unwired = run_main(argv)

    assert exported.status == 0
    assert exported.stdout == f"{tmp_path / 'bundle'}\n"
    assert calls == [
        (tmp_path / "runs" / "point", tmp_path / "release.yaml", tmp_path / "bundle")
    ]
    assert unwired.status == 1
    assert "no exporter" in unwired.stderr


def test_export_reports_a_refused_bundle_with_status_two(tmp_path: Path) -> None:
    def refuse(run: Path, release: Path, destination: Path, /) -> None:
        del release, destination
        msg = f"the run directory {run} records no completed run"
        raise ValueError(msg)

    result = run_main(
        [
            "export",
            "--run",
            str(tmp_path),
            "--release",
            str(tmp_path / "release.yaml"),
            "--out",
            str(tmp_path / "bundle"),
        ],
        export=refuse,
    )

    assert result.status == 2
    assert result.stderr == (
        f"lofi export: the run directory {tmp_path} records no completed run\n"
    )
    assert result.stdout == ""


def test_bead_predict_hands_its_four_paths_to_the_prediction_operation(
    tmp_path: Path,
) -> None:
    calls: list[tuple[Path, Path, Path, Path]] = []

    def predict(
        run: Path,
        serving: Path,
        items: Path,
        destination: Path,
        /,
    ) -> int:
        calls.append((run, serving, items, destination))
        return 3

    run = tmp_path / "runs" / "point"
    serving = tmp_path / "serving.json"
    items = tmp_path / "items.jsonl"
    destination = tmp_path / "predictions.jsonl"
    argv = [
        "bead-predict",
        "--run",
        str(run),
        "--serving",
        str(serving),
        "--items",
        str(items),
        "--out",
        str(destination),
    ]

    predicted = run_main(argv, bead_predict=predict)
    unwired = run_main(argv)

    assert predicted.status == 0
    assert predicted.stdout == f"3 {destination}\n"
    assert calls == [(run, serving, items, destination)]
    assert unwired.status == 1
    assert "no bead prediction operation" in unwired.stderr


def test_bead_predict_reports_refused_input_with_status_two(tmp_path: Path) -> None:
    def refuse(
        run: Path,
        serving: Path,
        items: Path,
        destination: Path,
        /,
    ) -> int:
        del serving, items, destination
        msg = f"the run directory {run} records no completed run"
        raise ValueError(msg)

    result = run_main(
        [
            "bead-predict",
            "--run",
            str(tmp_path),
            "--serving",
            str(tmp_path / "serving.json"),
            "--items",
            str(tmp_path / "items.jsonl"),
            "--out",
            str(tmp_path / "predictions.jsonl"),
        ],
        bead_predict=refuse,
    )

    assert result.status == 2
    assert result.stderr == (
        f"lofi bead-predict: the run directory {tmp_path} records no completed run\n"
    )
    assert result.stdout == ""


def test_publish_hands_its_four_paths_to_the_publication_and_reports_each_role(
    tmp_path: Path,
) -> None:
    calls: list[tuple[Path, Path, Path, Path]] = []
    out = tmp_path / "records"

    def publish(
        run: Path, tokenizations: Path, selection: Path, destination: Path, /
    ) -> tuple[Published, ...]:
        calls.append((run, tokenizations, selection, destination))
        return (
            Published("decoded", destination / "decoded.jsonl", 3, 1),
            Published("viterbi", destination / "viterbi.jsonl", 4, 0),
        )

    run = tmp_path / "runs" / "point"
    argv = [
        "publish",
        "--run",
        str(run),
        "--tokenizations",
        str(tmp_path / "tokenizations.yaml"),
        "--selection",
        str(tmp_path / "selection.yaml"),
        "--out",
        str(out),
    ]

    published = run_main(argv, publish=publish)
    unwired = run_main(argv)

    assert published.status == 0
    assert published.stdout == (
        f"decoded: 3 records, 1 unlabeled, {out / 'decoded.jsonl'}\n"
        f"viterbi: 4 records, 0 unlabeled, {out / 'viterbi.jsonl'}\n"
    )
    assert calls == [
        (run, tmp_path / "tokenizations.yaml", tmp_path / "selection.yaml", out)
    ]
    assert unwired.status == 1
    assert "no publication" in unwired.stderr


def test_publish_reports_a_refused_publication_with_status_two(
    tmp_path: Path,
) -> None:
    def refuse(
        run: Path, tokenizations: Path, selection: Path, destination: Path, /
    ) -> tuple[Published, ...]:
        del tokenizations, selection, destination
        msg = f"the point at {run} is a pilot's"
        raise ValueError(msg)

    result = run_main(
        [
            "publish",
            "--run",
            str(tmp_path),
            "--tokenizations",
            str(tmp_path / "tokenizations.yaml"),
            "--selection",
            str(tmp_path / "selection.yaml"),
            "--out",
            str(tmp_path / "records"),
        ],
        publish=refuse,
    )

    assert result.status == 2
    assert result.stderr == f"lofi publish: the point at {tmp_path} is a pilot's\n"
    assert result.stdout == ""


def _active_manifest(**changed: JsonValue) -> ActiveLearningManifest:
    document: dict[str, JsonValue] = {
        "document": json.loads(an_active_learning_document()),
        "status": "completed",
        "rounds": 2,
        "awaiting": None,
        "stop_reason": "max-rounds",
        "initial_judgments": 24,
        "initial_skipped": 0,
        "source_digests": {},
        "seed": 3,
        "seed_derivation": None,
        "package_versions": {},
        "git_revision": "0" * 40,
    }
    document.update(changed)
    return ActiveLearningManifest.model_validate_json(json.dumps(document))


def test_active_learn_hands_its_document_seed_and_directory_to_the_run(
    tmp_path: Path,
) -> None:
    calls: list[tuple[Path, int, Path]] = []

    def learn(document: Path, seed: int, out: Path, /) -> ActiveLearningManifest:
        calls.append((document, seed, out))
        return _active_manifest()

    document = tmp_path / "active.yaml"
    out = tmp_path / "out"
    argv = ["active-learn", str(document), "--out", str(out), "--seed", "3"]

    learned = run_main(argv, active_learn=learn)
    unwired = run_main(argv)

    assert learned.status == 0
    assert learned.stdout == f"completed 2 {out}\n"
    assert calls == [(document, 3, out)]
    assert unwired.status == 1
    assert "no active-learning operation" in unwired.stderr


def test_active_learn_reports_the_batch_a_stopped_run_awaits(tmp_path: Path) -> None:
    def learn(document: Path, seed: int, out: Path, /) -> ActiveLearningManifest:
        del document, seed, out
        return _active_manifest(
            status="awaiting-responses", rounds=1, awaiting=2, stop_reason=None
        )

    out = tmp_path / "out"
    result = run_main(
        ["active-learn", str(tmp_path / "active.yaml"), "--out", str(out)],
        active_learn=learn,
    )

    assert result.status == 0
    assert result.stdout == (
        f"awaiting-responses 2 {out / 'rounds' / '2' / 'batch.jsonl'}\n"
    )


def test_active_learn_reports_refused_input_with_status_two(tmp_path: Path) -> None:
    def refuse(document: Path, seed: int, out: Path, /) -> ActiveLearningManifest:
        del seed, out
        msg = f"the judgment-fit document of {document} declares no comparison"
        raise ValueError(msg)

    document = tmp_path / "active.yaml"
    result = run_main(
        ["active-learn", str(document), "--out", str(tmp_path / "out")],
        active_learn=refuse,
    )

    assert result.status == 2
    assert result.stderr == (
        f"lofi active-learn: the judgment-fit document of {document} declares no "
        "comparison\n"
    )


def test_probe_composes_over_the_runs_recorded_configuration(tmp_path: Path) -> None:
    calls: list[tuple[Path, Selection, Path]] = []
    run = tmp_path / "runs" / "point"
    overlay = tmp_path / "probes.yaml"

    def probe(run: Path, selection: Selection, destination: Path, /) -> int:
        calls.append((run, selection, destination))
        return 7

    argv = [
        "probe",
        "--run",
        str(run),
        "--out",
        str(tmp_path / "rows.json"),
        "--extra",
        str(overlay),
        "--set",
        "probes.diag.kind=diagnostics",
    ]

    probed = run_main(argv, probe=probe)
    unwired = run_main(argv)

    assert probed.status == 0
    assert probed.stdout == f"7 {tmp_path / 'rows.json'}\n"
    assert len(calls) == 1
    given, selection, destination = calls[0]

    assert given == run
    assert destination == tmp_path / "rows.json"
    assert selection.config == run / "config.resolved.json"
    assert selection.profile is None
    assert selection.overrides == ("probes.diag.kind=diagnostics",)
    assert selection.overlays == (overlay,)
    assert unwired.status == 1
    assert "no probe operation" in unwired.stderr


def test_probe_reports_a_refused_run_with_status_two(tmp_path: Path) -> None:
    def refuse(run: Path, selection: Selection, destination: Path, /) -> int:
        del selection, destination
        msg = f"the run directory {run} records no completed run"
        raise ValueError(msg)

    result = run_main(
        ["probe", "--run", str(tmp_path), "--out", str(tmp_path / "rows.json")],
        probe=refuse,
    )

    assert result.status == 2
    assert result.stderr == (
        f"lofi probe: the run directory {tmp_path} records no completed run\n"
    )
    assert result.stdout == ""


def test_probe_requires_a_run_and_a_destination(tmp_path: Path) -> None:
    for argv in (["probe", "--out", str(tmp_path)], ["probe", "--run", str(tmp_path)]):
        with pytest.raises(SystemExit):
            build_parser().parse_args(argv)


def test_contract_applies_a_declared_rung_and_records_it_in_the_ledger(
    tmp_path: Path,
) -> None:
    path = write_factorial_experiment(
        tmp_path,
        {"a": [0, 1]},
        factors={},
        preregistration=NO_FACTORS,
        rungs=[{"kind": "seeds", "floor": 3}],
        seeds=(0, 1, 2, 3),
    )
    declared = sweep_id(expand_experiment(read_experiment(path), tmp_path))

    contracted = run_main(
        [
            "contract",
            "--experiment",
            "experiments/headline.yaml",
            "--rung",
            "seeds",
            "--root",
            str(tmp_path),
        ]
    )
    amended = read_experiment(path)
    points = expand_experiment(amended, tmp_path)

    assert contracted.status == 0, contracted.stderr
    assert contracted.stdout.splitlines() == [
        "rung: seeds",
        "dropped: [3]",
        f"sweep: {declared} -> {sweep_id(points)}",
        "points: 6",
        f"ledger: {amendment_ledger(path)}",
    ]
    assert amended.sweep.seeds == (0, 1, 2)


def test_contract_refuses_a_rung_the_ladder_does_not_declare(tmp_path: Path) -> None:
    write_experiment(tmp_path, an_experiment_document())

    refused = run_main(
        [
            "contract",
            "--experiment",
            "experiments/headline.yaml",
            "--rung",
            "resolution-iv",
            "--root",
            str(tmp_path),
        ]
    )

    assert refused.status == 2
    assert "declares no rung 'resolution-iv'" in refused.stderr


@pytest.mark.parametrize(
    "command",
    [
        ["launch", "--experiment", "e.yaml", "--pilot", "--out", "launch.json"],
        ["status", "--experiment", "e.yaml", "--interval", "0", "--polls", "1"],
    ],
)
def test_a_launch_or_poll_without_launching_launches_nothing(
    command: list[str],
) -> None:
    result = run_main(command)

    assert result.status == 1
    assert "no launching" in result.stderr


def test_a_launch_reads_its_experiment_under_the_overrides_it_is_given() -> None:
    namespace = build_parser().parse_args(
        [
            "launch",
            "--experiment",
            "e.yaml",
            "--freeze",
            "FREEZE.json",
            "--set",
            "launch.kind=array",
            "--set",
            "launch.chunk_size=8",
            "--out",
            "launch.json",
        ]
    )

    assert namespace.overrides == ["launch.kind=array", "launch.chunk_size=8"]


def a_pilot(root: Path, scores: list[float]) -> Path:
    pilot = root / "results" / "A" / "0" / "pilot"
    pilot.mkdir(parents=True)
    rows: list[dict[str, JsonValue]] = [
        {"metric": "recovery", "value": value, "seed": seed, "config_hash": "cell"}
        for seed, value in enumerate(scores)
    ]
    pq.write_table(pa.Table.from_pylist(rows), pilot / "rows.parquet")

    return pilot


def test_seed_budget_writes_the_priced_seed_count_beside_the_seed_floor(
    tmp_path: Path,
) -> None:
    experiment = write_experiment(tmp_path, an_experiment_document())
    pilot = a_pilot(tmp_path, [0.40, 0.80, 0.55, 0.70, 0.45])

    result = run_main(
        [
            "seed-budget",
            "--experiment",
            str(experiment),
            "--pilot",
            str(pilot),
            "--root",
            str(tmp_path),
        ],
        seed_budget=write_seed_budget,
    )
    written = experiment.with_name(f"{experiment.stem}{SEED_BUDGET_SUFFIX}")
    record = json.loads(written.read_text(encoding="utf-8"))

    assert result.status == 0
    assert json.loads(result.stdout) == record
    assert record["seed_floor"] == 3
    assert record["seeds_required"] == record["cells"]["cell"]
    assert record["seeds_required"] > 3
    assert (record["target"], record["precision"], record["level"]) == (
        "mean",
        0.05,
        0.95,
    )


@pytest.mark.parametrize("command", ["freeze", "seed-budget", "release"])
def test_a_release_command_without_its_operation_does_nothing(command: str) -> None:
    arguments = {
        "freeze": ["--number", "1", "--readiness", "r.txt"],
        "seed-budget": ["--experiment", "e.yaml", "--pilot", "pilot"],
        "release": ["--bundle", "b", "--repo-id", "o/n", "--run", "r"],
    }

    result = run_main([command, *arguments[command]])

    assert result.status == 1
    assert "was given no" in result.stderr


def test_a_publication_names_its_evidence_reviewer_and_license(
    tmp_path: Path,
) -> None:
    def release(
        bundle: Path,
        repo_id: str,
        run: Path,
        /,
        *,
        publication: ReleasePublication | None,
        environ: Mapping[str, str],
    ) -> Mapping[str, JsonValue]:
        del bundle, repo_id, run, publication, environ
        msg = "a partial publication reached the release"
        raise AssertionError(msg)

    stdout = io.StringIO()
    stderr = io.StringIO()
    status = main(
        [
            "release",
            "--bundle",
            str(tmp_path),
            "--repo-id",
            "owner/name",
            "--run",
            str(tmp_path),
            "--publish",
            "--reviewer",
            "A Reviewer",
        ],
        release=release,
        stdout=stdout,
        stderr=stderr,
    )

    assert status == 2
    assert "--evidence, --authorized-license is required" in stderr.getvalue()
