import io
import json
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import pyarrow.parquet as pq
import pytest
from tests.support.installed import install
from tests.unit.run.conftest import (
    EXPERIMENT_SEED,
    a_price,
    a_profile,
    record_manifest,
)
from tests.unit.run.test__experiment import (
    a_judgment_experiment_document,
    an_experiment_document,
    write_experiment,
)
from tests.unit.run.test_active import an_active_learning_experiment, write_inputs

from lofi.eval import UNFROZEN, BaselineMeasurement, BaselineResult
from lofi.run import (
    MANIFEST_NAME,
    POINTS_NAME,
    RUNS_NAME,
    SCRIPT_NAME,
    SITE_PROFILE_VARIABLE,
    SWEEPS_NAME,
    AdmissionError,
    ArrayJob,
    ArtifactClass,
    BaselineFitSpec,
    ExecutionEnvironment,
    ExperimentIdentity,
    ExperimentSeed,
    ExperimentStore,
    FailureKind,
    GridSweepExpander,
    LaunchContext,
    Launcher,
    LauncherPlugin,
    LaunchSpec,
    ManifestStatus,
    PointBinding,
    PointExecutionError,
    PointFailure,
    QuotaExceededError,
    RecordedRows,
    ReseededRunError,
    RunHandle,
    RunOutcome,
    RunStatus,
    RunStore,
    SerialLauncher,
    SerialLaunchSpec,
    SiteAllocation,
    SiteProfile,
    SiteQuota,
    Sweep,
    SweepPoint,
    TrainingOperations,
    UnfrozenRevisionError,
    build_parser,
    expand_experiment,
    read_experiment,
    read_points,
    subprocess_output,
    subprocess_runner,
    sweep_id,
    write_failure,
    write_price,
    write_rows,
)
from lofi.run import _baseline_fit_configuration as baseline_fit_configuration
from lofi.run import _run_configuration as run_configuration
from lofi.run import launching as launching_module
from lofi.run import main as cli_main
from lofi.run._boundary.launch import LaunchReport, LaunchRequest
from lofi.run.launching import (
    ExperimentLaunching,
    UnlaunchableExperimentError,
    UnlaunchedPointError,
    UnpricedLaunchError,
    launch,
    plan_launch,
    task_program,
)
from lofi.run.operations import TRAINING_OPERATIONS
from lofi.run.results import (
    MANIFESTS_NAME,
    ROWS_NAME,
    STATUS_NAME,
    FreezeMismatchError,
    FreezeRecordError,
    pilot_directory,
)
from lofi.spec.baselines import BiskHockenmaierBaselineSpec

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from lofi._types import JsonValue
    from lofi.run import CommandRunner, FailurePolicySpec
    from lofi.spec import RunSpecCodec


def main(
    argv: Sequence[str],
    *,
    run: CommandRunner | None = None,
    environ: Mapping[str, str] | None = None,
    stdout: io.StringIO | None = None,
    checkout: Path | None = None,
    quota_report: Callable[[Sequence[str]], str] | None = None,
    training: TrainingOperations = TRAINING_OPERATIONS,
) -> int:
    """Launch as ``lofi launch`` does, raising what the command reports."""
    namespace = build_parser().parse_args(["launch", *argv])
    launching = ExperimentLaunching(
        run=subprocess_runner if run is None else run,
        quota_report=subprocess_output if quota_report is None else quota_report,
        checkout=checkout,
    )
    out = io.StringIO() if stdout is None else stdout

    if namespace.pilot:
        return launching.pilot(
            namespace.experiment,
            root=namespace.root,
            overrides=tuple(namespace.overrides),
            training=training,
            out=namespace.out,
            stdout=out,
        )

    return launching.launch(
        namespace.experiment,
        freeze=namespace.freeze,
        root=namespace.root,
        overrides=tuple(namespace.overrides),
        dry_run=namespace.dry_run,
        companions=tuple(namespace.companions),
        training=training,
        environ={} if environ is None else environ,
        out=namespace.out,
        stdout=out,
    )


class Workspace(NamedTuple):
    root: Path
    experiment: Path
    profile: SiteProfile
    freeze: Path


FREEZE_ID = "freeze/1"


def bind(
    store: RunStore, point: SweepPoint, *, seed: ExperimentSeed = EXPERIMENT_SEED
) -> None:
    """Bind a point's directory to the launcher tests' experiment."""
    RunStore(store.root, freeze_id=store.freeze_id, seed=seed).claim(point)


def complete(store: RunStore, point: SweepPoint) -> None:
    """Record a point completed under the launcher tests' experiment."""
    bind(store, point)
    record_manifest(store, point, ManifestStatus.COMPLETED)
    derived = EXPERIMENT_SEED.point(point.point_id, point.seed)
    write_rows(store.directory(point), RecordedRows(FREEZE_ID, derived, ()))


class RecordingExecutor:
    """Trains a point by writing a running manifest and then closing it."""

    def __init__(self, store: RunStore, *, outcome: ManifestStatus) -> None:
        self._store = store
        self._outcome = outcome
        self.trained: list[SweepPoint] = []

    def __call__(self, point: SweepPoint) -> RunOutcome:
        record_manifest(self._store, point, ManifestStatus.RUNNING)
        self.trained.append(point)

        if self._outcome is ManifestStatus.COMPLETED:
            complete(self._store, point)
        else:
            record_manifest(self._store, point, self._outcome)

        return RunOutcome(self._store.directory(point), self._outcome, cached=False)


class RecordingRunner:
    """Records the submit commands it is given and reports each one successful."""

    def __init__(self) -> None:
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, argv: Sequence[str]) -> int:
        self.commands.append(tuple(argv))

        return 0


class SilentLauncher:
    """Accepts every point and returns a handle for none of them."""

    def admitted(self, points: Sequence[SweepPoint], /) -> tuple[SweepPoint, ...]:
        return tuple(points)

    def rehearse(self, points: Sequence[SweepPoint], /) -> str:
        return f"{len(points)} points would be dropped"

    def submit(self, points: Sequence[SweepPoint], /) -> list[RunHandle]:
        del points

        return []

    def status(self, handle: RunHandle, /) -> RunStatus:
        del handle

        return RunStatus.COMPLETED


class NoStopRule:
    """Reports that no stop rule fired."""

    @property
    def fired(self) -> tuple[str, ...]:
        return ()


GIT = shutil.which("git") or "git"


def git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        (GIT, "-C", str(root), *arguments),
        check=True,
        capture_output=True,
        encoding="utf-8",
    )

    return completed.stdout.strip()


def head(root: Path) -> str:
    return git(root, "rev-parse", "HEAD")


def tag_checkout(root: Path, tag: str) -> None:
    """Commit everything under a root and tag the commit."""
    git(root, "init", "--quiet", "--initial-branch", "main")
    git(root, "config", "user.name", "Launch Test")
    git(root, "config", "user.email", "launch@example.org")
    git(root, "config", "commit.gpgSign", "false")
    git(root, "config", "tag.gpgSign", "false")
    git(root, "add", "--all")
    git(root, "commit", "--quiet", "--message", "Gate commit")
    git(root, "tag", tag)


def write_freeze(
    root: Path,
    experiment: Path,
    *,
    freeze_id: str = FREEZE_ID,
    sweep: str | None = None,
    commit: str | None = None,
    name: str | None = None,
) -> Path:
    """Write the freeze record a tagged checkout's experiment is launched against."""
    loaded = read_experiment(experiment)
    frozen = sweep_id(expand_experiment(loaded, root)) if sweep is None else sweep
    record = {
        "freeze_id": freeze_id,
        "sweep_id": {loaded.name if name is None else name: frozen},
        "git_tag": {"name": freeze_id, "commit": commit or head(root)},
    }
    path = root / "freezes" / freeze_id.removeprefix("freeze/") / "FREEZE.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8", newline="\n")

    return path


def ordinal_response(document: dict[str, JsonValue], /) -> dict[str, JsonValue]:
    """Return the ordinal response body from a fit document."""
    response = document["response"]

    if not isinstance(response, dict):
        msg = "the fit document has no response declaration"
        raise TypeError(msg)

    body = response["ResponseDeclaration_responsespec"]

    if not isinstance(body, dict):
        msg = "the fit document has no ordinal response body"
        raise TypeError(msg)

    return body


def a_preflight_document() -> dict[str, JsonValue]:
    """Return the launcher experiment as a pre-flight, which is admitted alone."""
    document = an_experiment_document()
    document["role"] = "preflight"

    return document


def a_baseline_experiment_document() -> dict[str, JsonValue]:
    """Return a complete experiment document over a baseline fit."""
    document = an_experiment_document()
    document["config"] = {"kind": "baseline-fit", "path": "baseline.json"}
    document["sweep"] = {
        "grid": [{"path": "system.em_max_iterations", "values": [1, 2]}],
        "seeds": [0, 1, 2],
        "expected_points": 6,
    }
    metrics = (
        "supertagging-length-10",
        "dependencies-length-10",
        "supertagging-length-20",
        "dependencies-length-20",
        "supertagging-all-lengths",
        "dependencies-all-lengths",
    )
    declared_rows: list[dict[str, JsonValue]] = [
        {"evaluator": "baseline", "metric": metric} for metric in metrics
    ]
    rows: list[JsonValue] = list(declared_rows)
    document["rows"] = rows
    primary = declared_rows[-1]
    document["primary"] = primary
    document["factors"] = {
        "trainer.epochs": {
            "axis": "system.em_max_iterations",
            "levels": [
                {"level": 1, "value": 1},
                {"level": 2, "value": 2},
            ],
        }
    }
    document["decision_point"] = {
        "date": "2026-12-01",
        "signal": {
            "kind": "metric",
            "row": primary,
            "statistic": "mean",
            "direction": "above",
            "threshold": 0.5,
        },
        "fallback": "The baseline remains below the target.",
    }
    evaluation = document["evaluation"]
    if not isinstance(evaluation, dict):
        msg = "the experiment fixture has no evaluation declaration"
        raise TypeError(msg)
    seed_budget = evaluation["seed_budget"]
    if not isinstance(seed_budget, dict):
        msg = "the experiment fixture has no seed budget"
        raise TypeError(msg)
    seed_budget["metric"] = "dependencies-all-lengths"

    return document


def a_baseline_fit_document(
    *,
    ccgbank_source: str = "licensed-ccgbank-auto",
    dependency_gold_source: str = "licensed-johansson-nugues-conll",
    reference_source: str = "lofi.resource:configs/eval/bisk_hockenmaier_2012.json",
    reference_tolerance: float = 0.02,
    training_sections: tuple[int, ...] = tuple(range(2, 22)),
    training_max_length: int = 10,
    max_arity: int = 3,
    category_induction_steps: int = 2,
    derived_constituent_pass: bool = True,
    second_order_modifiers: bool = True,
    em_max_iterations: int = 50,
) -> dict[str, JsonValue]:
    """Return one declared Bisk--Hockenmaier fit."""
    return {
        "system": {
            "kind": "bisk-hockenmaier",
            "ccgbank_source": ccgbank_source,
            "dependency_gold_source": dependency_gold_source,
            "reference_source": reference_source,
            "reference_tolerance": reference_tolerance,
            "training_sections": list(training_sections),
            "training_max_length": training_max_length,
            "nominal_category": "N",
            "sentence_category": "S",
            "coordination_category": "conj",
            "gold_sentence_atom": "S",
            "gold_coordination_atom": "conj",
            "nominal_tags": ["NN"],
            "verbal_tags": ["VB", "VBG"],
            "coordination_tags": ["CC"],
            "other_tags": ["JJ", "."],
            "punctuation_tags": ["."],
            "headed_or_modifier_arguments": ["N", "S"],
            "sentence_arguments": ["N"],
            "second_order_modifiers": second_order_modifiers,
            "max_arity": max_arity,
            "category_induction_steps": category_induction_steps,
            "derived_constituent_pass": derived_constituent_pass,
            "em_max_iterations": em_max_iterations,
            "convergence_tolerance": 1e-6,
        }
    }


def write_baseline_experiment(root: Path) -> Path:
    """Write one launcher experiment and its baseline fit."""
    experiment = write_experiment(root, a_baseline_experiment_document())
    (experiment.parent / "baseline.json").write_text(
        json.dumps(a_baseline_fit_document()), encoding="utf-8", newline="\n"
    )
    return experiment


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    experiment = write_experiment(tmp_path, a_preflight_document())
    tag_checkout(tmp_path, FREEZE_ID)
    freeze = write_freeze(tmp_path, experiment)

    return Workspace(tmp_path, experiment, a_profile(tmp_path, tag="site"), freeze)


def expanded(workspace: Workspace) -> tuple[SweepPoint, ...]:
    return expand_experiment(read_experiment(workspace.experiment), workspace.root)


def record_price(
    workspace: Workspace,
    *,
    seconds: tuple[float, ...] = (1.0, 2.0, 3.0),
    peak: int = 3 * 1024**3,
    refused: int = 0,
) -> dict[str, str]:
    """Record the experiment's price, refusing its last points, and name the site."""
    points = expanded(workspace)
    kept = len(points) - refused
    write_price(
        workspace.root / SWEEPS_NAME / sweep_id(points),
        a_price(points[:kept], seconds=seconds, peak=peak, refused=points[kept:]),
    )
    settings = workspace.root / "site.json"
    settings.write_text(
        workspace.profile.model_dump_json(), encoding="utf-8", newline="\n"
    )

    return {SITE_PROFILE_VARIABLE: str(settings)}


def run_main(
    workspace: Workspace,
    *extra: str,
    execute: RecordingExecutor | None = None,
    run: RecordingRunner | None = None,
    environ: Mapping[str, str] | None = None,
    seeds: list[ExperimentSeed] | None = None,
) -> tuple[int, LaunchReport]:
    stdout = io.StringIO()

    with pytest.MonkeyPatch.context() as patch:
        if execute is not None:

            def point_executor(
                store: RunStore,
                *,
                codec: RunSpecCodec,
                operations: TrainingOperations,
                infrastructure_attempts: int,
                failure_policy: FailurePolicySpec,
                environment: ExecutionEnvironment,
            ) -> RecordingExecutor:
                del codec, operations
                assert infrastructure_attempts == failure_policy.infrastructure_attempts
                assert store.freeze_id == FREEZE_ID
                assert environment.git_revision == head(workspace.root)

                if seeds is not None:
                    seeds.append(store.seed)

                return execute

            patch.setattr(run_configuration, "PointExecutor", point_executor)

        status = main(
            [
                "--experiment",
                str(workspace.experiment),
                "--freeze",
                str(workspace.freeze),
                "--out",
                str(workspace.root / "launch.json"),
                *extra,
            ],
            run=run,
            environ=environ,
            stdout=stdout,
            checkout=workspace.root,
        )

    return status, LaunchReport.model_validate_json(stdout.getvalue())


def test_the_second_run_of_an_experiment_sweep_is_all_cache_hits(
    workspace: Workspace,
) -> None:
    store = RunStore(workspace.root, freeze_id=FREEZE_ID, seed=EXPERIMENT_SEED)
    executor = RecordingExecutor(store, outcome=ManifestStatus.COMPLETED)
    arguments = ("--root", str(workspace.root))

    first_status, first = run_main(workspace, *arguments, execute=executor)
    trained = list(executor.trained)
    second_status, second = run_main(workspace, *arguments, execute=executor)

    runs = workspace.root / FREEZE_ID / RUNS_NAME
    directories = sorted(path.name for path in runs.iterdir())
    manifests = sorted(path.parent.name for path in runs.glob(f"*/{MANIFEST_NAME}"))

    assert (first_status, second_status) == (0, 0)
    assert (first.offered, first.launched, first.cache_hits) == (6, 6, 0)
    assert (second.offered, second.launched, second.cache_hits) == (6, 0, 6)
    assert len(directories) == 6
    assert directories == sorted({outcome.point_id for outcome in first.outcomes})
    assert manifests == directories
    assert executor.trained == trained
    assert all(point.provenance is not None for point in trained)
    assert [outcome.cached for outcome in second.outcomes] == [True] * 6


def test_a_serial_launch_seeds_its_points_from_the_experiment_root(
    workspace: Workspace,
) -> None:
    store = RunStore(workspace.root, freeze_id=FREEZE_ID, seed=EXPERIMENT_SEED)
    executor = RecordingExecutor(store, outcome=ManifestStatus.COMPLETED)
    seeds: list[ExperimentSeed] = []

    run_main(workspace, "--root", str(workspace.root), execute=executor, seeds=seeds)

    assert seeds == [read_experiment(workspace.experiment).seed_policy]


def test_the_cli_constructs_the_selected_launcher_plugin(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = RunStore(workspace.root, freeze_id=FREEZE_ID, seed=EXPERIMENT_SEED)
    executor = RecordingExecutor(store, outcome=ManifestStatus.COMPLETED)
    constructions: list[str] = []

    def build(declaration: LaunchSpec, context: LaunchContext) -> Launcher:
        constructions.append(str(declaration.kind))
        return SerialLauncher(
            context.store, execute=context.execute(), stop=context.stop
        )

    plugin = LauncherPlugin(
        spec=type(read_experiment(workspace.experiment).launch),
        build=build,
        requires_site=False,
    )

    def selected(declaration: LaunchSpec) -> LauncherPlugin[LaunchSpec]:
        del declaration
        return plugin

    monkeypatch.setattr(launching_module, "launcher_plugin", selected)

    status, report = run_main(
        workspace,
        "--root",
        str(workspace.root),
        execute=executor,
    )

    assert status == 0
    assert report.launched == 6
    assert constructions == ["serial"]


def experiment_store(workspace: Workspace) -> ExperimentStore:
    return ExperimentStore(workspace.root, read_experiment(workspace.experiment))


def test_a_point_bound_under_another_experiment_root_is_refused(
    workspace: Workspace,
) -> None:
    store = experiment_store(workspace)
    points = expanded(workspace)
    bind(store.runs, points[0], seed=ExperimentSeed(8))
    record_manifest(store.runs, points[0], ManifestStatus.COMPLETED)

    with pytest.raises(ReseededRunError) as refused:
        plan_launch(store, points)

    assert refused.value.point_id == points[0].point_id


def test_a_point_already_completed_is_a_cache_hit_on_its_first_offer(
    workspace: Workspace,
) -> None:
    store = RunStore(workspace.root, freeze_id=FREEZE_ID, seed=EXPERIMENT_SEED)
    executor = RecordingExecutor(store, outcome=ManifestStatus.COMPLETED)
    points = expanded(workspace)
    complete(store, points[1])

    status, report = run_main(
        workspace, "--root", str(workspace.root), execute=executor
    )

    assert status == 0
    assert (report.cache_hits, report.launched) == (1, 5)
    assert [outcome.point_id for outcome in report.outcomes] == [
        points[1].point_id,
        points[0].point_id,
        *(point.point_id for point in points[2:]),
    ]
    assert points[1] not in executor.trained


def test_a_point_left_running_is_launched_again(workspace: Workspace) -> None:
    store = RunStore(workspace.root, freeze_id=FREEZE_ID, seed=EXPERIMENT_SEED)
    executor = RecordingExecutor(store, outcome=ManifestStatus.COMPLETED)
    points = expanded(workspace)
    bind(store, points[2])
    record_manifest(store, points[2], ManifestStatus.RUNNING)

    status, report = run_main(
        workspace, "--root", str(workspace.root), execute=executor
    )

    assert status == 0
    assert (report.cache_hits, report.launched) == (0, 6)
    assert points[2] in executor.trained


def test_a_point_id_is_invariant_under_configuration_key_order(
    workspace: Workspace,
) -> None:
    points = expanded(workspace)
    reordered = SweepPoint(
        dict(reversed(list(points[0].config.items()))), seed=points[0].seed
    )
    store = experiment_store(workspace)
    complete(store.runs, points[0])

    plan = plan_launch(store, (reordered,))

    assert reordered.point_id == points[0].point_id
    assert plan == ((reordered,), ())


def test_plan_launch_keeps_both_halves_in_expansion_order(
    workspace: Workspace,
) -> None:
    store = experiment_store(workspace)
    points = expanded(workspace)

    for index in (0, 3):
        complete(store.runs, points[index])

    plan = plan_launch(store, points)

    assert plan.cached == (points[0], points[3])
    assert plan.pending == (points[1], points[2], points[4], points[5])


def test_the_report_names_the_launch_by_its_request_digest(
    workspace: Workspace,
) -> None:
    store = RunStore(workspace.root, freeze_id=FREEZE_ID, seed=EXPERIMENT_SEED)
    executor = RecordingExecutor(store, outcome=ManifestStatus.COMPLETED)

    _, report = run_main(workspace, "--root", str(workspace.root), execute=executor)

    expected = LaunchRequest(
        experiment="headline",
        freeze_id=FREEZE_ID,
        experiment_seed=7,
        launch=SerialLaunchSpec(),
        root=str(store.root),
        points=tuple(outcome.point_id for outcome in report.outcomes),
    )

    assert report.launch_id == expected.digest()


def test_two_serial_launches_under_different_roots_are_named_apart(
    workspace: Workspace,
) -> None:
    reports = []

    for name in ("alpha", "beta"):
        root = workspace.root / name
        executor = RecordingExecutor(
            RunStore(root, freeze_id=FREEZE_ID, seed=EXPERIMENT_SEED),
            outcome=ManifestStatus.COMPLETED,
        )
        _, report = run_main(workspace, "--root", str(root), execute=executor)
        reports.append(report)

    one, other = reports

    assert one.launch_id != other.launch_id
    assert [outcome.point_id for outcome in one.outcomes] == [
        outcome.point_id for outcome in other.outcomes
    ]
    assert one.outcomes[0].directory != other.outcomes[0].directory


def test_the_written_record_is_the_record_read_back(workspace: Workspace) -> None:
    store = RunStore(workspace.root, freeze_id=FREEZE_ID, seed=EXPERIMENT_SEED)
    executor = RecordingExecutor(store, outcome=ManifestStatus.COMPLETED)

    _, echoed = run_main(workspace, "--root", str(workspace.root), execute=executor)
    written = LaunchReport.read(workspace.root / "launch.json")

    assert written == echoed


def test_an_array_launch_submits_one_job_over_the_pending_points(
    workspace: Workspace,
) -> None:
    profile = workspace.profile
    scratch = Path(profile.storage.scratch_root)
    environ = record_price(workspace)
    runner = RecordingRunner()

    status, report = run_main(
        workspace,
        "--set",
        "launch.kind=array",
        "--set",
        "launch.chunk_size=2",
        run=runner,
        environ=environ,
    )

    (directory,) = (scratch / FREEZE_ID / SWEEPS_NAME).iterdir()
    script = (directory / SCRIPT_NAME).read_text(encoding="utf-8")

    assert status == 0
    assert (report.offered, report.launched, report.cache_hits) == (6, 6, 0)
    assert report.launcher == "array"
    assert runner.commands == [
        (profile.submission.command, str(directory / SCRIPT_NAME))
    ]
    assert (directory / POINTS_NAME).is_file()
    assert (
        shlex.join(task_program(str(scratch), Path("experiments/headline.yaml")))
        in script
    )
    assert all(outcome.status is RunStatus.SUBMITTED for outcome in report.outcomes)
    results = workspace.root / "results" / "1" / "headline"
    status_record = json.loads((results / STATUS_NAME).read_text(encoding="utf-8"))
    assert status_record["counts"] == {"submitted": 6}
    assert read_points(directory / POINTS_NAME).identity == ExperimentIdentity(
        "headline", "freeze/1"
    )


def test_a_remote_launch_runs_one_session_per_pending_point_of_the_experiment(
    workspace: Workspace,
) -> None:
    profile = workspace.profile
    scratch = Path(profile.storage.scratch_root)
    settings = workspace.root / "site.json"
    settings.write_text(profile.model_dump_json(), encoding="utf-8", newline="\n")
    runner = RecordingRunner()

    status, report = run_main(
        workspace,
        "--set",
        "launch.kind=remote",
        run=runner,
        environ={SITE_PROFILE_VARIABLE: str(settings)},
    )
    (directory,) = (scratch / FREEZE_ID / SWEEPS_NAME).iterdir()
    program = shlex.join(task_program(str(scratch), Path("experiments/headline.yaml")))

    assert status == 0
    assert report.launcher == "remote"
    assert (report.offered, report.launched) == (6, 6)
    assert len(runner.commands) == 6
    assert all(program in command[-1] for command in runner.commands)
    assert read_points(directory / POINTS_NAME).identity == ExperimentIdentity(
        "headline", "freeze/1"
    )


def test_an_array_launch_names_its_request_by_the_site_it_submits_to(
    workspace: Workspace,
) -> None:
    environ = record_price(workspace)
    settings = Path(environ[SITE_PROFILE_VARIABLE])

    _, one = run_main(
        workspace, "--set", "launch.kind=array", run=RecordingRunner(), environ=environ
    )

    elsewhere = a_profile(workspace.root / "elsewhere", tag="site")
    settings.write_text(elsewhere.model_dump_json(), encoding="utf-8", newline="\n")

    _, other = run_main(
        workspace, "--set", "launch.kind=array", run=RecordingRunner(), environ=environ
    )

    assert one.outcomes[0].point_id == other.outcomes[0].point_id
    assert one.launch_id != other.launch_id


def dry_run_script(workspace: Workspace, environ: Mapping[str, str]) -> list[str]:
    """Return the script a dry array launch renders, after checking nothing ran."""
    runner = RecordingRunner()
    stdout = io.StringIO()
    status = main(
        [
            "--experiment",
            str(workspace.experiment),
            "--freeze",
            str(workspace.freeze),
            "--out",
            str(workspace.root / "launch.json"),
            "--set",
            "launch.kind=array",
            "--set",
            "launch.chunk_size=2",
            "--dry-run",
        ],
        run=runner,
        environ=environ,
        stdout=stdout,
        checkout=workspace.root,
    )
    scratch = Path(workspace.profile.storage.scratch_root)
    (directory,) = (scratch / FREEZE_ID / SWEEPS_NAME).iterdir()
    script = directory / SCRIPT_NAME

    assert status == 0
    assert runner.commands == []
    assert str(script) in stdout.getvalue()
    assert not (workspace.root / "launch.json").exists()
    assert not (scratch / FREEZE_ID / RUNS_NAME).exists()

    lines = script.read_text(encoding="utf-8").splitlines()
    shutil.rmtree(directory)

    return lines


def test_a_dry_run_renders_resource_lines_that_change_with_the_price(
    workspace: Workspace,
) -> None:
    prefix = workspace.profile.submission.directive_prefix
    quick = dry_run_script(workspace, record_price(workspace))
    slow = dry_run_script(
        workspace,
        record_price(workspace, seconds=(10.0, 20.0, 30.0), peak=6 * 1024**3),
    )
    quick_time, slow_time = (
        next(line for line in lines if line.startswith(f"{prefix} --time-site="))
        for lines in (quick, slow)
    )

    assert f"{prefix} --mem-site=3072" in quick
    assert f"{prefix} --mem-site=6144" in slow
    assert quick_time != slow_time
    assert int(slow_time.rsplit("=", 1)[1]) > 5 * int(quick_time.rsplit("=", 1)[1])


def test_an_over_budget_stack_is_absent_from_the_points_file(
    workspace: Workspace,
) -> None:
    environ = record_price(workspace, refused=2)
    points = expanded(workspace)
    runner = RecordingRunner()

    status, report = run_main(
        workspace, "--set", "launch.kind=array", run=runner, environ=environ
    )

    scratch = Path(workspace.profile.storage.scratch_root)
    (directory,) = (scratch / FREEZE_ID / SWEEPS_NAME).iterdir()

    assert status == 0
    assert read_points(directory / POINTS_NAME).points == points[:4]
    assert report.refused == tuple(point.point_id for point in points[4:])
    assert (report.offered, report.launched) == (4, 4)
    assert {outcome.point_id for outcome in report.outcomes}.isdisjoint(report.refused)


def test_an_array_launch_without_a_price_record_is_refused(
    workspace: Workspace,
) -> None:
    settings = workspace.root / "site.json"
    settings.write_text(
        workspace.profile.model_dump_json(), encoding="utf-8", newline="\n"
    )

    with pytest.raises(UnpricedLaunchError, match="lofi price"):
        run_main(
            workspace,
            "--set",
            "launch.kind=array",
            run=RecordingRunner(),
            environ={SITE_PROFILE_VARIABLE: str(settings)},
        )


def test_task_program_names_the_experiment_and_its_store_root() -> None:
    assert task_program("/scratch/lofi", Path("experiments/headline.yaml")) == (
        "-m",
        "lofi.run",
        "task",
        "--root",
        "/scratch/lofi",
        "--experiment",
        "experiments/headline.yaml",
    )


def test_a_pending_point_the_backend_drops_is_refused(workspace: Workspace) -> None:
    store = experiment_store(workspace)
    points = expanded(workspace)
    plan = plan_launch(store, points)
    request = LaunchRequest(
        experiment="headline",
        freeze_id=FREEZE_ID,
        experiment_seed=7,
        launch=SerialLaunchSpec(),
        root=str(store.root),
        points=tuple(p.point_id for p in plan.pending),
    )

    with pytest.raises(UnlaunchedPointError) as refused:
        launch(plan, request, store=store, launcher=SilentLauncher(), stop=NoStopRule())

    assert refused.value.point_id == points[0].point_id


def test_the_five_factor_product_of_the_series_plans_as_one_launch(
    tmp_path: Path,
) -> None:
    factors = {
        "data.datasets.train.dataset.vocabulary": (64, 128, 256, 512, 1024, 2048),
        "data.datasets.train.dataset.predicates": (2, 4, 8, 16, 32, 64),
        "data.datasets.train.dataset.constructors": (1, 2, 3, 4),
        "model.parser.combinators": ("a", "b", "c", "d", "e"),
        "data.datasets.train.dataset.ambiguity": (1.0, 1.5, 2.0),
    }
    sweep = Sweep.model_validate(
        {
            "grid": [
                {"path": path, "values": list(values)}
                for path, values in factors.items()
            ],
            "seeds": [0, 1],
            "expected_points": 4320,
        }
    )
    points = GridSweepExpander(_assigned).expand(sweep)

    plan = plan_launch(RunStore(tmp_path, freeze_id=None, seed=EXPERIMENT_SEED), points)

    assert len(points) == 4320
    assert plan.pending == points
    assert plan.cached == ()
    assert len({point.point_id for point in points}) == 4320


def _assigned(overrides: tuple[str, ...], _seed: int) -> Mapping[str, JsonValue]:
    """Return a resolved document that is the overrides one point was rendered from."""
    return {"overrides": list(overrides)}


def test_an_array_launch_of_a_judgment_fit_experiment_is_refused_before_anything(
    tmp_path: Path,
) -> None:
    experiment = write_experiment(
        tmp_path, a_judgment_experiment_document(), name="fits"
    )
    tag_checkout(tmp_path, "freeze/1")
    profile = a_profile(tmp_path, tag="site")
    settings = tmp_path / "site.json"
    settings.write_text(profile.model_dump_json(), encoding="utf-8", newline="\n")
    runner = RecordingRunner()
    out = tmp_path / "launch.json"

    with pytest.raises(UnlaunchableExperimentError, match="judgment-fit") as refused:
        main(
            [
                "--experiment",
                str(experiment),
                "--freeze",
                str(tmp_path / "FREEZE.json"),
                "--set",
                "launch.kind=array",
                "--root",
                str(tmp_path),
                "--out",
                str(out),
            ],
            run=runner,
            environ={SITE_PROFILE_VARIABLE: str(settings)},
            stdout=io.StringIO(),
        )

    assert refused.value.experiment == "fits"
    assert refused.value.launcher == "array"
    assert runner.commands == []
    assert not out.exists()
    assert not (tmp_path / FREEZE_ID).exists()
    assert not Path(profile.storage.scratch_root).exists()


def test_an_array_launch_of_a_baseline_fit_experiment_is_refused_before_anything(
    tmp_path: Path,
) -> None:
    assert BiskHockenmaierBaselineSpec.family() == "baseline_system"
    experiment = write_baseline_experiment(tmp_path)
    tag_checkout(tmp_path, FREEZE_ID)
    profile = a_profile(tmp_path, tag="site")
    settings = tmp_path / "site.json"
    settings.write_text(profile.model_dump_json(), encoding="utf-8", newline="\n")
    runner = RecordingRunner()
    out = tmp_path / "launch.json"

    with pytest.raises(UnlaunchableExperimentError, match="baseline-fit"):
        main(
            [
                "--experiment",
                str(experiment),
                "--freeze",
                str(tmp_path / "FREEZE.json"),
                "--set",
                "launch.kind=array",
                "--root",
                str(tmp_path),
                "--out",
                str(out),
            ],
            run=runner,
            environ={SITE_PROFILE_VARIABLE: str(settings)},
            stdout=io.StringIO(),
        )

    assert runner.commands == []
    assert not out.exists()
    assert not (tmp_path / FREEZE_ID).exists()
    assert not Path(profile.storage.scratch_root).exists()


def test_an_array_launch_of_an_active_learning_experiment_is_refused(
    tmp_path: Path,
) -> None:
    write_inputs(tmp_path)
    experiment = an_active_learning_experiment(tmp_path)
    profile = a_profile(tmp_path, tag="site")
    settings = tmp_path / "site.json"
    settings.write_text(profile.model_dump_json(), encoding="utf-8", newline="\n")
    runner = RecordingRunner()
    out = tmp_path / "launch.json"

    with pytest.raises(UnlaunchableExperimentError, match="active-learning"):
        main(
            [
                "--experiment",
                str(experiment),
                "--freeze",
                str(tmp_path / "FREEZE.json"),
                "--set",
                "launch.kind=array",
                "--root",
                str(tmp_path),
                "--out",
                str(out),
            ],
            run=runner,
            environ={SITE_PROFILE_VARIABLE: str(settings)},
            stdout=io.StringIO(),
        )

    assert runner.commands == []
    assert not out.exists()
    assert not Path(profile.storage.scratch_root).exists()


def test_a_serial_baseline_fit_uses_the_registered_operation_by_default(
    tmp_path: Path,
) -> None:
    assert BiskHockenmaierBaselineSpec.family() == "baseline_system"
    experiment = write_baseline_experiment(tmp_path)
    tag_checkout(tmp_path, FREEZE_ID)
    freeze = write_freeze(tmp_path, experiment)
    out = tmp_path / "launch.json"

    def operation(fit: BaselineFitSpec, root: Path, seed: int, /) -> BaselineResult:
        del root, seed
        return BaselineResult(
            tuple(
                BaselineMeasurement(
                    "ccg-induction",
                    "bisk-hockenmaier-2012",
                    metric,
                    "re-running",
                    "Bisk and Hockenmaier 2012",
                    "the published predictions are not released",
                    0.5,
                    1,
                    "proportion",
                )
                for metric in fit.system.metrics()
            ),
            {
                "ccgbank-auto": f"sha256:{'0' * 64}",
                "dependency-gold": f"sha256:{'1' * 64}",
            },
        )

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            baseline_fit_configuration, "RegisteredBaselines", lambda: operation
        )
        status = main(
            [
                "--experiment",
                str(experiment),
                "--freeze",
                str(freeze),
                "--root",
                str(tmp_path),
                "--out",
                str(out),
            ],
            stdout=io.StringIO(),
            checkout=tmp_path,
        )

    assert status == 0
    assert out.exists()
    assert (tmp_path / FREEZE_ID).exists()


def test_a_serial_cli_runs_the_registered_baseline_over_repository_data(
    tmp_path: Path,
) -> None:
    fixture = Path(__file__).parents[2] / "data/eval/bisk_hockenmaier_tiny"
    document = a_baseline_experiment_document()
    experiment = write_experiment(tmp_path, document)
    (experiment.parent / "baseline.json").write_text(
        json.dumps(
            a_baseline_fit_document(
                ccgbank_source=str(fixture / "ccgbank"),
                dependency_gold_source=str(fixture / "evaluation.conll"),
                reference_source=str(fixture / "reference.json"),
                reference_tolerance=0.0,
                training_sections=(2,),
                training_max_length=1,
                max_arity=1,
                category_induction_steps=1,
                derived_constituent_pass=False,
                second_order_modifiers=False,
                em_max_iterations=1,
            )
        ),
        encoding="utf-8",
        newline="\n",
    )
    tag_checkout(tmp_path, FREEZE_ID)
    freeze = write_freeze(tmp_path, experiment)
    out = tmp_path / "launch.json"

    status = main(
        [
            "--experiment",
            str(experiment),
            "--freeze",
            str(freeze),
            "--root",
            str(tmp_path),
            "--out",
            str(out),
        ],
        stdout=io.StringIO(),
        checkout=tmp_path,
    )

    assert status == 0
    assert out.exists()
    assert (tmp_path / FREEZE_ID).exists()


def test_a_launch_from_a_checkout_past_its_freeze_is_refused_before_anything(
    workspace: Workspace,
) -> None:
    (workspace.root / "NOTES.md").write_text(
        "after the freeze\n", encoding="utf-8", newline="\n"
    )
    git(workspace.root, "add", "--all")
    git(workspace.root, "commit", "--quiet", "--message", "Move past the freeze")
    store = RunStore(workspace.root, freeze_id=FREEZE_ID, seed=EXPERIMENT_SEED)
    executor = RecordingExecutor(store, outcome=ManifestStatus.COMPLETED)

    with pytest.raises(UnfrozenRevisionError, match="freeze/1"):
        run_main(workspace, "--root", str(workspace.root), execute=executor)

    assert executor.trained == []
    assert not (workspace.root / FREEZE_ID).exists()
    assert not (workspace.root / "launch.json").exists()


def a_stopping_workspace(root: Path) -> Workspace:
    """Return a tagged launcher workspace whose experiment stops at two completions."""
    document = a_preflight_document()
    document["stop_rules"] = [
        {
            "name": "two completed",
            "when": {"kind": "points", "statuses": ["completed"], "at_least": 2},
        }
    ]
    experiment = write_experiment(root, document)
    tag_checkout(root, FREEZE_ID)
    freeze = write_freeze(root, experiment)

    return Workspace(root, experiment, a_profile(root, tag="site"), freeze)


def test_a_serial_launch_stops_submitting_once_a_stop_rule_fires(
    tmp_path: Path,
) -> None:
    workspace = a_stopping_workspace(tmp_path)
    store = RunStore(workspace.root, freeze_id=FREEZE_ID, seed=EXPERIMENT_SEED)
    executor = RecordingExecutor(store, outcome=ManifestStatus.COMPLETED)
    points = expanded(workspace)

    status, report = run_main(
        workspace, "--root", str(workspace.root), execute=executor
    )
    results = workspace.root / "results" / "1" / "headline"
    record = json.loads((results / STATUS_NAME).read_text(encoding="utf-8"))

    assert status == 3
    assert executor.trained == list(points[:2])
    assert report.stop_rules == ("two completed",)
    assert report.halted == tuple(point.point_id for point in points[2:])
    assert (report.offered, report.cache_hits, report.launched) == (6, 0, 2)
    assert record["counts"] == {"completed": 2, "not-submitted": 4}
    assert record["stop_rules"] == ["two completed"]


def test_a_relaunch_after_a_stop_rule_fired_submits_nothing(tmp_path: Path) -> None:
    workspace = a_stopping_workspace(tmp_path)
    store = RunStore(workspace.root, freeze_id=FREEZE_ID, seed=EXPERIMENT_SEED)
    executor = RecordingExecutor(store, outcome=ManifestStatus.COMPLETED)
    points = expanded(workspace)
    run_main(workspace, "--root", str(workspace.root), execute=executor)

    status, report = run_main(
        workspace, "--root", str(workspace.root), execute=executor
    )

    assert status == 3
    assert executor.trained == list(points[:2])
    assert (report.cache_hits, report.launched) == (2, 0)
    assert report.halted == tuple(point.point_id for point in points[2:])


def test_an_array_launch_whose_stop_rule_fired_submits_nothing(
    tmp_path: Path,
) -> None:
    workspace = a_stopping_workspace(tmp_path)
    environ = record_price(workspace)
    scratch = RunStore(
        Path(workspace.profile.storage.scratch_root),
        freeze_id=FREEZE_ID,
        seed=EXPERIMENT_SEED,
    )
    points = expanded(workspace)

    for point in points[:2]:
        complete(scratch, point)

    runner = RecordingRunner()
    status, report = run_main(
        workspace, "--set", "launch.kind=array", run=runner, environ=environ
    )

    assert status == 3
    assert runner.commands == []
    assert report.halted == tuple(point.point_id for point in points[2:])
    unclaimed = ExperimentStore(
        Path(workspace.profile.storage.scratch_root),
        read_experiment(workspace.experiment),
    )
    assert all(
        unclaimed.entry(point)["status"] == "not-submitted" for point in points[2:]
    )


def test_a_launch_writes_the_experiment_results_directory_from_the_store(
    workspace: Workspace,
) -> None:
    store = RunStore(workspace.root, freeze_id=FREEZE_ID, seed=EXPERIMENT_SEED)
    executor = RecordingExecutor(store, outcome=ManifestStatus.COMPLETED)

    run_main(workspace, "--root", str(workspace.root), execute=executor)

    results = workspace.root / "results" / "1" / "headline"
    status = json.loads((results / STATUS_NAME).read_text(encoding="utf-8"))
    points = expanded(workspace)

    assert status["freeze_id"] == "freeze/1"
    assert status["offered"] == len(points)
    assert list(status["points"]) == [point.point_id for point in points]
    assert sorted(path.stem for path in (results / MANIFESTS_NAME).iterdir()) == (
        sorted(point.point_id for point in points)
    )


def test_a_launch_checks_the_executing_checkout_not_the_experiments_repository(
    workspace: Workspace,
) -> None:
    out = workspace.root / "launch.json"

    with pytest.raises(UnfrozenRevisionError, match="executing checkout"):
        main(
            [
                "--experiment",
                str(workspace.experiment),
                "--freeze",
                str(workspace.freeze),
                "--out",
                str(out),
            ],
            stdout=io.StringIO(),
        )

    assert not (workspace.root / FREEZE_ID).exists()
    assert not out.exists()


def test_an_experiment_outside_the_executing_checkout_is_refused(
    workspace: Workspace, tmp_path_factory: pytest.TempPathFactory
) -> None:
    elsewhere = tmp_path_factory.mktemp("elsewhere")
    (elsewhere / "README.md").write_text(
        "another tree\n", encoding="utf-8", newline="\n"
    )
    tag_checkout(elsewhere, "freeze/1")
    out = workspace.root / "launch.json"

    with pytest.raises(UnfrozenRevisionError, match="lies outside"):
        main(
            [
                "--experiment",
                str(workspace.experiment),
                "--freeze",
                str(workspace.freeze),
                "--out",
                str(out),
            ],
            stdout=io.StringIO(),
            checkout=elsewhere,
        )

    assert not (workspace.root / FREEZE_ID).exists()


def test_a_fit_launch_compiles_its_linkings_from_the_installed_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = write_experiment(
        tmp_path, a_judgment_experiment_document(), name="fits"
    )
    fit = json.loads((experiment.parent / "fit.json").read_text(encoding="utf-8"))
    ordinal_response(fit)["predictor"] = [
        {"kind": "link", "readout": "score", "linking": {"family": "linear"}}
    ]
    (experiment.parent / "fit.json").write_text(
        json.dumps(fit), encoding="utf-8", newline="\n"
    )
    tag_checkout(tmp_path, FREEZE_ID)
    freeze = write_freeze(tmp_path, experiment)
    install(monkeypatch, "linking", ())

    with pytest.raises(
        ValueError,
        match=r"installed linking families \[\], and it declares \['linear'\]",
    ):
        main(
            [
                "--experiment",
                str(experiment),
                "--freeze",
                str(freeze),
                "--root",
                str(tmp_path),
                "--out",
                str(tmp_path / "launch.json"),
            ],
            stdout=io.StringIO(),
            checkout=tmp_path,
        )


def test_an_uncalibratable_fit_experiment_is_refused_before_any_point_is_claimed(
    tmp_path: Path,
) -> None:
    experiment = write_experiment(
        tmp_path, a_judgment_experiment_document(), name="fits"
    )
    fit = json.loads((experiment.parent / "fit.json").read_text(encoding="utf-8"))
    ordinal_response(fit)["predictor"] = [
        {"kind": "link", "readout": "score", "linking": {"family": "linear"}}
    ]
    (experiment.parent / "fit.json").write_text(
        json.dumps(fit), encoding="utf-8", newline="\n"
    )
    tag_checkout(tmp_path, FREEZE_ID)
    freeze = write_freeze(tmp_path, experiment)
    out = tmp_path / "launch.json"

    with pytest.raises(PointExecutionError, match="do not vary over participants"):
        main(
            [
                "--experiment",
                str(experiment),
                "--freeze",
                str(freeze),
                "--root",
                str(tmp_path),
                "--out",
                str(out),
            ],
            stdout=io.StringIO(),
            checkout=tmp_path,
        )

    assert not (tmp_path / FREEZE_ID).exists()
    assert not out.exists()


@pytest.mark.parametrize(
    ("record", "refusal", "detail"),
    [
        ({"name": "analysis"}, FreezeRecordError, "headline"),
        ({"sweep": "0" * 16}, FreezeMismatchError, "expands to the sweep"),
        ({"commit": "f" * 40}, FreezeMismatchError, "records the commit"),
        ({"freeze_id": "freeze/2"}, FreezeMismatchError, "runs against"),
    ],
    ids=["unlisted", "resweeped", "other-commit", "other-freeze"],
)
def test_an_experiment_its_freeze_record_does_not_freeze_is_refused_unclaimed(
    workspace: Workspace,
    record: dict[str, str],
    refusal: type[Exception],
    detail: str,
) -> None:
    freeze = write_freeze(workspace.root, workspace.experiment, **record)
    store = RunStore(workspace.root, freeze_id=FREEZE_ID, seed=EXPERIMENT_SEED)
    executor = RecordingExecutor(store, outcome=ManifestStatus.COMPLETED)

    with pytest.raises(refusal, match=detail):
        run_main(
            workspace._replace(freeze=freeze),
            "--root",
            str(workspace.root),
            execute=executor,
        )

    assert executor.trained == []
    assert not (workspace.root / FREEZE_ID).exists()
    assert not (workspace.root / "launch.json").exists()


def refreeze(workspace: Workspace, freeze_id: str) -> Workspace:
    """Move the workspace's experiment to a new freeze, tagged at a new commit."""
    document = json.loads(workspace.experiment.read_text(encoding="utf-8"))
    document["freeze_id"] = freeze_id
    workspace.experiment.write_text(
        json.dumps(document), encoding="utf-8", newline="\n"
    )
    git(workspace.root, "add", str(workspace.experiment))
    git(workspace.root, "commit", "--quiet", "--message", "Refreeze")
    git(workspace.root, "tag", freeze_id)
    freeze = write_freeze(workspace.root, workspace.experiment, freeze_id=freeze_id)

    return workspace._replace(freeze=freeze)


def test_a_relaunch_under_a_new_freeze_keeps_the_first_freezes_results(
    workspace: Workspace,
) -> None:
    first = RunStore(workspace.root, freeze_id=FREEZE_ID, seed=EXPERIMENT_SEED)
    _, launched = run_main(
        workspace,
        "--root",
        str(workspace.root),
        execute=RecordingExecutor(first, outcome=ManifestStatus.COMPLETED),
    )
    points = expanded(workspace)
    refrozen = refreeze(workspace, "freeze/2")
    second = RunStore(workspace.root, freeze_id="freeze/2", seed=EXPERIMENT_SEED)
    executor = RecordingExecutor(second, outcome=ManifestStatus.KILLED)

    with pytest.MonkeyPatch.context() as patch:

        def point_executor(
            store: RunStore,
            *,
            codec: RunSpecCodec,
            operations: TrainingOperations,
            infrastructure_attempts: int,
            failure_policy: FailurePolicySpec,
            environment: ExecutionEnvironment,
        ) -> RecordingExecutor:
            del codec, operations, environment
            assert infrastructure_attempts == failure_policy.infrastructure_attempts
            assert store.freeze_id == "freeze/2"

            return executor

        patch.setattr(run_configuration, "PointExecutor", point_executor)
        stdout = io.StringIO()
        main(
            [
                "--experiment",
                str(refrozen.experiment),
                "--freeze",
                str(refrozen.freeze),
                "--root",
                str(workspace.root),
                "--out",
                str(workspace.root / "relaunch.json"),
            ],
            stdout=stdout,
            checkout=workspace.root,
        )

    report = LaunchReport.model_validate_json(stdout.getvalue())

    assert (report.cache_hits, report.launched) == (0, len(points))
    assert report.launch_id != launched.launch_id
    assert executor.trained == list(points)
    assert all(first.is_completed(point) for point in points)
    assert all(first.directory(point) != second.directory(point) for point in points)


def test_a_final_failure_is_reported_as_recorded_and_its_stop_rule_halts_the_launch(
    workspace: Workspace,
) -> None:
    store = RunStore(workspace.root, freeze_id=FREEZE_ID, seed=EXPERIMENT_SEED)
    executor = RecordingExecutor(store, outcome=ManifestStatus.COMPLETED)
    points = expanded(workspace)
    failed = points[2]
    bind(store, failed)
    record_manifest(store, failed, ManifestStatus.KILLED)
    derivation = EXPERIMENT_SEED.point(failed.point_id, failed.seed)
    write_failure(
        store.directory(failed),
        PointFailure(FailureKind.NUMERICAL, "the loss is non-finite", 1, derivation),
    )

    exit_status, report = run_main(
        workspace, "--root", str(workspace.root), execute=executor
    )
    results = workspace.root / "results" / "1" / "headline"
    status = json.loads((results / STATUS_NAME).read_text(encoding="utf-8"))
    (outcome,) = report.outcomes

    assert exit_status == 3
    assert executor.trained == []
    assert report.stop_rules == ("numerical failure",)
    assert report.halted == tuple(point.point_id for point in points if point != failed)
    assert (outcome.point_id, outcome.status, outcome.cached) == (
        failed.point_id,
        RunStatus.FAILED,
        True,
    )
    assert status["points"][failed.point_id]["status"] == "numerical-failure"
    assert status["counts"] == {
        "not-submitted": len(points) - 1,
        "numerical-failure": 1,
    }


def test_an_experiment_whose_p90_exceeds_its_admission_is_refused_before_submission(
    workspace: Workspace,
) -> None:
    allocation = SiteAllocation(accelerator_hours=30.0, share=0.5)
    workspace = workspace._replace(
        profile=workspace.profile.with_(allocation=allocation)
    )
    environ = record_price(workspace)
    runner = RecordingRunner()

    with pytest.raises(AdmissionError, match=r"above the 15\.0 one paper may hold"):
        run_main(workspace, "--set", "launch.kind=array", run=runner, environ=environ)

    assert runner.commands == []
    assert not Path(workspace.profile.storage.scratch_root).exists()


def test_a_headline_launch_is_admitted_only_with_its_analysis_experiment(
    tmp_path: Path,
) -> None:
    headline = write_experiment(tmp_path, an_experiment_document(), name="headline")
    analysis = an_experiment_document()
    analysis["role"] = "analysis"
    companion = write_experiment(tmp_path, analysis, name="analysis")
    tag_checkout(tmp_path, FREEZE_ID)
    workspace = Workspace(
        tmp_path,
        headline,
        a_profile(tmp_path, tag="site"),
        write_freeze(tmp_path, headline),
    )
    environ = record_price(workspace)

    with pytest.raises(AdmissionError, match="no analysis experiment"):
        run_main(
            workspace,
            "--set",
            "launch.kind=array",
            run=RecordingRunner(),
            environ=environ,
        )

    status, report = run_main(
        workspace,
        "--set",
        "launch.kind=array",
        "--with",
        str(companion),
        run=RecordingRunner(),
        environ=environ,
    )

    assert status == 0
    assert report.launched == 6


def over_quota_settled_artifacts(
    workspace: Workspace,
) -> tuple[Workspace, Mapping[str, str], Path]:
    """Return a workspace over its quota whose settled point holds an artifact."""
    storage = workspace.profile.storage.with_(
        quota=SiteQuota(
            command=(sys.executable, "-c", "print('used 9 bytes')"),
            used_pattern=r"used (?P<used>\d+)",
            unit_bytes=1_000_000,
            limit_bytes=5_000_000,
        ),
        artifacts=(
            ArtifactClass(
                name="representations",
                pattern="representations",
                regeneration_hours=0.1,
            ),
        ),
    )
    workspace = workspace._replace(profile=workspace.profile.with_(storage=storage))
    environ = record_price(workspace)
    experiment = read_experiment(workspace.experiment)
    store = ExperimentStore(Path(storage.scratch_root), experiment)
    settled = expanded(workspace)[0]
    store.claim(settled)
    binding = store.binding(settled)
    record_manifest(store.runs, settled, ManifestStatus.COMPLETED)
    write_rows(
        store.directory(settled),
        RecordedRows(binding.freeze_id, binding.seed_derivation, ()),
    )
    representations = store.directory(settled) / "representations"
    representations.mkdir()
    (representations / "layer.bin").write_bytes(b"\0" * 4_000_000)

    return workspace, environ, representations


def test_an_array_launch_reclaims_settled_artifacts_before_it_submits(
    workspace: Workspace,
) -> None:
    workspace, environ, representations = over_quota_settled_artifacts(workspace)

    status, _ = run_main(
        workspace, "--set", "launch.kind=array", run=RecordingRunner(), environ=environ
    )

    assert status == 0
    assert not representations.exists()


def dry_run(
    workspace: Workspace, environ: Mapping[str, str], launcher: str
) -> tuple[int, str]:
    """Run a dry launch that must submit nothing, returning its status and text."""
    runner = RecordingRunner()
    stdout = io.StringIO()

    try:
        status = main(
            [
                "--experiment",
                str(workspace.experiment),
                "--freeze",
                str(workspace.freeze),
                "--out",
                str(workspace.root / "launch.json"),
                "--set",
                f"launch.kind={launcher}",
                "--dry-run",
            ],
            run=runner,
            environ=environ,
            stdout=stdout,
            checkout=workspace.root,
        )

    finally:
        assert runner.commands == []

    return status, stdout.getvalue()


def test_a_dry_array_launch_over_its_quota_removes_no_settled_artifact(
    workspace: Workspace,
) -> None:
    workspace, environ, representations = over_quota_settled_artifacts(workspace)

    assert dry_run(workspace, environ, "array")[0] == 0
    assert (representations / "layer.bin").stat().st_size == 4_000_000


def test_a_refused_dry_remote_launch_removes_no_settled_artifact(
    workspace: Workspace,
) -> None:
    workspace, environ, representations = over_quota_settled_artifacts(workspace)

    status, written = dry_run(workspace, environ, "remote")

    assert status == 0
    assert "sessions would be opened" in written

    assert (representations / "layer.bin").stat().st_size == 4_000_000


def test_an_array_launch_over_a_quota_it_cannot_reclaim_is_refused(
    workspace: Workspace,
) -> None:
    quota = workspace.profile.storage.quota.with_(
        command=(sys.executable, "-c", "print('used 9 bytes')"),
        limit_bytes=5,
    )
    storage = workspace.profile.storage.with_(quota=quota)
    workspace = workspace._replace(profile=workspace.profile.with_(storage=storage))
    environ = record_price(workspace)
    runner = RecordingRunner()

    with pytest.raises(QuotaExceededError, match="nothing is submitted"):
        run_main(workspace, "--set", "launch.kind=array", run=runner, environ=environ)

    assert runner.commands == []


class PilotExecutor:
    """Trains a pilot point by recording a completed manifest and unfrozen rows."""

    def __init__(self, store: RunStore) -> None:
        self._store = store
        self.trained: list[SweepPoint] = []

    def __call__(self, point: SweepPoint) -> RunOutcome:
        self.trained.append(point)
        record_manifest(self._store, point, ManifestStatus.COMPLETED)
        derived = EXPERIMENT_SEED.point(point.point_id, point.seed)
        write_rows(
            self._store.directory(point),
            RecordedRows(
                None,
                derived,
                ({"point_id": point.point_id, "freeze_id": UNFROZEN, "value": 1.0},),
            ),
        )

        return RunOutcome(
            self._store.directory(point), ManifestStatus.COMPLETED, cached=False
        )


def an_unfrozen_checkout(root: Path) -> Path:
    """Return an experiment committed in a checkout that no freeze tags."""
    experiment = write_experiment(root, a_preflight_document())
    git(root, "init", "--quiet", "--initial-branch", "main")
    git(root, "config", "user.name", "Launch Test")
    git(root, "config", "user.email", "launch@example.org")
    git(root, "config", "commit.gpgSign", "false")
    git(root, "add", "--all")
    git(root, "commit", "--quiet", "--message", "Draft an experiment")

    return experiment


def run_pilot(
    root: Path, experiment: Path, *extra: str, execute: PilotExecutor | None = None
) -> tuple[int, LaunchReport]:
    stdout = io.StringIO()

    with pytest.MonkeyPatch.context() as patch:

        def point_executor(
            store: RunStore,
            *,
            codec: RunSpecCodec,
            operations: TrainingOperations,
            infrastructure_attempts: int,
            failure_policy: FailurePolicySpec,
            environment: ExecutionEnvironment,
        ) -> PilotExecutor:
            del codec, operations, infrastructure_attempts, failure_policy
            assert store.freeze_id is None
            assert environment.git_revision == head(root)

            return PilotExecutor(store) if execute is None else execute

        patch.setattr(run_configuration, "PointExecutor", point_executor)
        status = main(
            [
                "--experiment",
                str(experiment),
                "--pilot",
                "--out",
                str(root / "pilot-launch.json"),
                *extra,
            ],
            stdout=stdout,
            checkout=root,
        )

    return status, LaunchReport.model_validate_json(stdout.getvalue())


def test_a_pilot_runs_every_point_without_a_freeze_under_the_pilot_root(
    tmp_path: Path,
) -> None:
    experiment = an_unfrozen_checkout(tmp_path)
    loaded = read_experiment(experiment)
    points = expanded_from(tmp_path, experiment)
    store = RunStore(tmp_path / "pilot", freeze_id=None, seed=EXPERIMENT_SEED)
    executor = PilotExecutor(store)

    status, report = run_pilot(tmp_path, experiment, execute=executor)

    results = pilot_directory(tmp_path, loaded)
    record = json.loads((results / STATUS_NAME).read_text(encoding="utf-8"))
    assert status == 0
    assert executor.trained == list(points)
    assert (report.offered, report.cache_hits, report.launched) == (6, 0, 6)
    assert report.launcher == "serial"
    assert record["freeze_id"] == UNFROZEN
    assert record["counts"] == {"completed": 6}
    assert (results / ROWS_NAME).is_file()
    assert all(
        store.directory(point) == tmp_path / "pilot" / RUNS_NAME / point.point_id
        for point in points
    )
    assert not (tmp_path / "results").exists()
    assert not (tmp_path / FREEZE_ID).exists()


def test_a_second_pilot_serves_its_points_from_the_pilot_store(tmp_path: Path) -> None:
    experiment = an_unfrozen_checkout(tmp_path)
    store = RunStore(tmp_path / "pilot", freeze_id=None, seed=EXPERIMENT_SEED)
    run_pilot(tmp_path, experiment, execute=PilotExecutor(store))
    executor = PilotExecutor(store)

    status, report = run_pilot(tmp_path, experiment, execute=executor)

    assert status == 0
    assert executor.trained == []
    assert report.cache_hits == report.offered == 6


def test_a_pilot_cannot_also_name_a_freeze(tmp_path: Path) -> None:
    experiment = an_unfrozen_checkout(tmp_path)

    with pytest.raises(SystemExit):
        run_pilot(tmp_path, experiment, "--freeze", "freezes/1/FREEZE.json")

    assert not (tmp_path / "pilot").exists()


@pytest.mark.parametrize(
    ("extra", "named"),
    [
        (("--dry-run",), "--dry-run"),
        (("--with", "experiments/analysis.yaml"), "--with"),
    ],
)
def test_a_pilot_refuses_every_option_of_a_launch_to_a_site(
    tmp_path: Path, extra: tuple[str, ...], named: str
) -> None:
    experiment = an_unfrozen_checkout(tmp_path)
    stderr = io.StringIO()

    status = cli_main(
        [
            "launch",
            "--experiment",
            str(experiment),
            "--pilot",
            "--out",
            str(tmp_path / "pilot-launch.json"),
            *extra,
        ],
        training=TRAINING_OPERATIONS,
        launching=ExperimentLaunching(checkout=tmp_path),
        stdout=io.StringIO(),
        stderr=stderr,
    )

    assert status == 2
    assert named in stderr.getvalue()
    assert not (tmp_path / "pilot").exists()


def test_a_launch_names_a_freeze_or_a_pilot(tmp_path: Path) -> None:
    experiment = an_unfrozen_checkout(tmp_path)

    with pytest.raises(SystemExit):
        main(
            ["--experiment", str(experiment), "--out", str(tmp_path / "out.json")],
            stdout=io.StringIO(),
            checkout=tmp_path,
        )


def expanded_from(root: Path, experiment: Path) -> tuple[SweepPoint, ...]:
    return expand_experiment(read_experiment(experiment), root)


def poll_main(
    argv: Sequence[str],
    *,
    stdout: io.StringIO,
    sleep: Callable[[float], None] = time.sleep,
    checkout: Path | None = None,
    run: CommandRunner | None = None,
    environ: Mapping[str, str] | None = None,
) -> int:
    """Poll as ``lofi status`` does, raising what the command reports."""
    namespace = build_parser().parse_args(["status", *argv])

    return ExperimentLaunching(
        run=subprocess_runner if run is None else run,
        sleep=sleep,
        checkout=checkout,
    ).poll(
        namespace.experiment,
        root=namespace.root,
        overrides=tuple(namespace.overrides),
        interval=namespace.interval,
        polls=namespace.polls,
        environ={} if environ is None else environ,
        stdout=stdout,
    )


def bind_polled(store: RunStore, point: SweepPoint) -> PointBinding:
    derived = EXPERIMENT_SEED.point(point.point_id, point.seed)
    binding = PointBinding("freeze/1", derived)
    store.claim(point)

    return binding


def complete_polled(store: RunStore, point: SweepPoint) -> None:
    binding = bind_polled(store, point)
    record_manifest(store, point, ManifestStatus.COMPLETED)
    write_rows(
        store.directory(point),
        RecordedRows(
            binding.freeze_id,
            binding.seed_derivation,
            ({"point_id": point.point_id, "metric": "f1", "value": 0.5},),
        ),
    )


def poll_once(
    experiment: Path, store: RunStore, *, exit_status: int = 0
) -> dict[str, JsonValue]:
    stdout = io.StringIO()

    status = poll_main(
        [
            "--experiment",
            str(experiment),
            "--root",
            str(store.root),
            "--interval",
            "0",
            "--polls",
            "1",
        ],
        stdout=stdout,
        checkout=experiment.parents[1],
    )

    assert status == exit_status

    return json.loads(stdout.getvalue())


def test_a_second_poll_reports_the_points_completed_since_without_duplication(
    tmp_path: Path,
) -> None:
    experiment = write_experiment(tmp_path, an_experiment_document())
    points = expand_experiment(read_experiment(experiment), tmp_path)
    store = RunStore(tmp_path / "scratch", freeze_id="freeze/1", seed=EXPERIMENT_SEED)
    results = tmp_path / "results" / "1" / "headline"

    for point in points[:2]:
        complete_polled(store, point)

    bind_polled(store, points[2])
    record_manifest(store, points[2], ManifestStatus.RUNNING)
    first = poll_once(experiment, store, exit_status=2)

    complete_polled(store, points[2])
    complete_polled(store, points[3])
    binding = bind_polled(store, points[4])
    record_manifest(store, points[4], ManifestStatus.KILLED)
    write_failure(
        store.directory(points[4]),
        PointFailure(
            FailureKind.INFRASTRUCTURE, "out of memory", 3, binding.seed_derivation
        ),
    )
    second = poll_once(experiment, store, exit_status=1)
    again = poll_once(experiment, store, exit_status=1)
    rows = pq.read_table(results / ROWS_NAME).to_pylist()

    assert first["counts"] == {"completed": 2, "running": 1, "not-submitted": 3}
    assert second["counts"] == {
        "completed": 4,
        "infra-failed": 1,
        "not-submitted": 1,
    }
    assert again == second
    assert json.loads((results / STATUS_NAME).read_text(encoding="utf-8")) == second
    assert sorted(row["point_id"] for row in rows) == sorted(
        point.point_id for point in points[:4]
    )
    assert sorted(path.stem for path in (results / MANIFESTS_NAME).iterdir()) == (
        sorted(point.point_id for point in points[:5])
    )


def test_a_poll_of_an_empty_store_reports_every_point_unsubmitted(
    tmp_path: Path,
) -> None:
    experiment = write_experiment(tmp_path, an_experiment_document())

    record = poll_once(
        experiment,
        RunStore(tmp_path / "scratch", freeze_id="freeze/1", seed=EXPERIMENT_SEED),
    )

    assert record["counts"] == {"not-submitted": 6}
    assert record["offered"] == 6


class Scheduler:
    """Finishes points between polls, as a scheduler's tasks would."""

    def __init__(self, store: RunStore, steps: list[Callable[[], None]]) -> None:
        self.store = store
        self.steps = steps
        self.waits: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.waits.append(seconds)
        self.steps.pop(0)()


class PollingRunner:
    """Records every scheduler command a poll runs, and runs none of them."""

    def __init__(self) -> None:
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, argv: Sequence[str]) -> int:
        self.commands.append(tuple(argv))

        return 0


def loop(
    experiment: Path,
    store: RunStore,
    scheduler: Scheduler,
    *,
    polls: int,
    run: PollingRunner | None = None,
    environ: Mapping[str, str] | None = None,
) -> tuple[int, list[dict[str, JsonValue]]]:
    stdout = io.StringIO()
    status = poll_main(
        [
            "--experiment",
            str(experiment),
            "--root",
            str(store.root),
            "--interval",
            "30",
            "--polls",
            str(polls),
        ],
        stdout=stdout,
        sleep=scheduler,
        checkout=experiment.parents[1],
        run=PollingRunner() if run is None else run,
        environ={} if environ is None else environ,
    )

    return status, [json.loads(line) for line in stdout.getvalue().splitlines()]


def test_the_poll_loop_updates_the_status_as_points_finish(tmp_path: Path) -> None:
    experiment = write_experiment(tmp_path, an_experiment_document())
    points = expand_experiment(read_experiment(experiment), tmp_path)
    store = RunStore(tmp_path / "scratch", freeze_id="freeze/1", seed=EXPERIMENT_SEED)
    results = tmp_path / "results" / "1" / "headline"

    for point in points:
        bind_polled(store, point)
        record_manifest(store, point, ManifestStatus.RUNNING)

    def finish(chosen: tuple[SweepPoint, ...]) -> Callable[[], None]:
        def step() -> None:
            for point in chosen:
                complete_polled(store, point)

        return step

    scheduler = Scheduler(store, [finish(points[:3]), finish(points[3:])])

    status, records = loop(experiment, store, scheduler, polls=10)

    assert status == 0
    assert [record["counts"] for record in records] == [
        {"running": 6},
        {"completed": 3, "running": 3},
        {"completed": 6},
    ]
    assert scheduler.waits == [30.0, 30.0]
    assert (
        json.loads((results / STATUS_NAME).read_text(encoding="utf-8")) == (records[-1])
    )


def test_the_poll_loop_ends_at_its_bound_nonzero_while_points_still_run(
    tmp_path: Path,
) -> None:
    experiment = write_experiment(tmp_path, an_experiment_document())
    points = expand_experiment(read_experiment(experiment), tmp_path)
    store = RunStore(tmp_path / "scratch", freeze_id="freeze/1", seed=EXPERIMENT_SEED)
    bind_polled(store, points[0])
    record_manifest(store, points[0], ManifestStatus.RUNNING)
    scheduler = Scheduler(store, [lambda: None, lambda: None])

    status, records = loop(experiment, store, scheduler, polls=3)

    assert status == 2
    assert len(records) == 3
    assert records[-1]["counts"] == {"running": 1, "not-submitted": 5}


@pytest.mark.parametrize("omitted", ["--interval", "--polls"])
def test_the_poll_loop_refuses_a_command_line_that_omits_its_cadence(
    tmp_path: Path, omitted: str
) -> None:
    experiment = write_experiment(tmp_path, an_experiment_document())
    arguments = {"--interval": "30", "--polls": "2"}
    del arguments[omitted]

    with pytest.raises(SystemExit):
        poll_main(
            [
                "--experiment",
                str(experiment),
                *(word for pair in arguments.items() for word in pair),
            ],
            stdout=io.StringIO(),
        )


def test_the_poll_loop_exits_nonzero_when_the_scheduler_reports_a_failure(
    tmp_path: Path,
) -> None:
    experiment = write_experiment(tmp_path, an_experiment_document())
    points = expand_experiment(read_experiment(experiment), tmp_path)
    store = RunStore(tmp_path / "scratch", freeze_id="freeze/1", seed=EXPERIMENT_SEED)

    for point in points:
        bind_polled(store, point)
        record_manifest(store, point, ManifestStatus.RUNNING)

    def fail() -> None:
        binding = bind_polled(store, points[0])
        record_manifest(store, points[0], ManifestStatus.KILLED)
        write_failure(
            store.directory(points[0]),
            PointFailure(
                FailureKind.INFRASTRUCTURE, "node lost", 3, binding.seed_derivation
            ),
        )

    status, records = loop(experiment, store, Scheduler(store, [fail]), polls=5)

    assert status == 1
    assert records[-1]["counts"] == {"infra-failed": 1, "running": 5}


def test_the_poll_loop_stops_when_a_declared_stop_rule_fires(tmp_path: Path) -> None:
    document = an_experiment_document()
    document["stop_rules"] = [
        {
            "name": "a killed point",
            "when": {"kind": "points", "statuses": ["killed"], "at_least": 1},
        }
    ]
    experiment = write_experiment(tmp_path, document)
    points = expand_experiment(read_experiment(experiment), tmp_path)
    store = RunStore(tmp_path / "scratch", freeze_id="freeze/1", seed=EXPERIMENT_SEED)

    for point in points:
        bind_polled(store, point)
        record_manifest(store, point, ManifestStatus.RUNNING)

    def kill() -> None:
        record_manifest(store, points[1], ManifestStatus.KILLED)

    status, records = loop(experiment, store, Scheduler(store, [kill]), polls=5)

    assert status == 3
    assert records[-1]["stop_rules"] == ["a killed point"]


def a_stopping_document(launch: dict[str, JsonValue]) -> dict[str, JsonValue]:
    document = an_experiment_document()
    document["launch"] = launch
    document["stop_rules"] = [
        {
            "name": "a killed point",
            "when": {"kind": "points", "statuses": ["killed"], "at_least": 1},
        }
    ]

    return document


def test_a_fired_stop_rule_cancels_the_array_job_through_the_site_scheduler(
    tmp_path: Path,
) -> None:
    profile = a_profile(tmp_path, tag="site")
    settings = tmp_path / "site.json"
    settings.write_text(profile.model_dump_json(), encoding="utf-8", newline="\n")
    experiment = write_experiment(
        tmp_path, a_stopping_document({"kind": "array", "chunk_size": 2})
    )
    loaded = read_experiment(experiment)
    points = expand_experiment(loaded, tmp_path)
    store = RunStore(tmp_path / "scratch", freeze_id="freeze/1", seed=EXPERIMENT_SEED)

    for point in points:
        bind_polled(store, point)
        record_manifest(store, point, ManifestStatus.RUNNING)

    def kill() -> None:
        record_manifest(store, points[1], ManifestStatus.KILLED)

    runner = PollingRunner()
    status, records = loop(
        experiment,
        store,
        Scheduler(store, [kill]),
        polls=5,
        run=runner,
        environ={SITE_PROFILE_VARIABLE: str(settings)},
    )
    job = ArrayJob(profile, ExperimentStore(store.root, loaded).identity, run=runner)

    assert status == 3
    assert records[-1]["stop_rules"] == ["a killed point"]
    assert runner.commands == [profile.submission.cancellation(name=job.name)]


def test_a_poll_whose_stop_rules_hold_cancels_nothing(tmp_path: Path) -> None:
    experiment = write_experiment(
        tmp_path, a_stopping_document({"kind": "array", "chunk_size": 2})
    )
    points = expand_experiment(read_experiment(experiment), tmp_path)
    store = RunStore(tmp_path / "scratch", freeze_id="freeze/1", seed=EXPERIMENT_SEED)
    bind_polled(store, points[0])
    record_manifest(store, points[0], ManifestStatus.RUNNING)
    runner = PollingRunner()

    status, _ = loop(
        experiment, store, Scheduler(store, [lambda: None]), polls=2, run=runner
    )

    assert status == 2
    assert runner.commands == []


def test_a_serial_experiment_whose_stop_rule_fires_leaves_no_job_to_cancel(
    tmp_path: Path,
) -> None:
    experiment = write_experiment(tmp_path, a_stopping_document({"kind": "serial"}))
    points = expand_experiment(read_experiment(experiment), tmp_path)
    store = RunStore(tmp_path / "scratch", freeze_id="freeze/1", seed=EXPERIMENT_SEED)
    bind_polled(store, points[0])
    record_manifest(store, points[0], ManifestStatus.KILLED)
    runner = PollingRunner()

    status, _ = loop(experiment, store, Scheduler(store, []), polls=1, run=runner)

    assert status == 3
    assert runner.commands == []
