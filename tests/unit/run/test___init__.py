import io
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from tests.unit.run.conftest import a_price, a_profile, a_scratch_store

import lofi._types as shared_types
import lofi.eval
import lofi.run
from lofi.data import Annotation, EvalDataset, Item, ItemId, Token
from lofi.eval import (
    NO_ENUMERATION_BUDGET,
    Approximation,
    ConventionVector,
    MetricResult,
    Resampling,
    Split,
)
from lofi.spec import ConstraintPolicySpec

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lofi._types import JsonValue
    from lofi.formalism import Grammar
    from lofi.spec import RunSpec


_SENTENCES = Resampling("sentence")


def test_the_published_surface_exposes_training_planning() -> None:
    for name in (
        "ExecutionEnvironment",
        "PointExecutor",
        "TrainingBlueprint",
        "TrainingOperations",
        "execution_environment",
        "plan_training",
    ):
        assert name in lofi.run.__all__


def test_every_published_name_resolves() -> None:
    assert [name for name in lofi.run.__all__ if not hasattr(lofi.run, name)] == []


def test_the_published_surface_declares_an_experiment_document() -> None:
    document = lofi.run.ExperimentDocument
    sections = {
        "failure": lofi.run.FailurePolicySpec,
        "seed_policy": lofi.run.ExperimentSeedSpec,
        "primary": lofi.run.RowReference,
        "compute": lofi.run.ComputeSpec,
        "pricing": lofi.run.PricingSpec,
        "contraction": lofi.run.ExperimentContraction,
        "decision_point": lofi.run.ExperimentDecision,
        "evaluation": lofi.run.EvaluationSpec,
    }

    for field, section in sections.items():
        assert document.__field_specs__[field].annotation is section
    assert (
        document.__field_specs__["rows"].annotation == tuple[lofi.run.RowReference, ...]
    )
    assert (
        document.__field_specs__["reports"].annotation
        == tuple[lofi.run.ReportSpec, ...]
    )
    assert (
        document.__field_specs__["stop_rules"].annotation
        == tuple[lofi.run.StopRuleSpec, ...]
    )
    assert (
        document.__field_specs__["factors"].annotation == dict[str, lofi.run.FactorSpan]
    )


def test_the_published_surface_exposes_comparison_point_metadata() -> None:
    assert "comparison_points" in lofi.run.__all__
    assert lofi.run.comparison_points.__module__ == "lofi.run._result_points"


def test_the_published_surface_exposes_only_the_judgment_lowerer() -> None:
    assert "judgment_report_row" in lofi.run.__all__
    for duplicate in (
        "JudgmentReportKey",
        "JudgmentReportRow",
        "Report",
        "ReportIdentity",
        "ReportRow",
    ):
        assert duplicate not in lofi.run.__all__
        assert not hasattr(lofi.run, duplicate)


def test_base_run_import_blocks_optional_reporting_and_neural_deps() -> None:
    script = """
import sys

for name in (
    "bead",
    "huggingface_hub",
    "lairs",
    "pyarrow",
    "pyarrow.parquet",
    "torch",
    "transformers",
):
    sys.modules[name] = None

blocked = set(sys.modules)
import lofi.run

loaded = set(sys.modules) - blocked
for dependency in (
    "bead",
    "huggingface_hub",
    "lairs",
    "pyarrow",
    "torch",
    "transformers",
):
    assert not any(
        name == dependency or name.startswith(f"{dependency}.") for name in loaded
    )

for name in ("LoadedHubModel", "ModelRelease", "export_hub_model", "load_hub_bundle"):
    assert name not in lofi.run.__all__
    assert not hasattr(lofi.run, name)
"""
    finished = subprocess.run(
        [sys.executable, "-"],
        input=script,
        capture_output=True,
        encoding="utf-8",
        check=False,
        cwd=Path(__file__).parents[3],
    )

    assert finished.returncode == 0, finished.stderr


class Agreement:
    def __init__(self, conventions: ConventionVector) -> None:
        self._conventions = conventions
        self._hits: dict[ItemId, bool] = {}

    @property
    def name(self) -> str:
        return "agreement"

    @property
    def conventions(self) -> ConventionVector:
        return self._conventions

    @property
    def order_invariant(self) -> bool:
        return True

    def reset(self) -> None:
        self._hits = {}

    def update(self, pred: str, gold: str, *, key: ItemId) -> None:
        self._hits[key] = pred == gold

    def compute(self) -> MetricResult:
        if not self._hits:
            return MetricResult(None, 0, _SENTENCES)

        return MetricResult(
            sum(self._hits.values()) / len(self._hits), len(self._hits), _SENTENCES
        )

    def per_item(self) -> Mapping[ItemId, MetricResult]:
        return {
            key: MetricResult(float(hit), 1, _SENTENCES)
            for key, hit in self._hits.items()
        }


class _Config:
    def model_dump(self) -> dict[str, JsonValue]:
        return {"run": "public-run-test"}


class _Grammar:
    fingerprint = "grammar:public-run-test"


def test_an_evaluation_runs_from_the_names_the_package_publishes() -> None:
    conventions = ConventionVector(
        punctuation="strip",
        unary_chains="collapse",
        trivial_spans="drop",
        minimum_length=0,
        aggregation="sentence",
        coverage="zero",
        self_arcs="forbid",
        gold_derivation="canonical",
    )
    split: Split = "dev"
    dataset = EvalDataset(
        [
            Item(ItemId("s-1"), (Token(0, "kim"), Token(1, "slept"))),
            Item(ItemId("s-2"), (Token(0, "kim"), Token(1, "slept"))),
        ],
        [Annotation(ItemId("s-1"), "yes"), Annotation(ItemId("s-2"), "no")],
    )
    evaluator = lofi.run.Evaluator(
        dataset,
        [Agreement(conventions)],
        run_spec=cast("RunSpec", _Config()),
        grammar=cast("Grammar", _Grammar()),
        freeze_id="freeze/1",
        fallback_conventions=conventions,
        suite="grammar-comparison",
        task="acceptability",
        dataset_name="megaacceptability",
        split=split,
        seed=4,
        approximation=Approximation(NO_ENUMERATION_BUDGET, residual=None),
        interval=None,
    )

    report = evaluator.evaluate({ItemId("s-1"): "yes", ItemId("s-2"): "yes"})
    (row,) = report.rows

    assert isinstance(report, lofi.eval.Report)
    assert isinstance(row, lofi.eval.ReportRow)
    assert row.metric == "agreement"
    assert row.value == 0.5
    assert row.n == 2
    assert row.split == "dev"
    assert row.identity == lofi.eval.ReportIdentity(
        _Config(), _Grammar(), conventions, freeze_id="freeze/1"
    )
    assert row.convention_hash == row.identity.convention_hash
    assert row.grammar_fingerprint == "grammar:public-run-test"


def test_the_default_sink_is_the_one_that_writes_nowhere() -> None:
    sink: lofi.run.ReportSink = lofi.run.NullSink()

    assert sink.name == "null"


def test_the_package_publishes_sweep_identity() -> None:
    config: dict[str, JsonValue] = {"model": {"width": 64}, "seed": 3}
    point = lofi.run.SweepPoint(config, seed=3)

    assert point.point_id == lofi.run.point_id(point.config, seed=3)
    assert lofi.run.Axis(path="model.width", values=(64,)).path == "model.width"


def test_the_package_publishes_constraint_policy_lowering() -> None:
    policy = lofi.run.resolve_constraint_policy(ConstraintPolicySpec(coverage=1.0))

    assert policy.coverage == 1.0
    assert "resolve_constraint_policy" in lofi.run.__all__


def _nest(overrides: tuple[str, ...], _seed: int) -> dict[str, JsonValue]:
    resolved: dict[str, JsonValue] = {}

    for expression in overrides:
        key, _, text = expression.partition("=")
        resolved[key] = int(text)

    return resolved


def test_the_package_publishes_a_sweep_expander() -> None:
    composer: lofi.run.Composer = _nest
    expander: lofi.run.SweepExpander = lofi.run.GridSweepExpander(composer)
    sweep = lofi.run.Sweep(
        grid=(lofi.run.Axis(path="width", values=(64, 128)),),
        seeds=(1, 2),
        expected_points=4,
    )
    points = expander.expand(sweep)

    assert [(point.config, point.seed) for point in points] == [
        ({"width": 64}, 1),
        ({"width": 64}, 2),
        ({"width": 128}, 1),
        ({"width": 128}, 2),
    ]
    assert lofi.run.unique_points((*points, points[0])) == points

    with pytest.raises(lofi.run.UnexpectedPointCountError):
        expander.expand(sweep.with_(expected_points=3))


def test_the_package_publishes_content_addressed_run_directories(
    tmp_path: Path,
) -> None:
    store = lofi.run.RunStore(tmp_path, freeze_id=None, seed=lofi.run.ExperimentSeed(7))
    point = lofi.run.SweepPoint({"width": 64}, seed=2)

    assert store.directory(point) == tmp_path / lofi.run.RUNS_NAME / point.point_id
    assert store.manifest(point) is None
    assert lofi.run.RunOutcome(
        store.directory(point), lofi.run.ManifestStatus.COMPLETED, cached=True
    ).cached
    assert lofi.run.MANIFEST_NAME.endswith(".json")


def test_the_package_publishes_the_experiment_manifest_boundary() -> None:
    assert lofi.run.ManifestIso.__name__ == "ManifestIso"
    assert lofi.run.ExperimentManifest.__name__ == "ExperimentManifest"
    assert tuple(lofi.run.ManifestStatus) == (
        lofi.run.ManifestStatus.RUNNING,
        lofi.run.ManifestStatus.COMPLETED,
        lofi.run.ManifestStatus.KILLED,
    )


def test_the_package_publishes_submodule_sharing_values() -> None:
    assert lofi.run.ModuleSharing is shared_types.ModuleSharing
    assert lofi.run.SubmoduleDeclaration is shared_types.SubmoduleDeclaration


def test_the_package_publishes_metric_sinks_and_their_laws() -> None:
    (sink,) = lofi.run.resolve_metric_sinks({}, lofi.run.SinkContext("point"))

    sink.log_scalars({"loss": 0.25}, 3)
    sink.log_table("losses", ({"name": "loss", "value": 0.25},))
    sink.close()

    assert len(lofi.run.METRIC_SINK_LAWS) == 3


def test_the_package_publishes_seed_and_device_scopes_lazily() -> None:
    policy = lofi.run.SeedPolicy(5, samplers=("fit",))
    streams = lofi.run.RandomStreams(policy)

    assert streams.sampler_seed("fit") == policy.sampler_seed("fit")

    for name in (
        "AcceleratorRandom",
        "RandomStreams",
        "SeedPolicy",
        "device_scope",
    ):
        assert name in lofi.run.__all__


class _Recording:
    def __init__(self) -> None:
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, argv: Sequence[str]) -> int:
        self.commands.append(tuple(argv))

        return 0


def test_the_package_publishes_three_launcher_backends(tmp_path: Path) -> None:
    profile = a_profile(tmp_path, tag="pkg")
    store = lofi.run.RunStore(tmp_path, freeze_id=None, seed=lofi.run.ExperimentSeed(7))
    point = lofi.run.SweepPoint({"width": 64}, seed=2)
    recording = _Recording()
    scratch = a_scratch_store(profile)
    array: lofi.run.Launcher = lofi.run.ArrayLauncher(
        profile,
        store=scratch,
        program=("-m", "lofi"),
        chunk_size=8,
        price=a_price((point,)),
        throttle=4,
        run=recording,
    )
    remote: lofi.run.Launcher = lofi.run.RemoteLauncher(
        profile, store=scratch, program=("-m", "lofi"), run=recording
    )
    serial: lofi.run.Launcher = lofi.run.SerialLauncher(
        store,
        execute=lambda point: lofi.run.RunOutcome(
            store.directory(point), lofi.run.ManifestStatus.COMPLETED, cached=False
        ),
    )
    rendered = array.render((point,))

    assert rendered is not None
    assert rendered.tasks == lofi.run.task_count(1, 8)
    assert lofi.run.read_points(rendered.points).points == (point,)
    assert recording.commands == []
    assert len(remote.submit((point,))) == 1
    assert recording.commands[0][-1].endswith(" 0 1")
    assert [handle.point_id for handle in serial.submit((point,))] == [point.point_id]
    assert serial.status(
        lofi.run.RunHandle(point.point_id, str(store.directory(point)))
    )


def test_the_command_line_runs_from_the_name_the_package_publishes() -> None:
    stdout = io.StringIO()

    status = lofi.run.main(
        ["configs"],
        stdout=stdout,
        stderr=io.StringIO(),
    )

    assert status == 0
    assert "model.parser" in stdout.getvalue()
