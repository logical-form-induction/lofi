import io
import json
import math
from datetime import date
from hashlib import sha256
from typing import TYPE_CHECKING, cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tests.unit.run._boundary.test_metrics import a_record
from tests.unit.run.conftest import EXPERIMENT_SEED, a_running_manifest
from tests.unit.run.test__experiment import (
    a_judgment_experiment_document,
    an_experiment_document,
    write_experiment,
)

from lofi.eval import (
    NO_ENUMERATION_BUDGET,
    UNFROZEN,
    Approximation,
    ConventionVector,
    DeclaredBudget,
    MetricResult,
    ReportIdentity,
    ReportRow,
    Resampling,
    Residual,
    config_digest,
)
from lofi.judgment import (
    DesignProvenance,
    FitArtifact,
    JudgmentFitManifest,
    JudgmentFitSpec,
)
from lofi.run import (
    MANIFEST_NAME,
    Axis,
    ExperimentSeed,
    ExperimentStore,
    FailureKind,
    GridSweepExpander,
    ManifestIso,
    ManifestStatus,
    PointFailure,
    RecordedRows,
    RunStatus,
    RunStore,
    Selection,
    Sweep,
    SweepPoint,
    amend_experiment,
    close_manifest,
    composer,
    encode_seed_derivation,
    expand_experiment,
    main,
    pending_points,
    read_experiment,
    sweep_id,
    write_failure,
    write_rows,
)
from lofi.run._boundary.metrics import write_metrics
from lofi.run.results import (
    AUDIT_NAME,
    INGESTED_NAME,
    SEED_BUDGET_SUFFIX,
    ApproximationError,
    FreezeMismatchError,
    FreezeRecordError,
    PilotResultsError,
    ResultsLayoutError,
    UnfrozenConfigurationError,
    frozen_experiment,
    ingest_experiment,
    ingest_results,
    pilot_directory,
    price_seed_budget,
    results_directory,
    write_ingested,
    write_results,
    write_seed_budget,
)
from lofi.spec import RunSpec, config_search_path
from lofi.stats import Dependence, Holm, paired_permutation

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from lofi._types import JsonValue
    from lofi.run import Experiment
    from lofi.run.results import IngestedResults


class _Grammar:
    fingerprint = "grammar:results-test"


TODAY = date(2026, 11, 1)
"""A day before the fixture experiment's decision point."""

DECIDED = date(2026, 12, 1)
"""The fixture experiment's decision point."""

REVISION = "0123456789abcdef"
"""The source revision every recorded manifest names."""

INFRASTRUCTURE_ATTEMPTS = 3

TAG: dict[str, JsonValue] = {"name": "freeze/1", "commit": REVISION}

UNBUDGETED = Approximation(NO_ENUMERATION_BUDGET, residual=None)
"""The approximation of a row no enumeration budget applies to."""
"""The tag a freeze record names, on the commit the manifests were run at."""


_MEASURED = Residual(
    {
        "beam-truncation": 0.02,
        "chart-cap": 0.0,
        "enumeration-budget": 0.5,
        "prover-timeout": 1.0,
    }
)


def _points(
    root: Path, experiment: Experiment, epochs: tuple[int, ...]
) -> list[SweepPoint]:
    selection = Selection(
        config=experiment.config.path,
        profile=None,
        overrides=(),
        overlays=(),
        search_path=config_search_path(root),
    )
    sweep = Sweep(grid=(Axis(path="trainer.epochs", values=epochs),), seeds=(0, 1, 2))

    return list(GridSweepExpander(composer(selection)).expand(sweep))


class _Layout:
    def __init__(self, tmp_path: Path) -> None:
        self.root = tmp_path / "repository"
        self.path = write_experiment(self.root, an_experiment_document())
        self.experiment = read_experiment(self.path)
        self.points = _points(self.root, self.experiment, (1, 2))
        self.unfrozen = _points(self.root, self.experiment, (3,))[0]
        self.freeze = self.root / "freezes" / "1" / "FREEZE.json"
        self.directory = results_directory(self.root, self.experiment)
        listing = "\n".join(point.point_id for point in self.points)
        self.write_freeze(
            {
                "freeze_id": "freeze/1",
                "sweep_id": {"headline": sha256(listing.encode()).hexdigest()[:16]},
                "git_tag": TAG,
            }
        )
        (self.directory / "manifests").mkdir(parents=True)

        for point in self.points[:2]:
            self.write_manifest(point.point_id, point)

        self.write_status({"completed": 2, "completed_fraction": 0.25})
        self.write_rows([self.row(self.points[0]), self.row(self.points[1])])
        self.write_measured([])

    def write_freeze(self, record: JsonValue) -> None:
        self.freeze.parent.mkdir(parents=True, exist_ok=True)
        self.freeze.write_text(json.dumps(record), encoding="utf-8", newline="\n")

    def write_manifest(
        self, name: str, point: SweepPoint, *, seed: ExperimentSeed = EXPERIMENT_SEED
    ) -> None:
        manifest = ManifestIso().forward(a_running_manifest(point, seed=seed))
        (self.directory / "manifests" / f"{name}.json").write_bytes(manifest)

    def write_status(self, status: JsonValue) -> None:
        (self.directory / "STATUS.json").write_text(
            json.dumps(status), encoding="utf-8", newline="\n"
        )

    def write_rows(self, rows: list[dict[str, JsonValue]]) -> None:
        pq.write_table(pa.Table.from_pylist(rows), self.directory / "rows.parquet")

    def write_measured(self, rows: list[dict[str, JsonValue]]) -> None:
        pq.write_table(pa.Table.from_pylist(rows), self.directory / "measured.parquet")

    def row(
        self,
        point: SweepPoint,
        *,
        freeze_id: str = "freeze/1",
        residual: Residual | None = _MEASURED,
        approximation: Approximation | None = None,
    ) -> dict[str, JsonValue]:
        return ReportRow(
            identity=ReportIdentity(
                RunSpec.model_validate(point.config),
                _Grammar(),
                ConventionVector(
                    punctuation="strip",
                    unary_chains="collapse",
                    trivial_spans="drop",
                    minimum_length=3,
                    aggregation="sentence",
                    coverage="zero",
                    self_arcs="forbid",
                    gold_derivation="canonical",
                ),
                freeze_id=freeze_id,
            ),
            method="ccg",
            metric="unlabeled-bracketing-f1",
            result=MetricResult(0.61, 40, Resampling("sentence", 3)),
            approximation=(
                Approximation(DeclaredBudget(6, 20, mass=None), residual=residual)
                if approximation is None
                else approximation
            ),
        ).to_spec()

    def ingest(
        self,
        *,
        experiment: Experiment | None = None,
        directory: Path | None = None,
    ) -> IngestedResults:
        return ingest_results(
            self.directory if directory is None else directory,
            experiment=self.experiment if experiment is None else experiment,
            freeze=self.freeze,
            root=self.root,
        )


@pytest.fixture
def layout(tmp_path: Path) -> _Layout:
    return _Layout(tmp_path)


def test_a_conforming_directory_is_accepted_and_counts_its_unmeasured_rows(
    layout: _Layout,
) -> None:
    partial = Residual({"beam-truncation": 0.1, "prover-timeout": 0.0})
    layout.write_rows(
        [
            layout.row(layout.points[0]),
            layout.row(layout.points[5], residual=None),
            layout.row(layout.points[3], residual=partial),
        ]
    )

    accepted = layout.ingest()

    assert accepted.freeze_id == "freeze/1"
    assert accepted.experiment == "headline"
    assert [row["freeze_id"] for row in accepted.rows] == ["freeze/1"] * 3
    assert accepted.rows[1]["residual_chart-cap"] is None
    assert accepted.residuals.rows == 3
    assert accepted.residuals.measured == 1
    assert dict(accepted.residuals.by_source) == {
        "beam-truncation": 1,
        "chart-cap": 2,
        "enumeration-budget": 2,
        "prover-timeout": 1,
    }
    assert accepted.audit.tested == ()


def test_an_experiment_run_against_another_freeze_is_refused(layout: _Layout) -> None:
    layout.write_freeze(
        {"freeze_id": "freeze/2", "sweep_id": {"headline": "0"}, "git_tag": TAG}
    )

    with pytest.raises(FreezeMismatchError, match="not the freeze 'freeze/2'"):
        layout.ingest()


def test_an_experiment_whose_sweep_changed_since_the_freeze_is_refused(
    layout: _Layout,
) -> None:
    document = an_experiment_document()
    document["sweep"] = {
        "grid": [{"path": "trainer.epochs", "values": [1, 2]}],
        "seeds": [0, 1, 2, 3],
        "expected_points": 8,
    }
    edited = read_experiment(write_experiment(layout.root, document))

    with pytest.raises(FreezeMismatchError, match="expands to the sweep"):
        layout.ingest(experiment=edited)


@pytest.mark.parametrize(
    "record",
    [
        None,
        [],
        {"sweep_id": {"headline": "0"}},
        {"freeze_id": "freeze/1", "sweep_id": {"analysis": "0"}},
        {"freeze_id": "freeze/1", "sweep_id": "0"},
        {"freeze_id": "freeze/1", "sweep_id": {"headline": "0"}},
        {"freeze_id": "freeze/1", "sweep_id": {"headline": "0"}, "git_tag": {}},
    ],
)
def test_a_freeze_record_that_does_not_record_the_experiment_is_refused(
    layout: _Layout, record: JsonValue
) -> None:
    if record is None:
        layout.freeze.write_text("{", encoding="utf-8", newline="\n")
    else:
        layout.write_freeze(record)

    with pytest.raises(FreezeRecordError):
        layout.ingest()


def test_a_manifest_named_by_another_frozen_point_is_refused(layout: _Layout) -> None:
    first, second = layout.points[:2]
    layout.write_manifest(first.point_id, second)
    layout.write_manifest(second.point_id, first)

    with pytest.raises(UnfrozenConfigurationError, match="not the point its name"):
        layout.ingest()


def test_a_manifest_of_a_point_outside_the_frozen_sweep_is_refused(
    layout: _Layout,
) -> None:
    layout.write_manifest(layout.unfrozen.point_id, layout.unfrozen)

    with pytest.raises(UnfrozenConfigurationError, match="outside the frozen sweep"):
        layout.ingest()


def test_a_manifest_seeded_under_another_experiment_root_is_refused(
    layout: _Layout,
) -> None:
    point = layout.points[0]
    layout.write_manifest(point.point_id, point, seed=ExperimentSeed(8))

    with pytest.raises(FreezeMismatchError, match="experiment root 8"):
        layout.ingest()


def test_a_manifest_names_the_point_its_seed_was_derived_for(
    layout: _Layout,
) -> None:
    point = layout.points[0]
    layout.write_manifest(point.point_id, point)

    accepted = layout.ingest()

    assert accepted.experiment == "headline"


def test_a_manifest_that_does_not_decode_is_refused(layout: _Layout) -> None:
    name = layout.points[0].point_id
    (layout.directory / "manifests" / f"{name}.json").write_text(
        '{"run_spec": {}}', encoding="utf-8", newline="\n"
    )

    with pytest.raises(ResultsLayoutError, match=f"{name}.json"):
        layout.ingest()


def test_a_row_whose_configuration_is_not_a_frozen_point_is_refused(
    layout: _Layout,
) -> None:
    layout.write_rows([layout.row(layout.points[0]), layout.row(layout.unfrozen)])

    with pytest.raises(UnfrozenConfigurationError, match="row 1"):
        layout.ingest()


def test_a_row_scored_outside_any_freeze_is_refused(layout: _Layout) -> None:
    layout.write_rows(
        [layout.row(layout.points[0]), layout.row(layout.points[1], freeze_id=UNFROZEN)]
    )

    with pytest.raises(FreezeMismatchError, match=r"row 1 .* outside any freeze"):
        layout.ingest()


def test_a_row_reported_against_another_freeze_is_refused(layout: _Layout) -> None:
    stray = layout.row(layout.points[0], freeze_id="freeze/2")
    unfrozen = layout.row(layout.points[0])
    del unfrozen["freeze_id"]

    layout.write_rows([layout.row(layout.points[0]), stray])

    with pytest.raises(FreezeMismatchError, match="'freeze/2'"):
        layout.ingest()

    layout.write_rows([unfrozen])

    with pytest.raises(FreezeMismatchError, match="reported against None"):
        layout.ingest()


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("residual_chart-cap", math.nan),
        ("residual_chart-cap", -0.5),
        ("residual_chart-cap", "0.1"),
        ("budget_depth", "6"),
        ("budget_beam", -1),
        ("budget_mass", math.inf),
    ],
)
def test_a_value_its_approximation_type_refuses_is_refused(
    layout: _Layout, column: str, value: JsonValue
) -> None:
    row = layout.row(layout.points[0])
    row[column] = value
    layout.write_rows([row])

    with pytest.raises(ApproximationError, match="refused"):
        layout.ingest()


def test_a_nan_residual_is_not_counted_as_measured(layout: _Layout) -> None:
    row = layout.row(layout.points[0], residual=None)
    row.update({f"residual_{source}": math.nan for source in _MEASURED.measurements})
    row["residual_measured"] = True
    layout.write_rows([row])

    with pytest.raises(ApproximationError, match="not finite"):
        layout.ingest()


def test_an_unmeasured_residual_written_as_zero_is_refused(layout: _Layout) -> None:
    row = layout.row(layout.points[0], residual=None)
    row.update({f"residual_{source}": 0.0 for source in _MEASURED.measurements})
    layout.write_rows([row])

    with pytest.raises(ApproximationError, match="never as a number"):
        layout.ingest()


def test_a_measured_flag_over_null_residuals_is_refused(layout: _Layout) -> None:
    row = layout.row(layout.points[0], residual=None)
    row["residual_measured"] = True
    layout.write_rows([row])

    with pytest.raises(ApproximationError, match="never as a number"):
        layout.ingest()


@pytest.mark.parametrize(
    "column", ["budget_beam", "budget_mass", "residual_prover-timeout"]
)
def test_a_row_missing_an_approximation_column_is_refused(
    layout: _Layout, column: str
) -> None:
    row = layout.row(layout.points[0])
    del row[column]
    layout.write_rows([row])

    with pytest.raises(ApproximationError, match=column):
        layout.ingest()


def test_a_row_missing_its_measured_flag_is_refused(layout: _Layout) -> None:
    row = layout.row(layout.points[0])
    del row["residual_measured"]
    layout.write_rows([row])

    with pytest.raises(ApproximationError, match="residual_measured"):
        layout.ingest()


def test_a_status_that_is_absent_or_not_an_object_is_refused(
    layout: _Layout,
) -> None:
    layout.write_status([])

    with pytest.raises(ResultsLayoutError, match="not a JSON object"):
        layout.ingest()

    (layout.directory / "STATUS.json").write_text("{", encoding="utf-8", newline="\n")

    with pytest.raises(ResultsLayoutError, match=r"STATUS\.json"):
        layout.ingest()

    (layout.directory / "STATUS.json").unlink()

    with pytest.raises(ResultsLayoutError, match=r"STATUS\.json"):
        layout.ingest()


def test_rows_that_are_absent_malformed_or_empty_are_refused(
    layout: _Layout,
) -> None:
    layout.write_rows([])

    with pytest.raises(ResultsLayoutError, match="holds no row"):
        layout.ingest()

    (layout.directory / "rows.parquet").write_bytes(b"not a parquet table")

    with pytest.raises(ResultsLayoutError, match=r"readable rows\.parquet"):
        layout.ingest()

    (layout.directory / "rows.parquet").unlink()

    with pytest.raises(ResultsLayoutError, match=r"readable rows\.parquet"):
        layout.ingest()


def test_the_measured_values_of_each_recorded_point_are_ingested(
    layout: _Layout,
) -> None:
    layout.write_measured(
        [
            {"point_id": layout.points[0].point_id, "point/parameter_count": 9},
            {"point_id": layout.points[1].point_id, "point/parameter_count": 12},
        ]
    )

    accepted = layout.ingest()

    assert [
        (row["point_id"], row["point/parameter_count"]) for row in accepted.points
    ] == [(layout.points[0].point_id, 9), (layout.points[1].point_id, 12)]


def test_a_measured_point_outside_the_frozen_sweep_is_refused(
    layout: _Layout,
) -> None:
    layout.write_measured([{"point_id": layout.unfrozen.point_id}])

    with pytest.raises(UnfrozenConfigurationError, match="outside the frozen sweep"):
        layout.ingest()


def test_a_measured_table_that_is_absent_or_malformed_is_refused(
    layout: _Layout,
) -> None:
    (layout.directory / "measured.parquet").write_bytes(b"not a parquet table")

    with pytest.raises(ResultsLayoutError, match=r"readable measured\.parquet"):
        layout.ingest()

    (layout.directory / "measured.parquet").unlink()

    with pytest.raises(ResultsLayoutError, match=r"readable measured\.parquet"):
        layout.ingest()


def test_a_directory_without_manifests_is_refused(layout: _Layout) -> None:
    for manifest in (layout.directory / "manifests").iterdir():
        manifest.unlink()

    (layout.directory / "manifests").rmdir()

    with pytest.raises(ResultsLayoutError, match="'manifests'"):
        layout.ingest()


@pytest.mark.parametrize(
    "parts",
    [
        ("2", "headline"),
        ("1", "analysis"),
        ("headline",),
        ("A", "1", "headline"),
        ("results", "1", "headline", "headline"),
    ],
)
def test_a_directory_not_filed_under_the_experiments_freeze_is_refused(
    layout: _Layout, tmp_path: Path, parts: tuple[str, ...]
) -> None:
    misfiled = tmp_path.joinpath("elsewhere", *parts)
    misfiled.parent.mkdir(parents=True)
    layout.directory.rename(misfiled)

    with pytest.raises(ResultsLayoutError, match=r"results[/\\]1[/\\]headline, not"):
        layout.ingest(directory=misfiled)


class _FitLayout:
    def __init__(self, layout: _Layout) -> None:
        self.layout = layout
        self.path = write_experiment(
            layout.root, a_judgment_experiment_document(), name="fits"
        )
        self.experiment = read_experiment(self.path)
        self.points = expand_experiment(self.experiment, layout.root)
        compose = self.experiment.config.composer(config_search_path(layout.root))
        self.unfrozen = SweepPoint(compose(("inference.warmup=300",), 0), seed=0)
        self.directory = layout.directory.parent / "fits"
        (self.directory / "manifests").mkdir(parents=True)
        layout.write_freeze(
            {
                "freeze_id": "freeze/1",
                "sweep_id": {"headline": "0", "fits": sweep_id(self.points)},
                "git_tag": TAG,
            }
        )

        for point in self.points[:2]:
            self.write_manifest(point.point_id, point)

        (self.directory / "STATUS.json").write_text(
            json.dumps({"completed": 2, "completed_fraction": 1.0}),
            encoding="utf-8",
            newline="\n",
        )
        self.write_rows(self.points[:2])
        pq.write_table(pa.Table.from_pylist([]), self.directory / "measured.parquet")

    def write_manifest(
        self, name: str, point: SweepPoint, *, seed: ExperimentSeed = EXPERIMENT_SEED
    ) -> None:
        fit = JudgmentFitSpec.model_validate_json(json.dumps(dict(point.config)))
        manifest = JudgmentFitManifest(
            fit=fit,
            design=DesignProvenance(
                tag=fit.dataset,
                checkpoint=None,
                checkpoint_dtype=None,
                unigram_digest=None,
                corpus_digests={},
                seed=None,
                participants=None,
                items=None,
                levels=None,
                population_digest=None,
                producer_run=None,
                producer_point=None,
            ),
            seed_derivation=encode_seed_derivation(
                seed.point(point.point_id, point.seed)
            ),
            artifact=FitArtifact(name="posterior.nc", sha256="0" * 64),
            package_versions={"quivers": "0.25.2"},
            git_revision="0123456789abcdef",
        )
        path = self.directory / "manifests" / f"{name}.json"
        path.write_text(manifest.model_dump_json(), encoding="utf-8", newline="\n")

    def write_rows(
        self, points: Sequence[SweepPoint], *, budget: Approximation = UNBUDGETED
    ) -> None:
        template = self.layout.row(self.layout.points[0], approximation=budget)
        rows = [
            {**template, "config_hash": config_digest(point.config)} for point in points
        ]
        pq.write_table(pa.Table.from_pylist(rows), self.directory / "rows.parquet")

    def ingest(self) -> IngestedResults:
        return ingest_results(
            self.directory,
            experiment=self.experiment,
            freeze=self.layout.freeze,
            root=self.layout.root,
        )


@pytest.fixture
def fits(layout: _Layout) -> _FitLayout:
    return _FitLayout(layout)


def test_a_judgment_fit_experiment_ingests_through_the_same_route(
    fits: _FitLayout,
) -> None:
    accepted = fits.ingest()

    assert accepted.freeze_id == "freeze/1"
    assert accepted.experiment == "fits"
    assert [row["config_hash"] for row in accepted.rows] == [
        config_digest(point.config) for point in fits.points[:2]
    ]
    assert accepted.audit.tested == ()


def test_a_judgment_fit_manifest_named_by_another_point_is_refused(
    fits: _FitLayout,
) -> None:
    first, second = fits.points[:2]
    fits.write_manifest(first.point_id, second)

    with pytest.raises(UnfrozenConfigurationError, match="not the point its name"):
        fits.ingest()


def test_a_judgment_fit_outside_the_frozen_sweep_is_refused(fits: _FitLayout) -> None:
    fits.write_manifest(fits.unfrozen.point_id, fits.unfrozen)

    with pytest.raises(UnfrozenConfigurationError, match="outside the frozen sweep"):
        fits.ingest()

    (fits.directory / "manifests" / f"{fits.unfrozen.point_id}.json").unlink()
    fits.write_rows([fits.points[0], fits.unfrozen])

    with pytest.raises(UnfrozenConfigurationError, match="row 1"):
        fits.ingest()


def test_a_judgment_fit_seeded_under_another_experiment_root_is_refused(
    fits: _FitLayout,
) -> None:
    point = fits.points[0]
    fits.write_manifest(point.point_id, point, seed=ExperimentSeed(8))

    with pytest.raises(FreezeMismatchError, match="experiment root 8"):
        fits.ingest()


def test_an_experiment_manifest_is_not_a_judgment_fit_manifest(
    fits: _FitLayout,
) -> None:
    point = fits.layout.points[0]
    path = fits.directory / "manifests" / f"{fits.points[0].point_id}.json"
    path.write_bytes(ManifestIso().forward(a_running_manifest(point)))

    with pytest.raises(ResultsLayoutError, match="judgment-fit manifest"):
        fits.ingest()


def test_a_judgment_fit_experiment_whose_sweep_changed_is_refused(
    fits: _FitLayout,
) -> None:
    document = a_judgment_experiment_document()
    document["sweep"] = {
        "grid": [{"path": "inference.warmup", "values": [100, 200, 300]}],
        "seeds": [0, 1, 2],
        "expected_points": 9,
    }
    edited = read_experiment(write_experiment(fits.layout.root, document, name="fits"))

    with pytest.raises(FreezeMismatchError, match="expands to the sweep"):
        ingest_results(
            fits.directory,
            experiment=edited,
            freeze=fits.layout.freeze,
            root=fits.layout.root,
        )


def test_a_manifest_run_at_another_revision_than_the_tag_is_refused(
    layout: _Layout,
) -> None:
    point = layout.points[0]
    manifest = a_running_manifest(point).with_(git_revision="fedcba9876543210")
    (layout.directory / "manifests" / f"{point.point_id}.json").write_bytes(
        ManifestIso().forward(manifest)
    )

    with pytest.raises(FreezeMismatchError, match="fedcba9876543210"):
        layout.ingest()


def _record(store: RunStore, point: SweepPoint, status: ManifestStatus) -> None:
    manifest = a_running_manifest(point)

    if status is not ManifestStatus.RUNNING:
        manifest = close_manifest(manifest, status)

    directory = store.directory(point)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / MANIFEST_NAME).write_bytes(ManifestIso().forward(manifest))


def _bound(store: ExperimentStore, point: SweepPoint) -> None:
    store.claim(point)


def _completed(
    store: ExperimentStore, point: SweepPoint, row: dict[str, JsonValue]
) -> None:
    _bound(store, point)
    _record(store.runs, point, ManifestStatus.COMPLETED)
    binding = store.binding(point)
    write_rows(
        store.directory(point),
        RecordedRows(binding.freeze_id, binding.seed_derivation, (row,)),
    )


def test_written_results_report_every_point_and_are_accepted(
    layout: _Layout, tmp_path: Path
) -> None:
    store = ExperimentStore(tmp_path / "store", layout.experiment)
    points = layout.points
    completed, failed, running, killed, submitted, unsubmitted = points
    _completed(store, completed, layout.row(completed))
    _bound(store, failed)
    _record(store.runs, failed, ManifestStatus.KILLED)
    write_failure(
        store.directory(failed),
        PointFailure(
            FailureKind.NUMERICAL,
            "the loss is non-finite",
            1,
            store.binding(failed).seed_derivation,
        ),
    )
    _bound(store, running)
    _record(store.runs, running, ManifestStatus.RUNNING)
    _bound(store, killed)
    _record(store.runs, killed, ManifestStatus.KILLED)
    _bound(store, submitted)
    write_metrics(store.directory(completed), a_record())
    write_metrics(store.directory(failed), a_record())
    write_metrics(store.directory(unsubmitted), a_record())
    directory = results_directory(layout.root, layout.experiment)

    record = write_results(directory, points=points, store=store, today=TODAY)
    accepted = layout.ingest(directory=directory)
    status = json.loads((directory / "STATUS.json").read_text(encoding="utf-8"))

    assert status == record
    assert status["sweep_id"] == sweep_id(points)
    assert status["offered"] == len(points)
    assert status["completed_fraction"] == 1 / len(points)
    assert status["counts"] == {
        "completed": 1,
        "numerical-failure": 1,
        "running": 1,
        "killed": 1,
        "submitted": 1,
        "not-submitted": 1,
    }
    assert status["points"] == {
        completed.point_id: {"status": "completed"},
        failed.point_id: {
            "status": "numerical-failure",
            "diagnostic": "the loss is non-finite",
            "attempts": 1,
        },
        running.point_id: {"status": "running"},
        killed.point_id: {"status": "killed"},
        submitted.point_id: {"status": "submitted"},
        unsubmitted.point_id: {"status": "not-submitted"},
    }
    assert sorted(path.stem for path in (directory / "manifests").iterdir()) == sorted(
        point.point_id for point in (completed, failed, running, killed)
    )
    assert [row["config_hash"] for row in accepted.rows] == [
        config_digest(completed.config)
    ]
    assert [
        (
            row["point_id"],
            row["config_hash"],
            row["status"],
            row["point/parameter_count"],
        )
        for row in accepted.points
    ] == [
        (completed.point_id, config_digest(completed.config), "completed", 9),
        (failed.point_id, config_digest(failed.config), "numerical-failure", 9),
    ]
    assert status["stop_rules"] == ["numerical failure"]
    assert status["decision"] == "pending"


def test_no_stop_rule_fires_while_no_point_has_failed(
    layout: _Layout, tmp_path: Path
) -> None:
    store = ExperimentStore(tmp_path / "store", layout.experiment)
    _completed(store, layout.points[0], layout.row(layout.points[0]))

    record = write_results(
        layout.directory, points=layout.points, store=store, today=TODAY
    )

    assert record["stop_rules"] == []


@pytest.mark.parametrize(
    ("value", "decision"), [(0.9, "signal-held"), (0.2, "fallback")]
)
def test_the_decision_point_reads_its_signal_from_the_rows_on_its_date(
    layout: _Layout, tmp_path: Path, value: float, decision: str
) -> None:
    store = ExperimentStore(tmp_path / "store", layout.experiment)

    for point in layout.points[:3]:
        row = {**layout.row(point), "task": "held-out", "metric": "recovery"}
        _completed(store, point, {**row, "value": value})

    record = write_results(
        layout.directory, points=layout.points, store=store, today=DECIDED
    )

    assert record["decision"] == decision


def test_the_status_counts_each_cell_across_its_seeds(
    layout: _Layout, tmp_path: Path
) -> None:
    store = ExperimentStore(tmp_path / "store", layout.experiment)
    points = layout.points

    for point in points[:3]:
        _completed(store, point, layout.row(point))

    record = write_results(
        tmp_path / "written" / "1" / "headline",
        points=points,
        store=store,
        today=TODAY,
    )
    cells = cast("dict[str, dict[str, JsonValue]]", record["cells"])
    by_cell: dict[str, list[str]] = {}

    for point in points:
        by_cell.setdefault(config_digest(point.config), []).append(point.point_id)

    assert set(cells) == set(by_cell)

    for cell, members in by_cell.items():
        done = sum(point.point_id in members for point in points[:3])

        assert cells[cell]["points"] == members
        assert cells[cell]["offered"] == len(members)
        assert cells[cell]["completed_fraction"] == done / len(members)


def test_a_point_completed_under_another_freeze_is_not_submitted_for_this_one(
    layout: _Layout, tmp_path: Path
) -> None:
    root = tmp_path / "store"
    point = layout.points[0]
    document = an_experiment_document(freeze_id="freeze/2")
    other = ExperimentStore(
        root, read_experiment(write_experiment(tmp_path / "other", document))
    )
    _completed(other, point, layout.row(point, freeze_id="freeze/2"))

    record = write_results(
        tmp_path / "written" / "1" / "headline",
        points=layout.points,
        store=ExperimentStore(root, layout.experiment),
        today=TODAY,
    )
    points = cast("dict[str, JsonValue]", record["points"])

    assert points[point.point_id] == {"status": "not-submitted"}
    assert other.is_completed(point)


@pytest.mark.parametrize(
    ("manifest", "expected"),
    [
        (ManifestStatus.RUNNING, "running"),
        (ManifestStatus.KILLED, "killed"),
    ],
)
def test_an_attempt_short_of_the_bound_reports_its_manifest_and_stays_pending(
    layout: _Layout, tmp_path: Path, manifest: ManifestStatus, expected: str
) -> None:
    store = ExperimentStore(tmp_path / "store", layout.experiment)
    point = layout.points[0]
    _bound(store, point)
    _record(store.runs, point, manifest)
    write_failure(
        store.directory(point),
        PointFailure(
            FailureKind.INTERRUPTED,
            "the attempt started and reached no outcome",
            INFRASTRUCTURE_ATTEMPTS,
            store.binding(point).seed_derivation,
        ),
    )

    assert store.entry(point) == {"status": expected}
    assert not store.is_final(point)
    assert pending_points(store, (point,)) == (point,)


@pytest.mark.parametrize(
    ("kind", "attempts"),
    [
        (FailureKind.NUMERICAL, 1),
        (FailureKind.PREPARATION, 1),
        (FailureKind.INFRASTRUCTURE, INFRASTRUCTURE_ATTEMPTS),
    ],
)
def test_a_final_failure_is_never_offered_as_pending(
    layout: _Layout, tmp_path: Path, kind: FailureKind, attempts: int
) -> None:
    store = ExperimentStore(tmp_path / "store", layout.experiment)
    point = layout.points[0]
    _bound(store, point)
    _record(store.runs, point, ManifestStatus.KILLED)
    write_failure(
        store.directory(point),
        PointFailure(kind, "diagnosed", attempts, store.binding(point).seed_derivation),
    )

    assert store.entry(point)["status"] == kind.value
    assert store.is_final(point)
    assert store.status(point) is RunStatus.FAILED
    assert pending_points(store, (point,)) == ()


def test_rows_produced_under_another_freeze_do_not_complete_a_point(
    layout: _Layout, tmp_path: Path
) -> None:
    store = ExperimentStore(tmp_path / "store", layout.experiment)
    point = layout.points[0]
    _bound(store, point)
    _record(store.runs, point, ManifestStatus.COMPLETED)
    write_rows(
        store.directory(point),
        RecordedRows(
            "freeze/2", store.binding(point).seed_derivation, (layout.row(point),)
        ),
    )

    assert store.entry(point) == {"status": "killed"}


def test_written_results_keep_a_column_the_first_row_does_not_carry(
    layout: _Layout, tmp_path: Path
) -> None:
    store = ExperimentStore(tmp_path / "store", layout.experiment)
    first, second = layout.points[:2]
    plain = layout.row(first)
    annotated = {**layout.row(second), "summary": "coverage", "covered": 3}

    for point, row in ((first, plain), (second, annotated)):
        _completed(store, point, row)

    directory = results_directory(layout.root, layout.experiment)
    write_results(directory, points=layout.points, store=store, today=TODAY)
    rows = layout.ingest(directory=directory).rows

    assert [row["summary"] for row in rows] == [None, "coverage"]
    assert [row["covered"] for row in rows] == [None, 3]
    assert rows[0]["metric"] == plain["metric"]


def test_written_rows_are_the_ones_the_configuration_kind_reports(
    layout: _Layout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ExperimentStore(tmp_path / "store", layout.experiment)
    point = layout.points[0]
    _completed(store, point, layout.row(point))
    reported: list[tuple[int, int]] = []

    def result_rows(
        configuration: object,
        rows: Sequence[Mapping[str, JsonValue]],
        points: Sequence[SweepPoint],
        experiment_store: ExperimentStore,
        /,
    ) -> tuple[Mapping[str, JsonValue], ...]:
        del configuration
        assert experiment_store is store
        reported.append((len(rows), len(points)))

        return ({**rows[0], "summary": "reported by the kind"},)

    monkeypatch.setattr(type(layout.experiment.config), "result_rows", result_rows)
    directory = results_directory(layout.root, layout.experiment)
    write_results(directory, points=layout.points, store=store, today=TODAY)
    written = pq.read_table(directory / "rows.parquet").to_pylist()

    assert reported == [(1, len(layout.points))]
    assert [row["summary"] for row in written] == ["reported by the kind"]


def test_rewritten_results_replace_what_an_earlier_poll_wrote(
    layout: _Layout, tmp_path: Path
) -> None:
    store = ExperimentStore(tmp_path / "store", layout.experiment)
    point = layout.points[0]
    directory = results_directory(layout.root, layout.experiment)
    _bound(store, point)
    _record(store.runs, point, ManifestStatus.RUNNING)
    write_results(directory, points=layout.points, store=store, today=TODAY)

    _completed(store, point, layout.row(point))
    record = write_results(directory, points=layout.points, store=store, today=TODAY)
    written = ManifestIso().backward(
        (directory / "manifests" / f"{point.point_id}.json").read_bytes()
    )
    points = cast("dict[str, JsonValue]", record["points"])

    assert points[point.point_id] == {"status": "completed"}
    assert written.status is ManifestStatus.COMPLETED
    assert len(layout.ingest(directory=directory).rows) == 1


def test_an_experiment_writes_its_results_under_its_freeze_number_and_name(
    layout: _Layout,
) -> None:
    assert results_directory(layout.root, layout.experiment) == (
        layout.root / "results" / "1" / "headline"
    )


def test_a_pilot_files_its_results_under_the_pilot_root(layout: _Layout) -> None:
    assert pilot_directory(layout.root, layout.experiment) == (
        layout.root / "pilot" / "results" / "headline"
    )


def test_written_pilot_results_are_marked_unfrozen(
    layout: _Layout, tmp_path: Path
) -> None:
    store = ExperimentStore(tmp_path / "pilot", layout.experiment, pilot=True)
    point = layout.points[0]
    store.claim(point)
    _record(store.runs, point, ManifestStatus.COMPLETED)
    write_rows(
        store.directory(point),
        RecordedRows(
            None,
            store.binding(point).seed_derivation,
            (layout.row(point, freeze_id=UNFROZEN),),
        ),
    )
    directory = pilot_directory(layout.root, layout.experiment)

    record = write_results(directory, points=layout.points, store=store, today=TODAY)

    assert record["freeze_id"] == UNFROZEN
    assert record["counts"] == {"completed": 1, "not-submitted": 5}
    assert [
        row["freeze_id"]
        for row in pq.read_table(directory / "rows.parquet").to_pylist()
    ] == [UNFROZEN]


def test_ingesting_pilot_results_is_refused_by_name(layout: _Layout) -> None:
    pilot = pilot_directory(layout.root, layout.experiment)
    pilot.mkdir(parents=True)
    (pilot / "STATUS.json").write_text(
        json.dumps({"freeze_id": UNFROZEN}), encoding="utf-8", newline="\n"
    )
    layout.write_status({"freeze_id": UNFROZEN, "completed_fraction": 0.25})

    with pytest.raises(PilotResultsError, match="pilot"):
        layout.ingest(directory=pilot)

    with pytest.raises(PilotResultsError, match="pilot"):
        layout.ingest()

    with pytest.raises(PilotResultsError, match="pilot"):
        ingest_experiment(layout.path, layout.freeze, layout.root)

    assert not (layout.directory / INGESTED_NAME).exists()


def test_a_row_under_no_enumeration_budget_is_accepted_and_counted_apart(
    layout: _Layout,
) -> None:
    layout.write_rows(
        [
            layout.row(layout.points[0]),
            layout.row(layout.points[1], approximation=UNBUDGETED),
        ]
    )

    accepted = layout.ingest()

    assert accepted.rows[1]["budget_kind"] == "none"
    assert accepted.residuals.rows == 2
    assert accepted.residuals.measured == 1
    assert accepted.residuals.not_applicable == 1
    assert accepted.residuals.unmeasured == 0


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("budget_kind", "partial", "budget kind"),
        ("budget_depth", 0, "rebuild"),
        ("residual_measured", True, "rebuild"),
    ],
)
def test_a_row_misstating_that_no_budget_applies_is_refused(
    layout: _Layout, column: str, value: JsonValue, message: str
) -> None:
    row = {**layout.row(layout.points[0], approximation=UNBUDGETED), column: value}
    layout.write_rows([row])

    with pytest.raises(ApproximationError, match=message):
        layout.ingest()


def test_a_row_without_its_budget_kind_is_refused(layout: _Layout) -> None:
    row = layout.row(layout.points[0])
    del row["budget_kind"]
    layout.write_rows([row])

    with pytest.raises(ApproximationError, match="budget_kind"):
        layout.ingest()


def test_a_judgment_fit_row_claiming_an_enumeration_budget_is_refused(
    fits: _FitLayout,
) -> None:
    fits.write_rows(
        fits.points[:2],
        budget=Approximation(DeclaredBudget(0, 0, mass=None), residual=None),
    )

    with pytest.raises(ApproximationError, match="enumeration budget"):
        fits.ingest()


def test_judgment_fit_rows_are_counted_as_under_no_enumeration_budget(
    fits: _FitLayout,
) -> None:
    accepted = fits.ingest()

    assert accepted.residuals.not_applicable == accepted.residuals.rows == 2


def test_ingested_results_record_the_completed_fraction(layout: _Layout) -> None:
    assert layout.ingest().completed_fraction == 0.25

    layout.write_status({"completed": 2})

    with pytest.raises(ResultsLayoutError, match="no completed fraction"):
        layout.ingest()

    layout.write_status({"completed": 2, "completed_fraction": 1.5})

    with pytest.raises(ResultsLayoutError, match="no completed fraction"):
        layout.ingest()


def _declaring(layout: _Layout, **evaluation: JsonValue) -> Experiment:
    document = an_experiment_document()
    declared = cast("dict[str, JsonValue]", document["evaluation"])
    document["evaluation"] = {**declared, **evaluation}

    return read_experiment(write_experiment(layout.root, document))


def _pilot(layout: _Layout, values: dict[int, list[float]]) -> Path:
    pilot = layout.root / "results" / "A" / "0" / "pilot"
    pilot.mkdir(parents=True)
    rows: list[dict[str, JsonValue]] = [
        {
            "metric": "recovery",
            "value": value,
            "seed": seed,
            "config_hash": f"cell-{cell}",
            "task": "brackets",
        }
        for cell, scores in values.items()
        for seed, value in enumerate(scores)
    ]
    rows.append(
        {
            "metric": "parity",
            "value": 0.5,
            "seed": 0,
            "config_hash": "cell-0",
            "task": "brackets",
        }
    )
    pq.write_table(pa.Table.from_pylist(rows), pilot / "rows.parquet")

    return pilot


def test_the_seed_budget_is_priced_from_a_pilot_beside_the_seed_floor(
    layout: _Layout,
) -> None:
    steady = [0.60, 0.61, 0.59, 0.60, 0.61]
    noisy = [0.40, 0.80, 0.55, 0.70, 0.45]
    pilot = _pilot(layout, {0: steady, 1: noisy})

    priced = price_seed_budget(layout.experiment, pilot)
    written = write_seed_budget(layout.experiment, pilot)
    record = json.loads(written.read_text(encoding="utf-8"))

    assert priced.seed_floor == 3
    assert priced.cells["cell-1/brackets"] > priced.cells["cell-0/brackets"]
    assert priced.seeds_required == priced.cells["cell-1/brackets"]
    assert written == layout.path.with_name(f"headline{SEED_BUDGET_SUFFIX}")
    assert record["seed_floor"] == 3
    assert record["seeds_required"] == priced.seeds_required
    assert record["metric"] == "recovery"


def test_a_tighter_declared_precision_prices_more_seeds(layout: _Layout) -> None:
    pilot = _pilot(layout, {0: [0.40, 0.80, 0.55, 0.70, 0.45]})
    loose = price_seed_budget(
        _declaring(
            layout,
            seed_budget={
                "metric": "recovery",
                "target": "mean",
                "precision": 0.2,
                "level": 0.95,
            },
        ),
        pilot,
    )
    tight = price_seed_budget(layout.experiment, pilot)

    assert tight.seeds_required > loose.seeds_required


def test_a_pilot_without_the_budgeted_metric_or_seeds_is_refused(
    layout: _Layout,
) -> None:
    pilot = _pilot(layout, {0: [0.5]})

    with pytest.raises(ValueError, match="pilot seeds"):
        price_seed_budget(layout.experiment, pilot)

    renamed = _declaring(
        layout,
        seed_budget={
            "metric": "absent",
            "target": "mean",
            "precision": 0.05,
            "level": 0.95,
        },
    )

    with pytest.raises(ResultsLayoutError, match="no row of the metric 'absent'"):
        price_seed_budget(renamed, pilot)


def test_the_declared_minimum_floors_the_priced_seed_count(layout: _Layout) -> None:
    pilot = _pilot(layout, {0: [0.60, 0.60001, 0.59999, 0.60]})
    lenient = _declaring(
        layout,
        seed_budget={
            "metric": "recovery",
            "target": "mean",
            "precision": 0.5,
            "level": 0.95,
        },
    )

    assert price_seed_budget(lenient, pilot).seeds_required == 3


def test_a_pilot_cell_scored_twice_at_one_seed_is_refused(layout: _Layout) -> None:
    pilot = layout.root / "results" / "A" / "0" / "pilot"
    pilot.mkdir(parents=True)
    rows: list[dict[str, JsonValue]] = [
        {"metric": "recovery", "value": value, "seed": 0, "config_hash": "cell"}
        for value in (0.4, 0.6)
    ]
    pq.write_table(pa.Table.from_pylist(rows), pilot / "rows.parquet")

    with pytest.raises(ResultsLayoutError, match="twice"):
        price_seed_budget(layout.experiment, pilot)


def test_pilot_cells_are_told_apart_by_dataset_and_split(layout: _Layout) -> None:
    pilot = layout.root / "results" / "A" / "0" / "pilot"
    pilot.mkdir(parents=True)
    rows: list[dict[str, JsonValue]] = [
        {
            "metric": "recovery",
            "value": value + shift,
            "seed": seed,
            "config_hash": "cell",
            "dataset": dataset,
            "split": "test",
        }
        for dataset, shift in (("one", 0.0), ("two", 0.1))
        for seed, value in enumerate((0.40, 0.80, 0.55, 0.70, 0.45))
    ]
    pq.write_table(pa.Table.from_pylist(rows), pilot / "rows.parquet")

    priced = price_seed_budget(layout.experiment, pilot)

    assert sorted(priced.cells) == ["cell/one/test", "cell/two/test"]


def test_an_amended_sweep_reached_from_the_frozen_sweep_is_accepted(
    layout: _Layout,
) -> None:
    document = an_experiment_document()
    document["sweep"] = {
        "grid": [{"path": "trainer.epochs", "values": [1, 2]}],
        "seeds": [0, 1, 2, 3],
        "expected_points": 8,
    }
    path = write_experiment(layout.root, document, name="seeded")
    declared = read_experiment(path)
    frozen = sweep_id(expand_experiment(declared, layout.root))
    layout.write_freeze(
        {"freeze_id": "freeze/1", "sweep_id": {"seeded": frozen}, "git_tag": TAG}
    )
    amended = amend_experiment(declared, "seeds", layout.root, on=TODAY)
    points = expand_experiment(amended, layout.root)

    record = frozen_experiment(layout.freeze, amended, points)

    assert record.sweep_id == frozen
    assert len(points) == 6

    with pytest.raises(FreezeMismatchError, match="expands to the sweep"):
        frozen_experiment(
            layout.freeze, declared, expand_experiment(amended, layout.root)
        )


def test_an_amended_sweep_reached_from_a_sweep_the_freeze_did_not_record_is_refused(
    layout: _Layout,
) -> None:
    document = an_experiment_document()
    document["sweep"] = {
        "grid": [{"path": "trainer.epochs", "values": [1, 2]}],
        "seeds": [0, 1, 2, 3],
        "expected_points": 8,
    }
    path = write_experiment(layout.root, document, name="seeded")
    declared = read_experiment(path)
    layout.write_freeze(
        {"freeze_id": "freeze/1", "sweep_id": {"seeded": "0" * 16}, "git_tag": TAG}
    )
    amended = amend_experiment(declared, "seeds", layout.root, on=TODAY)

    with pytest.raises(FreezeMismatchError, match="expands to the sweep"):
        frozen_experiment(
            layout.freeze, amended, expand_experiment(amended, layout.root)
        )


def test_one_command_ingests_the_results_and_writes_the_tables_and_their_audit(
    layout: _Layout,
) -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()

    status = main(
        [
            "ingest",
            "--experiment",
            "experiments/headline.yaml",
            "--freeze",
            "freezes/1/FREEZE.json",
            "--root",
            str(layout.root),
        ],
        ingest=ingest_experiment,
        environ={},
        stdout=stdout,
        stderr=stderr,
    )
    table = pq.read_table(layout.directory / INGESTED_NAME).to_pylist()
    audit = json.loads((layout.directory / AUDIT_NAME).read_text(encoding="utf-8"))

    assert status == 0, stderr.getvalue()
    assert stdout.getvalue().splitlines() == [
        "rows: 2",
        f"ingested: {layout.directory / INGESTED_NAME}",
        f"audit: {layout.directory / AUDIT_NAME}",
    ]
    assert [row["config_hash"] for row in table] == [
        config_digest(point.config) for point in layout.points[:2]
    ]
    assert audit["freeze_id"] == "freeze/1"
    assert audit["experiment"] == "headline"
    assert audit["rows"] == 2
    assert audit["residuals"]["measured"] == 2
    assert audit["preregistration"]["declared"] == 2
    assert audit["preregistration"]["tested"] == 0
    assert audit["completed_fraction"] == 0.25
    assert audit["decisions"] == {}
    assert audit["fallback"] is None


def test_the_audit_records_each_familys_declared_correction(
    layout: _Layout,
) -> None:
    pvalue = paired_permutation(
        (1.0, 2.0, 3.0, 4.0),
        (0.0, 0.5, 1.0, 1.5),
        replicates=99,
        seed=3,
        tail="greater",
    )
    decisions = Holm().apply((pvalue, pvalue), 0.05, Dependence.ARBITRARY)
    accepted = layout.ingest()._replace(decisions={"primary": decisions})

    audit = json.loads(
        write_ingested(layout.directory, accepted).read_text(encoding="utf-8")
    )

    assert audit["decisions"] == {
        "primary": {
            "correction": "holm",
            "alpha": 0.05,
            "dependence": "arbitrary",
            "raw_p": [pvalue.value, pvalue.value],
            "adjusted_p": list(decisions.adjusted),
            "reject": list(decisions.reject),
        }
    }


def test_the_audit_names_the_fallback_claim_once_the_fallback_rung_applies(
    layout: _Layout,
) -> None:
    assert layout.ingest().fallback is None

    amended = amend_experiment(layout.experiment, "fallback", layout.root, on=TODAY)
    accepted = layout.ingest(experiment=amended)
    audit = json.loads(
        write_ingested(layout.directory, accepted).read_text(encoding="utf-8")
    )

    assert accepted.fallback == amended.decision_point.fallback
    assert audit["fallback"] == amended.decision_point.fallback


def test_the_ingest_command_refuses_results_off_their_freeze(layout: _Layout) -> None:
    layout.write_freeze(
        {"freeze_id": "freeze/1", "sweep_id": {"headline": "0" * 16}, "git_tag": TAG}
    )
    stderr = io.StringIO()

    status = main(
        [
            "ingest",
            "--experiment",
            "experiments/headline.yaml",
            "--freeze",
            "freezes/1/FREEZE.json",
            "--root",
            str(layout.root),
        ],
        ingest=ingest_experiment,
        environ={},
        stdout=io.StringIO(),
        stderr=stderr,
    )

    assert status == 1
    assert "expands to the sweep" in stderr.getvalue()
    assert not (layout.directory / AUDIT_NAME).exists()
