import os
import stat
from hashlib import sha256
from typing import TYPE_CHECKING, cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tests.unit.run.test__experiment import (
    an_experiment_document,
    write_experiment,
)

from lofi.eval import (
    NO_ENUMERATION_BUDGET,
    Approximation,
    MetricResult,
    ReportIdentity,
    ReportRow,
    Resampling,
)
from lofi.report import REPORT_FIELDS, report_renderer
from lofi.run import read_experiment, reporting
from lofi.run.reporting import (
    REPORT_SCHEMA,
    DuplicateReportRowError,
    ParquetReportSink,
    ReportSelector,
    RequiredReportRowsError,
    render_experiment_reports,
    render_table,
    select_required_rows,
)
from lofi.run.results import ROWS_NAME, pilot_directory, results_directory

if TYPE_CHECKING:
    from pathlib import Path

    from lofi._types import JsonValue


class _Config:
    def __init__(self, run: str = "reporting-test") -> None:
        self._run = run

    def model_dump(self) -> dict[str, JsonValue]:
        return {"run": self._run}


class _Grammar:
    def __init__(self, fingerprint: str = "grammar:reporting-test") -> None:
        self.fingerprint = fingerprint


def _identity(model: str = "reporting-test") -> ReportIdentity:
    return ReportIdentity(
        _Config(model),
        _Grammar(f"grammar:{model}"),
        None,
        freeze_id="freeze/8",
    )


def _row(
    metric: str,
    value: float | None,
    *,
    fold: int = 0,
    task: str = "held-out-fit",
    model: str = "reporting-test",
) -> ReportRow:
    return ReportRow(
        identity=_identity(model),
        method="verb-by-frame",
        metric=metric,
        result=MetricResult(value, 0 if value is None else 4, Resampling("response")),
        approximation=Approximation(NO_ENUMERATION_BUDGET, residual=None),
        suite="judgment-models",
        task=task,
        dataset="mega-veridicality",
        split="test",
        seed=23,
        specification_hash="specification-d4",
        fold=fold,
        rung="verb-by-frame",
        held_out_convention="marginal",
    )


def _selector(row: ReportRow) -> ReportSelector:
    return ReportSelector(
        freeze_id=row.freeze_id,
        config_hash=row.config_hash,
        grammar_fingerprint=row.grammar_fingerprint,
        convention_hash=row.convention_hash,
        suite="judgment-models",
        task="held-out-fit",
        dataset="mega-veridicality",
        split="test",
        method=row.method,
        metric=row.metric,
        seed=23,
        specification_hash="specification-d4",
        fold=row.fold,
        rung="verb-by-frame",
        held_out_convention="marginal",
    )


def test_the_schema_is_the_canonical_row_schema() -> None:
    assert REPORT_SCHEMA.names == list(_row("recovery", 0.5).to_spec())
    assert REPORT_SCHEMA.names[:10] == [
        "freeze_id",
        "config_hash",
        "grammar_fingerprint",
        "convention_hash",
        "suite",
        "task",
        "dataset",
        "split",
        "method",
        "metric",
    ]
    assert "budget_kind" in REPORT_SCHEMA.names
    assert "residual_beam-truncation" in REPORT_SCHEMA.names
    assert [(field.name, field.nullable) for field in REPORT_SCHEMA] == [
        (field.name, field.nullable) for field in REPORT_FIELDS
    ]


def test_repeated_emissions_append_in_order_and_preserve_null_and_zero(
    tmp_path: Path,
) -> None:
    path = tmp_path / "report.parquet"
    absent = _row("unavailable", None)
    zero = _row("pseudo-r-squared", 0.0)
    sink = ParquetReportSink(path)

    sink.emit((absent,))
    sink.emit((zero,))
    table = pq.read_table(path)

    assert table.schema == REPORT_SCHEMA
    assert table.column("metric").to_pylist() == ["unavailable", "pseudo-r-squared"]
    assert table.column("value").to_pylist() == [None, 0.0]


def test_a_duplicate_complete_selector_is_refused_before_the_file_changes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "report.parquet"
    original = _row("predictive-density", -0.5)
    sink = ParquetReportSink(path)
    sink.emit((original,))
    digest = sha256(path.read_bytes()).digest()

    with pytest.raises(DuplicateReportRowError) as raised:
        sink.emit((_row("predictive-density", -0.25),))

    assert raised.value.selectors == (_selector(original),)
    assert sha256(path.read_bytes()).digest() == digest


def test_a_selector_repeated_within_one_append_creates_no_file(tmp_path: Path) -> None:
    path = tmp_path / "report.parquet"
    sink = ParquetReportSink(path)

    with pytest.raises(DuplicateReportRowError):
        sink.emit(
            (
                _row("predictive-density", -0.5),
                _row("predictive-density", -0.25),
            )
        )

    assert not path.exists()


def test_a_failed_replacement_leaves_the_prior_report_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "report.parquet"
    original = _row("predictive-density", -0.5)
    sink = ParquetReportSink(path)
    sink.emit((original,))
    before = path.read_bytes()

    def fail_write(table: pa.Table, where: Path) -> None:
        del table, where
        msg = "disk refused the staged write"
        raise OSError(msg)

    monkeypatch.setattr(pq, "write_table", fail_write)

    with pytest.raises(OSError, match="staged write"):
        sink.emit((_row("posterior-variance", 0.25),))

    assert path.read_bytes() == before
    assert not any(path.parent.glob(f".{path.name}.*.staged"))


def test_required_selection_preserves_declared_order(tmp_path: Path) -> None:
    path = tmp_path / "report.parquet"
    rows = (
        _row("predictive-density", -0.5),
        _row("posterior-variance", 0.25),
        _row("pseudo-r-squared", 0.0),
    )
    ParquetReportSink(path).emit(rows)

    selected = select_required_rows(path, (_selector(rows[2]), _selector(rows[0])))

    assert tuple(row["metric"] for row in selected) == (
        "pseudo-r-squared",
        "predictive-density",
    )


def test_required_selection_names_missing_rows(tmp_path: Path) -> None:
    path = tmp_path / "report.parquet"
    present = _row("predictive-density", -0.5)
    missing = _row("pseudo-r-squared", 0.0)
    ParquetReportSink(path).emit((present,))

    with pytest.raises(RequiredReportRowsError) as raised:
        select_required_rows(path, (_selector(missing),))

    assert raised.value.missing == (_selector(missing),)


def test_required_selection_names_ambiguous_rows(tmp_path: Path) -> None:
    path = tmp_path / "report.parquet"
    repeated = _row("predictive-density", -0.5)
    pq.write_table(
        pa.Table.from_pylist(
            (repeated.to_spec(), repeated.to_spec()), schema=REPORT_SCHEMA
        ),
        path,
    )

    with pytest.raises(RequiredReportRowsError) as raised:
        select_required_rows(path, (_selector(repeated),))

    assert raised.value.ambiguous == (_selector(repeated),)


@pytest.mark.parametrize(
    ("renderer", "expected"),
    [
        ("markdown", ("| pseudo_r_squared | 0.00 |", "| predictive_density | -0.51 |")),
        ("latex", (r"pseudo\_r\_squared & 0.00 \\", r"predictive\_density & -0.51 \\")),
    ],
)
def test_one_renderer_writes_the_selected_rows_in_selector_order(
    tmp_path: Path, renderer: str, expected: tuple[str, str]
) -> None:
    path = tmp_path / "report.parquet"
    rows = (_row("predictive_density", -0.5123), _row("pseudo_r_squared", 0.0))
    ParquetReportSink(path).emit(rows)
    selected = select_required_rows(path, (_selector(rows[1]), _selector(rows[0])))

    rendered = render_table(
        report_renderer(renderer),
        selected,
        columns=("metric", "value"),
        labels=("Metric", "Value"),
        precision=2,
    ).decode()

    assert all(line in rendered.splitlines() for line in expected)
    assert rendered.index(expected[0]) < rendered.index(expected[1])


@pytest.mark.parametrize(
    ("columns", "labels", "message"),
    [
        ((), (), "0 columns but 0 labels"),
        (("metric",), (), "1 columns but 0 labels"),
        (("metric", "not-stored"), ("a", "b"), "lack declared columns"),
    ],
)
def test_one_renderer_refuses_a_table_its_rows_cannot_fill(
    tmp_path: Path,
    columns: tuple[str, ...],
    labels: tuple[str, ...],
    message: str,
) -> None:
    path = tmp_path / "report.parquet"
    ParquetReportSink(path).emit((_row("recovery", 0.5),))
    selected = select_required_rows(path, (_selector(_row("recovery", 0.5)),))

    with pytest.raises(ValueError, match=message):
        render_table(
            report_renderer("markdown"),
            selected,
            columns=columns,
            labels=labels,
            precision=2,
        )


def test_a_selector_refuses_a_held_out_convention_no_policy_declares() -> None:
    row = _row("recovery", 0.5)
    with pytest.raises(ValueError, match="unknown held-out convention"):
        ReportSelector(
            freeze_id=row.freeze_id,
            config_hash=row.config_hash,
            grammar_fingerprint=row.grammar_fingerprint,
            convention_hash=row.convention_hash,
            suite="judgment-models",
            task="held-out-fit",
            dataset="mega-veridicality",
            split="test",
            method=row.method,
            metric=row.metric,
            seed=23,
            held_out_convention="posterior",
        )


def test_a_declared_latex_report_renders_three_columns_and_nan_rates_by_model(
    tmp_path: Path,
) -> None:
    document = an_experiment_document()
    document["reports"] = [
        {
            "name": "headline",
            "rows": ["held-out/recovery"],
            "columns": ["metric", "value", "nan_rate"],
            "labels": ["Metric", "Value", "NaN rate"],
            "precision": 2,
            "format": "latex",
        }
    ]
    path = write_experiment(tmp_path, document)
    experiment = read_experiment(path)
    directory = results_directory(tmp_path, experiment)
    rows = (
        _row("recovery", 0.75, fold=0, task="held-out", model="model-a"),
        _row("recovery", None, fold=1, task="held-out", model="model-a"),
        _row("recovery", 0.25, fold=0, task="held-out", model="model-b"),
    )
    ParquetReportSink(directory / ROWS_NAME).emit(rows)

    written = render_experiment_reports(path, tmp_path, pilot=False)

    assert written == (directory / "headline.tex",)
    rendered = written[0].read_text(encoding="utf-8")
    assert "Metric & Value & NaN rate" in rendered
    assert "recovery & 0.75 & 0.50" in rendered
    assert "recovery &  & 0.50" in rendered
    assert "recovery & 0.25 & 0.00" in rendered


def test_a_pilot_report_renders_the_pilot_rows_and_not_the_frozen_ones(
    tmp_path: Path,
) -> None:
    document = an_experiment_document()
    document["reports"] = [
        {
            "name": "headline",
            "rows": ["held-out/recovery"],
            "columns": ["metric", "value"],
            "labels": ["Metric", "Value"],
            "precision": 2,
        }
    ]
    path = write_experiment(tmp_path, document)
    experiment = read_experiment(path)
    pilot = pilot_directory(tmp_path, experiment)
    ParquetReportSink(pilot / ROWS_NAME).emit(
        (_row("recovery", 0.75, task="held-out"),)
    )

    written = render_experiment_reports(path, tmp_path, pilot=True)

    assert written == (pilot / "headline.md",)
    assert "recovery | 0.75" in written[0].read_text(encoding="utf-8")
    assert not results_directory(tmp_path, experiment).exists()


def test_the_sink_refuses_a_file_with_another_schema(tmp_path: Path) -> None:
    path = tmp_path / "report.parquet"
    pq.write_table(pa.table({"metric": ("predictive-density",)}), path)

    with pytest.raises(ValueError, match="expected"):
        ParquetReportSink(path).emit((_row("posterior-variance", 0.25),))


def test_an_empty_emission_does_not_create_a_file(tmp_path: Path) -> None:
    path = tmp_path / "report.parquet"
    sink = ParquetReportSink(path)

    sink.emit(())

    assert not path.exists()
    assert repr(sink) == f"ParquetReportSink({path!r})"


def test_the_sink_accepts_only_the_canonical_report_row(tmp_path: Path) -> None:
    sink = ParquetReportSink(tmp_path / "report.parquet")

    with pytest.raises(TypeError, match="not canonical"):
        sink.emit((cast("ReportRow", {"metric": "not-a-row"}),))


def test_a_report_is_synchronized_through_a_writable_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fsync = os.fsync
    written: list[int] = []

    def sync(descriptor: int) -> None:
        if stat.S_ISREG(os.fstat(descriptor).st_mode):
            written.append(os.write(descriptor, b""))
        fsync(descriptor)

    monkeypatch.setattr(reporting.os, "fsync", sync)
    ParquetReportSink(tmp_path / "report.parquet").emit((_row("metric", 1.0),))

    assert written == [0]
