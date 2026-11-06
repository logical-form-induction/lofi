import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from tests.conformance.report_sink import ReportSinkConformance, ReportSinkFixture

from lofi.eval import (
    NO_ENUMERATION_BUDGET,
    Approximation,
    ConventionVector,
    DeclaredBudget,
    EvaluationArtifact,
    MetricResult,
    Report,
    ReportIdentity,
    ReportRow,
    Resampling,
    Residual,
    report,
)
from lofi.run._sinks import (
    ARTIFACT_DIRECTORY,
    HELD_OUT_TABLE,
    NullSink,
    PointReportSink,
    log_held_out_rows,
    write_artifacts,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lofi._types import JsonValue
    from lofi.run._ports import MetricSink


class _Config:
    def model_dump(self) -> dict[str, JsonValue]:
        return {"run": "sink-test"}


class _Grammar:
    fingerprint = "grammar:sink-test"


def _identity() -> ReportIdentity:
    return ReportIdentity(
        _Config(),
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
        freeze_id="freeze/1",
    )


def _report(
    metric: str = "bracketing-f1",
    approximation: Approximation | None = None,
) -> Report:
    return report(
        (
            ReportRow(
                identity=_identity(),
                method="bracketing",
                metric=metric,
                result=MetricResult(0.71, 40, Resampling("item")),
                approximation=(
                    Approximation(NO_ENUMERATION_BUDGET, residual=None)
                    if approximation is None
                    else approximation
                ),
                suite="grammar-comparison",
                task="bracketing",
                dataset="ccgbank",
                split="dev",
                seed=3,
            ),
        )
    )


class TestNullReportSink(ReportSinkConformance):
    @pytest.fixture
    def report_sink_fixture(self) -> ReportSinkFixture[Report]:
        sink = NullSink()

        return ReportSinkFixture(
            name=sink.name,
            emit=sink.emit,
            read=lambda: (),
            first=_report(),
            second=_report("coverage"),
            retains=False,
        )


def test_the_null_sink_accepts_a_report_and_leaves_it_alone() -> None:
    report = _report()

    assert NullSink().emit(report) is None
    assert report == _report()


def test_the_null_sink_names_itself_and_takes_report_after_report() -> None:
    sink = NullSink()

    sink.emit(_report())
    sink.emit(_report())

    assert sink.name == "null"
    assert repr(sink) == "NullSink()"


def test_the_null_sink_accepts_the_complete_metric_protocol() -> None:
    sink = cast("MetricSink", NullSink())
    scalars = {"loss/first": 0.25, "loss/second": 0.75}
    rows: tuple[dict[str, JsonValue], ...] = (
        {"split": "dev", "score": 0.5},
        {"split": "test", "score": 0.25},
    )

    assert sink.log_scalars(scalars, 3) is None
    assert sink.log_table("scores", rows) is None
    assert sink.close() is None
    assert scalars == {"loss/first": 0.25, "loss/second": 0.75}
    assert rows[0] == {"split": "dev", "score": 0.5}


def test_building_the_null_sink_imports_no_telemetry_backend() -> None:
    script = """
import sys

for name in ("wandb", "mlflow", "trackio"):
    sys.modules[name] = None

from lofi.run._sinks import NullSink

NullSink()
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


def test_discovery_and_the_empty_default_import_no_telemetry_backend() -> None:
    script = """
import sys

for name in ("wandb", "mlflow", "trackio"):
    sys.modules[name] = None

from lofi.registry import Registry
from lofi.run import SinkContext, resolve_metric_sinks
resolve_metric_sinks({}, SinkContext("point"))
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


def test_a_point_report_sink_serializes_each_row_with_its_approximation() -> None:
    sink = PointReportSink("0123abcd")
    measured = Approximation(
        DeclaredBudget(6, 20, mass=None),
        residual=Residual(
            {
                "beam-truncation": 0.1,
                "chart-cap": 0.0,
                "enumeration-budget": 0.0,
                "prover-timeout": 0.0,
            }
        ),
    )
    unmeasured = Approximation(DeclaredBudget(5, 8, mass=1.0), residual=None)

    sink.emit(_report(approximation=measured))
    sink.emit(_report("coverage", unmeasured))

    first, second = sink.rows
    assert first == {
        **_report().rows[0].to_spec(),
        **measured.to_spec(),
        "point_id": "0123abcd",
    }
    assert (second["metric"], second["budget_mass"], second["residual_measured"]) == (
        "coverage",
        1.0,
        False,
    )
    assert second["residual_beam-truncation"] is None
    assert repr(sink) == "PointReportSink('0123abcd', rows=2)"


class _TableSink:
    def __init__(self) -> None:
        self.tables: list[tuple[str, tuple[Mapping[str, JsonValue], ...]]] = []

    def log_scalars(self, mapping: Mapping[str, float], step: int) -> None:
        del mapping, step

    def log_table(self, name: str, rows: Sequence[Mapping[str, JsonValue]]) -> None:
        self.tables.append((name, tuple(rows)))

    def close(self) -> None:
        return None


def test_held_out_rows_reach_every_declared_metric_sink() -> None:
    sink = PointReportSink("0123abcd")
    sink.emit(_report())
    sink.emit(_report("coverage"))
    first, second = _TableSink(), _TableSink()

    log_held_out_rows((first, second), sink.rows)

    assert first.tables == second.tables == [(HELD_OUT_TABLE, sink.rows)]


def test_a_point_report_sink_persists_role_scoped_artifacts_atomically(
    tmp_path: Path,
) -> None:
    sink = PointReportSink("0123abcd")
    artifact = EvaluationArtifact("categories.json", b'{"digest":"abc"}\n')

    sink.emit_artifact("expert", artifact)
    (written,) = write_artifacts(tmp_path, sink.artifacts)

    assert written == tmp_path / ARTIFACT_DIRECTORY / "expert" / "categories.json"
    assert written.read_bytes() == artifact.content

    with pytest.raises(ValueError, match="emitted twice"):
        sink.emit_artifact("expert", artifact)
