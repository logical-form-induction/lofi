from typing import TYPE_CHECKING

from lofi.eval import (
    NO_ENUMERATION_BUDGET,
    Approximation,
    ConventionVector,
    MetricResult,
    Report,
    ReportIdentity,
    ReportRow,
    Resampling,
    report,
)
from lofi.run._ports import MetricSink, ReportSink
from lofi.run._sinks import NullSink

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lofi._types import JsonValue


class RecordingSink:
    def __init__(self) -> None:
        self.written: list[Report] = []

    @property
    def name(self) -> str:
        return "recording"

    def emit(self, report: Report) -> None:
        self.written.append(report)


class HalfASink:
    @property
    def name(self) -> str:
        return "half"


class RecordingMetricSink:
    def __init__(self) -> None:
        self.scalars: list[tuple[dict[str, float], int]] = []
        self.tables: list[tuple[str, tuple[dict[str, JsonValue], ...]]] = []
        self.closed = False

    def log_scalars(self, mapping: Mapping[str, float], step: int) -> None:
        self.scalars.append((dict(mapping), step))

    def log_table(self, name: str, rows: Sequence[Mapping[str, JsonValue]]) -> None:
        self.tables.append((name, tuple(dict(row) for row in rows)))

    def close(self) -> None:
        self.closed = True


class HalfAMetricSink:
    def log_scalars(self, mapping: Mapping[str, float], step: int) -> None:
        del mapping, step


class _Config:
    def model_dump(self) -> dict[str, JsonValue]:
        return {"run": "port-test"}


class _Grammar:
    fingerprint = "grammar:port-test"


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


def _report() -> Report:
    return report(
        (
            ReportRow(
                identity=_identity(),
                method="bracketing",
                metric="bracketing-f1",
                result=MetricResult(0.71, 40, Resampling("item")),
                approximation=Approximation(NO_ENUMERATION_BUDGET, residual=None),
                suite="grammar-comparison",
                task="bracketing",
                dataset="ccgbank",
                split="test",
                seed=3,
            ),
        )
    )


def test_a_sink_that_names_itself_and_emits_conforms() -> None:
    sink = RecordingSink()
    report = _report()

    assert isinstance(sink, ReportSink)

    sink.emit(report)

    assert sink.written == [report]
    assert sink.name == "recording"


def test_the_null_sink_conforms() -> None:
    sink: ReportSink = NullSink()

    sink.emit(_report())

    assert isinstance(sink, ReportSink)
    assert sink.name == "null"


def test_something_that_cannot_emit_is_not_a_sink() -> None:
    assert not isinstance(HalfASink(), ReportSink)


def test_one_report_reaches_several_sinks_unchanged() -> None:
    first = RecordingSink()
    second = RecordingSink()
    report = _report()

    for sink in (first, second):
        sink.emit(report)

    assert first.written[0] is report
    assert second.written[0] is report
    assert report.rows[0].value == 0.71


def test_a_metric_sink_is_driven_through_its_complete_protocol() -> None:
    sink = RecordingMetricSink()
    rows: tuple[dict[str, JsonValue], ...] = ({"metric": "loss", "value": 0.25},)

    assert isinstance(sink, MetricSink)

    sink.log_scalars({"loss": 0.25}, 4)
    sink.log_table("validation", rows)
    sink.close()

    assert sink.scalars == [({"loss": 0.25}, 4)]
    assert sink.tables == [("validation", rows)]
    assert sink.closed


def test_a_partial_metric_sink_does_not_conform() -> None:
    assert not isinstance(HalfAMetricSink(), MetricSink)
