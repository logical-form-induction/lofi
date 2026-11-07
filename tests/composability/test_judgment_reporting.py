from typing import TYPE_CHECKING

import pyarrow.parquet as pq
import torch

from lofi.eval import ConventionVector, ReportIdentity, ReportRow
from lofi.judgment import (
    Measurement,
    constraint_cost,
    posterior_variance_row,
    predictive_density_row,
    pseudo_r_squared,
    scale_profile,
)
from lofi.report import report_renderer
from lofi.run import judgment_report_row
from lofi.run.reporting import (
    ParquetReportSink,
    ReportSelector,
    render_table,
    select_required_rows,
)

if TYPE_CHECKING:
    from pathlib import Path

    from lofi._types import JsonValue


class _Config:
    def model_dump(self) -> dict[str, JsonValue]:
        return {"run": "judgment-report-composition"}


class _Grammar:
    fingerprint = "grammar:judgment-report-composition"


def _identity() -> ReportIdentity:
    return ReportIdentity(
        _Config(),
        _Grammar(),
        ConventionVector(
            punctuation="strip",
            unary_chains="collapse",
            trivial_spans="drop",
            minimum_length=0,
            aggregation="sentence",
            coverage="zero",
            self_arcs="forbid",
            gold_derivation="canonical",
        ),
        freeze_id="freeze/8",
    )


def _lower(metric: str, measurement: Measurement) -> ReportRow:
    return judgment_report_row(
        measurement,
        _identity(),
        suite="judgment-models",
        task="held-out-fit",
        dataset="mega-veridicality",
        split="test",
        metric=metric,
        seed=29,
        normal_form=None,
        specification_hash="specification-a1",
        fold=0,
        rung="verb_by_frame",
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
        method="verb_by_frame",
        metric=row.metric,
        seed=29,
        specification_hash="specification-a1",
        fold=0,
        rung="verb_by_frame",
        held_out_convention="marginal",
    )


def test_judgment_measurements_flow_through_one_file_driven_report(
    tmp_path: Path,
) -> None:
    density = predictive_density_row(
        "verb_by_frame", torch.log(torch.tensor((0.5, 0.25, 0.125)))
    ).predictive_density
    variance = posterior_variance_row(
        "verb_by_frame",
        "participant",
        torch.tensor((0.1, 0.2, 0.3, 0.4)),
    ).posterior_variance
    pseudo = pseudo_r_squared(
        "verb_by_frame",
        Measurement(value=-2.0, n=3, unit="response"),
        reference="intercept_only",
        references={"intercept_only": Measurement(value=-4.0, n=3, unit="response")},
    ).pseudo_r_squared
    rows = (
        _lower("predictive_density", density),
        _lower("posterior_variance", variance),
        _lower("pseudo_r_squared", pseudo),
    )
    path = tmp_path / "judgments.parquet"
    ParquetReportSink(path).emit(rows)
    requested = (_selector(rows[2]), _selector(rows[0]), _selector(rows[1]))
    selected = select_required_rows(path, requested)
    rendered = render_table(
        report_renderer("markdown"),
        selected,
        columns=("metric", "value", "n", "resampling_unit"),
        labels=("Metric", "Value", "Count", "Unit"),
        precision=None,
    )
    stored = pq.read_table(path)

    assert tuple(row["metric"] for row in selected) == (
        "pseudo_r_squared",
        "predictive_density",
        "posterior_variance",
    )
    assert stored.column("metric").to_pylist() == [row.metric for row in rows]

    for row in selected:
        assert str(row["metric"]).encode() in rendered
        assert str(row["value"]).encode() in rendered
        assert str(row["n"]).encode() in rendered
        assert str(row["resampling_unit"]).encode() in rendered


def test_computed_identification_results_flow_through_the_generic_report(
    tmp_path: Path,
) -> None:
    profile = scale_profile(
        (0.5, 1.0, 2.0),
        (-3.0, 0.0, -2.0),
        minimum_tail_drop=2.0,
    )
    cost = constraint_cost(
        Measurement(value=-3.0, n=20, unit="response"),
        Measurement(value=-2.5, n=20, unit="response"),
    )
    measurements = (
        ("scale_profile_peak", profile.peak_measurement()),
        ("scale_profile_lower_tail_drop", profile.lower_tail_measurement()),
        ("scale_profile_upper_tail_drop", profile.upper_tail_measurement()),
        ("scale_constraint_cost", cost.as_measurement()),
    )
    rows = tuple(_lower(metric, measurement) for metric, measurement in measurements)
    path = tmp_path / "scale-identification.parquet"
    ParquetReportSink(path).emit(rows)
    stored = pq.read_table(path)

    assert stored.column("metric").to_pylist() == [metric for metric, _ in measurements]
    assert stored.column("value").to_pylist() == [0.0, 3.0, 2.0, 0.5]
    assert stored.column("n").to_pylist() == [3, 3, 3, 20]
    assert stored.column("resampling_unit").to_pylist() == [
        "scale_factor",
        "scale_factor",
        "scale_factor",
        "response",
    ]
