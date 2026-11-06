import json
from typing import TYPE_CHECKING, Final, cast, override

import pytest

from lofi.eval._identity import ReportIdentity
from lofi.eval._inference import ENTAILMENT_ACCURACY_METRIC
from lofi.eval._ports import HYPOTHESIS_ONLY
from lofi.eval._report import (
    MissingControlError,
    PooledJudgment,
    Report,
    ReportRow,
    report,
)
from lofi.eval._residual import (
    NO_ENUMERATION_BUDGET,
    RESIDUAL_SOURCES,
    Approximation,
    Residual,
    ResidualSummary,
)
from lofi.eval._values import ConventionVector, MetricResult, Resampling
from lofi.spec import HeldOutParticipantPolicy
from lofi.stats import Interval

if TYPE_CHECKING:
    from collections.abc import Callable

    from lofi._types import JsonValue


class _Config:
    def model_dump(self) -> dict[str, JsonValue]:
        return {"run": "report-test"}


class _Grammar:
    fingerprint = "grammar:report-test"


def _report_identity() -> ReportIdentity:
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


class _Budget:
    def __init__(self, depth: int, beam: int, mass: float | None) -> None:
        self._depth = depth
        self._beam = beam
        self._mass = mass

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def beam(self) -> int:
        return self._beam

    @property
    def mass(self) -> float | None:
        return self._mass

    @override
    def __repr__(self) -> str:
        return f"_Budget({self._depth!r}, {self._beam!r}, {self._mass!r})"


_SENTENCES: Final = Resampling("sentence", 3)

_BUDGET: Final = _Budget(6, 20, 0.01)

_MEASURED: Final = Residual(
    {
        "beam-truncation": 0.02,
        "chart-cap": 0.0,
        "enumeration-budget": 0.5,
        "prover-timeout": 1.0,
    }
)


def _result(value: float | None = 0.61, n: int = 40) -> MetricResult:
    return MetricResult(
        value,
        n,
        _SENTENCES,
        components={"matched": 24.0},
        ci=None if value is None else Interval(0.55, 0.67, 0.95),
    )


def _row(
    residual: Residual | None,
    *,
    method: str = "ccg",
    metric: str = "unlabeled-bracketing-f1",
) -> ReportRow:
    return ReportRow(
        identity=_report_identity(),
        method=method,
        metric=metric,
        result=_result(),
        approximation=Approximation(_BUDGET, residual=residual),
    )


def test_a_row_refuses_an_approximation_that_is_not_one() -> None:
    with pytest.raises(TypeError, match="approximation is not one"):
        ReportRow(
            identity=_report_identity(),
            method="ccg",
            metric="f1",
            result=_result(),
            approximation=cast("Approximation", _BUDGET),
        )


def test_a_row_requires_a_real_identity() -> None:
    constructor = cast("Callable[..., object]", ReportRow)

    with pytest.raises(TypeError, match="identity"):
        constructor(
            method="ccg",
            metric="f1",
            result=_result(),
            approximation=Approximation(_BUDGET, residual=_MEASURED),
        )

    with pytest.raises(TypeError, match="identity is not one"):
        ReportRow(
            identity=cast("ReportIdentity", {}),
            method="ccg",
            metric="f1",
            result=_result(),
            approximation=Approximation(_BUDGET, residual=_MEASURED),
        )


def test_a_row_refuses_a_result_that_is_not_one() -> None:
    with pytest.raises(TypeError, match="result is not one"):
        ReportRow(
            identity=_report_identity(),
            method="ccg",
            metric="f1",
            result=cast("MetricResult", 0.61),
            approximation=Approximation(_BUDGET, residual=_MEASURED),
        )


def test_a_row_refuses_an_empty_method_or_metric() -> None:
    with pytest.raises(ValueError, match="method is empty"):
        _row(_MEASURED, method="")

    with pytest.raises(ValueError, match="metric is empty"):
        _row(_MEASURED, metric="")


def test_a_row_carries_its_freeze_into_its_identity_columns() -> None:
    row = _row(_MEASURED)
    spec = row.to_spec()

    assert row.freeze_id == "freeze/1"
    assert spec["freeze_id"] == "freeze/1"
    assert spec["config_hash"] == row.config_hash


def test_a_measured_row_writes_every_residual_column() -> None:
    spec = _row(_MEASURED).to_spec()

    assert spec["residual_measured"] is True
    assert spec["residual_beam-truncation"] == 0.02
    assert spec["residual_chart-cap"] == 0.0
    assert spec["residual_prover-timeout"] == 1.0
    assert spec["budget_depth"] == 6
    assert spec["budget_beam"] == 20
    assert spec["budget_mass"] == 0.01
    assert spec["value"] == 0.61
    assert spec["ci_low"] == 0.55
    assert spec["ci_level"] == 0.95


def test_an_unmeasured_row_serializes_as_null_and_never_as_zero() -> None:
    spec = _row(None).to_spec()

    assert spec["residual_measured"] is False

    for column in (
        "residual_beam-truncation",
        "residual_chart-cap",
        "residual_enumeration-budget",
        "residual_prover-timeout",
    ):
        assert spec[column] is None

    text = json.dumps(spec)

    assert '"residual_beam-truncation": null' in text


def test_a_partly_measured_row_keeps_a_measured_zero_apart_from_an_absence() -> None:
    partial = Residual({"chart-cap": 0.0})
    spec = _row(partial).to_spec()

    assert spec["residual_chart-cap"] == 0.0
    assert spec["residual_beam-truncation"] is None
    assert spec["residual_measured"] is False


def test_a_report_counts_the_rows_that_state_what_their_budget_cost() -> None:
    built = report([_row(_MEASURED), _row(None, method="cfg"), _row(_MEASURED)])

    assert built.residuals.rows == 3
    assert built.residuals.measured == 2
    assert built.residuals.unmeasured == 1
    assert built.residuals.by_source["prover-timeout"] == 1


def test_a_report_counts_a_partly_measured_row_against_the_sources_it_missed() -> None:
    built = report([_row(Residual({"beam-truncation": 0.1, "chart-cap": 0.0}))])

    assert built.residuals.measured == 0
    assert "beam-truncation" not in built.residuals.by_source
    assert "chart-cap" not in built.residuals.by_source
    assert built.residuals.by_source["enumeration-budget"] == 1
    assert built.residuals.by_source["prover-timeout"] == 1
    assert built.to_spec()["residuals"] == {
        "rows": 1,
        "measured": 0,
        "unmeasured": 1,
        "not_applicable": 0,
        "by_source": {
            "beam-truncation": 0,
            "chart-cap": 0,
            "enumeration-budget": 1,
            "prover-timeout": 1,
        },
    }


def test_two_summaries_making_one_claim_are_equal_however_they_were_built() -> None:
    stated = ResidualSummary(
        2, 1, dict.fromkeys(RESIDUAL_SOURCES, 0) | {"chart-cap": 1}, not_applicable=0
    )
    sparse = ResidualSummary(2, 1, {"chart-cap": 1}, not_applicable=0)

    assert stated == sparse
    assert hash(stated) == hash(sparse)
    assert dict(stated.by_source) == dict(sparse.by_source)


def test_a_report_writes_its_rows_and_its_summary() -> None:
    built = report([_row(_MEASURED), _row(None, method="cfg")])
    spec = built.to_spec()
    rows = spec["rows"]
    residuals = spec["residuals"]

    assert isinstance(rows, list)
    assert isinstance(residuals, dict)
    assert len(rows) == 2
    assert residuals["measured"] == 1
    assert residuals["unmeasured"] == 1
    assert json.loads(json.dumps(spec)) == spec


def test_a_report_refuses_a_summary_that_counts_other_rows() -> None:
    with pytest.raises(ValueError, match="summarizes"):
        Report([_row(_MEASURED)], ResidualSummary(4, 4, {}, not_applicable=0))


def test_a_report_refuses_a_summary_its_own_rows_deny() -> None:
    with pytest.raises(ValueError, match="rows measuring every source"):
        Report([_row(None)], ResidualSummary(1, 1, {}, not_applicable=0))

    with pytest.raises(ValueError, match="rows measuring every source"):
        Report(
            [_row(_MEASURED)],
            ResidualSummary(1, 0, dict.fromkeys(RESIDUAL_SOURCES, 1), not_applicable=0),
        )

    accepted = Report([_row(_MEASURED)], ResidualSummary(1, 1, {}, not_applicable=0))

    assert accepted.rows == report([_row(_MEASURED)]).rows
    assert accepted.residuals.measured == 1


def test_a_report_refuses_a_summary_that_is_not_one() -> None:
    with pytest.raises(TypeError, match="residual summary is not one"):
        Report([_row(_MEASURED)], cast("ResidualSummary", 3))


def test_a_report_refuses_a_row_that_is_not_one() -> None:
    with pytest.raises(TypeError, match="report row is not one"):
        report([_row(_MEASURED), cast("ReportRow", 0.5)])


def test_two_reports_over_the_same_rows_are_equal_and_hash_alike() -> None:
    first = report([_row(_MEASURED)])
    second = report([_row(_MEASURED)])

    assert first == second
    assert hash(first) == hash(second)
    assert first != report([_row(None)])
    assert first.rows[0] != _row(None)
    assert first != "a report"


def test_a_row_repr_names_the_fields_that_rebuild_it() -> None:
    text = repr(_row(None))

    assert text.startswith("ReportRow(identity=ReportIdentity(")
    assert "approximation=Approximation(" in text
    assert repr(report([_row(None)])).startswith("Report([ReportRow(")


def test_a_row_and_report_are_frozen() -> None:
    row = _row(_MEASURED)
    built = report([row])

    with pytest.raises(AttributeError):
        row.__setattr__("_metric", "changed")

    with pytest.raises(AttributeError):
        built.__setattr__("_rows", ())


def test_a_row_refuses_partial_contexts() -> None:
    with pytest.raises(ValueError, match="incomplete experiment context"):
        ReportRow(
            identity=_report_identity(),
            method="ccg",
            metric="f1",
            result=_result(),
            approximation=Approximation(_BUDGET, residual=_MEASURED),
            split="validation",
        )

    with pytest.raises(ValueError, match="incomplete judgment context"):
        ReportRow(
            identity=_report_identity(),
            method="ccg",
            metric="f1",
            result=_result(),
            approximation=Approximation(_BUDGET, residual=_MEASURED),
            fold=0,
        )

    with pytest.raises(ValueError, match="without judgment context"):
        ReportRow(
            identity=_report_identity(),
            method="ccg",
            metric="f1",
            result=_result(),
            approximation=Approximation(_BUDGET, residual=_MEASURED),
            held_out_convention="marginal",
        )


def test_a_row_with_no_value_still_carries_its_approximation() -> None:
    empty = ReportRow(
        identity=_report_identity(),
        method="ccg",
        metric="f1",
        result=MetricResult(None, 0, _SENTENCES),
        approximation=Approximation(_BUDGET, residual=_MEASURED),
    )
    spec = empty.to_spec()

    assert spec["value"] is None
    assert spec["ci_low"] is None
    assert spec["residual_measured"] is True
    assert empty.is_measured


# ---------------------------------------------------------------------------
# the hypothesis-only control an inference accuracy has to be read against
# ---------------------------------------------------------------------------


class _SecondConfig:
    def model_dump(self) -> dict[str, JsonValue]:
        return {"run": "report-test-second"}


def _second_identity() -> ReportIdentity:
    return ReportIdentity(
        _SecondConfig(),
        _Grammar(),
        ConventionVector(
            punctuation="strip",
            unary_chains="collapse",
            trivial_spans="drop",
            minimum_length=3,
            aggregation="sentence",
            coverage="zero",
            gold_derivation="canonical",
            self_arcs="forbid",
        ),
        freeze_id="freeze/1",
    )


def _inference_row(
    method: str,
    *,
    metric: str = ENTAILMENT_ACCURACY_METRIC,
    identity: ReportIdentity | None = None,
    required_control: str | None = HYPOTHESIS_ONLY,
) -> ReportRow:
    return ReportRow(
        identity=_report_identity() if identity is None else identity,
        method=method,
        metric=metric,
        result=_result(),
        approximation=Approximation(_BUDGET, residual=_MEASURED),
        required_control=required_control,
    )


def test_a_report_refuses_an_inference_accuracy_with_no_control_beside_it() -> None:
    with pytest.raises(MissingControlError, match="required control"):
        report([_inference_row("ccg")])


def test_a_report_admits_an_inference_accuracy_once_the_control_is_present() -> None:
    built = report([_inference_row("ccg"), _inference_row(HYPOTHESIS_ONLY)])

    assert {row.method for row in built.rows} == {"ccg", HYPOTHESIS_ONLY}


def test_the_control_is_required_for_the_label_accuracy_of_any_class_count() -> None:
    with pytest.raises(MissingControlError, match="inference-label-3-way"):
        report([_inference_row("ccg", metric="inference-label-3-way")])

    admitted = report(
        [
            _inference_row("ccg", metric="inference-label-3-way"),
            _inference_row(HYPOTHESIS_ONLY, metric="inference-label-3-way"),
        ]
    )

    assert len(admitted.rows) == 2


def test_a_control_on_one_metric_does_not_cover_another() -> None:
    with pytest.raises(MissingControlError, match="inference-label-2-way"):
        report(
            [
                _inference_row("ccg", metric="inference-label-2-way"),
                _inference_row(HYPOTHESIS_ONLY),
                _inference_row("ccg"),
            ]
        )


def test_a_control_in_one_run_does_not_cover_another_run() -> None:
    with pytest.raises(MissingControlError, match="under run"):
        report(
            [
                _inference_row("ccg"),
                _inference_row(HYPOTHESIS_ONLY),
                _inference_row("ccg", identity=_second_identity()),
            ]
        )

    covered = report(
        [
            _inference_row("ccg"),
            _inference_row(HYPOTHESIS_ONLY),
            _inference_row("ccg", identity=_second_identity()),
            _inference_row(HYPOTHESIS_ONLY, identity=_second_identity()),
        ]
    )

    assert len(covered.rows) == 4


def test_a_report_over_a_metric_with_no_control_capability_needs_no_control() -> None:
    assert len(report([_row(_MEASURED)]).rows) == 1
    assert (
        len(
            report(
                [
                    _inference_row(
                        "ccg",
                        metric="uncontrolled-metric",
                        required_control=None,
                    )
                ]
            ).rows
        )
        == 1
    )


def _unbudgeted_row() -> ReportRow:
    return ReportRow(
        identity=_report_identity(),
        method="posterior",
        metric="max-rhat",
        result=_result(),
        approximation=Approximation(NO_ENUMERATION_BUDGET, residual=None),
    )


def test_a_report_counts_rows_no_budget_applies_to_apart() -> None:
    built = report([_row(_MEASURED), _unbudgeted_row()])
    residuals = built.to_spec()["residuals"]

    assert isinstance(residuals, dict)
    assert (
        residuals["measured"],
        residuals["unmeasured"],
        residuals["not_applicable"],
    ) == (1, 0, 1)


def test_a_report_refuses_a_summary_misstating_its_rows_without_a_budget() -> None:
    with pytest.raises(ValueError, match="no budget applies"):
        Report(
            [_row(_MEASURED), _unbudgeted_row()],
            ResidualSummary(2, 1, {}, not_applicable=0),
        )


# ---------------------------------------------------------------------------
# statistics pooled over the held-out folds of a cross-fitted judgment fit
# ---------------------------------------------------------------------------


def _pooled(
    *,
    rungs: tuple[str, ...] = ("two_scalar", "one_scalar"),
    specification_hashes: tuple[str, ...] = ("spec-two", "spec-one"),
    folds: tuple[int, ...] = (2, 0, 1),
) -> PooledJudgment:
    return PooledJudgment(
        rungs=rungs, specification_hashes=specification_hashes, folds=folds
    )


def _pooled_row(pooled: PooledJudgment) -> ReportRow:
    return ReportRow(
        identity=_report_identity(),
        method=pooled.rung,
        metric="predictive_density_improvement",
        result=_result(),
        approximation=Approximation(NO_ENUMERATION_BUDGET, residual=None),
        suite="judgment",
        task="acceptability",
        dataset="megaacceptability",
        split="test",
        seed=0,
        pooled=pooled,
        held_out_convention="marginal",
    )


def test_a_pooled_judgment_records_its_folds_and_rungs() -> None:
    pooled = _pooled()

    assert pooled.rungs == ("two_scalar", "one_scalar")
    assert pooled.specification_hashes == ("spec-two", "spec-one")
    assert pooled.folds == (0, 1, 2)
    assert pooled.rung == "two_scalar"


def test_a_pooled_judgment_digest_names_exactly_its_folds_and_rungs() -> None:
    digest = _pooled().specification_hash

    assert digest == _pooled(folds=(0, 1, 2)).specification_hash
    assert digest != _pooled(folds=(0, 1)).specification_hash
    assert (
        digest
        != _pooled(
            rungs=("one_scalar", "two_scalar"),
            specification_hashes=("spec-one", "spec-two"),
        ).specification_hash
    )
    assert (
        digest
        != _pooled(
            specification_hashes=("spec-two", "spec-one-revised")
        ).specification_hash
    )
    assert digest not in {"spec-two", "spec-one"}


@pytest.mark.parametrize(
    ("rungs", "specification_hashes", "folds", "match"),
    [
        ((), (), (0, 1), "pools no rung"),
        (("one",), ("a", "b"), (0, 1), "2 specification hashes for 1 rungs"),
        (("one", "one"), ("a", "b"), (0, 1), "repeats a rung"),
        (("",), ("a",), (0, 1), "rung"),
        (("one",), ("",), (0, 1), "specification hash"),
        (("one",), ("a",), (0,), "fewer than two folds"),
        (("one",), ("a",), (0, 0), "repeats a fold"),
        (("one",), ("a",), (0, -1), "fold"),
    ],
)
def test_a_pooled_judgment_refuses_a_malformed_key(
    rungs: tuple[str, ...],
    specification_hashes: tuple[str, ...],
    folds: tuple[int, ...],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        PooledJudgment(
            rungs=rungs, specification_hashes=specification_hashes, folds=folds
        )


def test_a_pooled_row_carries_a_complete_judgment_identity() -> None:
    pooled = _pooled()
    row = _pooled_row(pooled)
    spec = row.to_spec()

    assert row.pooled == pooled
    assert (row.specification_hash, row.fold, row.rung) == (
        pooled.specification_hash,
        None,
        "two_scalar",
    )
    assert (spec["specification_hash"], spec["fold"], spec["rung"]) == (
        pooled.specification_hash,
        None,
        "two_scalar",
    )
    assert spec["held_out_convention"] == "marginal"


@pytest.mark.parametrize(
    "convention", [policy.value for policy in HeldOutParticipantPolicy]
)
def test_a_judgment_row_admits_each_declared_held_out_convention(
    convention: str,
) -> None:
    pooled = _pooled()
    row = ReportRow(
        identity=_report_identity(),
        method=pooled.rung,
        metric="predictive_density_improvement",
        result=_result(),
        approximation=Approximation(NO_ENUMERATION_BUDGET, residual=None),
        suite="judgment",
        task="acceptability",
        dataset="megaacceptability",
        split="test",
        seed=0,
        pooled=pooled,
        held_out_convention=convention,
    )

    assert row.held_out_convention == convention


def test_a_judgment_row_refuses_an_undeclared_held_out_convention() -> None:
    pooled = _pooled()

    with pytest.raises(ValueError, match="unknown held-out convention"):
        ReportRow(
            identity=_report_identity(),
            method=pooled.rung,
            metric="predictive_density_improvement",
            result=_result(),
            approximation=Approximation(NO_ENUMERATION_BUDGET, residual=None),
            suite="judgment",
            task="acceptability",
            dataset="megaacceptability",
            split="test",
            seed=0,
            pooled=pooled,
            held_out_convention="posterior",
        )


def test_a_pooled_row_differs_from_one_pooling_other_folds() -> None:
    row = _pooled_row(_pooled())

    assert row == _pooled_row(_pooled())
    assert hash(row) == hash(_pooled_row(_pooled()))
    assert row != _pooled_row(_pooled(folds=(0, 1)))
    assert "pooled=PooledJudgment(" in repr(row)


def test_a_pooled_row_refuses_a_second_judgment_context() -> None:
    with pytest.raises(ValueError, match="both a held-out fold and pooled folds"):
        ReportRow(
            identity=_report_identity(),
            method="two_scalar",
            metric="predictive_density_improvement",
            result=_result(),
            approximation=Approximation(NO_ENUMERATION_BUDGET, residual=None),
            rung="two_scalar",
            pooled=_pooled(),
        )


def test_a_row_refuses_a_pooled_key_that_is_not_one() -> None:
    with pytest.raises(TypeError, match="pooled judgment is not one"):
        ReportRow(
            identity=_report_identity(),
            method="two_scalar",
            metric="predictive_density_improvement",
            result=_result(),
            approximation=Approximation(NO_ENUMERATION_BUDGET, residual=None),
            pooled=cast("PooledJudgment", (0, 1)),
        )
