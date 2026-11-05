"""Exercise known-grammar logical-form induction through public boundaries."""

from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from lofi.run import (
    BaselineExecutor,
    ExecutionEnvironment,
    ExperimentSeed,
    ExperimentStore,
    ManifestStatus,
    PointExecutor,
    RegisteredBaselines,
    RunStore,
    SerialLauncher,
    expand_experiment,
    read_experiment,
    read_rows,
)
from lofi.run.operations import TRAINING_OPERATIONS
from lofi.spec import RunSpecCodec

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi._types import JsonValue
    from lofi.run import SweepPoint


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = Path(__file__).resolve().parent

FORM_ROWS = {
    "logical-form-normal-form",
    "many-to-one-accuracy",
    "random-relabeling-floor",
    "fiber-relative-recovery",
    "fixed-vocabulary-accuracy",
    "fiber-floor",
    "fiber-size",
}
CATEGORY_ROWS = {
    "supertag-accuracy",
    "dependency:labeled-directed:zero:micro-f-score",
    "lexical-type-f1",
    "primitive-one-to-one-accuracy",
    "primitive-random-relabeling-floor",
}
BASELINE_ROWS = {
    "logical-form-normal-form",
    "many-to-one-accuracy",
    "random-relabeling-floor",
    "inference-accuracy",
    "inference-coverage",
    "prover-calls",
    "prover-memo-hit-rate",
    "prover-cpu-seconds",
    "prover-abstention-rate",
}


def _check(condition: object, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _mapping(value: JsonValue, label: str) -> Mapping[str, JsonValue]:
    if not isinstance(value, dict):
        msg = f"{label} is not a mapping"
        raise TypeError(msg)
    return value


def _at(document: Mapping[str, JsonValue], path: str) -> JsonValue:
    value: JsonValue = dict(document)
    for part in path.split("."):
        value = _mapping(value, path)[part]
    return value


def _treatment(point: SweepPoint) -> str:
    grounding = _mapping(
        _mapping(point.config["model"], "model")["grounding"],
        "model.grounding",
    )
    if grounding["kind"] == "expected-embedding":
        _check(
            grounding["vectors"] == "sentence",
            "the NLI arm is not the standard sentence-vector architecture",
        )
        return "nli-baseline"
    scorer = _mapping(grounding["scorer"], "model.grounding.scorer")
    _check(scorer["kind"] == "prover", "the induced arm has no theorem prover")
    return "induced-with-prover"


def test_documents_load_and_expand_the_exact_known_grammar_design() -> None:
    """The neural and replayed-baseline documents retain every declared cell."""
    induction = read_experiment(EXPERIMENTS / "induction.yaml")
    reduced = read_experiment(EXPERIMENTS / "reduced-freeze.yaml")
    linc = read_experiment(EXPERIMENTS / "lm-prover.yaml")
    logic_lm = read_experiment(EXPERIMENTS / "logic-lm.yaml")
    reduced_linc = read_experiment(EXPERIMENTS / "lm-prover-reduced.yaml")
    reduced_logic_lm = read_experiment(EXPERIMENTS / "logic-lm-reduced.yaml")

    induction_points = expand_experiment(induction, ROOT)
    reduced_points = expand_experiment(reduced, ROOT)
    linc_points = expand_experiment(linc, ROOT)
    logic_lm_points = expand_experiment(logic_lm, ROOT)
    reduced_linc_points = expand_experiment(reduced_linc, ROOT)
    reduced_logic_lm_points = expand_experiment(reduced_logic_lm, ROOT)

    _check(len(induction_points) == 16, "the production neural grid is not 16 points")
    _check(len(reduced_points) == 2, "the reduced neural grid is not two points")
    _check(len(linc_points) == 4, "the LINC baseline omits a seed")
    _check(len(logic_lm_points) == 4, "the Logic-LM baseline omits a seed")
    _check(len(reduced_linc_points) == 1, "the reduced LINC fit is not one point")
    _check(
        len(reduced_logic_lm_points) == 1,
        "the reduced Logic-LM fit is not one point",
    )
    _check(
        {_treatment(point) for point in induction_points}
        == {"induced-with-prover", "nli-baseline"},
        "the production grid omits an inference treatment",
    )
    _check(
        {
            cast("int", _at(point.config, "model.capacity.target"))
            for point in induction_points
        }
        == {65_536, 262_144},
        "the production grid omits a matched parameter budget",
    )
    for point in (*induction_points, *reduced_points):
        _check(
            _at(point.config, "data.datasets.train.dataset.grammar.name") == "l1-slash",
            "a neural point changed the known grammar",
        )
        _check(
            _at(
                point.config,
                "data.datasets.train.dataset.source.language.logic.prover.kind",
            )
            == "truth-table",
            "a neural point changed the known logic's prover",
        )
        _check(
            _at(point.config, "data.datasets.train.constraint_source") is None,
            "a neural point exposes lexical logical forms as constraints",
        )
    _check(
        {(row.evaluator, row.metric) for row in induction.rows}
        == {
            *(("logical-forms", metric) for metric in FORM_ROWS),
            *(("category-side", metric) for metric in CATEGORY_ROWS),
        },
        "the production document omits a recovery row",
    )
    _check(
        _at(linc_points[0].config, "system.pipeline.kind") == "linc"
        and _at(logic_lm_points[0].config, "system.pipeline.kind") == "logic-lm",
        "the two baseline documents do not select their named pipelines",
    )
    _check(
        _at(linc_points[0].config, "system.data_source")
        == _at(logic_lm_points[0].config, "system.data_source"),
        "the LM-plus-prover pipelines do not score the same population",
    )
    for point in (*linc_points, *logic_lm_points):
        _check(
            _at(point.config, "system.language.name") == "l1-propositional",
            "an LM-plus-prover point changed the known logical language",
        )
        _check(
            _at(point.config, "system.language.logic.prover.kind") == "truth-table",
            "an LM-plus-prover point changed the known logic's prover",
        )


@pytest.mark.integration
@pytest.mark.parametrize(
    ("point_index", "expected_treatment"),
    [(0, "induced-with-prover"), (1, "nli-baseline")],
)
def test_reduced_neural_freeze_reports_fiber_and_category_recovery(
    tmp_path: Path,
    point_index: int,
    expected_treatment: str,
) -> None:
    """Both reduced neural treatments train and emit computed recovery rows."""
    experiment = read_experiment(EXPERIMENTS / "reduced-freeze.yaml")
    points = expand_experiment(experiment, ROOT)
    point = points[point_index]
    _check(_treatment(point) == expected_treatment, "the reduced treatment moved")
    store = RunStore(
        tmp_path / "runs",
        freeze_id=None,
        seed=ExperimentSeed(experiment.seed_policy.root),
    )
    executor = PointExecutor(
        store,
        codec=RunSpecCodec(),
        operations=TRAINING_OPERATIONS,
        infrastructure_attempts=1,
        failure_policy=experiment.failure,
    )

    handles = SerialLauncher(store, execute=executor).submit((point,))
    _check(len(handles) == 1, "the serial launcher omitted the reduced point")
    _check(store.is_completed(point), "the reduced neural point did not complete")
    recorded = read_rows(store.directory(point))
    if recorded is None:
        msg = "the completed reduced neural point emitted no rows"
        raise AssertionError(msg)
    rows = {
        cast("str", row["metric"]): row
        for row in recorded.rows
        if row["task"] in {"logical-forms", "category-side"}
    }
    required = FORM_ROWS | CATEGORY_ROWS
    _check(required <= set(rows), "the reduced neural point omitted recovery rows")
    for metric in required:
        value = rows[metric]["value"]
        _check(
            isinstance(value, int | float) and isfinite(float(value)),
            f"the computed {metric} row has no finite value",
        )


@pytest.mark.integration
@pytest.mark.parametrize(
    ("document", "pipeline"),
    [("lm-prover-reduced.yaml", "linc"), ("logic-lm-reduced.yaml", "logic-lm")],
)
def test_reduced_lm_plus_prover_baselines_replay_and_score(
    tmp_path: Path,
    document: str,
    pipeline: str,
) -> None:
    """Both raw generation traces are proved and scored at the baseline boundary."""
    experiment = read_experiment(EXPERIMENTS / document)
    [point] = expand_experiment(experiment, ROOT)
    store = ExperimentStore(tmp_path / "runs", experiment)
    executor = BaselineExecutor(
        store,
        operation=RegisteredBaselines(),
        environment=ExecutionEnvironment({"lofi": "test"}, "test", "cpu"),
        infrastructure_attempts=1,
    )

    outcome = executor(point)
    recorded = read_rows(store.directory(point))
    _check(
        (outcome.status, outcome.cached) == (ManifestStatus.COMPLETED, False),
        f"the {pipeline} reduced replay did not complete",
    )
    if recorded is None:
        msg = f"the {pipeline} reduced replay emitted no rows"
        raise AssertionError(msg)
    rows = {cast("str", row["metric"]): row for row in recorded.rows}
    _check(set(rows) == BASELINE_ROWS, f"the {pipeline} replay omitted a metric")
    for metric, row in rows.items():
        value = row["value"]
        _check(
            isinstance(value, int | float) and isfinite(float(value)),
            f"the computed {pipeline} {metric} row has no finite value",
        )
        _check(cast("int", row["n"]) > 0, f"the {pipeline} {metric} row has no sample")
