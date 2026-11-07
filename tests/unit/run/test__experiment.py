import itertools
import json
import os
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast
from uuid import UUID

import pytest
from bead.items.item import Item
from tests.unit.judgment.test__spec import a_fit_document
from tests.unit.run.conftest import a_document

from lofi import judgment
from lofi.eval import read_preregistration
from lofi.judgment import (
    JudgmentFitSpec,
    PredictedJudgmentSource,
)
from lofi.lf import InferenceLabel
from lofi.report.latex import LATEX
from lofi.run import (
    FIT_MANIFEST_NAME,
    MANIFEST_NAME,
    UNPRICED,
    Amendment,
    Axis,
    ComputeEstimate,
    Experiment,
    ExperimentError,
    ExperimentSeed,
    GridSweepExpander,
    ReportedRow,
    ReportSpec,
    Selection,
    Sweep,
    SweepPoint,
    amend_experiment,
    amendment_ledger,
    composer,
    execute_predicted_judgments,
    expand_experiment,
    point_id,
    read_experiment,
    sweep_id,
)
from lofi.run._cost_model_builtins import (
    LengthBatchCostModel,
    StackBootstrapCostModel,
)
from lofi.spec import OptimizerSpec, config_search_path

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi._types import JsonValue
    from lofi.judgment import (
        InferenceText,
        TrainedPointPredictor,
    )
    from lofi.spec.judgment_sources import PredictedJudgmentSourceSettings

PREREGISTRATION: dict[str, JsonValue] = {
    "grid": {"trainer.epochs": [1, 2]},
    "hypotheses": [
        {"name": "recovery", "left": "induced", "right": "gold", "family": "primary"},
        {
            "name": "parity",
            "left": "small",
            "right": "large",
            "family": "equivalence",
            "margin": 0.5,
        },
    ],
}

RECOVERY: dict[str, JsonValue] = {"evaluator": "held-out", "metric": "recovery"}
PARITY: dict[str, JsonValue] = {"evaluator": "held-out", "metric": "parity"}

SWEEP: dict[str, JsonValue] = {
    "grid": [{"path": "trainer.epochs", "values": [1, 2]}],
    "seeds": [0, 1, 2],
    "expected_points": 6,
}


def an_experiment_document(*, freeze_id: str = "freeze/1") -> dict[str, JsonValue]:
    """Return a complete experiment document over the launcher run document."""
    return {
        "role": "headline",
        "freeze_id": freeze_id,
        "config": {"kind": "run", "path": "run.json"},
        "launch": {"kind": "serial"},
        "failure": {"numerical": "final", "out_of_memory": "retry"},
        "sweep": SWEEP,
        "seed_policy": {"root": 7},
        "rows": [RECOVERY, PARITY],
        "primary": RECOVERY,
        "preregistration": "preregistration.json",
        "compute": {
            "kind": "estimate",
            "p50_hours": 12.5,
            "p90_hours": 20,
            "cost_model": "price-v3",
        },
        "contraction": {
            "rungs": [
                {"kind": "unpreregistered-cells"},
                {"kind": "seeds", "floor": 3},
                {"kind": "fallback"},
            ],
            "factorial": "two epoch counts, fully crossed",
        },
        "stop_rules": [
            {
                "name": "numerical failure",
                "when": {
                    "kind": "points",
                    "statuses": ["numerical-failure"],
                    "at_least": 1,
                },
            }
        ],
        "decision_point": {
            "date": "2026-12-01",
            "signal": {
                "kind": "metric",
                "row": RECOVERY,
                "statistic": "mean",
                "direction": "above",
                "threshold": 0.5,
            },
            "fallback": "Induced grammars recover gold derivations on toy data.",
        },
        "evaluation": {
            "alpha": 0.05,
            "correction": {"kind": "benjamini-hochberg"},
            "dependence": "prds",
            "minimum_seeds": 3,
            "seed_budget": {
                "metric": "recovery",
                "target": "mean",
                "precision": 0.05,
                "level": 0.95,
            },
        },
    }


def a_judgment_experiment_document() -> dict[str, JsonValue]:
    """Return a complete experiment document over the judgment-fit document."""
    document = an_experiment_document()
    document["config"] = {"kind": "judgment-fit", "path": "fit.json"}
    document["sweep"] = {
        "grid": [{"path": "inference.warmup", "values": [100, 200]}],
        "seeds": [0, 1, 2],
        "expected_points": 6,
    }
    rhat: dict[str, JsonValue] = {"evaluator": "fit", "metric": "max-rhat"}
    document["rows"] = [rhat, {"evaluator": "fit", "metric": "min-ess"}]
    document["primary"] = rhat
    document["factors"] = {
        "trainer.epochs": {
            "axis": "inference.warmup",
            "levels": [
                {"level": 1, "value": 100},
                {"level": 2, "value": 200},
            ],
        }
    }
    document["decision_point"] = {
        "date": "2026-12-01",
        "signal": {
            "kind": "metric",
            "row": rhat,
            "statistic": "max",
            "direction": "below",
            "threshold": 1.01,
        },
        "fallback": "The fit converges on one warmup length.",
    }

    return document


def write_experiment(
    root: Path,
    document: dict[str, JsonValue],
    *,
    name: str = "headline",
) -> Path:
    """Write an experiment, its run document and its preregistration under a root."""
    folder = root / "experiments"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "run.json").write_text(
        json.dumps(a_document()), encoding="utf-8", newline="\n"
    )
    (folder / "fit.json").write_text(
        json.dumps(a_fit_document({"objective": "conditional"})),
        encoding="utf-8",
        newline="\n",
    )
    (folder / "preregistration.json").write_text(
        json.dumps(PREREGISTRATION), encoding="utf-8", newline="\n"
    )
    path = folder / f"{name}.yaml"
    path.write_text(json.dumps(document), encoding="utf-8", newline="\n")

    return path


def refusal(tmp_path: Path, document: dict[str, JsonValue]) -> str:
    """Return the reason the loader refuses a document."""
    path = write_experiment(tmp_path, document)

    with pytest.raises(ExperimentError) as refused:
        read_experiment(path)

    assert refused.value.document == path

    return refused.value.reason


def test_a_complete_document_loads_into_one_frozen_experiment(tmp_path: Path) -> None:
    path = write_experiment(tmp_path, an_experiment_document())

    experiment = read_experiment(path)

    assert isinstance(experiment, Experiment)
    assert experiment.name == "headline"
    assert experiment.role == "headline"
    assert experiment.freeze_id == "freeze/1"
    assert experiment.config.kind == "run"
    assert experiment.config.path == path.parent / "run.json"
    declared_sweep = Sweep.model_validate(SWEEP)
    assert tuple((axis.key, axis.points) for axis in experiment.sweep.grid) == tuple(
        (axis.path, axis.points) for axis in declared_sweep.grid
    )
    assert experiment.sweep.clauses == declared_sweep.clauses
    assert experiment.sweep.seeds == declared_sweep.seeds
    assert experiment.sweep.expected_points == declared_sweep.expected_points
    assert experiment.seed_policy == ExperimentSeed(7)
    assert experiment.rows == (
        ReportedRow("held-out", "recovery"),
        ReportedRow("held-out", "parity"),
    )
    assert experiment.reports == ()
    assert experiment.primary == ReportedRow("held-out", "recovery")
    assert experiment.preregistration == read_preregistration(
        json.dumps(PREREGISTRATION)
    )
    assert experiment.preregistration_document == path.parent / "preregistration.json"
    assert dict(experiment.factors) == {"trainer.epochs": "trainer.epochs"}
    assert experiment.compute == ComputeEstimate(12.5, 20.0, "price-v3")
    assert tuple(rung.step for rung in experiment.contraction.rungs) == (
        "unpreregistered-cells",
        "seeds",
        "fallback",
    )
    assert experiment.contraction.factorial == "two epoch counts, fully crossed"
    assert tuple(rule.name for rule in experiment.stop_rules) == ("numerical failure",)
    assert experiment.decision_point.decided_on == date(2026, 12, 1)
    assert experiment.amendments == ()
    assert not experiment.falls_back

    with pytest.raises(AttributeError):
        experiment.__setattr__("_name", "analysis")

    with pytest.raises(TypeError):
        cast("dict[str, str]", experiment.factors)["trainer.epochs"] = "seed"


def test_declared_report_layouts_lower_with_the_experiment(tmp_path: Path) -> None:
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

    experiment = read_experiment(write_experiment(tmp_path, document))

    assert experiment.reports == (
        ReportSpec(
            name="headline",
            rows=("held-out/recovery",),
            columns=("metric", "value", "nan_rate"),
            labels=("Metric", "Value", "NaN rate"),
            precision=2,
            format="latex",
        ),
    )
    assert experiment.reports[0].renderer() is LATEX


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("format", "html", "report_format"),
        ("columns", ["metric", "not-a-column"], "unknown columns"),
        ("name", "../headline", "filename stem"),
    ],
)
def test_a_report_declares_an_installed_format_known_columns_and_a_stem(
    tmp_path: Path, field: str, value: JsonValue, message: str
) -> None:
    document = an_experiment_document()
    report: dict[str, JsonValue] = {
        "name": "headline",
        "rows": ["held-out/recovery"],
        "columns": ["metric", "value"],
        "labels": ["Metric", "Value"],
    }
    report[field] = value
    document["reports"] = [report]

    assert message in refusal(tmp_path, document)


def test_a_report_cannot_read_an_undeclared_experiment_row(tmp_path: Path) -> None:
    document = an_experiment_document()
    document["reports"] = [
        {
            "name": "headline",
            "rows": ["held-out/not-declared"],
            "columns": ["metric"],
            "labels": ["Metric"],
        }
    ]

    assert "reads undeclared rows" in refusal(tmp_path, document)


def test_a_report_may_repeat_a_heading_but_not_a_column(tmp_path: Path) -> None:
    document = an_experiment_document()
    report: dict[str, JsonValue] = {
        "name": "headline",
        "rows": ["held-out/recovery"],
        "columns": ["metric", "value"],
        "labels": ["Result", "Result"],
    }
    document["reports"] = [report]

    assert read_experiment(write_experiment(tmp_path, document)).reports[0].labels == (
        "Result",
        "Result",
    )

    report["columns"] = ["metric", "metric"]

    assert "repeats an entry" in refusal(tmp_path, document)


def test_a_yaml_document_reads_like_its_json_spelling(tmp_path: Path) -> None:
    path = write_experiment(tmp_path, an_experiment_document())
    from_json = read_experiment(path)
    path.write_text(
        """\
role: headline
freeze_id: freeze/1
config:
  kind: run
  path: run.json
launch: {kind: serial}
failure: {numerical: final, out_of_memory: retry}
sweep:
  grid:
    - path: trainer.epochs
      values: [1, 2]
  seeds: [0, 1, 2]
  expected_points: 6
seed_policy:
  root: 7
rows:
  - {evaluator: held-out, metric: recovery}
  - {evaluator: held-out, metric: parity}
primary: {evaluator: held-out, metric: recovery}
preregistration: preregistration.json
compute:
  kind: estimate
  p50_hours: 12.5
  p90_hours: 20
  cost_model: price-v3
contraction:
  rungs:
    - kind: unpreregistered-cells
    - {kind: seeds, floor: 3}
    - kind: fallback
  factorial: two epoch counts, fully crossed
stop_rules:
  - name: numerical failure
    when: {kind: points, statuses: [numerical-failure], at_least: 1}
decision_point:
  date: "2026-12-01"
  signal:
    kind: metric
    row: {evaluator: held-out, metric: recovery}
    statistic: mean
    direction: above
    threshold: 0.5
  fallback: Induced grammars recover gold derivations on toy data.
evaluation:
  alpha: 0.05
  correction: {kind: benjamini-hochberg}
  dependence: prds
  minimum_seeds: 3
  seed_budget:
    metric: recovery
    target: mean
    precision: 0.05
    level: 0.95
""",
        encoding="utf-8",
        newline="\n",
    )

    from_yaml = read_experiment(path)

    assert from_yaml.sweep == from_json.sweep
    assert from_yaml.seed_policy == from_json.seed_policy
    assert from_yaml.compute == from_json.compute
    assert from_yaml.rows == from_json.rows
    assert from_yaml.primary == from_json.primary
    assert tuple(rung.step for rung in from_yaml.contraction.rungs) == tuple(
        rung.step for rung in from_json.contraction.rungs
    )
    assert from_yaml.decision_point.decided_on == from_json.decision_point.decided_on
    assert from_yaml.evaluation.alpha == from_json.evaluation.alpha
    assert from_yaml.evaluation.correction.name == from_json.evaluation.correction.name
    assert from_yaml.evaluation.dependence == from_json.evaluation.dependence
    assert from_yaml.evaluation.minimum_seeds == from_json.evaluation.minimum_seeds
    assert from_yaml.evaluation.seed_budget == from_json.evaluation.seed_budget


def test_an_experiment_expands_to_the_sweep_id_of_its_declared_grid(
    tmp_path: Path,
) -> None:
    experiment = read_experiment(write_experiment(tmp_path, an_experiment_document()))
    independent = Sweep(
        grid=(Axis(path="trainer.epochs", values=(1, 2)),), seeds=(0, 1, 2)
    )
    other = Sweep(grid=(Axis(path="trainer.epochs", values=(1, 3)),), seeds=(0, 1, 2))
    selection = Selection(
        config=experiment.config.path,
        profile=None,
        overrides=(),
        overlays=(),
        search_path=config_search_path(tmp_path),
    )
    expander = GridSweepExpander(composer(selection))

    points = expand_experiment(experiment, tmp_path)

    assert len(points) == 6
    assert {point.seed for point in points} == {0, 1, 2}
    assert sweep_id(points) == sweep_id(expander.expand(independent))
    assert sweep_id(points) != sweep_id(expander.expand(other))


@pytest.mark.parametrize(
    "field", ["freeze_id", "contraction", "primary", "compute", "stop_rules"]
)
def test_a_document_missing_a_required_field_is_refused(
    tmp_path: Path, field: str
) -> None:
    document = an_experiment_document()
    del document[field]

    assert field in refusal(tmp_path, document)


def test_a_preregistration_family_assignment_is_not_an_experiment_field(
    tmp_path: Path,
) -> None:
    document = an_experiment_document()
    document["prereg_family"] = {"recovery": "primary"}

    assert "prereg_family" in refusal(tmp_path, document)


CAPACITY_PREREGISTRATION: dict[str, JsonValue] = {
    "grid": {
        "model-width": ["w1", "w2"],
        "embedding-width": ["e16", "e64"],
        "primitives": ["k2", "k4"],
    },
    "hypotheses": [
        {"name": "capacity", "left": "w2", "right": "w1", "family": "capacity"}
    ],
}


class FactorialOptimizer(OptimizerSpec):
    """An optimizer declaration with settings a factorial sweep can vary."""

    kind: Literal["factorial-unit-optimizer"] = "factorial-unit-optimizer"
    a: int = 0
    b: int = 0
    c: int = 0
    d: int = 0
    e: int = 0
    f: int = 0
    language: str = "deu"


def write_factorial_experiment(
    root: Path,
    axes: dict[str, list[JsonValue]],
    *,
    factors: dict[str, JsonValue],
    preregistration: dict[str, JsonValue],
    rungs: list[JsonValue],
    seeds: tuple[int, ...] = (0, 1, 2),
) -> Path:
    """Write an experiment sweeping the factorial optimizer's settings."""
    count = len(seeds)

    for values in axes.values():
        count *= len(values)

    document = an_experiment_document()
    document["sweep"] = {
        "grid": [
            {"path": f"optimizer.{name}", "values": values}
            for name, values in axes.items()
        ],
        "seeds": list(seeds),
        "expected_points": count,
    }
    document["factors"] = factors
    document["contraction"] = {"rungs": rungs, "factorial": "a factorial over a-f"}
    path = write_experiment(root, document)
    run = a_document()
    run["optimizer"] = {"kind": "factorial-unit-optimizer"}
    (path.parent / "run.json").write_text(
        json.dumps(run), encoding="utf-8", newline="\n"
    )
    (path.parent / "preregistration.json").write_text(
        json.dumps(preregistration), encoding="utf-8", newline="\n"
    )

    return path


def write_capacity_experiment(
    root: Path, axes: tuple[str, ...], factors: dict[str, JsonValue]
) -> Path:
    """Write an experiment whose preregistration crosses three factors."""
    values: dict[str, list[JsonValue]] = {"a": [1, 2], "b": [16, 64], "c": [2, 4]}

    return write_factorial_experiment(
        root,
        {axis: values[axis] for axis in axes},
        factors=factors,
        preregistration=CAPACITY_PREREGISTRATION,
        rungs=[{"kind": "fallback"}],
    )


CAPACITY_FACTORS: dict[str, JsonValue] = {
    "model-width": {
        "axis": "optimizer.a",
        "levels": [{"level": "w1", "value": 1}, {"level": "w2", "value": 2}],
    },
    "embedding-width": {
        "axis": "optimizer.b",
        "levels": [{"level": "e16", "value": 16}, {"level": "e64", "value": 64}],
    },
    "primitives": {
        "axis": "optimizer.c",
        "levels": [{"level": "k2", "value": 2}, {"level": "k4", "value": 4}],
    },
}


def test_an_experiment_is_refused_until_its_sweep_spans_every_preregistered_factor(
    tmp_path: Path,
) -> None:
    capacity = {"model-width": CAPACITY_FACTORS["model-width"]}
    width_only = write_capacity_experiment(tmp_path, ("a",), capacity)

    with pytest.raises(ExperimentError) as refused:
        read_experiment(width_only)

    assert "'embedding-width'" in refused.value.reason
    assert "'primitives'" in refused.value.reason

    spanned = write_capacity_experiment(
        tmp_path,
        ("a", "b", "c"),
        CAPACITY_FACTORS,
    )

    assert read_experiment(spanned).factors == {
        "model-width": "optimizer.a",
        "embedding-width": "optimizer.b",
        "primitives": "optimizer.c",
    }


def test_a_factor_whose_level_maps_to_no_axis_value_is_refused(
    tmp_path: Path,
) -> None:
    factors = dict(CAPACITY_FACTORS)
    factors["primitives"] = {
        "axis": "optimizer.c",
        "levels": [{"level": "k2", "value": 2}, {"level": "k4", "value": 8}],
    }
    path = write_capacity_experiment(
        tmp_path,
        ("a", "b", "c"),
        factors,
    )

    with pytest.raises(ExperimentError, match=r"does not take the values \['8'\]"):
        read_experiment(path)


def test_a_factor_with_an_unmapped_level_is_refused(tmp_path: Path) -> None:
    factors = dict(CAPACITY_FACTORS)
    factors["primitives"] = {
        "axis": "optimizer.c",
        "levels": [{"level": "k2", "value": 2}],
    }
    path = write_capacity_experiment(
        tmp_path,
        ("a", "b", "c"),
        factors,
    )

    with pytest.raises(ExperimentError, match="maps no axis value"):
        read_experiment(path)


def test_a_binding_for_an_undeclared_factor_is_refused(tmp_path: Path) -> None:
    document = an_experiment_document()
    document["factors"] = {"depth": {"axis": "trainer.epochs"}}

    assert "'depth'" in refusal(tmp_path, document)


def test_a_row_naming_no_declared_evaluator_is_refused_at_load(tmp_path: Path) -> None:
    document = an_experiment_document()
    recovery: dict[str, JsonValue] = {"evaluator": "recovery", "metric": "recovery"}
    document["rows"] = [recovery]
    document["primary"] = recovery
    signal = cast("dict[str, JsonValue]", document["decision_point"])
    signal["signal"] = {"kind": "points", "statuses": ["completed"], "at_least": 1}

    reason = refusal(tmp_path, document)

    assert "'recovery'" in reason
    assert "does not declare" in reason


def test_a_row_naming_a_metric_its_evaluator_does_not_report_is_refused(
    tmp_path: Path,
) -> None:
    document = an_experiment_document()
    coverage: dict[str, JsonValue] = {"evaluator": "held-out", "metric": "coverage"}
    document["rows"] = [RECOVERY, coverage]

    reason = refusal(tmp_path, document)

    assert "'coverage'" in reason
    assert "does not report" in reason


def test_a_primary_row_outside_the_declared_rows_is_refused(tmp_path: Path) -> None:
    document = an_experiment_document()
    document["primary"] = {"evaluator": "held-out", "metric": "coverage"}

    assert "primary row" in refusal(tmp_path, document)


def test_a_predicate_reading_an_undeclared_row_is_refused(tmp_path: Path) -> None:
    document = an_experiment_document()
    document["rows"] = [PARITY]
    document["primary"] = PARITY

    assert "predicate reads 'held-out/recovery'" in refusal(tmp_path, document)


def test_a_point_count_predicate_over_an_unknown_status_is_refused(
    tmp_path: Path,
) -> None:
    document = an_experiment_document()
    document["stop_rules"] = [
        {
            "name": "lost",
            "when": {"kind": "points", "statuses": ["lost"], "at_least": 1},
        }
    ]

    assert "'lost'" in refusal(tmp_path, document)


def test_a_point_whose_sweep_removes_a_rows_evaluator_is_refused_at_expansion(
    tmp_path: Path,
) -> None:
    fragments = tmp_path / "configs" / "evaluators" / "held-out"
    fragments.mkdir(parents=True)
    (fragments / "unit.json").write_text(
        json.dumps(
            {
                "kind": "launcher-unit-evaluator",
                "interval": {"replicates": 50, "level": 0.95},
            }
        ),
        encoding="utf-8",
        newline="\n",
    )
    document = an_experiment_document()
    document["sweep"] = {
        "grid": [
            {"path": "trainer.epochs", "values": [1, 2]},
            {"group": "evaluators.held-out", "values": ["unit", None]},
        ],
        "seeds": [0, 1, 2],
        "expected_points": 12,
    }
    path = write_experiment(tmp_path, document)
    run = a_document()
    del run["evaluators"]
    (path.parent / "run.json").write_text(
        json.dumps(run), encoding="utf-8", newline="\n"
    )
    experiment = read_experiment(path)

    with pytest.raises(ExperimentError, match="declares no evaluator under"):
        expand_experiment(experiment, tmp_path)


def test_a_document_declaring_itself_unpriced_loads_as_unpriced(
    tmp_path: Path,
) -> None:
    priced = read_experiment(write_experiment(tmp_path, an_experiment_document()))
    document = an_experiment_document()
    document["compute"] = {"kind": "unpriced"}

    experiment = read_experiment(write_experiment(tmp_path, document))
    points = expand_experiment(experiment, tmp_path)

    assert experiment.compute is UNPRICED
    assert experiment.compute != ComputeEstimate(12.5, 20.0, "price-v3")
    assert len(points) == 6
    assert sweep_id(points) == sweep_id(expand_experiment(priced, tmp_path))


@pytest.mark.parametrize(
    "compute",
    ["unpriced", "priced", "", None, 12.5, ["unpriced"], {"kind": "priced"}],
    ids=repr,
)
def test_a_compute_that_is_neither_an_estimate_nor_unpriced_is_refused(
    tmp_path: Path, compute: JsonValue
) -> None:
    document = an_experiment_document()
    document["compute"] = compute

    assert "compute" in refusal(tmp_path, document)


@pytest.mark.parametrize(
    "compute",
    [
        {"kind": "estimate", "p50_hours": 1.0, "p90_hours": 2.0},
        {
            "kind": "estimate",
            "p50_hours": 1.0,
            "p90_hours": 2.0,
            "cost_model": "v",
            "extra": 1,
        },
        {"p50_hours": 1.0, "p90_hours": 2.0, "cost_model": "v"},
        {"kind": "unpriced", "p50_hours": 1.0},
    ],
    ids=("incomplete", "extra", "untagged", "unpriced-with-hours"),
)
def test_a_malformed_estimate_is_refused(tmp_path: Path, compute: JsonValue) -> None:
    document = an_experiment_document()
    document["compute"] = compute

    assert "compute" in refusal(tmp_path, document)


def test_a_p90_estimate_below_the_p50_estimate_is_refused(tmp_path: Path) -> None:
    document = an_experiment_document()
    document["compute"] = {
        "kind": "estimate",
        "p50_hours": 20,
        "p90_hours": 12.5,
        "cost_model": "v3",
    }

    assert "below the P50 estimate" in refusal(tmp_path, document)


def test_a_custom_experiment_role_is_accepted(tmp_path: Path) -> None:
    document = an_experiment_document()
    document["role"] = "pilot"

    experiment = read_experiment(write_experiment(tmp_path, document))

    assert experiment.role == "pilot"


def test_a_blank_experiment_role_is_refused(tmp_path: Path) -> None:
    document = an_experiment_document()
    document["role"] = ""

    assert "experiment role" in refusal(tmp_path, document)


@pytest.mark.parametrize(
    ("freeze_id", "reason"),
    [
        ("freeze/A/1", "positive integer"),
        ("freeze/0", "positive integer"),
        ("freeze/01", "positive integer"),
        ("freeze/one", "positive integer"),
        ("1", "positive integer"),
    ],
)
def test_a_freeze_identifier_that_is_not_a_positive_number_is_refused(
    tmp_path: Path, freeze_id: str, reason: str
) -> None:
    document = an_experiment_document(freeze_id=freeze_id)

    assert reason in refusal(tmp_path, document)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("paper", "A"),
        ("claim", "Induced grammars recover the gold derivations."),
        ("falsification", "Recovery at the shuffled-label baseline."),
    ],
)
def test_a_document_declaring_a_field_the_boundary_does_not_hold_is_refused(
    tmp_path: Path, field: str, value: JsonValue
) -> None:
    document = an_experiment_document()
    document[field] = value

    assert field in refusal(tmp_path, document)


def test_an_experiment_document_loads_from_any_directory(tmp_path: Path) -> None:
    path = write_experiment(tmp_path, an_experiment_document())
    nested = tmp_path / "studies" / "lexicon"
    nested.mkdir(parents=True)

    for sibling in path.parent.iterdir():
        (nested / sibling.name).write_bytes(sibling.read_bytes())

    experiment = read_experiment(nested / path.name)

    assert experiment.name == "headline"
    assert experiment.config.path == nested / "run.json"


@pytest.mark.parametrize(
    ("rungs", "reason"),
    [
        ([], "no rung"),
        (
            [
                {"kind": "unpreregistered-cells"},
                {"kind": "unpreregistered-cells"},
            ],
            "repeats",
        ),
        ([{"kind": "languages", "axis": "trainer.epochs", "order": []}], "drop order"),
        ([{"kind": "seeds", "floor": 2}], "below the 3 seeds"),
        ([{"kind": "seeds", "floor": 4}], "exceeds the 3 seeds"),
        ([{"kind": "resolution-iv", "generators": []}], "no generator"),
        (
            [
                {
                    "kind": "resolution-iv",
                    "generators": [{"factor": "c", "word": ["a", "b"]}],
                }
            ],
            "resolution 4",
        ),
        (
            [
                {
                    "kind": "resolution-iv",
                    "generators": [{"factor": "a", "word": ["a", "b", "c"]}],
                }
            ],
            "its own word",
        ),
        (
            [
                {
                    "kind": "corpus-scale",
                    "axis": "trainer.epochs",
                    "order": [1, 1],
                }
            ],
            "repeats",
        ),
    ],
)
def test_a_malformed_contraction_ladder_is_refused(
    tmp_path: Path, rungs: list[JsonValue], reason: str
) -> None:
    document = an_experiment_document()
    document["contraction"] = {"rungs": rungs, "factorial": "two epoch counts"}

    assert reason in refusal(tmp_path, document)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed_policy", {"root": 7, "derivation": "blake2b-32-point-v1"}),
        (
            "decision_point",
            {
                "date": "December",
                "signal": {"kind": "points", "statuses": ["completed"], "at_least": 1},
                "fallback": "f",
            },
        ),
        ("decision_point", {"date": "2026-12-01", "signal": "s", "fallback": "f"}),
        ("preregistration", "absent.json"),
        ("config", "absent.json"),
        ("rows", []),
        ("stop_rules", ["halt"]),
        (
            "stop_rules",
            [
                {
                    "name": "halt",
                    "when": {"kind": "points", "statuses": ["killed"], "at_least": 1},
                },
                {
                    "name": "halt",
                    "when": {"kind": "points", "statuses": ["running"], "at_least": 1},
                },
            ],
        ),
        ("preregistration", " "),
    ],
)
def test_a_malformed_field_is_refused(
    tmp_path: Path, field: str, value: JsonValue
) -> None:
    document = an_experiment_document()
    document[field] = value

    refusal(tmp_path, document)


def test_a_judgment_fit_experiment_loads_and_expands_over_its_fit(
    tmp_path: Path,
) -> None:
    path = write_experiment(tmp_path, a_judgment_experiment_document())

    experiment = read_experiment(path)
    points = expand_experiment(experiment, tmp_path)

    assert experiment.config.kind == "judgment-fit"
    assert experiment.config.path == path.parent / "fit.json"
    assert len(points) == 6
    assert [
        (
            JudgmentFitSpec.model_validate_json(
                json.dumps(dict(point.config))
            ).inference.warmup,
            point.seed,
        )
        for point in points
    ] == [(warmup, seed) for warmup in (100, 200) for seed in (0, 1, 2)]
    assert all(
        point.point_id == point_id(point.config, seed=point.seed) for point in points
    )
    assert all(point.provenance is not None for point in points)
    assert expand_experiment(read_experiment(path), tmp_path) == points


def test_judgment_fit_point_ids_are_stable_across_processes_and_hash_seeds(
    tmp_path: Path,
) -> None:
    path = write_experiment(tmp_path, a_judgment_experiment_document())
    script = (
        "import sys\n"
        "from pathlib import Path\n"
        "from lofi.run import expand_experiment, read_experiment, sweep_id\n"
        "experiment = read_experiment(Path(sys.argv[1]))\n"
        "points = expand_experiment(experiment, Path(sys.argv[2]))\n"
        "print(sweep_id(points), *(point.point_id for point in points))\n"
    )
    listings = {
        subprocess.run(
            [sys.executable, "-c", script, str(path), str(tmp_path)],
            capture_output=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
            encoding="utf-8",
        ).stdout
        for hash_seed in ("0", "1")
    }
    points = expand_experiment(read_experiment(path), tmp_path)

    assert listings == {
        " ".join((sweep_id(points), *(point.point_id for point in points))) + "\n"
    }


def test_a_configuration_of_neither_kind_is_refused(tmp_path: Path) -> None:
    document = an_experiment_document()
    document["config"] = {"kind": "corpus", "path": "run.json"}

    assert "config" in refusal(tmp_path, document)


@pytest.mark.parametrize("document", ["fit.json", "preregistration.json"])
def test_a_run_experiment_over_a_document_that_is_no_run_is_refused_at_load(
    tmp_path: Path, document: str
) -> None:
    declared = an_experiment_document()
    declared["config"] = {"kind": "run", "path": document}

    assert "not a run document" in refusal(tmp_path, declared)


@pytest.mark.parametrize("document", ["run.json", "preregistration.json"])
def test_a_fit_experiment_over_a_document_that_is_no_fit_is_refused_at_expansion(
    tmp_path: Path, document: str
) -> None:
    declared = a_judgment_experiment_document()
    declared["config"] = {"kind": "judgment-fit", "path": document}
    path = write_experiment(tmp_path, declared)
    experiment = read_experiment(path)

    with pytest.raises(ExperimentError) as refused:
        expand_experiment(experiment, tmp_path)

    assert refused.value.document == path
    assert "not a judgment-fit document" in refused.value.reason


def test_each_configuration_kind_names_the_file_its_manifest_is_recorded_in(
    tmp_path: Path,
) -> None:
    run = read_experiment(write_experiment(tmp_path, an_experiment_document()))
    fits = read_experiment(
        write_experiment(tmp_path, a_judgment_experiment_document(), name="fits")
    )

    assert run.config.manifest_name == MANIFEST_NAME
    assert fits.config.manifest_name == FIT_MANIFEST_NAME
    assert MANIFEST_NAME != FIT_MANIFEST_NAME


def test_only_a_run_configurations_rows_may_claim_an_enumeration_budget(
    tmp_path: Path,
) -> None:
    run = read_experiment(write_experiment(tmp_path, an_experiment_document()))
    fits = read_experiment(
        write_experiment(tmp_path, a_judgment_experiment_document(), name="fits")
    )

    assert run.config.enumerates
    assert not fits.config.enumerates


@pytest.mark.parametrize(
    "seed_policy",
    [
        {"root": 7, "derivation": "blake2b-32-point-v1"},
        {"root": -7},
        {},
    ],
    ids=("derivation", "root", "incomplete"),
)
def test_a_seed_policy_naming_no_experiment_seed_is_refused(
    tmp_path: Path, seed_policy: JsonValue
) -> None:
    document = an_experiment_document()
    document["seed_policy"] = seed_policy

    assert "seed_policy" in refusal(tmp_path, document)


@pytest.mark.parametrize(
    "path",
    [
        pytest.param(
            path,
            marks=pytest.mark.skipif(
                path.stem == "source-factors" and shutil.which("vampire") is None,
                reason="the source-factor declaration requires the Vampire prover",
            ),
        )
        for pattern in ("experiments/*/*.yaml", "docs/examples/experiments/*.yaml")
        for path in sorted(Path(__file__).resolve().parents[3].glob(pattern))
        if not path.stem.endswith("-run")
    ],
    ids=lambda path: path.stem,
)
def test_every_checked_in_experiment_document_loads_and_expands(path: Path) -> None:
    experiment = read_experiment(path)
    points = expand_experiment(experiment, Path(__file__).resolve().parents[3])

    assert len(points) == experiment.sweep.expected_points


def test_an_experiment_prices_each_stack_off_its_own_steps_unless_it_declares_a_grid(
    tmp_path: Path,
) -> None:
    plain = read_experiment(write_experiment(tmp_path, an_experiment_document()))
    document = an_experiment_document()
    document["pricing"] = {"kind": "length-batch"}
    default = read_experiment(write_experiment(tmp_path, document))
    document["pricing"] = {"kind": "length-batch", "lengths": [5, 9], "batches": [2, 4]}
    declared = read_experiment(write_experiment(tmp_path, document))

    assert isinstance(plain.pricing, StackBootstrapCostModel)
    assert isinstance(default.pricing, LengthBatchCostModel)
    assert isinstance(declared.pricing, LengthBatchCostModel)
    assert (default.pricing.lengths, default.pricing.batches) == (
        (10, 20, 30, 40),
        (8, 16, 32),
    )
    assert (declared.pricing.lengths, declared.pricing.batches) == ((5, 9), (2, 4))
    assert plain.pricing.resamples == default.pricing.resamples == 10_000


def test_an_experiment_declares_the_bootstrap_replicates_its_price_reads(
    tmp_path: Path,
) -> None:
    document = an_experiment_document()
    document["pricing"] = {"kind": "stack-bootstrap", "resamples": 500}

    assert read_experiment(write_experiment(tmp_path, document)).pricing.resamples == (
        500
    )


@pytest.mark.parametrize(
    "pricing",
    [
        {"kind": "length-batch", "lengths": [10]},
        {"kind": "length-batch", "batches": [16, 8]},
        {"kind": "length-batch", "lengths": [0, 10]},
        {"kind": "interpolated"},
    ],
    ids=("one-length", "descending", "zero", "unknown"),
)
def test_a_pricing_grid_that_cannot_interpolate_is_refused(
    tmp_path: Path, pricing: JsonValue
) -> None:
    document = an_experiment_document()
    document["pricing"] = pricing

    assert "pricing" in refusal(tmp_path, document)


NO_FACTORS: dict[str, JsonValue] = {
    "grid": {},
    "hypotheses": [
        {"name": "recovery", "left": "induced", "right": "gold", "family": "primary"}
    ],
}

TWO_LEVELS: dict[str, list[JsonValue]] = {name: [0, 1] for name in "abcdef"}

RESOLUTION_IV: dict[str, JsonValue] = {
    "kind": "resolution-iv",
    "generators": [
        {
            "factor": "optimizer.e",
            "word": ["optimizer.a", "optimizer.b", "optimizer.c"],
        },
        {
            "factor": "optimizer.f",
            "word": ["optimizer.b", "optimizer.c", "optimizer.d"],
        },
    ],
}

AMENDED_ON = date(2026, 11, 2)


def coded(points: tuple[SweepPoint, ...]) -> list[dict[str, int]]:
    """Return each point's six two-level settings as signs."""
    return [
        {
            name: -1
            if cast("dict[str, int]", point.config["optimizer"])[name] == 0
            else 1
            for name in "abcdef"
        }
        for point in points
    ]


def test_the_resolution_iv_rung_reduces_a_two_to_the_six_grid_to_sixteen_points(
    tmp_path: Path,
) -> None:
    path = write_factorial_experiment(
        tmp_path,
        TWO_LEVELS,
        factors={},
        preregistration=NO_FACTORS,
        rungs=[RESOLUTION_IV],
        seeds=(0,),
    )
    experiment = read_experiment(path)
    full = expand_experiment(experiment, tmp_path)

    amended = amend_experiment(experiment, "resolution-iv", tmp_path, on=AMENDED_ON)
    points = expand_experiment(amended, tmp_path)
    runs = coded(points)

    assert len(full) == 64
    assert len(points) == 16
    assert len({tuple(run.values()) for run in runs}) == 16

    for factor in "abcdef":
        assert sum(run[factor] for run in runs) == 0

        for left, right in itertools.combinations("abcdef", 2):
            assert sum(run[factor] * run[left] * run[right] for run in runs) == 0

    assert all(run["e"] == run["a"] * run["b"] * run["c"] for run in runs)
    assert amended.amendments == (
        Amendment(
            "resolution-iv",
            (),
            sweep_id(full),
            sweep_id(points),
            16,
            AMENDED_ON,
        ),
    )


def test_a_recorded_amendment_is_applied_again_when_the_experiment_is_loaded(
    tmp_path: Path,
) -> None:
    path = write_factorial_experiment(
        tmp_path,
        TWO_LEVELS,
        factors={},
        preregistration=NO_FACTORS,
        rungs=[RESOLUTION_IV],
        seeds=(0,),
    )
    amended = amend_experiment(
        read_experiment(path),
        "resolution-iv",
        tmp_path,
        on=AMENDED_ON,
    )

    reloaded = read_experiment(path)

    assert amendment_ledger(path).is_file()
    assert reloaded.sweep == amended.sweep
    assert reloaded.amendments == amended.amendments
    assert sweep_id(expand_experiment(reloaded, tmp_path)) == (
        amended.amendments[-1].sweep_id
    )


def test_an_amended_experiment_whose_points_changed_since_is_refused_at_expansion(
    tmp_path: Path,
) -> None:
    path = write_factorial_experiment(
        tmp_path,
        TWO_LEVELS,
        factors={},
        preregistration=NO_FACTORS,
        rungs=[RESOLUTION_IV],
        seeds=(0,),
    )
    amend_experiment(
        read_experiment(path),
        "resolution-iv",
        tmp_path,
        on=AMENDED_ON,
    )
    run = json.loads((path.parent / "run.json").read_text(encoding="utf-8"))
    run["trainer"]["epochs"] = 3
    (path.parent / "run.json").write_text(
        json.dumps(run), encoding="utf-8", newline="\n"
    )

    with pytest.raises(ExperimentError, match="its last amendment recorded"):
        expand_experiment(read_experiment(path), tmp_path)


def test_a_ledger_that_disagrees_with_its_rung_is_refused(tmp_path: Path) -> None:
    path = write_factorial_experiment(
        tmp_path,
        {"a": [0, 1]},
        factors={},
        preregistration=NO_FACTORS,
        rungs=[{"kind": "seeds", "floor": 3}],
        seeds=(0, 1, 2, 3),
    )
    amend_experiment(read_experiment(path), "seeds", tmp_path, on=AMENDED_ON)
    ledger = amendment_ledger(path)
    recorded = json.loads(ledger.read_text(encoding="utf-8"))
    recorded["amendments"][0]["dropped"] = [2]
    ledger.write_text(json.dumps(recorded), encoding="utf-8", newline="\n")

    with pytest.raises(ExperimentError, match="removing"):
        read_experiment(path)


def test_the_seed_rung_keeps_the_first_seeds_down_to_its_floor(tmp_path: Path) -> None:
    path = write_factorial_experiment(
        tmp_path,
        {"a": [0, 1]},
        factors={},
        preregistration=NO_FACTORS,
        rungs=[{"kind": "seeds", "floor": 3}],
        seeds=(0, 1, 2, 3),
    )

    amended = amend_experiment(read_experiment(path), "seeds", tmp_path, on=AMENDED_ON)

    assert amended.sweep.seeds == (0, 1, 2)
    assert amended.amendments[-1].dropped == (3,)
    assert len(expand_experiment(amended, tmp_path)) == 6

    with pytest.raises(ExperimentError, match="already at the floor"):
        amend_experiment(amended, "seeds", tmp_path, on=AMENDED_ON)


def test_the_language_rung_drops_languages_in_its_declared_order(
    tmp_path: Path,
) -> None:
    path = write_factorial_experiment(
        tmp_path,
        {"language": ["deu", "nld", "eng"]},
        factors={},
        preregistration=NO_FACTORS,
        rungs=[
            {
                "kind": "languages",
                "axis": "optimizer.language",
                "order": ["nld", "eng"],
            }
        ],
    )
    experiment = read_experiment(path)

    once = amend_experiment(experiment, "languages", tmp_path, on=AMENDED_ON)
    twice = amend_experiment(once, "languages", tmp_path, on=AMENDED_ON)

    assert once.amendments[-1].dropped == ("nld",)
    assert twice.amendments[-1].dropped == ("eng",)
    assert {
        cast("dict[str, str]", point.config["optimizer"])["language"]
        for point in expand_experiment(twice, tmp_path)
    } == {"deu"}

    with pytest.raises(ExperimentError, match="no value left in the drop order"):
        amend_experiment(twice, "languages", tmp_path, on=AMENDED_ON)


def test_the_corpus_scale_rung_drops_scales_in_its_declared_order(
    tmp_path: Path,
) -> None:
    path = write_factorial_experiment(
        tmp_path,
        {"b": [100, 1000, 10000]},
        factors={},
        preregistration=NO_FACTORS,
        rungs=[{"kind": "corpus-scale", "axis": "optimizer.b", "order": [10000]}],
    )

    amended = amend_experiment(
        read_experiment(path), "corpus-scale", tmp_path, on=AMENDED_ON
    )

    assert amended.amendments[-1].dropped == (10000,)
    assert len(expand_experiment(amended, tmp_path)) == 6


def test_the_unpreregistered_cells_rung_drops_values_realizing_no_level(
    tmp_path: Path,
) -> None:
    path = write_factorial_experiment(
        tmp_path,
        {"a": [1, 2, 3], "b": [0, 1]},
        factors={"width": {"axis": "optimizer.a"}},
        preregistration={
            "grid": {"width": [1, 2]},
            "hypotheses": NO_FACTORS["hypotheses"],
        },
        rungs=[{"kind": "unpreregistered-cells"}, {"kind": "seeds", "floor": 3}],
        seeds=(0, 1, 2, 3),
    )
    experiment = read_experiment(path)

    amended = amend_experiment(
        experiment, "unpreregistered-cells", tmp_path, on=AMENDED_ON
    )

    assert amended.amendments[-1].dropped == (3,)
    assert len(expand_experiment(amended, tmp_path)) == 16

    seeded = amend_experiment(amended, "seeds", tmp_path, on=AMENDED_ON)

    with pytest.raises(ExperimentError, match="precedes the applied rung 'seeds'"):
        amend_experiment(seeded, "unpreregistered-cells", tmp_path, on=AMENDED_ON)


def test_the_unpreregistered_cells_rung_keeps_a_value_any_factor_on_its_axis_realizes(
    tmp_path: Path,
) -> None:
    path = write_factorial_experiment(
        tmp_path,
        {"a": [1, 2, 3, 4], "b": [0, 1]},
        factors={"width": {"axis": "optimizer.a"}, "depth": {"axis": "optimizer.a"}},
        preregistration={
            "grid": {"width": [1, 2], "depth": [2, 3]},
            "hypotheses": NO_FACTORS["hypotheses"],
        },
        rungs=[{"kind": "unpreregistered-cells"}],
    )

    amended = amend_experiment(
        read_experiment(path),
        "unpreregistered-cells",
        tmp_path,
        on=AMENDED_ON,
    )

    assert amended.amendments[-1].dropped == (4,)
    assert len(expand_experiment(amended, tmp_path)) == 18


def test_a_rung_that_changes_no_point_is_refused(tmp_path: Path) -> None:
    experiment = read_experiment(write_experiment(tmp_path, an_experiment_document()))

    with pytest.raises(ExperimentError, match="changes no point"):
        amend_experiment(experiment, "unpreregistered-cells", tmp_path, on=AMENDED_ON)

    assert not amendment_ledger(experiment.document).exists()


def test_the_fallback_rung_keeps_the_points_and_switches_to_the_fallback_claim(
    tmp_path: Path,
) -> None:
    experiment = read_experiment(write_experiment(tmp_path, an_experiment_document()))
    before = sweep_id(expand_experiment(experiment, tmp_path))

    amended = amend_experiment(experiment, "fallback", tmp_path, on=AMENDED_ON)

    assert amended.falls_back
    assert amended.amendments[-1].sweep_id == before
    assert read_experiment(experiment.document).falls_back


def test_a_rung_the_ladder_does_not_declare_is_refused(tmp_path: Path) -> None:
    experiment = read_experiment(write_experiment(tmp_path, an_experiment_document()))

    with pytest.raises(ExperimentError, match="declares no rung 'resolution-iv'"):
        amend_experiment(
            experiment,
            "resolution-iv",
            tmp_path,
            on=AMENDED_ON,
        )


def test_the_resolution_iv_rung_refuses_an_axis_without_two_values(
    tmp_path: Path,
) -> None:
    axes = dict(TWO_LEVELS)
    axes["a"] = [0, 1, 2]
    path = write_factorial_experiment(
        tmp_path,
        axes,
        factors={},
        preregistration=NO_FACTORS,
        rungs=[RESOLUTION_IV],
        seeds=(0,),
    )

    with pytest.raises(ExperimentError, match="exactly two values"):
        amend_experiment(
            read_experiment(path),
            "resolution-iv",
            tmp_path,
            on=AMENDED_ON,
        )


class _Point:
    @property
    def response_levels(self) -> int:
        return 3

    @property
    def relation_labels(self) -> tuple[InferenceLabel, ...]:
        return (
            InferenceLabel.ENTAILS,
            InferenceLabel.CONTRADICTS,
            InferenceLabel.NEUTRAL,
        )

    def predict_responses(
        self, pairs: Sequence[InferenceText], /
    ) -> tuple[tuple[float, ...], ...]:
        return tuple((0.6, 0.3, 0.1) for _ in pairs)

    def predict_relations(
        self, pairs: Sequence[InferenceText], /
    ) -> tuple[tuple[float, ...], ...]:
        return tuple((0.7, 0.2, 0.1) for _ in pairs)


def test_predicted_source_document_names_producer_and_item_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots: list[Path] = []

    def source(
        settings: PredictedJudgmentSourceSettings,
        root: Path,
        predictor: TrainedPointPredictor,
    ) -> PredictedJudgmentSource:
        roots.append(root)
        item = Item(
            id=UUID(int=1),
            item_template_id=UUID(int=0),
            rendered_elements={"context": "someone admires", "claim": "someone likes"},
        )
        return PredictedJudgmentSource(settings, (item,), predictor)

    monkeypatch.setattr(judgment, "predicted_judgment_source_for", source)
    document = tmp_path / "predicted.json"
    document.write_text(
        """{
          "kind": "predicted",
          "producer_run": "runs/synthetic",
          "producer_point": "point-17",
          "item_set": "items.json",
          "inputs": {"premise": "context", "hypothesis": "claim"},
          "factors": {},
          "question": "inference",
          "labels": ["entails", "contradicts", "neutral"],
          "participant": "lofi-point-17"
        }""",
        encoding="utf-8",
        newline="\n",
    )

    predicted = execute_predicted_judgments(document, _Point())

    assert roots == [tmp_path]
    assert predicted.rows()[0].item == str(UUID(int=1))
    assert predicted.predictions[0].response_probabilities == (0.6, 0.3, 0.1)
    assert predicted.predictions[0].relation_probabilities == {
        InferenceLabel.ENTAILS: 0.7,
        InferenceLabel.CONTRADICTS: 0.2,
        InferenceLabel.NEUTRAL: 0.1,
    }
    assert predicted.provenance("predicted-inference").producer_run == (
        "runs/synthetic"
    )
    assert predicted.provenance("predicted-inference").producer_point == "point-17"
