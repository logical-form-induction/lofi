"""Tests for registered experiment-row reducers."""

import json
from typing import TYPE_CHECKING, cast

import pytest
from tests.support.installed import installed

from lofi.eval._prereg import DeclaredHypothesis, Preregistration
from lofi.eval._reducers import (
    RANK_CHURN_PLUGIN,
    RESCORING_PLUGIN,
    VARIANCE_PLUGIN,
    ExperimentReducer,
    ReductionContext,
    reduce_experiment_rows,
)
from lofi.eval._rescoring import (
    RecallHomogeneityMetric,
    read_predictions,
)
from lofi.eval._values import ConventionVector
from lofi.registry import ComponentPlugin, Registry
from lofi.spec.reducers import (
    RankChurnReducerSpec,
    RescoringReducerSpec,
    VarianceReducerSpec,
)
from lofi.stats import Bonferroni, Dependence

if TYPE_CHECKING:
    from pathlib import Path

    from lofi._types import JsonValue

type ReducerPlugin = ComponentPlugin[object, ReductionContext, ExperimentReducer]


def _prediction_text() -> str:
    items: list[JsonValue] = []

    for split, suffix in (("dev", "1"), ("test", "1")):
        items.append(
            {
                "item_id": f"{split}-{suffix}",
                "split": split,
                "length": 2,
                "punctuation": [],
                "gold": [{"start": 0, "end": 2, "label": "S", "depth": 0}],
                "spans": [{"start": 0, "end": 2, "label": "c0"}],
            }
        )

    document: JsonValue = {
        "system": "released-parser",
        "budget": {"depth": 2, "beam": 4, "mass": 0.0},
        "items": items,
    }
    return json.dumps(document)


def _context(root: Path) -> ReductionContext:
    contributed = ConventionVector(
        punctuation="strip",
        unary_chains="keep",
        trivial_spans="keep",
        minimum_length=0,
        aggregation="corpus",
        coverage="ignore",
        self_arcs="allow",
        gold_derivation="canonical",
    ).extended({"provider": "laboratory"})
    readers = installed("prediction_format", {"lofi-json": read_predictions})
    metrics = installed(
        "rescoring_metric", {"recall-homogeneity": RecallHomogeneityMetric()}
    )
    conventions = installed("convention", {"laboratory": lambda: (contributed,)})
    return ReductionContext(
        root,
        seed=7,
        prediction_formats=readers,
        rescoring_metrics=metrics,
        conventions=conventions,
    )


def _registry() -> Registry[ReducerPlugin]:
    return installed(
        "reducer",
        {
            "variance": cast("ReducerPlugin", VARIANCE_PLUGIN),
            "rescoring": cast("ReducerPlugin", RESCORING_PLUGIN),
            "rank-churn": cast("ReducerPlugin", RANK_CHURN_PLUGIN),
        },
    )


def _variance_rows() -> tuple[dict[str, JsonValue], ...]:
    effects = {"convention": (-0.3, 0.3), "seed": (-0.2, 0.2), "sample": (-0.1, 0.1)}
    return tuple(
        {
            "metric": "accuracy",
            "method": f"method-{method}",
            "convention_hash": convention,
            "seed": seed,
            "data_subsample": sample,
            "value": effects["convention"][convention]
            + effects["seed"][seed]
            + effects["sample"][sample]
            + (0.01 if method == 0 else -0.01),
        }
        for convention in range(2)
        for seed in range(2)
        for sample in range(2)
        for method in range(2)
    )


def test_an_experiment_reduces_rows_through_variance_and_rescoring(
    tmp_path: Path,
) -> None:
    tmp_path.joinpath("predictions.json").write_text(
        _prediction_text(), encoding="utf-8", newline="\n"
    )
    variance = VarianceReducerSpec.model_validate(
        {
            "kind": "variance",
            "metric": "accuracy",
            "factors": [
                {"name": "convention", "column": "convention_hash"},
                {"name": "seed", "column": "seed"},
                {"name": "data-subsample", "column": "data_subsample"},
            ],
            "estimator": "moments",
            "interval": {"replicates": 9, "level": 0.9},
            "method_column": "method",
            "value_column": "value",
        }
    )
    rescoring = RescoringReducerSpec.model_validate(
        {
            "kind": "rescoring",
            "source": "predictions.json",
            "alignments": [
                {
                    "kind": "many-to-one",
                    "tie_break": "least-gold-label",
                    "fit_on": "dev",
                    "score_on": "test",
                    "random_floor": {"replicates": 3, "level": 0.9},
                    "stability": {"replicates": 3, "level": 0.9},
                }
            ],
            "metric": "recall-homogeneity",
            "conventions": {
                "punctuation": ["keep"],
                "unary_chains": ["keep"],
                "trivial_spans": ["keep"],
                "minimum_length": [0],
                "aggregation": ["corpus"],
                "coverage": ["ignore"],
                "self_arcs": ["allow"],
                "gold_derivation": ["canonical"],
            },
            "prediction_format": "lofi-json",
            "register": ["laboratory"],
        }
    )

    reduced = reduce_experiment_rows(
        _variance_rows(),
        (variance, rescoring),
        _context(tmp_path),
        registry=_registry(),
    )

    variance_rows = tuple(row for row in reduced.rows if row["reducer"] == "variance")
    rescored_rows = tuple(row for row in reduced.rows if row["reducer"] == "rescoring")
    assert {row["source"] for row in variance_rows} == {
        "convention",
        "seed",
        "data-subsample",
        "interaction",
        "residual",
    }
    assert len(rescored_rows) == 2
    assert {row.get("provider") for row in rescored_rows} == {None, "laboratory"}


def test_rank_churn_is_a_registered_row_reducer(tmp_path: Path) -> None:
    declaration = RankChurnReducerSpec.model_validate(
        {
            "kind": "rank-churn",
            "metric": "score",
            "constrained": "eisner",
            "unconstrained": "free",
            "method_column": "method",
            "condition_column": "normal_form",
            "value_column": "value",
        }
    )
    rows: tuple[dict[str, JsonValue], ...] = (
        {"metric": "score", "method": "a", "normal_form": "eisner", "value": 0.9},
        {"metric": "score", "method": "b", "normal_form": "eisner", "value": 0.8},
        {"metric": "score", "method": "a", "normal_form": "free", "value": 0.7},
        {"metric": "score", "method": "b", "normal_form": "free", "value": 0.8},
    )

    assert reduce_experiment_rows(
        rows,
        (declaration,),
        _context(tmp_path),
        registry=_registry(),
    ).rows == (
        {
            "reducer": "rank-churn",
            "metric": "score",
            "left": "eisner",
            "right": "free",
            "methods": 2,
            "changed": 2,
            "value": 1.0,
        },
    )


def test_a_context_carries_the_declared_correction_with_its_level(
    tmp_path: Path,
) -> None:
    plan = Preregistration(
        {"candidate": ["source", "rival"]},
        [
            DeclaredHypothesis(
                "source-against-rival", "source", "rival", family="primary"
            )
        ],
    )
    context = ReductionContext(
        tmp_path,
        seed=7,
        preregistration=plan,
        alpha=0.05,
        correction=Bonferroni(),
        dependence=Dependence.ARBITRARY,
    )

    assert context.correction.name == "bonferroni"
    assert context.dependence is Dependence.ARBITRARY

    with pytest.raises(ValueError, match="supplied together"):
        ReductionContext(
            tmp_path,
            seed=7,
            preregistration=plan,
            alpha=0.05,
            dependence=Dependence.ARBITRARY,
        )
