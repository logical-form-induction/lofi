"""Experiment-document integration for registered evaluation reducers."""

import json
from typing import TYPE_CHECKING, cast

from tests.acceptance.driver import resolved
from tests.support.installed import installed
from tests.unit.run.test__experiment import (
    an_experiment_document,
    write_experiment,
)

from lofi.eval._comparison import PointMetadata
from lofi.eval._diagnostic_reducers import (
    COHERENCE_COLLAPSE_PLUGIN,
    CROSSED_RECOVERY_PLUGIN,
    CROSSLINGUAL_PLUGIN,
    ENCODER_ABLATION_PLUGIN,
    RESIDUAL_STRUCTURE_PLUGIN,
    TRANSFER_PLUGIN,
)
from lofi.eval._reducers import (
    ReductionContext,
    reduce_experiment_rows,
)
from lofi.eval._restitutive_coverage import PLUGIN as RESTITUTIVE_COVERAGE_PLUGIN
from lofi.eval._values import ConventionVector
from lofi.run import read_experiment

if TYPE_CHECKING:
    from pathlib import Path

    from lofi._types import JsonValue


def _predictions() -> str:
    items: list[JsonValue] = [
        {
            "item_id": f"{split}-1",
            "split": split,
            "length": 2,
            "punctuation": [],
            "gold": [{"start": 0, "end": 2, "label": "S", "depth": 0}],
            "spans": [{"start": 0, "end": 2, "label": "c0"}],
        }
        for split in ("dev", "test")
    ]

    document: JsonValue = {
        "system": "checkpoint-parser",
        "budget": {"depth": 2, "beam": 4, "mass": 0.0},
        "items": items,
    }
    return json.dumps(document)


def _variance() -> dict[str, JsonValue]:
    return {
        "kind": "variance",
        "metric": "recovery",
        "factors": [
            {"name": "seed", "column": "seed"},
            {"name": "data-subsample", "column": "data_subsample"},
        ],
        "estimator": "moments",
        "interval": {"replicates": 5, "level": 0.9},
        "method_column": "method",
        "value_column": "value",
    }


def _rescoring() -> dict[str, JsonValue]:
    return {
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


def _rows() -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        {
            "metric": "recovery",
            "method": f"parser-{method}",
            "seed": seed,
            "data_subsample": sample,
            "value": 0.1 * seed + 0.2 * sample + 0.01 * method,
        }
        for seed in range(2)
        for sample in range(2)
        for method in range(2)
    )


def test_an_experiment_document_runs_variance_and_registered_rescoring(
    tmp_path: Path,
) -> None:
    document = an_experiment_document()
    document["reducers"] = [_variance(), _rescoring()]
    path = write_experiment(tmp_path, document)
    path.parent.joinpath("predictions.json").write_text(
        _predictions(), encoding="utf-8", newline="\n"
    )
    experiment = read_experiment(path)

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

    conventions = installed("convention", {"laboratory": lambda: (contributed,)})
    reduced = reduce_experiment_rows(
        _rows(),
        experiment.reducers,
        ReductionContext(
            path.parent,
            seed=experiment.seed_policy.root,
            conventions=conventions,
        ),
    )

    variance = tuple(row for row in reduced.rows if row["reducer"] == "variance")
    rescored = tuple(row for row in reduced.rows if row["reducer"] == "rescoring")
    assert {row["source"] for row in variance} == {
        "seed",
        "data-subsample",
        "interaction",
        "residual",
    }
    assert len(rescored) == 2
    assert {row.get("provider") for row in rescored} == {None, "laboratory"}


def test_an_experiment_document_queries_restitutive_lexicon_coverage(
    tmp_path: Path,
) -> None:
    document = an_experiment_document()
    document["reducers"] = [
        {
            "kind": "restitutive-coverage",
            "dataset": "train",
            "change_class": "change-state",
            "modifier_class": "repetition",
        }
    ]
    experiment = read_experiment(write_experiment(tmp_path, document))
    configuration = resolved("restitutive-coverage.yaml")
    point = PointMetadata(
        configuration,
        point_id="coverage-point",
        seed=0,
        result_seed=0,
        parameter_count=1,
        signature_fingerprint="coverage-signature",
    )

    reducers = installed(
        "reducer", {"restitutive-coverage": RESTITUTIVE_COVERAGE_PLUGIN}
    )
    reduction = reduce_experiment_rows(
        (),
        experiment.reducers,
        ReductionContext(
            tmp_path,
            seed=experiment.seed_policy.root,
            points={"coverage-config": (point,)},
        ),
        registry=reducers,
    )

    statuses = {
        cast("str", row["form"]): cast("str", row["status"]) for row in reduction.rows
    }
    assert statuses["breaks"] == "covered"
    assert statuses["taps"] == "uncovered"


def _diagnostic_reducers() -> list[dict[str, JsonValue]]:
    return [
        {
            "kind": "transfer",
            "metrics": ["language-accuracy"],
            "source_language": "en",
            "languages": {"en": "gold", "de": "gold"},
            "metric_column": "metric",
            "language_column": "language",
            "value_column": "value",
            "count_column": "n",
        },
        {
            "kind": "cross-lingual",
            "metrics": ["language-accuracy"],
            "source_language": "en",
            "metric_column": "metric",
            "language_column": "language",
            "value_column": "value",
        },
        {
            "kind": "encoder-ablation",
            "checkpoints": ["mt5", "xlmr"],
            "freeze_states": ["frozen"],
            "metrics": ["encoder-accuracy"],
            "minimum_seeds": 4,
            "checkpoint_column": "checkpoint",
            "freeze_column": "freeze_state",
            "metric_column": "metric",
            "seed_column": "seed",
            "value_column": "value",
            "weights_column": "weights",
        },
        {
            "kind": "crossed-recovery",
            "metrics": ["recovery"],
            "factor_columns": ["architecture", "width"],
            "minimum_seeds": 4,
            "null_replicates": 16,
            "metric_column": "metric",
            "seed_column": "seed",
            "value_column": "value",
            "floor_column": "floor",
        },
        {
            "kind": "coherence-collapse",
            "metric": "collapse",
            "kl": "kl",
            "cross_entropy": "cross-entropy",
            "minimum_seeds": 4,
            "metric_column": "metric",
            "divergence_column": "coherence_divergence",
            "seed_column": "seed",
            "value_column": "value",
        },
        {
            "kind": "residual-structure",
            "response_column": "residual",
            "root_entropy_column": "root_entropy",
            "non_target_column": "non_target",
            "derivation_count_column": "derivation_count",
        },
    ]


def _diagnostic_rows() -> tuple[dict[str, JsonValue], ...]:
    rows: list[dict[str, JsonValue]] = []
    identities = (
        *(
            ("language-accuracy", language, seed)
            for language in ("en", "de")
            for seed in range(4)
        ),
        *(
            ("encoder-accuracy", checkpoint, seed)
            for checkpoint in ("mt5", "xlmr")
            for seed in range(4)
        ),
        *(("recovery", "recovery", seed) for seed in range(4)),
        *(
            ("collapse", divergence, seed)
            for divergence in ("kl", "cross-entropy")
            for seed in range(4)
        ),
    )
    for index, (metric, identity, seed) in enumerate(identities):
        entropy = ((index * 7) % 17 + 1) / 18.0
        count = (index * 3) % 5 + 1
        category = ("noun", "verb", "adjective")[index % 3]
        rows.append(
            {
                "metric": metric,
                "language": identity if metric == "language-accuracy" else "en",
                "checkpoint": identity if metric == "encoder-accuracy" else "mt5",
                "freeze_state": "frozen",
                "weights": "pretrained",
                "architecture": "rnn",
                "width": 1,
                "coherence_divergence": identity if metric == "collapse" else "kl",
                "seed": seed,
                "value": 0.5
                + 0.03 * seed
                + (0.1 if identity in ("en", "mt5", "kl") else 0.0),
                "n": 10,
                "floor": 0.25,
                "root_entropy": entropy,
                "non_target": category,
                "derivation_count": count,
                "residual": 1.0
                + 2.0 * entropy
                + 0.5 * count
                + (0.75 if category == "verb" else 0.0)
                - (0.25 if category == "adjective" else 0.0),
            }
        )
    return tuple(rows)


def test_experiment_document_reaches_every_diagnostic_reducer(tmp_path: Path) -> None:
    document = an_experiment_document()
    document["reducers"] = cast("JsonValue", _diagnostic_reducers())
    experiment = read_experiment(write_experiment(tmp_path, document))

    reducers = installed(
        "reducer",
        {
            "transfer": TRANSFER_PLUGIN,
            "cross-lingual": CROSSLINGUAL_PLUGIN,
            "encoder-ablation": ENCODER_ABLATION_PLUGIN,
            "crossed-recovery": CROSSED_RECOVERY_PLUGIN,
            "coherence-collapse": COHERENCE_COLLAPSE_PLUGIN,
            "residual-structure": RESIDUAL_STRUCTURE_PLUGIN,
        },
    )

    reduced = reduce_experiment_rows(
        _diagnostic_rows(),
        experiment.reducers,
        ReductionContext(tmp_path, seed=experiment.seed_policy.root),
        registry=reducers,
    )

    assert {row["reducer"] for row in reduced.rows} == {
        "transfer",
        "cross-lingual",
        "encoder-ablation",
        "crossed-recovery",
        "coherence-collapse",
        "residual-structure",
    }
