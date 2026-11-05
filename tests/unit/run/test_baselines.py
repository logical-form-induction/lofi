"""Tests for serial external-baseline execution."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

from tests.unit.run.test__experiment import (
    an_experiment_document,
    write_experiment,
)

from lofi.eval import BaselineMeasurement, BaselineResult, config_digest
from lofi.run import (
    BASELINE_FIT_MANIFEST_NAME,
    BaselineExecutor,
    ExecutionEnvironment,
    ExperimentStore,
    ManifestStatus,
    decode_baseline_manifest,
    expand_experiment,
    read_experiment,
    read_rows,
)
from lofi.spec.baselines import BiskHockenmaierBaselineSpec

if TYPE_CHECKING:
    from pathlib import Path

    from lofi._types import JsonValue
    from lofi.run import BaselineFitSpec, Experiment

BASELINE_METRICS = (
    "supertagging-length-10",
    "dependencies-length-10",
    "supertagging-length-20",
    "dependencies-length-20",
    "supertagging-all-lengths",
    "dependencies-all-lengths",
)


def fit_document() -> dict[str, object]:
    """Return the baseline fit exercised end to end."""
    return {
        "system": {
            "kind": "bisk-hockenmaier",
            "ccgbank_source": "licensed-ccgbank-auto",
            "dependency_gold_source": "licensed-johansson-nugues-conll",
            "reference_source": "lofi.resource:configs/eval/bisk_hockenmaier_2012.json",
            "reference_tolerance": 0.02,
            "nominal_category": "N",
            "sentence_category": "S",
            "coordination_category": "conj",
            "gold_sentence_atom": "S",
            "gold_coordination_atom": "conj",
            "nominal_tags": ["NN"],
            "verbal_tags": ["VB", "VBG"],
            "coordination_tags": ["CC"],
            "other_tags": ["JJ", "."],
            "punctuation_tags": ["."],
            "headed_or_modifier_arguments": ["N", "S"],
            "sentence_arguments": ["N"],
            "em_max_iterations": 2,
            "convergence_tolerance": 1e-6,
        }
    }


def baseline_experiment(root: Path) -> Experiment:
    """Write and load an experiment over the baseline-fit boundary."""
    document = an_experiment_document()
    document["config"] = {"kind": "baseline-fit", "path": "baseline.json"}
    document["sweep"] = {
        "grid": [{"path": "system.em_max_iterations", "values": [1, 2]}],
        "seeds": [0, 1, 2],
        "expected_points": 6,
    }
    declared_rows: list[dict[str, JsonValue]] = [
        {"evaluator": "baseline", "metric": metric} for metric in BASELINE_METRICS
    ]
    rows: list[JsonValue] = list(declared_rows)
    document["rows"] = rows
    primary = declared_rows[-1]
    document["primary"] = primary
    document["decision_point"] = {
        "date": "2026-12-01",
        "signal": {
            "kind": "metric",
            "row": primary,
            "statistic": "mean",
            "direction": "above",
            "threshold": 0.5,
        },
        "fallback": "The declared baseline remains below the target.",
    }
    evaluation = document["evaluation"]
    assert isinstance(evaluation, dict)
    seed_budget = evaluation["seed_budget"]
    assert isinstance(seed_budget, dict)
    seed_budget["metric"] = "dependencies-all-lengths"
    document["factors"] = {
        "trainer.epochs": {
            "axis": "system.em_max_iterations",
            "levels": [
                {"level": 1, "value": 1},
                {"level": 2, "value": 2},
            ],
        }
    }
    path = write_experiment(root, document, name="baseline")
    (path.parent / "baseline.json").write_text(
        json.dumps(fit_document()), encoding="utf-8", newline="\n"
    )
    return read_experiment(path)


class DeterministicFit:
    """Emit fixed rows while recording the boundary values it received."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Path, int]] = []

    def __call__(
        self, fit: BaselineFitSpec, root: Path, seed: int, /
    ) -> BaselineResult:
        self.calls.append((cast("str", fit.system.kind), root, seed))
        return BaselineResult(
            tuple(
                BaselineMeasurement(
                    "ccg-induction:dependencies",
                    "bisk-hockenmaier-2012",
                    metric,
                    "re-running",
                    config_digest(fit.model_dump()),
                    "the published predictions are not released",
                    value,
                    count,
                    "directed-dependencies",
                )
                for metric, value, count in zip(
                    BASELINE_METRICS,
                    (0.8, 0.7, 0.7, 0.6, 0.6, 0.5),
                    (10, 10, 20, 20, 30, 30),
                    strict=True,
                )
            ),
            {
                "ccgbank-auto": f"sha256:{'0' * 64}",
                "dependency-gold": f"sha256:{'1' * 64}",
            },
        )


def test_config_to_serial_execution_persists_rows_and_manifest(tmp_path: Path) -> None:
    assert BiskHockenmaierBaselineSpec.family() == "baseline_system"
    experiment = baseline_experiment(tmp_path / "repository")
    point = expand_experiment(experiment, tmp_path / "repository")[0]
    store = ExperimentStore(tmp_path / "store", experiment)
    operation = DeterministicFit()
    executor = BaselineExecutor(
        store,
        operation=operation,
        environment=ExecutionEnvironment({"lofi": "0.1"}, "0123abcd", "cpu"),
        infrastructure_attempts=1,
    )

    outcome = executor(point)
    derived = experiment.seed_policy.point(point.point_id, point.seed)
    recorded = read_rows(store.directory(point))
    manifest = decode_baseline_manifest(
        (store.directory(point) / BASELINE_FIT_MANIFEST_NAME).read_bytes()
    )

    assert experiment.config.kind == "baseline-fit"
    assert (outcome.status, outcome.cached) == (ManifestStatus.COMPLETED, False)
    assert operation.calls == [
        ("bisk-hockenmaier", experiment.config.path.parent, derived.root)
    ]
    assert recorded is not None
    assert tuple(row["metric"] for row in recorded.rows) == BASELINE_METRICS
    expected_conventions: dict[str, JsonValue] = dict(manifest.fit.system.conventions())
    expected_conventions["source_digests"] = dict(manifest.source_digests)
    assert {row["convention_hash"] for row in recorded.rows} == {
        config_digest(expected_conventions)
    }
    assert all(row["budget_kind"] == "none" for row in recorded.rows)
    assert manifest.git_revision == "0123abcd"
    assert manifest.source_digests == {
        "ccgbank-auto": f"sha256:{'0' * 64}",
        "dependency-gold": f"sha256:{'1' * 64}",
    }
    assert json.loads(manifest.fit.model_dump_json()) == dict(point.config)

    cached = executor(point)

    assert (cached.status, cached.cached) == (ManifestStatus.COMPLETED, True)
    assert len(operation.calls) == 1
