"""Tests for the registered experiment-level diagnostic reducers."""

from itertools import product
from typing import TYPE_CHECKING, cast

import pytest

from lofi.eval._diagnostic_reducers import (
    COHERENCE_COLLAPSE_PLUGIN,
    CROSSED_RECOVERY_PLUGIN,
    CROSSLINGUAL_PLUGIN,
    ENCODER_ABLATION_PLUGIN,
    RESIDUAL_STRUCTURE_PLUGIN,
    TRANSFER_PLUGIN,
)
from lofi.eval._reducers import ExperimentReducer, ReductionContext
from lofi.spec.reducers import (
    CoherenceCollapseReducerSpec,
    CrossedRecoveryReducerSpec,
    CrosslingualReducerSpec,
    EncoderAblationReducerSpec,
    ResidualStructureReducerSpec,
    TransferReducerSpec,
)

if TYPE_CHECKING:
    from pathlib import Path

    from lofi._types import JsonValue
    from lofi.registry import ComponentPlugin


def _build(plugin: object, settings: object, root: Path) -> ExperimentReducer:
    held = cast("ComponentPlugin[object, ReductionContext, ExperimentReducer]", plugin)
    return held.build(settings, ReductionContext(root, seed=13))


def _language_rows() -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        {
            "metric": "accuracy",
            "language": language,
            "value": value + seed * 0.02,
            "n": 10,
            "seed": seed,
        }
        for language, value in (("en", 0.9), ("de", 0.7), ("es", 0.6))
        for seed in (0, 1)
    )


def test_transfer_emits_scores_gaps_summaries_and_unavailable_rows(
    tmp_path: Path,
) -> None:
    spec = TransferReducerSpec.model_validate(
        {
            "kind": "transfer",
            "metrics": ["accuracy"],
            "source_language": "en",
            "languages": {
                "en": "gold",
                "de": "gold",
                "es": "silver",
                "zh": "silver",
            },
            "metric_column": "metric",
            "language_column": "language",
            "value_column": "value",
            "count_column": "n",
        }
    )
    rows = (
        _build(TRANSFER_PLUGIN, spec.settings(), tmp_path).reduce(_language_rows()).rows
    )

    assert {row["kind"] for row in rows} >= {
        "score",
        "gap",
        "score-summary",
        "gap-summary",
    }
    german_gap = next(
        row for row in rows if row["language"] == "de" and row["kind"] == "gap"
    )
    unavailable = tuple(row for row in rows if row["language"] == "zh")
    silver = tuple(
        row
        for row in rows
        if row.get("source") == "silver"
        and isinstance(row["kind"], str)
        and row["kind"].endswith("summary")
    )
    assert german_gap["value"] == pytest.approx(0.2)
    assert unavailable
    assert not any(row["available"] for row in unavailable)
    assert silver
    assert not any(row["available"] for row in silver)


def test_crosslingual_metrics_emit_each_registered_summary(tmp_path: Path) -> None:
    spec = CrosslingualReducerSpec.model_validate(
        {
            "kind": "cross-lingual",
            "metrics": ["accuracy"],
            "source_language": "en",
            "metric_column": "metric",
            "language_column": "language",
            "value_column": "value",
        }
    )
    rows = (
        _build(CROSSLINGUAL_PLUGIN, spec.settings(), tmp_path)
        .reduce(_language_rows())
        .rows
    )

    assert {row["summary"] for row in rows} == {
        "macro",
        "minimum",
        "disparity",
        "source-gap",
    }
    assert all(row["metric"] == "accuracy" for row in rows)


def _encoder_rows() -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        {
            "checkpoint": checkpoint,
            "freeze_state": freeze,
            "metric": "accuracy",
            "seed": seed,
            "value": base + seed * 0.1,
            "weights": "stand-in" if checkpoint == "xlmr" else "pretrained",
        }
        for checkpoint, freeze, base in (
            ("mt5", "frozen", 0.7),
            ("xlmr", "frozen", 0.5),
            ("mt5", "fine-tuned", 0.9),
            ("xlmr", "fine-tuned", 0.8),
        )
        for seed in (0, 1, 2, 3)
    )


def test_encoder_ablation_emits_complete_cells_and_seed_scaled_differences(
    tmp_path: Path,
) -> None:
    spec = EncoderAblationReducerSpec.model_validate(
        {
            "kind": "encoder-ablation",
            "checkpoints": ["mt5", "xlmr"],
            "freeze_states": ["frozen", "fine-tuned"],
            "metrics": ["accuracy"],
            "minimum_seeds": 4,
            "checkpoint_column": "checkpoint",
            "freeze_column": "freeze_state",
            "metric_column": "metric",
            "seed_column": "seed",
            "value_column": "value",
            "weights_column": "weights",
        }
    )
    rows = (
        _build(ENCODER_ABLATION_PLUGIN, spec.settings(), tmp_path)
        .reduce(_encoder_rows())
        .rows
    )
    cells = tuple(row for row in rows if row["kind"] == "cell")
    differences = tuple(row for row in rows if row["kind"] == "difference")

    assert len(cells) == 4
    assert len(differences) == 2
    assert all(row["value"] is not None for row in differences)
    assert differences[0]["effect"] == pytest.approx(0.2)
    assert any(row["involves_stand_in"] for row in differences)


def test_encoder_ablation_preserves_metric_scale_when_seed_spread_is_zero(
    tmp_path: Path,
) -> None:
    spec = EncoderAblationReducerSpec.model_validate(
        {
            "kind": "encoder-ablation",
            "checkpoints": ["mt5", "xlmr"],
            "freeze_states": ["frozen"],
            "metrics": ["accuracy"],
            "minimum_seeds": 4,
            "checkpoint_column": "checkpoint",
            "freeze_column": "freeze_state",
            "metric_column": "metric",
            "seed_column": "seed",
            "value_column": "value",
            "weights_column": "weights",
        }
    )
    rows: tuple[dict[str, JsonValue], ...] = tuple(
        {
            "checkpoint": checkpoint,
            "freeze_state": "frozen",
            "metric": "accuracy",
            "seed": seed,
            "value": value,
            "weights": "pretrained",
        }
        for checkpoint, value in (("mt5", 0.8), ("xlmr", 0.5))
        for seed in range(4)
    )
    reduced = (
        _build(ENCODER_ABLATION_PLUGIN, spec.settings(), tmp_path).reduce(rows).rows
    )
    difference = next(row for row in reduced if row["kind"] == "difference")

    assert difference["value"] == pytest.approx(0.3)
    assert difference["effect"] == pytest.approx(0.3)
    assert difference["standardized"] is None


def _recovery_rows() -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        {
            "metric": "recovery",
            "architecture": architecture,
            "width": width,
            "seed": seed,
            "value": 0.6 + 0.1 * seed + 0.05 * width,
            "floor": 0.25,
        }
        for architecture, width, seed in product(
            ("rnn", "transformer"), (1, 2), range(4)
        )
    )


def test_crossed_recovery_emits_every_factor_cell(tmp_path: Path) -> None:
    spec = CrossedRecoveryReducerSpec.model_validate(
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
        }
    )
    rows = (
        _build(CROSSED_RECOVERY_PLUGIN, spec.settings(), tmp_path)
        .reduce(_recovery_rows())
        .rows
    )

    assert len(rows) == 4
    assert {(row["architecture"], row["width"]) for row in rows} == set(
        product(("rnn", "transformer"), (1, 2))
    )
    assert all(row["value"] is not None for row in rows)
    assert all(0.0 < cast("float", row["p_value"]) <= 1.0 for row in rows)


def test_residual_structure_regresses_every_declared_predictor(tmp_path: Path) -> None:
    spec = ResidualStructureReducerSpec.model_validate(
        {
            "kind": "residual-structure",
            "response_column": "residual",
            "root_entropy_column": "root_entropy",
            "non_target_column": "non_target",
            "derivation_count_column": "derivation_count",
        }
    )
    rows: tuple[dict[str, JsonValue], ...] = tuple(
        {
            "root_entropy": entropy,
            "non_target": category,
            "derivation_count": count,
            "residual": 1.0
            + 2.0 * entropy
            + 0.5 * count
            + (0.75 if category == "verb" else 0.0)
            - (0.25 if category == "adjective" else 0.0),
        }
        for entropy, category, count in (
            (0.1, "noun", 1),
            (0.2, "verb", 3),
            (0.4, "adjective", 2),
            (0.8, "verb", 1),
            (0.3, "noun", 4),
            (0.7, "adjective", 5),
            (0.9, "noun", 3),
            (0.5, "verb", 2),
            (0.15, "adjective", 4),
            (0.65, "noun", 5),
        )
    )
    reduced = (
        _build(RESIDUAL_STRUCTURE_PLUGIN, spec.settings(), tmp_path).reduce(rows).rows
    )
    coefficients = {
        row["term"]: row["value"] for row in reduced if row["metric"] == "coefficient"
    }

    assert coefficients["root-entropy"] == pytest.approx(2.0)
    assert coefficients["derivation-count"] == pytest.approx(0.5)
    assert coefficients["non-target:noun"] == pytest.approx(0.25)
    assert coefficients["non-target:verb"] == pytest.approx(1.0)
    assert reduced[-1]["value"] == pytest.approx(1.0)


def test_coherence_collapse_emits_both_cells_and_the_paired_difference(
    tmp_path: Path,
) -> None:
    spec = CoherenceCollapseReducerSpec.model_validate(
        {
            "kind": "coherence-collapse",
            "metric": "type-distribution-collapse-rate",
            "kl": "kl",
            "cross_entropy": "cross-entropy",
            "minimum_seeds": 4,
            "metric_column": "metric",
            "divergence_column": "coherence_divergence",
            "seed_column": "seed",
            "value_column": "value",
        }
    )
    rows: tuple[dict[str, JsonValue], ...] = tuple(
        {
            "metric": "type-distribution-collapse-rate",
            "coherence_divergence": divergence,
            "seed": seed,
            "value": base + seed * 0.01,
        }
        for divergence, base in (("kl", 0.2), ("cross-entropy", 0.5))
        for seed in range(4)
    )
    reduced = (
        _build(COHERENCE_COLLAPSE_PLUGIN, spec.settings(), tmp_path).reduce(rows).rows
    )

    assert [row["kind"] for row in reduced] == ["cell", "cell", "difference"]
    assert reduced[-1]["value"] == pytest.approx(-0.3)
    assert reduced[-1]["n"] == 4
