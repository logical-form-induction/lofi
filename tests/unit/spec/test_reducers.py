"""Tests for experiment reducer declarations."""

from typing import TYPE_CHECKING, cast

import didactic.api as dx
import pytest

from lofi.spec.reducers import (
    CoherenceCollapseReducerSpec,
    ConventionGridSpec,
    CrossedRecoveryReducerSpec,
    CrosslingualReducerSpec,
    EncoderAblationReducerSpec,
    NodeVocabularyReducerSpec,
    RankChurnReducerSpec,
    ReducerSpec,
    RescoringReducerSpec,
    ResidualStructureReducerSpec,
    RestitutiveCoverageReducerSpec,
    SourceComparisonReducerSpec,
    TransferReducerSpec,
    VarianceReducerSpec,
)

if TYPE_CHECKING:
    from lofi._types import JsonValue

INTERVAL: dict[str, JsonValue] = {"replicates": 19, "level": 0.9}
ALIGNMENT: dict[str, JsonValue] = {
    "kind": "many-to-one",
    "tie_break": "least-gold-label",
    "fit_on": "dev",
    "score_on": "test",
    "random_floor": INTERVAL,
    "stability": INTERVAL,
}
GRID: dict[str, JsonValue] = {
    "punctuation": ["keep", "strip"],
    "unary_chains": ["keep"],
    "trivial_spans": ["drop"],
    "minimum_length": [0, 3],
    "aggregation": ["corpus"],
    "coverage": ["zero"],
    "self_arcs": ["forbid"],
    "gold_derivation": ["canonical"],
}


def test_restitutive_coverage_declares_the_lexicon_and_semantic_classes() -> None:
    declared = RestitutiveCoverageReducerSpec.model_validate(
        {
            "kind": "restitutive-coverage",
            "dataset": "train",
            "change_class": "change-state",
            "modifier_class": "repetition",
        }
    )

    assert declared.settings().dataset == "train"
    assert declared.settings().change_class == "change-state"
    assert declared.settings().modifier_class == "repetition"

    with pytest.raises(dx.ValidationError, match="distinct"):
        RestitutiveCoverageReducerSpec.model_validate(
            declared.model_dump() | {"modifier_class": "change-state"}
        )


def test_variance_settings_declare_factors_and_estimator() -> None:
    declared = VarianceReducerSpec.model_validate(
        {
            "kind": "variance",
            "metric": "unlabeled-bracketing-f1",
            "factors": [
                {"name": "convention", "column": "convention_hash"},
                {"name": "seed", "column": "seed"},
                {"name": "data-subsample", "column": "data_subsample"},
            ],
            "estimator": "moments",
            "interval": INTERVAL,
            "method_column": "method",
            "value_column": "value",
        }
    )

    assert tuple(factor.name for factor in declared.settings().factors) == (
        "convention",
        "seed",
        "data-subsample",
    )
    assert declared.settings().estimator == "moments"


def test_variance_refuses_an_unidentified_or_repeated_factor_design() -> None:
    common: dict[str, JsonValue] = {
        "kind": "variance",
        "metric": "accuracy",
        "estimator": "moments",
        "interval": INTERVAL,
        "method_column": "method",
        "value_column": "value",
    }

    with pytest.raises(dx.ValidationError, match="at least two factors"):
        VarianceReducerSpec.model_validate(
            common | {"factors": [{"name": "seed", "column": "seed"}]}
        )

    with pytest.raises(dx.ValidationError, match="repeat"):
        VarianceReducerSpec.model_validate(
            common
            | {
                "factors": [
                    {"name": "seed", "column": "seed"},
                    {"name": "seed", "column": "subsample"},
                ]
            }
        )


def test_rescoring_declares_every_previous_fixed_choice() -> None:
    declared = RescoringReducerSpec.model_validate(
        {
            "kind": "rescoring",
            "source": "released.json",
            "alignments": [ALIGNMENT],
            "metric": "recall-homogeneity",
            "conventions": GRID,
            "prediction_format": "lofi-json",
            "register": ["laboratory-conventions"],
        }
    )
    settings = declared.settings()

    assert settings.source == "released.json"
    assert settings.alignments[0].kind == "many-to-one"
    assert settings.metric == "recall-homogeneity"
    assert settings.conventions.minimum_length == (0, 3)
    assert settings.prediction_format == "lofi-json"
    assert settings.register == ("laboratory-conventions",)


def test_rescoring_refuses_an_empty_grid_axis_or_repeated_register_entry() -> None:
    with pytest.raises(dx.ValidationError, match="declares no level"):
        ConventionGridSpec.model_validate(GRID | {"coverage": []})

    document: dict[str, JsonValue] = {
        "kind": "rescoring",
        "source": "released.json",
        "alignments": [ALIGNMENT],
        "metric": "recall-homogeneity",
        "conventions": GRID,
        "prediction_format": "lofi-json",
        "register": ["lab", "lab"],
    }

    with pytest.raises(dx.ValidationError, match="repeats an entry"):
        RescoringReducerSpec.model_validate(document)


def test_rank_churn_declares_the_two_conditions_and_input_columns() -> None:
    declared = RankChurnReducerSpec.model_validate(
        {
            "kind": "rank-churn",
            "metric": "derivational-log-likelihood",
            "constrained": "eisner",
            "unconstrained": "unconstrained",
            "method_column": "method",
            "condition_column": "normal_form",
            "value_column": "value",
        }
    )

    assert declared.settings().condition_column == "normal_form"

    with pytest.raises(dx.ValidationError, match="distinct conditions"):
        RankChurnReducerSpec.model_validate(
            declared.model_dump() | {"unconstrained": "eisner"}
        )


def test_source_comparison_declares_point_dimensions_and_statistical_tests() -> None:
    declared = SourceComparisonReducerSpec.model_validate(
        {
            "kind": "source-comparison",
            "metric": "held-out-fit",
            "source_path": "comparison.source",
            "candidate_path": "comparison.candidate",
            "data_size_path": "data.training_size",
            "condition_path": "comparison.condition",
            "direction_path": "grammar.direction",
            "twin_path": "comparison.twin",
            "description_length_metric": "description-length-nats",
            "primary_condition": "primary",
            "reversed_ground_truth_condition": "reversed-ground-truth",
            "lexicon_scrambled_condition": "lexicon-scrambled",
            "observed_twin": "observed",
            "permutation": {"replicates": 999, "seed": 13},
            "equivalence": {"replicates": 799, "seed": 17},
            "falsifier": "the generating source does not win",
            "smoke_test": False,
        }
    )

    settings = declared.settings()
    assert settings.source_path == "comparison.source"
    assert settings.value_column == "value"
    assert settings.method_column == "method"
    assert settings.capacity_field == "parameter_count"
    assert settings.permutation.replicates == 999
    assert type(ReducerSpec.model_validate(declared.model_dump())) is (
        SourceComparisonReducerSpec
    )

    with pytest.raises(dx.ValidationError, match="repeats a condition"):
        SourceComparisonReducerSpec.model_validate(
            declared.model_dump()
            | {"lexicon_scrambled_condition": "reversed-ground-truth"}
        )


def test_transfer_is_a_registered_parameterized_reducer() -> None:
    declared = TransferReducerSpec.model_validate(
        {
            "kind": "transfer",
            "metrics": ["supertag-accuracy", "bracketing-f1"],
            "source_language": "en",
            "languages": {"en": "gold", "de": "gold", "zh": "silver"},
            "metric_column": "metric",
            "language_column": "language",
            "value_column": "value",
            "count_column": "n",
        }
    )

    assert declared.settings().metrics == ("supertag-accuracy", "bracketing-f1")
    assert declared.settings().source_language == "en"
    assert type(ReducerSpec.model_validate(declared.model_dump())) is (
        TransferReducerSpec
    )

    with pytest.raises(dx.ValidationError, match="declaration is empty"):
        TransferReducerSpec.model_validate(declared.model_dump() | {"metrics": []})

    with pytest.raises(dx.ValidationError, match="repeats a value"):
        TransferReducerSpec.model_validate(
            declared.model_dump() | {"metrics": ["bracketing-f1", "bracketing-f1"]}
        )


def test_every_diagnostic_reducer_declares_its_row_schema() -> None:
    coherence = CoherenceCollapseReducerSpec.model_validate(
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
    crosslingual = CrosslingualReducerSpec.model_validate(
        {
            "kind": "cross-lingual",
            "metrics": ["accuracy"],
            "source_language": "en",
            "metric_column": "metric",
            "language_column": "language",
            "value_column": "value",
        }
    )
    encoder = EncoderAblationReducerSpec.model_validate(
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
    recovery = CrossedRecoveryReducerSpec.model_validate(
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
    residual = ResidualStructureReducerSpec.model_validate(
        {
            "kind": "residual-structure",
            "response_column": "residual",
            "root_entropy_column": "root_entropy",
            "non_target_column": "non_target",
            "derivation_count_column": "derivation_count",
        }
    )

    assert coherence.settings().kl == "kl"
    assert coherence.settings().cross_entropy == "cross-entropy"
    assert crosslingual.settings().source_language == "en"
    assert encoder.settings().freeze_states == ("frozen", "fine-tuned")
    assert recovery.settings().factor_columns == ("architecture", "width")
    assert residual.settings().non_target_column == "non_target"
    assert type(ReducerSpec.model_validate(crosslingual.model_dump())) is (
        CrosslingualReducerSpec
    )
    assert type(ReducerSpec.model_validate(encoder.model_dump())) is (
        EncoderAblationReducerSpec
    )
    assert type(ReducerSpec.model_validate(recovery.model_dump())) is (
        CrossedRecoveryReducerSpec
    )
    assert type(ReducerSpec.model_validate(residual.model_dump())) is (
        ResidualStructureReducerSpec
    )
    assert type(ReducerSpec.model_validate(coherence.model_dump())) is (
        CoherenceCollapseReducerSpec
    )

    with pytest.raises(dx.ValidationError, match="distinct KL"):
        CoherenceCollapseReducerSpec.model_validate(
            coherence.model_dump() | {"cross_entropy": "kl"}
        )


ARM_SOURCES: list[JsonValue] = [
    {
        "representation": representation,
        "source": f"arms/{representation}/probe-records.json",
        "probe": "causal",
        "record": "pass-0",
    }
    for representation in (
        "derivation-variables",
        "trained-transformer-sae-latents",
        "dense-linear-probe",
        "random-transformer-sae-control",
    )
]


def test_node_vocabulary_declares_one_persisted_record_per_arm() -> None:
    declared = ReducerSpec.model_validate(
        {"kind": "node-vocabulary", "arms": ARM_SOURCES}
    )

    assert isinstance(declared, NodeVocabularyReducerSpec)
    settings = declared.settings()
    assert [arm.representation for arm in settings.arms] == [
        "derivation-variables",
        "trained-transformer-sae-latents",
        "dense-linear-probe",
        "random-transformer-sae-control",
    ]
    assert settings.arms[2].source == "arms/dense-linear-probe/probe-records.json"
    assert (settings.arms[0].probe, settings.arms[0].record) == ("causal", "pass-0")


def test_node_vocabulary_refuses_a_missing_or_repeated_arm() -> None:
    with pytest.raises(dx.ValidationError, match="every arm once"):
        ReducerSpec.model_validate({"kind": "node-vocabulary", "arms": ARM_SOURCES[:3]})

    with pytest.raises(dx.ValidationError, match="every arm once"):
        ReducerSpec.model_validate(
            {"kind": "node-vocabulary", "arms": [*ARM_SOURCES[:3], ARM_SOURCES[0]]}
        )

    with pytest.raises(dx.ValidationError, match="node-vocabulary field"):
        ReducerSpec.model_validate(
            {
                "kind": "node-vocabulary",
                "arms": [
                    {**cast("dict[str, JsonValue]", ARM_SOURCES[0]), "probe": ""},
                    *ARM_SOURCES[1:],
                ],
            }
        )
