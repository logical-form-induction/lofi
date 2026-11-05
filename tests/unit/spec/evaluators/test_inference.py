from typing import TYPE_CHECKING

import didactic.api as dx
import pytest

import lofi.spec
from lofi.spec import EvaluationData
from lofi.spec.evaluators.inference import (
    InferenceEvaluatorSpec,
)

if TYPE_CHECKING:
    from lofi._types import JsonValue

CONVENTIONS = {
    "punctuation": "keep",
    "unary_chains": "keep",
    "trivial_spans": "keep",
    "minimum_length": 0,
    "aggregation": "corpus",
    "coverage": "zero",
    "self_arcs": "allow",
    "gold_derivation": "canonical",
}


def test_inference_declaration_is_public_at_the_spec_boundary() -> None:
    assert lofi.spec.InferenceEvaluatorSpec is InferenceEvaluatorSpec


def declared(reduction: JsonValue) -> InferenceEvaluatorSpec:
    return InferenceEvaluatorSpec.model_validate(
        {
            "kind": "inference",
            "interval": {"replicates": 50, "level": 0.95},
            "conventions": CONVENTIONS,
            "reduction": reduction,
            "partitions": {"fit_on": "dev", "score_on": "test"},
            "bins": 2,
            "proof": None,
        }
    )


def test_native_scale_declares_classification_and_abstention_rows() -> None:
    spec = declared(None)

    assert spec.classes() == ("contradicts", "entails", "neutral")
    assert spec.metrics() == (
        "inference-label-3-way",
        "inference-label-3-way-macro-f1",
        "inference-label-3-way-contradicts-precision",
        "inference-label-3-way-entails-precision",
        "inference-label-3-way-neutral-precision",
        "inference-label-3-way-contradicts-recall",
        "inference-label-3-way-entails-recall",
        "inference-label-3-way-neutral-recall",
        "inference-label-3-way-contradicts-f1",
        "inference-label-3-way-entails-f1",
        "inference-label-3-way-neutral-f1",
        "inference-label-3-way-abstention",
        "entailment-coverage",
        "entailment-calibration-ece",
        "entailment-calibration-brier",
        "majority-class",
        "hypothesis-only",
    )
    assert all("micro" not in metric for metric in spec.metrics())


def test_two_way_metrics_require_an_explicit_total_reduction() -> None:
    spec = declared(
        {
            "mapping": {
                "entails": "entails",
                "contradicts": "not-entails",
                "neutral": "not-entails",
            }
        }
    )

    assert spec.classes() == ("entails", "not-entails")
    assert len(spec.metrics()) == 14
    assert "inference-label-2-way-abstention" in spec.metrics()


def test_native_two_way_metrics_come_from_the_data_inventory() -> None:
    spec = declared(None)

    assert spec.evaluation_data() is EvaluationData.INFERENCE_PAIRS
    assert spec.classes(("entails", "non-entails")) == (
        "entails",
        "non-entails",
    )
    assert spec.metrics(("entails", "non-entails"))[0] == ("inference-label-2-way")


def test_a_partial_or_non_two_way_reduction_is_refused() -> None:
    with pytest.raises(dx.ValidationError, match="must coarsen"):
        declared({"mapping": {"entails": "yes", "neutral": "no"}})

    with pytest.raises(dx.ValidationError, match="not two"):
        declared(
            {
                "mapping": {
                    "entails": "one",
                    "contradicts": "two",
                    "neutral": "three",
                }
            }
        )


def test_one_inference_declaration_includes_controls_and_calibration() -> None:
    spec = InferenceEvaluatorSpec.model_validate(
        {
            "kind": "inference",
            "interval": {"replicates": 50, "level": 0.95},
            "conventions": CONVENTIONS,
            "reduction": None,
            "partitions": {"fit_on": "dev", "score_on": "test"},
            "bins": 2,
            "proof": None,
        }
    )
    assert "hypothesis-only" in spec.metrics()
    assert "majority-class" in spec.metrics()
    assert "entailment-calibration-ece" in spec.metrics()
    assert spec.settings().partitions.fit_on == "dev"


def test_inference_refuses_control_fitting_on_the_scored_partition() -> None:
    with pytest.raises(dx.ValidationError, match="disjoint"):
        InferenceEvaluatorSpec.model_validate(
            {
                "kind": "inference",
                "interval": {"replicates": 50, "level": 0.95},
                "conventions": CONVENTIONS,
                "reduction": None,
                "partitions": {"fit_on": "test", "score_on": "test"},
                "bins": 2,
                "proof": None,
            }
        )
