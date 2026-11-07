import pytest
from tests.acceptance.driver import composed
from tests.conformance.evaluator import held_out_run

from lofi._types import ItemId
from lofi.deduce import ItemKey
from lofi.eval._coherence_square_evaluator import (
    CoherencePrediction,
    CoherenceSquareEvaluator,
    _CoherenceMetric,
    _selected,
)
from lofi.eval._values import ConventionVector
from lofi.spec.evaluators.coherence_square import CoherenceSquareSpec

CONVENTIONS = ConventionVector(
    punctuation="keep",
    unary_chains="keep",
    trivial_spans="keep",
    minimum_length=0,
    aggregation="sentence",
    coverage="zero",
    self_arcs="allow",
    gold_derivation="canonical",
)


def test_off_sample_and_all_select_exactly_the_declared_chart_items() -> None:
    keys = tuple(ItemKey((0, 1), name) for name in ("a", "b", "c", "d"))

    assert _selected(keys, mode="off", count=None, seed=2, item_id=ItemId("x")) == ()
    sampled = _selected(keys, mode="sample", count=2, seed=2, item_id=ItemId("x"))
    assert len(sampled) == 2
    assert sampled == _selected(
        keys, mode="sample", count=2, seed=2, item_id=ItemId("x")
    )
    assert _selected(keys, mode="all", count=None, seed=2, item_id=ItemId("x")) == keys


def test_coherence_and_collapse_rows_use_derivation_specific_counts() -> None:
    prediction = CoherencePrediction(4, 3, 8, 2, 5)
    expected = {
        "coherence-square-rate": 0.75,
        "coherence-square-items": 4.0,
        "coherence-square-instances": 8.0,
        "coherence-square-strict-rate": 0.25,
        "coherence-square-collapse-rate": 0.375,
    }
    for name, value in expected.items():
        metric = _CoherenceMetric(name, CONVENTIONS)
        metric.update(prediction, ItemId("one"), key=ItemId("one"))
        assert metric.compute().value == value


def test_sampled_square_checks_interpretable_items_without_fabricated_forms() -> None:
    held_out = held_out_run(composed("meaning-evaluator.yaml"))
    settings = CoherenceSquareSpec.model_validate(
        {
            "kind": "coherence-square",
            "interval": {"replicates": 20, "level": 0.95},
            "readout": "category-terms",
            "astar": {},
            "decode": {"kind": "viterbi"},
            "mode": "sample",
            "sample_items": 1,
            "seed": 0,
            "conventions": {
                "punctuation": "keep",
                "unary_chains": "keep",
                "trivial_spans": "keep",
                "minimum_length": 0,
                "aggregation": "sentence",
                "coverage": "zero",
                "self_arcs": "allow",
                "gold_derivation": "canonical",
            },
        }
    ).settings()
    evaluator = CoherenceSquareEvaluator(held_out.context, settings)

    for outputs in held_out.published:
        predictions = evaluator.predict(outputs)

        assert set(predictions) == set(outputs.batch.item_ids)
        assert all(prediction.selected == 1 for prediction in predictions.values())


def test_a_row_name_no_reading_declares_is_refused() -> None:
    with pytest.raises(ValueError, match="no coherence-square row is named"):
        _CoherenceMetric("coherence-square-ratio", CONVENTIONS)
