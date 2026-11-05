from typing import Final

import pytest
import torch
from tests.conformance.evaluator import packaged_held_out_pass

from lofi.eval._bracketing import Bracketing
from lofi.eval._category_side import (
    PLUGIN,
    CategorySideEvaluator,
    CategorySidePrediction,
)
from lofi.eval._lexical_diagnostics import LexicalDiagnosticPrediction
from lofi.eval._structural_diagnostics import StructuralPrediction
from lofi.nn.forms import Estimate, WeightedTerms
from lofi.spec.evaluators.category_side import (
    CategorySideEvaluatorSettings,
    CategorySideEvaluatorSpec,
)

DECLARATION: Final = CategorySideEvaluatorSpec.model_validate(
    {
        "kind": "category-side",
        "interval": {"replicates": 20, "level": 0.95},
        "readout": "category-terms",
        "supertag_beta": 0.1,
        "astar": {"proposer": {"kind": "supertag-beam"}},
        "decode": {"kind": "viterbi"},
        "dependency_interpretation": "dependency",
        "dependency_mode": "labeled-directed",
        "conventions": {
            "punctuation": "keep",
            "unary_chains": "keep",
            "trivial_spans": "keep",
            "minimum_length": 0,
            "aggregation": "corpus",
            "coverage": "zero",
            "self_arcs": "forbid",
            "gold_derivation": "canonical",
        },
        "alignment": {
            "kind": "identity",
            "fit_on": "dev",
            "score_on": "test",
            "random_floor": {"replicates": 11, "level": 0.9},
            "stability": {"replicates": 13, "level": 0.9},
        },
    }
)


def _evaluator() -> CategorySideEvaluator:
    held_out = packaged_held_out_pass()
    return CategorySideEvaluator(held_out.context, DECLARATION.settings())


def test_the_plugin_reads_every_declared_category_side_choice() -> None:
    built = PLUGIN.construct(DECLARATION, packaged_held_out_pass().context)
    factors = built.conventions.factors

    assert PLUGIN.spec is CategorySideEvaluatorSpec
    assert factors["category_readout"] == "category-terms"
    assert factors["astar_proposer"] == "supertag-beam"
    assert factors["dependency_mode"] == "labeled-directed"
    assert factors["alignment"] == "identity"
    assert built.alignment.fit_on == "dev"
    assert "readout='category-terms'" in repr(built)


def test_one_published_pass_supplies_all_three_category_predictions() -> None:
    held_out = packaged_held_out_pass()
    evaluator = _evaluator()

    for outputs in held_out.published:
        predictions = evaluator.predict(outputs)

        assert set(predictions) == set(outputs.batch.item_ids)
        assert all(
            prediction.supertags.token_beams for prediction in predictions.values()
        )
        assert all(prediction.lexical_types for prediction in predictions.values())


def test_a_gold_derivation_control_scores_every_headline_at_ceiling() -> None:
    held_out = packaged_held_out_pass()
    evaluator = _evaluator()
    gold = {
        item.item_id: evaluator.annotate(item, derivation)
        for item, derivation in held_out.gold
    }
    controls = {
        item_id: CategorySidePrediction(
            LexicalDiagnosticPrediction(
                analysis.supertags.words,
                tuple(
                    WeightedTerms(
                        (category,),
                        torch.tensor([0.0]),
                        torch.tensor(float("-inf")),
                        Estimate.EXACT,
                    )
                    for category in analysis.supertags.categories
                ),
                (),
            ),
            StructuralPrediction(
                Bracketing(
                    analysis.structure.bracketing.length,
                    tuple(
                        constituent.span
                        for constituent in analysis.structure.bracketing.constituents
                    ),
                ),
                analysis.structure.attachment,
                analysis.structure.dependency,
                0,
                0,
                None,
            ),
            analysis.lexical_types.categories,
        )
        for item_id, analysis in gold.items()
    }
    ordered = tuple(sorted(gold))
    dev = ordered[:2]
    test = ordered[2:]

    evaluator.fit_alignment(
        {item_id: controls[item_id] for item_id in dev},
        {item_id: gold[item_id] for item_id in dev},
    )
    for item_id in test:
        for metric in evaluator.metrics:
            metric.update(controls[item_id], gold[item_id], key=item_id)

    results = tuple(metric.compute() for metric in evaluator.metrics)
    alignment = evaluator.score_alignment(
        {item_id: controls[item_id] for item_id in test},
        {item_id: gold[item_id] for item_id in test},
        seed=7,
    )

    assert (
        tuple(metric.name for metric in evaluator.metrics) == DECLARATION.metrics()[:3]
    )
    assert all(result.value == pytest.approx(1.0) for result in results)
    assert alignment.primitives.observed.value == pytest.approx(1.0)
    assert alignment.constructors is not None
    assert alignment.constructors.observed.value == pytest.approx(1.0)


def test_metrics_are_unavailable_until_development_alignment_is_frozen() -> None:
    evaluator = _evaluator()

    with pytest.raises(ValueError, match="alignment fitted on development"):
        _ = evaluator.metrics


def test_settings_remain_immutable_below_the_document_boundary() -> None:
    settings = DECLARATION.settings()
    field = "readout"

    assert isinstance(settings, CategorySideEvaluatorSettings)
    with pytest.raises(AttributeError):
        setattr(settings, field, "other")
