from typing import TYPE_CHECKING, Final

import pytest
from tests.acceptance.driver import composed
from tests.conformance.evaluator import (
    HeldOutEvaluatorSuite,
    HeldOutPass,
    HeldOutSubject,
    held_out_run,
)

from lofi.eval import ConventionVector
from lofi.eval._equivalence_evaluator import PLUGIN, EquivalenceEvaluator
from lofi.lf.provers import ProofBudget
from lofi.registry import plugin_for
from lofi.spec.components.run import EvaluatorSpec
from lofi.spec.evaluators.equivalence import EquivalenceEvaluatorSpec

if TYPE_CHECKING:
    from lofi.terms import Term

CONVENTIONS: Final = ConventionVector(
    punctuation="keep",
    unary_chains="keep",
    trivial_spans="keep",
    minimum_length=1,
    aggregation="corpus",
    coverage="zero",
    self_arcs="allow",
    gold_derivation="canonical",
)


@pytest.fixture(scope="module")
def interpreted() -> HeldOutPass:
    return held_out_run(composed("meaning-evaluator.yaml"))


def evaluator(interpreted: HeldOutPass, /) -> EquivalenceEvaluator:
    return EquivalenceEvaluator(
        interpreted.context,
        readout="decoded-forms",
        budget=ProofBudget(),
        conventions=CONVENTIONS,
    )


class TestTheEquivalenceEvaluatorObeysTheLaws(HeldOutEvaluatorSuite):
    @pytest.fixture
    def held_out_subject(self, interpreted: HeldOutPass) -> HeldOutSubject[str, Term]:
        return HeldOutSubject(evaluator(interpreted), interpreted)


def test_the_evaluator_reports_the_rows_its_declaration_names(
    interpreted: HeldOutPass,
) -> None:
    declaration = EvaluatorSpec.model_validate(
        {
            "kind": "equivalence",
            "interval": {"replicates": 20, "level": 0.95},
            "readout": "decoded-forms",
            "budget": {"calls_per_step": 4},
            "conventions": {
                "punctuation": "keep",
                "unary_chains": "keep",
                "trivial_spans": "keep",
                "minimum_length": 1,
                "aggregation": "corpus",
                "coverage": "zero",
                "self_arcs": "allow",
                "gold_derivation": "canonical",
            },
        }
    )
    assert type(declaration) is EquivalenceEvaluatorSpec
    built = PLUGIN.construct(declaration, interpreted.context)

    assert plugin_for(declaration) is PLUGIN
    assert isinstance(built, EquivalenceEvaluator)
    assert tuple(metric.name for metric in built.metrics) == declaration.metrics()
    assert built.conventions.factors["proof_calls_per_step"] == 4


def test_predictions_are_root_forms_written_in_the_run_language(
    interpreted: HeldOutPass,
) -> None:
    built = evaluator(interpreted)
    interpretation = interpreted.context.interpretation
    assert interpretation is not None

    for outputs in interpreted.published:
        predictions = built.predict(outputs)

        assert set(predictions) == set(outputs.batch.item_ids)
        assert all(
            interpretation.render(interpretation.read_text(text)) == text
            for text in predictions.values()
        )


def test_gold_forms_are_read_from_the_corpus_boundary(
    interpreted: HeldOutPass,
) -> None:
    built = evaluator(interpreted)

    for item, derivation in interpreted.gold:
        assert built.annotate(item, derivation).is_closed
