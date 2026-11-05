"""Tests for the registered lexical diagnostic battery."""

from pathlib import Path
from typing import Final, cast

import pytest
from tests.conformance.evaluator import (
    HeldOutEvaluatorSuite,
    HeldOutSubject,
    packaged_held_out_pass,
)

from lofi.eval import ConventionVector
from lofi.eval._lexical_diagnostics import (
    PLUGIN,
    LexicalDiagnosticGold,
    LexicalDiagnosticPrediction,
    LexicalDiagnosticsEvaluator,
)
from lofi.spec import PACKAGED_CONFIG_ROOT, RunSpecCodec
from lofi.spec.evaluators.conventions import ConventionSpec
from lofi.spec.evaluators.lexical_diagnostics import (
    CountStratumSettings,
    LexicalDiagnosticsSettings,
    LexicalDiagnosticsSpec,
)

CONVENTIONS: Final = ConventionVector(
    punctuation="keep",
    unary_chains="keep",
    trivial_spans="keep",
    minimum_length=0,
    aggregation="sentence",
    coverage="zero",
    self_arcs="allow",
    gold_derivation="canonical",
)
CODEC: Final = RunSpecCodec(
    search_path=(PACKAGED_CONFIG_ROOT / "run", PACKAGED_CONFIG_ROOT)
)


def _evaluator() -> LexicalDiagnosticsEvaluator:
    held_out = packaged_held_out_pass()
    return LexicalDiagnosticsEvaluator(
        held_out.context,
        LexicalDiagnosticsSettings(
            readout="category-terms",
            beta=0.1,
            clustering_beta=2.0,
            top_k=2,
            frequency_strata=(("all", CountStratumSettings(0, None)),),
            ambiguity_strata=(("all", CountStratumSettings(0, None)),),
            word_classes=(("nouns", ("word-0",)), ("predicates", ("word-1",))),
            case_studies=(
                (
                    "lexical-class",
                    (("nouns", ("word-0",)), ("predicates", ("word-1",))),
                ),
            ),
            conventions=ConventionSpec.model_validate(
                dict(CONVENTIONS.factors)
            ).settings(),
            metrics=(
                "supertag-accuracy",
                "supertag-multi-accuracy",
                "supertag-ambiguity",
                "supertag-confusion",
                "clustering-homogeneity",
                "clustering-completeness",
                "clustering-v-measure",
                "clustering-adjusted-mutual-information",
                "clustering-adjusted-rand",
                "supertag-accuracy-frequency-all",
                "supertag-accuracy-ambiguity-all",
                "lexicon-mass-nouns",
                "lexicon-mass-predicates",
                "case-study-lexical-class-nouns",
                "case-study-lexical-class-predicates",
                "collapse-rate",
                "decoded-residual-mass",
                "decode-refusal-rate",
            ),
        ),
    )


class TestTheLexicalDiagnosticsObeyTheLaws(HeldOutEvaluatorSuite):
    @pytest.fixture
    def held_out_subject(
        self,
    ) -> HeldOutSubject[LexicalDiagnosticPrediction, LexicalDiagnosticGold]:
        return HeldOutSubject(_evaluator(), packaged_held_out_pass())


def test_every_declared_metric_family_emits_a_row() -> None:
    held_out = packaged_held_out_pass()
    evaluator = _evaluator()
    predictions = {}

    for outputs in held_out.published:
        predictions.update(evaluator.predict(outputs))

    for item, derivation in held_out.gold:
        predicted = predictions[item.item_id]
        gold = evaluator.annotate(item, derivation)
        for metric in evaluator.metrics:
            metric.update(predicted, gold, key=item.item_id)

    results = tuple(metric.compute() for metric in evaluator.metrics)

    assert len(results) == len(evaluator.metrics)
    assert all(result.n >= 0 for result in results)
    assert {
        result.resampling.unit
        for metric, result in zip(evaluator.metrics, results, strict=True)
        if metric.name.startswith("case-study-")
    } == {"sentence"}
    assert {
        result.resampling.unit
        for metric, result in zip(evaluator.metrics, results, strict=True)
        if not metric.name.startswith("case-study-")
    } == {"token"}
    v_measure = next(
        result
        for metric, result in zip(evaluator.metrics, results, strict=True)
        if metric.name == "clustering-v-measure"
    )
    assert v_measure.components["beta"] == 2.0
    assert {metric.name for metric in evaluator.metrics} >= {
        "supertag-confusion",
        "clustering-v-measure",
        "lexicon-mass-nouns",
        "case-study-lexical-class-nouns",
        "decode-refusal-rate",
    }
    by_name = {
        metric.name: result
        for metric, result in zip(evaluator.metrics, results, strict=True)
    }
    assert (
        sum(
            result.n
            for name, result in by_name.items()
            if name.startswith("supertag-accuracy-frequency-")
        )
        == by_name["supertag-accuracy"].n
    )
    assert (
        sum(
            result.n
            for name, result in by_name.items()
            if name.startswith("supertag-accuracy-ambiguity-")
        )
        == by_name["supertag-accuracy"].n
    )
    case_study = next(
        result
        for metric, result in zip(evaluator.metrics, results, strict=True)
        if metric.name == "case-study-lexical-class-nouns"
    )
    assert case_study.n > 0
    assert case_study.components["occurrences"] > 0.0
    assert any(name.startswith("mass:") for name in case_study.components)


def test_every_declared_choice_is_part_of_the_convention_vector() -> None:
    factors = dict(_evaluator().conventions.factors)

    assert factors["diagnostic_readout"] == "category-terms"
    assert factors["supertag_beta"] == 0.1
    assert factors["clustering_beta"] == 2.0
    assert factors["lexicon_mass_top_k"] == 2
    assert factors["frequency_strata"] == ('{"all":{"at_least":0,"at_most":null}}')
    assert factors["ambiguity_strata"] == ('{"all":{"at_least":0,"at_most":null}}')
    assert factors["word_classes"] == ('{"nouns":["word-0"],"predicates":["word-1"]}')
    assert factors["case_studies"] == (
        '{"lexical-class":{"nouns":["word-0"],"predicates":["word-1"]}}'
    )


def test_the_plugin_builds_the_single_declared_treatment() -> None:
    path = (
        Path(__file__).parents[3]
        / "lofi/configs/run/synthetic-lexical-diagnostics.yaml"
    )
    declaration = cast(
        "LexicalDiagnosticsSpec",
        CODEC(path).evaluators["lexical-diagnostics"],
    )
    built = PLUGIN.construct(declaration, packaged_held_out_pass().context)

    assert PLUGIN.spec is LexicalDiagnosticsSpec
    assert tuple(metric.name for metric in built.metrics) == declaration.metrics()
    assert built.conventions.factors["diagnostic_readout"] == "category-terms"
