"""Tests for the lexical diagnostic evaluator declaration."""

from typing import TYPE_CHECKING

import didactic.api as dx
import pytest

from lofi.spec import Dependency
from lofi.spec.components.run import EvaluatorSpec
from lofi.spec.evaluators.lexical_diagnostics import LexicalDiagnosticsSpec

if TYPE_CHECKING:
    from lofi._types import JsonValue


CONVENTIONS: dict[str, JsonValue] = {
    "punctuation": "keep",
    "unary_chains": "keep",
    "trivial_spans": "keep",
    "minimum_length": 0,
    "aggregation": "sentence",
    "coverage": "zero",
    "self_arcs": "allow",
    "gold_derivation": "canonical",
}


def _document() -> dict[str, JsonValue]:
    return {
        "kind": "lexical-diagnostics",
        "interval": {"replicates": 50, "level": 0.95},
        "readout": "category-terms",
        "beta": 0.1,
        "clustering_beta": 2.0,
        "top_k": 3,
        "frequency_strata": {
            "frequent": {"at_least": 2, "at_most": None},
            "rare": {"at_least": 1, "at_most": 1},
            "unseen": {"at_least": 0, "at_most": 0},
        },
        "ambiguity_strata": {
            "ambiguous": {"at_least": 2, "at_most": None},
            "unambiguous": {"at_least": 1, "at_most": 1},
            "unseen": {"at_least": 0, "at_most": 0},
        },
        "word_classes": {"nouns": ["word-0"], "predicates": ["word-1"]},
        "case_studies": {
            "lexical-class": {
                "nouns": ["word-0"],
                "predicates": ["word-1"],
            }
        },
        "conventions": CONVENTIONS,
    }


def test_the_variant_declares_every_metric_and_parameter() -> None:
    declared = EvaluatorSpec.model_validate(_document())

    assert type(declared) is LexicalDiagnosticsSpec
    assert declared.dependencies() == (Dependency("category-terms", "terms-readout"),)
    assert declared.metrics() == (
        "supertag-accuracy",
        "supertag-multi-accuracy",
        "supertag-ambiguity",
        "supertag-confusion",
        "clustering-homogeneity",
        "clustering-completeness",
        "clustering-v-measure",
        "clustering-adjusted-mutual-information",
        "clustering-adjusted-rand",
        "supertag-accuracy-frequency-frequent",
        "supertag-accuracy-frequency-rare",
        "supertag-accuracy-frequency-unseen",
        "supertag-accuracy-ambiguity-ambiguous",
        "supertag-accuracy-ambiguity-unambiguous",
        "supertag-accuracy-ambiguity-unseen",
        "lexicon-mass-nouns",
        "lexicon-mass-predicates",
        "case-study-lexical-class-nouns",
        "case-study-lexical-class-predicates",
        "collapse-rate",
        "decoded-residual-mass",
        "decode-refusal-rate",
    )
    assert declared.settings().word_classes == (
        ("nouns", ("word-0",)),
        ("predicates", ("word-1",)),
    )


@pytest.mark.parametrize(
    "change",
    [
        {"readout": ""},
        {"beta": 1.0},
        {"clustering_beta": -0.1},
        {"top_k": 0},
        {"frequency_strata": {}},
        {
            "frequency_strata": {
                "low": {"at_least": 0, "at_most": 1},
                "high": {"at_least": 3, "at_most": None},
            }
        },
        {
            "ambiguity_strata": {
                "first": {"at_least": 0, "at_most": None},
                "overlap": {"at_least": 1, "at_most": None},
            }
        },
        {"word_classes": {}},
        {"word_classes": {"a": ["word-0"], "b": ["word-0"]}},
        {"case_studies": {"bad": {"only": ["word-0"]}}},
        {"case_studies": {"bad": {"left": ["word-0"], "right": ["word-0"]}}},
        {"beam": 4},
    ],
)
def test_invalid_or_superseded_routes_are_refused(
    change: dict[str, JsonValue],
) -> None:
    document = _document()
    document.update(change)

    with pytest.raises(dx.ValidationError):
        EvaluatorSpec.model_validate(document)


def test_the_variant_round_trips_through_its_json_shape() -> None:
    declared = EvaluatorSpec.model_validate(_document())

    assert EvaluatorSpec.model_validate_json(declared.model_dump_json()) == declared


def test_case_studies_are_omitted_when_the_data_cannot_locate_byte_spans() -> None:
    document = _document()
    document["case_studies"] = {}

    declared = EvaluatorSpec.model_validate(document)

    assert type(declared) is LexicalDiagnosticsSpec
    assert not any(metric.startswith("case-study-") for metric in declared.metrics())
    assert declared.settings().case_studies == ()
