from typing import TYPE_CHECKING

import didactic.api as dx
import pytest

from lofi.spec.components.run import EvaluatorSpec
from lofi.spec.evaluators.bracketing import (
    BracketingEvaluatorSettings,
    BracketingEvaluatorSpec,
)
from lofi.spec.evaluators.conventions import ConventionSpec

if TYPE_CHECKING:
    from lofi._types import JsonValue

CONVENTIONS: dict[str, JsonValue] = {
    "punctuation": "strip",
    "unary_chains": "collapse",
    "trivial_spans": "drop",
    "minimum_length": 3,
    "aggregation": "sentence",
    "coverage": "zero",
    "self_arcs": "allow",
    "gold_derivation": "canonical",
}


def test_the_bracketing_variant_declares_its_conventions_and_no_dependency() -> None:
    declared = EvaluatorSpec.model_validate(
        {
            "kind": "bracketing",
            "interval": {"replicates": 50, "level": 0.95},
            "conventions": CONVENTIONS,
            "punctuation": {"source": "surface", "forms": [",", "."]},
            "decoder": "viterbi",
        }
    )

    assert type(declared) is BracketingEvaluatorSpec
    assert declared.conventions == ConventionSpec.model_validate(CONVENTIONS)
    assert declared.dependencies() == ()
    assert declared.declared_output is None
    assert declared.metrics() == ("unlabeled-bracketing-f1",)


@pytest.mark.parametrize(
    "payload",
    [
        {"kind": "bracketing", "interval": {"replicates": 50, "level": 0.95}},
        {
            "kind": "bracketing",
            "interval": {"replicates": 50, "level": 0.95},
            "conventions": CONVENTIONS,
            "punctuation": {"source": "none", "forms": []},
            "decoder": "minimum-risk",
            "labels": True,
        },
        {
            "kind": "bracketing",
            "interval": {"replicates": 50, "level": 0.95},
            "conventions": CONVENTIONS,
            "punctuation": {"source": "surface", "forms": []},
            "decoder": "minimum-risk",
        },
        {
            "kind": "bracketing",
            "interval": {"replicates": 50, "level": 0.95},
            "conventions": CONVENTIONS,
            "punctuation": {"source": "none", "forms": []},
            "decoder": "minimum-risk",
            "labeled": True,
        },
    ],
)
def test_missing_conventions_or_an_extra_field_is_refused(
    payload: dict[str, JsonValue],
) -> None:
    with pytest.raises(dx.ValidationError):
        EvaluatorSpec.model_validate(payload)


def test_the_variant_round_trips_through_its_json_shape() -> None:
    declared = EvaluatorSpec.model_validate(
        {
            "kind": "bracketing",
            "interval": {"replicates": 50, "level": 0.95},
            "conventions": CONVENTIONS,
            "punctuation": {"source": "none", "forms": []},
            "decoder": "minimum-risk",
        }
    )

    assert EvaluatorSpec.model_validate_json(declared.model_dump_json()) == declared


def test_the_variant_reports_the_evaluator_family_and_its_settings() -> None:
    declared = BracketingEvaluatorSpec.model_validate(
        {
            "kind": "bracketing",
            "interval": {"replicates": 50, "level": 0.95},
            "conventions": CONVENTIONS,
            "punctuation": {"source": "surface", "forms": [",", "."]},
            "decoder": "viterbi",
        }
    )

    assert BracketingEvaluatorSpec.family() == "evaluator"
    assert declared.settings() == BracketingEvaluatorSettings(
        conventions=ConventionSpec.model_validate(CONVENTIONS).settings(),
        punctuation_source="surface",
        punctuation_forms=(",", "."),
        decoder="viterbi",
    )
