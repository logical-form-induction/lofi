from typing import TYPE_CHECKING, cast

import didactic.api as dx
import pytest

from lofi.spec import Dependency
from lofi.spec.components.run import EvaluatorSpec
from lofi.spec.evaluators.alignment import AlignmentDeclaration, AlignmentSpec
from lofi.spec.evaluators.conventions import ConventionSpec
from lofi.spec.evaluators.lexical_types import (
    LexicalTypeEvaluatorSettings,
    LexicalTypeEvaluatorSpec,
)

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
ALIGNMENT: dict[str, JsonValue] = {
    "kind": "one-to-one",
    "tie_break": "least-gold-label",
    "fit_on": "dev",
    "score_on": "test",
    "random_floor": {"replicates": 31, "level": 0.9},
    "stability": {"replicates": 41, "level": 0.95},
}


def test_the_lexical_variant_declares_its_term_readout_dependency() -> None:
    declared = EvaluatorSpec.model_validate(
        {
            "kind": "lexical-types",
            "interval": {"replicates": 50, "level": 0.95},
            "readout": "types",
            "conventions": CONVENTIONS,
            "alignment": ALIGNMENT,
        }
    )

    assert type(declared) is LexicalTypeEvaluatorSpec
    assert declared.readout == "types"
    assert declared.dependencies() == (Dependency("types", "terms-readout"),)
    assert declared.metrics() == (
        "lexical-type-f1",
        "primitive-one-to-one-accuracy",
        "primitive-random-relabeling-floor",
        "constructor-one-to-one-accuracy",
        "constructor-random-relabeling-floor",
    )


@pytest.mark.parametrize(
    "payload",
    [
        {
            "kind": "lexical-types",
            "interval": {"replicates": 50, "level": 0.95},
            "conventions": CONVENTIONS,
            "alignment": ALIGNMENT,
        },
        {
            "kind": "lexical-types",
            "interval": {"replicates": 50, "level": 0.95},
            "readout": "",
            "conventions": CONVENTIONS,
            "alignment": ALIGNMENT,
        },
        {
            "kind": "lexical-types",
            "interval": {"replicates": 50, "level": 0.95},
            "readout": "types",
            "alignment": ALIGNMENT,
        },
        {
            "kind": "lexical-types",
            "interval": {"replicates": 50, "level": 0.95},
            "readout": "types",
            "conventions": CONVENTIONS,
            "alignment": ALIGNMENT,
            "beam": 4,
        },
        {
            "kind": "lexical-types",
            "interval": {"replicates": 50, "level": 0.95},
            "readout": "types",
            "conventions": CONVENTIONS,
        },
    ],
)
def test_a_missing_empty_or_extra_field_is_refused(
    payload: dict[str, JsonValue],
) -> None:
    with pytest.raises(dx.ValidationError):
        EvaluatorSpec.model_validate(payload)


def test_the_variant_round_trips_through_its_json_shape() -> None:
    declared = EvaluatorSpec.model_validate(
        {
            "kind": "lexical-types",
            "interval": {"replicates": 50, "level": 0.95},
            "readout": "types",
            "conventions": CONVENTIONS,
            "alignment": ALIGNMENT,
        }
    )

    assert EvaluatorSpec.model_validate_json(declared.model_dump_json()) == declared


def test_the_variant_reports_the_evaluator_family_and_its_settings() -> None:
    declared = LexicalTypeEvaluatorSpec.model_validate(
        {
            "kind": "lexical-types",
            "interval": {"replicates": 50, "level": 0.95},
            "readout": "lexicon",
            "conventions": CONVENTIONS,
            "alignment": ALIGNMENT,
        }
    )

    assert LexicalTypeEvaluatorSpec.family() == "evaluator"
    assert declared.settings() == LexicalTypeEvaluatorSettings(
        readout="lexicon",
        conventions=ConventionSpec.model_validate(CONVENTIONS).settings(),
        alignment=cast(
            "AlignmentDeclaration", AlignmentSpec.model_validate(ALIGNMENT)
        ).settings(),
    )
