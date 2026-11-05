from typing import TYPE_CHECKING, cast

import didactic.api as dx
import pytest

from lofi.spec import Dependency
from lofi.spec.components.run import EvaluatorSpec
from lofi.spec.evaluators.alignment import AlignmentDeclaration, AlignmentSpec
from lofi.spec.evaluators.conventions import ConventionSpec
from lofi.spec.evaluators.logical_form import (
    LogicalFormEvaluatorSettings,
    LogicalFormEvaluatorSpec,
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
    "kind": "many-to-one",
    "tie_break": "largest-gold-marginal",
    "fit_on": "dev",
    "score_on": "test",
    "random_floor": {"replicates": 31, "level": 0.9},
    "stability": {"replicates": 41, "level": 0.95},
}


def test_the_logical_form_variant_declares_its_complete_term_readout() -> None:
    declared = EvaluatorSpec.model_validate(
        {
            "kind": "logical-form",
            "interval": {"replicates": 50, "level": 0.95},
            "readout": "decoded-forms",
            "measure": "exact",
            "conventions": CONVENTIONS,
            "alignment": ALIGNMENT,
        }
    )

    assert type(declared) is LogicalFormEvaluatorSpec
    assert declared.dependencies() == (Dependency("decoded-forms", "terms-readout"),)
    assert declared.metrics() == (
        "logical-form-exact",
        "many-to-one-accuracy",
        "random-relabeling-floor",
        "fiber-relative-recovery",
        "fixed-vocabulary-accuracy",
        "fiber-floor",
        "fiber-size",
    )
    assert declared.settings() == LogicalFormEvaluatorSettings(
        "decoded-forms",
        "exact",
        4,
        5_040,
        ConventionSpec.model_validate(CONVENTIONS).settings(),
        cast(
            "AlignmentDeclaration", AlignmentSpec.model_validate(ALIGNMENT)
        ).settings(),
    )


@pytest.mark.parametrize(
    "payload",
    [
        {
            "kind": "logical-form",
            "interval": {"replicates": 50, "level": 0.95},
            "measure": "exact",
            "conventions": CONVENTIONS,
            "alignment": ALIGNMENT,
        },
        {
            "kind": "logical-form",
            "interval": {"replicates": 50, "level": 0.95},
            "readout": "",
            "measure": "exact",
            "conventions": CONVENTIONS,
            "alignment": ALIGNMENT,
        },
        {
            "kind": "logical-form",
            "interval": {"replicates": 50, "level": 0.95},
            "readout": "decoded-forms",
            "measure": "beta",
            "conventions": CONVENTIONS,
            "alignment": ALIGNMENT,
        },
        {
            "kind": "logical-form",
            "interval": {"replicates": 50, "level": 0.95},
            "readout": "decoded-forms",
            "measure": "exact",
            "restarts": -1,
            "conventions": CONVENTIONS,
            "alignment": ALIGNMENT,
        },
        {
            "kind": "logical-form",
            "interval": {"replicates": 50, "level": 0.95},
            "readout": "decoded-forms",
            "measure": "exact",
            "permutation_limit": 0,
            "conventions": CONVENTIONS,
            "alignment": ALIGNMENT,
        },
        {
            "kind": "logical-form",
            "interval": {"replicates": 50, "level": 0.95},
            "readout": "decoded-forms",
            "measure": "exact",
            "conventions": CONVENTIONS,
            "alignment": ALIGNMENT,
            "beam": 4,
        },
        {
            "kind": "logical-form",
            "interval": {"replicates": 50, "level": 0.95},
            "readout": "decoded-forms",
            "measure": "exact",
            "conventions": CONVENTIONS,
        },
    ],
)
def test_an_incomplete_unknown_or_extra_setting_is_refused(
    payload: dict[str, JsonValue],
) -> None:
    with pytest.raises(dx.ValidationError):
        EvaluatorSpec.model_validate(payload)


def test_the_variant_round_trips_through_its_json_shape() -> None:
    declared = LogicalFormEvaluatorSpec.model_validate(
        {
            "kind": "logical-form",
            "interval": {"replicates": 50, "level": 0.95},
            "readout": "decoded-forms",
            "measure": "smatch",
            "conventions": CONVENTIONS,
            "alignment": ALIGNMENT,
        }
    )

    assert EvaluatorSpec.model_validate_json(declared.model_dump_json()) == declared
