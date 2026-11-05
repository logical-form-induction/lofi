from typing import TYPE_CHECKING

import didactic.api as dx
import pytest

from lofi.lf.provers import ProofBudget
from lofi.spec import Dependency
from lofi.spec.components.run import EvaluatorSpec
from lofi.spec.evaluators.equivalence import (
    EQUIVALENCE_ROWS,
    EquivalenceEvaluatorSettings,
    EquivalenceEvaluatorSpec,
)

if TYPE_CHECKING:
    from lofi._types import JsonValue


def document() -> dict[str, JsonValue]:
    """Return an equivalence declaration with a bounded call budget."""
    return {
        "kind": "equivalence",
        "interval": {"replicates": 20, "level": 0.95},
        "readout": "decoded-forms",
        "budget": {"calls_per_step": 0},
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


def test_a_document_declares_the_rates_and_service_counts() -> None:
    declared = EvaluatorSpec.model_validate(document())

    assert type(declared) is EquivalenceEvaluatorSpec
    assert declared.dependencies() == (Dependency("decoded-forms", "terms-readout"),)
    assert declared.metrics() == EQUIVALENCE_ROWS
    settings = declared.settings()
    assert isinstance(settings, EquivalenceEvaluatorSettings)
    assert settings.budget == ProofBudget(calls_per_step=0)


@pytest.mark.parametrize(
    ("field", "value"),
    [("readout", ""), ("budget", {"calls_per_step": -1}), ("budget", None)],
)
def test_an_unnamed_readout_or_invalid_budget_is_refused(
    field: str, value: JsonValue
) -> None:
    payload = document()
    payload[field] = value

    with pytest.raises(dx.ValidationError):
        EvaluatorSpec.model_validate(payload)


def test_the_declaration_round_trips_through_its_json_shape() -> None:
    declared = EvaluatorSpec.model_validate(document())

    assert EvaluatorSpec.model_validate_json(declared.model_dump_json()) == declared
