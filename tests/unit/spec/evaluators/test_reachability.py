from typing import TYPE_CHECKING

import didactic.api as dx
import pytest

from lofi.spec.components.run import EvaluatorSpec
from lofi.spec.evaluators.reachability import (
    REACHABILITY_ROWS,
    ReachabilityEvaluatorSettings,
    ReachabilityEvaluatorSpec,
)

if TYPE_CHECKING:
    from lofi._types import JsonValue


def document() -> dict[str, JsonValue]:
    """Return a reachability declaration over a named inventory."""
    return {
        "kind": "reachability",
        "interval": {"replicates": 20, "level": 0.95},
        "inventory": "slash",
        "corpus": "synthetic",
        "language": "en",
        "conventions": {
            "punctuation": "keep",
            "unary_chains": "keep",
            "trivial_spans": "keep",
            "minimum_length": 0,
            "aggregation": "corpus",
            "coverage": "zero",
            "self_arcs": "allow",
            "gold_derivation": "canonical",
        },
    }


def test_a_document_declares_every_reachability_row() -> None:
    declared = EvaluatorSpec.model_validate(document())

    assert type(declared) is ReachabilityEvaluatorSpec
    assert declared.dependencies() == ()
    assert declared.metrics() == REACHABILITY_ROWS
    settings = declared.settings()
    assert isinstance(settings, ReachabilityEvaluatorSettings)
    assert (settings.inventory, settings.corpus, settings.language) == (
        "slash",
        "synthetic",
        "en",
    )


@pytest.mark.parametrize("field", ["inventory", "corpus", "language"])
def test_an_empty_report_name_is_refused(field: str) -> None:
    payload = document()
    payload[field] = ""

    with pytest.raises(dx.ValidationError):
        EvaluatorSpec.model_validate(payload)


def test_the_declaration_round_trips_through_its_json_shape() -> None:
    declared = EvaluatorSpec.model_validate(document())

    assert EvaluatorSpec.model_validate_json(declared.model_dump_json()) == declared
