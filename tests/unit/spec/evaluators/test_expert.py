from typing import TYPE_CHECKING

import didactic.api as dx
import pytest

from lofi.spec.components.run import EvaluatorSpec
from lofi.spec.evaluators.expert import (
    ExpertCategoryEvaluatorSettings,
    ExpertCategoryEvaluatorSpec,
)

if TYPE_CHECKING:
    from lofi._types import JsonValue

CONVENTIONS: dict[str, JsonValue] = {
    "punctuation": "keep",
    "unary_chains": "keep",
    "trivial_spans": "keep",
    "minimum_length": 0,
    "aggregation": "corpus",
    "coverage": "zero",
    "self_arcs": "allow",
    "gold_derivation": "canonical",
}
ALIGNMENT: dict[str, JsonValue] = {
    "kind": "many-to-one",
    "tie_break": "largest-gold-marginal",
    "fit_on": "dev",
    "score_on": "test",
    "random_floor": {"replicates": 50, "level": 0.95},
    "stability": {"replicates": 50, "level": 0.95},
}


def _payload() -> dict[str, JsonValue]:
    return {
        "kind": "expert-categories",
        "interval": {"replicates": 100, "level": 0.95},
        "readout": "terms",
        "ratings": {"kind": "expert-ratings", "source": "ratings.json"},
        "taxonomy": ["wrong-arity", "wrong-result"],
        "conventions": CONVENTIONS,
        "alignment": ALIGNMENT,
    }


def test_the_expert_variant_declares_its_rows_and_settings() -> None:
    declared = EvaluatorSpec.model_validate(_payload())

    assert type(declared) is ExpertCategoryEvaluatorSpec
    assert declared.metrics() == (
        "expert-rating-alpha",
        "expert-error-wrong-arity",
        "expert-error-wrong-result",
    )
    settings = declared.settings()
    assert isinstance(settings, ExpertCategoryEvaluatorSettings)
    assert settings.readout == "terms"
    assert settings.ratings is not None
    assert settings.ratings.source == "ratings.json"
    assert settings.taxonomy == ("wrong-arity", "wrong-result")
    assert settings.alignment.fit_on == "dev"
    assert tuple(
        (dependency.role, dependency.port) for dependency in declared.dependencies()
    ) == (("terms", "terms-readout"),)


@pytest.mark.parametrize(
    "update",
    [
        {"taxonomy": []},
        {"taxonomy": ["wrong-arity", "wrong-arity"]},
        {"readout": ""},
        {"ratings": {"kind": "expert-ratings", "source": ""}},
    ],
)
def test_empty_and_duplicate_names_are_refused(update: dict[str, JsonValue]) -> None:
    payload = _payload() | update

    with pytest.raises(dx.ValidationError):
        EvaluatorSpec.model_validate(payload)


def test_the_expert_variant_round_trips_through_json() -> None:
    declared = EvaluatorSpec.model_validate(_payload())

    assert EvaluatorSpec.model_validate_json(declared.model_dump_json()) == declared


def test_omitting_ratings_declares_annotation_export_mode() -> None:
    payload = _payload()
    del payload["ratings"]

    declared = EvaluatorSpec.model_validate(payload)

    assert isinstance(declared, ExpertCategoryEvaluatorSpec)
    assert declared.settings().ratings is None
