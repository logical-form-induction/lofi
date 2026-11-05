from typing import TYPE_CHECKING

import didactic.api as dx
import pytest

from lofi.spec import Dependency
from lofi.spec.components.run import ParseSpec, ViterbiSemiringSpec
from lofi.spec.evaluators.category_side import CategorySideEvaluatorSpec

if TYPE_CHECKING:
    from lofi._types import JsonValue

CONVENTIONS: dict[str, JsonValue] = {
    "punctuation": "keep",
    "unary_chains": "keep",
    "trivial_spans": "keep",
    "minimum_length": 0,
    "aggregation": "corpus",
    "coverage": "zero",
    "self_arcs": "forbid",
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


def _payload() -> dict[str, JsonValue]:
    return {
        "kind": "category-side",
        "interval": {"replicates": 50, "level": 0.95},
        "readout": "decoded-categories",
        "supertag_beta": 0.1,
        "astar": {"proposer": {"kind": "supertag-beam"}},
        "decode": {"kind": "viterbi"},
        "dependency_interpretation": "dependency",
        "dependency_mode": "labeled-directed",
        "conventions": CONVENTIONS,
        "alignment": ALIGNMENT,
    }


def test_the_category_side_declares_one_category_readout_and_every_row() -> None:
    declared = CategorySideEvaluatorSpec.model_validate(_payload())

    assert declared.dependencies() == (
        Dependency("decoded-categories", "terms-readout"),
    )
    assert declared.fit_split() == "dev"
    assert declared.score_split() == "test"
    assert declared.metrics() == (
        "supertag-accuracy",
        "dependency:labeled-directed:zero:micro-f-score",
        "lexical-type-f1",
        "primitive-one-to-one-accuracy",
        "primitive-random-relabeling-floor",
        "constructor-one-to-one-accuracy",
        "constructor-random-relabeling-floor",
    )
    settings = declared.settings()
    proposer = settings.astar.proposer
    assert settings.metrics == declared.metrics()
    assert proposer is not None
    assert proposer.name == "supertag-beam"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("readout", ""),
        ("dependency_interpretation", ""),
        ("supertag_beta", -0.1),
        ("supertag_beta", 1.0),
        ("dependency_mode", "unlabeled-directed"),
    ],
)
def test_an_unnamed_or_out_of_range_category_choice_is_refused(
    field: str,
    value: JsonValue,
) -> None:
    payload = _payload()
    payload[field] = value

    with pytest.raises(dx.ValidationError):
        CategorySideEvaluatorSpec.model_validate(payload)


def test_the_category_side_json_shape_round_trips() -> None:
    declared = CategorySideEvaluatorSpec.model_validate(_payload())

    assert (
        CategorySideEvaluatorSpec.model_validate_json(declared.model_dump_json())
        == declared
    )


def test_an_mbr_category_decode_is_refused_a_viterbi_run_parse() -> None:
    viterbi = CategorySideEvaluatorSpec.model_validate(_payload())
    payload = _payload()
    payload["decode"] = {"kind": "mbr", "candidates": 4}
    mbr = CategorySideEvaluatorSpec.model_validate(payload)
    parse = ParseSpec(
        retain="forest", reachability="all", semiring=ViterbiSemiringSpec()
    )

    assert viterbi.parse_refusal(parse) is None
    assert mbr.parse_refusal(ParseSpec(retain="forest", reachability="all")) is None
    assert "does not supply" in str(mbr.parse_refusal(parse))
