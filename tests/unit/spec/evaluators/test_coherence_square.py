from typing import TYPE_CHECKING

import didactic.api as dx
import pytest

from lofi.spec.components.run import ParseSpec, ViterbiSemiringSpec
from lofi.spec.evaluators.coherence_square import CoherenceSquareSpec

if TYPE_CHECKING:
    from lofi._types import JsonValue


def document(mode: str, sample_items: int | None) -> dict[str, JsonValue]:
    return {
        "kind": "coherence-square",
        "interval": {"replicates": 20, "level": 0.95},
        "readout": "types",
        "astar": {},
        "decode": {"kind": "viterbi"},
        "mode": mode,
        "sample_items": sample_items,
        "seed": 7,
        "conventions": {
            "punctuation": "keep",
            "unary_chains": "keep",
            "trivial_spans": "keep",
            "minimum_length": 0,
            "aggregation": "sentence",
            "coverage": "zero",
            "self_arcs": "allow",
            "gold_derivation": "canonical",
        },
    }


@pytest.mark.parametrize(
    ("mode", "sample_items"), [("off", None), ("sample", 3), ("all", None)]
)
def test_all_three_coherence_policies_are_current_declarations(
    mode: str, sample_items: int | None
) -> None:
    spec = CoherenceSquareSpec.model_validate(document(mode, sample_items))

    assert spec.settings().mode == mode
    assert spec.settings().metrics == spec.metrics()


@pytest.mark.parametrize(
    ("mode", "sample_items"), [("sample", None), ("off", 2), ("all", 2)]
)
def test_coherence_refuses_implicit_or_unused_sampling(
    mode: str, sample_items: int | None
) -> None:
    with pytest.raises(dx.ValidationError, match="sample"):
        CoherenceSquareSpec.model_validate(document(mode, sample_items))


def test_coherence_squares_decode_with_the_run_parse() -> None:
    viterbi = CoherenceSquareSpec.model_validate(document("off", None))
    payload = document("off", None)
    payload["decode"] = {"kind": "mbr", "candidates": 4}
    mbr = CoherenceSquareSpec.model_validate(payload)
    parse = ParseSpec(
        retain="forest", reachability="all", semiring=ViterbiSemiringSpec()
    )

    assert viterbi.parse_refusal(parse) is None
    assert "does not supply" in str(mbr.parse_refusal(parse))
