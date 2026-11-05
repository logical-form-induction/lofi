"""Tests for validated LM-plus-prover sources."""

import json
from typing import TYPE_CHECKING, cast

import pytest

from lofi.spec.baselines.lm_prover_trace import (
    read_lm_prover_data,
    read_lm_prover_generations,
)

if TYPE_CHECKING:
    from lofi._types import JsonValue


def _pairs() -> list[dict[str, JsonValue]]:
    return [
        {
            "item_id": "dev-1",
            "split": "dev",
            "premise_tokens": ["a"],
            "hypothesis_tokens": ["a"],
            "label": "entails",
            "gold_premise": "a",
            "gold_hypothesis": "a",
        },
        {
            "item_id": "test-1",
            "split": "test",
            "premise_tokens": ["a"],
            "hypothesis_tokens": ["not", "a"],
            "label": "contradicts",
            "gold_premise": "a",
            "gold_hypothesis": "not(a)",
        },
    ]


def _generation_items() -> list[dict[str, JsonValue]]:
    return [
        {
            "item_id": item_id,
            "candidates": [{"premise": "a", "hypothesis": hypothesis}],
        }
        for item_id, hypothesis in (("dev-1", "a"), ("test-1", "not(a)"))
    ]


def test_data_requires_distinct_items_in_nonempty_dev_and_test_partitions() -> None:
    decoded = read_lm_prover_data(json.dumps({"pairs": _pairs()}))

    assert [pair.item_id for pair in decoded.pairs] == ["dev-1", "test-1"]
    assert [pair.split for pair in decoded.pairs] == ["dev", "test"]

    one_split = _pairs()[:1]
    with pytest.raises(ValueError, match="dev and test"):
        read_lm_prover_data(json.dumps({"pairs": one_split}))

    repeated = [*_pairs(), _pairs()[0]]
    with pytest.raises(ValueError, match="repeats"):
        read_lm_prover_data(json.dumps({"pairs": repeated}))


def test_data_refuses_a_coarsened_output_in_place_of_a_logical_relation() -> None:
    pairs = _pairs()
    pairs[0]["label"] = "non-entails"

    with pytest.raises(ValueError, match="logical relation"):
        read_lm_prover_data(json.dumps({"pairs": pairs}))


def test_generation_trace_records_inputs_but_accepts_no_scores() -> None:
    document = cast(
        "dict[str, JsonValue]",
        {
            "pipeline": "logic-lm",
            "model": "org/model@revision",
            "prompt_digest": f"sha256:{'a' * 64}",
            "items": _generation_items(),
        },
    )
    decoded = read_lm_prover_generations(json.dumps(document))

    assert decoded.pipeline == "logic-lm"
    assert decoded.items[1].candidates[0].hypothesis == "not(a)"

    scored = dict(document)
    scored["accuracy"] = 1.0
    with pytest.raises(ValueError, match="do not validate"):
        read_lm_prover_generations(json.dumps(scored))


def test_generation_trace_requires_a_sha256_prompt_identity() -> None:
    document = {
        "pipeline": "linc",
        "model": "org/model@revision",
        "prompt_digest": "latest",
        "items": _generation_items(),
    }

    with pytest.raises(ValueError, match="SHA-256"):
        read_lm_prover_generations(json.dumps(document))


def test_generation_trace_requires_one_nonempty_candidate_list_per_item() -> None:
    items = _generation_items()
    items[0]["candidates"] = []
    document = {
        "pipeline": "linc",
        "model": "org/model@revision",
        "prompt_digest": f"sha256:{'a' * 64}",
        "items": items,
    }

    with pytest.raises(ValueError, match="at least one candidate"):
        read_lm_prover_generations(json.dumps(document))
