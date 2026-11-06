"""Tests for replayable LM-plus-prover baseline execution."""

import json
from hashlib import sha256
from typing import TYPE_CHECKING, cast

import pytest

from lofi.eval._lm_prover_baseline import _build
from lofi.eval._ports import BaselineBuildContext
from lofi.spec.baselines.lm_prover import LmProverBaselineSpec
from lofi.spec.language import ConstantSpec, registered_language

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from lofi._types import JsonValue


def _pairs() -> list[dict[str, JsonValue]]:
    return [
        {
            "item_id": "d1",
            "split": "dev",
            "premise_tokens": ["p"],
            "hypothesis_tokens": ["p"],
            "label": "entails",
            "gold_premise": "p",
            "gold_hypothesis": "p",
        },
        {
            "item_id": "d2",
            "split": "dev",
            "premise_tokens": ["p"],
            "hypothesis_tokens": ["q"],
            "label": "neutral",
            "gold_premise": "p",
            "gold_hypothesis": "q",
        },
        {
            "item_id": "d3",
            "split": "dev",
            "premise_tokens": ["p"],
            "hypothesis_tokens": ["not", "p"],
            "label": "contradicts",
            "gold_premise": "p",
            "gold_hypothesis": "~p",
        },
        {
            "item_id": "t1",
            "split": "test",
            "premise_tokens": ["p"],
            "hypothesis_tokens": ["p"],
            "label": "entails",
            "gold_premise": "p",
            "gold_hypothesis": "p",
        },
        {
            "item_id": "t2",
            "split": "test",
            "premise_tokens": ["p"],
            "hypothesis_tokens": ["not", "p"],
            "label": "contradicts",
            "gold_premise": "p",
            "gold_hypothesis": "~p",
        },
    ]


def _document(
    pipeline: dict[str, JsonValue], language: JsonValue
) -> dict[str, JsonValue]:
    return {
        "kind": "lm-prover",
        "data_source": "data.json",
        "generations_source": "generations.json",
        "prompt_source": "prompt.txt",
        "model": "org/model@revision",
        "pipeline": pipeline,
        "language": language,
        "budget": {"calls_per_step": 100, "wall_per_step": 30.0},
        "normalization_fuel": 100,
        "measure": "normal-form",
        "alignment": {
            "kind": "one-to-one",
            "tie_break": "least-gold-label",
            "fit_on": "dev",
            "score_on": "test",
            "random_floor": {"replicates": 10, "level": 0.95},
            "stability": {"replicates": 10, "level": 0.95},
        },
        "scoring": {
            "punctuation": "keep",
            "unary_chains": "keep",
            "trivial_spans": "keep",
            "minimum_length": 0,
            "aggregation": "corpus",
            "coverage": "zero",
            "self_arcs": "forbid",
            "gold_derivation": "canonical",
        },
    }


@pytest.mark.parametrize(
    ("pipeline", "attempts"),
    [
        (
            {"kind": "linc", "samples": 3},
            lambda premise, hypothesis: [
                {"premise": premise, "hypothesis": hypothesis},
                {"premise": premise, "hypothesis": hypothesis},
                {"premise": premise, "hypothesis": hypothesis},
            ],
        ),
        (
            {"kind": "logic-lm", "refinements": 1},
            lambda premise, hypothesis: [
                {"premise": "(", "hypothesis": hypothesis},
                {
                    "premise": premise,
                    "hypothesis": hypothesis,
                    "feedback": "the first formalization did not parse",
                },
            ],
        ),
    ],
)
def test_pipeline_replay_scores_forms_labels_and_prover_effort(
    tmp_path: Path,
    pipeline: dict[str, JsonValue],
    attempts: Callable[[str, str], list[dict[str, str]]],
) -> None:
    pairs = _pairs()
    prompt = b"Translate each pair into the declared formal language."
    (tmp_path / "prompt.txt").write_bytes(prompt)
    (tmp_path / "data.json").write_text(
        json.dumps({"pairs": pairs}), encoding="utf-8", newline="\n"
    )
    generations = {
        "pipeline": pipeline["kind"],
        "model": "org/model@revision",
        "prompt_digest": f"sha256:{sha256(prompt).hexdigest()}",
        "items": [
            {
                "item_id": pair["item_id"],
                "candidates": attempts(
                    cast("str", pair["gold_premise"]),
                    cast("str", pair["gold_hypothesis"]),
                ),
            }
            for pair in pairs
        ],
    }
    (tmp_path / "generations.json").write_text(
        json.dumps(generations), encoding="utf-8", newline="\n"
    )
    base = registered_language("propositional")
    language = base.with_(
        constants=(
            *base.constants,
            ConstantSpec(name="p", type="t"),
            ConstantSpec(name="q", type="t"),
        )
    )
    language_document = cast("JsonValue", json.loads(language.model_dump_json()))
    spec = LmProverBaselineSpec.model_validate(_document(pipeline, language_document))

    result = _build(spec.settings(), BaselineBuildContext(tmp_path, 7))()

    assert tuple(row.metric for row in result.measurements) == spec.metrics()
    measured = {row.metric: row.value for row in result.measurements}
    assert measured["logical-form-normal-form"] == 1.0
    assert measured["one-to-one-accuracy"] == 1.0
    assert measured["inference-accuracy"] == 1.0
    assert measured["inference-coverage"] == 1.0
    assert isinstance(measured["prover-calls"], int)
    assert measured["prover-abstention-rate"] == 0.0
    assert set(result.source_digests) == {"data", "generations", "prompt"}


def test_generation_item_set_must_exactly_match_the_data(tmp_path: Path) -> None:
    pairs = _pairs()
    prompt = b"Translate."
    (tmp_path / "prompt.txt").write_bytes(prompt)
    (tmp_path / "data.json").write_text(
        json.dumps({"pairs": pairs}), encoding="utf-8", newline="\n"
    )
    (tmp_path / "generations.json").write_text(
        json.dumps(
            {
                "pipeline": "linc",
                "model": "org/model@revision",
                "prompt_digest": f"sha256:{sha256(prompt).hexdigest()}",
                "items": [
                    {
                        "item_id": pair["item_id"],
                        "candidates": [
                            {
                                "premise": pair["gold_premise"],
                                "hypothesis": pair["gold_hypothesis"],
                            }
                        ],
                    }
                    for pair in pairs[:-1]
                ],
            }
        ),
        encoding="utf-8",
        newline="\n",
    )
    spec = LmProverBaselineSpec.model_validate(
        _document({"kind": "linc", "samples": 1}, "propositional")
    )

    with pytest.raises(ValueError, match="do not exactly match"):
        _build(spec.settings(), BaselineBuildContext(tmp_path, 7))()
