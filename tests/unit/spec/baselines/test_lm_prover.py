"""Tests for the language-model-plus-prover baseline declaration."""

from typing import TYPE_CHECKING, cast

import didactic.api as dx
import pytest

from lofi.spec.baselines.lm_prover import (
    LincPipeline,
    LmProverBaselineSpec,
    LogicLmPipeline,
)

if TYPE_CHECKING:
    from lofi._types import JsonValue


def _document(pipeline: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return {
        "data_source": "data/quantifier-free.json",
        "generations_source": "traces/lm-prover.json",
        "prompt_source": "prompts/quantifier-free.txt",
        "model": "org/model@revision",
        "pipeline": pipeline,
        "language": "propositional",
        "budget": {"calls_per_step": 100, "wall_per_step": 30.0},
        "normalization_fuel": 100,
        "measure": "normal-form",
        "alignment": {
            "kind": "one-to-one",
            "tie_break": "least-gold-label",
            "fit_on": "dev",
            "score_on": "test",
            "random_floor": {"replicates": 100, "level": 0.95},
            "stability": {"replicates": 100, "level": 0.95},
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
    ("pipeline", "expected"),
    [
        ({"kind": "linc", "samples": 5}, ("linc", 5, 0)),
        ({"kind": "logic-lm", "refinements": 3}, ("logic-lm", 1, 3)),
    ],
)
def test_both_pipeline_policies_lower_through_one_current_component(
    pipeline: dict[str, JsonValue], expected: tuple[str, int, int]
) -> None:
    spec = LmProverBaselineSpec.model_validate(_document(pipeline))

    pipeline_settings = spec.settings().pipeline

    assert (
        pipeline_settings.name,
        pipeline_settings.candidates,
        pipeline_settings.refinements,
    ) == expected
    assert spec.metrics() == (
        "logical-form-normal-form",
        "one-to-one-accuracy",
        "random-relabeling-floor",
        "inference-accuracy",
        "inference-coverage",
        "prover-calls",
        "prover-memo-hit-rate",
        "prover-cpu-seconds",
        "prover-abstention-rate",
    )
    assert spec.conventions()["pipeline"] == expected[0]
    assert spec.conventions()["formalization_candidates"] == expected[1]
    assert spec.conventions()["solver_feedback_refinements"] == expected[2]


@pytest.mark.parametrize(
    "pipeline",
    [
        {"kind": "linc", "samples": 0},
        {"kind": "logic-lm", "refinements": -1},
        {"kind": "logic-lm", "refinements": 1, "samples": 2},
    ],
)
def test_pipeline_specific_controls_are_strict(
    pipeline: dict[str, JsonValue],
) -> None:
    with pytest.raises(dx.ValidationError):
        LmProverBaselineSpec.model_validate(_document(pipeline))


def test_language_logic_is_the_only_prover_authority() -> None:
    document = _document({"kind": "linc", "samples": 5})
    document["prover"] = {"kind": "truth-table", "atoms": 4}

    with pytest.raises(dx.ValidationError):
        LmProverBaselineSpec.model_validate(document)

    spec = LmProverBaselineSpec.model_validate(
        _document({"kind": "linc", "samples": 5})
    )

    assert spec.settings().language.logic.prover == spec.language.logic.prover
    prover = cast("dict[str, JsonValue]", spec.conventions()["prover"])
    assert prover["kind"] == "truth-table"


def test_linc_selects_the_unique_majority_label_and_nothing_on_a_tie() -> None:
    linc = LincPipeline(3)

    assert linc.select(("yes", None, "yes")) == 0
    assert linc.select((None, "no", "no")) == 1
    assert linc.select(("yes", "no", None)) is None
    assert linc.select((None, None)) is None


def test_linc_refuses_a_trace_of_another_size_or_with_feedback() -> None:
    linc = LincPipeline(2)
    linc.check_trace("a", (False, False), (True, False))

    with pytest.raises(ValueError, match="records 1 candidates"):
        linc.check_trace("a", (False,), (True,))

    with pytest.raises(ValueError, match="records solver feedback"):
        linc.check_trace("a", (False, True), (True, True))


def test_logic_lm_keeps_its_one_valid_attempt_whatever_its_label() -> None:
    assert LogicLmPipeline(2).select((None,)) == 0
    assert LogicLmPipeline(2).select(()) is None


def test_logic_lm_refuses_a_trace_its_refinement_loop_cannot_produce() -> None:
    logic = LogicLmPipeline(1)
    logic.check_trace("a", (False, True), (False, True))

    with pytest.raises(ValueError, match="exceeds its declared refinement limit"):
        logic.check_trace("a", (False, True, True), (False, False, True))

    with pytest.raises(ValueError, match="feedback before its first attempt"):
        logic.check_trace("a", (True,), (True,))

    with pytest.raises(ValueError, match="has no solver feedback"):
        logic.check_trace("a", (False, False), (False, True))

    with pytest.raises(ValueError, match="continues after a valid formalization"):
        logic.check_trace("a", (False, True), (True, False))
