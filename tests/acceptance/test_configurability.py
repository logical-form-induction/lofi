"""Every choice a practitioner would vary is declared in a document.

Each case writes the choice in a configuration document, composes it through
the composition API and trains or plans the point, then checks that the
declared value reached the run. The probes cover rule matching, category
bounds, formalisms, supervision strength, the neural chart, datasets, trainers,
checkpoint release, derivation sampling, validation splits, decoder structure
and evaluators, together with the layering that lets a document refine
another and a project shadow a packaged option.
"""

from typing import TYPE_CHECKING, cast

import pytest
from tests.acceptance.driver import (
    DOCUMENTS,
    leaf,
    lofi,
    resolved,
    train,
)

if TYPE_CHECKING:
    from pathlib import Path

    from tests.acceptance.driver import TrainedPoint

    from lofi._types import JsonValue


def test_token_level_supervision_extends_the_inherited_objectives(
    tmp_path: Path,
) -> None:
    point = train("pinned-point.yaml", tmp_path)
    kinds = [leaf(point.resolved, f"loss.terms.{at}.objective.kind") for at in range(3)]

    assert kinds == ["grammaticality", "coherence", "pinning"]
    assert (
        leaf(point.resolved, "data.datasets.train.constraint_source.kind")
        == "gold-lexicon"
    )


def test_a_project_fragment_shadows_the_packaged_option() -> None:
    selection = "data/datasets/train/dataset/noise/grammatical=tenth"
    packaged = resolved("slash-point.yaml", selection)
    shadowed = resolved(
        "slash-point.yaml",
        selection,
        project=DOCUMENTS / "project",
    )

    path = "data.datasets.train.dataset.noise.grammatical.rate"
    assert leaf(packaged, path) == 0.1
    assert leaf(shadowed, path) == 0.05


@pytest.mark.parametrize(
    ("document", "language", "style"),
    [
        ("propositional-language.yaml", "propositional", "curry"),
        ("neo-davidsonian-language.yaml", "neo-davidsonian", "church"),
    ],
)
def test_each_packaged_language_trains_with_field_overrides(
    tmp_path: Path,
    document: str,
    language: str,
    style: str,
) -> None:
    point = train(document, tmp_path / language)

    assert leaf(point.resolved, "data.datasets.train.dataset.language.name") == language
    assert (
        leaf(point.resolved, "data.datasets.train.dataset.language.binders.style")
        == style
    )


def test_show_config_names_the_source_of_an_overridden_leaf(
    capsys: pytest.CaptureFixture[str],
) -> None:
    status = lofi()(
        [
            "show-config",
            "--config",
            str(DOCUMENTS / "slash-point.yaml"),
            "--set",
            "optimizer.learning_rate=0.01",
        ]
    )
    printed = capsys.readouterr().out

    assert status == 0
    assert "learning_rate: 0.01  # override optimizer.learning_rate=0.01" in printed


def test_one_rule_declares_exact_matching_under_a_feature_erasing_grammar(
    tmp_path: Path,
) -> None:
    point = train("rule-matching.yaml", tmp_path)
    assert leaf(
        point.resolved, "data.datasets.train.dataset.grammar.rules.rules.0.matching"
    ) == {"kind": "identity"}


def test_a_grammar_bounds_the_categories_it_may_build(tmp_path: Path) -> None:
    point = train("category-bound.yaml", tmp_path)

    assert (
        leaf(
            point.resolved,
            "data.datasets.train.dataset.grammar.budget.max_category_size",
        )
        == 7
    )
    assert (
        leaf(
            point.resolved,
            "data.datasets.train.dataset.grammar.budget.max_category_arity",
        )
        == 2
    )


def test_a_document_reads_its_grammar_as_an_ab_grammar(tmp_path: Path) -> None:
    point = train("ab-formalism.yaml", tmp_path)

    assert leaf(point.resolved, "data.datasets.train.dataset.grammar.formalism") == "ab"


def test_pinned_categories_are_a_weighted_preference(tmp_path: Path) -> None:
    point = train("soft-clamp.yaml", tmp_path)

    assert (
        leaf(point.resolved, "data.datasets.train.constraint_source.clamp.mode")
        == "soft"
    )
    assert (
        leaf(point.resolved, "data.datasets.train.constraint_source.clamp.penalty")
        == 0.5
    )


def test_the_neural_chart_uses_a_declared_reduction(tmp_path: Path) -> None:
    point = train("maximum-chart.yaml", tmp_path)

    assert leaf(point.resolved, "model.parser.reduction.kind") == "maximum"


def test_the_tagged_chart_decodes_a_declared_k_best(tmp_path: Path) -> None:
    point = train("tagged-chart.yaml", tmp_path)

    assert leaf(point.resolved, "model.parser.k_best") == 2


def test_a_point_trains_on_declared_corpus_partitions(tmp_path: Path) -> None:
    point = train("corpus-point.yaml", tmp_path)

    assert point.rows


def test_a_trainer_alternates_two_declared_steps(tmp_path: Path) -> None:
    point = train("alternating-trainer.yaml", tmp_path)

    assert leaf(point.resolved, "trainer.kind") == "alternating"


@pytest.mark.parametrize(
    "sampler",
    ["length-bucket", "random", "sequential"],
)
def test_each_declared_batch_sampler_trains(
    tmp_path: Path,
    sampler: str,
) -> None:
    point = train(
        "slash-point.yaml",
        tmp_path / sampler,
        f"trainer/batch_sampler={sampler}",
    )

    assert leaf(point.resolved, "trainer.batch_sampler.kind") == sampler


def test_a_staged_trainer_warms_up_tokens_before_fitting_the_parser(
    tmp_path: Path,
) -> None:
    point = train("staged-trainer.yaml", tmp_path)

    assert leaf(point.resolved, "trainer.kind") == "staged"
    assert leaf(point.resolved, "trainer.stages.0.step.kind") == "token-warm-up"
    assert leaf(point.resolved, "trainer.stages.1.step.kind") == "parser"


def test_a_run_declares_the_checkpoint_it_releases(tmp_path: Path) -> None:
    point = train("release-checkpoint.yaml", tmp_path)

    assert leaf(point.resolved, "release.checkpoint") == "validation-best"


def test_synthetic_derivations_are_drawn_under_declared_rule_weights(
    tmp_path: Path,
) -> None:
    point = train("weighted-sampler.yaml", tmp_path)

    assert (
        leaf(point.resolved, "data.datasets.train.dataset.sampler.kind")
        == "weighted-rules"
    )


def test_synthetic_derivations_meet_declared_shape_and_length_controls(
    tmp_path: Path,
) -> None:
    point = train("shape-sampler.yaml", tmp_path)

    assert leaf(point.resolved, "data.datasets.train.dataset.length.kind") == "tabular"
    assert (
        leaf(
            point.resolved,
            "data.datasets.train.dataset.shape.default.right_branching.value",
        )
        == 0.5
    )
    assert (
        leaf(
            point.resolved,
            "data.datasets.train.dataset.shape.default.slash_balance.value",
        )
        == 0.5
    )


def test_declared_retention_and_epoch_reshuffling_are_composable(
    tmp_path: Path,
) -> None:
    point = train("stratified-validation.yaml", tmp_path)

    assert (
        leaf(point.resolved, "data.datasets.train.dataset.retained.validation") == 0.25
    )
    assert leaf(point.resolved, "trainer.batch_sampler.reshuffle_each_epoch") is True


def test_a_transformer_decoder_declares_its_attention_structure(
    tmp_path: Path,
) -> None:
    point = train("decoder-structure.yaml", tmp_path)

    assert (
        leaf(point.resolved, "model.decoders.application.structure")
        == "ancestor-and-sibling-bias"
    )


def test_an_evaluator_scores_decoded_logical_forms(tmp_path: Path) -> None:
    point = train("meaning-evaluator.yaml", tmp_path)

    assert any(
        row["suite"] == "held-out"
        and row["task"] == "meanings"
        and row["metric"] == "logical-form-exact"
        for row in point.rows
    )


def _equivalence_rows(point: TrainedPoint, /) -> dict[str, JsonValue]:
    """Return the held-out equivalence rows of a trained point by metric."""
    return {
        cast("str", row["metric"]): row["value"]
        for row in point.rows
        if row["suite"] == "held-out" and row["task"] == "equivalence"
    }


def test_an_evaluator_decides_decoded_forms_by_proof(tmp_path: Path) -> None:
    rows = _equivalence_rows(train("equivalence-evaluator.yaml", tmp_path))

    assert set(rows) == {
        "equivalence-equivalence",
        "equivalence-well-formed",
        "equivalence-timeout",
        "equivalence-proof-time",
        "equivalence-proof-calls",
        "equivalence-memo-hits",
    }
    assert rows["equivalence-well-formed"] == 1.0
    assert cast("float", rows["equivalence-proof-calls"]) > 0.0
    assert isinstance(rows["equivalence-memo-hits"], float)


def test_a_zero_call_proof_budget_makes_every_check_abstain(tmp_path: Path) -> None:
    rows = _equivalence_rows(
        train(
            "equivalence-evaluator.yaml",
            tmp_path,
            "evaluators.equivalence.budget.calls_per_step=0",
        )
    )

    assert rows["equivalence-proof-calls"] == 0.0
    assert rows["equivalence-equivalence"] == 0.0


def test_an_evaluator_reports_gold_reachability_under_the_run_inventory(
    tmp_path: Path,
) -> None:
    point = train("reachability-evaluator.yaml", tmp_path)
    rows = {
        cast("str", row["metric"]): row
        for row in point.rows
        if row["suite"] == "held-out" and row["task"] == "reachability"
    }

    assert set(rows) == {
        "reachability-steps",
        "reachability-derivations",
        "reachability-blocked-label",
        "reachability-blocked-rule",
        "reachability-blocked-construction",
    }
    assert rows["reachability-derivations"]["value"] == 1.0
    assert rows["reachability-blocked-construction"]["value"] == 0.0
