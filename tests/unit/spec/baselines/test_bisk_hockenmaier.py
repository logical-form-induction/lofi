"""Tests for the Bisk--Hockenmaier baseline declaration."""

from __future__ import annotations

from hashlib import sha256
from json import dumps, loads
from math import exp
from typing import TYPE_CHECKING, cast

import didactic.api as dx
import pytest

from lofi.spec import (
    CombinatorSetSpec,
    PatternRuleSpec,
    RuleSpec,
    TypeRaisingSpec,
)
from lofi.spec.baselines.bisk_hockenmaier import (
    BiskHockenmaierBaselineSpec,
    bisk_hockenmaier_grammar,
)
from lofi.spec.grammar import canonical_bytes

if TYPE_CHECKING:
    from lofi._types import JsonValue


def document() -> dict[str, JsonValue]:
    """Return a small declaration with both licensed sources named."""
    return {
        "kind": "bisk-hockenmaier",
        "ccgbank_source": "licensed-ccgbank-auto",
        "dependency_gold_source": "licensed-johansson-nugues-conll",
        "reference_source": "lofi.resource:configs/eval/bisk_hockenmaier_2012.json",
        "reference_tolerance": 0.02,
        "nominal_category": "N",
        "sentence_category": "S",
        "coordination_category": "conj",
        "gold_sentence_atom": "S",
        "gold_coordination_atom": "conj",
        "nominal_tags": ["NN"],
        "verbal_tags": ["VB", "VBG"],
        "coordination_tags": ["CC"],
        "other_tags": ["JJ", "."],
        "punctuation_tags": ["."],
        "headed_or_modifier_arguments": ["N", "S"],
        "sentence_arguments": ["N"],
        "em_max_iterations": 50,
        "convergence_tolerance": 1e-6,
    }


def declared() -> BiskHockenmaierBaselineSpec:
    """Return the validated declaration."""
    return BiskHockenmaierBaselineSpec.model_validate(document())


def test_the_declaration_resolves_the_paper_treatment() -> None:
    spec = declared()
    settings = spec.settings()

    assert spec.metrics() == (
        "supertagging-length-10",
        "dependencies-length-10",
        "supertagging-length-20",
        "dependencies-length-20",
        "supertagging-all-lengths",
        "dependencies-all-lengths",
    )
    assert settings.ccgbank_source == "licensed-ccgbank-auto"
    assert settings.dependency_gold_source == "licensed-johansson-nugues-conll"
    assert settings.training_sections == tuple(range(2, 22))
    assert settings.training_max_length == 10
    assert settings.development_section == 0
    assert settings.evaluation_section == 23
    assert settings.evaluation_length_bands == (10, 20, None)
    assert settings.grammar == bisk_hockenmaier_grammar()
    assert settings.language.name == "propositional"
    assert settings.pos_root_policy == "s-to-top-except-vbg"
    assert settings.noun_argument_policy == "none"
    assert settings.headed_or_modifier_arguments == ("N", "S")
    assert settings.sentence_arguments == ("N",)
    assert settings.second_order_modifiers
    assert settings.max_arity == 3
    assert settings.transitive_category_policy == ("prefer-(S\\N)/N-over-(S/N)\\N")
    assert settings.lexicon_induction == "iterative"
    assert settings.category_induction_steps == 2
    assert settings.derived_constituent_pass
    assert settings.expansion_factors == ("Lex", "Unary", "Left", "Right")
    assert settings.em_max_iterations == 50
    assert settings.estimator == "smoothed-k-best-em"
    assert settings.k == 100
    assert settings.k_best_refresh == "every-iteration"
    assert settings.forest_interpolation == "c-k-over-c-k-plus-c-full"
    assert settings.initialization == "uniform-observed-events"
    assert settings.probability_floor == exp(-6.0)
    assert settings.stopping_rule == "convergence"
    assert settings.convergence_tolerance == 1e-6
    assert settings.k_provenance == "published-development-section-00"
    assert settings.dependency_conversion == "johansson-nugues-ptb"
    assert settings.dependency_scoring == ("conll-2008-unlabeled-directed-attachment")
    assert settings.dependency_direction == "modifier-to-head"
    assert settings.dependency_root == 0
    assert settings.conjunction_convention == "bisk-hockenmaier"
    assert settings.conventions == spec.conventions()
    assert settings.conventions["formalism"] == "ccg"
    assert settings.conventions["grammar_digest"] == (
        "sha256:" + sha256(canonical_bytes(bisk_hockenmaier_grammar())).hexdigest()
    )
    assert settings.conventions["dependency_conversion"] == "johansson-nugues-ptb"


@pytest.mark.parametrize(
    "changed",
    [
        {"ccgbank_source": ""},
        {"dependency_gold_source": ""},
        {"training_sections": []},
        {"training_sections": [0, 2]},
        {"training_max_length": 0},
        {"development_section": 23},
        {"evaluation_section": -1},
        {"evaluation_length_bands": []},
        {"evaluation_length_bands": [20, 10, None]},
        {"evaluation_length_bands": [10, 20]},
        {"evaluation_length_bands": [10, None, None]},
        {"nominal_tags": []},
        {"verbal_tags": ["VB"]},
        {"other_tags": ["NN"]},
        {"nominal_category": "TOP"},
        {"coordination_category": "X"},
        {"headed_or_modifier_arguments": ["N", "X"]},
        {"type_raised_results": "declared"},
        {"atomic_inventory": ["N", "S", "conj", "TOP"]},
        {"root_category": "TOP"},
        {"top_categories": ["N", "S"]},
        {"composition_order": "B1"},
        {"category_induction_steps": 0},
        {"k": 99},
        {"expansion_factors": ["Lex", "Unary", "Left"]},
        {"dependency_root": 1},
        {"em_max_iterations": 0},
        {"probability_floor": 0.0},
        {"probability_floor": -1.0},
        {"probability_floor": 1.01},
        {"convergence_tolerance": -1.0},
        {"corpus": "ccgbank"},
        {"evaluation_sections": [10, 20, 23]},
        {"lexical_constraint_set": "bisk-hockenmaier-six"},
        {"iterations": 50},
    ],
)
def test_malformed_or_ambiguous_controls_are_refused(
    changed: dict[str, JsonValue],
) -> None:
    changed_document = {**document(), **changed}

    with pytest.raises(dx.ValidationError):
        BiskHockenmaierBaselineSpec.model_validate(changed_document)


@pytest.mark.parametrize(
    "changed",
    [
        {"training_sections": list(range(3, 22))},
        {"training_max_length": 9},
        {"development_section": 1},
        {"evaluation_section": 24},
        {"evaluation_length_bands": [9, 20, None]},
        {"nominal_tags": ["NNS"]},
        {"verbal_tags": ["VB", "VBG", "VBZ"]},
        {"coordination_tags": ["CC", "CONJ"]},
        {"other_tags": ["."]},
        {"punctuation_tags": ["JJ"]},
        {"headed_or_modifier_arguments": ["S", "N"]},
        {"sentence_arguments": ["N", "S"]},
        {"second_order_modifiers": False},
        {"max_arity": 4},
        {"category_induction_steps": 3},
        {"derived_constituent_pass": False},
        {"em_max_iterations": 51},
        {"probability_floor": exp(-5.0)},
        {"convergence_tolerance": 2e-6},
        {"reference_tolerance": 0.03},
        {"language": "neo-davidsonian"},
    ],
)
def test_every_variable_treatment_control_changes_the_convention_identity(
    changed: dict[str, JsonValue],
) -> None:
    original = declared()
    varied = BiskHockenmaierBaselineSpec.model_validate({**document(), **changed})

    assert varied.conventions() != original.conventions()


def test_a_declared_grammar_change_changes_the_convention_identity() -> None:
    grammar = bisk_hockenmaier_grammar()
    varied = declared().with_(
        grammar=grammar.with_(
            normal_form=grammar.normal_form.with_(raised_application=False)
        )
    )

    assert varied.conventions() != declared().conventions()


def _grammar_with(**changes: JsonValue) -> dict[str, JsonValue]:
    grammar = cast(
        "dict[str, JsonValue]", loads(canonical_bytes(bisk_hockenmaier_grammar()))
    )
    return {**document(), "grammar": {**grammar, **changes}}


@pytest.mark.parametrize(
    ("grammar", "match"),
    [
        (
            {"goal": {"kind": "root", "categories": ["S", "TOP"]}},
            "not one root category",
        ),
        (
            {"goal": {"kind": "root", "categories": ["S"]}},
            "are not distinct",
        ),
        (
            {"budget": {"unary_rounds": 1, "closure_rounds": 1, "threads": 2}},
            "threads",
        ),
    ],
)
def test_a_grammar_the_baseline_cannot_read_is_refused(
    grammar: dict[str, JsonValue], match: str
) -> None:
    with pytest.raises(dx.ValidationError, match=match):
        BiskHockenmaierBaselineSpec.model_validate(_grammar_with(**grammar))


def test_the_packaged_grammar_raises_only_the_sentence_and_nominal_atoms() -> None:
    combinators, *others = bisk_hockenmaier_grammar().rules.rules

    assert isinstance(combinators, CombinatorSetSpec)
    assert combinators.type_raising
    assert combinators.raising_targets == ("N", "S")
    assert combinators.raising_arguments == ("N", "S")
    assert not any(isinstance(rule, TypeRaisingSpec) for rule in others)
    assert not any(
        isinstance(rule, PatternRuleSpec) and rule.normal_form_role.kind == "raising"
        for rule in others
    )


def _with_raising(rule: RuleSpec) -> dict[str, JsonValue]:
    declared = bisk_hockenmaier_grammar().rules
    _, *others = declared.rules
    return _grammar_with(
        rules=cast(
            "dict[str, JsonValue]",
            loads(declared.with_(rules=(rule, *others)).model_dump_json()),
        )
    )


@pytest.mark.parametrize(
    "rule",
    [
        cast("CombinatorSetSpec", bisk_hockenmaier_grammar().rules.rules[0]).with_(
            raising_targets=None
        ),
        cast("CombinatorSetSpec", bisk_hockenmaier_grammar().rules.rules[0]).with_(
            raising_arguments=None
        ),
        TypeRaisingSpec(view="/"),
        TypeRaisingSpec(view="/", targets=("N", "S"), arguments=None),
    ],
    ids=[
        "combinator-set-targets",
        "combinator-set-arguments",
        "type-raising-targets",
        "type-raising-arguments",
    ],
)
def test_raising_without_finite_declared_targets_and_arguments_is_refused(
    rule: RuleSpec,
) -> None:
    with pytest.raises(dx.ValidationError, match="finite raising targets"):
        BiskHockenmaierBaselineSpec.model_validate(_with_raising(rule))


def test_declared_raising_targets_change_the_convention_identity() -> None:
    combinators = cast("CombinatorSetSpec", bisk_hockenmaier_grammar().rules.rules[0])
    varied = BiskHockenmaierBaselineSpec.model_validate(
        _with_raising(combinators.with_(raising_targets=("S",)))
    )

    assert varied.conventions() != declared().conventions()


@pytest.mark.parametrize("kind", ["counting", "boolean", "forest"])
def test_a_parse_whose_semiring_has_no_log_probabilities_is_refused(
    kind: str,
) -> None:
    changed = {
        **document(),
        "parse": {
            "retain": "forest",
            "reachability": "roots",
            "semiring": {"kind": kind},
        },
    }

    with pytest.raises(
        dx.ValidationError,
        match=(
            "the Bisk--Hockenmaier baseline needs float log-probability weights, "
            f"which parse.semiring '{kind}' does not supply"
        ),
    ):
        BiskHockenmaierBaselineSpec.model_validate(changed)


def test_a_measured_beam_reaches_the_baseline_parse_settings() -> None:
    changed = {
        **document(),
        "parse": {
            "retain": "forest",
            "reachability": "roots",
            "pruner": {"kind": "top-k", "width": 4},
            "measure_pruned_mass": True,
        },
    }

    settings = BiskHockenmaierBaselineSpec.model_validate(changed).settings()

    assert settings.parse.measure_pruned_mass


def test_the_declared_threads_reach_the_settings_and_default_to_one() -> None:
    threaded = BiskHockenmaierBaselineSpec.model_validate({**document(), "threads": 4})

    assert declared().settings().threads == 1
    assert threaded.settings().threads == 4


def test_the_thread_count_leaves_the_convention_identity_unchanged() -> None:
    threaded = BiskHockenmaierBaselineSpec.model_validate({**document(), "threads": 4})

    assert threaded.conventions() == declared().conventions()
    assert "threads" not in threaded.conventions()


@pytest.mark.parametrize("threads", [0, -1])
def test_a_non_positive_thread_count_is_refused(threads: int) -> None:
    with pytest.raises(dx.ValidationError, match="threads"):
        BiskHockenmaierBaselineSpec.model_validate({**document(), "threads": threads})


@pytest.mark.parametrize("language", ["propositional", "neo-davidsonian"])
def test_a_declaration_read_back_in_sorted_key_order_is_the_same(
    language: str,
) -> None:
    spec = BiskHockenmaierBaselineSpec.model_validate(
        {**document(), "language": language}
    )
    sorted_form = dumps(loads(spec.model_dump_json()), sort_keys=True)

    assert BiskHockenmaierBaselineSpec.model_validate_json(sorted_form) == spec
