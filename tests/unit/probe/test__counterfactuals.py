import os
import re
import subprocess
import sys
from copy import copy, deepcopy
from typing import TYPE_CHECKING, Final, cast
from unittest.mock import patch

import pytest
import torch
from tests.support.rules import ccg_signature, declared_rules

from lofi.deduce import (
    CoverGoal,
    LogInside,
    ParsePolicy,
    Reachability,
    Retain,
    RootGoal,
    SpanFree,
    parse,
)
from lofi.judge import NoNormalForm, RuleSet
from lofi.lexicon import Clamp, ClampedSource, Entry, ErrorPolicy, StaticLexicon
from lofi.probe._causal import (
    AlignedInput,
    AlignmentPolicy,
    CausalPair,
    CombinatorAt,
    CounterfactualPair,
    JointCounterfactualPair,
    NamedVariable,
    Side,
    SplitAt,
    TokenizerAlignment,
    TypeAt,
)
from lofi.probe._counterfactuals import (
    ALIGNMENT_EXCLUSION_REASONS,
    ALIGNMENT_REFUSAL_REASONS,
    AlignmentExclusion,
    AlignmentManifest,
    AlignmentRefusal,
    CounterfactualGeneration,
    CounterfactualGenerator,
    GeneratedPair,
    GeneratedPairId,
    ModelCausalPair,
    ModelJointCausalPair,
    confirm_pair,
)
from lofi.probe._provenance import Provenance
from lofi.probe._variables import extract_variables
from lofi.spec import (
    ApplicationSpec,
    CompositionSpec,
    ConstantPatternSpec,
    PatternRuleSpec,
    RuleSetSpec,
    TermPatternSpec,
)
from lofi.spec.grammar import (
    ConjunctionPairSpec,
    ConjunctionsSpec,
    FeatureErasureMatchingSpec,
    OperatorSpec,
    SignatureSpec,
)
from lofi.terms import (
    TargetSet,
    Term,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lofi.deduce import Cell, Goal
    from lofi.judge import (
        Rule,
    )
    from lofi.probe._causal import CellAssignment, HighLevelVariable
    from lofi.probe._counterfactuals import (
        AlignmentExclusionReason,
        AlignmentRefusalReason,
    )
    from lofi.spec import PatternSpec


TOY_CATEGORIES: Final = SignatureSpec(
    name="toy-counterfactuals",
    primitives=("S", "S/S", "Sdcl", "Sng", "Sb", "A"),
    syntax="s-expression",
)


def toy_rule(
    name: str, premises: tuple[PatternSpec, ...], conclusion: str
) -> Rule[Term]:
    """Compile one pattern rule over the toy categories."""
    (rule,) = declared_rules(
        TOY_CATEGORIES,
        RuleSetSpec(
            name=name,
            rules=(
                PatternRuleSpec(
                    name=name,
                    premises=premises,
                    conclusion=ConstantPatternSpec(name=conclusion),
                ),
            ),
        ),
    ).rule_set

    return rule


def toy_constant(name: str) -> ConstantPatternSpec:
    """Return the fixed pattern for one toy category."""
    return ConstantPatternSpec(name=name)


TOY_CONTEXT: Final = declared_rules(
    TOY_CATEGORIES,
    RuleSetSpec(name="toy", rules=()),
).context
TOY_STORE: Final = TOY_CONTEXT.store


def toy_category(name: str) -> Term:
    """Return one interned toy category."""
    return TOY_STORE.const("Cat", name)


class Unweighted:
    """Gives every axiom and every rule application the same log weight."""

    lowering = SpanFree()

    def axiom_weight(self, position: int, entry: Entry[Term], /) -> float:
        del position, entry

        return 0.0

    def edge_weight(
        self,
        rule: Rule[Term],
        inputs: tuple[Term, ...],
        output: Term,
        span: Cell,
        split: int | None,
        /,
    ) -> float:
        del rule, inputs, output, span, split

        return 0.0


APPLY: Final = toy_rule(
    "apply",
    (toy_constant("S/S"), TermPatternSpec(name="X", sort="Cat")),
    "S",
)
JOIN: Final = toy_rule("join", (toy_constant("A"), toy_constant("A")), "A")
LIFT: Final = toy_rule("lift", (toy_constant("Sng"),), "S")
LEFT: Final = toy_rule("left", (toy_constant("A"), toy_constant("A")), "A")
RIGHT: Final = toy_rule("right", (toy_constant("A"), toy_constant("A")), "A")
EXACT: Final = APPLY.matching
TOY: Final = RuleSet("toy", (APPLY,), EXACT, NoNormalForm[Term]())
AMBIGUOUS: Final = RuleSet("ambiguous", (JOIN,), EXACT, NoNormalForm[Term]())
TWO_WAYS: Final = RuleSet("two-ways", (LEFT, RIGHT), EXACT, NoNormalForm[Term]())
LIFTING: Final = RuleSet("lifting", (APPLY, LIFT), EXACT, NoNormalForm[Term]())
TOY_GOAL: Final = RootGoal(TargetSet((toy_category("S"),), EXACT))
AMBIGUOUS_GOAL: Final = RootGoal(TargetSet((toy_category("A"),), EXACT))
TOY_LEXICON: Final = StaticLexicon(
    [
        Entry("really", toy_category("S/S")),
        Entry("sleeps", toy_category("Sdcl")),
        Entry("sleeps", toy_category("Sng")),
    ]
)
WIDE_LEXICON: Final = StaticLexicon(
    [
        Entry("really", toy_category("S/S")),
        Entry("sleeps", toy_category("Sdcl")),
        Entry("sleeps", toy_category("Sng")),
        Entry("sleeps", toy_category("Sb")),
        Entry("sleeps", toy_category("S")),
    ]
)
WORD_ALIGNMENT_POLICY: Final = AlignmentPolicy("all", "exclude", contiguous=True)


def a_render(term: Term) -> str:
    return TOY_STORE.payload_name(term)


def a_generator(
    rules: RuleSet[Term] = TOY,
    lexicon: StaticLexicon[Term] = TOY_LEXICON,
    goal: Goal[Term] = TOY_GOAL,
    readings: int = 64,
    threads: int = 1,
) -> CounterfactualGenerator[Term, float]:
    return CounterfactualGenerator(
        rules,
        lexicon,
        goal,
        a_render,
        "toy:1",
        unary_depth=0,
        parse_policy=ParsePolicy(Retain.FOREST, Reachability.ALL),
        readings=readings,
        threads=threads,
    )


def an_aligned_input(
    tokens: Sequence[str],
    alignments: tuple[TokenizerAlignment, ...] | None = None,
) -> AlignedInput:
    held = tuple(tokens)
    evidence = alignments or (
        TokenizerAlignment(
            "word-tokenizer",
            tuple((position,) for position in range(len(held))),
            (),
            len(held),
        ),
    )

    return AlignedInput(held, evidence)


def a_lexical_assignment(
    first: str, second: str, joined: str
) -> dict[HighLevelVariable, str | int]:
    return {
        TypeAt((0, 1)): first,
        TypeAt((1, 2)): second,
        TypeAt((0, 2)): joined,
        SplitAt((0, 2)): 1,
        CombinatorAt((0, 2), 1): "apply",
    }


def a_record(
    base_assignment: CellAssignment,
    variant_assignment: CellAssignment,
    varied: HighLevelVariable,
    *,
    tokens: tuple[str, ...] = ("really", "sleeps"),
    base: tuple[str, ...] = ("S/S", "Sdcl"),
    variant: tuple[str, ...] = ("S/S", "Sng"),
    grammar: str = "toy:1",
    provenance: Provenance | None = None,
) -> GeneratedPair:
    provenance = provenance or Provenance("grammar-counterfactual", "unit", 0)

    return GeneratedPair(
        tokens,
        base,
        variant,
        base_assignment,
        variant_assignment,
        varied,
        grammar,
        provenance,
    )


def a_generation() -> tuple[CounterfactualGeneration, TokenizerAlignment]:
    varied = TypeAt((1, 2))
    record = a_record(
        a_lexical_assignment("S/S", "Sdcl", "S"),
        a_lexical_assignment("S/S", "Sng", "S"),
        varied,
    )
    alignment = TokenizerAlignment("word-tokenizer", ((0,), (1,)), (), 2)
    aligned = AlignedInput(record.tokens, (alignment,))
    manifest = AlignmentManifest(WORD_ALIGNMENT_POLICY, aligned.tokenizers, 1, 1)

    return (
        CounterfactualGeneration((record,), {record.id: aligned}, (), manifest),
        alignment,
    )


def a_model_pair() -> ModelCausalPair:
    generation, alignment = a_generation()
    record = generation.pairs[0]
    evidence = generation.aligned_input(record.id)

    return ModelCausalPair(
        record.id,
        confirm_pair(record),
        evidence.tokens,
        generation.manifest.policy,
        base_inputs={"vectors": torch.zeros(1, 2, 3)},
        variant_inputs={"vectors": torch.ones(1, 2, 3)},
        base_alignment=alignment,
        variant_alignment=alignment,
    )


def test_a_string_with_one_lexically_ambiguous_argument_yields_one_pair() -> None:
    generated = a_generator().generate(
        (an_aligned_input(("really", "sleeps")),),
        policy=WORD_ALIGNMENT_POLICY,
        dataset="unit",
        seed=7,
    )

    assert len(generated.pairs) == 1
    assert generated.refusals == ()
    assert generated.manifest == AlignmentManifest(
        WORD_ALIGNMENT_POLICY, ("word-tokenizer",), 1, 1
    )
    assert generated.pairs[0].tokens == ("really", "sleeps")
    assert generated.pairs[0].base == ("S/S", "Sdcl")
    assert generated.pairs[0].variant == ("S/S", "Sng")
    assert generated.pairs[0].span == (1, 2)
    assert generated.pairs[0].form == "sleeps"
    assert generated.pairs[0].varied == TypeAt((1, 2))


@pytest.mark.parametrize("threads", [1, 2])
def test_the_declared_thread_count_reaches_the_counterfactual_parser(
    threads: int,
) -> None:
    generator = a_generator(threads=threads)

    with patch("lofi.probe._counterfactuals.parse", wraps=parse) as parser:
        generator.generate(
            (an_aligned_input(("really", "sleeps")),),
            policy=WORD_ALIGNMENT_POLICY,
            dataset="unit",
            seed=7,
        )

    assert parser.call_args is not None
    assert parser.call_args.kwargs["threads"] == threads


def test_a_generated_pair_differs_at_the_varied_category_alone() -> None:
    generated = a_generator().generate(
        (an_aligned_input(("really", "sleeps")),),
        policy=WORD_ALIGNMENT_POLICY,
        dataset="unit",
        seed=7,
    )

    confirmed = confirm_pair(generated.pairs[0])

    assert generated.refusals == ()
    assert confirmed.varied == TypeAt((1, 2))
    assert confirmed.span == (1, 2)
    assert confirmed.base[TypeAt((1, 2))] == "Sdcl"
    assert confirmed.variant[TypeAt((1, 2))] == "Sng"
    assert confirmed.base[TypeAt((0, 1))] == confirmed.variant[TypeAt((0, 1))]
    assert confirmed.base[TypeAt((0, 2))] == confirmed.variant[TypeAt((0, 2))]
    assert confirmed.base[SplitAt((0, 2))] == confirmed.variant[SplitAt((0, 2))]
    assert (
        confirmed.base[CombinatorAt((0, 2), 1)]
        == confirmed.variant[CombinatorAt((0, 2), 1)]
    )


def test_a_generated_pair_records_its_grammar_its_dataset_and_its_seed() -> None:
    generated = a_generator().generate(
        (an_aligned_input(("really", "sleeps")),),
        policy=WORD_ALIGNMENT_POLICY,
        dataset="unit",
        seed=7,
    )
    found = generated.pairs[0]

    assert generated.refusals == ()
    assert found.grammar == "toy:1"
    assert found.seed == 7
    assert found.provenance == Provenance("grammar-counterfactual", "unit", 7)


def test_a_generated_pair_has_a_golden_content_address() -> None:
    base = a_lexical_assignment("S/S", "Sdcl", "S")
    variant = a_lexical_assignment("S/S", "Sng", "S")
    record = a_record(base, variant, TypeAt((1, 2)))

    assert re.fullmatch(r"[0-9a-f]{64}", record.id)
    assert record.id == GeneratedPairId(
        "3662aa2967d88a2b29aad56831956d932e7b644315d83274d5662fed781398bf"
    )
    assert record.to_json()["id"] == record.id


def test_a_generated_pair_id_is_canonical_over_assignment_insertion_order() -> None:
    base = a_lexical_assignment("S/S", "Sdcl", "S")
    variant = a_lexical_assignment("S/S", "Sng", "S")
    reversed_base = dict(reversed(tuple(base.items())))
    reversed_variant = dict(reversed(tuple(variant.items())))

    assert (
        a_record(base, variant, TypeAt((1, 2))).id
        == a_record(reversed_base, reversed_variant, TypeAt((1, 2))).id
    )


def test_a_generated_pair_id_excludes_all_provenance_fields() -> None:
    base = a_lexical_assignment("S/S", "Sdcl", "S")
    variant = a_lexical_assignment("S/S", "Sng", "S")
    first = a_record(
        base,
        variant,
        TypeAt((1, 2)),
        provenance=Provenance("one-method", "one-dataset", 1),
    )
    second = a_record(
        base,
        variant,
        TypeAt((1, 2)),
        provenance=Provenance("other-method", "other-dataset", 2),
    )

    assert first.id == second.id


def test_every_semantic_pair_field_changes_the_generated_pair_id() -> None:
    base = a_lexical_assignment("S/S", "Sdcl", "S")
    variant = a_lexical_assignment("S/S", "Sng", "S")
    changed_base_assignment = dict(base)
    changed_base_assignment[TypeAt((0, 2))] = "Sother"
    changed_variant_assignment = dict(variant)
    changed_variant_assignment[TypeAt((0, 2))] = "Sother"
    records = (
        a_record(base, variant, TypeAt((1, 2))),
        a_record(
            base,
            variant,
            TypeAt((1, 2)),
            tokens=("truly", "sleeps"),
        ),
        a_record(
            base,
            variant,
            TypeAt((1, 2)),
            base=("S\\S", "Sdcl"),
        ),
        a_record(
            base,
            variant,
            TypeAt((1, 2)),
            variant=("S/S", "Sb"),
        ),
        a_record(changed_base_assignment, variant, TypeAt((1, 2))),
        a_record(base, changed_variant_assignment, TypeAt((1, 2))),
        a_record(base, variant, TypeAt((0, 1))),
        a_record(base, variant, TypeAt((1, 2)), grammar="toy:2"),
    )

    assert len({record.id for record in records}) == len(records)


def test_a_generated_pair_id_is_stable_across_hash_seeds() -> None:
    script = """
from lofi.probe._causal import CombinatorAt, SplitAt, TypeAt
from lofi.probe._counterfactuals import GeneratedPair
from lofi.probe._provenance import Provenance

base = {
    TypeAt((0, 1)): "S/S",
    TypeAt((1, 2)): "Sdcl",
    TypeAt((0, 2)): "S",
    SplitAt((0, 2)): 1,
    CombinatorAt((0, 2), 1): "apply",
}
variant = dict(base)
variant[TypeAt((1, 2))] = "Sng"
record = GeneratedPair(
    ("really", "sleeps"),
    ("S/S", "Sdcl"),
    ("S/S", "Sng"),
    base,
    variant,
    TypeAt((1, 2)),
    "toy:1",
    Provenance("grammar-counterfactual", "unit", 0),
)
print(record.id)
"""
    found = [
        subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            encoding="utf-8",
            env={**os.environ, "PYTHONHASHSEED": seed},
        ).stdout.strip()
        for seed in ("1", "247")
    ]

    assert found == [found[0], found[0]]


def test_a_string_with_one_reading_yields_no_pair() -> None:
    lexicon = StaticLexicon(
        [
            Entry("really", toy_category("S/S")),
            Entry("sleeps", toy_category("Sdcl")),
        ]
    )
    generated = a_generator(lexicon=lexicon).generate(
        (an_aligned_input(("really", "sleeps")),),
        policy=WORD_ALIGNMENT_POLICY,
        dataset="unit",
        seed=7,
    )

    assert generated.pairs == ()
    assert generated.refusals == ()
    assert generated.manifest.candidate_count == 0


def test_a_string_with_no_derivation_yields_no_pair() -> None:
    generated = a_generator().generate(
        (an_aligned_input(("sleeps", "really")),),
        policy=WORD_ALIGNMENT_POLICY,
        dataset="unit",
        seed=7,
    )

    assert generated.pairs == ()
    assert generated.refusals == ()
    assert generated.manifest.candidate_count == 0


def test_readings_that_differ_only_in_structure_yield_no_pair() -> None:
    lexicon = StaticLexicon(
        [Entry(form, toy_category("A")) for form in ("a", "b", "c")]
    )
    generated = a_generator(
        rules=AMBIGUOUS, lexicon=lexicon, goal=AMBIGUOUS_GOAL
    ).generate(
        (an_aligned_input(("a", "b", "c")),),
        policy=WORD_ALIGNMENT_POLICY,
        dataset="unit",
        seed=7,
    )

    assert generated.pairs == ()
    assert generated.refusals == ()
    assert generated.manifest.candidate_count == 0


def test_a_cell_whose_category_a_unary_step_changed_is_not_varied() -> None:
    generated = a_generator(rules=LIFTING).generate(
        (an_aligned_input(("really", "sleeps")),),
        policy=WORD_ALIGNMENT_POLICY,
        dataset="unit",
        seed=7,
    )

    assert len(generated.pairs) == 1
    assert generated.refusals == ()
    assert generated.pairs[0].base == ("S/S", "Sdcl")
    assert generated.pairs[0].variant == ("S/S", "Sng")
    assert confirm_pair(generated.pairs[0]).variant[TypeAt((1, 2))] == "Sng"


def test_readings_that_differ_only_in_the_combinator_yield_no_pair() -> None:
    lexicon = StaticLexicon([Entry(form, toy_category("A")) for form in ("a", "b")])
    generated = a_generator(
        rules=TWO_WAYS, lexicon=lexicon, goal=AMBIGUOUS_GOAL
    ).generate(
        (an_aligned_input(("a", "b")),),
        policy=WORD_ALIGNMENT_POLICY,
        dataset="unit",
        seed=7,
    )

    assert generated.pairs == ()
    assert generated.refusals == ()
    assert generated.manifest.candidate_count == 0


def test_a_derivation_covering_part_of_the_string_is_not_read() -> None:
    generated = a_generator(goal=CoverGoal[Term](2)).generate(
        (an_aligned_input(("sleeps", "really")),),
        policy=WORD_ALIGNMENT_POLICY,
        dataset="unit",
        seed=7,
    )

    assert generated.pairs == ()
    assert generated.refusals == ()
    assert generated.manifest.candidate_count == 0


def test_the_reading_budget_bounds_what_one_string_is_read_for() -> None:
    aligned = (an_aligned_input(("really", "sleeps")),)
    generous = a_generator(lexicon=WIDE_LEXICON).generate(
        aligned,
        policy=WORD_ALIGNMENT_POLICY,
        dataset="unit",
        seed=7,
    )
    narrow = a_generator(lexicon=WIDE_LEXICON, readings=2).generate(
        aligned,
        policy=WORD_ALIGNMENT_POLICY,
        dataset="unit",
        seed=7,
    )

    assert len(generous.pairs) == 6
    assert len(narrow.pairs) == 1
    assert generous.refusals == narrow.refusals == ()
    assert generous.manifest.candidate_count == 6
    assert narrow.manifest.candidate_count == 1


def test_generation_is_reproducible_at_a_fixed_seed() -> None:
    aligned = (an_aligned_input(("really", "sleeps")),)
    once = a_generator(lexicon=WIDE_LEXICON).generate(
        aligned,
        policy=WORD_ALIGNMENT_POLICY,
        dataset="unit",
        seed=11,
    )
    twice = a_generator(lexicon=WIDE_LEXICON).generate(
        aligned,
        policy=WORD_ALIGNMENT_POLICY,
        dataset="unit",
        seed=11,
    )

    assert [record.to_json() for record in once.pairs] == [
        record.to_json() for record in twice.pairs
    ]
    assert once.refusals == twice.refusals == ()
    assert once.manifest == twice.manifest


def test_a_different_seed_draws_the_same_pairs_in_another_order() -> None:
    aligned = (an_aligned_input(("really", "sleeps")),)
    here_generation = a_generator(lexicon=WIDE_LEXICON).generate(
        aligned,
        policy=WORD_ALIGNMENT_POLICY,
        dataset="unit",
        seed=11,
    )
    there_generation = a_generator(lexicon=WIDE_LEXICON).generate(
        aligned,
        policy=WORD_ALIGNMENT_POLICY,
        dataset="unit",
        seed=12,
    )
    here = [(record.base, record.variant) for record in here_generation.pairs]
    there = [(record.base, record.variant) for record in there_generation.pairs]

    assert here_generation.refusals == there_generation.refusals == ()
    assert here_generation.manifest.candidate_count == 6
    assert there_generation.manifest.candidate_count == 6
    assert sorted(here) == sorted(there)
    assert here != there


def test_every_named_alignment_must_accept_before_a_pair_is_emitted() -> None:
    direct = TokenizerAlignment("direct", ((0,), (1,)), (), 2)
    shifted = TokenizerAlignment("shifted", ((1,), (2,)), (0,), 3)
    gapped = TokenizerAlignment("gapped", ((0,), (1, 3)), (2,), 4)
    aligned = an_aligned_input(
        ("really", "sleeps"),
        (direct, shifted, gapped),
    )
    policy = AlignmentPolicy("all", "exclude", contiguous=True)

    generated = a_generator().generate(
        (aligned,), policy=policy, dataset="unit", seed=7
    )

    assert generated.pairs == ()
    assert generated.refusals == (
        AlignmentRefusal(
            generated.refusals[0].pair_id,
            "gapped",
            (1, 2),
            "noncontiguous-model-row-span",
        ),
    )
    assert generated.manifest == AlignmentManifest(
        policy,
        ("direct", "shifted", "gapped"),
        1,
        0,
    )
    assert generated.manifest.aligned_fraction == 0.0


def test_a_noncontiguous_policy_accepts_the_same_exact_gapped_evidence() -> None:
    gapped = TokenizerAlignment("gapped", ((0,), (1, 3)), (2,), 4)
    policy = AlignmentPolicy("all", "exclude", contiguous=False)

    generated = a_generator().generate(
        (an_aligned_input(("really", "sleeps"), (gapped,)),),
        policy=policy,
        dataset="unit",
        seed=7,
    )

    assert len(generated.pairs) == 1
    assert generated.refusals == ()
    assert generated.manifest.policy is policy
    assert generated.manifest.aligned_fraction == 1.0
    confirmed = confirm_pair(generated.pairs[0])
    assert type(confirmed.varied) is TypeAt
    assert confirmed.varied == generated.pairs[0].varied


def test_partial_complete_and_empty_populations_have_exact_finite_counts() -> None:
    direct = TokenizerAlignment("shared", ((0,), (1,)), (), 2)
    gapped = TokenizerAlignment("shared", ((0,), (1, 3)), (2,), 4)
    policy = AlignmentPolicy("all", "exclude", contiguous=True)
    partial = a_generator().generate(
        (
            an_aligned_input(("really", "sleeps"), (direct,)),
            an_aligned_input(("really", "sleeps"), (gapped,)),
        ),
        policy=policy,
        dataset="unit",
        seed=7,
    )
    complete = a_generator().generate(
        (an_aligned_input(("really", "sleeps"), (direct,)),),
        policy=policy,
        dataset="unit",
        seed=7,
    )
    one_reading = StaticLexicon(
        [
            Entry("really", toy_category("S/S")),
            Entry("sleeps", toy_category("Sdcl")),
        ]
    )
    empty = a_generator(lexicon=one_reading).generate(
        (an_aligned_input(("really", "sleeps"), (direct,)),),
        policy=policy,
        dataset="unit",
        seed=7,
    )

    assert (
        partial.manifest.candidate_count,
        partial.manifest.aligned_count,
        partial.manifest.aligned_fraction,
    ) == (2, 1, 0.5)
    assert (
        complete.manifest.candidate_count,
        complete.manifest.aligned_count,
        complete.manifest.aligned_fraction,
    ) == (1, 1, 1.0)
    assert (
        empty.manifest.candidate_count,
        empty.manifest.aligned_count,
        empty.manifest.aligned_fraction,
    ) == (0, 0, None)


def test_generation_values_are_frozen_and_copy_no_mutable_collection() -> None:
    direct = TokenizerAlignment("direct", ((0,), (1,)), (), 2)
    policy = AlignmentPolicy("all", "exclude", contiguous=True)
    generated = a_generator().generate(
        (an_aligned_input(("really", "sleeps"), (direct,)),),
        policy=policy,
        dataset="unit",
        seed=7,
    )

    with pytest.raises(AttributeError):
        generated.__setattr__("_pairs", ())

    with pytest.raises(AttributeError):
        generated.manifest.__setattr__("_aligned_count", 0)

    assert isinstance(generated, CounterfactualGeneration)
    assert generated.manifest.aligned_count == 1


def test_alignment_records_refuse_inconsistent_public_construction() -> None:
    policy = AlignmentPolicy("all", "exclude", contiguous=True)

    with pytest.raises(TypeError, match="reason is not a string"):
        AlignmentRefusal(
            GeneratedPairId("pair"),
            "tokenizer",
            (0, 1),
            cast("AlignmentRefusalReason", 1),
        )

    with pytest.raises(ValueError, match="at least one tokenizer"):
        AlignmentManifest(policy, (), 1, 0)

    with pytest.raises(ValueError, match="only 0 refusals"):
        CounterfactualGeneration((), {}, (), AlignmentManifest(policy, ("tok",), 1, 0))


def test_every_published_refusal_reason_constructs_and_no_other_does() -> None:
    refusals = tuple(
        AlignmentRefusal(GeneratedPairId("pair"), "tokenizer", (0, 1), reason)
        for reason in ALIGNMENT_REFUSAL_REASONS
    )

    assert tuple(refusal.reason for refusal in refusals) == ALIGNMENT_REFUSAL_REASONS
    assert len(set(ALIGNMENT_REFUSAL_REASONS)) == len(ALIGNMENT_REFUSAL_REASONS)

    with pytest.raises(ValueError, match="reason"):
        AlignmentRefusal(
            GeneratedPairId("pair"),
            "tokenizer",
            (0, 1),
            cast("AlignmentRefusalReason", "truncated-source-span"),
        )


def test_every_published_exclusion_reason_constructs_and_no_other_does() -> None:
    exclusions = tuple(
        AlignmentExclusion(("really", "sleeps"), "tokenizer", reason)
        for reason in ALIGNMENT_EXCLUSION_REASONS
    )

    assert (
        tuple(exclusion.reason for exclusion in exclusions)
        == ALIGNMENT_EXCLUSION_REASONS
    )
    assert set(ALIGNMENT_EXCLUSION_REASONS).isdisjoint(ALIGNMENT_REFUSAL_REASONS)

    with pytest.raises(ValueError, match="exclusion reason"):
        AlignmentExclusion(
            ("really",),
            "tokenizer",
            cast("AlignmentExclusionReason", "empty-model-row-span"),
        )

    with pytest.raises(ValueError, match="no source token"):
        AlignmentExclusion((), "tokenizer", "context-overflow")


def test_a_generation_carries_the_sentences_excluded_before_it() -> None:
    direct = TokenizerAlignment("direct", ((0,), (1,)), (), 2)
    excluded = AlignmentExclusion(
        ("really", "really", "sleeps"), "direct", "context-overflow"
    )
    generated = a_generator().generate(
        (an_aligned_input(("really", "sleeps"), (direct,)),),
        policy=AlignmentPolicy("all", "exclude", contiguous=True),
        dataset="unit",
        seed=7,
        exclusions=(excluded,),
    )

    assert generated.exclusions == (excluded,)
    assert generated.manifest.candidate_count == 1

    with pytest.raises(TypeError, match="AlignmentExclusion"):
        CounterfactualGeneration(
            (),
            {},
            (),
            AlignmentManifest(WORD_ALIGNMENT_POLICY, ("direct",), 0, 0),
            exclusions=(cast("AlignmentExclusion", "direct"),),
        )


def test_unaligned_token_sequences_are_not_generation_inputs() -> None:
    with pytest.raises(TypeError, match="consumes AlignedInput"):
        a_generator().generate(
            (cast("AlignedInput", ("really", "sleeps")),),
            policy=WORD_ALIGNMENT_POLICY,
            dataset="unit",
            seed=7,
        )


def test_a_generation_requires_one_tokenizer_order() -> None:
    first = TokenizerAlignment("first", ((0,), (1,)), (), 2)
    second = TokenizerAlignment("second", ((0,), (1,)), (), 2)

    with pytest.raises(ValueError, match="tokenizer order"):
        a_generator().generate(
            (
                an_aligned_input(("really", "sleeps"), (first, second)),
                an_aligned_input(("really", "sleeps"), (second, first)),
            ),
            policy=AlignmentPolicy("all", "exclude", contiguous=True),
            dataset="unit",
            seed=7,
        )


def test_the_variant_derivation_is_derivable_under_the_same_grammar() -> None:
    generated = a_generator().generate(
        (an_aligned_input(("really", "sleeps")),),
        policy=WORD_ALIGNMENT_POLICY,
        dataset="unit",
        seed=7,
    )
    record = generated.pairs[0]
    forest = parse(
        record.tokens,
        rules=TOY,
        source=ClampedSource(
            TOY_LEXICON,
            [
                Clamp((position, position + 1), (toy_category(category),))
                for position, category in enumerate(record.variant)
            ],
            EXACT,
            EXACT,
            ErrorPolicy(),
        ),
        semiring=LogInside(),
        scorer=Unweighted(),
        goal=TOY_GOAL,
        unary_depth=0,
        policy=ParsePolicy(Retain.FOREST, Reachability.ALL),
    )
    decided = [
        extract_variables(
            derivation,
            TOY.normal_form,
            render=a_render,
            dataset="unit",
            arm="unconstrained",
        ).assignment
        for root in TOY_GOAL.roots(forest)
        for derivation in forest.derivations(root)
    ]

    assert generated.refusals == ()
    assert dict(record.variant_assignment) in [dict(found) for found in decided]


def test_a_confirmation_rejects_a_record_deciding_different_variables() -> None:
    base = a_lexical_assignment("S/S", "Sdcl", "S")
    variant = a_lexical_assignment("S/S", "Sng", "S")
    del variant[SplitAt((0, 2))]

    with pytest.raises(ValueError, match="different variables"):
        confirm_pair(a_record(base, variant, TypeAt((1, 2))))


def test_a_confirmation_rejects_a_record_differing_at_two_variables() -> None:
    base = a_lexical_assignment("S/S", "Sdcl", "S")
    variant = a_lexical_assignment("S/S", "Sng", "Sother")

    with pytest.raises(ValueError, match="differ at 2 variables"):
        confirm_pair(a_record(base, variant, TypeAt((1, 2))))


def test_a_confirmation_rejects_a_record_differing_at_no_variable() -> None:
    base = a_lexical_assignment("S/S", "Sdcl", "S")

    with pytest.raises(ValueError, match="differ at 0 variables"):
        confirm_pair(a_record(base, dict(base), TypeAt((1, 2))))


def test_a_confirmation_rejects_a_record_varying_a_cell_above_one_token() -> None:
    base = a_lexical_assignment("S/S", "Sdcl", "S")
    variant = a_lexical_assignment("S/S", "Sdcl", "Sother")

    with pytest.raises(ValueError, match="not the category at a cell"):
        confirm_pair(a_record(base, variant, TypeAt((0, 2))))


def test_a_confirmation_rejects_a_record_varying_a_split() -> None:
    base = a_lexical_assignment("S/S", "Sdcl", "S")
    variant = a_lexical_assignment("S/S", "Sdcl", "S")
    variant[SplitAt((0, 2))] = 2

    with pytest.raises(ValueError, match="not the category at a cell"):
        confirm_pair(a_record(base, variant, SplitAt((0, 2))))


def test_a_confirmation_rejects_a_record_claiming_the_wrong_variable() -> None:
    base = a_lexical_assignment("S/S", "Sdcl", "S")
    variant = a_lexical_assignment("S/Sother", "Sdcl", "S")

    with pytest.raises(ValueError, match="claims to vary"):
        confirm_pair(
            a_record(
                base,
                variant,
                TypeAt((1, 2)),
                base=("S/S", "Sdcl"),
                variant=("S/Sother", "Sdcl"),
            )
        )


def test_a_confirmation_rejects_a_record_whose_lexical_items_both_vary() -> None:
    base = a_lexical_assignment("S/S", "Sdcl", "S")
    variant = a_lexical_assignment("S/S", "Sng", "S")

    with pytest.raises(ValueError, match="differ at 0, 1"):
        confirm_pair(
            a_record(
                base,
                variant,
                TypeAt((1, 2)),
                base=("S/S", "Sdcl"),
                variant=("S/Sother", "Sng"),
            )
        )


def test_a_confirmation_rejects_a_record_whose_lexical_items_agree() -> None:
    base = a_lexical_assignment("S/S", "Sdcl", "S")
    variant = a_lexical_assignment("S/S", "Sng", "S")

    with pytest.raises(ValueError, match="differ at no position"):
        confirm_pair(
            a_record(
                base,
                variant,
                TypeAt((1, 2)),
                base=("S/S", "Sdcl"),
                variant=("S/S", "Sdcl"),
            )
        )


def test_a_confirmation_rejects_a_record_disagreeing_with_its_own_lexicon() -> None:
    base = a_lexical_assignment("S/S", "Sdcl", "S")
    variant = a_lexical_assignment("S/S", "Sng", "S")

    with pytest.raises(ValueError, match="is not the pair of categories"):
        confirm_pair(
            a_record(
                base,
                variant,
                TypeAt((1, 2)),
                base=("S/S", "Sother"),
                variant=("S/S", "Sng"),
            )
        )


def test_a_confirmation_reads_a_generated_pair() -> None:
    with pytest.raises(TypeError, match="reads a GeneratedPair"):
        confirm_pair("really sleeps")  # ty: ignore[invalid-argument-type]


def test_a_record_names_a_high_level_variable() -> None:
    base = a_lexical_assignment("S/S", "Sdcl", "S")

    with pytest.raises(TypeError, match="varies exactly a TypeAt"):
        a_record(base, dict(base), (1, 2))  # ty: ignore[invalid-argument-type]


def test_a_structural_record_refuses_named_and_subclassed_variables() -> None:
    class SpecializedTypeAt(TypeAt):
        pass

    base = a_lexical_assignment("S/S", "Sdcl", "S")

    with pytest.raises(TypeError, match="does not admit NamedVariable"):
        a_record(base, dict(base), NamedVariable("external", "category"))

    with pytest.raises(TypeError, match="varies exactly a TypeAt"):
        a_record(base, dict(base), SpecializedTypeAt((1, 2)))


def test_a_record_requires_a_provenance() -> None:
    base = a_lexical_assignment("S/S", "Sdcl", "S")

    with pytest.raises(TypeError, match="requires a Provenance"):
        GeneratedPair(
            ("really", "sleeps"),
            ("S/S", "Sdcl"),
            ("S/S", "Sng"),
            base,
            dict(base),
            TypeAt((1, 2)),
            "toy:1",
            "unit",  # ty: ignore[invalid-argument-type]
        )


def test_a_record_covers_a_nonempty_string() -> None:
    base = a_lexical_assignment("S/S", "Sdcl", "S")

    with pytest.raises(ValueError, match="nonempty string"):
        a_record(base, dict(base), TypeAt((0, 1)), tokens=(), base=(), variant=())


def test_a_record_names_one_category_per_token() -> None:
    base = a_lexical_assignment("S/S", "Sdcl", "S")

    with pytest.raises(ValueError, match="names 1 categories for 2 tokens"):
        a_record(base, dict(base), TypeAt((1, 2)), variant=("S/S",))


def test_a_record_varies_a_cell_inside_the_string() -> None:
    base = a_lexical_assignment("S/S", "Sdcl", "S")

    with pytest.raises(ValueError, match="runs past the 2 tokens"):
        a_record(base, dict(base), TypeAt((2, 3)))


def test_a_record_names_the_seed_it_was_drawn_under() -> None:
    base = a_lexical_assignment("S/S", "Sdcl", "S")

    with pytest.raises(ValueError, match="names the seed"):
        GeneratedPair(
            ("really", "sleeps"),
            ("S/S", "Sdcl"),
            ("S/S", "Sng"),
            base,
            dict(base),
            TypeAt((1, 2)),
            "toy:1",
            Provenance("grammar-counterfactual", "unit", None),
        )


def test_a_record_names_the_grammar_it_came_from() -> None:
    base = a_lexical_assignment("S/S", "Sdcl", "S")

    with pytest.raises(ValueError, match="record's grammar"):
        GeneratedPair(
            ("really", "sleeps"),
            ("S/S", "Sdcl"),
            ("S/S", "Sng"),
            base,
            dict(base),
            TypeAt((1, 2)),
            "",
            Provenance("grammar-counterfactual", "unit", 0),
        )


def test_a_record_copies_the_assignments_it_was_handed() -> None:
    base = a_lexical_assignment("S/S", "Sdcl", "S")
    variant = a_lexical_assignment("S/S", "Sng", "S")
    record = a_record(base, variant, TypeAt((1, 2)))

    base[TypeAt((1, 2))] = "Sother"
    variant[TypeAt((1, 2))] = "Sother"

    assert record.base_assignment[TypeAt((1, 2))] == "Sdcl"
    assert record.variant_assignment[TypeAt((1, 2))] == "Sng"
    assert confirm_pair(record).varied == TypeAt((1, 2))


def test_a_record_serializes_its_string_its_assignments_and_its_provenance() -> None:
    generated = a_generator().generate(
        (an_aligned_input(("really", "sleeps")),),
        policy=WORD_ALIGNMENT_POLICY,
        dataset="unit",
        seed=7,
    )
    record = generated.pairs[0]

    written = record.to_json()

    assert generated.refusals == ()
    assert written["tokens"] == ["really", "sleeps"]
    assert written["base"] == ["S/S", "Sdcl"]
    assert written["variant"] == ["S/S", "Sng"]
    assert written["varied"] == {"kind": "type", "cell": [1, 2], "split": None}
    assert written["span"] == [1, 2]
    assert written["form"] == "sleeps"
    assert written["grammar"] == "toy:1"
    assert written["provenance"] == {
        "method": "grammar-counterfactual",
        "dataset": "unit",
        "seed": 7,
    }
    assert written["base_assignment"] == [
        {
            "variable": {"kind": "combinator", "cell": [0, 2], "split": 1},
            "value": "apply",
        },
        {"variable": {"kind": "split", "cell": [0, 2], "split": None}, "value": 1},
        {"variable": {"kind": "type", "cell": [0, 1], "split": None}, "value": "S/S"},
        {"variable": {"kind": "type", "cell": [0, 2], "split": None}, "value": "S"},
        {"variable": {"kind": "type", "cell": [1, 2], "split": None}, "value": "Sdcl"},
    ]
    assert written["variant_assignment"] == [
        {
            "variable": {"kind": "combinator", "cell": [0, 2], "split": 1},
            "value": "apply",
        },
        {"variable": {"kind": "split", "cell": [0, 2], "split": None}, "value": 1},
        {"variable": {"kind": "type", "cell": [0, 1], "split": None}, "value": "S/S"},
        {"variable": {"kind": "type", "cell": [0, 2], "split": None}, "value": "S"},
        {"variable": {"kind": "type", "cell": [1, 2], "split": None}, "value": "Sng"},
    ]


def test_a_generator_reads_at_least_two_derivations() -> None:
    with pytest.raises(ValueError, match="reading 1 of them generates nothing"):
        a_generator(readings=1)


def test_a_generator_names_the_grammar_it_generates_from() -> None:
    with pytest.raises(ValueError, match="generator's grammar"):
        CounterfactualGenerator(
            TOY,
            TOY_LEXICON,
            TOY_GOAL,
            a_render,
            "",
            unary_depth=0,
            parse_policy=ParsePolicy(Retain.FOREST, Reachability.ALL),
        )


def test_a_generator_names_the_dataset_it_generates_over() -> None:
    with pytest.raises(ValueError, match="generated pair's dataset"):
        a_generator().generate(
            (an_aligned_input(("really", "sleeps")),),
            policy=WORD_ALIGNMENT_POLICY,
            dataset="",
            seed=7,
        )


def test_a_generator_draws_under_a_nonnegative_seed() -> None:
    with pytest.raises(ValueError, match="generator's seed"):
        a_generator().generate(
            (an_aligned_input(("really", "sleeps")),),
            policy=WORD_ALIGNMENT_POLICY,
            dataset="unit",
            seed=-1,
        )


CCG_FEATURES: Final = ("dcl", "ng", "b", "pss", "to", "adj", "conj")
CCG_COMPILED: Final = declared_rules(
    ccg_signature(
        "N",
        "NP",
        "S",
        "conj",
        operators=(
            OperatorSpec(name="N[conj]", sort="Cat", erases_to="N"),
            OperatorSpec(name="NP[conj]", sort="Cat", erases_to="NP"),
            *(
                OperatorSpec(name=f"S[{feature}]", sort="Cat", erases_to="S")
                for feature in CCG_FEATURES
            ),
        ),
        conjunctions=ConjunctionsSpec(
            markers=("conj",),
            pairs=tuple(
                ConjunctionPairSpec(unmarked=unmarked, marked=f"{unmarked}[conj]")
                for unmarked in ("N", "NP", "S")
            ),
        ),
        name="toy-ccg",
    ),
    RuleSetSpec(
        name="toy-ccg",
        rules=(
            ApplicationSpec(view="/"),
            ApplicationSpec(view="\\"),
            CompositionSpec(direction="unrestricted", order=1, view="/", crossed=False),
            CompositionSpec(
                direction="unrestricted", order=1, view="\\", crossed=False
            ),
        ),
    ),
    matching=FeatureErasureMatchingSpec(),
)
CCG_STORE: Final = CCG_COMPILED.context.store
CCG_SYNTAX: Final = CCG_COMPILED.context.syntax


def a_category(text: str) -> Term:
    return CCG_SYNTAX.parse(CCG_STORE, text)


def a_ccg_render(term: Term) -> str:
    return CCG_SYNTAX.render(CCG_STORE, term)


CCG_RULES: Final = CCG_COMPILED.rule_set
SUBJECTS: Final = ("kim", "sandy", "lee", "robin", "jamie", "ash", "dana", "sam")
VERBS: Final = ("sleeps", "runs", "waits", "sings", "laughs", "reads")
ADVERBS: Final = ("today", "here", "quickly", "often")
FEATURES: Final = ("dcl", "ng", "b", "pss", "to", "adj")


def a_ccg_lexicon() -> StaticLexicon[Term]:
    entries = [Entry(form, a_category("NP")) for form in SUBJECTS]
    entries.extend(
        Entry(form, a_category(f"S[{feature}]\\NP"))
        for form in VERBS
        for feature in FEATURES
    )
    entries.extend(Entry(form, a_category("(S\\NP)\\(S\\NP)")) for form in ADVERBS)

    return StaticLexicon(entries)


def ccg_sentences() -> list[Sequence[str]]:
    short: list[Sequence[str]] = [
        [subject, verb, adverb]
        for subject in SUBJECTS
        for verb in VERBS
        for adverb in ADVERBS
    ]
    long: list[Sequence[str]] = [
        [subject, verb, first, second]
        for subject in SUBJECTS
        for verb in VERBS
        for first in ADVERBS
        for second in ADVERBS
    ]

    return short + long


def test_ten_thousand_confirmed_pairs_come_from_the_ccg_pack() -> None:
    generator = CounterfactualGenerator(
        CCG_RULES,
        a_ccg_lexicon(),
        RootGoal(TargetSet((a_category("S"),), CCG_RULES.matching)),
        a_ccg_render,
        "toy-ccg:1",
        unary_depth=1,
        parse_policy=ParsePolicy(Retain.FOREST, Reachability.ALL),
    )
    generated = generator.generate(
        tuple(an_aligned_input(sentence) for sentence in ccg_sentences()),
        policy=WORD_ALIGNMENT_POLICY,
        dataset="ccg-unit",
        seed=5,
    )
    found = generated.pairs[:10_000]

    assert len(found) == 10_000
    assert generated.refusals == ()
    assert generated.manifest.candidate_count == len(generated.pairs)
    assert generated.manifest.aligned_count == len(generated.pairs)
    assert all(confirm_pair(record).varied == record.varied for record in found)
    assert {record.grammar for record in found} == {"toy-ccg:1"}
    assert all(record.span[1] - record.span[0] == 1 for record in found)
    assert all(
        record.base[record.span[0]] != record.variant[record.span[0]]
        for record in found
    )


def test_a_generation_retains_the_exact_alignment_evidence_it_accepted() -> None:
    generation, alignment = a_generation()
    record = generation.pairs[0]

    assert generation.pair(record.id) is record
    assert generation.aligned_input(record.id) is generation.aligned_inputs[record.id]
    assert generation.aligned_input(record.id).alignments == (alignment,)


def test_retained_evidence_must_cover_exactly_the_accepted_pairs() -> None:
    generation, _ = a_generation()
    record = generation.pairs[0]

    with pytest.raises(ValueError, match=r"missing=.*foreign"):
        CounterfactualGeneration((record,), {}, (), generation.manifest)

    with pytest.raises(ValueError, match="duplicate pair identity"):
        CounterfactualGeneration(
            (record, record),
            {record.id: generation.aligned_input(record.id)},
            (),
            AlignmentManifest(
                WORD_ALIGNMENT_POLICY,
                generation.manifest.tokenizers,
                2,
                2,
            ),
        )


def test_a_model_pair_preserves_structure_inputs_and_exact_tokenizer_rows() -> None:
    pair = a_model_pair()

    assert isinstance(pair, CausalPair)
    assert pair.id
    assert pair.tokens == ("really", "sleeps")
    assert pair.base[pair.varied] == "Sdcl"
    assert pair.variant[pair.varied] == "Sng"
    assert pair.rows(Side.BASE) == (1,)
    assert pair.rows(Side.VARIANT) == (1,)
    assert pair.live_rows(Side.BASE) == 2
    assert pair.tokenizer == "word-tokenizer"
    assert copy(pair) is pair
    assert deepcopy(pair) is pair

    with pytest.raises(AttributeError, match="model causal pair is immutable"):
        pair.tokenizer = "other"  # ty: ignore[invalid-assignment]


def test_a_model_pair_carries_a_named_variable_on_the_unified_pair_route() -> None:
    variable = NamedVariable("external", "syntactic-category")
    causal = CounterfactualPair(
        {variable: "N"},
        {variable: "V"},
        variable,
        (0, 1),
        Provenance("external-counterfactual", "unit", 3),
    )
    alignment = TokenizerAlignment("word-tokenizer", ((0,),), (), 1)

    pair = ModelCausalPair(
        "named-pair",
        causal,
        ("runs",),
        WORD_ALIGNMENT_POLICY,
        base_inputs={"vectors": torch.zeros(1, 1, 3)},
        variant_inputs={"vectors": torch.ones(1, 1, 3)},
        base_alignment=alignment,
        variant_alignment=alignment,
    )

    assert pair.varied is variable
    assert pair.base[variable] == "N"
    assert pair.variant[variable] == "V"
    assert pair.span == (0, 1)
    assert pair.rows(Side.BASE) == (0,)


def test_a_model_pair_snapshots_caller_inputs_and_returns_fresh_copies() -> None:
    generation, alignment = a_generation()
    record = generation.pairs[0]
    evidence = generation.aligned_input(record.id)
    base = torch.zeros(1, 2, 3)
    variant = torch.ones(1, 2, 3)
    pair = ModelCausalPair(
        record.id,
        confirm_pair(record),
        evidence.tokens,
        generation.manifest.policy,
        base_inputs={"vectors": base},
        variant_inputs={"vectors": variant},
        base_alignment=alignment,
        variant_alignment=alignment,
    )
    base.fill_(9.0)
    variant.fill_(8.0)
    returned = pair.inputs(Side.BASE)["vectors"]
    returned.fill_(7.0)

    assert torch.equal(pair.inputs(Side.BASE)["vectors"], torch.zeros(1, 2, 3))
    assert torch.equal(pair.inputs(Side.VARIANT)["vectors"], torch.ones(1, 2, 3))


@pytest.mark.parametrize(
    ("base", "variant", "message"),
    [
        ({}, {"vectors": torch.ones(1, 2, 3)}, "nonempty mapping"),
        (
            {"vectors": torch.zeros(1, 2, 3)},
            {"other": torch.ones(1, 2, 3)},
            "different keys",
        ),
        (
            {"vectors": torch.zeros(1, 2, 3)},
            {"vectors": torch.ones(1, 2, 4)},
            "nonbatch shape",
        ),
        (
            {"vectors": torch.zeros(1, 2, 3)},
            {"vectors": torch.ones(1, 2, 3, dtype=torch.float64)},
            "dtype",
        ),
        (
            {"vectors": torch.zeros(1, 2, 3)},
            {"vectors": torch.zeros(1, 2, 3)},
            "identical",
        ),
        (
            {"vectors": torch.tensor([[[float("nan")]]])},
            {"vectors": torch.ones(1, 1, 1)},
            "nonfinite",
        ),
    ],
)
def test_malformed_or_noncausal_model_inputs_are_refused(
    base: dict[str, torch.Tensor],
    variant: dict[str, torch.Tensor],
    message: str,
) -> None:
    generation, alignment = a_generation()
    record = generation.pairs[0]

    with pytest.raises(ValueError, match=message):
        ModelCausalPair(
            record.id,
            confirm_pair(record),
            record.tokens,
            generation.manifest.policy,
            base_inputs=base,
            variant_inputs=variant,
            base_alignment=alignment,
            variant_alignment=alignment,
        )


def test_model_inputs_that_are_not_mappings_are_refused() -> None:
    generation, alignment = a_generation()
    record = generation.pairs[0]

    with pytest.raises(TypeError, match="mapping of named model inputs"):
        ModelCausalPair(
            record.id,
            confirm_pair(record),
            record.tokens,
            generation.manifest.policy,
            base_inputs=cast("Mapping[str, torch.Tensor]", 3),
            variant_inputs={"vectors": torch.ones(1, 2, 3)},
            base_alignment=alignment,
            variant_alignment=alignment,
        )


def test_a_model_pair_refuses_a_nontensor_named_input() -> None:
    generation, alignment = a_generation()
    record = generation.pairs[0]

    with pytest.raises(TypeError, match="is not a tensor"):
        ModelCausalPair(
            record.id,
            confirm_pair(record),
            record.tokens,
            generation.manifest.policy,
            base_inputs=cast("Mapping[str, torch.Tensor]", {"vectors": "not-a-tensor"}),
            variant_inputs={"vectors": torch.ones(1, 2, 3)},
            base_alignment=alignment,
            variant_alignment=alignment,
        )


def test_a_model_pair_refuses_an_alignment_for_another_token_count() -> None:
    generation, alignment = a_generation()
    record = generation.pairs[0]
    inconsistent = TokenizerAlignment("word-tokenizer", ((0,), (1,), (2,)), (), 3)

    with pytest.raises(ValueError, match="maps 3 source tokens, not the 2 tokens"):
        ModelCausalPair(
            record.id,
            confirm_pair(record),
            record.tokens,
            generation.manifest.policy,
            base_inputs={"vectors": torch.zeros(1, 2, 3)},
            variant_inputs={"vectors": torch.ones(1, 2, 3)},
            base_alignment=alignment,
            variant_alignment=inconsistent,
        )


@pytest.mark.parametrize(
    ("identity", "error", "message"),
    [
        ("", ValueError, "identity is empty"),
        (cast("str", 3), TypeError, "identity is not a string"),
    ],
)
def test_a_model_pair_requires_a_stable_identity(
    identity: str, error: type[TypeError | ValueError], message: str
) -> None:
    generation, alignment = a_generation()
    record = generation.pairs[0]

    with pytest.raises(error, match=message):
        ModelCausalPair(
            identity,
            confirm_pair(record),
            record.tokens,
            generation.manifest.policy,
            base_inputs={"vectors": torch.zeros(1, 2, 3)},
            variant_inputs={"vectors": torch.ones(1, 2, 3)},
            base_alignment=alignment,
            variant_alignment=alignment,
        )


def test_a_model_pair_requires_a_counterfactual_pair_and_policy() -> None:
    generation, alignment = a_generation()
    record = generation.pairs[0]

    def build(pair: CounterfactualPair, policy: AlignmentPolicy) -> ModelCausalPair:
        return ModelCausalPair(
            record.id,
            pair,
            record.tokens,
            policy,
            base_inputs={"vectors": torch.zeros(1, 2, 3)},
            variant_inputs={"vectors": torch.ones(1, 2, 3)},
            base_alignment=alignment,
            variant_alignment=alignment,
        )

    with pytest.raises(TypeError, match="requires a CounterfactualPair"):
        build(cast("CounterfactualPair", record), generation.manifest.policy)

    with pytest.raises(TypeError, match="requires an AlignmentPolicy"):
        build(
            confirm_pair(record),
            cast("AlignmentPolicy", "all"),
        )


@pytest.mark.parametrize(
    ("tokens", "message"),
    [
        (cast("tuple[str, ...]", ["really", "sleeps"]), "immutable tuple"),
        ((), "at least one source token"),
        (("really", ""), "source token 1 is empty"),
    ],
)
def test_a_model_pair_requires_exact_source_tokens(
    tokens: tuple[str, ...], message: str
) -> None:
    generation, alignment = a_generation()
    record = generation.pairs[0]
    error = TypeError if type(tokens) is not tuple else ValueError

    with pytest.raises(error, match=message):
        ModelCausalPair(
            record.id,
            confirm_pair(record),
            tokens,
            generation.manifest.policy,
            base_inputs={"vectors": torch.zeros(1, 2, 3)},
            variant_inputs={"vectors": torch.ones(1, 2, 3)},
            base_alignment=alignment,
            variant_alignment=alignment,
        )


def test_a_model_pair_refuses_a_structural_span_past_its_source_tokens() -> None:
    variable = TypeAt((1, 3))
    pair = CounterfactualPair(
        {variable: "A"},
        {variable: "B"},
        variable,
        variable.cell,
        Provenance("structural-counterfactual", "unit", 1),
    )
    alignment = TokenizerAlignment("word-tokenizer", ((0,), (1,)), (), 2)

    with pytest.raises(ValueError, match="past its 2 tokens"):
        ModelCausalPair(
            "past-source",
            pair,
            ("really", "sleeps"),
            WORD_ALIGNMENT_POLICY,
            base_inputs={"vectors": torch.zeros(1, 2, 3)},
            variant_inputs={"vectors": torch.ones(1, 2, 3)},
            base_alignment=alignment,
            variant_alignment=alignment,
        )


def test_a_model_pair_requires_two_alignments_for_one_tokenizer() -> None:
    generation, alignment = a_generation()
    record = generation.pairs[0]
    other = TokenizerAlignment("other-tokenizer", ((0,), (1,)), (), 2)

    with pytest.raises(TypeError, match="base alignment is a TokenizerAlignment"):
        ModelCausalPair(
            record.id,
            confirm_pair(record),
            record.tokens,
            generation.manifest.policy,
            base_inputs={"vectors": torch.zeros(1, 2, 3)},
            variant_inputs={"vectors": torch.ones(1, 2, 3)},
            base_alignment=cast("TokenizerAlignment", "word-tokenizer"),
            variant_alignment=alignment,
        )

    with pytest.raises(ValueError, match="do not identify one model"):
        ModelCausalPair(
            record.id,
            confirm_pair(record),
            record.tokens,
            generation.manifest.policy,
            base_inputs={"vectors": torch.zeros(1, 2, 3)},
            variant_inputs={"vectors": torch.ones(1, 2, 3)},
            base_alignment=alignment,
            variant_alignment=other,
        )


def test_a_model_pair_requires_equal_selected_row_extents() -> None:
    generation, _ = a_generation()
    record = generation.pairs[0]
    base = TokenizerAlignment("word-tokenizer", ((0,), (1, 2)), (), 3)
    variant = TokenizerAlignment("word-tokenizer", ((0,), (1,)), (), 2)

    with pytest.raises(ValueError, match="select 2 and 1 rows"):
        ModelCausalPair(
            record.id,
            confirm_pair(record),
            record.tokens,
            generation.manifest.policy,
            base_inputs={"vectors": torch.zeros(1, 3, 3)},
            variant_inputs={"vectors": torch.ones(1, 2, 3)},
            base_alignment=base,
            variant_alignment=variant,
        )


def test_a_model_joint_pair_keeps_distinct_rows_for_both_variables() -> None:
    outer = TypeAt((0, 4))
    inner = TypeAt((1, 3))
    structural = JointCounterfactualPair(
        {outer: "NP", inner: "N"},
        {outer: "S", inner: "NP"},
        (inner, outer),
        outer.cell,
        Provenance("joint-fixture", "unit", 4),
    )
    alignment = TokenizerAlignment("word-tokenizer", ((0,), (1,), (2,), (3,)), (), 4)
    pair = ModelJointCausalPair(
        "joint-model-pair",
        structural,
        ("a", "b", "c", "d"),
        AlignmentPolicy("all", "exclude", contiguous=True),
        base_inputs={"input_ids": torch.tensor([[1, 2, 3, 4]])},
        variant_inputs={"input_ids": torch.tensor([[1, 5, 6, 4]])},
        base_alignment=alignment,
        variant_alignment=alignment,
    )

    assert pair.variables == (outer, inner)
    assert pair.rows(Side.BASE, outer) == (0, 1, 2, 3)
    assert pair.rows(Side.BASE, inner) == (1, 2)
    assert pair.selected_rows(Side.VARIANT) == (0, 1, 2, 3)
    assert pair.inputs(Side.BASE)["input_ids"].tolist() == [[1, 2, 3, 4]]
