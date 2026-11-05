from typing import TYPE_CHECKING, cast

import pytest

from lofi.probe._causal import (
    PIECE_POLICIES,
    SPECIAL_TOKEN_POLICIES,
    AlignedInput,
    AlignmentPolicy,
    CausalPair,
    CellAssignment,
    CombinatorAt,
    CounterfactualPair,
    HighLevelVariable,
    IIAResult,
    JointCounterfactualPair,
    NamedVariable,
    PiecePolicy,
    RowAligner,
    RowEvidence,
    Side,
    SpecialTokenPolicy,
    SplitAt,
    TokenizerAlignment,
    TypeAt,
    strictly_dominates,
    tokenizer_alignment,
    variable_label,
)
from lofi.probe._provenance import Provenance
from lofi.stats import NullProvenance, p_value_against_replicates

if TYPE_CHECKING:
    from collections.abc import Callable

GENERATED_BY = Provenance("grammar-counterfactuals", "ccgbank-dev", 7)

CATEGORY = TypeAt((0, 2))
SPLIT = SplitAt((0, 4))
COMBINATOR = CombinatorAt((0, 4), 2)
NAMED = NamedVariable("external", "syntactic-category")

BASE: CellAssignment = {CATEGORY: "NP", SPLIT: 2, COMBINATOR: "forward-application"}


def a_null(name: str = "cell-permutation") -> NullProvenance:
    tested = p_value_against_replicates(
        3.0,
        [1.0, 2.0, 2.5],
        tail="greater",
        null=name,
        family="permutation",
        exact=False,
        seed=3,
    )

    return tested.provenance


def a_variant(**changed: str | int) -> CellAssignment:
    by_name = {"category": CATEGORY, "split": SPLIT, "combinator": COMBINATOR}

    return dict(BASE) | {by_name[field]: value for field, value in changed.items()}


def test_the_variables_are_the_three_decisions_a_derivation_makes() -> None:
    assert CATEGORY.kind == "type"
    assert SPLIT.kind == "split"
    assert COMBINATOR.kind == "combinator"


def test_every_variable_has_one_shared_stable_report_label() -> None:
    assert variable_label(CATEGORY) == "type-0-2"
    assert variable_label(SPLIT) == "split-0-4"
    assert variable_label(COMBINATOR) == "combinator-0-4-2"
    assert variable_label(NAMED) == "named-external-syntactic-category"


def test_every_variable_names_the_cell_it_is_about() -> None:
    assert CATEGORY.cell == (0, 2)
    assert SPLIT.cell == (0, 4)
    assert COMBINATOR.cell == (0, 4)
    assert COMBINATOR.split == 2


def test_a_combinator_at_one_split_is_a_different_variable_from_another() -> None:
    assert CombinatorAt((0, 4), 2) != CombinatorAt((0, 4), 3)
    assert CombinatorAt((0, 4), 2) == COMBINATOR
    assert hash(CombinatorAt((0, 4), 2)) == hash(COMBINATOR)


def test_two_variants_over_one_cell_are_not_the_same_variable() -> None:
    assert TypeAt((0, 4)) != SplitAt((0, 4))
    assert len({TypeAt((0, 4)), SplitAt((0, 4))}) == 2
    assert TypeAt((0, 4)) != (0, 4)


def test_a_named_variable_has_an_immutable_qualified_identity() -> None:
    same = NamedVariable("external", "syntactic-category")

    assert NAMED.kind == "named"
    assert NAMED.namespace == "external"
    assert NAMED.name == "syntactic-category"
    assert same == NAMED
    assert hash(NAMED) == hash(same)
    assert NamedVariable("other", "syntactic-category") != NAMED
    assert repr(NAMED) == "NamedVariable('external', 'syntactic-category')"
    with pytest.raises(AttributeError):
        NAMED.__setattr__("_name", "changed")
    assert not hasattr(NAMED, "__dict__")


@pytest.mark.parametrize(("namespace", "name"), [("", "category"), ("ext", "")])
def test_a_named_variable_requires_a_namespace_and_name(
    namespace: str, name: str
) -> None:
    with pytest.raises(ValueError, match="is empty"):
        NamedVariable(namespace, name)


def test_a_variable_over_a_cell_that_covers_nothing_is_refused() -> None:
    for build in (TypeAt, SplitAt):
        with pytest.raises(ValueError, match="covers nothing"):
            build((2, 2))


def test_a_combinator_whose_split_falls_outside_its_cell_is_refused() -> None:
    with pytest.raises(ValueError, match="does not lie inside"):
        CombinatorAt((0, 4), 4)


def test_a_variable_cannot_be_rebound() -> None:
    for held in ("cell", "kind"):
        with pytest.raises(AttributeError):
            setattr(CATEGORY, held, None)

    for held in ("cell", "split"):
        with pytest.raises(AttributeError):
            setattr(COMBINATOR, held, None)

    assert not hasattr(CATEGORY, "__dict__")
    assert CATEGORY.cell == (0, 2)


def test_a_pair_records_what_each_side_decided_and_which_decision_moved() -> None:
    variant = a_variant(category="S")
    pair = CounterfactualPair(BASE, variant, CATEGORY, (0, 4), GENERATED_BY)

    assert pair.base[CATEGORY] == "NP"
    assert pair.variant[CATEGORY] == "S"
    assert pair.varied == CATEGORY
    assert pair.span == (0, 4)
    assert pair.provenance is GENERATED_BY
    assert isinstance(pair, CausalPair)


def test_a_pair_varying_a_split_point_records_a_position_and_not_a_name() -> None:
    pair = CounterfactualPair(BASE, a_variant(split=1), SPLIT, (0, 4), GENERATED_BY)

    assert pair.variant[SPLIT] == 1


def test_a_named_variable_uses_the_pairs_local_source_span() -> None:
    pair = CounterfactualPair({NAMED: "NP"}, {NAMED: "S"}, NAMED, (2, 3), GENERATED_BY)

    assert pair.varied is NAMED
    assert pair.span == (2, 3)


def test_structural_joint_pairs_refuse_named_variables_explicitly() -> None:
    with pytest.raises(TypeError, match="NamedVariable"):
        JointCounterfactualPair(
            {SPLIT: 2, NAMED: "NP"},
            {SPLIT: 1, NAMED: "S"},
            (SPLIT, NAMED),
            (0, 4),
            GENERATED_BY,
        )

    with pytest.raises(TypeError, match="NamedVariable"):
        strictly_dominates(SPLIT, NAMED)


def test_a_pair_differing_in_two_variables_is_refused() -> None:
    with pytest.raises(ValueError, match="differ at 2 variables"):
        CounterfactualPair(
            BASE,
            a_variant(category="S", split=1),
            CATEGORY,
            (0, 4),
            GENERATED_BY,
        )


def test_a_pair_that_varies_nothing_is_refused() -> None:
    with pytest.raises(ValueError, match="everything the same way"):
        CounterfactualPair(BASE, dict(BASE), CATEGORY, (0, 4), GENERATED_BY)


def test_a_pair_claiming_to_vary_one_thing_and_varying_another_is_refused() -> None:
    with pytest.raises(ValueError, match="claims to vary"):
        CounterfactualPair(BASE, a_variant(split=1), CATEGORY, (0, 4), GENERATED_BY)


def test_two_sides_deciding_different_variables_are_not_a_counterfactual() -> None:
    thinner: CellAssignment = {CATEGORY: "S", SPLIT: 2}

    with pytest.raises(ValueError, match="decide different variables"):
        CounterfactualPair(BASE, thinner, CATEGORY, (0, 4), GENERATED_BY)


def test_a_varied_cell_outside_the_span_being_intervened_on_is_refused() -> None:
    with pytest.raises(ValueError, match="falls outside the span"):
        CounterfactualPair(
            BASE, a_variant(category="S"), CATEGORY, (1, 4), GENERATED_BY
        )


def test_a_decision_that_is_neither_a_name_nor_a_position_is_refused() -> None:
    with pytest.raises(TypeError, match="neither a name nor a position"):
        CounterfactualPair(
            cast("CellAssignment", {CATEGORY: 0.5}),
            {CATEGORY: "S"},
            CATEGORY,
            (0, 4),
            GENERATED_BY,
        )


def test_an_assignment_keyed_by_something_that_is_not_a_variable_is_refused() -> None:
    with pytest.raises(TypeError, match="decides high-level variables"):
        CounterfactualPair(
            cast("CellAssignment", {(0, 2): "NP"}),
            {CATEGORY: "S"},
            CATEGORY,
            (0, 4),
            GENERATED_BY,
        )


def test_a_pair_varies_a_declared_variable_and_not_a_logical_form() -> None:
    with pytest.raises(TypeError, match="varies a high-level variable"):
        CounterfactualPair(
            BASE,
            a_variant(category="S"),
            cast("HighLevelVariable", "LF(0,2)"),
            (0, 4),
            GENERATED_BY,
        )


def test_a_pairs_assignments_cannot_be_written_through() -> None:
    pair = CounterfactualPair(
        BASE, a_variant(category="S"), CATEGORY, (0, 4), GENERATED_BY
    )

    with pytest.raises(TypeError):
        cast("dict[HighLevelVariable, str | int]", pair.base)[CATEGORY] = "S"

    assert pair.base[CATEGORY] == "NP"


def test_a_policy_states_the_three_rules_that_lay_a_span_on_tokens() -> None:
    policy = AlignmentPolicy("last", "include", contiguous=False)

    assert policy.pieces == "last"
    assert policy.specials == "include"
    assert policy.contiguous is False


def test_every_declared_policy_builds_and_reports_itself() -> None:
    for pieces in PIECE_POLICIES:
        for specials in SPECIAL_TOKEN_POLICIES:
            policy = AlignmentPolicy(pieces, specials, contiguous=True)

            assert (policy.pieces, policy.specials) == (pieces, specials)


def test_a_piece_policy_nobody_declared_is_refused() -> None:
    with pytest.raises(ValueError, match="no piece policy is called"):
        AlignmentPolicy(cast("PiecePolicy", "middle"), "exclude", contiguous=True)


def test_a_special_token_policy_nobody_declared_is_refused() -> None:
    with pytest.raises(ValueError, match="no special-token policy"):
        AlignmentPolicy("first", cast("SpecialTokenPolicy", "drop"), contiguous=True)


def test_contiguity_is_a_boolean_and_not_a_truthy_number() -> None:
    with pytest.raises(TypeError, match="contiguity is not a boolean"):
        AlignmentPolicy("first", "exclude", contiguous=cast("bool", 1))


def test_a_policy_is_frozen() -> None:
    policy = AlignmentPolicy("first", "exclude", contiguous=True)

    with pytest.raises(AttributeError):
        policy.__setattr__("_pieces", "last")

    with pytest.raises(AttributeError):
        policy.__delattr__("_pieces")

    assert not hasattr(policy, "__dict__")
    assert policy.pieces == "first"


def test_policies_stating_the_same_three_rules_are_one_policy() -> None:
    policy = AlignmentPolicy("first", "exclude", contiguous=True)

    assert policy == AlignmentPolicy("first", "exclude", contiguous=True)
    assert hash(policy) == hash(AlignmentPolicy("first", "exclude", contiguous=True))
    assert policy != AlignmentPolicy("all", "exclude", contiguous=True)
    assert policy != AlignmentPolicy("first", "include", contiguous=True)
    assert policy != AlignmentPolicy("first", "exclude", contiguous=False)
    assert policy != ("first", "exclude", True)


def test_a_tokenizer_alignment_preserves_exact_row_boundaries() -> None:
    alignment = TokenizerAlignment(
        "bert-base@revision",
        ((1,), (2, 3)),
        (0, 4),
        5,
    )

    assert alignment.tokenizer == "bert-base@revision"
    assert alignment.token_rows == ((1,), (2, 3))
    assert alignment.special_rows == (0, 4)
    assert alignment.live_rows == 5
    assert alignment.model_rows(
        (1, 2), AlignmentPolicy("all", "exclude", contiguous=True)
    ) == (2, 3)
    assert alignment.model_rows(
        (1, 2), AlignmentPolicy("first", "exclude", contiguous=True)
    ) == (2,)
    assert alignment.model_rows(
        (0, 2), AlignmentPolicy("all", "include", contiguous=True)
    ) == (0, 1, 2, 3, 4)


def test_equal_row_counts_do_not_erase_different_or_gapped_boundaries() -> None:
    gapped = TokenizerAlignment("wordpiece", ((0,), (1, 3)), (2,), 4)
    adjacent = TokenizerAlignment("wordpiece", ((0,), (1, 2)), (3,), 4)

    assert gapped != adjacent
    assert len(gapped.token_rows[1]) == len(adjacent.token_rows[1])
    assert len({gapped, adjacent}) == 2


@pytest.mark.parametrize("identity", ["", cast("str", 7)])
def test_a_tokenizer_alignment_requires_a_well_formed_identity(identity: str) -> None:
    with pytest.raises((TypeError, ValueError), match="identity"):
        TokenizerAlignment(identity, ((0,),), (), 1)


@pytest.mark.parametrize(
    ("token_rows", "special_rows", "live_rows", "detail"),
    [
        ((), (0,), 1, "at least one source token"),
        (((0,), ()), (), 1, "zero model rows"),
        (((0, 0),), (), 1, "repeats or overlaps"),
        (((0, 1), (1, 2)), (), 3, "repeats or overlaps"),
        (((1,), (0,)), (), 2, "unordered"),
        (((0, 2),), (1,), 2, "outside"),
    ],
)
def test_malformed_source_row_mappings_are_refused(
    token_rows: tuple[tuple[int, ...], ...],
    special_rows: tuple[int, ...],
    live_rows: int,
    detail: str,
) -> None:
    with pytest.raises(ValueError, match=detail):
        TokenizerAlignment("tokenizer", token_rows, special_rows, live_rows)


@pytest.mark.parametrize(
    ("special_rows", "detail"),
    [
        ((0,), "also belongs"),
        ((1, 1), "repeated or unordered"),
        ((3,), "outside"),
        ((), "do not complete"),
    ],
)
def test_inconsistent_special_rows_are_refused(
    special_rows: tuple[int, ...], detail: str
) -> None:
    with pytest.raises(ValueError, match=detail):
        TokenizerAlignment("tokenizer", ((0,), (2,)), special_rows, 3)


def test_an_aligned_input_requires_total_distinct_named_alignments() -> None:
    one_word = TokenizerAlignment("one-word", ((0,),), (), 1)
    first = TokenizerAlignment("same", ((0,), (1,)), (), 2)
    second = TokenizerAlignment("same", ((0,), (1,)), (), 2)

    with pytest.raises(ValueError, match="maps 1 source tokens"):
        AlignedInput(("really", "sleeps"), (one_word,))

    with pytest.raises(ValueError, match="appears more than once"):
        AlignedInput(("really", "sleeps"), (first, second))


def test_aligned_inputs_and_tokenizer_alignments_are_deeply_frozen() -> None:
    alignment = TokenizerAlignment("words", ((0,), (1,)), (), 2)
    aligned = AlignedInput(("really", "sleeps"), (alignment,))

    with pytest.raises(AttributeError):
        alignment.__setattr__("_token_rows", ((0,),))

    with pytest.raises(AttributeError):
        aligned.__setattr__("_tokens", ("changed",))

    assert aligned.tokens == ("really", "sleeps")
    assert aligned.alignments == (alignment,)
    assert aligned.tokenizers == ("words",)


class _Rows:
    def __init__(
        self,
        word_rows: tuple[tuple[int, ...], ...],
        special_rows: tuple[int, ...],
        live_rows: int,
    ) -> None:
        self._word_rows = word_rows
        self._special_rows = special_rows
        self._live_rows = live_rows

    @property
    def word_rows(self) -> tuple[tuple[int, ...], ...]:
        return self._word_rows

    @property
    def special_rows(self) -> tuple[int, ...]:
        return self._special_rows

    @property
    def live_rows(self) -> int:
        return self._live_rows


class _Aligner:
    def __init__(self, identity: str, rows: _Rows) -> None:
        self._identity = identity
        self._rows = rows
        self.read: list[tuple[str, ...]] = []

    @property
    def identity(self) -> str:
        return self._identity

    def align(self, words: tuple[str, ...], /) -> RowEvidence:
        self.read.append(words)

        return self._rows


def test_a_row_aligner_yields_the_exact_alignment_under_its_identity() -> None:
    aligner = _Aligner("an-org/a-model@" + "a" * 40, _Rows(((1,), (2, 4)), (0, 3), 5))

    alignment = tokenizer_alignment(aligner, ("really", "sleeps"))

    assert isinstance(aligner, RowAligner)
    assert aligner.read == [("really", "sleeps")]
    assert alignment == TokenizerAlignment(
        "an-org/a-model@" + "a" * 40, ((1,), (2, 4)), (0, 3), 5
    )
    assert alignment.token_rows[1] == (2, 4)


def test_row_evidence_for_another_token_count_is_refused() -> None:
    aligner = _Aligner("words", _Rows(((0,),), (), 1))

    with pytest.raises(ValueError, match="rows for 1 words, not the 2 source"):
        tokenizer_alignment(aligner, ("really", "sleeps"))


def test_malformed_row_evidence_is_refused_as_a_tokenizer_alignment() -> None:
    aligner = _Aligner("words", _Rows(((0,), (0,)), (), 1))

    with pytest.raises(ValueError, match="repeats or overlaps"):
        tokenizer_alignment(aligner, ("really", "sleeps"))


def test_a_result_carries_the_accuracy_the_variable_the_rank_and_the_null() -> None:
    null = a_null()
    result = IIAResult(0.875, NAMED, 8, null)

    assert result.accuracy == 0.875
    assert result.variable is NAMED
    assert not hasattr(result, "cell")
    assert result.rank == 8
    assert result.null is null
    assert result.null.null == "cell-permutation"


def test_a_result_has_no_way_to_omit_the_null_it_was_tested_against() -> None:
    build = cast("Callable[..., IIAResult]", IIAResult)

    with pytest.raises(TypeError, match="'null'"):
        build(0.875, NAMED, 8)


def test_an_accuracy_tested_against_a_bare_name_is_refused() -> None:
    with pytest.raises(TypeError, match="tested against a NullProvenance"):
        IIAResult(0.875, NAMED, 8, cast("NullProvenance", "cell-permutation"))


def test_an_accuracy_outside_the_unit_interval_is_refused() -> None:
    for accuracy in (-0.1, 1.5):
        with pytest.raises(ValueError, match="not a proportion"):
            IIAResult(accuracy, NAMED, 8, a_null())


def test_a_zero_width_result_reports_an_unaligned_featurizer() -> None:
    result = IIAResult(0.875, NAMED, 0, a_null())

    assert result.rank == 0


def test_a_result_requires_an_exact_high_level_variable() -> None:
    with pytest.raises(TypeError, match="high-level variable"):
        IIAResult(0.875, cast("HighLevelVariable", (0, 2)), 8, a_null())


def test_results_recording_the_same_four_fields_are_one_result() -> None:
    result = IIAResult(0.875, NAMED, 8, a_null())

    assert result == IIAResult(0.875, NAMED, 8, a_null())
    assert hash(result) == hash(IIAResult(0.875, NAMED, 8, a_null()))
    assert result != IIAResult(0.9, NAMED, 8, a_null())
    assert result != IIAResult(0.875, CATEGORY, 8, a_null())
    assert result != IIAResult(0.875, NAMED, 16, a_null())
    assert result != IIAResult(0.875, NAMED, 8, a_null("scrambled-grammar"))
    assert result != 0.875


def test_a_result_cannot_be_rebound() -> None:
    result = IIAResult(0.875, NAMED, 8, a_null())

    for held in ("accuracy", "variable", "rank", "null"):
        with pytest.raises(AttributeError):
            setattr(result, held, None)

    assert not hasattr(result, "__dict__")


def test_a_pair_has_two_sides_and_each_names_one_of_its_derivations() -> None:
    pair = CounterfactualPair(
        BASE, a_variant(category="S"), CATEGORY, (0, 4), GENERATED_BY
    )
    by_side = {Side.BASE: pair.base, Side.VARIANT: pair.variant}

    assert set(Side) == {Side.BASE, Side.VARIANT}
    assert by_side[Side.BASE][CATEGORY] == "NP"
    assert by_side[Side.VARIANT][CATEGORY] == "S"
    assert Side.BASE is not Side.VARIANT
