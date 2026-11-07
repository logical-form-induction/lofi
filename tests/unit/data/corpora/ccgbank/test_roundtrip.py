from copy import copy, deepcopy
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Never, override

import pytest
from tests.unit.data.deps.test__ccg import (
    CategoryAlgebraCase,
    CCGCategoryAlgebraConformanceTests,
)

from lofi.data import DerivationNode, Token, TokenSpan
from lofi.data.corpora.ccgbank import (
    CCGBANK_HEAD_SELECTIONS,
    AutoLeaf,
    AutoTree,
    MarkedCategoryAlgebra,
    RoundTripCounts,
    ccgbank_round_trip_projector,
    evaluated,
    gold_sentence,
    round_trip,
    score_arcs,
)
from lofi.data.corpora.ccgbank._auto_parse import parse_auto
from lofi.data.corpora.ccgbank._parg_parse import parse_parg
from lofi.data.deps import CCGDependencyProjector, project_dependencies
from lofi.spec import HeadSelection

if TYPE_CHECKING:
    from lofi.data import Derivation
    from lofi.data.corpora.ccgbank import LabeledArc
    from lofi.data.deps import DecoratedCategory, HeadProjection, ProjectedDependency

AUTO: Final = (
    "ID=synthetic.1 PARSER=GOLD NUMPARSE=1\n"
    "(<T S[dcl] 0 2> (<T S[dcl] 1 2> (<T NP 1 2> "
    "(<L NP[nb]/N DT DT the NP[nb]_1/N_1>) (<L N NN NN dog N>) ) "
    "(<T S[dcl]\\NP 0 2> (<L (S[dcl]\\NP)/NP VBD VBD saw (S[dcl]\\NP_2)/NP_3>) "
    "(<T NP 1 2> (<L NP[nb]/N DT DT a NP[nb]_4/N_4>) (<L N NN NN cat N>) ) ) ) "
    "(<L . . . . .>) )\n"
)
PARG_LINES: Final = (
    "1\t0\tNP[nb]/N\t1\tdog the\n",
    "1\t2\t(S[dcl]\\NP)/NP\t1\tdog saw\n",
    "4\t2\t(S[dcl]\\NP)/NP\t2\tcat saw\n",
    "4\t3\tNP[nb]/N\t1\tcat a\n",
)
COORDINATED: Final = (
    "(<T S[dcl] 1 2> (<T NP 0 2> (<T NP 0 1> (<L N NNS NNS cats N>) ) "
    "(<T NP[conj] 1 2> (<L conj CC CC and conj>) "
    "(<T NP 0 1> (<L N NNS NNS dogs N>) ) ) ) "
    "(<L S[dcl]\\NP VBD VBD barked S[dcl]\\NP_1>) )"
)
TRAILING_COMMA: Final = (
    "(<T S[dcl] 1 2> (<T NP 0 2> (<T NP 0 1> (<L N NNS NNS dogs N>) ) "
    "(<L , , , , ,>) ) (<L S[dcl]\\NP VBD VBD barked S[dcl]\\NP_1>) )"
)
LEADING_COMMA: Final = (
    "(<T S[dcl] 1 2> (<T NP 1 2> (<L , , , , ,>) "
    "(<T NP 0 1> (<L N NNS NNS dogs N>) ) ) "
    "(<L S[dcl]\\NP VBD VBD barked S[dcl]\\NP_1>) )"
)
COMPOSED: Final = (
    "(<T S[dcl] 1 2> (<T NP 0 1> (<L N NNP NNP John N>) ) "
    "(<T S[dcl]\\NP 0 2> (<T (S[dcl]\\NP)/NP 0 2> "
    "(<T (S[dcl]\\NP)/NP 1 2> "
    "(<L (S\\NP)/(S\\NP) RB RB probably (S_1\\NP_2)_1/(S_1\\NP_2)_1>) "
    "(<L (S[dcl]\\NP)/NP VBD VBD saw (S[dcl]\\NP_3)/NP_4>) ) "
    "(<L (S\\NP)\\(S\\NP) RB RB yesterday (S_5\\NP_6)_5\\(S_5\\NP_6)_5>) ) "
    "(<T NP 0 1> (<L N NNP NNP Mary N>) ) ) )"
)
ALGEBRA: Final = MarkedCategoryAlgebra(("NP/N",))
PROJECTOR: Final = CCGDependencyProjector(MarkedCategoryAlgebra())


def parg(*lines: str, identifier: str = "synthetic.1", last: int = 5) -> str:
    return f'<s id="{identifier}"> {last}\n{"".join(lines)}<\\s>\n'


def leaf(category: str, word: str = "w", slotted: str = "") -> AutoLeaf:
    return AutoLeaf(category, "X", "X", word, slotted or category)


def rules(derivation: Derivation[str]) -> tuple[str | None, ...]:
    return tuple(
        node.rule
        for _, node in sorted(derivation.nodes.items())
        if node.rule is not None
    )


class TestMarkedCategoryAlgebra(CCGCategoryAlgebraConformanceTests[str]):
    @override
    def make_category_algebra_case(self) -> CategoryAlgebraCase[str]:
        return CategoryAlgebraCase(
            ALGEBRA,
            "N",
            "(S[dcl]\\NP)/NP",
            "S\\NP",
            "NP",
            "N_1/N_1",
            "(S[dcl]\\NP_2)/NP_3",
        )


@pytest.mark.parametrize("category", ["(S\\NP)", "((S\\NP))", "(S\\NP)_4"])
def test_a_wrapped_functor_is_still_a_functor(category: str) -> None:
    assert ALGEBRA.is_functor(category)
    assert ALGEBRA.result(category) == "S"
    assert ALGEBRA.argument(category) == "NP"


def test_functor_parts_keep_their_markup() -> None:
    auxiliary = "(S[dcl]\\NP_10)/(S[b]_11\\NP_10:B)_11"
    complement = ALGEBRA.argument(auxiliary)

    assert ALGEBRA.result(auxiliary) == "S[dcl]\\NP_10"
    assert complement == "(S[b]_11\\NP_10:B)_11"
    assert ALGEBRA.is_functor(complement)
    assert ALGEBRA.result(complement) == "S[b]_11"
    assert ALGEBRA.argument(complement) == "NP_10:B"


@pytest.mark.parametrize(
    ("category", "shares"),
    [
        ("(S_1\\NP_2)_1\\(S_1\\NP_2)_1", True),
        ("NP[nb]_29/N_29", True),
        ("(S[dcl]\\NP_10)/(S[b]_11\\NP_10:B)_11", False),
        ("(NP_1\\NP_1)/NP_2", False),
        ("(S[dcl]\\NP_1)/(S[b]\\NP_1)", False),
        ("NP_1\\NP_1", True),
        ("(NP[nb]_9/N_9)\\NP_8", False),
        ("(S\\NP)\\(S\\NP)", True),
        ("NP[nb]/N", True),
        ("(S\\NP)/NP", False),
        ("NP", False),
    ],
)
def test_head_sharing_reads_markup_then_declarations(
    category: str,
    *,
    shares: bool,
) -> None:
    assert ALGEBRA.shares_head(category) is shares


def test_an_undeclared_unmarked_determiner_does_not_share_its_head() -> None:
    assert not MarkedCategoryAlgebra().shares_head("NP[nb]/N")
    assert MarkedCategoryAlgebra().shares_head("NP[nb]_1/N_1")


@pytest.mark.parametrize(
    ("left", "right", "same"),
    [
        ("S[dcl]\\NP_8", "(S\\NP)", True),
        ("((S[b]\\NP)/PP)/NP", "(S\\NP)/PP/NP", True),
        ("NP[nb]", "NP", True),
        ("NP", "N", False),
        ("(S\\NP)/NP", "S\\(NP/NP)", False),
        ("S/NP", "S\\NP", False),
    ],
)
def test_equivalence_erases_features_and_markup(
    left: str,
    right: str,
    *,
    same: bool,
) -> None:
    assert ALGEBRA.equivalent(left, right) is same


@pytest.mark.parametrize("category", ["NP", "S[dcl]_3"])
def test_an_atom_has_no_functor_parts(category: str) -> None:
    with pytest.raises(ValueError, match="not a CCG functor"):
        ALGEBRA.result(category)

    with pytest.raises(ValueError, match="not a CCG functor"):
        ALGEBRA.argument(category)


def test_an_unbalanced_category_is_refused() -> None:
    with pytest.raises(ValueError, match="unbalanced"):
        ALGEBRA.is_functor("(S\\NP")


def test_the_algebra_is_an_immutable_value() -> None:
    assert MarkedCategoryAlgebra(("NP[nb]/N",)) == MarkedCategoryAlgebra(("NP/N",))
    assert hash(MarkedCategoryAlgebra(("NP/N",))) == hash(ALGEBRA)
    assert MarkedCategoryAlgebra() != ALGEBRA
    assert ALGEBRA != "algebra"
    assert ALGEBRA.head_sharing == frozenset({"(NP/N)"})
    assert repr(ALGEBRA) == "MarkedCategoryAlgebra(('(NP/N)',))"
    assert copy(ALGEBRA) is ALGEBRA
    assert deepcopy(ALGEBRA) is ALGEBRA

    with pytest.raises(AttributeError):
        ALGEBRA.__setattr__("_head_sharing", frozenset())


@pytest.mark.parametrize(
    ("parent", "left", "right", "head", "rule"),
    [
        ("NP", "NP[nb]/N", "N", 1, "forward-application"),
        ("S[dcl]", "NP", "S[dcl]\\NP", 1, "backward-application"),
        ("NP[conj]", "conj", "NP", 1, "conjunct"),
        ("NP", "NP", "NP[conj]", 0, "coordination"),
        ("NP", ",", "NP", 1, "punctuation-left"),
        ("NP", "NP", ".", 0, "punctuation-right"),
        ("S/NP", "S/(S\\NP)", "(S\\NP)/NP", 0, "forward-composition"),
        (
            "(S[dcl]\\NP)/NP",
            "(S[dcl]\\NP)/NP",
            "(S\\NP)\\(S\\NP)",
            0,
            "backward-composition",
        ),
        (
            "(S[dcl]\\NP)/(S[b]\\NP)",
            "(S[dcl]\\NP)/(S[b]\\NP)",
            "(S\\NP)\\(S\\NP)",
            0,
            "backward-composition",
        ),
        ("S/S", "NP", ",", 0, "binary-type-changing-left"),
        ("(S\\NP)\\(S\\NP)", ",", "NP", 1, "binary-type-changing-right"),
        ("S", "NP", "S/NP", 1, "binary-type-changing-right"),
        ("S/S", "NP/N", "N", 0, "binary-type-changing-left"),
        ("S", "NP", "S\\N", 1, "binary-type-changing-right"),
        ("S/NP", "S\\NP", "NP/NP", 0, "binary-type-changing-left"),
        ("S/NP", "S/NP", "NP\\NP", 0, "binary-type-changing-left"),
        (
            "(S/PP)/NP",
            "S/(S\\NP)",
            "((S\\NP)/PP)/NP",
            0,
            "forward-composition",
        ),
        (
            "(S/NP)/PP",
            "S/(S\\NP)",
            "((S\\NP)/PP)/NP",
            0,
            "binary-type-changing-left",
        ),
        (
            "((S\\NP)/PP)/NP",
            "((S\\NP)/PP)/NP",
            "(S\\NP)\\(S\\NP)",
            0,
            "backward-composition",
        ),
        (
            "((S\\NP)/NP)/PP",
            "((S\\NP)/PP)/NP",
            "(S\\NP)\\(S\\NP)",
            0,
            "binary-type-changing-left",
        ),
        ("S", "NP", "NP[conj]", 0, "binary-type-changing-left"),
        ("N", "conj", "N", 1, "binary-type-changing-right"),
    ],
)
def test_binary_steps_are_named_structurally(
    parent: str,
    left: str,
    right: str,
    head: int,
    rule: str,
) -> None:
    tree = AutoTree(parent, head, (leaf(left), leaf(right)))

    assert rules(gold_sentence(tree).derivation) == (rule,)
    assert rule in CCGBANK_HEAD_SELECTIONS


@pytest.mark.parametrize(
    ("parent", "child", "rule"),
    [
        ("NP", "N", "type-changing"),
        ("S/(S\\NP)", "NP", "type-raising"),
        ("(S\\NP)\\((S\\NP)/NP)", "NP", "type-raising"),
        ("S/(S/NP)", "NP", "type-changing"),
        ("S/(S\\N)", "NP", "type-changing"),
        ("NP\\(S\\NP)", "NP", "type-changing"),
        ("S/(N\\NP)", "NP", "type-changing"),
        ("S\\(N/NP)", "NP", "type-changing"),
    ],
)
def test_unary_steps_are_named_structurally(
    parent: str,
    child: str,
    rule: str,
) -> None:
    tree = AutoTree(parent, 0, (leaf(child),))

    assert rules(gold_sentence(tree).derivation) == (rule,)
    assert rule in CCGBANK_HEAD_SELECTIONS


def test_a_gold_sentence_has_marked_labels_and_unmarked_categories() -> None:
    (derivation,) = parse_auto(AUTO)
    prepared = gold_sentence(derivation.tree)
    lexical = tuple(
        node.label
        for _, node in sorted(prepared.derivation.nodes.items())
        if node.rule is None
    )

    assert tuple(token.surface for token in prepared.tokens) == (
        "the",
        "dog",
        "saw",
        "a",
        "cat",
        ".",
    )
    assert tuple(token.index for token in prepared.tokens) == tuple(range(6))
    assert prepared.categories[2] == "(S[dcl]\\NP)/NP"
    assert lexical[2] == "(S[dcl]\\NP_2)/NP_3"
    assert rules(prepared.derivation) == (
        "forward-application",
        "forward-application",
        "forward-application",
        "backward-application",
        "punctuation-right",
    )
    assert prepared.derivation.node(prepared.derivation.root).span.length == 6
    assert prepared.covering[0, 1] == "forward-application"
    assert prepared.covering[1, 0] == "forward-application"
    assert prepared.covering[1, 4] == "backward-application"
    assert prepared.covering[2, 4] == "forward-application"
    assert prepared.covering[5, 0] == "punctuation-right"


def test_a_conjunct_label_drops_its_marker() -> None:
    tree = AutoTree(
        "NP",
        0,
        (leaf("NP"), AutoTree("NP[conj]", 1, (leaf("conj"), leaf("NP")))),
    )
    prepared = gold_sentence(tree)
    labels = {node.rule: node.label for node in prepared.derivation.nodes.values()}

    assert labels["conjunct"] == "NP"
    assert labels["coordination"] == "NP"


def test_a_node_with_three_daughters_is_refused() -> None:
    tree = AutoTree("NP", 0, (leaf("NP"), leaf("NP"), leaf("NP")))

    with pytest.raises(ValueError, match="3 daughters"):
        gold_sentence(tree)


def test_hand_computed_arcs_score_as_multisets() -> None:
    first: LabeledArc = (1, 0, "N/N", 1)
    second: LabeledArc = (2, 1, "(S\\NP)/NP", 1)
    third: LabeledArc = (2, 3, "(S\\NP)/NP", 2)
    fourth: LabeledArc = (4, 3, "N/N", 1)
    reflexive: LabeledArc = (2, 2, "(S\\NP)/NP", 2)

    counts = score_arcs(
        (
            (first, "forward-application"),
            (second, "forward-application"),
            (second, "forward-application"),
            (third, "backward-application"),
            (reflexive, "forward-application"),
        ),
        (
            (first, "forward-application"),
            (second, "backward-application"),
            (fourth, "backward-composition"),
            (reflexive, "forward-application"),
        ),
    )

    assert (counts.matched, counts.predicted, counts.gold) == (2, 4, 3)
    assert counts.labeled_precision == 0.5
    assert counts.labeled_recall == 2 / 3
    assert counts.spurious == (
        ("backward-application", 1),
        ("forward-application", 1),
    )
    assert counts.missed == (("backward-composition", 1),)
    assert counts.unprojected == 0


def test_repeated_arcs_match_with_multiplicity() -> None:
    arc: LabeledArc = (2, 1, "S\\NP", 1)
    counts = score_arcs(((arc, "a"), (arc, "a"), (arc, "a")), ((arc, "b"), (arc, "b")))

    assert (counts.matched, counts.predicted, counts.gold) == (2, 3, 2)
    assert counts.spurious == (("a", 1),)
    assert counts.missed == ()


def test_rule_counts_are_ordered_largest_first() -> None:
    counts = RoundTripCounts(spurious=(("a", 1), ("c", 3), ("b", 1)))

    assert counts.spurious == (("c", 3), ("a", 1), ("b", 1))


def test_the_slot_and_category_are_part_of_a_labeled_arc() -> None:
    counts = score_arcs(
        (((2, 1, "S\\NP", 2), "a"), ((2, 3, "S/NP", 1), "b")),
        (((2, 1, "S\\NP", 1), "c"), ((2, 3, "S\\NP", 1), "d")),
    )

    assert counts.matched == 0
    assert counts.spurious == (("a", 1), ("b", 1))
    assert counts.missed == (("c", 1), ("d", 1))


def test_corpus_counts_are_micro_sums() -> None:
    small = RoundTripCounts(1, 1, 1)
    large = RoundTripCounts(
        2,
        8,
        4,
        spurious=(("forward-application", 6),),
        missed=(("forward-application", 1), ("coordination", 1)),
        unprojected=1,
    )
    total = small + large

    assert total == RoundTripCounts(
        3,
        9,
        5,
        spurious=(("forward-application", 6),),
        missed=(("coordination", 1), ("forward-application", 1)),
        unprojected=1,
    )
    assert total.labeled_precision == 3 / 9
    assert total.labeled_recall == 3 / 5
    assert total.missed == (("coordination", 1), ("forward-application", 1))
    assert (large + large).spurious == (("forward-application", 12),)
    assert (large + large).missed == (
        ("coordination", 2),
        ("forward-application", 2),
    )
    assert (large + large).unprojected == 2


def test_raised_arcs_are_summed_by_step() -> None:
    first = RoundTripCounts(gold=3, raised=(("a(lexical)", 3),), unprojected=1)
    second = RoundTripCounts(gold=2, raised=(("a(lexical)", 2),), unprojected=1)
    third = RoundTripCounts(gold=1, raised=(("b(lexical)", 1),), unprojected=1)
    total = first + second + third

    assert total.raised == (("a(lexical)", 5), ("b(lexical)", 1))
    assert total.unprojected == 3
    assert total.gold == 6


def test_a_rule_named_twice_has_its_counts_summed() -> None:
    counts = RoundTripCounts(
        spurious=(("a", 1), ("a", 2)),
        missed=(("b", 2), ("c", 1), ("b", 2)),
        raised=(("d", 1), ("d", 1)),
    )

    assert counts.spurious == (("a", 3),)
    assert counts.missed == (("b", 4), ("c", 1))
    assert counts.raised == (("d", 2),)


def test_empty_counts_have_zero_ratios() -> None:
    empty = RoundTripCounts()

    assert empty.labeled_precision == 0.0
    assert empty.labeled_recall == 0.0
    assert RoundTripCounts(0, 3, 0).labeled_precision == 0.0
    assert RoundTripCounts(0, 0, 3).labeled_recall == 0.0


@pytest.mark.parametrize(
    ("matched", "predicted", "gold", "unprojected", "message"),
    [
        (-1, 0, 0, 0, "negative"),
        (0, -1, 0, 0, "negative"),
        (0, 0, -1, 0, "negative"),
        (0, 0, 0, -1, "negative"),
        (2, 1, 3, 0, "exceed"),
        (2, 3, 1, 0, "exceed"),
    ],
)
def test_inconsistent_counts_are_refused(
    matched: int,
    predicted: int,
    gold: int,
    unprojected: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        RoundTripCounts(matched, predicted, gold, unprojected=unprojected)


def test_counts_are_immutable_values() -> None:
    counts = RoundTripCounts(1, 2, 3, spurious=(("a", 1),), unprojected=1)

    assert counts == RoundTripCounts(1, 2, 3, spurious=(("a", 1),), unprojected=1)
    assert counts != RoundTripCounts(1, 2, 3, spurious=(("a", 1),))
    assert counts != "counts"
    assert hash(counts) == hash(
        RoundTripCounts(1, 2, 3, spurious=(("a", 1),), unprojected=1)
    )
    assert repr(counts) == (
        "RoundTripCounts(1, 2, 3, spurious=(('a', 1),), missed=(), raised=(), "
        "unprojected=1)"
    )
    assert counts != RoundTripCounts(
        1, 2, 3, spurious=(("a", 1),), raised=(("b", 1),), unprojected=1
    )
    assert RoundTripCounts(spurious=(("a", 0),)).spurious == ()

    with pytest.raises(AttributeError):
        counts.__setattr__("_matched", 0)


def test_a_gold_derivation_round_trips_exactly() -> None:
    (derivation,) = parse_auto(AUTO)
    (sentence,) = parse_parg(parg(*PARG_LINES))
    counts = round_trip(
        derivation, sentence, CCGDependencyProjector(MarkedCategoryAlgebra())
    )

    assert counts == RoundTripCounts(4, 4, 4)
    assert counts.labeled_precision == 1.0
    assert counts.labeled_recall == 1.0


def test_round_trip_failures_name_their_rules() -> None:
    (derivation,) = parse_auto(AUTO)
    (sentence,) = parse_parg(
        parg(
            *PARG_LINES[:3],
            "0\t4\tN\t1\tthe cat\n",
            "5\t5\tN\t1\t. .\n",
        )
    )
    counts = round_trip(derivation, sentence, PROJECTOR)

    assert (counts.matched, counts.predicted, counts.gold) == (3, 4, 4)
    assert counts.spurious == (("forward-application", 1),)
    assert counts.missed == (("backward-application", 1),)


def test_a_gold_arc_beyond_the_derivation_is_unspanned() -> None:
    (derivation,) = parse_auto(AUTO)
    (sentence,) = parse_parg(
        parg(*PARG_LINES, "7\t6\tN\t1\tx y\n", last=7),
    )

    assert round_trip(derivation, sentence, PROJECTOR).missed == (("unspanned", 1),)


@pytest.mark.parametrize(
    ("auto", "lines"),
    [
        (
            COORDINATED,
            (
                "0\t3\tS[dcl]\\NP\t1\tcats barked\n",
                "2\t3\tS[dcl]\\NP\t1\tdogs barked\n",
            ),
        ),
        (
            TRAILING_COMMA,
            ("0\t2\tS[dcl]\\NP\t1\tdogs barked\n",),
        ),
        (
            LEADING_COMMA,
            ("1\t2\tS[dcl]\\NP\t1\tdogs barked\n",),
        ),
    ],
    ids=["coordination", "punctuation-right", "punctuation-left"],
)
def test_heads_pass_through_coordination_and_punctuation(
    auto: str,
    lines: tuple[str, ...],
) -> None:
    (derivation,) = parse_auto(f"ID=synthetic.1\n{auto}\n")
    (sentence,) = parse_parg(parg(*lines, last=auto.count("<L ") - 1))
    counts = round_trip(derivation, sentence, PROJECTOR)

    assert counts == RoundTripCounts(len(lines), len(lines), len(lines))


class RefusingProjector:
    def __init__(self, refused: str, error: type[Exception] = ValueError) -> None:
        self._refused = refused
        self._error = error

    def lexical(
        self,
        node: DerivationNode[str],
        token: Token,
        /,
    ) -> HeadProjection[ProjectedDependency[str], DecoratedCategory[str]]:
        if self._refused == "lexical":
            self._refuse()

        return PROJECTOR.lexical(node, token)

    def step(
        self,
        node: DerivationNode[str],
        rule_name: str,
        children: tuple[
            HeadProjection[ProjectedDependency[str], DecoratedCategory[str]], ...
        ],
        selection: HeadSelection,
        /,
    ) -> HeadProjection[ProjectedDependency[str], DecoratedCategory[str]]:
        if rule_name == self._refused:
            self._refuse()

        return PROJECTOR.step(node, rule_name, children, selection)

    def _refuse(self) -> Never:
        msg = "projection refused"
        raise self._error(msg)


@pytest.mark.parametrize(
    ("refused", "step"),
    [
        ("lexical", "lexical"),
        ("forward-application", "forward-application(lexical, lexical)"),
        (
            "backward-application",
            "backward-application(forward-application, forward-application)",
        ),
        ("punctuation-right", "punctuation-right(backward-application, lexical)"),
    ],
)
def test_a_sentence_whose_projection_raises_keeps_its_gold_arcs(
    refused: str,
    step: str,
) -> None:
    (derivation,) = parse_auto(AUTO)
    (sentence,) = parse_parg(parg(*PARG_LINES))
    counts = round_trip(derivation, sentence, RefusingProjector(refused))

    assert counts == RoundTripCounts(0, 0, 4, raised=((step, 4),), unprojected=1)
    assert counts.missed == ()
    assert counts.labeled_recall == 0.0


def test_a_raising_sentence_drops_its_reflexive_gold_arcs() -> None:
    (derivation,) = parse_auto(AUTO)
    (sentence,) = parse_parg(parg(*PARG_LINES, "5\t5\tN\t1\t. .\n"))
    counts = round_trip(derivation, sentence, RefusingProjector("lexical"))

    assert counts == RoundTripCounts(gold=4, raised=(("lexical", 4),), unprojected=1)


def test_a_projector_error_other_than_a_value_error_propagates() -> None:
    (derivation,) = parse_auto(AUTO)
    (sentence,) = parse_parg(parg(*PARG_LINES))

    with pytest.raises(TypeError, match="projection refused"):
        round_trip(derivation, sentence, RefusingProjector("lexical", TypeError))


def test_composition_steps_select_the_functor_as_head() -> None:
    (derivation,) = parse_auto(f"ID=synthetic.1\n{COMPOSED}\n")
    (sentence,) = parse_parg(
        parg(
            "2\t1\t(S\\NP)/(S\\NP)\t2\tsaw probably\n",
            "2\t3\t(S\\NP)\\(S\\NP)\t2\tsaw yesterday\n",
            "4\t2\t(S[dcl]\\NP)/NP\t2\tMary saw\n",
            "0\t2\t(S[dcl]\\NP)/NP\t1\tJohn saw\n",
            last=4,
        )
    )
    prepared = gold_sentence(derivation.tree)

    assert "forward-composition" in rules(prepared.derivation)
    assert "backward-composition" in rules(prepared.derivation)
    assert round_trip(derivation, sentence, PROJECTOR) == RoundTripCounts(4, 4, 4)


@pytest.mark.parametrize(
    ("parent", "daughters", "head"),
    [
        ("S/S", (leaf("NP"), leaf(",", ",")), 0),
        ("(S\\NP)\\(S\\NP)", (leaf(",", ","), leaf("NP")), 1),
    ],
    ids=["left", "right"],
)
def test_binary_type_changing_keeps_the_auto_head_daughter(
    parent: str,
    daughters: tuple[AutoLeaf, AutoLeaf],
    head: int,
) -> None:
    prepared = gold_sentence(AutoTree(parent, head, daughters))
    projection = project_dependencies(
        prepared.derivation, prepared.tokens, CCGBANK_HEAD_SELECTIONS, PROJECTOR
    )

    assert projection.heads == (head,)


def test_a_gold_arc_keeps_the_category_parg_states() -> None:
    (derivation,) = parse_auto(AUTO)
    (sentence,) = parse_parg(
        parg(*PARG_LINES[:2], "4\t2\t(S\\NP)/NP\t2\tcat saw\n", PARG_LINES[3])
    )
    counts = round_trip(derivation, sentence, PROJECTOR)

    assert (counts.matched, counts.predicted, counts.gold) == (3, 4, 4)
    assert counts.missed == (("forward-application", 1),)


def test_the_round_trip_projector_declares_the_ccgbank_configuration() -> None:
    projector = ccgbank_round_trip_projector()

    assert projector.algebra == MarkedCategoryAlgebra(("NP/N",))
    assert projector.convention is None
    (derivation,) = parse_auto(AUTO)
    (sentence,) = parse_parg(parg(*PARG_LINES))

    assert round_trip(derivation, sentence, projector) == RoundTripCounts(4, 4, 4)


def test_published_mappings_are_read_only() -> None:
    (derivation,) = parse_auto(AUTO)

    assert isinstance(CCGBANK_HEAD_SELECTIONS, MappingProxyType)
    assert isinstance(gold_sentence(derivation.tree).covering, MappingProxyType)


def test_a_derivation_scored_against_another_sentence_is_refused() -> None:
    (derivation,) = parse_auto(AUTO)
    (sentence,) = parse_parg(parg(*PARG_LINES, identifier="synthetic.2"))

    with pytest.raises(ValueError, match=r"synthetic\.2"):
        round_trip(derivation, sentence, PROJECTOR)


AUXILIARY: Final = (
    "(<T S[dcl] 1 2> (<T NP 0 1> (<L N NNP NNP Vinken N>) ) "
    "(<T S[dcl]\\NP 0 2> "
    "(<L (S[dcl]\\NP)/(S[b]\\NP) MD MD will (S[dcl]\\NP_1)/(S[b]_2\\NP_1:B)_2>) "
    "(<L S[b]\\NP VB VB join S[b]\\NP_3>) ) )"
)
ADJUNCT: Final = (
    "(<T S[dcl] 1 2> (<T NP 0 1> (<L N NNP NNP Kim N>) ) "
    "(<T S[dcl]\\NP 0 2> (<L S[dcl]\\NP VBD VBD left S[dcl]\\NP_1>) "
    "(<T (S\\NP)\\(S\\NP) 0 1> (<L S[ng]\\NP VBG VBG smiling S[ng]\\NP_2>) ) ) )"
)
REDUCED_RELATIVE: Final = (
    "(<T NP 0 2> (<T NP 0 1> (<L N NNS NNS dogs N>) ) "
    "(<T NP\\NP 0 1> (<L S[pss]\\NP VBN VBN seen S[pss]\\NP_1>) ) )"
)
CONTROL: Final = (
    "(<T S[dcl] 1 2> (<T NP 0 1> (<L N NNP NNP Kim N>) ) "
    "(<T S[dcl]\\NP 0 2> "
    "(<L (S[dcl]\\NP)/(S[to]\\NP) VBZ VBZ wants "
    "(S[dcl]\\NP_1)/(S[to]_2\\NP_1:B)_2>) "
    "(<T S[to]\\NP 0 2> "
    "(<L (S[to]\\NP)/(S[b]\\NP) TO TO to (S[to]\\NP_3)/(S[b]_4\\NP_3:B)_4>) "
    "(<L S[b]\\NP VB VB leave S[b]\\NP_5>) ) ) )"
)


@pytest.mark.parametrize(
    ("auto", "lines"),
    [
        (
            AUXILIARY,
            (
                "0\t1\t(S[dcl]\\NP)/(S[b]\\NP)\t1\tVinken will\n",
                "0\t2\tS[b]\\NP\t1\tVinken join\n",
                "2\t1\t(S[dcl]\\NP)/(S[b]\\NP)\t2\tjoin will\n",
            ),
        ),
        (
            ADJUNCT,
            (
                "0\t1\tS[dcl]\\NP\t1\tKim left\n",
                "0\t2\tS[ng]\\NP\t1\tKim smiling\n",
            ),
        ),
        (REDUCED_RELATIVE, ("0\t1\tS[pss]\\NP\t1\tdogs seen\n",)),
        (
            CONTROL,
            (
                "0\t1\t(S[dcl]\\NP)/(S[to]\\NP)\t1\tKim wants\n",
                "2\t1\t(S[dcl]\\NP)/(S[to]\\NP)\t2\tto wants\n",
                "3\t2\t(S[to]\\NP)/(S[b]\\NP)\t2\tleave to\n",
                "0\t3\tS[b]\\NP\t1\tKim leave\n",
            ),
        ),
    ],
    ids=["auxiliary", "type-changed-adjunct", "reduced-relative", "control"],
)
def test_marked_categories_round_trip_their_nonlocal_dependencies(
    auto: str,
    lines: tuple[str, ...],
) -> None:
    (derivation,) = parse_auto(f"ID=synthetic.1\n{auto}\n")
    (sentence,) = parse_parg(parg(*lines, last=auto.count("<L ") - 1))

    assert round_trip(
        derivation, sentence, ccgbank_round_trip_projector()
    ) == RoundTripCounts(len(lines), len(lines), len(lines))


def test_the_infinitival_subject_is_projected_but_left_unevaluated() -> None:
    (derivation,) = parse_auto(f"ID=synthetic.1\n{CONTROL}\n")
    prepared = gold_sentence(derivation.tree)
    projection = project_dependencies(
        prepared.derivation,
        prepared.tokens,
        CCGBANK_HEAD_SELECTIONS,
        ccgbank_round_trip_projector(),
    )
    arcs = {
        (dependency.predicate_form, dependency.slot, dependency.argument_form)
        for dependency in projection.dependencies
    }

    assert ("to", 1, "Kim") in arcs
    assert not evaluated((2, 0, "(S[to]\\NP)/(S[b]\\NP)", 1), ("Kim", "wants", "to"))


@pytest.mark.parametrize(
    ("arc", "words", "scored"),
    [
        ((2, 0, "(S[to]\\NP)/(S[b]\\NP)", 1), ("Kim", "x", "to"), False),
        ((2, 3, "(S[to]\\NP)/(S[b]\\NP)", 2), ("Kim", "x", "to", "go"), True),
        ((1, 0, "(S[b]\\NP)/(S[pss]\\NP)", 1), ("it", "be"), False),
        ((1, 0, "(S[b]\\NP)/(S[pss]\\NP)", 1), ("it", "Be"), True),
        ((1, 0, "(S[b]\\NP)/(S[pss]\\NP)", 2), ("it", "be"), True),
        ((1, 0, "(S[b]\\NP)/(S[b]\\NP)", 1), ("it", "help"), True),
        ((1, 0, "(S[b]\\NP)/NP", 1), ("there", "be"), False),
        ((1, 0, "(S[b]\\NP)/NP", 1), ("it", "be"), True),
        ((1, 0, "conj", 1), ("x", "and"), False),
        ((1, 7, "(S[dcl]\\NP)/NP", 1), ("x", "saw"), True),
    ],
)
def test_only_the_declared_dependencies_are_left_unevaluated(
    arc: LabeledArc,
    words: tuple[str, ...],
    *,
    scored: bool,
) -> None:
    assert evaluated(arc, words) is scored


def test_unevaluated_arcs_leave_the_predictions_but_not_the_gold() -> None:
    (derivation,) = parse_auto(f"ID=synthetic.1\n{CONTROL}\n")
    lines = (
        "0\t1\t(S[dcl]\\NP)/(S[to]\\NP)\t1\tKim wants\n",
        "2\t1\t(S[dcl]\\NP)/(S[to]\\NP)\t2\tto wants\n",
        "3\t2\t(S[to]\\NP)/(S[b]\\NP)\t2\tleave to\n",
        "0\t3\tS[b]\\NP\t1\tKim leave\n",
        "0\t2\t(S[to]\\NP)/(S[b]\\NP)\t1\tKim to\n",
    )
    (sentence,) = parse_parg(parg(*lines, last=3))

    assert round_trip(
        derivation, sentence, ccgbank_round_trip_projector()
    ) == RoundTripCounts(4, 4, 5, missed=(("backward-application", 1),))


@pytest.mark.parametrize(
    ("category", "expected"),
    [
        (
            "(S[dcl]\\NP_10)/(S[b]_11\\NP_10:B)_11",
            (
                (("argument", "argument"), 10),
                (("result", "argument"), 10),
            ),
        ),
        (
            "((S_1\\NP_2)_1\\(S_1\\NP_2)_1)/NP_3",
            (
                (("result", "argument", "argument"), 2),
                (("result", "result", "argument"), 2),
            ),
        ),
        ("(S[dcl]\\NP_2)/NP_3", ()),
        ("(S[dcl]\\NP)/(S[b]\\NP)", ()),
        ("N", ()),
    ],
)
def test_coindices_are_indices_shared_by_argument_nodes(
    category: str,
    expected: tuple[tuple[tuple[str, ...], int], ...],
) -> None:
    assert (
        tuple((binding.path, binding.index) for binding in ALGEBRA.coindices(category))
        == expected
    )


def test_a_relabeled_state_keeps_the_head_passing_of_its_marked_category() -> None:
    marked = "(S[dcl]\\NP_1)/(S[b]_2\\NP_1:B)_2"
    unmarked = "(S[dcl]\\NP)/(S[b]\\NP)"
    auxiliary = PROJECTOR.lexical(
        DerivationNode(TokenSpan(0, 1), marked), Token(0, "will")
    )
    relabeled = PROJECTOR.step(
        DerivationNode(TokenSpan(0, 1), unmarked, rule="unary", children=(0,)),
        "unary",
        (auxiliary,),
        HeadSelection("pass-through", 0),
    )
    complement = PROJECTOR.lexical(
        DerivationNode(TokenSpan(1, 2), "S[b]\\NP_3"), Token(1, "join")
    )
    applied = PROJECTOR.step(
        DerivationNode(TokenSpan(0, 2), "S[dcl]\\NP", rule="fa", children=(0, 1)),
        "fa",
        (relabeled, complement),
        HeadSelection("functor", 0),
    )

    assert ALGEBRA.shares_head(unmarked)
    assert relabeled.state is not None
    assert not relabeled.state.shares_head
    assert applied.heads == (0,)


COORDINATED_ADVERBS: Final = (
    "(<T S[dcl] 1 2> (<T NP 0 1> (<L N NNP NNP Kim N>) ) "
    "(<T S[dcl]\\NP 0 2> (<L S[dcl]\\NP VBD VBD left S[dcl]\\NP_1>) "
    "(<T (S\\NP)\\(S\\NP) 0 2> "
    "(<L (S\\NP)\\(S\\NP) RB RB quickly (S_2\\NP_3)_2\\(S_2\\NP_3)_2>) "
    "(<T (S\\NP)\\(S\\NP)[conj] 1 2> (<L conj CC CC and conj>) "
    "(<L (S\\NP)\\(S\\NP) RB RB quietly (S_4\\NP_5)_4\\(S_4\\NP_5)_4>) ) ) ) )"
)


def test_coordinated_modifiers_pass_the_head_they_modify() -> None:
    (derivation,) = parse_auto(f"ID=synthetic.1\n{COORDINATED_ADVERBS}\n")
    (sentence,) = parse_parg(
        parg(
            "0\t1\tS[dcl]\\NP\t1\tKim left\n",
            "1\t2\t(S\\NP)\\(S\\NP)\t2\tleft quickly\n",
            "1\t4\t(S\\NP)\\(S\\NP)\t2\tleft quietly\n",
            last=4,
        )
    )
    prepared = gold_sentence(derivation.tree)
    projector = ccgbank_round_trip_projector()
    projection = project_dependencies(
        prepared.derivation, prepared.tokens, CCGBANK_HEAD_SELECTIONS, projector
    )

    assert projection.heads == (1,)
    assert round_trip(derivation, sentence, projector) == RoundTripCounts(3, 3, 3)
