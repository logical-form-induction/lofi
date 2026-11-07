from math import log

import pytest

from lofi.deduce import Derivation, Edge, ItemKey
from lofi.interpret._render import BraceString, ProofFigure

TOKENS = ("the", "dog", "barks", "loudly")
TAGS = ("DT", "NN", "VBZ", "RB")
SYMBOLS = {"fa": ">", "ba": "<"}
HEADS = {"fa": 0, "ba": 1}


def key(span: tuple[int, int], category: str) -> ItemKey[str]:
    return ItemKey(span, category)


def leaf(span: tuple[int, int], category: str) -> Derivation[str, float]:
    return Derivation(key(span, category), Edge("lex", (), category, log(0.5)))


def step(
    span: tuple[int, int],
    category: str,
    rule: str,
    children: tuple[Derivation[str, float], ...],
) -> Derivation[str, float]:
    return Derivation(
        key(span, category),
        Edge(rule, tuple(child.key for child in children), category, log(0.8)),
        children,
    )


def sentence() -> Derivation[str, float]:
    subject = step((0, 2), "NP", "fa", (leaf((0, 1), "NP/N"), leaf((1, 2), "N")))
    predicate = step(
        (2, 4),
        "S\\NP",
        "ba",
        (leaf((2, 3), "S\\NP"), leaf((3, 4), "(S\\NP)\\(S\\NP)")),
    )

    return step((0, 4), "S", "ba", (subject, predicate))


FIGURE = (
    "the  dog  barks    loudly\n"
    "NP/N  N   S\\NP  (S\\NP)\\(S\\NP)\n"
    "--------> -------------------<\n"
    "   NP            S\\NP\n"
    "-----------------------------<\n"
    "              S"
)

BRACES = (
    "(<T S 1 2> (<T NP 0 2> (<L NP/N DT DT the NP/N>) (<L N NN NN dog N>) ) "
    "(<T S\\NP 1 2> (<L S\\NP VBZ VBZ barks S\\NP>) "
    "(<L (S\\NP)\\(S\\NP) RB RB loudly (S\\NP)\\(S\\NP)>) ) )"
)


def test_the_proof_figure_draws_the_tokens_over_the_categories() -> None:
    figure = ProofFigure(TOKENS, str, SYMBOLS)

    assert figure.render(sentence()) == FIGURE


def test_every_line_of_a_figure_keeps_its_own_columns() -> None:
    lines = ProofFigure(TOKENS, str, SYMBOLS).render(sentence()).split("\n")

    # sibling steps occupy separate runs on the same row.
    assert lines[2].count(" ") == 1
    assert lines[2].endswith("<")
    assert lines[2].index(">") + 2 == lines[2].index("-", lines[2].index(">"))
    assert all(line == line.rstrip() for line in lines)


def test_a_wide_category_widens_the_column_beneath_it() -> None:
    narrow = ProofFigure(("a", "b"), str, SYMBOLS)
    tree = step(
        (0, 2), "S[dcl]\\NP[expl]", "fa", (leaf((0, 1), "X"), leaf((1, 2), "Y"))
    )

    lines = narrow.render(tree).split("\n")

    assert lines[-1].strip() == "S[dcl]\\NP[expl]"
    assert len(lines[2]) >= len("S[dcl]\\NP[expl]")
    assert lines[0].split() == ["a", "b"]


def test_a_rule_the_symbol_table_does_not_name_writes_its_own_name() -> None:
    figure = ProofFigure(TOKENS, str)

    assert figure.render(sentence()).split("\n")[2].endswith("ba")


def test_a_lexical_axiom_renders_as_two_lines() -> None:
    figure = ProofFigure(TOKENS, str, SYMBOLS)

    assert figure.render(leaf((1, 2), "N")) == "dog\n N"


def test_a_figure_refuses_a_derivation_over_tokens_it_was_not_given() -> None:
    figure = ProofFigure(("the", "dog"), str, SYMBOLS)

    with pytest.raises(ValueError, match="covers tokens up to 4"):
        figure.render(sentence())


def test_the_figure_names_itself_and_reports_what_it_holds() -> None:
    figure = ProofFigure(TOKENS, str, SYMBOLS)

    assert figure.name == "proof-figure"
    assert repr(figure) == "ProofFigure(tokens=4, symbols=2)"


def test_the_brace_string_writes_one_node_per_pair_of_braces() -> None:
    braces = BraceString(TOKENS, TAGS, str, HEADS)

    assert braces.render(sentence()) == BRACES


def test_the_brace_string_counts_the_premises_of_every_step() -> None:
    rendered = BraceString(TOKENS, TAGS, str, HEADS).render(sentence())

    assert rendered.count("(<T ") == 3
    assert rendered.count("(<L ") == 4
    assert rendered.startswith("(<T S 1 2>")


def test_a_unary_step_heads_its_only_premise_without_a_table() -> None:
    braces = BraceString(("dog",), ("NN",), str, {})
    raised = step((0, 1), "NP", "tr", (leaf((0, 1), "N"),))

    assert braces.render(raised) == "(<T NP 0 1> (<L N NN NN dog N>) )"


def test_a_binary_rule_the_head_table_does_not_name_is_refused() -> None:
    braces = BraceString(TOKENS, TAGS, str, {"fa": 0})

    with pytest.raises(ValueError, match="names no premise of 'ba'"):
        braces.render(sentence())


def test_a_head_the_rule_does_not_have_is_refused() -> None:
    braces = BraceString(TOKENS, TAGS, str, {"fa": 0, "ba": 5})

    with pytest.raises(ValueError, match="premise 5 of 'ba'"):
        braces.render(sentence())


def test_a_brace_string_refuses_a_derivation_over_tokens_it_lacks() -> None:
    braces = BraceString(("the", "dog"), ("DT", "NN"), str, HEADS)

    with pytest.raises(ValueError, match="covers tokens up to 4"):
        braces.render(sentence())


def test_a_brace_string_requires_one_tag_for_each_token() -> None:
    with pytest.raises(ValueError, match="4 tokens cannot align with 2 tags"):
        BraceString(TOKENS, ("DT", "NN"), str, HEADS)


def test_the_brace_string_names_itself_and_reports_what_it_holds() -> None:
    braces = BraceString(TOKENS, TAGS, str, HEADS)

    assert braces.name == "brace-string"
    assert repr(braces) == "BraceString(tokens=4, heads=2)"


def test_a_style_spells_a_term_the_way_it_was_told_to() -> None:
    figure = ProofFigure(("dog",), lambda term: term.lower(), SYMBOLS)

    assert figure.render(leaf((0, 1), "NP")) == "dog\nnp"


def test_two_nodes_of_one_shape_stay_apart_in_the_figure_it_lays_out() -> None:
    # structurally equal premises remain distinct nodes.
    left = leaf((0, 1), "N")
    right = leaf((1, 2), "N")
    tree = step((0, 2), "N", "fa", (left, step((1, 2), "N", "tr", (right,))))

    assert hash(leaf((0, 1), "N")) != hash(left)
    assert (
        ProofFigure(("a", "b"), str, SYMBOLS).render(tree)
        == "a b\nN N\n  -tr\n  N\n--->\n N"
    )
