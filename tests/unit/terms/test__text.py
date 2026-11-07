import json
import re
from itertools import product
from sys import getrecursionlimit, hash_info
from typing import TYPE_CHECKING, Final, cast, override

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from tests.support.term_reference import categories, formulas, infix_syntax
from tests.support.term_strategies import terms

from lofi.terms import (
    AngleSyntax,
    BindingSite,
    BoundRecord,
    ConstantRecord,
    ConstructorNotation,
    ConstructorSyntax,
    FreeRecord,
    NotationReader,
    NotationTable,
    Operator,
    OperatorRecord,
    SExpSyntax,
    Signature,
    SortError,
    Term,
    TermStore,
    TextCursor,
    TextDepthError,
    TextParseError,
    UnknownConstantError,
    UnknownOperatorError,
    Vocabulary,
    from_record,
    record_differences,
    record_from_json,
    record_to_json,
    to_record,
)


def test_declared_constructor_syntax_obeys_each_notation_and_precedence() -> None:
    signature = Signature(
        "declared-category-text",
        ("Cat",),
        (
            Operator("product", "Cat", arg_sorts=("Cat", "Cat")),
            Operator("arrow", "Cat", arg_sorts=("Cat", "Cat")),
            Operator("box", "Cat", arg_sorts=("Cat",)),
            Operator("conditional", "Cat", arg_sorts=("Cat", "Cat", "Cat")),
        ),
        vocabularies=(Vocabulary("Cat", ("A", "B", "C")),),
    )
    store = TermStore(signature)
    syntax = ConstructorSyntax(
        signature,
        (
            ConstructorNotation.infix(
                "product", "*", precedence=20, associativity="left"
            ),
            ConstructorNotation.infix(
                "arrow", "->", precedence=10, associativity="right"
            ),
            ConstructorNotation.prefix("box", "box"),
            ConstructorNotation.mixfix(
                "conditional",
                ("if ", " then ", " else ", ""),
                precedence=5,
            ),
        ),
    )

    for text in ("box(A*B)->C->A", "if A then B else C", "(A->B)*C"):
        term = syntax.parse(store, text)
        written = syntax.render(store, term)

        assert syntax.parse(store, written) is term

    arrow = signature.by_name("arrow")
    nested = syntax.parse(store, "A->B->C")
    assert nested.op is arrow
    assert nested.args[1].op is arrow


if TYPE_CHECKING:
    from lofi._types import JsonValue

type _Tree = str | tuple[str, tuple[_Tree, ...]]


class _TreeOperands:
    """Operands that read identifiers as leaves and build labeled tuples."""

    def atom(self, cursor: TextCursor, context: str, /) -> _Tree:
        start = cursor.offset
        name = cursor.name()

        if name is None:
            cursor.fail(context, start, "expected a leaf")

        return name

    def build(
        self,
        cursor: TextCursor,
        notation: ConstructorNotation,
        arguments: tuple[_Tree, ...],
        start: int,
        /,
    ) -> _Tree:
        del cursor, start
        return (notation.name, arguments)

    def decorate(self, cursor: TextCursor, node: _Tree, /) -> _Tree:
        del cursor
        return node

    def label(self, node: _Tree, /) -> str:
        return node if isinstance(node, str) else node[0]

    def arity(self, notation: ConstructorNotation, /) -> int:
        del notation
        return 2


_TREE_NOTATIONS: Final = NotationTable(
    (
        ConstructorNotation.infix("to", "to", precedence=1, associativity="right"),
        ConstructorNotation.infix("and", "&", precedence=2, associativity="none"),
        ConstructorNotation.prefix("pair", "pair"),
        ConstructorNotation.mixfix("box", ("<<", ">>"), precedence=3),
    )
)


def _tree(text: str, *, word_boundaries: bool = True) -> _Tree:
    cursor = TextCursor(text, word_boundaries=word_boundaries)
    return NotationReader(cursor, _TREE_NOTATIONS, _TreeOperands()).complete()


def test_one_notation_reader_serves_any_operand_builder() -> None:
    assert _tree("a to b to c") == ("to", ("a", ("to", ("b", "c"))))
    assert _tree("pair(a, b & c)") == ("pair", ("a", ("and", ("b", "c"))))
    assert _tree("<< (a to b) >> & toe") == (
        "and",
        (("box", (("to", ("a", "b")),)), "toe"),
    )


def test_a_word_token_does_not_match_inside_a_longer_identifier() -> None:
    with pytest.raises(TextParseError, match="found trailing input") as caught:
        _tree("a tow")

    assert caught.value.offset == 2


def test_without_word_boundaries_a_word_token_delimits_its_operands() -> None:
    cursor = TextCursor("a tob", word_boundaries=False)
    reader = NotationReader(cursor, _TREE_NOTATIONS, _TreeOperands())

    assert reader.expression() == ("to", ("a", "b"))


def test_a_chained_nonassociative_notation_is_refused_at_its_offset() -> None:
    with pytest.raises(TextParseError, match="nonassociative") as caught:
        _tree("a & b & c")

    assert caught.value.operator == "and"


def test_a_reader_stops_at_unread_input_without_completing() -> None:
    cursor = TextCursor("a to b . rest", word_boundaries=True)
    held = NotationReader(cursor, _TREE_NOTATIONS, _TreeOperands()).expression()

    assert held == ("to", ("a", "b"))
    assert cursor.accept(".") is not None


FORMULA_SIGNATURE: Final = formulas().with_vocabulary(
    Vocabulary("Lf", ("sleep", "crossing-name", "f", "new-name", "deep", "edge"))
)
FORMULA_STORE: Final = TermStore(FORMULA_SIGNATURE)
FORMULA_SYNTAX: Final = SExpSyntax(FORMULA_SIGNATURE)
FORMULA_SIZE: Final = 16


REFERENCE_SEXP_CASES: Final = (
    '("lam" ("and" ("#bvar[Lf]" 0) ("#bvar[Lf]" 1)))',
    (
        '("lam" ("lam" ("app" ("app" ("#const[Lf]" "f") ("#bvar[Lf]" 1)) ("#bvar[Lf]" '
        "0))))"
    ),
    (
        '("forall" ("implies" ("app" ("#const[Lf]" "p") ("#bvar[Lf]" 0)) '
        '("app" ("#const[Lf]" "q") ("#bvar[Lf]" 0))))'
    ),
    (
        '("forall" ("exists" ("and" ("app" ("app" ("#const[Lf]" "p") '
        '("#bvar[Lf]" 1)) ("#bvar[Lf]" 0)) ("app" ("app" ("#const[Lf]" "q") '
        '("#bvar[Lf]" 1)) ("#bvar[Lf]" 0)))))'
    ),
    (
        '("lam" ("forall" ("implies" ("app" ("app" ("#const[Lf]" "f") '
        '("#bvar[Lf]" 0)) ("#bvar[Lf]" 1)) ("and" ("app" ("#const[Lf]" "p") '
        '("#bvar[Lf]" 0)) ("exists" ("app" ("#const[Lf]" "q") ("#bvar[Lf]" 0)))))))'
    ),
    (
        '("app" ("app" ("app" ("#const[Lf]" "f") ("#const[Lf]" "TRUE")) ("#const[Lf]" '
        '"FALSE")) ("#bvar[Lf]" 0))'
    ),
    (
        '("implies" ("and" ("app" ("app" ("#const[Lf]" "f") '
        '("app" ("#const[Lf]" "g") ("#bvar[Lf]" 0))) ("app" ("#const[Lf]" "h") '
        '("#bvar[Lf]" 1))) ("app" ("#const[Lf]" "p") ("#bvar[Lf]" 0))) '
        '("app" ("#const[Lf]" "q") ("#bvar[Lf]" 1)))'
    ),
    (
        '("lam" ("lam" ("lam" ("and" ("app" ("app" ("app" '
        '("#const[Lf]" "f") ("#bvar[Lf]" 2)) ("#bvar[Lf]" 1)) ("#bvar[Lf]" 0)) '
        '("implies" ("app" ("app" ("#const[Lf]" "g") ("#bvar[Lf]" 1)) '
        '("#bvar[Lf]" 0)) ("app" ("#const[Lf]" "h") ("#bvar[Lf]" 2)))))))'
    ),
    (
        '("forall" ("exists" ("forall" ("implies" ("app" ("app" '
        '("app" ("#const[Lf]" "p") ("#bvar[Lf]" 2)) ("#bvar[Lf]" 1)) ("#bvar[Lf]" 0)) '
        '("and" ("app" ("app" ("#const[Lf]" "q") ("#bvar[Lf]" 1)) ("#bvar[Lf]" 0)) '
        '("app" ("#const[Lf]" "r") ("#bvar[Lf]" 2)))))))'
    ),
    (
        '("lam" ("forall" ("implies" ("app" ("app" ("#const[Lf]" "f") '
        '("#bvar[Lf]" 1)) ("#bvar[Lf]" 0)) ("exists" ("and" ("app" ("app" '
        '("app" ("#const[Lf]" "g") ("#bvar[Lf]" 2)) ("#bvar[Lf]" 1)) ("#bvar[Lf]" 0)) '
        '("or" ("app" ("app" ("#const[Lf]" "h") ("#const[Lf]" "TRUE")) '
        '("#bvar[Lf]" 1)) ("not" ("app" ("#const[Lf]" "p") ("#bvar[Lf]" 0)))))))))'
    ),
    (
        '("lam" ("lam" ("lam" ("lam" ("lam" ("app" ("app" '
        '("app" ("app" ("app" ("#const[Lf]" "f") ("#bvar[Lf]" 4)) ("#bvar[Lf]" 3)) '
        '("#bvar[Lf]" 2)) ("#bvar[Lf]" 1)) ("#bvar[Lf]" 0)))))))'
    ),
    (
        '("implies" ("or" ("and" ("app" ("#const[Lf]" "p") ("#bvar[Lf]" 0)) '
        '("app" ("#const[Lf]" "q") ("#bvar[Lf]" 0))) ("and" ("app" '
        '("#const[Lf]" "r") ("#bvar[Lf]" 0)) ("app" ("#const[Lf]" "s") ("#bvar[Lf]" '
        "0)))) "
        '("iff" ("not" ("app" ("#const[Lf]" "t") ("#bvar[Lf]" 0))) '
        '("app" ("#const[Lf]" "u") ("#bvar[Lf]" 0))))'
    ),
    (
        '("forall" ("exists" ("forall" ("exists" ("iff" ("app" ("app" '
        '("app" ("app" ("#const[Lf]" "p") ("#bvar[Lf]" 3)) ("#bvar[Lf]" 2)) '
        '("#bvar[Lf]" 1)) ("#bvar[Lf]" 0)) ("and" ("app" ("app" ("#const[Lf]" "q") '
        '("#bvar[Lf]" 2)) ("#bvar[Lf]" 1)) ("app" ("app" ("#const[Lf]" "r") '
        '("#bvar[Lf]" 3)) ("#bvar[Lf]" 0))))))))'
    ),
    (
        '("lam" ("lam" ("forall" ("or" ("not" ("app" ("app" '
        '("app" ("#const[Lf]" "p") ("#bvar[Lf]" 2)) ("#bvar[Lf]" 1)) ("#bvar[Lf]" 0))) '
        '("exists" ("implies" ("app" ("app" ("#const[Lf]" "q") '
        '("#bvar[Lf]" 0)) ("#bvar[Lf]" 1)) ("app" ("app" ("#const[Lf]" "r") '
        '("#bvar[Lf]" 3)) ("#bvar[Lf]" 2))))))))'
    ),
    (
        '("app" ("#const[Lf]" "f") ("app" ("#const[Lf]" "g") ("app" '
        '("#const[Lf]" "h") ("app" ("#const[Lf]" "i") ("app" ("#const[Lf]" "j") '
        '("app" ("#const[Lf]" "k") ("#bvar[Lf]" 0)))))))'
    ),
    (
        '("app" ("app" ("app" ("#const[Lf]" "f") ("app" ("#const[Lf]" "g") '
        '("#bvar[Lf]" 0))) ("app" ("#const[Lf]" "h") ("app" ("#const[Lf]" "i") '
        '("#bvar[Lf]" 1)))) ("app" ("#const[Lf]" "j") ("app" ("app" '
        '("#const[Lf]" "k") ("#bvar[Lf]" 2)) ("#bvar[Lf]" 3))))'
    ),
    (
        '("lam" ("iff" ("forall" ("app" ("app" ("#const[Lf]" "p") '
        '("#bvar[Lf]" 1)) ("#bvar[Lf]" 0))) ("exists" ("app" ("app" '
        '("#const[Lf]" "q") ("#bvar[Lf]" 1)) ("#bvar[Lf]" 0)))))'
    ),
    (
        '("implies" ("app" ("#const[Lf]" "p") ("#bvar[Lf]" 0)) '
        '("implies" ("app" ("#const[Lf]" "q") ("#bvar[Lf]" 0)) '
        '("implies" ("app" ("#const[Lf]" "r") ("#bvar[Lf]" 0)) '
        '("implies" ("app" ("#const[Lf]" "s") ("#bvar[Lf]" 0)) '
        '("app" ("#const[Lf]" "t") ("#bvar[Lf]" 0))))))'
    ),
    (
        '("forall" ("exists" ("implies" ("iff" ("app" ("app" '
        '("#const[Lf]" "p") ("#bvar[Lf]" 1)) ("#bvar[Lf]" 0)) ("app" ("app" '
        '("#const[Lf]" "q") ("#bvar[Lf]" 1)) ("#bvar[Lf]" 0))) ("iff" ("app" '
        '("#const[Lf]" "r") ("#bvar[Lf]" 1)) ("app" ("#const[Lf]" "s") '
        '("#bvar[Lf]" 0))))))'
    ),
    (
        '("lam" ("forall" ("lam" ("exists" ("lam" ("forall" ("app" '
        '("app" ("app" ("app" ("app" ("app" ("#const[Lf]" "p") '
        '("#bvar[Lf]" 5)) ("#bvar[Lf]" 4)) ("#bvar[Lf]" 3)) ("#bvar[Lf]" 2)) '
        '("#bvar[Lf]" 1)) '
        '("#bvar[Lf]" 0))))))))'
    ),
    (
        '("forall" ("forall" ("implies" ("and" ("app" ("app" '
        '("#const[Lf]" "friend") ("#bvar[Lf]" 1)) ("#bvar[Lf]" 0)) ("app" '
        '("#const[Lf]" "happy") ("#bvar[Lf]" 1))) ("app" ("#const[Lf]" "happy") '
        '("#bvar[Lf]" 0)))))'
    ),
    (
        '("lam" ("lam" ("forall" ("implies" ("app" ("#bvar[Lf]" 1) '
        '("#bvar[Lf]" 0)) ("exists" ("app" ("app" ("#bvar[Lf]" 2) '
        '("#bvar[Lf]" 0)) ("#bvar[Lf]" 1)))))))'
    ),
    (
        '("implies" ("and" ("or" ("app" ("#const[Lf]" "p") '
        '("#bvar[Lf]" 0)) ("app" ("#const[Lf]" "q") ("#bvar[Lf]" 0))) ("or" '
        '("app" ("#const[Lf]" "r") ("#bvar[Lf]" 0)) ("app" ("#const[Lf]" "s") '
        '("#bvar[Lf]" 0)))) ("or" ("and" ("app" ("#const[Lf]" "t") '
        '("#bvar[Lf]" 0)) ("app" ("#const[Lf]" "u") ("#bvar[Lf]" 0))) ("and" '
        '("app" ("#const[Lf]" "v") ("#bvar[Lf]" 0)) ("app" ("#const[Lf]" "w") '
        '("#bvar[Lf]" 0)))))'
    ),
    (
        '("lam" ("lam" ("lam" ("app" ("app" ("app" '
        '("#const[Lf]" "compose") ("#bvar[Lf]" 2)) ("#bvar[Lf]" 1)) ("#bvar[Lf]" 0)))))'
    ),
    (
        '("forall" ("exists" ("and" ("iff" ("app" ("#const[Lf]" "f") '
        '("app" ("#const[Lf]" "g") ("#bvar[Lf]" 1))) ("app" ("#const[Lf]" "h") '
        '("app" ("#const[Lf]" "i") ("#bvar[Lf]" 0)))) ("implies" ("app" '
        '("#const[Lf]" "j") ("app" ("app" ("#const[Lf]" "k") ("#bvar[Lf]" 1)) '
        '("#bvar[Lf]" 0))) ("app" ("#const[Lf]" "l") ("app" ("#const[Lf]" "m") '
        '("app" ("#const[Lf]" "n") ("#bvar[Lf]" 0))))))))'
    ),
    (
        '("forall" ("forall" ("forall" ("implies" ("app" ("app" '
        '("#const[Lf]" "equal") ("#bvar[Lf]" 2)) ("#bvar[Lf]" 1)) ("implies" '
        '("app" ("app" ("#const[Lf]" "equal") ("#bvar[Lf]" 1)) ("#bvar[Lf]" 0)) '
        '("app" ("app" ("#const[Lf]" "equal") ("#bvar[Lf]" 2)) '
        '("#bvar[Lf]" 0)))))))'
    ),
    (
        '("lam" ("lam" ("implies" ("implies" ("implies" '
        '("#bvar[Lf]" 0) ("#bvar[Lf]" 1)) ("#bvar[Lf]" 0)) ("#bvar[Lf]" 0))))'
    ),
    (
        '("lam" ("lam" ("app" ("#bvar[Lf]" 0) ("app" ("#bvar[Lf]" 0) '
        '("app" ("#bvar[Lf]" 0) ("app" ("#bvar[Lf]" 0) ("#bvar[Lf]" 1)))))))'
    ),
    (
        '("lam" ("app" ("lam" ("app" ("#bvar[Lf]" 0) ("app" '
        '("#bvar[Lf]" 1) ("#bvar[Lf]" 1)))) ("lam" ("app" ("#bvar[Lf]" 0) '
        '("app" ("#bvar[Lf]" 1) ("#bvar[Lf]" 1))))))'
    ),
    (
        '("lam" ("forall" ("exists" ("iff" ("app" ("app" '
        '("#const[Lf]" "f") ("#bvar[Lf]" 2)) ("app" ("app" ("#const[Lf]" "g") '
        '("#bvar[Lf]" 1)) ("app" ("#const[Lf]" "h") ("#bvar[Lf]" 0)))) ("or" ("not" '
        '("app" ("#const[Lf]" "j") ("app" ("app" ("#const[Lf]" "k") '
        '("#bvar[Lf]" 2)) ("#bvar[Lf]" 1)))) ("and" ("app" ("#const[Lf]" "l") '
        '("#bvar[Lf]" 0)) ("app" ("#const[Lf]" "m") ("#bvar[Lf]" 1))))))))'
    ),
)

REFERENCE_SIGNATURE: Final = Signature(
    "reference-formulas",
    ("Lf",),
    (
        Operator("app", "Lf", arg_sorts=("Lf", "Lf"), family="application"),
        Operator(
            "lam",
            "Lf",
            arg_sorts=("Lf",),
            binding=(BindingSite((0,), "Lf"),),
            family="binder",
        ),
        Operator(
            "forall",
            "Lf",
            arg_sorts=("Lf",),
            binding=(BindingSite((0,), "Lf"),),
            family="binder",
        ),
        Operator(
            "exists",
            "Lf",
            arg_sorts=("Lf",),
            binding=(BindingSite((0,), "Lf"),),
            family="binder",
        ),
        Operator("and", "Lf", arg_sorts=("Lf", "Lf"), family="connective"),
        Operator("or", "Lf", arg_sorts=("Lf", "Lf"), family="connective"),
        Operator("implies", "Lf", arg_sorts=("Lf", "Lf"), family="connective"),
        Operator("iff", "Lf", arg_sorts=("Lf", "Lf"), family="connective"),
        Operator("not", "Lf", arg_sorts=("Lf",), family="connective"),
        Operator("equal", "Lf", arg_sorts=("Lf", "Lf"), family="connective"),
    ),
    vocabularies=(
        Vocabulary(
            "Lf",
            tuple(
                sorted(
                    {
                        name
                        for case in REFERENCE_SEXP_CASES
                        for name in re.findall(r'"#const\[Lf\]" "([^"]+)"', case)
                    }
                )
            ),
        ),
    ),
)


def test_s_expression_preserves_binders_and_named_payloads() -> None:
    pred = FORMULA_STORE.const("Lf", "sleep")
    bound = FORMULA_STORE.op(FORMULA_SIGNATURE.bound("Lf"), payload=0)
    applied = FORMULA_STORE.op(FORMULA_SIGNATURE.by_name("app"), pred, bound)
    term = FORMULA_STORE.op(FORMULA_SIGNATURE.by_name("lam"), applied)
    written = '("lam" ("app" ("#const[Lf]" "sleep") ("#bvar[Lf]" 0)))'

    assert FORMULA_SYNTAX.render(FORMULA_STORE, term) == written
    assert FORMULA_SYNTAX.parse(FORMULA_STORE, written) is term


@pytest.mark.parametrize("written", REFERENCE_SEXP_CASES)
def test_reference_semantic_forms_have_canonical_trees(written: str) -> None:
    store = TermStore(REFERENCE_SIGNATURE)
    syntax = SExpSyntax(REFERENCE_SIGNATURE)
    parsed = syntax.parse(store, written)

    assert syntax.render(store, parsed) == written
    assert syntax.parse(store, written) is parsed


@given(
    terms(
        FORMULA_STORE,
        sort="Lf",
        max_size=FORMULA_SIZE,
        closed=True,
    )
)
@settings(max_examples=2_000, deadline=None)
def test_thousands_of_generated_logical_forms_round_trip(
    generated: Term,
) -> None:
    written = FORMULA_SYNTAX.render(FORMULA_STORE, generated)

    assert FORMULA_SYNTAX.parse(FORMULA_STORE, written) is generated
    assert (
        FORMULA_SYNTAX.render(
            FORMULA_STORE,
            FORMULA_SYNTAX.parse(FORMULA_STORE, f" \n {written}\t "),
        )
        == written
    )


def test_s_expression_quotes_arbitrary_operator_and_atom_names() -> None:
    signature = Signature(
        "quoted",
        ("Node",),
        (Operator("operator with spaces", "Node", arg_sorts=("Node",)),),
        vocabularies=(Vocabulary("Node", ('a "quoted" name',)),),
    )
    store = TermStore(signature)
    term = store.op(
        signature.by_name("operator with spaces"),
        store.const("Node", 'a "quoted" name'),
    )
    syntax = SExpSyntax(signature)
    written = '("operator with spaces" ("#const[Node]" "a \\"quoted\\" name"))'

    assert syntax.render(store, term) == written
    assert syntax.parse(store, written) is term


def test_s_expression_is_a_store_independent_crossing_format() -> None:
    source = FORMULA_STORE.const("Lf", "crossing-name")
    written = FORMULA_SYNTAX.render(FORMULA_STORE, source)
    target = TermStore(FORMULA_SIGNATURE)
    decoded = FORMULA_SYNTAX.parse(target, written)

    assert FORMULA_SYNTAX.render(target, decoded) == written
    assert target.payload_name(decoded) == "crossing-name"


def test_s_expression_reports_unknown_operators_at_their_offsets() -> None:
    with pytest.raises(TextParseError) as caught:
        FORMULA_SYNTAX.parse(FORMULA_STORE, '("missing")')

    assert caught.value.operator == "missing"
    assert caught.value.offset == 1
    assert "character offset 1" in str(caught.value)


def test_s_expression_reports_constructor_and_offset_for_bad_shapes() -> None:
    malformed = '("app" ("#const[Lf]" "f"))'

    with pytest.raises(TextParseError) as caught:
        FORMULA_SYNTAX.parse(FORMULA_STORE, malformed)

    assert caught.value.operator == "app"
    assert caught.value.offset == len(malformed) - 1


def test_s_expression_refuses_an_unquoted_operator_name() -> None:
    with pytest.raises(TextParseError) as caught:
        FORMULA_SYNTAX.parse(FORMULA_STORE, "(1)")

    assert caught.value.operator == "<operator>"
    assert caught.value.offset == 1
    assert caught.value.reason == "expected a quoted name"


def test_s_expression_locates_invalid_named_payloads() -> None:
    with pytest.raises(TextParseError) as caught:
        FORMULA_SYNTAX.parse(FORMULA_STORE, '("#const[Lf]" "")')

    assert caught.value.operator == "#const[Lf]"
    assert caught.value.offset == 14


def test_s_expression_reports_sort_failures_at_the_parent_operator() -> None:
    signature = Signature(
        "sorted",
        ("A", "B"),
        (
            Operator("a", "A"),
            Operator("b", "B"),
            Operator("take-a", "A", arg_sorts=("A",)),
        ),
    )
    store = TermStore(signature)

    with pytest.raises(TextParseError) as caught:
        SExpSyntax(signature).parse(store, '("take-a" ("b"))')

    assert caught.value.operator == "take-a"
    assert caught.value.offset == 1
    assert "sort" in caught.value.reason


@pytest.mark.parametrize(
    "payload",
    [
        "\N{SUPERSCRIPT TWO}",
        "\N{ARABIC-INDIC DIGIT ONE}",
        "\N{FULLWIDTH DIGIT ONE}",
    ],
)
def test_s_expression_accepts_only_ascii_index_digits(payload: str) -> None:
    with pytest.raises(TextParseError) as caught:
        FORMULA_SYNTAX.parse(FORMULA_STORE, f'("#bvar[Lf]" {payload})')

    assert caught.value.operator == "#bvar[Lf]"
    assert caught.value.offset == 13
    assert "nonnegative integer" in caught.value.reason


def test_s_expression_locates_oversized_integer_payloads() -> None:
    payload = "9" * 5_000

    with pytest.raises(TextParseError) as caught:
        FORMULA_SYNTAX.parse(FORMULA_STORE, f'("#bvar[Lf]" {payload})')

    assert caught.value.operator == "#bvar[Lf]"
    assert caught.value.offset == 13
    assert caught.value.reason == "integer payload is too large"


@pytest.mark.parametrize(
    "malformed",
    [
        '("app" ("#const[Lf]" "new-name") ("missing"))',
        '("#const[Lf]" "new-name") trailing',
    ],
)
def test_malformed_s_expressions_do_not_mutate_the_store(malformed: str) -> None:
    store = TermStore(FORMULA_SIGNATURE)
    before = (store.local_nodes, store.hits, store.misses)

    with pytest.raises(TextParseError):
        FORMULA_SYNTAX.parse(store, malformed)

    assert (store.local_nodes, store.hits, store.misses) == before
    assert store.atom("Lf", "after-failure") == 0


def test_s_expression_depth_bound_applies_to_parsing_and_rendering() -> None:
    syntax = SExpSyntax(FORMULA_SIGNATURE, max_depth=3)
    store = TermStore(FORMULA_SIGNATURE)
    text = '("lam" ("lam" ("lam" ("#const[Lf]" "deep"))))'
    limit = getrecursionlimit()

    with pytest.raises(TextParseError, match="maximum depth 3") as parse_error:
        syntax.parse(store, text)

    assert parse_error.value.operator == "lam"
    assert parse_error.value.offset > 0
    assert parse_error.value.reason == "nesting exceeds maximum depth 3"

    pred = store.const("Lf", "deep")
    term = pred

    for _ in range(3):
        term = store.op(FORMULA_SIGNATURE.by_name("lam"), term)

    with pytest.raises(TextDepthError, match="maximum depth 3") as render_error:
        syntax.render(store, term)

    assert render_error.value.operator == "#const[Lf]"
    assert render_error.value.max_depth == 3
    assert getrecursionlimit() == limit


def test_s_expression_accepts_its_exact_depth_bound() -> None:
    syntax = SExpSyntax(FORMULA_SIGNATURE, max_depth=3)
    store = TermStore(FORMULA_SIGNATURE)
    text = '("lam" ("lam" ("#const[Lf]" "edge")))'
    parsed = syntax.parse(store, text)

    assert syntax.render(store, parsed) == text


@pytest.mark.parametrize("max_depth", [0, -1])
def test_syntaxes_require_a_positive_depth_bound(max_depth: int) -> None:
    with pytest.raises(ValueError, match="max_depth"):
        SExpSyntax(FORMULA_SIGNATURE, max_depth=max_depth)

    with pytest.raises(ValueError, match="max_depth"):
        AngleSyntax(CHURCH_SIGNATURE, "->", max_depth=max_depth)

    with pytest.raises(ValueError, match="max_depth"):
        infix_syntax(
            CCG_SIGNATURE,
            ("/", "fwd"),
            ("\\", "bwd"),
            max_depth=max_depth,
        )


@pytest.mark.parametrize(
    "max_depth",
    [cast("int", 1.0), cast("int", "3")],
)
def test_syntaxes_require_an_integer_depth_bound(max_depth: int) -> None:
    with pytest.raises(TypeError, match="max_depth"):
        SExpSyntax(FORMULA_SIGNATURE, max_depth=max_depth)

    with pytest.raises(TypeError, match="max_depth"):
        infix_syntax(
            CCG_SIGNATURE,
            ("/", "fwd"),
            ("\\", "bwd"),
            max_depth=max_depth,
        )


CCG_SIGNATURE: Final = categories()
CCG_STORE: Final = TermStore(CCG_SIGNATURE)
CCG_SYNTAX: Final = infix_syntax(CCG_SIGNATURE, ("/", "fwd"), ("\\", "bwd"))


@pytest.mark.parametrize(
    ("source", "canonical"),
    [
        ("s", "s"),
        (" s [dcl] ", "s[dcl]"),
        ("s\\np", "s\\np"),
        ("(s[b] \\ np) / np", "s[b]\\np/np"),
        ("s / (s \\ np)", "s/(s\\np)"),
    ],
)
def test_ccg_text_canonicalizes_and_round_trips(
    source: str,
    canonical: str,
) -> None:
    parsed = CCG_SYNTAX.parse(CCG_STORE, source)

    assert CCG_SYNTAX.render(CCG_STORE, parsed) == canonical
    assert CCG_SYNTAX.parse(CCG_STORE, canonical) is parsed


OPEN_CCGBANK_PRIMITIVES: Final = (
    "s",
    "np",
    "n",
    "pp",
    "conj",
    ".",
    ",",
    ":",
    ";",
    "lrb",
    "rrb",
)
OPEN_CCGBANK_FEATURES: Final = {
    "s": (
        "dcl",
        "wq",
        "q",
        "qem",
        "em",
        "bem",
        "b",
        "frg",
        "for",
        "intj",
        "inv",
        "to",
        "pss",
        "pt",
        "ng",
        "adj",
        "as",
        "poss",
        "asup",
    ),
    "n": ("num",),
    "np": ("expl", "thr", "nb", "conj"),
}


def open_ccgbank_signature() -> Signature:
    primitives = tuple(Operator(name, "Cat") for name in OPEN_CCGBANK_PRIMITIVES)

    return Signature(
        "open-ccgbank-text",
        ("Cat",),
        (
            *primitives,
            Operator("fwd", "Cat", arg_sorts=("Cat", "Cat"), family="slash"),
            Operator("bwd", "Cat", arg_sorts=("Cat", "Cat"), family="slash"),
        ),
        features=OPEN_CCGBANK_FEATURES,
    )


def test_open_ccgbank_inventory_round_trips_canonically() -> None:
    signature = open_ccgbank_signature()
    store = TermStore(signature)
    syntax = infix_syntax(signature, ("/", "fwd"), ("\\", "bwd"))
    atoms = tuple(store.op(op) for op in signature.by_arity(0))
    inventory = (
        *atoms,
        *(
            store.op(op, left, right)
            for op in signature.by_arity(2)
            for left, right in product(atoms, repeat=2)
        ),
    )

    for term in inventory:
        written = syntax.render(store, term)
        parsed = syntax.parse(store, written)

        assert parsed is term
        assert syntax.render(store, parsed) == written


def test_ccg_syntax_uses_the_supplied_undirected_constructor() -> None:
    signature = Signature(
        "undirected",
        ("Cat",),
        (
            Operator("s", "Cat"),
            Operator("np", "Cat"),
            Operator("arrow", "Cat", arg_sorts=("Cat", "Cat")),
        ),
    )
    store = TermStore(signature)
    syntax = infix_syntax(signature, ("|", "arrow"))
    term = syntax.parse(store, "s|(np|s)")

    assert term.op is signature.by_name("arrow")
    assert syntax.render(store, term) == "s|(np|s)"
    assert syntax.parse(store, syntax.render(store, term)) is term


def test_ccg_syntax_requires_one_token_for_each_constructor() -> None:
    signature = Signature(
        "undirected",
        ("Cat",),
        (
            Operator("s", "Cat"),
            Operator("arrow", "Cat", arg_sorts=("Cat", "Cat")),
        ),
    )

    with pytest.raises(ValueError, match="more than one"):
        ConstructorSyntax(
            signature,
            (
                ConstructorNotation.infix(
                    "arrow", "|", precedence=10, associativity="left"
                ),
                ConstructorNotation.infix(
                    "arrow", "^", precedence=10, associativity="left"
                ),
            ),
        )

    with pytest.raises(ValueError, match="no surface token"):
        infix_syntax(
            Signature(
                "two",
                ("Cat",),
                (
                    Operator("s", "Cat"),
                    Operator("left", "Cat", arg_sorts=("Cat", "Cat")),
                    Operator("right", "Cat", arg_sorts=("Cat", "Cat")),
                ),
            ),
            ("|", "left"),
        )


def test_ccg_syntax_round_trips_features_on_compound_categories() -> None:
    signature = Signature(
        "compound-feature",
        ("Cat",),
        (
            Operator("s", "Cat"),
            Operator("np", "Cat"),
            Operator("arrow", "Cat", arg_sorts=("Cat", "Cat")),
        ),
        features={"arrow": ("conj",)},
    )
    store = TermStore(signature)
    syntax = infix_syntax(signature, ("|", "arrow"))
    term = store.op(
        signature.featured("arrow", "conj"),
        store.op(signature.by_name("s")),
        store.op(signature.by_name("np")),
    )
    written = "(s|np)[conj]"

    assert syntax.render(store, term) == written
    assert syntax.parse(store, written) is term


def test_declared_category_syntax_uses_its_selected_feature_delimiter() -> None:
    signature = Signature(
        "colon-features",
        ("Cat",),
        (
            Operator("s", "Cat"),
            Operator("s:dcl", "Cat", erases_to="s"),
            Operator("np", "Cat"),
            Operator("fwd", "Cat", arg_sorts=("Cat", "Cat")),
            Operator("bwd", "Cat", arg_sorts=("Cat", "Cat")),
        ),
    )
    store = TermStore(signature)
    syntax = ConstructorSyntax(
        signature,
        (
            ConstructorNotation.infix("fwd", "/", precedence=10, associativity="left"),
            ConstructorNotation.infix("bwd", "\\", precedence=10, associativity="left"),
        ),
        feature_delimiter="colon",
    )

    category = syntax.parse(store, "s:dcl\\np")

    assert category.op is signature.by_name("bwd")
    assert category.args[0].op is signature.by_name("s:dcl")
    assert signature.erase(category.args[0].op) is signature.by_name("s")
    assert syntax.render(store, category) == "s:dcl\\np"
    with pytest.raises(TextParseError):
        syntax.parse(store, "s[dcl]\\np")


def test_ccg_syntax_reads_stacked_and_category_level_features() -> None:
    signature = Signature(
        "category-features",
        ("Cat",),
        (
            Operator("s", "Cat"),
            Operator("n", "Cat"),
            Operator("fwd", "Cat", arg_sorts=("Cat", "Cat")),
        ),
        features={
            "s": ("dcl", "dcl][conj"),
            "n": ("conj",),
            "fwd": ("conj",),
        },
    )
    store = TermStore(signature)
    syntax = infix_syntax(signature, ("/", "fwd"))
    stacked = syntax.parse(store, "s[dcl][conj]")
    category_level = syntax.parse(store, "(n/n)[conj]")
    featured_right = store.op(
        signature.by_name("fwd"),
        store.op(signature.by_name("n")),
        store.op(signature.featured("n", "conj")),
    )

    assert stacked.op is signature.featured("s", "dcl][conj")
    assert syntax.render(store, stacked) == "s[dcl][conj]"
    assert category_level.op is signature.featured("fwd", "conj")
    assert category_level.args[1].op is signature.by_name("n")
    assert syntax.render(store, category_level) == "(n/n)[conj]"
    assert syntax.parse(store, "(n/n)[conj]") is category_level
    assert syntax.render(store, featured_right) == "n/n[conj]"
    assert syntax.parse(store, syntax.render(store, featured_right)) is featured_right


def test_ccg_syntax_rejects_features_that_conflict_with_bracket_syntax() -> None:
    signature = Signature(
        "bad-feature",
        ("Cat",),
        (
            Operator("s", "Cat"),
            Operator("arrow", "Cat", arg_sorts=("Cat", "Cat")),
        ),
        features={"s": ("bad]feature",)},
    )

    with pytest.raises(ValueError, match="bracketed suffix"):
        infix_syntax(signature, ("|", "arrow"))


def test_ccg_parse_errors_name_the_operator_and_offset() -> None:
    with pytest.raises(TextParseError) as unknown:
        CCG_SYNTAX.parse(CCG_STORE, "missing/np")

    assert unknown.value.operator == "missing"
    assert unknown.value.offset == 0

    with pytest.raises(TextParseError) as incomplete:
        CCG_SYNTAX.parse(CCG_STORE, "s/")

    assert incomplete.value.operator == "fwd"
    assert incomplete.value.offset == 2


def test_malformed_ccg_text_does_not_mutate_the_store() -> None:
    store = TermStore(CCG_SIGNATURE)
    before = (store.local_nodes, store.hits, store.misses)

    with pytest.raises(TextParseError):
        CCG_SYNTAX.parse(store, "s/np trailing")

    assert (store.local_nodes, store.hits, store.misses) == before


@pytest.mark.parametrize(
    ("text", "operator", "offset"),
    [
        ("s/s/s/s", "fwd", 5),
        ("((((s))))", "<expression>", 3),
    ],
)
def test_ccg_depth_bound_rejects_deep_text(
    text: str,
    operator: str,
    offset: int,
) -> None:
    syntax = infix_syntax(
        CCG_SIGNATURE,
        ("/", "fwd"),
        ("\\", "bwd"),
        max_depth=3,
    )
    store = TermStore(CCG_SIGNATURE)
    limit = getrecursionlimit()

    with pytest.raises(TextParseError, match="maximum depth 3") as caught:
        syntax.parse(store, text)

    assert caught.value.operator == operator
    assert caught.value.offset == offset
    assert caught.value.reason == "nesting exceeds maximum depth 3"
    assert (store.local_nodes, store.hits, store.misses) == (0, 0, 0)
    assert getrecursionlimit() == limit


@pytest.mark.parametrize("text", ["s/s/s", "(((s)))"])
def test_ccg_parser_accepts_its_exact_depth_bound(text: str) -> None:
    syntax = infix_syntax(
        CCG_SIGNATURE,
        ("/", "fwd"),
        ("\\", "bwd"),
        max_depth=3,
    )
    store = TermStore(CCG_SIGNATURE)
    parsed = syntax.parse(store, text)

    assert syntax.parse(store, syntax.render(store, parsed)) is parsed


def test_ccg_depth_bound_applies_to_rendering() -> None:
    syntax = infix_syntax(
        CCG_SIGNATURE,
        ("/", "fwd"),
        ("\\", "bwd"),
        max_depth=3,
    )
    store = TermStore(CCG_SIGNATURE)
    atom = store.op(CCG_SIGNATURE.by_name("s"))
    term = atom

    for _ in range(3):
        term = store.op(CCG_SIGNATURE.by_name("fwd"), term, atom)

    with pytest.raises(TextDepthError, match="maximum depth 3") as caught:
        syntax.render(store, term)

    assert caught.value.operator == "s"
    assert caught.value.max_depth == 3


def test_syntax_values_are_frozen_and_constructor_maps_are_read_only() -> None:
    with pytest.raises(AttributeError, match="immutable"):
        FORMULA_SYNTAX.__setattr__("signature", categories())

    with pytest.raises(AttributeError, match="immutable"):
        FORMULA_SYNTAX.__delattr__("_signature")

    with pytest.raises(AttributeError, match="immutable"):
        CCG_SYNTAX.__delattr__("_table")

    with pytest.raises(TypeError):
        cast("dict[str, ConstructorNotation]", CCG_SYNTAX.notations)["fwd"] = (
            ConstructorNotation.infix("fwd", "|", precedence=10, associativity="left")
        )


def test_a_syntax_refuses_a_store_for_another_signature() -> None:
    other = TermStore(formulas())

    with pytest.raises(ValueError, match="cannot use store"):
        FORMULA_SYNTAX.parse(other, '("#const[Lf]" "sleep")')


CHURCH_SIGNATURE: Final = Signature(
    "church",
    ("Ty", "Tm"),
    (
        Operator("e", "Ty"),
        Operator("t", "Ty"),
        Operator("->", "Ty", arg_sorts=("Ty", "Ty"), family="constructor"),
        Operator("app", "Tm", arg_sorts=("Tm", "Tm"), family="application"),
        Operator(
            "lam",
            "Tm",
            arg_sorts=("Ty", "Tm"),
            binding=(BindingSite((1,), "Tm", 0),),
            family="binder",
        ),
    ),
    vocabularies=(Vocabulary("Tm", ("sleep", "t", "two words", "e")),),
    open_sorts=("Ty",),
    classifiers={"Tm": "Ty"},
)
TYPES: Final = AngleSyntax(CHURCH_SIGNATURE, "->")
FORMS: Final = AngleSyntax(CHURCH_SIGNATURE, "->", sort="Tm")
CHURCH_SEXP: Final = SExpSyntax(CHURCH_SIGNATURE)


def test_angle_brackets_spell_the_pair_constructor_as_the_s_expression_does() -> None:
    store = TermStore(CHURCH_SIGNATURE)
    quantifier = TYPES.parse(store, "⟨⟨e,t⟩,t⟩")

    assert quantifier is CHURCH_SEXP.parse(store, '("->" ("->" ("e") ("t")) ("t"))')
    assert quantifier.args[0].args[0].op.name == "e"
    assert TYPES.parse(store, " ⟨ ⟨e , t⟩,\tt ⟩ ") is quantifier
    assert TYPES.render(store, quantifier) == "⟨⟨e,t⟩,t⟩"
    assert TYPES.render(store, TYPES.parse(store, "⟨e,⟨e,t⟩⟩")) == "⟨e,⟨e,t⟩⟩"
    assert TYPES.pair == "->"
    assert TYPES.sort == "Ty"
    assert FORMS.sort == "Tm"


def test_angle_text_reads_leaves_by_the_sort_its_position_expects() -> None:
    store = TermStore(CHURCH_SIGNATURE)
    form = FORMS.parse(store, "lam⟨⟨e,t⟩,app⟨t,#bvar[Tm]{0}⟩⟩")
    annotation, body = form.args

    assert annotation is TYPES.parse(store, "⟨e,t⟩")
    assert body.args[0] is store.const("Tm", "t")
    assert body.args[1] is store.bvar("Tm", 0)
    assert FORMS.render(store, form) == "lam⟨⟨e,t⟩,app⟨t,#bvar[Tm]{0}⟩⟩"


def test_angle_text_writes_shadowed_and_reserved_constants_explicitly() -> None:
    store = TermStore(CHURCH_SIGNATURE)
    plain = FORMS.parse(store, "app⟨e,sleep⟩")
    quoted = FORMS.parse(store, '#const[Tm]{"two words"}')
    variable = TYPES.parse(store, '#fvar[Ty]{"a b"}')

    assert plain.args[0] is store.const("Tm", "e")
    assert FORMS.render(store, plain) == "app⟨e,sleep⟩"
    assert FORMS.render(store, quoted) == '#const[Tm]{"two words"}'
    assert FORMS.parse(store, "#const[Tm]{sleep}") is store.const("Tm", "sleep")
    assert TYPES.render(store, variable) == '#fvar[Ty]{"a b"}'
    assert variable is store.fvar("Ty", "a b")


def test_a_constant_named_like_a_nullary_operator_of_its_sort_is_explicit() -> None:
    signature = Signature(
        "shadow",
        ("Ty",),
        (
            Operator("e", "Ty"),
            Operator("->", "Ty", arg_sorts=("Ty", "Ty")),
        ),
        vocabularies=(Vocabulary("Ty", ("e", "s")),),
    )
    store = TermStore(signature)
    syntax = AngleSyntax(signature, "->")
    term = store.op(
        signature.by_name("->"), store.const("Ty", "e"), store.const("Ty", "s")
    )
    written = syntax.render(store, term)

    assert written == "⟨#const[Ty]{e},s⟩"
    assert syntax.parse(store, written) is term
    assert syntax.parse(store, "e").op.name == "e"


def test_angle_text_quotes_operator_names_holding_reserved_characters() -> None:
    signature = Signature(
        "quoted",
        ("S",),
        (
            Operator("a,b", "S"),
            Operator("pair", "S", arg_sorts=("S", "S")),
            Operator("f {x}", "S", arg_sorts=("S",)),
        ),
    )
    store = TermStore(signature)
    syntax = AngleSyntax(signature, "pair")
    term = syntax.parse(store, '"f {x}"⟨⟨"a,b","a,b"⟩⟩')

    assert term.op.name == "f {x}"
    assert syntax.render(store, term) == '"f {x}"⟨⟨"a,b","a,b"⟩⟩'


@pytest.mark.parametrize(
    ("syntax", "text", "operator", "offset", "reason"),
    [
        (TYPES, "⟨e,⟩", "->", 3, "expected a name"),
        (TYPES, "⟨e,t", "->", 4, "expected '⟩'"),
        (TYPES, "w", "w", 0, "neither a nullary operator nor a constant of 'Ty'"),
        (TYPES, "e⟨t⟩", "e", 1, "takes no arguments"),
        (TYPES, "⟨e,t⟩ t", "->", 6, "found trailing input"),
        (TYPES, "sleep", "sleep", 0, "neither a nullary operator nor a constant"),
        (TYPES, "app⟨e,t⟩", "app", 0, "has sort 'Tm' where 'Ty' is due"),
        (FORMS, "⟨e,t⟩", "->", 0, "has sort 'Ty' where 'Tm' is due"),
        (FORMS, "#bvar[Tm]⟨⟩", "#bvar[Tm]", 9, "expected '{'"),
        (FORMS, "#bvar[Tm]{x}", "#bvar[Tm]", 10, "nonnegative integer payload"),
        (FORMS, "#bvar[Tm]{\u0663}", "#bvar[Tm]", 10, "nonnegative integer payload"),
        (FORMS, "#const[Tm]{snore}", "#const[Tm]", 11, "not a constant of 'Tm'"),
        (FORMS, "app⟨sleep,snore⟩", "snore", 10, "nor a constant of 'Tm'"),
        (FORMS, "app{0}⟨t,t⟩", "app", 3, "takes no payload"),
        (FORMS, "nope⟨t⟩", "nope", 0, "not declared by the signature"),
        (FORMS, 'app⟨t,"⟩', "app", 6, "expected a quoted name"),
        (FORMS, 'app⟨t,""⟩', "app", 6, "expected a quoted name"),
        (FORMS, "app⟨t⟩", "app", 5, "expected ','"),
    ],
)
def test_angle_parse_errors_name_the_operator_and_offset(
    syntax: AngleSyntax, text: str, operator: str, offset: int, reason: str
) -> None:
    with pytest.raises(TextParseError, match=re.escape(reason)) as caught:
        syntax.parse(TermStore(CHURCH_SIGNATURE), text)

    assert caught.value.operator == operator
    assert caught.value.offset == offset


def test_malformed_angle_text_does_not_mutate_the_store() -> None:
    store = TermStore(CHURCH_SIGNATURE)
    before = (store.local_nodes, store.hits, store.misses)

    with pytest.raises(TextParseError):
        TYPES.parse(store, "⟨#fvar[Ty]{fresh},w⟩")

    assert (store.local_nodes, store.hits, store.misses) == before
    assert store.atom("Ty", "after-failure") == 0


def test_angle_depth_bound_applies_to_parsing_and_rendering() -> None:
    store = TermStore(CHURCH_SIGNATURE)
    shallow = AngleSyntax(CHURCH_SIGNATURE, "->", max_depth=2)
    deep = TYPES.parse(store, "⟨⟨e,t⟩,t⟩")

    assert shallow.parse(store, "⟨e,t⟩") is TYPES.parse(store, "⟨e,t⟩")

    with pytest.raises(TextParseError, match="maximum depth 2"):
        shallow.parse(store, "⟨⟨e,t⟩,t⟩")

    with pytest.raises(TextDepthError) as caught:
        shallow.render(store, deep)

    assert caught.value.max_depth == 2


def test_angle_syntax_refuses_a_term_of_another_sort() -> None:
    store = TermStore(CHURCH_SIGNATURE)

    with pytest.raises(SortError) as caught:
        TYPES.render(store, store.const("Tm", "sleep"))

    assert (caught.value.expected, caught.value.received) == ("Ty", "Tm")


@pytest.mark.parametrize(
    ("pair", "sort", "error", "message"),
    [
        ("lam", None, ValueError, "not a payload-free non-binding pair"),
        ("e", None, ValueError, "not a payload-free non-binding pair"),
        ("#bvar[Tm]", None, ValueError, "not a payload-free non-binding pair"),
        ("->", "Lf", ValueError, "sort 'Lf' is not declared"),
        ("missing", None, UnknownOperatorError, "missing"),
    ],
)
def test_angle_syntax_requires_a_binary_pair_and_a_declared_sort(
    pair: str, sort: str | None, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=re.escape(message)):
        AngleSyntax(CHURCH_SIGNATURE, pair, sort=sort)


@st.composite
def declared_languages(draw: st.DrawFn) -> Signature:
    """Draw a many-sorted signature with a pair, binders, constants and variables."""
    sorts = tuple(f"S{index}" for index in range(draw(st.integers(1, 3))))
    names = st.text(alphabet='ab ⟨⟩,{}"x', min_size=1, max_size=3)
    operators: list[Operator] = [
        Operator(f"n{sort}{index}", sort)
        for sort in sorts
        for index in range(draw(st.integers(0, 2)))
    ]
    operators.append(
        Operator(
            "pair",
            sorts[0],
            arg_sorts=(draw(st.sampled_from(sorts)), draw(st.sampled_from(sorts))),
        )
    )
    operators.extend(
        Operator(
            f"g{index}",
            draw(st.sampled_from(sorts)),
            arg_sorts=tuple(
                draw(st.lists(st.sampled_from(sorts), min_size=1, max_size=3))
            ),
        )
        for index in range(draw(st.integers(0, 3)))
    )
    bound = draw(st.sampled_from(sorts))

    if draw(st.booleans()):
        operators.append(
            Operator(
                "bind",
                bound,
                arg_sorts=(bound,),
                binding=(BindingSite((0,), bound),),
            )
        )

    vocabularies = tuple(
        Vocabulary(
            sort,
            draw(
                st.lists(
                    st.one_of(names, st.sampled_from([f"n{sort}0", "pair"])),
                    unique=True,
                    min_size=1,
                    max_size=3,
                )
            ),
        )
        for sort in sorts
        if draw(st.booleans())
    )

    return Signature(
        "generated",
        sorts,
        operators,
        vocabularies=vocabularies,
        open_sorts=draw(st.lists(st.sampled_from(sorts), unique=True)),
        holes=draw(st.booleans()),
    )


@given(st.data())
@settings(max_examples=300, deadline=None)
def test_generated_languages_round_trip_every_sort_through_angles_and_records(
    data: st.DataObject,
) -> None:
    signature = data.draw(declared_languages())
    sort = data.draw(st.sampled_from(signature.sorts))
    store = TermStore(signature)
    term = data.draw(terms(store, sort=sort, max_size=12, closed=False))
    syntax = AngleSyntax(signature, "pair", sort=sort)
    written = syntax.render(store, term)
    elsewhere = TermStore(signature)
    published = json.loads(json.dumps(record_to_json(to_record(store, term))))
    rebuilt = from_record(elsewhere, record_from_json(published))

    assert syntax.parse(store, written) is term
    assert syntax.parse(store, f" {written} ") is term
    assert rebuilt.key() == term.key()
    assert to_record(elsewhere, rebuilt) == to_record(store, term)
    assert hash(to_record(elsewhere, rebuilt)) == hash(to_record(store, term))
    assert syntax.render(elsewhere, rebuilt) == written


def test_a_church_binder_reads_its_annotation_in_the_type_sort() -> None:
    store = TermStore(CHURCH_SIGNATURE)
    form = FORMS.parse(store, "lam⟨#fvar[Ty]{a},lam⟨⟨#fvar[Ty]{a},t⟩,#bvar[Tm]{1}⟩⟩")

    with pytest.raises(TextParseError, match="neither"):
        FORMS.parse(store, "lam⟨a,t⟩")

    assert form.args[0] is store.fvar("Ty", "a")
    assert FORMS.parse(store, FORMS.render(store, form)) is form


def test_a_term_record_names_its_leaves_and_survives_a_vocabulary_reordering() -> None:
    reordered = Signature(
        "church",
        CHURCH_SIGNATURE.sorts,
        CHURCH_SIGNATURE.declarations,
        vocabularies=(Vocabulary("Tm", ("e", "two words", "t", "sleep")),),
        open_sorts=("Ty",),
        classifiers={"Tm": "Ty"},
    )
    here = TermStore(CHURCH_SIGNATURE)
    there = TermStore(reordered)
    form = FORMS.parse(here, 'lam⟨#fvar[Ty]{a},app⟨sleep,#const[Tm]{"two words"}⟩⟩')
    record = to_record(here, form)
    moved = from_record(there, record)

    assert record == OperatorRecord(
        "lam",
        (
            FreeRecord("Ty", "a"),
            OperatorRecord(
                "app",
                (ConstantRecord("Tm", "sleep"), ConstantRecord("Tm", "two words")),
            ),
        ),
    )
    assert moved.key() != form.key()
    assert to_record(there, moved) == record
    assert record_to_json(record) == {
        "operator": "lam",
        "arguments": [
            {"free": "a", "sort": "Ty"},
            {
                "operator": "app",
                "arguments": [
                    {"constant": "sleep", "sort": "Tm"},
                    {"constant": "two words", "sort": "Tm"},
                ],
            },
        ],
    }


def test_records_of_different_kinds_differ_even_with_one_name() -> None:
    leaves = (
        ConstantRecord("Tm", "a"),
        FreeRecord("Tm", "a"),
        ConstantRecord("Ty", "a"),
        BoundRecord("Tm", 0),
        BoundRecord("Tm", 1),
        OperatorRecord("a"),
    )

    for left, right in product(leaves, repeat=2):
        assert (left == right) is (left is right)

    assert {BoundRecord("Tm", 0), BoundRecord("Tm", 0)} == {BoundRecord("Tm", 0)}
    assert repr(BoundRecord("Tm", 0)) == "BoundRecord('Tm', 0)"
    assert repr(OperatorRecord("app", leaves[:2])) == "OperatorRecord('app', arity=2)"


def test_records_with_colliding_hashes_compare_by_structure() -> None:
    low = BoundRecord("Tm", 0)
    high = BoundRecord("Tm", hash_info.modulus)
    left = OperatorRecord("f", (low,))
    right = OperatorRecord("f", (high,))

    assert hash(left) == hash(right)
    assert left != right
    assert record_differences(left, right) == ((0,),)


class _UniformName(str):
    """A name whose hash is the same as every other name of its kind."""

    __slots__ = ()

    @override
    def __hash__(self) -> int:
        return 0


def test_records_whose_operator_names_hash_alike_compare_by_name() -> None:
    leaf = BoundRecord("Tm", 0)
    left = OperatorRecord(_UniformName("f"), (leaf,))
    right = OperatorRecord(_UniformName("g"), (leaf,))

    assert hash(left) == hash(right)
    assert left != right
    assert OperatorRecord("h", (left,)) != OperatorRecord("h", (right,))
    assert record_differences(left, right) == ((),)


def test_record_differences_name_the_outermost_differing_paths() -> None:
    store = TermStore(CHURCH_SIGNATURE)
    left = to_record(store, TYPES.parse(store, "⟨⟨e,t⟩,⟨e,t⟩⟩"))
    right = to_record(store, TYPES.parse(store, "⟨⟨e,e⟩,⟨t,t⟩⟩"))
    other = to_record(store, TYPES.parse(store, "e"))

    assert record_differences(left, left) == ()
    assert record_differences(left, right) == ((0, 1), (1, 0))
    assert record_differences(left, other) == ((),)
    assert record_differences(OperatorRecord("f", (other,)), OperatorRecord("f")) == (
        (),
    )


def test_unbounded_syntaxes_read_and_write_text_deeper_than_the_recursion_limit() -> (
    None
):
    depth = 4 * getrecursionlimit()
    formulas_store = TermStore(FORMULA_SIGNATURE)
    form = formulas_store.const("Lf", "deep")

    for _ in range(depth):
        form = formulas_store.op(FORMULA_SIGNATURE.by_name("lam"), form)

    ccg_store = TermStore(CCG_SIGNATURE)
    right_nested = "s/(" * depth + "s/s" + ")" * depth
    church_store = TermStore(CHURCH_SIGNATURE)
    arrows = "⟨e," * depth + "t" + "⟩" * depth

    assert (FORMULA_SYNTAX.max_depth, CCG_SYNTAX.max_depth, TYPES.max_depth) == (
        None,
        None,
        None,
    )
    assert (
        FORMULA_SYNTAX.parse(
            formulas_store, FORMULA_SYNTAX.render(formulas_store, form)
        )
        is form
    )
    assert CCG_SYNTAX.render(ccg_store, CCG_SYNTAX.parse(ccg_store, right_nested)) == (
        right_nested
    )
    assert CCG_SYNTAX.parse(ccg_store, "(" * depth + "s" + ")" * depth) is (
        CCG_SYNTAX.parse(ccg_store, "s")
    )
    assert TYPES.render(church_store, TYPES.parse(church_store, arrows)) == arrows


def test_records_far_deeper_than_the_recursion_limit_compare_and_publish() -> None:
    store = TermStore(CHURCH_SIGNATURE)
    depth = 4 * getrecursionlimit()
    arrow = CHURCH_SIGNATURE.by_name("->")
    term = store.op(CHURCH_SIGNATURE.by_name("t"))

    for _ in range(depth):
        term = store.op(arrow, store.op(CHURCH_SIGNATURE.by_name("e")), term)

    record = to_record(store, term)
    again = to_record(TermStore(CHURCH_SIGNATURE), from_record(store, record))
    published = record_from_json(record_to_json(record))

    assert record == again == published
    assert record_differences(record, published) == ()


@pytest.mark.parametrize(
    ("value", "error", "message"),
    [
        ([], TypeError, "not a JSON object"),
        ({"operator": "f", "arguments": [3]}, TypeError, "not a JSON object"),
        ({"operator": 3}, ValueError, "operator node is malformed"),
        ({"operator": "f", "extra": 1}, ValueError, "operator node is malformed"),
        ({"constant": "a"}, ValueError, "one kind with a sort"),
        ({"constant": "a", "sort": 3}, ValueError, "one kind with a sort"),
        ({"constant": "a", "free": "b", "sort": "S"}, ValueError, "one kind"),
        ({"bound": True, "sort": "S"}, ValueError, "leaf is malformed"),
        ({"bound": "0", "sort": "S"}, ValueError, "leaf is malformed"),
        ({"constant": 1, "sort": "S"}, ValueError, "leaf is malformed"),
        ({"other": "a", "sort": "S"}, ValueError, "leaf is malformed"),
        ({"bound": -1, "sort": "S"}, ValueError, "index"),
    ],
)
def test_record_json_refuses_malformed_forms(
    value: JsonValue,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        record_from_json(value)


def test_a_record_naming_what_the_store_lacks_is_refused() -> None:
    store = TermStore(CHURCH_SIGNATURE)

    with pytest.raises(UnknownConstantError):
        from_record(store, ConstantRecord("Tm", "snore"))

    with pytest.raises(UnknownOperatorError):
        from_record(store, OperatorRecord("missing"))

    with pytest.raises(UnknownOperatorError):
        from_record(store, ConstantRecord("Ty", "e"))
