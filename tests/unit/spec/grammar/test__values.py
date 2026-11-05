import pickle
from copy import copy, deepcopy
from typing import Final

import pytest
from tests.support.term_reference import infix_syntax

from lofi._types import HeadSelection
from lofi.spec import (
    ConstantPatternSpec,
    PatternRuleSpec,
    RuleSetSpec,
)
from lofi.spec.grammar import (
    GRAMMAR_SOURCE_VERSION,
    GRAMMAR_TARGET_VERSION,
    BudgetSpec,
    CategoryMeasure,
    GrammarAssembly,
    GrammarDecl,
    GrammarIso,
    InlineLexiconSpec,
    LexicalEntrySpec,
    RootGoalSpec,
    SignatureSpec,
)
from lofi.spec.language import registered_language
from lofi.terms import Operator, Signature, TermStore, Vocabulary


def _node(name: str) -> ConstantPatternSpec:
    return ConstantPatternSpec(name=name)


DECLARATION: Final = GrammarDecl(
    name="values",
    signature=SignatureSpec(
        name="cfg", primitives=("S", "NP", "VP"), syntax="s-expression"
    ),
    rules=RuleSetSpec(
        name="productions",
        rules=(
            PatternRuleSpec(
                name="s",
                premises=(_node("NP"), _node("VP")),
                conclusion=_node("S"),
                head_policy="pass-through",
            ),
        ),
    ),
    lexicon=InlineLexiconSpec(
        entries=(
            LexicalEntrySpec(form="john", category='("#const[Cat]" "NP")'),
            LexicalEntrySpec(form="sleeps", category='("#const[Cat]" "VP")'),
        )
    ),
    goal=RootGoalSpec(categories=('("#const[Cat]" "S")',)),
    budget=BudgetSpec(unary_rounds=0),
)


def _assembly(declaration: GrammarDecl = DECLARATION) -> GrammarAssembly:
    return GrammarIso().forward(declaration, registered_language("propositional"))


def test_an_assembly_exposes_one_compiled_native_grammar() -> None:
    assembly = _assembly()

    assert assembly.syntax.signature is assembly.store.signature
    assert [rule.name for rule in assembly.grammar.rules] == ["s"]
    assert (
        assembly.grammar.theory_version
        == f"{GRAMMAR_SOURCE_VERSION}->{GRAMMAR_TARGET_VERSION}"
    )
    assert copy(assembly) is assembly
    assert deepcopy(assembly) is assembly


def test_an_assembly_exposes_each_compiled_rule_head_selection() -> None:
    selections = _assembly().head_selections

    assert dict(selections) == {"s": HeadSelection("pass-through", None)}


def test_an_assembly_is_equal_by_declaration_and_fingerprint() -> None:
    first = _assembly()
    second = _assembly()
    renamed = _assembly(DECLARATION.with_(name="renamed"))

    assert first is not second
    assert first == second
    assert hash(first) == hash(second)
    assert first != renamed
    assert first != DECLARATION
    assert first.declaration_digest != renamed.declaration_digest
    assert first.grammar.fingerprint != renamed.grammar.fingerprint
    assert repr(first).startswith("GrammarAssembly('values', declaration=")

    forged = GrammarAssembly(
        first.declaration,
        renamed.grammar,
        first.lexicon,
        first.syntax,
        first.language,
        first.compile_report,
        category_measure=first.category_measure,
        head_selections=first.head_selections,
        max_category_arguments=first.max_category_arguments,
        max_category_order=first.max_category_order,
        source_version=GRAMMAR_SOURCE_VERSION,
        target_version=GRAMMAR_TARGET_VERSION,
    )

    assert forged != first
    assert forged.declaration_digest == first.declaration_digest


def test_an_assembly_holds_an_immutable_category_measure() -> None:
    built = _assembly()
    measure = CategoryMeasure((("fwd", 0), ("bwd", 0)))
    assembly = GrammarAssembly(
        built.declaration,
        built.grammar,
        built.lexicon,
        built.syntax,
        built.language,
        built.compile_report,
        category_measure=measure,
        head_selections=built.head_selections,
        max_category_arguments=None,
        max_category_order=None,
        source_version=GRAMMAR_SOURCE_VERSION,
        target_version=GRAMMAR_TARGET_VERSION,
    )
    assert assembly.category_measure is measure
    assert assembly.category_measure.constructors == (("fwd", 0), ("bwd", 0))
    assert assembly.max_category_arguments is None
    assert assembly.max_category_order is None
    assert built.category_measure.constructors == ()


def test_a_category_measure_reports_depth_argument_count_and_order() -> None:
    """The three inventory factors distinguish depth, arity and higher order."""
    signature = Signature(
        "categories",
        ("Cat",),
        (
            Operator("/", "Cat", arg_sorts=("Cat", "Cat"), family="slash"),
            Operator("\\", "Cat", arg_sorts=("Cat", "Cat"), family="slash"),
        ),
        vocabularies=(Vocabulary("Cat", ("S", "NP")),),
    )
    store = TermStore(signature)
    syntax = infix_syntax(signature, ("/", "/"), ("\\", "\\"))
    measure = CategoryMeasure((("/", 0), ("\\", 0)))
    categories = {
        text: syntax.parse(store, text)
        for text in ("NP", "S\\NP", "(S\\NP)/NP", "S/(S\\NP)")
    }

    assert measure.depth(categories["NP"]) == 0
    assert measure.depth(categories["(S\\NP)/NP"]) == 2
    assert measure.arity(categories["NP"]) == 0
    assert measure.arity(categories["S\\NP"]) == 1
    assert measure.arity(categories["(S\\NP)/NP"]) == 2
    assert measure.order(categories["S\\NP"]) == 1
    assert measure.order(categories["(S\\NP)/NP"]) == 1
    assert measure.order(categories["S/(S\\NP)"]) == 2


def test_an_assembly_refuses_change_and_copies_to_itself() -> None:
    assembly = _assembly()

    assert copy(assembly) is assembly
    assert deepcopy(assembly) is assembly
    assert assembly.store is assembly.grammar.store

    with pytest.raises(AttributeError, match="immutable"):
        assembly.__setattr__("_language", None)

    with pytest.raises(AttributeError, match="immutable"):
        assembly.__delattr__("_language")

    with pytest.raises(TypeError, match="cannot be pickled"):
        pickle.dumps(assembly)
