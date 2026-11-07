import pytest

from lofi.grammars.ccg import FunctorView
from lofi.spec import ApplicationSpec, RuleSetSpec, TypeRaisingSpec
from lofi.spec._rule_laws import LAW_ATOMS, LawGrammar, declared_rules
from lofi.spec.grammar import GrammarIso, IdentityMatchingSpec, SignatureSpec
from lofi.spec.language import registered_language


def test_every_rule_a_law_grammar_compiles_shares_its_term_store() -> None:
    grammar = LawGrammar("direct")
    application = grammar.rule(ApplicationSpec(view="fwd"))
    raising = grammar.rule(TypeRaisingSpec(view="fwd", targets=("t",), arguments=None))
    functors, slash = grammar.functor("forward")
    principal = functors.make(slash, grammar.atom("x"), grammar.atom("y"))

    assert tuple(application.apply((principal, grammar.atom("y")))) == (
        grammar.atom("x"),
    )
    assert raising.admits(
        (grammar.atom("y"),),
        functors.make(
            slash,
            grammar.atom("t"),
            functors.make(
                functors.backward_slash, grammar.atom("t"), grammar.atom("y")
            ),
        ),
    )


def test_the_arrangements_are_the_declared_functor_views() -> None:
    swapped = LawGrammar("swapped")
    undirected = LawGrammar("undirected")
    functors, slash = swapped.functor("backward")
    built = functors.make(slash, swapped.atom("x"), swapped.atom("y"))

    assert type(functors) is FunctorView
    assert (functors.result_path, functors.argument_path) == ((1,), (0,))
    assert built.args == (swapped.atom("y"), swapped.atom("x"))
    assert undirected.functor("forward") == undirected.functor("backward")
    assert undirected.view_of("arrow") == "arrow"
    assert swapped.view_of("bwd") == "bwd"
    assert {atom: swapped.atom(atom) for atom in LAW_ATOMS}.keys() == set(LAW_ATOMS)


def test_a_slash_outside_the_arrangement_names_no_view() -> None:
    with pytest.raises(KeyError):
        LawGrammar("undirected").view_of("fwd")


def test_a_rule_grammar_needs_a_primitive_to_anchor_its_lexicon() -> None:
    with pytest.raises(ValueError, match="needs a primitive"):
        declared_rules(
            GrammarIso(),
            registered_language("propositional"),
            SignatureSpec(syntax="s-expression"),
            RuleSetSpec(name="empty", rules=()),
            matching=IdentityMatchingSpec(),
        )
