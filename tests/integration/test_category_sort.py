"""Syntactic categories in the learner language's one declared store."""

from lofi.data.synthetic.spec.corpus import (
    SyntheticCorpusDecl,
    SyntheticCorpusLowerer,
    UniformLengthSpec,
)
from lofi.spec import ConstructorSpec, PrefixConstructorNotationSpec, RuleSetSpec
from lofi.spec.grammar import (
    GrammarDecl,
    InlineLexiconSpec,
    LexicalEntrySpec,
    OperatorSpec,
    RootGoalSpec,
    SignatureSpec,
)
from lofi.spec.language import SortBoundSpec, registered_language


def test_categories_are_constants_of_the_declared_language_sort() -> None:
    """Lower grammar categories into the exact language store published by data."""
    base = registered_language("propositional")
    language = base.with_(
        bounds={
            **base.bounds,
            "Cat": SortBoundSpec(layout="complete-tree", depth=4),
        }
    )
    declaration = SyntheticCorpusDecl(
        retained={"validation": 0.25},
        language=language,
        grammar=GrammarDecl(
            name="category-sort",
            signature=SignatureSpec(
                primitives=("S", "NP"),
                operators=(OperatorSpec(name="S[dcl]", sort="Cat", erases_to="S"),),
                constructors=(
                    ConstructorSpec(
                        name="box",
                        kind=("Cat", "Cat"),
                        notation=PrefixConstructorNotationSpec(spelling="box"),
                    ),
                ),
                syntax="s-expression",
            ),
            rules=RuleSetSpec(name="empty"),
            lexicon=InlineLexiconSpec(
                entries=(
                    LexicalEntrySpec(form="sentence", category='("#const[Cat]" "S")'),
                    LexicalEntrySpec(form="noun", category='("#const[Cat]" "NP")'),
                    LexicalEntrySpec(
                        form="declarative", category='("#const[Cat]" "S[dcl]")'
                    ),
                )
            ),
            goal=RootGoalSpec(categories=('("#const[Cat]" "S")',)),
        ),
        size=0,
        seed=1,
        length=UniformLengthSpec(min_length=1, max_length=1),
    )

    grammar = SyntheticCorpusLowerer().check(declaration).source_grammar
    lowered = grammar.language
    sentence = grammar.syntax.parse(grammar.store, '("#const[Cat]" "S")')
    declarative = grammar.syntax.parse(grammar.store, '("#const[Cat]" "S[dcl]")')

    assert grammar.store is lowered.store
    assert lowered.store.signature.sorts == ("Tm", "Ty", "Cat")
    assert sentence is lowered.store.const("Cat", "S")
    assert declarative.core is sentence
    assert sentence.op.payload_sort == "constant"
    assert lowered.codec.at("Cat") is lowered.codec.at(lowered.category_sort)
