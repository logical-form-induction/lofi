from lofi.spec.grammar import (
    CFGFormalismSpec,
    CFGProductionSpec,
    GrammarSpec,
    InlineLexiconSpec,
    LexicalEntrySpec,
)


def test_a_formalism_pack_is_one_open_grammar_declaration() -> None:
    declaration = CFGFormalismSpec(
        nonterminals=("S",),
        productions=(CFGProductionSpec(name="unary", parent="S", children=("S",)),),
        lexicon=InlineLexiconSpec(
            entries=(LexicalEntrySpec(form="token", category="S"),)
        ),
        goal="S",
    )

    assert isinstance(declaration, GrammarSpec)
    assert declaration.formalism == "cfg"
    assert declaration.lexicon.entries == (
        LexicalEntrySpec(form="token", category="S"),
    )
