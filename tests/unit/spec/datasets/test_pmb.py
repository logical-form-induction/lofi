from pathlib import Path
from typing import Final

import pytest

from lofi.data.corpora.pmb import PmbCcgCodec, PmbLeaf, PmbTree
from lofi.spec.datasets import PmbGrammarLanguage, pmb_grammar
from lofi.spec.grammar import GrammarAssembly, GrammarIso
from lofi.spec.language import registered_language

FIXTURES: Final = Path(__file__).parents[2] / "data" / "corpora" / "pmb" / "fixtures"
LANGUAGES: Final[tuple[PmbGrammarLanguage, ...]] = ("en", "de", "it", "nl")


@pytest.mark.parametrize("language", LANGUAGES)
def test_each_packaged_pmb_grammar_compiles(language: PmbGrammarLanguage) -> None:
    declaration = pmb_grammar(language)
    grammar = GrammarIso().forward(
        declaration,
        registered_language("propositional"),
    )

    assert declaration.name == f"pmb-{language}"
    assert grammar.syntax.parse(grammar.store, "s:dcl\\np")


def _recover(tree: PmbLeaf | PmbTree, grammar: GrammarAssembly) -> None:
    if isinstance(tree, PmbLeaf):
        return

    inputs = tuple(
        grammar.syntax.parse(grammar.store, child.category)
        for child in tree.children
        if isinstance(child, PmbLeaf | PmbTree)
    )
    output = grammar.syntax.parse(grammar.store, tree.category)

    assert grammar.grammar.rules.recover(inputs, output), (
        tree.rule,
        tuple(child.category for child in tree.children),
        tree.category,
    )

    for child in tree.children:
        if isinstance(child, PmbLeaf | PmbTree):
            _recover(child, grammar)


def test_the_english_grammar_reaches_the_repository_pmb_derivation() -> None:
    declaration = pmb_grammar("en")
    grammar = GrammarIso().forward(
        declaration,
        registered_language("propositional"),
    )
    parsed = PmbCcgCodec().iso.backward(
        PmbCcgCodec().decode((FIXTURES / "pawnshop.tags").read_text(encoding="utf-8"))
    )

    for derivation in parsed:
        _recover(derivation.tree, grammar)
