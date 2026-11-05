from lofi.spec import ConstantPatternSpec, PatternRuleSpec, RuleSetSpec
from lofi.spec.grammar import (
    FormalismPlugin,
    GrammarDecl,
    InlineLexiconSpec,
    LexicalEntrySpec,
    SignatureSpec,
    declare_grammar,
)


def _declaration() -> GrammarDecl:
    return GrammarDecl(
        name="declared",
        signature=SignatureSpec(name="cfg", primitives=("S",), syntax="s-expression"),
        rules=RuleSetSpec(
            name="identity",
            rules=(
                PatternRuleSpec(
                    name="identity",
                    premises=(ConstantPatternSpec(name="S"),),
                    conclusion=ConstantPatternSpec(name="S"),
                    head_policy="pass-through",
                ),
            ),
        ),
        lexicon=InlineLexiconSpec(
            entries=(LexicalEntrySpec(form="sentence", category="S"),)
        ),
    )


def test_a_generic_declaration_is_already_at_the_formalism_boundary() -> None:
    declaration = _declaration()

    assert declare_grammar(declaration) is declaration


def test_a_formalism_plugin_exposes_and_runs_its_declared_expansion() -> None:
    declaration = _declaration()
    plugin = FormalismPlugin(spec=GrammarDecl, declare=lambda source: source)

    assert plugin.spec is GrammarDecl
    assert plugin.construct(declaration) is declaration
