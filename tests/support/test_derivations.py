from tests.support.derivations import FormLexicon, axiom, step
from tests.support.rules import declared_rules

from lofi.interpret import axiom_entries, is_axiom, step_context
from lofi.lexicon import Entry
from lofi.spec import (
    ConstantPatternSpec,
    PatternRuleSpec,
    RuleSetSpec,
    TermPatternSpec,
)
from lofi.spec.grammar import SignatureSpec

RULES = declared_rules(
    SignatureSpec(name="support-derivations", primitives=("t",), syntax="s-expression"),
    RuleSetSpec(
        name="support-derivations",
        rules=(
            PatternRuleSpec(
                name="pair",
                premises=(
                    TermPatternSpec(name="X", sort="Cat"),
                    TermPatternSpec(name="Y", sort="Cat"),
                ),
                conclusion=ConstantPatternSpec(name="t"),
            ),
        ),
    ),
)
STORE = RULES.context.store
T = STORE.const("Cat", "t")
(PAIR,) = RULES.rule_set.rules


def test_a_built_step_spans_its_axioms_and_reads_through_the_form_lexicon() -> None:
    left = Entry("a", T)
    right = Entry("b", T)
    lexicon = FormLexicon((PAIR,), (left, right, left))
    tree = step(PAIR, (axiom(left), axiom(right, 1)), T, 0.5, 2)

    assert tree.key.span == (0, 2)
    assert tree.edge.split == 1
    assert not is_axiom(tree)
    assert is_axiom(tree.children[1])
    assert axiom_entries(tree.children[0], lexicon) == (left,)
    assert step_context(tree, lexicon).order == 2
