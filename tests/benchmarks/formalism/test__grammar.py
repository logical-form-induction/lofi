from typing import Final

from tests.support.rules import ccg_signature, declared_rules

from lofi.deduce import RootGoal
from lofi.formalism import Budget, Formalism, FormalismDeclaration, TheoryArtifact
from lofi.judge import ClosureReport, capacity_report
from lofi.spec import ConstantPatternSpec, PatternRuleSpec, RuleSetSpec
from lofi.terms import TargetSet

ADOPTED: Final = declared_rules(
    ccg_signature("A", "B", name="formalism-benchmark"),
    RuleSetSpec(
        name="formalism-benchmark",
        rules=(
            PatternRuleSpec(
                name="a-to-b",
                premises=(ConstantPatternSpec(name="A"),),
                conclusion=ConstantPatternSpec(name="B"),
            ),
        ),
    ),
)
STORE: Final = ADOPTED.context.store
A: Final = ADOPTED.context.syntax.parse(STORE, "A")
B: Final = ADOPTED.context.syntax.parse(STORE, "B")
RULES: Final = ADOPTED.rule_set
MATCHING: Final = RULES.matching
CLOSURE: Final = ClosureReport(
    (A, B),
    saturated=True,
    steps=1,
    budget=2,
)
TARGETS: Final = TargetSet((B,), MATCHING)
DECLARATION: Final = FormalismDeclaration(
    CLOSURE,
    TheoryArtifact.from_specification("v1", b'{"category":"Cat","order":1}'),
    1,
    Budget(
        depth=5,
        mass=1.0,
        beam=8,
        unary_rounds=1,
        threads=1,
        beta_fuel=1_000,
        max_category_arity=None,
        max_category_size=None,
    ),
    capacity_report(RULES),
    lexicalized=True,
)
FORMALISM: Final = Formalism(
    "formalism-benchmark",
    STORE,
    RULES,
    TARGETS,
    RootGoal(TARGETS),
    DECLARATION,
)


def test_a_cached_grammar_compile_returns_the_held_grammar() -> None:
    expected = FORMALISM.compile()

    assert FORMALISM.compile() is expected
