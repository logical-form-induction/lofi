"""Rule-law cases for pattern rules."""

from typing import TYPE_CHECKING, Final

from hypothesis import strategies as st

from lofi.judge.laws import (
    RuleFixture,
    RuleLawCase,
    RuleLawCases,
    check_rule_laws,
)
from lofi.spec._rule import (
    ConstantPatternSpec,
    FixedOperatorSpec,
    NodePatternSpec,
    PatternRuleSpec,
    TermPatternSpec,
)
from lofi.spec._rule_laws import LawGrammar

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi._types import JsonValue
    from lofi.judge import Rule
    from lofi.terms import Term

__all__ = ["provide_rule_laws"]

_PAIRING: Final = PatternRuleSpec(
    name="pair-right",
    premises=(TermPatternSpec(name="X", sort="Cat"), ConstantPatternSpec(name="y")),
    conclusion=NodePatternSpec(
        operator=FixedOperatorSpec(name="fwd"),
        arguments=(
            TermPatternSpec(name="X", sort="Cat"),
            ConstantPatternSpec(name="y"),
        ),
    ),
)
"""A binary rule pairing any category with the atom ``y``."""

_UNARY: Final = PatternRuleSpec(
    name="atom-to-phrase",
    premises=(ConstantPatternSpec(name="x"),),
    conclusion=ConstantPatternSpec(name="y"),
)
"""A unary rule rewriting the atom ``x`` to the atom ``y``."""


class _Rebuild:
    """Rebuild one pattern rule from its declaration."""

    __slots__ = ("_declaration", "_grammar", "_spec")

    def __init__(
        self, grammar: LawGrammar, declaration: PatternRuleSpec, rule: Rule[Term], /
    ) -> None:
        self._grammar = grammar
        self._declaration = declaration
        self._spec = rule.to_spec()

    def __call__(self, spec: Mapping[str, JsonValue], /) -> Rule[Term]:
        """Compile the grammar the declaration names.

        Parameters
        ----------
        spec : Mapping[str, JsonValue]
            Rule declaration.

        Raises
        ------
        ValueError
            If the declaration differs from the case's rule.
        """
        if spec != self._spec:
            msg = f"unexpected pattern rule declaration: {spec!r}"
            raise ValueError(msg)

        return self._grammar.rule(self._declaration)


def _case(grammar: LawGrammar, name: str, declaration: PatternRuleSpec) -> RuleLawCase:
    """Build one pattern-rule law case."""
    rule = grammar.rule(declaration)
    functors, slash = grammar.functor("forward")
    x = grammar.atom("x")
    y = grammar.atom("y")
    terms = (x, y, functors.make(slash, x, y), functors.make(slash, y, y))
    fixture = RuleFixture(
        rule, st.sampled_from(terms), _Rebuild(grammar, declaration, rule), 500
    )

    def check() -> None:
        """Run every law for the captured pattern fixture."""
        check_rule_laws(fixture)

    return RuleLawCase(name, rule.name, check)


def provide_rule_laws() -> RuleLawCases:
    """Return the two ordinary pattern-rule cases."""
    grammar = LawGrammar("direct")

    return RuleLawCases(
        "pattern",
        (
            _case(grammar, "pattern/pairing", _PAIRING),
            _case(grammar, "pattern/unary", _UNARY),
        ),
    )
