"""Installed-wheel smoke test for the native symbolic parser."""

from typing import TYPE_CHECKING, Final

from lofi.deduce import (
    Counting,
    ParsePolicy,
    Reachability,
    Retain,
    RootGoal,
    UnitEdges,
    parse,
)
from lofi.lexicon import Entry
from lofi.spec import (
    ConstantPatternSpec,
    ConstructorSpec,
    InfixConstructorNotationSpec,
    PatternRuleSpec,
    RuleSetSpec,
)
from lofi.spec.grammar import (
    GrammarDecl,
    GrammarIso,
    InlineLexiconSpec,
    LexicalEntrySpec,
    SignatureSpec,
)
from lofi.spec.language import registered_language
from lofi.terms import TargetSet, Term

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from lofi.deduce import Cell
    from lofi.judge import Rule
    from lofi.lexicon import Clamp


GRAMMAR: Final = GrammarDecl(
    name="wheel-smoke",
    signature=SignatureSpec(
        name="wheel-smoke",
        primitives=("S", "A", "B"),
        constructors=(
            ConstructorSpec(
                name="/",
                kind=("Cat", "Cat", "Cat"),
                notation=InfixConstructorNotationSpec(
                    spelling="/", precedence=10, associativity="left"
                ),
            ),
        ),
    ),
    rules=RuleSetSpec(
        name="wheel-smoke",
        rules=(
            PatternRuleSpec(
                name="combine",
                premises=(ConstantPatternSpec(name="A"), ConstantPatternSpec(name="B")),
                conclusion=ConstantPatternSpec(name="S"),
                head_policy="non-absorbing",
            ),
        ),
    ),
    lexicon=InlineLexiconSpec(entries=(LexicalEntrySpec(form="s", category="S"),)),
)
ADOPTED: Final = GrammarIso().rules(GRAMMAR, registered_language("propositional"))
RULES: Final = ADOPTED.rule_set
MATCHING: Final = RULES.matching
A: Final = ADOPTED.context.syntax.parse(ADOPTED.context.store, "A")
B: Final = ADOPTED.context.syntax.parse(ADOPTED.context.store, "B")
S: Final = ADOPTED.context.syntax.parse(ADOPTED.context.store, "S")


class _Source:
    """Supply the two lexical axioms for the smoke sentence."""

    @property
    def is_re_callable(self) -> bool:
        """Return whether the source can be enumerated again."""
        return True

    @property
    def clamps(self) -> tuple[Clamp[Term], ...]:
        """Return no lexical clamps."""
        return ()

    def axioms(
        self, tokens: Sequence[str], /
    ) -> Iterable[tuple[int, Entry[Term], int]]:
        """Return one unit-weight category for each declared token."""
        if tuple(tokens) != ("a", "b"):
            return ()

        return (
            (0, Entry("a", A), 1),
            (1, Entry("b", B), 1),
        )


class _Scorer:
    """Assign unit weight to every lexical and rule edge."""

    @property
    def lowering(self) -> UnitEdges:
        """Select the native unit-edge scorer."""
        return UnitEdges()

    def axiom_weight(self, position: int, entry: Entry[Term], /) -> int:
        """Return the multiplicative identity for a lexical edge."""
        del position, entry
        return 1

    def edge_weight(
        self,
        rule: Rule[Term],
        inputs: tuple[Term, ...],
        output: Term,
        span: Cell,
        split: int | None,
        /,
    ) -> int:
        """Return the multiplicative identity for a rule edge."""
        del rule, inputs, output, span, split
        return 1


def test_the_installed_wheel_parses_through_the_native_chart() -> None:
    """Build a spanning derivation using only the wheel's public interface."""
    chart = parse(
        ("a", "b"),
        rules=RULES,
        source=_Source(),
        semiring=Counting(),
        scorer=_Scorer(),
        goal=RootGoal(TargetSet((S,), MATCHING)),
        unary_depth=0,
        policy=ParsePolicy(Retain.FOREST, Reachability.ALL),
    )

    roots = tuple(chart.items_at((0, 2)))

    assert tuple(root.reading for root in roots) == (S,)
    assert chart.inside(roots[0]) == 1
    assert tuple(edge.rule_name for edge in chart.edges(roots[0])) == ("combine",)
