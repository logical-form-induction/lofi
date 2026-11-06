"""Benchmarks the chart on generalized composition, features and many coercions.

The grammar has the shape of a treebank-derived CCG: feature-bearing
primitives under feature-erasure matching, application, harmonic and crossed
composition at orders one and two in both directions, crossed substitution at
order one, 190 unary coercions drawn from a fixed seed, and two rounds of
unary closure. The benchmarks time the inside-only parse under the
Hockenmaier-Bisk normal form, which is what a reader of supports and counts
runs; the untimed tests beside them check that this parse keeps every
category the unrestricted chart derives on each sentence suffix.
"""

from functools import cache
from random import Random
from typing import TYPE_CHECKING

import pytest

from lofi.deduce import (
    Counting,
    PackedForest,
    ParsePolicy,
    Reachability,
    Retain,
    UnitEdges,
    parse,
)
from lofi.spec import (
    ApplicationSpec,
    CompositionSpec,
    ConstantPatternSpec,
    ConstructorSpec,
    FixedOperatorSpec,
    InfixConstructorNotationSpec,
    NodePatternSpec,
    PatternRuleSpec,
    RuleSetSpec,
    SubstitutionSpec,
)
from lofi.spec.grammar import (
    BudgetSpec,
    FeatureErasureMatchingSpec,
    FunctorSpec,
    GrammarDecl,
    GrammarIso,
    HockenmaierBiskNormalFormSpec,
    InlineLexiconSpec,
    LexicalEntrySpec,
    NoNormalFormSpec,
    NormalFormSpec,
    OperatorSpec,
    RootGoalSpec,
    SignatureSpec,
)
from lofi.spec.language import registered_language

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from pytest_benchmark.fixture import BenchmarkFixture
    from tests.benchmarks.conftest import WallClockBudget

    from lofi.deduce import Cell, InsideChart
    from lofi.judge import Rule
    from lofi.lexicon import Entry, LexicalSource
    from lofi.spec.grammar import GrammarAssembly
    from lofi.terms import Term

type _Category = str | tuple[str, _Category, _Category]
"""A category: a primitive, or a slash over a result and an argument."""

_ATOMS = ("S", "NP", "N", "PP")
_FEATURED = ("S[dcl]", "S[b]", "S[ng]", "NP[nb]")

_LEXICON = {
    "the": ("NP[nb]/N", "NP/N"),
    "a": ("NP[nb]/N",),
    "dog": ("N",),
    "cat": ("N",),
    "man": ("N",),
    "park": ("N",),
    "old": ("N/N",),
    "big": ("N/N",),
    "sees": ("(S[dcl]\\NP)/NP",),
    "see": ("(S[b]\\NP)/NP",),
    "seeing": ("(S[ng]\\NP)/NP", "N"),
    "will": ("(S[dcl]\\NP)/(S[b]\\NP)",),
    "is": ("(S[dcl]\\NP)/(S[ng]\\NP)", "(S[dcl]\\NP)/NP"),
    "in": ("(NP\\NP)/NP", "((S\\NP)\\(S\\NP))/NP", "PP/NP"),
    "with": ("(NP\\NP)/NP", "((S\\NP)\\(S\\NP))/NP"),
    "near": ("(NP\\NP)/NP", "PP/NP"),
    "quickly": ("(S\\NP)\\(S\\NP)",),
}

_SENTENCES = {
    16: "the dog will see a big cat in the park near the man with a dog",
    17: "the dog is seeing a big cat in the park near the man with a dog quickly",
}


def _text(category: _Category) -> str:
    """Spell a category in infix syntax."""
    if isinstance(category, str):
        return category

    slash, result, argument = category

    return f"({_text(result)}){slash}({_text(argument)})"


def _pattern(category: _Category) -> NodePatternSpec | ConstantPatternSpec:
    """Return the pattern matching exactly one category."""
    if isinstance(category, str):
        return ConstantPatternSpec(name=category)

    slash, result, argument = category

    return NodePatternSpec(
        operator=FixedOperatorSpec(name=slash),
        arguments=(_pattern(result), _pattern(argument)),
    )


def _common() -> tuple[_Category, ...]:
    """Return primitives, functors over two primitives and verb-phrase functors."""
    verb_phrase = ("\\", "S", "NP")
    modifier = ("\\", "NP", "NP")

    return (
        *_ATOMS,
        *(
            (slash, result, argument)
            for slash in "/\\"
            for result in _ATOMS
            for argument in _ATOMS
        ),
        *((slash, verb_phrase, argument) for slash in "/\\" for argument in _ATOMS),
        *((slash, verb_phrase, verb_phrase) for slash in "/\\"),
        *((slash, modifier, argument) for slash in "/\\" for argument in _ATOMS),
    )


def _conclusions(common: Sequence[_Category]) -> tuple[_Category, ...]:
    """Return coercion conclusions, featured verb phrases among them."""
    return (
        *common,
        *(("\\", featured, "NP") for featured in _FEATURED[:3]),
        "NP[nb]",
    )


def _rare(common: Sequence[_Category]) -> tuple[_Category, ...]:
    """Return three-argument functors that only long compositions reach."""
    pairs = tuple(category for category in common if not isinstance(category, str))[:8]

    return tuple(
        (outer, (inner, pair, atom), argument)
        for pair in pairs
        for inner in "/\\"
        for atom in _ATOMS
        for outer in "/\\"
        for argument in pairs[:4]
    )


def _coercions(
    count: int,
    premises: Sequence[_Category],
    conclusions: Sequence[_Category],
    random: Random,
    taken: set[tuple[_Category, _Category]],
) -> list[tuple[_Category, _Category]]:
    """Draw distinct coercions between two category pools."""
    drawn: list[tuple[_Category, _Category]] = []

    while len(drawn) < count:
        pair = (random.choice(premises), random.choice(conclusions))

        if pair[0] != pair[1] and pair not in taken:
            taken.add(pair)
            drawn.append(pair)

    return drawn


def declaration(normal_form: NormalFormSpec) -> GrammarDecl:
    """Return the coercion grammar under one normal form."""
    random = Random(0)
    common = _common()
    taken: set[tuple[_Category, _Category]] = set()
    coercions = (
        *_coercions(100, common, _conclusions(common), random, taken),
        *_coercions(90, _rare(common), common, random, taken),
    )
    rules = [
        *(ApplicationSpec(view=slash) for slash in "/\\"),
        *(
            CompositionSpec(
                order=order, view=slash, crossed=False, direction="unrestricted"
            )
            for order in (1, 2)
            for slash in "/\\"
        ),
        *(SubstitutionSpec(order=1, view=slash, crossed=True) for slash in "/\\"),
        *(
            PatternRuleSpec(
                name=f"coercion-{at}",
                premises=(_pattern(premise),),
                conclusion=_pattern(conclusion),
            )
            for at, (premise, conclusion) in enumerate(coercions)
        ),
    ]

    return GrammarDecl(
        name="ccg-coercions",
        signature=SignatureSpec(
            name="ccg-coercions",
            primitives=_ATOMS,
            operators=tuple(
                OperatorSpec(name=name, sort="Cat", erases_to=name.split("[")[0])
                for name in _FEATURED
            ),
            constructors=(
                ConstructorSpec(
                    name="/",
                    kind=("Cat", "Cat", "Cat"),
                    notation=InfixConstructorNotationSpec(
                        spelling="/", precedence=50, associativity="left"
                    ),
                ),
                ConstructorSpec(
                    name="\\",
                    kind=("Cat", "Cat", "Cat"),
                    notation=InfixConstructorNotationSpec(
                        spelling="\\", precedence=50, associativity="left"
                    ),
                ),
            ),
            functors=(
                FunctorSpec(
                    name="/",
                    constructor="/",
                    argument_side="right",
                    opposite="\\",
                ),
                FunctorSpec(
                    name="\\",
                    constructor="\\",
                    argument_side="left",
                    opposite="/",
                ),
            ),
        ),
        rules=RuleSetSpec(name="ccg-coercions", rules=tuple(rules)),
        lexicon=InlineLexiconSpec(
            entries=tuple(
                LexicalEntrySpec(form=form, category=category)
                for form, categories in _LEXICON.items()
                for category in categories
            )
        ),
        goal=RootGoalSpec(categories=("S", "S\\NP")),
        matching=FeatureErasureMatchingSpec(),
        normal_form=normal_form,
        budget=BudgetSpec(unary_rounds=2, threads=8),
    )


@cache
def assembly(kind: str) -> GrammarAssembly:
    """Return the grammar lowered under the named normal form."""
    normal_form = (
        HockenmaierBiskNormalFormSpec()
        if kind == "hockenmaier-bisk"
        else NoNormalFormSpec()
    )

    return GrammarIso().forward(
        declaration(normal_form), registered_language("propositional")
    )


class Unit:
    """Counts every derivation once: unit axioms and unit edges."""

    __slots__ = ("_source",)
    lowering = UnitEdges()

    def __init__(self, source: LexicalSource[Term, float]) -> None:
        self._source = source

    @property
    def is_re_callable(self) -> bool:
        """Return whether the wrapped source can be enumerated again."""
        return self._source.is_re_callable

    @property
    def clamps(self) -> tuple[()]:
        """Return no clamps."""
        return ()

    def axioms(
        self, tokens: Sequence[str], /
    ) -> Iterable[tuple[int, Entry[Term], int]]:
        """Return the wrapped source's axioms at unit weight."""
        return tuple(
            (position, entry, 1) for position, entry, _ in self._source.axioms(tokens)
        )

    def axiom_weight(self, position: int, entry: Entry[Term], /) -> int:
        """Return one."""
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
        """Return one."""
        del rule, inputs, output, span, split

        return 1


def supports(kind: str, tokens: tuple[str, ...]) -> InsideChart[Term, int]:
    """Count the normal-form derivations of every exact category of one sentence."""
    lowered = assembly(kind)
    unit = Unit(lowered.lexicon)

    return parse(
        tokens,
        rules=lowered.grammar.rules,
        source=unit,
        semiring=Counting(),
        scorer=unit,
        goal=lowered.grammar.goal,
        category_measure=lowered.category_measure,
        max_category_arity=lowered.grammar.budget.max_category_arity,
        max_category_size=lowered.grammar.budget.max_category_size,
        unary_depth=lowered.grammar.budget.unary_rounds,
        threads=lowered.grammar.budget.threads,
        policy=ParsePolicy(Retain.INSIDE, Reachability.ALL),
    )


def forest(kind: str, tokens: tuple[str, ...]) -> PackedForest[Term, int]:
    """Count the derivations of one sentence into a packed forest."""
    lowered = assembly(kind)
    unit = Unit(lowered.lexicon)

    return parse(
        tokens,
        rules=lowered.grammar.rules,
        source=unit,
        semiring=Counting(),
        scorer=unit,
        goal=lowered.grammar.goal,
        category_measure=lowered.category_measure,
        max_category_arity=lowered.grammar.budget.max_category_arity,
        max_category_size=lowered.grammar.budget.max_category_size,
        unary_depth=lowered.grammar.budget.unary_rounds,
        threads=lowered.grammar.budget.threads,
        policy=ParsePolicy(Retain.FOREST, Reachability.ALL),
    )


def suffix_categories(
    chart: PackedForest[Term, int] | InsideChart[Term, int],
) -> frozenset[tuple[Cell, Term]]:
    """Return every exact category over every span ending at the sentence end."""
    end = chart.index.n

    return frozenset(
        ((start, end), category)
        for start in range(end)
        for category, _ in chart.categories_at((start, end))
    )


def test_the_grammar_declares_190_coercions_and_feature_erasure() -> None:
    rules = assembly("hockenmaier-bisk").grammar.rules

    assert sum(rule.name.startswith("coercion-") for rule in rules) == 190
    assert rules.matching.name == "feature-erasure"
    assert rules.normal_form.name == "hockenmaier-bisk"


@pytest.mark.parametrize("length", [6, 8])
def test_the_normal_form_keeps_every_suffix_category_of_a_prefix(length: int) -> None:
    tokens = tuple(_SENTENCES[16].split())[:length]
    unrestricted = forest("none", tokens)
    restricted = supports("hockenmaier-bisk", tokens)

    assert suffix_categories(restricted) == suffix_categories(unrestricted)
    assert suffix_categories(unrestricted)


@pytest.mark.parametrize(
    ("length", "budget"),
    [(16, 35.0), (17, 70.0)],
    ids=("16-tokens", "17-tokens"),
)
def test_a_long_sentence_parses_within_its_budget(
    benchmark: BenchmarkFixture,
    length: int,
    budget: float,
    wall_clock_budget: WallClockBudget,
) -> None:
    tokens = tuple(_SENTENCES[length].split())
    benchmark.pedantic(
        supports, args=("hockenmaier-bisk", tokens), rounds=1, iterations=1
    )
    measured = benchmark.stats

    if measured is None:
        pytest.fail("the parse ran but pytest-benchmark timed nothing")

    benchmark.extra_info["tokens"] = len(tokens)
    benchmark.extra_info["budget_seconds"] = budget

    if wall_clock_budget.enforced:
        assert measured.stats.min <= budget
