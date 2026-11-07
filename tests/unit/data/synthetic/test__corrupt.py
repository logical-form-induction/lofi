"""Tests for installed corruption operators."""

from collections import Counter
from random import Random
from typing import TYPE_CHECKING, Final

import pytest
from tests.unit.spec.grammar.test__signature import slash_views, slashes

from lofi.data.synthetic._corrupt import (
    DELETION_PLUGIN,
    INSERTION_PLUGIN,
    SUBSTITUTION_PLUGIN,
    SWAP_PLUGIN,
    CorruptionContext,
    CorruptionOperator,
)
from lofi.data.synthetic.spec.operators import (
    DeletionSpec,
    InsertionSpec,
    SubstitutionSpec,
    SwapSpec,
)
from lofi.spec import ApplicationSpec, RuleSetSpec
from lofi.spec.grammar import (
    BudgetSpec,
    ConjunctionPairSpec,
    ConjunctionsSpec,
    FeatureErasureMatchingSpec,
    GrammarDecl,
    GrammarIso,
    InlineLexiconSpec,
    LexicalEntrySpec,
    OperatorSpec,
    RootGoalSpec,
    SignatureSpec,
)
from lofi.spec.language import registered_language

if TYPE_CHECKING:
    from lofi.spec.grammar import GrammarAssembly

CONJUNCTIONS: Final = ConjunctionsSpec(
    markers=("conj",),
    pairs=(ConjunctionPairSpec(unmarked="NP", marked="NP[conj]"),),
)
ENTRIES: Final = (
    ("a", "NP"),
    ("b", "S\\NP"),
    ("c", "(S\\NP)/NP"),
    ("d", "NP/NP"),
    ("e", "NP"),
    ("e", "NP[conj]"),
)


def _assembly() -> GrammarAssembly:
    declaration = GrammarDecl(
        name="operators",
        signature=SignatureSpec(
            constructors=slashes(),
            functors=slash_views(),
            primitives=("S", "NP", "conj"),
            operators=(OperatorSpec(name="NP[conj]", sort="Cat", erases_to="NP"),),
            conjunctions=CONJUNCTIONS,
        ),
        rules=RuleSetSpec(
            name="application",
            rules=(ApplicationSpec(view="/"), ApplicationSpec(view="\\")),
        ),
        lexicon=InlineLexiconSpec(
            entries=tuple(
                LexicalEntrySpec(form=form, category=category)
                for form, category in ENTRIES
            )
        ),
        goal=RootGoalSpec(categories=("S",)),
        matching=FeatureErasureMatchingSpec(),
        budget=BudgetSpec(unary_rounds=1),
    )
    return GrammarIso().forward(declaration, registered_language("propositional"))


@pytest.fixture(scope="module")
def context() -> CorruptionContext:
    assembly = _assembly()
    return CorruptionContext(assembly.lexicon, assembly.grammar.matching, 4)


def _draws(
    operator: CorruptionOperator,
    forms: tuple[str, ...],
    seed: int = 0,
    count: int = 200,
) -> Counter[tuple[str, ...] | None]:
    random = Random(seed)
    return Counter(operator.apply(random, forms) for _ in range(count))


def test_a_swap_exchanges_adjacent_distinct_forms(context: CorruptionContext) -> None:
    operator = SWAP_PLUGIN.compile(SwapSpec())(context)
    drawn = _draws(operator, ("a", "c", "a", "a"))

    assert set(drawn) == {("c", "a", "a", "a"), ("a", "a", "c", "a")}
    assert min(drawn.values()) > 50
    assert set(_draws(operator, ("a", "a", "a"))) == {None}


def test_a_deletion_applies_the_declared_number_of_edits(
    context: CorruptionContext,
) -> None:
    operator = DELETION_PLUGIN.compile(DeletionSpec(edits=2))(context)
    drawn = _draws(operator, ("a", "b", "c"))

    assert set(drawn) == {("a",), ("b",), ("c",)}
    assert set(_draws(operator, ("a",))) == {None}


def test_an_insertion_draws_a_lexicon_form_below_the_bound(
    context: CorruptionContext,
) -> None:
    operator = INSERTION_PLUGIN.compile(InsertionSpec())(context)
    drawn = _draws(operator, ("a", "b"), count=600)

    assert len(drawn) == 13
    assert min(drawn.values()) > 15
    assert set(_draws(operator, ("a", "b", "c", "d"))) == {None}


def test_a_substitution_uses_category_structure_from_the_matching(
    context: CorruptionContext,
) -> None:
    operator = SUBSTITUTION_PLUGIN.compile(SubstitutionSpec())(context)
    drawn = _draws(operator, ("a", "b"), count=400)

    assert set(drawn) == {
        ("b", "b"),
        ("c", "b"),
        ("d", "b"),
        ("a", "a"),
        ("a", "c"),
        ("a", "d"),
        ("a", "e"),
    }
    assert min(drawn.values()) > 20


def test_the_context_validates_the_length_bound() -> None:
    assembly = _assembly()
    with pytest.raises(ValueError, match="max_length is negative"):
        CorruptionContext(assembly.lexicon, assembly.grammar.matching, -1)
