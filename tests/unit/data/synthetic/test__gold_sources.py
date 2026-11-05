"""Gold constraint sources built from their declarations through their plugins."""

from typing import TYPE_CHECKING, Final, cast

import pytest
from tests.conformance.configured_component import (
    ConfiguredComponentConformance,
    ConfiguredComponentFixture,
)

from lofi._types import ItemId
from lofi.data import (
    Example,
    GoldDerivationSource,
    GoldLexiconSource,
    Item,
    SelectionPolicy,
    Token,
    TokenSpan,
    TypeScope,
    UnionRule,
    Unlabeled,
)
from lofi.data.synthetic._gold_sources import (
    ConstraintSourceContext,
    UnservedPolicyError,
)
from lofi.lexicon import Entry, StaticLexicon
from lofi.registry import plugin_for
from lofi.spec.constraint_policy import PolicyDomain
from lofi.spec.constraint_sources import (
    GoldDerivationConstraintSourceSpec,
    GoldLexiconConstraintSourceSpec,
    GoldLogicalFormConstraintSourceSpec,
)
from lofi.terms import Operator, Signature, TermStore

if TYPE_CHECKING:
    from lofi.spec import ConfiguredSpec


SIGNATURE: Final = Signature(
    "constraint-factory-test",
    ("Cat",),
    (Operator("N", "Cat", family="primitive"),),
)
STORE: Final = TermStore(SIGNATURE)
NOUN: Final = STORE.op(SIGNATURE.by_name("N"))
LEXICON: Final = StaticLexicon((Entry("dog", NOUN),))
SYNTACTIC: Final = SelectionPolicy[str, str](UnionRule())
SEMANTIC: Final = SelectionPolicy[str, str](UnionRule(), coverage=0.5)


def context() -> ConstraintSourceContext:
    return ConstraintSourceContext(
        syntactic=SYNTACTIC,
        semantic=SEMANTIC,
        lexicon=LEXICON,
        render=lambda term: term.op.name,
        interpretation=None,
    )


def built(declaration: ConfiguredSpec[object]) -> object:
    return plugin_for(declaration).construct(declaration, context())


def test_a_context_names_each_of_the_runs_policies() -> None:
    held = context()

    assert held.policy(PolicyDomain.SYNTACTIC) is SYNTACTIC
    assert held.policy(PolicyDomain.SEMANTIC) is SEMANTIC
    assert held.lexicon is LEXICON
    assert held.render(NOUN) == "N"
    assert held.interpretation is None


def test_a_context_is_frozen() -> None:
    held = context()

    with pytest.raises(AttributeError, match="immutable"):
        held.__setattr__("_lexicon", LEXICON)


class TestGoldDerivationSourcePlugin(ConfiguredComponentConformance):
    @pytest.fixture
    def configured_component_fixture(self) -> ConfiguredComponentFixture:
        return ConfiguredComponentFixture(
            declaration=cast(
                "ConfiguredSpec[object]",
                GoldDerivationConstraintSourceSpec(policy=PolicyDomain.SYNTACTIC),
            ),
            context=context(),
            expected_type=GoldDerivationSource,
            observe=lambda source: {
                "selects under the declared policy": cast(
                    "GoldDerivationSource[str, str]", source
                ).policy
                is SYNTACTIC
            },
            unchanged=lambda: True,
        )


class TestGoldLexiconSourcePlugin(ConfiguredComponentConformance):
    @pytest.fixture
    def configured_component_fixture(self) -> ConfiguredComponentFixture:
        return ConfiguredComponentFixture(
            declaration=cast(
                "ConfiguredSpec[object]",
                GoldLexiconConstraintSourceSpec(policy=PolicyDomain.SYNTACTIC),
            ),
            context=context(),
            expected_type=GoldLexiconSource,
            observe=lambda source: {
                "selects under the declared policy": cast(
                    "GoldLexiconSource[str, str]", source
                ).policy
                is SYNTACTIC
            },
            unchanged=lambda: tuple(LEXICON.forms) == ("dog",),
        )


def test_the_gold_lexicon_source_renders_terms_and_pins_every_equal_form() -> None:
    source = cast(
        "GoldLexiconSource[str, str]",
        built(
            cast(
                "ConfiguredSpec[object]",
                GoldLexiconConstraintSourceSpec(policy=PolicyDomain.SYNTACTIC),
            )
        ),
    )
    example = Example(
        Item(
            ItemId("at://example.invalid/expression/repeated"),
            (Token(0, "dog"), Token(1, "dog")),
        ),
        Unlabeled(),
    )
    selection = source.select((example,))
    (constrained,) = selection.examples

    assert len(constrained.supervision.constraints) == 2
    assert tuple(
        constraint.span for constraint in constrained.supervision.constraints
    ) == (
        TokenSpan(0, 1),
        TokenSpan(1, 2),
    )
    assert all(
        constraint.permitted == {"N"} and constraint.scope == TypeScope(("dog",))
        for constraint in constrained.supervision.constraints
    )
    assert selection.budget.types_pinned == 1
    assert selection.budget.tokens_covered == 2
    assert selection.budget.mass_share == 1.0


def test_the_gold_lexicon_source_keeps_the_weight_of_every_rendered_entry() -> None:
    signature = Signature(
        "weighted-constraint-test",
        ("Cat",),
        (
            Operator("N", "Cat", family="primitive"),
            Operator("V", "Cat", family="primitive"),
        ),
    )
    store = TermStore(signature)
    lexicon = StaticLexicon(
        (
            Entry("run", store.op(signature.by_name("N")), weight=0.25),
            Entry("run", store.op(signature.by_name("V")), weight=0.75),
        )
    )
    declaration = cast(
        "ConfiguredSpec[object]",
        GoldLexiconConstraintSourceSpec(policy=PolicyDomain.SYNTACTIC),
    )
    source = cast(
        "GoldLexiconSource[str, str]",
        plugin_for(declaration).construct(
            declaration,
            ConstraintSourceContext(
                syntactic=SYNTACTIC,
                semantic=SEMANTIC,
                lexicon=lexicon,
                render=lambda term: term.op.name,
                interpretation=None,
            ),
        ),
    )

    assert sorted(
        (entry.category, weight) for _, entry, weight in source.lexicon.axioms(("run",))
    ) == sorted(
        (entry.category.op.name, weight)
        for _, entry, weight in lexicon.axioms(("run",))
    )


def test_a_logical_form_source_needs_an_interpreted_corpus() -> None:
    declaration = cast(
        "ConfiguredSpec[object]",
        GoldLogicalFormConstraintSourceSpec(policy=PolicyDomain.SEMANTIC),
    )

    with pytest.raises(UnservedPolicyError, match="without an interpretation"):
        built(declaration)
