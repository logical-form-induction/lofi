from collections import Counter
from collections.abc import MutableMapping
from math import log
from random import Random
from typing import TYPE_CHECKING, Final

import pytest
from tests.conformance.constraint_source import ConstraintSourceConformance

from lofi._types import ItemId
from lofi.data import (
    ConstraintPattern,
    LogicalFormSource,
    PinnedConstraintSource,
    QuantilePopulation,
    ScoreQuantile,
)
from lofi.data._constraint import (
    SpanConstraint,
    SpanConstraints,
    TokenScope,
    TypeScope,
    occurrences,
)
from lofi.data._constraint_sources import (
    NO_GOLD_DERIVATION,
    DeclaredSource,
    GoldDerivationSource,
    GoldLexiconSource,
)
from lofi.data._derivation import Derivation, DerivationNode
from lofi.data._ports import ConstraintSource
from lofi.data._report import LoadReport
from lofi.data._selection import (
    CategoryBudget,
    CategoryClass,
    CorpusProfile,
    FrequencyCount,
    FrequencyQuantile,
    MajorityRule,
    PerOccurrenceRule,
    RankBand,
    ScopeKind,
    SelectionPolicy,
    UnionRule,
    WeightedRule,
    agreement,
)
from lofi.data._supervision import Example, GoldDerivation, Unlabeled, Weighted
from lofi.data._values import Item, Token, TokenSpan
from lofi.data.laws import ConstraintSourceFixture
from lofi.lexicon import Entry, StaticLexicon

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping

    from lofi.data._selection import FormSelector
    from lofi.data._supervision import Supervision

type Form = tuple[str, ...]
type Corpus = tuple[Example[Supervision[str]], ...]


EVERY_FORM: Final = RankBand(0.0, 1.0)


def item(identifier: str, *surfaces: str) -> Item:
    return Item(
        ItemId(f"at://example.invalid/expression/{identifier}"),
        tuple(Token(position, surface) for position, surface in enumerate(surfaces)),
    )


def node(start: int, end: int, label: str, *children: int) -> DerivationNode[str]:
    return DerivationNode(TokenSpan(start, end), label, children=children)


EVERY_PERSON_LEFT: Final = Example(
    item("1", "every", "person", "left"),
    GoldDerivation(
        Derivation(
            0,
            {
                0: node(0, 3, "S", 1, 4),
                1: node(0, 2, "NP", 2, 3),
                2: node(0, 1, "NP/N"),
                3: node(1, 2, "N"),
                4: node(2, 3, "S\\NP"),
            },
        )
    ),
)

EVERY_PERSON_SAW_EVERY_PERSON: Final = Example(
    item("2", "every", "person", "saw", "every", "person"),
    GoldDerivation(
        Derivation(
            0,
            {
                0: node(0, 5, "S", 1, 4),
                1: node(0, 2, "NP", 2, 3),
                2: node(0, 1, "NP/N"),
                3: node(1, 2, "N"),
                4: node(2, 5, "S\\NP", 5, 6),
                5: node(2, 3, "(S\\NP)/NP"),
                6: node(3, 5, "NP", 7, 8),
                7: node(3, 4, "NP/N"),
                8: node(4, 5, "N"),
            },
        )
    ),
)

NOBODY_LEFT: Final = Example(
    item("3", "nobody", "left"),
    GoldDerivation(
        Derivation(
            0, {0: node(0, 2, "S", 1, 2), 1: node(0, 1, "NP"), 2: node(1, 2, "S\\NP")}
        )
    ),
)

NOBODY_SAW_NOBODY: Final = Example(
    item("4", "nobody", "saw", "nobody"),
    GoldDerivation(
        Derivation(
            0,
            {
                0: node(0, 3, "S", 1, 2),
                1: node(0, 1, "NP"),
                2: node(1, 3, "S\\NP", 3, 4),
                3: node(1, 2, "(S\\NP)/NP"),
                4: node(2, 3, "NP"),
            },
        )
    ),
)

THE_LEFT: Final = Example(
    item("5", "the", "left"),
    GoldDerivation(
        Derivation(
            0, {0: node(0, 2, "NP", 1, 2), 1: node(0, 1, "NP/N"), 2: node(1, 2, "N")}
        )
    ),
)

LEFT_THE: Final = Example(item("6", "left", "the"), Unlabeled())

CORPUS: Final[Corpus] = (
    EVERY_PERSON_LEFT,
    EVERY_PERSON_SAW_EVERY_PERSON,
    NOBODY_LEFT,
    NOBODY_SAW_NOBODY,
    THE_LEFT,
    LEFT_THE,
)


def derived() -> tuple[Example[GoldDerivation[str]], ...]:
    held: list[Example[GoldDerivation[str]]] = []

    for example in CORPUS:
        supervision = example.supervision

        if isinstance(supervision, GoldDerivation):
            held.append(Example(example.item, supervision))

    return tuple(held)


DERIVED: Final = derived()

LEXICON: Final = StaticLexicon(
    (
        Entry("every", "NP/N"),
        Entry("person", "N"),
        Entry("left", "S\\NP", weight=log(2.0)),
        Entry("left", "N"),
        Entry("saw", "(S\\NP)/NP"),
        Entry("nobody", "NP"),
        Entry("the", "NP/N"),
    )
)

DECLARED_BY_ITEM: Final[Mapping[ItemId, tuple[SpanConstraint[str], ...]]] = {
    EVERY_PERSON_LEFT.item.item_id: (
        SpanConstraint(TokenSpan(0, 2), frozenset({"NP"})),
        SpanConstraint(TokenSpan(2, 3), frozenset({"S\\NP"})),
    ),
    EVERY_PERSON_SAW_EVERY_PERSON.item.item_id: (
        SpanConstraint(TokenSpan(0, 2), frozenset({"NP"})),
        SpanConstraint(TokenSpan(3, 5), frozenset({"NP"})),
        SpanConstraint(TokenSpan(2, 5), frozenset({"S\\NP"})),
    ),
    NOBODY_SAW_NOBODY.item.item_id: (
        SpanConstraint(TokenSpan(1, 3), frozenset({"S\\NP"})),
        SpanConstraint(TokenSpan(0, 1), frozenset({"NP"})),
    ),
    LEFT_THE.item.item_id: (SpanConstraint(TokenSpan(0, 1), frozenset({"N"})),),
}

DECLARED_BY_FORM: Final[Mapping[Form, frozenset[str]]] = {
    ("every", "person"): frozenset({"NP"}),
    ("left",): frozenset({"S\\NP", "N"}),
    ("nobody",): frozenset({"NP"}),
    ("saw",): frozenset({"(S\\NP)/NP"}),
    ("the",): frozenset({"NP/N"}),
    ("every", "person", "left"): frozenset({"S"}),
}


def surfaces(example: Example[Supervision[str]]) -> tuple[str, ...]:
    return tuple(token.surface for token in example.item.tokens)


def frequency(form: Form, corpus: Corpus = CORPUS) -> int:
    return sum(len(occurrences(form, example.item.tokens)) for example in corpus)


def constituents() -> dict[ItemId, frozenset[SpanConstraint[str]]]:
    return {
        example.item.item_id: frozenset(
            SpanConstraints.from_derivation(example.supervision.derivation).constraints
        )
        for example in DERIVED
    }


def lexical() -> dict[ItemId, frozenset[SpanConstraint[str]]]:
    expected: dict[ItemId, set[SpanConstraint[str]]] = {}

    for example in CORPUS:
        held = expected.setdefault(example.item.item_id, set())

        for position, surface in enumerate(surfaces(example)):
            entries = LEXICON.entries(surface)

            if not entries:
                continue

            held.add(
                SpanConstraint(
                    TokenSpan(position, position + 1),
                    frozenset(entry.category for entry in entries),
                    TypeScope((surface,)),
                )
            )

    return {item_id: frozenset(held) for item_id, held in expected.items()}


def declared() -> dict[ItemId, frozenset[SpanConstraint[str]]]:
    return {
        example.item.item_id: frozenset(DECLARED_BY_ITEM.get(example.item.item_id, ()))
        for example in CORPUS
    }


def policy(
    rule: UnionRule[str] | MajorityRule[str] | PerOccurrenceRule[str],
    *,
    coverage: float = 1.0,
    seed: int = 0,
    selector: FormSelector[str] = EVERY_FORM,
) -> SelectionPolicy[str, str]:
    return SelectionPolicy(rule, selector=selector, coverage=coverage, seed=seed)


def top(count: int, types: int) -> RankBand:
    return RankBand(0.0, count / types)


def gold_source(
    rule: UnionRule[str] | MajorityRule[str] | PerOccurrenceRule[str],
    *,
    coverage: float = 1.0,
    seed: int = 0,
    selector: FormSelector[str] = EVERY_FORM,
) -> GoldDerivationSource[str, str]:
    return GoldDerivationSource(
        policy(rule, coverage=coverage, seed=seed, selector=selector)
    )


def lexicon_source(
    rule: UnionRule[str] | MajorityRule[str] | PerOccurrenceRule[str],
    *,
    coverage: float = 1.0,
    seed: int = 0,
) -> GoldLexiconSource[str, str]:
    return GoldLexiconSource(LEXICON, policy(rule, coverage=coverage, seed=seed))


def by_item_source(*, coverage: float = 1.0, seed: int = 0) -> DeclaredSource[str, str]:
    return DeclaredSource(
        policy(PerOccurrenceRule(), coverage=coverage, seed=seed),
        by_item=DECLARED_BY_ITEM,
    )


def by_form_source(*, coverage: float = 1.0, seed: int = 0) -> DeclaredSource[str, str]:
    return DeclaredSource(
        policy(UnionRule(), coverage=coverage, seed=seed), by_form=DECLARED_BY_FORM
    )


def emitted_by[L: Hashable](
    source: ConstraintSource[str, L], corpus: Corpus = CORPUS
) -> dict[ItemId, tuple[SpanConstraint[L], ...]]:
    examples = source.select(corpus).examples

    return {
        example.item.item_id: example.supervision.constraints for example in examples
    }


def type_level[L: Hashable](
    source: ConstraintSource[str, L],
) -> dict[Form, frozenset[L]]:
    return type_level_over(source, CORPUS)


def type_level_over[C: Hashable, L: Hashable](
    source: ConstraintSource[C, L], corpus: tuple[Example[Supervision[C]], ...]
) -> dict[Form, frozenset[L]]:
    emitted = source.select(corpus).examples
    bound: dict[Form, frozenset[L]] = {}

    for example in emitted:
        for constraint in example.supervision.constraints:
            form = constraint.scope.form

            if form is not None:
                bound[form] = constraint.permitted

    return bound


class TestGoldDerivationSource(ConstraintSourceConformance):
    @pytest.fixture
    def constraint_source_fixture(self) -> ConstraintSourceFixture[str, str]:
        return ConstraintSourceFixture(
            gold_source(PerOccurrenceRule()),
            gold_source(UnionRule()),
            lambda seed: gold_source(PerOccurrenceRule(), coverage=0.5, seed=seed),
            CORPUS,
            constituents(),
        )


class TestGoldLexiconSource(ConstraintSourceConformance):
    @pytest.fixture
    def constraint_source_fixture(self) -> ConstraintSourceFixture[str, str]:
        return ConstraintSourceFixture(
            lexicon_source(UnionRule()),
            lexicon_source(UnionRule()),
            lambda seed: lexicon_source(UnionRule(), coverage=0.5, seed=seed),
            CORPUS,
            lexical(),
        )


class TestDeclaredSource(ConstraintSourceConformance):
    @pytest.fixture
    def constraint_source_fixture(self) -> ConstraintSourceFixture[str, str]:
        return ConstraintSourceFixture(
            by_item_source(),
            by_form_source(),
            lambda seed: by_item_source(coverage=0.5, seed=seed),
            CORPUS,
            declared(),
        )


@pytest.mark.parametrize(
    "source",
    [
        gold_source(PerOccurrenceRule()),
        lexicon_source(UnionRule()),
        by_item_source(),
        by_form_source(),
    ],
)
def test_each_implementer_satisfies_the_port_by_shape(
    source: ConstraintSource[str, str],
) -> None:
    assert isinstance(source, ConstraintSource)


def test_a_gold_derivation_source_takes_the_scope_of_its_rule() -> None:
    assert gold_source(PerOccurrenceRule()).scope is ScopeKind.OCCURRENCE
    assert gold_source(UnionRule()).scope is ScopeKind.FORM
    assert gold_source(UnionRule()).policy == SelectionPolicy(UnionRule())


def test_full_coverage_over_gold_derivations_is_every_constituent_as_a_singleton() -> (
    None
):
    emitted = emitted_by(gold_source(PerOccurrenceRule()))

    for example in DERIVED:
        derivation = example.supervision.derivation
        nodes = [derivation.node(node_id) for node_id in sorted(derivation.nodes)]

        assert emitted[example.item.item_id] == tuple(
            SpanConstraint(held.span, frozenset({held.label})) for held in nodes
        )
        assert all(
            constraint.scope == TokenScope()
            for constraint in emitted[example.item.item_id]
        )


def test_a_gold_derivation_source_skips_and_counts_an_item_without_a_derivation() -> (
    None
):
    selection = gold_source(PerOccurrenceRule()).select(CORPUS)
    examples, report = selection.examples, selection.report

    assert LEFT_THE.item.item_id not in {example.item.item_id for example in examples}
    assert report == LoadReport(len(CORPUS), len(DERIVED), {NO_GOLD_DERIVATION: 1})


def test_the_union_rule_over_derivations_pools_the_types_of_a_form() -> None:
    bound = type_level(gold_source(UnionRule()))

    assert bound[("left",)] == {"S\\NP", "N"}
    assert bound[("every", "person")] == {"NP"}
    assert bound[("the", "left")] == {"NP"}


def test_the_majority_rule_over_derivations_keeps_the_commonest_type() -> None:
    bound = type_level(gold_source(MajorityRule()))

    assert bound[("left",)] == {"S\\NP"}
    assert bound[("nobody",)] == {"NP"}


def test_the_weighted_rule_over_derivations_shares_the_occurrences_out() -> None:
    source = GoldDerivationSource[str, Weighted[str]](SelectionPolicy(WeightedRule()))
    bound = type_level(source)
    observed = Counter(
        held.label
        for example in DERIVED
        for held in example.supervision.derivation.nodes.values()
        if surfaces(example)[held.span.start : held.span.end] == ("left",)
    )

    assert bound[("left",)] == {
        Weighted(label, count / observed.total()) for label, count in observed.items()
    }


def test_a_type_level_constraint_is_anchored_at_every_occurrence_in_each_item() -> None:
    emitted = emitted_by(gold_source(UnionRule()))
    twice = emitted[EVERY_PERSON_SAW_EVERY_PERSON.item.item_id]
    every_person = [
        constraint
        for constraint in twice
        if constraint.scope.form == ("every", "person")
    ]

    assert every_person == [
        SpanConstraint(
            TokenSpan(0, 2), frozenset({"NP"}), TypeScope(("every", "person"))
        ),
        SpanConstraint(
            TokenSpan(3, 5), frozenset({"NP"}), TypeScope(("every", "person"))
        ),
    ]
    assert [constraint.span for constraint in every_person] == [
        TokenSpan(0, 2),
        TokenSpan(3, 5),
    ]


def test_a_type_level_form_is_bound_in_every_item_holding_it_derived_or_not() -> None:
    emitted = emitted_by(gold_source(UnionRule()))

    assert LEFT_THE.item.item_id not in emitted

    for example in DERIVED:
        held = {constraint.scope.form for constraint in emitted[example.item.item_id]}

        assert ("left",) in held or "left" not in surfaces(example)


def test_a_count_band_reads_counts_from_the_corpus() -> None:
    forms = set(type_level(gold_source(UnionRule())))
    common = {form for form in forms if frequency(form) >= 2}
    rare = forms - common

    assert common
    assert rare
    assert (
        set(type_level(gold_source(UnionRule(), selector=FrequencyCount(2, None))))
        == common
    )
    assert (
        set(type_level(gold_source(UnionRule(), selector=FrequencyCount(1, 1)))) == rare
    )


def test_a_count_band_counts_occurrences_in_every_item_of_the_corpus() -> None:
    lowest = min(frequency(form) for form in type_level(gold_source(UnionRule())))
    highest = max(frequency(form) for form in type_level(gold_source(UnionRule())))
    width = highest - lowest

    assert width > 0
    assert frequency(("left",)) > frequency(("left",), DERIVED)
    assert set(
        type_level(gold_source(UnionRule(), selector=FrequencyCount(highest, None)))
    ) == {
        form
        for form in type_level(gold_source(UnionRule()))
        if frequency(form) == highest
    }


def test_a_count_band_over_occurrences_reads_the_span_form_count() -> None:
    emitted = emitted_by(
        gold_source(PerOccurrenceRule(), selector=FrequencyCount(2, None))
    )

    for example in DERIVED:
        derivation = example.supervision.derivation

        assert emitted[example.item.item_id] == tuple(
            SpanConstraint(held.span, frozenset({held.label}))
            for held in (
                derivation.node(node_id) for node_id in sorted(derivation.nodes)
            )
            if frequency(surfaces(example)[held.span.start : held.span.end]) >= 2
        )


def kept_by(coverage: float) -> int:
    emitted = emitted_by(gold_source(PerOccurrenceRule(), coverage=coverage))

    return sum(len(held) for held in emitted.values())


def test_coverage_keeps_a_rounded_fraction_of_the_admitted_occurrences() -> None:
    total = sum(len(held) for held in constituents().values())

    assert total == 25
    assert kept_by(0.0) == 0
    assert kept_by(0.25) == 6
    assert kept_by(1.0) == total


def test_coverage_rounds_a_half_occurrence_to_even() -> None:
    assert kept_by(0.3) == 8
    assert kept_by(0.5) == 12
    assert kept_by(0.7) == 18


def test_coverage_keeps_a_rounded_fraction_of_the_admitted_forms() -> None:
    total = len(type_level(gold_source(UnionRule())))

    assert len(type_level(gold_source(UnionRule(), coverage=0.5))) == round(0.5 * total)
    assert type_level(gold_source(UnionRule(), coverage=0.0)) == {}


def test_covered_occurrences_are_a_subset_of_full_coverage_in_the_same_order() -> None:
    full = emitted_by(gold_source(PerOccurrenceRule()))
    half = emitted_by(gold_source(PerOccurrenceRule(), coverage=0.5, seed=5))

    for item_id, constraints in half.items():
        kept = [held for held in full[item_id] if held in constraints]

        assert list(constraints) == kept


def test_zero_coverage_still_emits_every_readable_item_unconstrained() -> None:
    selection = gold_source(PerOccurrenceRule(), coverage=0.0).select(CORPUS)
    examples, report = selection.examples, selection.report

    assert [example.item for example in examples] == [
        example.item for example in DERIVED
    ]
    assert all(example.supervision.constraints == () for example in examples)
    assert report.emitted == len(DERIVED)


def test_an_empty_corpus_emits_nothing_and_reports_nothing_seen() -> None:
    for source in (
        gold_source(PerOccurrenceRule()),
        lexicon_source(UnionRule()),
        by_form_source(),
    ):
        selection = source.select(())
        examples, report = selection.examples, selection.report

        assert examples == ()
        assert report == LoadReport(0, 0)


def test_a_lexicon_source_takes_the_scope_of_its_rule() -> None:
    assert lexicon_source(UnionRule()).scope is ScopeKind.FORM
    assert lexicon_source(MajorityRule()).scope is ScopeKind.FORM
    assert lexicon_source(PerOccurrenceRule()).scope is ScopeKind.OCCURRENCE
    assert lexicon_source(UnionRule()).lexicon is LEXICON


def test_a_lexicon_source_uses_gold_leaves_for_token_level_labels() -> None:
    selection = lexicon_source(PerOccurrenceRule()).select(CORPUS)
    emitted = {
        example.item.item_id: example.supervision.constraints
        for example in selection.examples
    }

    for example in DERIVED:
        derivation = example.supervision.derivation
        leaves = [
            derivation.node(node_id)
            for node_id in sorted(derivation.nodes)
            if derivation.node(node_id).is_lexical
        ]

        assert emitted[example.item.item_id] == tuple(
            SpanConstraint(leaf.span, frozenset({leaf.label})) for leaf in leaves
        )
        assert all(
            constraint.scope == TokenScope()
            for constraint in emitted[example.item.item_id]
        )

    assert LEFT_THE.item.item_id not in emitted
    assert selection.report == LoadReport(
        len(CORPUS), len(DERIVED), {NO_GOLD_DERIVATION: 1}
    )


def test_a_lexicon_source_reads_every_item_and_pools_a_form_across_them() -> None:
    selection = lexicon_source(UnionRule()).select(CORPUS)
    examples, report = selection.examples, selection.report

    assert [example.item for example in examples] == [
        example.item for example in CORPUS
    ]
    assert report == LoadReport(len(CORPUS), len(CORPUS))
    assert type_level(lexicon_source(UnionRule()))[("left",)] == {"S\\NP", "N"}
    assert type_level(lexicon_source(MajorityRule()))[("left",)] == {"S\\NP"}


def test_a_lexicon_source_weights_a_form_by_the_lexicon_not_by_its_frequency() -> None:
    source = GoldLexiconSource[str, Weighted[str]](
        LEXICON, SelectionPolicy(WeightedRule())
    )
    weighted = {label.value: label.weight for label in type_level(source)[("left",)]}

    assert weighted == {
        "S\\NP": pytest.approx(2.0 / 3.0),
        "N": pytest.approx(1.0 / 3.0),
    }
    assert frequency(("left",)) > 1


def test_a_lexicon_source_leaves_a_form_the_lexicon_lacks_alone() -> None:
    lexicon = StaticLexicon((Entry("nobody", "NP"),))
    source = GoldLexiconSource[str, str](lexicon, SelectionPolicy(UnionRule()))

    assert set(type_level(source)) == {("nobody",)}
    assert emitted_by(source)[THE_LEFT.item.item_id] == ()


def test_a_declared_source_takes_the_scope_of_its_list() -> None:
    assert by_item_source().scope is ScopeKind.OCCURRENCE
    assert by_form_source().scope is ScopeKind.FORM
    assert by_item_source().by_item == DECLARED_BY_ITEM
    assert by_form_source().by_form == DECLARED_BY_FORM


def test_a_declared_source_emits_its_list_as_written() -> None:
    emitted = emitted_by(by_item_source())

    for item_id, constraints in DECLARED_BY_ITEM.items():
        assert emitted[item_id] == constraints

    assert emitted[THE_LEFT.item.item_id] == ()
    assert type_level(by_form_source()) == dict(DECLARED_BY_FORM)


def test_a_declared_form_is_bound_in_every_item_holding_it() -> None:
    emitted = emitted_by(by_form_source())

    for example in CORPUS:
        held = {constraint.scope.form for constraint in emitted[example.item.item_id]}

        assert held == {
            form for form in DECLARED_BY_FORM if occurrences(form, example.item.tokens)
        }


def test_a_declared_source_holds_exactly_one_list() -> None:
    with pytest.raises(ValueError, match="one list"):
        DeclaredSource[str, str](SelectionPolicy(UnionRule()))

    with pytest.raises(ValueError, match="one list"):
        DeclaredSource(
            SelectionPolicy(UnionRule()),
            by_item=DECLARED_BY_ITEM,
            by_form=DECLARED_BY_FORM,
        )


def test_a_declared_source_refuses_a_rule_at_the_other_scope() -> None:
    with pytest.raises(ValueError, match="occurrence level"):
        DeclaredSource(SelectionPolicy(UnionRule()), by_item=DECLARED_BY_ITEM)

    with pytest.raises(ValueError, match="form level"):
        DeclaredSource(SelectionPolicy(PerOccurrenceRule()), by_form=DECLARED_BY_FORM)


def test_a_declared_source_refuses_a_form_scoped_constraint_by_item() -> None:
    with pytest.raises(ValueError, match="declared by form"):
        DeclaredSource(
            SelectionPolicy(PerOccurrenceRule()),
            by_item={
                NOBODY_LEFT.item.item_id: (
                    SpanConstraint(
                        TokenSpan(0, 1), frozenset({"NP"}), TypeScope(("nobody",))
                    ),
                )
            },
        )


def test_a_declared_source_refuses_an_empty_form_or_an_empty_label_set() -> None:
    with pytest.raises(ValueError, match="no words"):
        DeclaredSource(SelectionPolicy(UnionRule()), by_form={(): frozenset({"NP"})})

    with pytest.raises(ValueError, match="no label"):
        DeclaredSource(
            SelectionPolicy(UnionRule()), by_form={("nobody",): frozenset[str]()}
        )


def test_a_declared_source_refuses_a_corpus_missing_a_declared_item() -> None:
    with pytest.raises(ValueError, match="does not hold"):
        by_item_source().select(CORPUS[:2])


def test_a_declared_form_the_corpus_lacks_binds_nowhere() -> None:
    source = DeclaredSource(
        SelectionPolicy(UnionRule()),
        by_form={**DECLARED_BY_FORM, ("nobody", "slept"): frozenset({"S"})},
    )

    selection = source.select(CORPUS)
    examples, report = selection.examples, selection.report

    assert emitted_by(source) == emitted_by(by_form_source())
    assert len(examples) == len(CORPUS)
    assert report == LoadReport(len(CORPUS), len(CORPUS))


def test_a_declared_list_is_copied_rather_than_shared() -> None:
    by_item = {
        NOBODY_LEFT.item.item_id: (SpanConstraint(TokenSpan(0, 1), frozenset({"NP"})),)
    }
    source = DeclaredSource(SelectionPolicy(PerOccurrenceRule()), by_item=by_item)
    by_item.clear()

    assert len(source.by_item) == 1
    assert not isinstance(source.by_item, MutableMapping)
    assert not isinstance(by_form_source().by_form, MutableMapping)


def test_the_sources_report_themselves_as_their_constructor_calls() -> None:
    policy = SelectionPolicy[str, str](UnionRule())

    assert repr(GoldDerivationSource(policy)) == f"GoldDerivationSource({policy!r})"
    assert repr(GoldLexiconSource(LEXICON, policy)) == (
        f"GoldLexiconSource({LEXICON!r}, {policy!r})"
    )
    assert repr(by_form_source()) == (
        f"DeclaredSource({policy!r}, by_item={{}}, by_form={dict(DECLARED_BY_FORM)!r})"
    )


CLOSED: Final = "D"
CLASS_SIZES: Final = {CLOSED: 4, "P": 12, "V": 40}
DRAWN_TOKENS: Final = 6_000
DRAWN_LENGTH: Final = 6
DRAWN_SEED: Final = 5


def coupled_lexicon() -> dict[str, str]:
    return {
        f"{category.lower()}{index}": category
        for category, size in CLASS_SIZES.items()
        for index in range(size)
    }


def coupled_corpus() -> Corpus:
    declared = coupled_lexicon()
    weights = [float(CLASS_SIZES[category]) ** -1.0 for category in declared.values()]
    stream = Random(DRAWN_SEED).choices(list(declared), weights=weights, k=DRAWN_TOKENS)

    return tuple(
        Example(
            item(str(start // DRAWN_LENGTH), *stream[start : start + DRAWN_LENGTH]),
            Unlabeled(),
        )
        for start in range(0, DRAWN_TOKENS, DRAWN_LENGTH)
    )


def coupled_source(selector: FormSelector[str]) -> GoldLexiconSource[str, str]:
    return GoldLexiconSource(
        StaticLexicon(
            tuple(
                Entry(surface, category)
                for surface, category in coupled_lexicon().items()
            )
        ),
        policy(UnionRule(), selector=selector),
    )


def constituent_forms() -> dict[Form, frozenset[str]]:
    forms: dict[Form, set[str]] = {}

    for example in DERIVED:
        derivation = example.supervision.derivation

        for node_id in derivation.nodes:
            node = derivation.node(node_id)
            span = example.item.tokens[node.span.start : node.span.end]
            forms.setdefault(tuple(token.surface for token in span), set()).add(
                node.label
            )

    return {form: frozenset(labels) for form, labels in forms.items()}


def candidate_profile() -> CorpusProfile[str]:
    forms = constituent_forms()

    return CorpusProfile({form: frequency(form) for form in forms}, forms)


def test_a_type_level_budget_counts_what_a_rank_policy_pinned() -> None:
    profile = candidate_profile()
    wanted = 2
    source = gold_source(UnionRule(), selector=top(wanted, profile.types))
    selection = source.select(CORPUS)
    highest = profile.ranked[:wanted]
    covered = sum(frequency(form) for form in highest)

    assert set(type_level(source)) == set(highest)
    assert selection.budget.types_pinned == wanted
    assert selection.budget.tokens_covered == covered
    assert selection.budget.mass_share == pytest.approx(covered / profile.tokens)
    assert selection.budget.mass_share < 1.0


def test_a_type_level_budget_breaks_its_counts_down_by_category() -> None:
    profile = candidate_profile()
    forms = constituent_forms()
    source = gold_source(UnionRule(), selector=top(3, profile.types))
    rows = source.select(CORPUS).budget.by_category
    pinned = frozenset(type_level(source))

    assert set(rows) == {label for form in pinned for label in forms[form]}

    for label, row in rows.items():
        borne = [form for form in pinned if label in forms[form]]
        tokens = sum(frequency(form) for form in borne)

        assert row == CategoryBudget(len(borne), tokens, tokens / profile.tokens)


def test_every_selection_names_the_concordance_of_the_corpus_it_read() -> None:
    profile = candidate_profile()
    selection = gold_source(UnionRule()).select(CORPUS)

    assert profile.concordance is not None
    assert selection.budget.concordance == pytest.approx(profile.concordance)


def test_a_token_level_budget_counts_the_spans_its_constraints_bind() -> None:
    profile = candidate_profile()
    selection = gold_source(PerOccurrenceRule(), selector=top(2, profile.types)).select(
        CORPUS
    )
    bound = {
        (example.item.item_id, constraint.span)
        for example in selection.examples
        for constraint in example.supervision.constraints
    }

    assert selection.budget.tokens_covered == len(bound)
    assert selection.budget.mass_share == pytest.approx(len(bound) / profile.tokens)


def test_select_returns_examples_report_and_budget_together() -> None:
    for source in (
        gold_source(UnionRule(), selector=top(2, candidate_profile().types)),
        lexicon_source(UnionRule()),
        by_form_source(),
    ):
        selection = source.select(CORPUS)

        assert selection.examples
        assert selection.report.records_seen == len(CORPUS)
        assert selection.budget.types_pinned > 0


def test_a_declared_list_names_no_categories_so_a_class_selects_nothing() -> None:
    selection = by_form_source().select(CORPUS)
    classed = DeclaredSource(
        policy(UnionRule(), selector=CategoryClass({"NP"})), by_form=DECLARED_BY_FORM
    ).select(CORPUS)

    assert selection.budget.by_category == {}
    assert selection.budget.concordance is None
    assert selection.budget.types_pinned > 0
    assert classed.budget.types_pinned == 0
    assert classed.budget.tokens_covered == 0


def test_a_rank_policy_and_a_category_class_pin_the_same_closed_class() -> None:
    corpus = coupled_corpus()
    by_rank = coupled_source(top(CLASS_SIZES[CLOSED], sum(CLASS_SIZES.values())))
    by_class = coupled_source(CategoryClass({CLOSED}))
    ranked = frozenset(type_level_over(by_rank, corpus))
    classed = frozenset(type_level_over(by_class, corpus))

    assert ranked == classed
    assert ranked == frozenset(
        (surface,)
        for surface, category in coupled_lexicon().items()
        if category == CLOSED
    )
    assert by_rank.select(corpus).budget.types_pinned == CLASS_SIZES[CLOSED]
    assert by_rank.select(corpus).budget.mass_share == pytest.approx(
        by_class.select(corpus).budget.mass_share
    )


def test_a_quantile_over_token_mass_covers_the_share_it_names() -> None:
    corpus = coupled_corpus()
    wanted = 0.1
    source = coupled_source(FrequencyQuantile(wanted, over=QuantilePopulation.TOKENS))
    budget = source.select(corpus).budget
    counts = Counter(
        token.surface for example in corpus for token in example.item.tokens
    )
    granularity = max(counts.values()) / sum(counts.values())

    assert budget.mass_share >= wanted
    assert budget.mass_share - wanted <= granularity
    assert budget.tokens_covered == sum(
        counts[surface] for (surface,) in type_level_over(source, corpus)
    )


type ConstantClass = tuple[str, int | None]
type SemanticCorpus = tuple[Example[Supervision[ConstantClass]], ...]

QUANTIFIER: Final[ConstantClass] = ("quantifier", None)
CONNECTIVE: Final[ConstantClass] = ("connective", None)
UNARY_PREDICATE: Final[ConstantClass] = ("predicate", 1)
BINARY_PREDICATE: Final[ConstantClass] = ("predicate", 2)

MEANINGS: Final[Mapping[Form, str]] = {
    ("every",): "forall",
    ("nobody",): "no",
    ("the",): "iota",
    ("person",): "person(x)",
    ("left",): "left(x)",
    ("saw",): "saw(x,y)",
    ("and",): "and(p,q)",
}

MEANING_CLASSES: Final[Mapping[Form, frozenset[ConstantClass]]] = {
    ("every",): frozenset({QUANTIFIER}),
    ("nobody",): frozenset({QUANTIFIER}),
    ("the",): frozenset({QUANTIFIER}),
    ("person",): frozenset({UNARY_PREDICATE}),
    ("left",): frozenset({UNARY_PREDICATE}),
    ("saw",): frozenset({BINARY_PREDICATE}),
    ("and",): frozenset({CONNECTIVE}),
}

SEMANTIC_CORPUS: Final[SemanticCorpus] = (
    Example(item("1", "every", "person", "left"), Unlabeled()),
    Example(item("2", "every", "person", "saw", "every", "person"), Unlabeled()),
    Example(item("3", "nobody", "left"), Unlabeled()),
    Example(item("4", "nobody", "saw", "nobody"), Unlabeled()),
    Example(item("5", "the", "left"), Unlabeled()),
    Example(item("6", "left", "the"), Unlabeled()),
    Example(item("7", "person", "left", "and", "nobody", "left"), Unlabeled()),
)
"""Twenty-two tokens over seven assigned forms.

The quantifiers occur nine times over three forms, the one-place predicates ten
times over two, the one two-place predicate twice and the one connective once.
"""

SEMANTIC_SIZES: Final[Mapping[ConstantClass, int]] = {
    QUANTIFIER: 2,
    CONNECTIVE: 10,
    UNARY_PREDICATE: 40,
    BINARY_PREDICATE: 160,
}
"""How many logical forms each constant class holds.

The classes differ by an order of magnitude, so a positive coupling makes the
quantifiers the most frequent forms and a coupling of zero makes every form
equally frequent whatever class its meaning belongs to.
"""

SEMANTIC_PREFIXES: Final[Mapping[ConstantClass, str]] = {
    QUANTIFIER: "q",
    CONNECTIVE: "c",
    UNARY_PREDICATE: "u",
    BINARY_PREDICATE: "b",
}

SEMANTIC_TOKENS: Final = 6_000
SEMANTIC_SEED: Final = 7
SEMANTIC_COUPLINGS: Final = (0.0, 0.1, 0.25, 0.5, 1.0)
SEMANTIC_CHANCE: Final = 0.1


def logical_form_source(
    *,
    coverage: float = 1.0,
    seed: int = 0,
    selector: FormSelector[ConstantClass] = EVERY_FORM,
) -> LogicalFormSource[ConstantClass, str]:
    return LogicalFormSource(
        MEANINGS,
        MEANING_CLASSES,
        SelectionPolicy(UnionRule(), selector=selector, coverage=coverage, seed=seed),
    )


def pinned() -> dict[ItemId, frozenset[SpanConstraint[str]]]:
    expected: dict[ItemId, set[SpanConstraint[str]]] = {}

    for example in SEMANTIC_CORPUS:
        held = expected.setdefault(example.item.item_id, set())

        for form, meaning in MEANINGS.items():
            starts = occurrences(form, example.item.tokens)

            for start in starts:
                held.add(
                    SpanConstraint(
                        TokenSpan(start, start + len(form)),
                        frozenset({meaning}),
                        TypeScope(form),
                    )
                )

    return {item_id: frozenset(held) for item_id, held in expected.items()}


def drawn_lexicon() -> dict[Form, ConstantClass]:
    return {
        (f"{SEMANTIC_PREFIXES[held]}{index}",): held
        for held, size in SEMANTIC_SIZES.items()
        for index in range(size)
    }


def drawn_corpus(*, coupling: float) -> SemanticCorpus:
    declared = drawn_lexicon()
    weights = [float(SEMANTIC_SIZES[held]) ** -coupling for held in declared.values()]
    stream = Random(SEMANTIC_SEED).choices(
        [surface for (surface,) in declared], weights=weights, k=SEMANTIC_TOKENS
    )

    return tuple(
        Example(
            item(str(start // DRAWN_LENGTH), *stream[start : start + DRAWN_LENGTH]),
            Unlabeled(),
        )
        for start in range(0, SEMANTIC_TOKENS, DRAWN_LENGTH)
    )


def drawn_source(
    selector: FormSelector[ConstantClass],
) -> LogicalFormSource[ConstantClass, str]:
    declared = drawn_lexicon()

    return LogicalFormSource(
        {form: f"{form[0]}(x)" for form in declared},
        {form: frozenset({held}) for form, held in declared.items()},
        SelectionPolicy(UnionRule(), selector=selector),
    )


def drawn_profile(corpus: SemanticCorpus) -> CorpusProfile[ConstantClass]:
    counts: Counter[Form] = Counter(
        (token.surface,) for example in corpus for token in example.item.tokens
    )
    declared = drawn_lexicon()

    return CorpusProfile(
        {form: counts[form] for form in declared},
        {form: frozenset({held}) for form, held in declared.items()},
    )


class TestLogicalFormSource(ConstraintSourceConformance):
    @pytest.fixture
    def constraint_source_fixture(self) -> ConstraintSourceFixture[ConstantClass, str]:
        return ConstraintSourceFixture(
            logical_form_source(),
            logical_form_source(),
            lambda seed: logical_form_source(coverage=0.5, seed=seed),
            SEMANTIC_CORPUS,
            pinned(),
        )


def test_a_logical_form_source_satisfies_the_port_by_shape() -> None:
    assert isinstance(logical_form_source(), ConstraintSource)


def test_a_logical_form_source_binds_one_meaning_at_the_type_level() -> None:
    source = logical_form_source()

    assert source.scope is ScopeKind.FORM
    assert source.assignment == MEANINGS
    assert source.classes == MEANING_CLASSES
    assert type_level_over(source, SEMANTIC_CORPUS) == {
        form: frozenset({meaning}) for form, meaning in MEANINGS.items()
    }


def test_a_logical_form_source_refuses_a_rule_binding_at_the_token_level() -> None:
    with pytest.raises(ValueError, match="pinned on a lexical form"):
        LogicalFormSource(
            MEANINGS, MEANING_CLASSES, SelectionPolicy(PerOccurrenceRule())
        )


def test_a_logical_form_source_refuses_a_form_it_assigns_but_does_not_classify() -> (
    None
):
    with pytest.raises(ValueError, match="named on one side only"):
        LogicalFormSource(
            {**MEANINGS, ("dog",): "dog(x)"},
            MEANING_CLASSES,
            SelectionPolicy(UnionRule()),
        )


def test_a_logical_form_source_refuses_an_empty_form_or_a_meaning_of_no_class() -> None:
    with pytest.raises(ValueError, match="form of no words"):
        LogicalFormSource(
            {(): "forall"}, {(): frozenset({QUANTIFIER})}, SelectionPolicy(UnionRule())
        )

    with pytest.raises(ValueError, match="meaning of no class"):
        LogicalFormSource(
            {("every",): "forall"},
            {("every",): frozenset[ConstantClass]()},
            SelectionPolicy(UnionRule()),
        )


def test_a_logical_form_assignment_is_copied_rather_than_shared() -> None:
    assignment: MutableMapping[Form, str] = dict(MEANINGS)
    classes: MutableMapping[Form, frozenset[ConstantClass]] = dict(MEANING_CLASSES)
    source = LogicalFormSource(assignment, classes, SelectionPolicy(UnionRule()))
    assignment[("dog",)] = "dog(x)"
    classes[("dog",)] = frozenset({UNARY_PREDICATE})

    assert ("dog",) not in source.assignment
    assert ("dog",) not in source.classes


def test_the_logical_form_budget_counts_forms_tokens_and_mass_per_class() -> None:
    budget = logical_form_source().select(SEMANTIC_CORPUS).budget

    assert budget.types_pinned == 7
    assert budget.tokens_covered == 22
    assert budget.mass_share == pytest.approx(1.0)
    assert budget.by_category == {
        QUANTIFIER: CategoryBudget(3, 9, 9 / 22),
        UNARY_PREDICATE: CategoryBudget(2, 10, 10 / 22),
        BINARY_PREDICATE: CategoryBudget(1, 2, 2 / 22),
        CONNECTIVE: CategoryBudget(1, 1, 1 / 22),
    }


def test_a_constant_class_pins_only_the_forms_whose_meaning_bears_it() -> None:
    source = logical_form_source(selector=CategoryClass({QUANTIFIER, BINARY_PREDICATE}))
    budget = source.select(SEMANTIC_CORPUS).budget

    assert set(type_level_over(source, SEMANTIC_CORPUS)) == {
        ("every",),
        ("nobody",),
        ("the",),
        ("saw",),
    }
    assert budget.types_pinned == 4
    assert budget.tokens_covered == 11
    assert budget.mass_share == pytest.approx(0.5)
    assert budget.by_category == {
        QUANTIFIER: CategoryBudget(3, 9, 9 / 22),
        BINARY_PREDICATE: CategoryBudget(1, 2, 2 / 22),
    }


def test_a_class_naming_the_two_place_predicates_pins_those_alone() -> None:
    source = logical_form_source(selector=CategoryClass({BINARY_PREDICATE}))

    assert type_level_over(source, SEMANTIC_CORPUS) == {
        ("saw",): frozenset({"saw(x,y)"})
    }
    assert source.select(SEMANTIC_CORPUS).budget.by_category == {
        BINARY_PREDICATE: CategoryBudget(1, 2, 2 / 22)
    }


def test_a_meaning_assigned_to_a_form_the_corpus_lacks_pins_nowhere() -> None:
    source = LogicalFormSource(
        {**MEANINGS, ("dog",): "dog(x)"},
        {**MEANING_CLASSES, ("dog",): frozenset({UNARY_PREDICATE})},
        SelectionPolicy(UnionRule()),
    )

    assert ("dog",) not in type_level_over(source, SEMANTIC_CORPUS)
    assert source.select(SEMANTIC_CORPUS).budget.types_pinned == 7


def test_a_logical_form_selection_emits_every_item_and_reports_what_it_emits() -> None:
    source = logical_form_source(selector=CategoryClass({BINARY_PREDICATE}))
    selection = source.select(SEMANTIC_CORPUS)
    selection = source.select(SEMANTIC_CORPUS)
    examples, report = selection.examples, selection.report
    constrained = [example for example in examples if example.supervision.constraints]

    assert examples == selection.examples
    assert report == selection.report
    assert report == LoadReport(len(SEMANTIC_CORPUS), len(SEMANTIC_CORPUS), {})
    assert len(examples) == len(SEMANTIC_CORPUS)
    assert len(constrained) == 2


def test_a_logical_form_source_reports_itself_as_its_constructor_call() -> None:
    source = LogicalFormSource(
        {("every",): "forall"},
        {("every",): frozenset({QUANTIFIER})},
        SelectionPolicy(UnionRule()),
    )

    assert repr(source) == (
        "LogicalFormSource({('every',): 'forall'}, "
        "{('every',): frozenset({('quantifier', None)})}, "
        "SelectionPolicy(UnionRule(), selector=RankBand(0.0, 1.0), "
        "coverage=1.0, seed=0))"
    )


def test_a_rank_policy_and_a_quantifier_class_pin_the_same_logical_forms() -> None:
    corpus = drawn_corpus(coupling=1.0)
    by_rank = drawn_source(
        RankBand(0.0, SEMANTIC_SIZES[QUANTIFIER] / sum(SEMANTIC_SIZES.values()))
    )
    by_class = drawn_source(CategoryClass({QUANTIFIER}))
    ranked = frozenset(type_level_over(by_rank, corpus))
    classed = frozenset(type_level_over(by_class, corpus))

    assert ranked == classed
    assert ranked == frozenset(
        form for form, held in drawn_lexicon().items() if held == QUANTIFIER
    )
    assert by_rank.select(corpus).budget.types_pinned == SEMANTIC_SIZES[QUANTIFIER]
    assert by_rank.select(corpus).budget.mass_share == pytest.approx(
        by_class.select(corpus).budget.mass_share
    )


def test_a_quantile_over_token_mass_pins_the_share_it_names() -> None:
    corpus = drawn_corpus(coupling=1.0)
    wanted = 0.1
    source = drawn_source(FrequencyQuantile(wanted, over=QuantilePopulation.TOKENS))
    budget = source.select(corpus).budget
    counts = Counter(
        token.surface for example in corpus for token in example.item.tokens
    )
    granularity = max(counts.values()) / sum(counts.values())

    assert budget.mass_share >= wanted
    assert budget.mass_share - wanted <= granularity
    assert budget.tokens_covered == sum(
        counts[surface] for (surface,) in type_level_over(source, corpus)
    )


def test_every_logical_form_selection_names_the_concordance_of_its_corpus() -> None:
    corpus = drawn_corpus(coupling=1.0)
    profile = drawn_profile(corpus)
    budget = drawn_source(CategoryClass({QUANTIFIER})).select(corpus).budget

    assert profile.concordance is not None
    assert budget.concordance == pytest.approx(profile.concordance)
    assert logical_form_source().select(SEMANTIC_CORPUS).budget.concordance is not None


def test_rank_and_class_agreement_over_meanings_rises_with_the_concordance() -> None:
    points = []

    for coupling in SEMANTIC_COUPLINGS:
        corpus = drawn_corpus(coupling=coupling)
        by_rank = drawn_source(
            RankBand(0.0, SEMANTIC_SIZES[QUANTIFIER] / sum(SEMANTIC_SIZES.values()))
        )
        by_class = drawn_source(CategoryClass({QUANTIFIER}))
        reported = by_rank.select(corpus).budget.concordance
        measured = agreement(
            frozenset(type_level_over(by_rank, corpus)),
            frozenset(type_level_over(by_class, corpus)),
            drawn_profile(corpus),
        )

        assert reported is not None
        assert measured is not None
        assert by_class.select(corpus).budget.concordance == pytest.approx(reported)

        points.append((reported, measured))

    by_concordance = [measured for _, measured in sorted(points)]
    half = len(by_concordance) // 2
    lower = by_concordance[:half]
    upper = by_concordance[half:]

    assert by_concordance == sorted(by_concordance)
    assert sum(upper) / len(upper) > sum(lower) / len(lower)
    assert abs(points[0][1]) < SEMANTIC_CHANCE
    assert points[-1][1] == pytest.approx(1.0)


def pinned_source(
    *patterns: ConstraintPattern[str],
    score_quantile: ScoreQuantile | None = None,
) -> PinnedConstraintSource[str, str]:
    """Return an occurrence-level source over declared patterns."""
    return PinnedConstraintSource(
        SelectionPolicy(PerOccurrenceRule()),
        patterns,
        score_quantile=score_quantile,
    )


def test_one_source_locates_lexical_items_patterns_and_sentence_tops() -> None:
    source = pinned_source(
        ConstraintPattern(("nobody",), frozenset({"ANIMATE"})),
        ConstraintPattern((None, "person"), frozenset({"NP"})),
        ConstraintPattern(None, frozenset({"S"})),
    )
    emitted = emitted_by(source)
    first = emitted[EVERY_PERSON_LEFT.item.item_id]
    nobody = emitted[NOBODY_SAW_NOBODY.item.item_id]

    assert first == (
        SpanConstraint(TokenSpan(0, 2), frozenset({"NP"})),
        SpanConstraint(TokenSpan(0, 3), frozenset({"S"})),
    )
    assert nobody == (
        SpanConstraint(TokenSpan(0, 1), frozenset({"ANIMATE"})),
        SpanConstraint(TokenSpan(2, 3), frozenset({"ANIMATE"})),
        SpanConstraint(TokenSpan(0, 3), frozenset({"S"})),
    )


def test_a_score_quantile_quantile_constrains_only_its_upper_tail() -> None:
    scores = {
        example.item.item_id: float(position) for position, example in enumerate(CORPUS)
    }
    source = pinned_source(
        ConstraintPattern(None, frozenset({"S"})),
        score_quantile=ScoreQuantile(0.5, scores),
    )
    selection = source.select(CORPUS)
    constrained = {
        example.item.item_id
        for example in selection.examples
        if example.supervision.constraints
    }

    assert constrained == {example.item.item_id for example in CORPUS[-3:]}
    assert selection.budget.types_pinned == 3
    assert selection.budget.tokens_covered == 3
    assert selection.budget.mass_share == pytest.approx(0.5)


def test_score_quantile_ties_at_the_quantile_boundary_are_all_retained() -> None:
    scores = {
        example.item.item_id: score
        for example, score in zip(CORPUS, (0.0, 1.0, 2.0, 2.0, 2.0, 3.0), strict=True)
    }
    selected = ScoreQuantile(0.5, scores).selected(
        tuple(example.item for example in CORPUS)
    )

    assert selected == {example.item.item_id for example in CORPUS[2:]}


def test_a_score_quantile_score_for_an_unknown_item_is_refused() -> None:
    source = pinned_source(
        ConstraintPattern(None, frozenset({"S"})),
        score_quantile=ScoreQuantile(
            0.9, {ItemId("at://example.invalid/expression/missing"): 1.0}
        ),
    )

    with pytest.raises(ValueError, match="outside the corpus"):
        source.select(CORPUS)


def test_a_pinned_source_requires_patterns_and_occurrence_selection() -> None:
    with pytest.raises(ValueError, match="at least one pattern"):
        PinnedConstraintSource(
            SelectionPolicy(PerOccurrenceRule()), (), score_quantile=None
        )

    with pytest.raises(ValueError, match="individual occurrences"):
        PinnedConstraintSource(
            SelectionPolicy(UnionRule()),
            (ConstraintPattern(None, frozenset({"S"})),),
            score_quantile=None,
        )


def test_declared_pattern_and_score_quantile_values_are_frozen() -> None:
    pattern = ConstraintPattern(("nobody",), frozenset({"NP"}))
    quantile = ScoreQuantile(0.5, {ItemId("at://example.invalid/expression/a"): 1.0})

    with pytest.raises(AttributeError, match="immutable"):
        pattern.__setattr__("_pattern", None)

    with pytest.raises(AttributeError, match="immutable"):
        quantile.__setattr__("_quantile", 0.0)
