"""Tests for the synthetic corpus values."""

from hashlib import sha256
from math import inf, nan
from random import Random
from typing import TYPE_CHECKING, Final, cast, overload, override

import pytest
from tests.unit.spec.grammar.test__signature import slash_views, slashes

from lofi._types import ItemId
from lofi.data import (
    Categorical,
    Derivation,
    DerivationNode,
    Example,
    GoldDerivation,
    Item,
    SamplingExhaustedError,
    Token,
    TokenSpan,
)
from lofi.data.synthetic import (
    GRAMMATICALITY_LABELS,
    AmbiguityReport,
    AmbiguityTarget,
    AmbiguityUnreachableError,
    Corrupted,
    Corruption,
    CorruptionDraw,
    CorruptionExhaustedError,
    CorruptionReport,
    GoldForm,
    GrammarKnobs,
    Histogram,
    IllTyped,
    IllTypedFormError,
    IllTypingReport,
    InventoryReport,
    LabelNoise,
    LengthReport,
    LexiconReport,
    NegativeBinomialLength,
    NegativeDataPolicy,
    NoiseReport,
    NonDerivability,
    ObservedLabel,
    ReadingCount,
    ScopeReport,
    ScopeTarget,
    ScopeUnreachableError,
    ShapeMeasurement,
    ShapeReport,
    ShapeSchedule,
    ShapeTarget,
    ShapeUnreachableError,
    SymbolInventory,
    SyntheticCorpus,
    SyntheticDataset,
    SyntheticGenerator,
    Target,
    TypingConflict,
    TypingFailure,
    UncertifiedDerivationError,
    UniformLength,
    UninterpretableDerivationError,
    UnproductiveGrammarError,
    WeightedOperator,
)
from lofi.data.synthetic.spec.interpretation import (
    InterpretationSpec,
    LexicalMeaningSourceSpec,
    MeaningSpec,
    lower_interpretation,
)
from lofi.deduce import ParsePolicy, Reachability, Retain
from lofi.spec import ApplicationSpec, RuleSetSpec
from lofi.spec.grammar import (
    BudgetSpec,
    ConjunctionPairSpec,
    ConjunctionsSpec,
    FeatureErasureMatchingSpec,
    GrammarAssembly,
    GrammarDecl,
    GrammarIso,
    InlineLexiconSpec,
    LexicalEntrySpec,
    OperatorSpec,
    SignatureMorphismErasureSpec,
    SignatureSpec,
)
from lofi.spec.language import ConstantSpec, LanguageIso, registered_language

if TYPE_CHECKING:
    from collections.abc import Callable

    from lofi.data.synthetic import FormTypingLabel, GrammaticalityLabel
    from lofi.terms import Term


class _NamedOperator:
    """A stable named operator for report-value tests."""

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @overload
    def apply(
        self, random: Random, value: tuple[str, ...], /
    ) -> tuple[str, ...] | None: ...

    @overload
    def apply(self, random: Random, value: Term, /) -> Term | None: ...

    def apply(
        self, random: Random, value: tuple[str, ...] | Term, /
    ) -> tuple[str, ...] | Term | None:
        del random
        return value

    @override
    def __eq__(self, other: object) -> bool:
        return type(other) is _NamedOperator and self._name == other._name

    @override
    def __hash__(self) -> int:
        return hash((_NamedOperator, self._name))

    @override
    def __repr__(self) -> str:
        return f"_NamedOperator({self._name!r})"


def _request(
    *names: str,
    proportion: float = 0.5,
    attempts: int = 3,
    draw: CorruptionDraw = "exact",
    policy: NegativeDataPolicy | None = None,
) -> Corruption[_NamedOperator]:
    return Corruption(
        proportion,
        operators=tuple(WeightedOperator(_NamedOperator(name), 1.0) for name in names),
        attempts=attempts,
        draw=draw,
        policy=policy,
    )


KNOBS: Final = GrammarKnobs(3, ("/", "\\"), ("application",))
INVENTORY: Final = SymbolInventory(
    {
        "individual": {"j": 0},
        "predicate": {"sleep": 1},
        "function": {"father": 1},
    }
)
DECLARATION: Final = b'{"name":"toy"}'
THEORY: Final = b'{"ops":[]}'
CERTIFICATE: Final = NonDerivability("d" * 64, "t" * 64, unary_rounds=1)
REPORT: Final = AmbiguityReport(
    Histogram({1: 2}), Histogram({1: 1}), rejected=0, target=None
)
REQUEST: Final = _request("swap", "deletion", "insertion", "substitution")
INJECTION: Final = _request("argument-swap", "over-application", "sort-mismatch")
CONFLICT: Final = TypingConflict("unification", "e", '(-> "e" "t")', (0,))
FAILURE: Final = TypingFailure((CONFLICT,), language_digest="l" * 64)
LEXICON: Final = LexiconReport(
    {"NP": Histogram({1: 2})}, Histogram({1: 2}), Histogram({1: 2}), None
)
CATEGORIES: Final = InventoryReport(
    {"NP": 0},
    {"NP": 0},
    {"NP": 0},
    ("NP",),
    max_arguments=None,
    max_order=None,
)
LENGTH: Final = LengthReport(
    Histogram({2: 1}), resampled=0, distribution=UniformLength(1, 3)
)


def _example(*forms: str) -> Example[GoldDerivation[str]]:
    tokens = tuple(Token(index, form) for index, form in enumerate(forms))
    nodes = {
        0: DerivationNode(
            TokenSpan(0, len(forms)),
            "S",
            rule="application",
            children=tuple(range(1, len(forms) + 1)),
        ),
        **{
            index + 1: DerivationNode(TokenSpan(index, index + 1), "NP")
            for index in range(len(forms))
        },
    }

    return Example(
        Item(ItemId(" ".join(forms)), tokens), GoldDerivation(Derivation(0, nodes))
    )


def _corrupted(*forms: str, kind: str = "swap") -> Corrupted:
    tokens = tuple(Token(index, form) for index, form in enumerate(forms))

    return Corrupted(
        Item(ItemId("!" + " ".join(forms)), tokens), kind, ("a", "b"), CERTIFICATE
    )


def _report(accepted: int, sample_size: int) -> CorruptionReport[_NamedOperator]:
    return CorruptionReport(
        REQUEST,
        accepted={"swap": accepted},
        rejected={},
        inapplicable={},
        sample_size=sample_size,
    )


def _labels(
    examples: tuple[Example[GoldDerivation[str]], ...],
    corrupted: tuple[Corrupted, ...],
    flipped: frozenset[str] = frozenset(),
) -> tuple[ObservedLabel, ...]:
    return tuple(
        ObservedLabel.of(item.item_id, truth, flipped=item.item_id in flipped)
        for item, truth in (
            *((example.item, "grammatical") for example in examples),
            *((record.item, "ungrammatical") for record in corrupted),
        )
    )


def _injection(
    accepted: int, sample_size: int, *, rejected: int = 0, inapplicable: int = 0
) -> CorruptionReport[_NamedOperator]:
    return CorruptionReport(
        INJECTION,
        accepted={"argument-swap": accepted},
        rejected={"argument-swap": rejected},
        inapplicable={"sort-mismatch": inapplicable},
        sample_size=sample_size,
    )


def _ill_typed(
    item_id: str = "a b",
    text: str = "a(b)",
    *,
    observed_label: FormTypingLabel = "ill-typed",
) -> IllTyped:
    return IllTyped(
        ItemId(item_id),
        "argument-swap",
        "b(a)",
        text,
        FAILURE,
        observed_label=observed_label,
    )


def _corpus(
    *,
    declaration: bytes = DECLARATION,
    theory: bytes = THEORY,
    seed: int = 7,
    length: LengthReport = LENGTH,
    knobs: GrammarKnobs = KNOBS,
    examples: tuple[Example[GoldDerivation[str]], ...] = (_example("a", "b"),),
    ambiguity: AmbiguityReport = REPORT,
    lexicon: LexiconReport = LEXICON,
    inventory: InventoryReport = CATEGORIES,
    corrupted: tuple[Corrupted, ...] = (),
    corruption: CorruptionReport[_NamedOperator] | None = None,
    labels: tuple[ObservedLabel, ...] | None = None,
    noise: NoiseReport | None = None,
    forms: tuple[GoldForm, ...] = (),
    ill_typed: tuple[IllTyped, ...] = (),
    ill_typing: IllTypingReport | None = None,
    scope: ScopeReport | None = None,
    shape: ShapeReport | None = None,
) -> SyntheticCorpus:
    return SyntheticCorpus(
        declaration,
        theory,
        seed=seed,
        length=length,
        knobs=knobs,
        examples=examples,
        ambiguity=ambiguity,
        lexicon=lexicon,
        inventory=inventory,
        corrupted=corrupted,
        corruption=corruption,
        labels=_labels(examples, corrupted) if labels is None else labels,
        noise=noise,
        forms=forms,
        ill_typed=ill_typed,
        ill_typing=ill_typing,
        scope=scope,
        shape=shape,
    )


def _length(
    min_length: int = 1, max_length: int = 3, *, achieved: Histogram | None = None
) -> LengthReport:
    return LengthReport(
        Histogram({2: 1}) if achieved is None else achieved,
        resampled=0,
        distribution=UniformLength(min_length, max_length),
    )


def test_knobs_expose_their_dimensions() -> None:
    assert KNOBS.primitive_count == 3
    assert KNOBS.constructors == ("/", "\\")
    assert KNOBS.combinators == ("application",)
    assert KNOBS.inventory is None
    assert repr(KNOBS) == (
        "GrammarKnobs(3, ('/', '\\\\'), ('application',), inventory=None)"
    )
    assert not hasattr(KNOBS, "vocabulary_size")
    assert not hasattr(KNOBS, "categories_per_form")


def test_knobs_are_equal_on_all_their_dimensions() -> None:
    same = GrammarKnobs(3, ("/", "\\"), ("application",))

    assert same == KNOBS
    assert hash(same) == hash(KNOBS)
    assert GrammarKnobs(2, ("/", "\\"), ("application",)) != KNOBS
    assert GrammarKnobs(3, ("->",), ("application",)) != KNOBS
    assert GrammarKnobs(3, ("/", "\\"), ("composition",)) != KNOBS
    assert KNOBS.with_inventory(INVENTORY) != KNOBS
    assert KNOBS.__eq__(("/", "\\")) is NotImplemented


def test_knobs_gain_the_symbol_inventory_beside_the_grammar_dimensions() -> None:
    knobs = KNOBS.with_inventory(INVENTORY)

    assert knobs.inventory is INVENTORY
    assert knobs.primitive_count == KNOBS.primitive_count
    assert knobs.constructors == KNOBS.constructors
    assert knobs.combinators == KNOBS.combinators
    assert knobs == GrammarKnobs(3, ("/", "\\"), ("application",), inventory=INVENTORY)
    assert knobs.with_inventory(None) == KNOBS
    assert repr(knobs).endswith(f"inventory={INVENTORY!r})")


def test_an_inventory_lists_its_symbols_by_role_and_counts_them_once() -> None:
    assert INVENTORY.roles == {
        "individual": {"j": 0},
        "predicate": {"sleep": 1},
        "function": {"father": 1},
    }
    assert INVENTORY.in_role("predicate") == {"sleep": 1}
    assert INVENTORY.in_role("absent") == {}
    assert INVENTORY.names == ("j", "sleep", "father")
    assert len(INVENTORY) == 3
    assert len(SymbolInventory()) == 0
    assert repr(INVENTORY) == (
        "SymbolInventory({'individual': {'j': 0}, 'predicate': {'sleep': 1}, "
        "'function': {'father': 1}})"
    )


def test_an_inventory_is_equal_on_its_symbols_and_arities() -> None:
    same = SymbolInventory(
        {
            "individual": {"j": 0},
            "predicate": {"sleep": 1},
            "function": {"father": 1},
        }
    )

    assert same == INVENTORY
    assert hash(same) == hash(INVENTORY)
    assert SymbolInventory({"individual": {"j": 0, "m": 0}}) != INVENTORY
    assert SymbolInventory({"predicate": {"sleep": 2}}) != INVENTORY
    assert SymbolInventory({"predicate": {"sleep": 1}}) != INVENTORY
    assert INVENTORY.__eq__(("j",)) is NotImplemented

    overlapping = SymbolInventory(
        {"predicate": {"most": 2}, "generalized-universal": {"most": 2}}
    )
    assert overlapping.names == ("most",)
    assert len(overlapping) == 1


@pytest.mark.parametrize(
    ("build", "error", "fault"),
    [
        (lambda: SymbolInventory({"": {"p": 1}}), ValueError, "inventory role"),
        (lambda: SymbolInventory({"predicate": {"": 1}}), ValueError, "symbol"),
        (lambda: SymbolInventory({"function": {"f": -1}}), ValueError, "arity"),
        (
            lambda: SymbolInventory({"predicate": {"p": cast("int", 1.5)}}),
            TypeError,
            "not an integer",
        ),
        (
            lambda: SymbolInventory({"predicate": {"p": 1}, "higher-order": {"p": 2}}),
            ValueError,
            "has arity 1.*and 2",
        ),
    ],
)
def test_an_inventory_refuses_empty_names_and_bad_arities(
    build: Callable[[], SymbolInventory], error: type[Exception], fault: str
) -> None:
    with pytest.raises(error, match=fault):
        build()


@pytest.mark.parametrize(
    ("build", "error", "fault"),
    [
        (lambda: GrammarKnobs(-1, (), ()), ValueError, "primitive_count"),
        (lambda: GrammarKnobs(3, ("/", "/"), ()), ValueError, "constructors"),
        (lambda: GrammarKnobs(3, (), ("a", "a")), ValueError, "combinators"),
        (
            lambda: GrammarKnobs(cast("int", 4.0), (), ()),
            TypeError,
            "not an integer",
        ),
    ],
)
def test_knobs_refuse_negative_counts_and_repeated_names(
    build: Callable[[], GrammarKnobs], error: type[Exception], fault: str
) -> None:
    with pytest.raises(error, match=fault):
        build()


def _joint(
    classes: dict[str, dict[int, int]],
    forms: dict[int, int] | None = None,
    ambiguity: dict[int, int] | None = None,
    correlation: float | None = None,
) -> LexiconReport:
    held = {category: Histogram(counts) for category, counts in classes.items()}
    merged: dict[int, int] = {}

    for histogram in held.values():
        for value, count in histogram.counts.items():
            merged[value] = merged.get(value, 0) + count

    measured_forms = Histogram(merged if forms is None else forms)

    return LexiconReport(
        held,
        measured_forms,
        Histogram({1: measured_forms.total} if ambiguity is None else ambiguity),
        correlation,
    )


def test_a_lexicon_report_reads_sizes_counts_and_shares_off_its_histograms() -> None:
    report = _joint({"Det": {30: 2}, "N": {2: 5, 0: 3}, "V": {0: 4}})

    assert report.class_sizes == {"Det": 2, "N": 8, "V": 4}
    assert report.token_counts == {"Det": 60, "N": 10, "V": 0}
    assert report.token_shares == {"Det": 60 / 70, "N": 10 / 70, "V": 0.0}
    assert report.forms == Histogram({30: 2, 2: 5, 0: 7})
    assert report.realized_size == 14
    assert report.realized_vocabulary_size == 7
    assert report.per_type_ambiguity == Histogram({1: 14})
    assert report.ambiguity_frequency_correlation is None
    assert report.type_token_ratio == pytest.approx(7 / 70)
    assert report.classes["N"] == Histogram({2: 5, 0: 3})
    assert _joint({"V": {0: 4}}).token_shares == {"V": 0.0}
    assert repr(report) == (
        "LexiconReport({'Det': Histogram({30: 2}), 'N': Histogram({0: 3, 2: 5}), "
        "'V': Histogram({0: 4})}, Histogram({0: 7, 2: 5, 30: 2}), "
        "Histogram({1: 14}), None)"
    )


def test_a_lexicon_report_fits_the_rank_frequency_exponent_of_its_forms() -> None:
    zipf = _joint({"N": {12: 1, 6: 1, 4: 1, 3: 1}})

    assert zipf.rank_frequency_exponent == pytest.approx(1.0)
    assert _joint({"N": {8: 1, 2: 1, 0: 2}}).rank_frequency_exponent == (
        pytest.approx(2.0)
    )
    assert _joint({"N": {5: 4, 0: 1}}).rank_frequency_exponent == pytest.approx(0.0)
    assert _joint({"N": {5: 1, 0: 3}}).rank_frequency_exponent is None
    assert _joint({"N": {0: 3}}).rank_frequency_exponent is None


def test_a_lexicon_report_fits_the_coupling_of_size_to_tokens_per_form() -> None:
    assert _joint({"Det": {8: 1}, "N": {4: 2}, "V": {2: 4}}).coupling == (
        pytest.approx(1.0)
    )
    assert _joint({"Det": {1: 1}, "N": {1: 4}}).coupling == pytest.approx(0.0)
    assert _joint({"Det": {1: 1}, "N": {4: 4}}).coupling == pytest.approx(-1.0)
    assert _joint({"Det": {8: 1}, "N": {0: 4}}).coupling is None
    assert _joint({"Det": {8: 1}, "N": {4: 1}}).coupling is None


def test_a_lexicon_report_measures_the_concordance_of_frequency_and_class() -> None:
    small_frequent = _joint({"Det": {9: 1, 8: 1}, "N": {3: 1, 2: 1, 1: 1, 0: 1}})
    large_frequent = _joint({"Det": {0: 1, 1: 1}, "N": {9: 1, 8: 1, 7: 1, 6: 1}})
    mixed = _joint({"Det": {9: 1, 0: 1}, "N": {9: 1, 0: 1, 5: 2}})

    assert small_frequent.concordance is not None
    assert large_frequent.concordance is not None
    assert small_frequent.concordance > 0.8
    assert large_frequent.concordance < -0.8
    assert mixed.concordance == pytest.approx(0.0)
    assert _joint({"Det": {3: 2}, "N": {3: 4}}).concordance is None
    assert _joint({"N": {5: 1, 2: 1}}).concordance is None


def test_a_lexicon_report_is_equal_on_its_histograms() -> None:
    assert _joint({"N": {1: 2}}) == _joint({"N": {1: 2}})
    assert hash(_joint({"N": {1: 2}})) == hash(_joint({"N": {1: 2}}))
    assert _joint({"N": {1: 2}}) != _joint({"V": {1: 2}})
    assert _joint({"N": {1: 2}}) != _joint({"N": {1: 2}}, forms={2: 1})
    assert _joint({"N": {1: 2}}) != _joint({"N": {1: 2}}, ambiguity={1: 1, 2: 1})
    assert _joint({"N": {1: 2}}) != _joint({"N": {1: 2}}, correlation=0.5)
    assert _joint({"N": {1: 2}}).__eq__(Histogram({1: 2})) is NotImplemented


@pytest.mark.parametrize(
    ("build", "fault"),
    [
        (lambda: _joint({"N": {}}), "'N' bears no form"),
        (lambda: _joint({"N": {2: 1}}, forms={1: 1}), "count 2 tokens but the forms"),
        (
            lambda: _joint({"N": {1: 2}}, ambiguity={1: 1}),
            "counts 2 forms but measures ambiguity for 1",
        ),
        (
            lambda: _joint({"N": {1: 2}}, correlation=1.1),
            r"outside \[-1, 1\]",
        ),
    ],
)
def test_a_lexicon_report_refuses_an_empty_class_or_disagreeing_totals(
    build: Callable[[], LexiconReport], fault: str
) -> None:
    with pytest.raises(ValueError, match=fault):
        build()


def test_an_inventory_report_reads_back_its_categories_depths_and_primitives() -> None:
    depths = {"S\\NP": 1, "NP": 0, "(S\\NP)/NP": 2}
    argument_counts = {"S\\NP": 1, "NP": 0, "(S\\NP)/NP": 2}
    composition_orders = {"S\\NP": 1, "NP": 0, "(S\\NP)/NP": 1}
    report = InventoryReport(
        depths,
        argument_counts,
        composition_orders,
        ("S", "NP"),
        max_arguments=2,
        max_order=1,
    )

    assert report.size == 3
    assert list(report.categories) == ["S\\NP", "NP", "(S\\NP)/NP"]
    assert report.depths == Histogram({0: 1, 1: 1, 2: 1})
    assert report.argument_counts == argument_counts
    assert report.arguments == Histogram({0: 1, 1: 1, 2: 1})
    assert report.composition_orders == composition_orders
    assert report.orders == Histogram({0: 1, 1: 2})
    assert report.max_arguments == 2
    assert report.max_order == 1
    assert report.primitives == ("S", "NP")
    assert report.realized_primitive_count == 2
    assert report == InventoryReport(
        depths,
        argument_counts,
        composition_orders,
        ("S", "NP"),
        max_arguments=2,
        max_order=1,
    )
    assert hash(report) == hash(
        InventoryReport(
            depths,
            argument_counts,
            composition_orders,
            ("S", "NP"),
            max_arguments=2,
            max_order=1,
        )
    )
    assert report != InventoryReport(
        {"NP": 0},
        {"NP": 0},
        {"NP": 0},
        ("NP",),
        max_arguments=None,
        max_order=None,
    )
    assert report.__eq__({"NP": 0}) is NotImplemented
    assert repr(CATEGORIES) == (
        "InventoryReport({'NP': 0}, {'NP': 0}, {'NP': 0}, ('NP',), "
        "max_arguments=None, max_order=None)"
    )


@pytest.mark.parametrize(
    ("build", "fault"),
    [
        (
            lambda: InventoryReport({}, {}, {}, (), max_arguments=None, max_order=None),
            "empty",
        ),
        (
            lambda: InventoryReport(
                {"NP": -1},
                {"NP": 0},
                {"NP": 0},
                ("NP",),
                max_arguments=None,
                max_order=None,
            ),
            "negative",
        ),
        (
            lambda: InventoryReport(
                {"NP": 0},
                {"NP": -1},
                {"NP": 0},
                ("NP",),
                max_arguments=None,
                max_order=None,
            ),
            "negative",
        ),
        (
            lambda: InventoryReport(
                {"NP": 0},
                {"N": 0},
                {"NP": 0},
                ("NP",),
                max_arguments=None,
                max_order=None,
            ),
            "different lexical categories",
        ),
        (
            lambda: InventoryReport(
                {"NP": 0},
                {"NP": 0},
                {"NP": 0},
                ("NP", "NP"),
                max_arguments=None,
                max_order=None,
            ),
            "twice",
        ),
    ],
)
def test_an_inventory_report_refuses_an_empty_inventory_or_a_malformed_field(
    build: Callable[[], InventoryReport], fault: str
) -> None:
    with pytest.raises(ValueError, match=fault):
        build()


def test_a_corpus_refuses_an_inventory_naming_other_categories_than_its_lexicon() -> (
    None
):
    with pytest.raises(ValueError, match="the inventory names"):
        _corpus(
            inventory=InventoryReport(
                {"N": 0},
                {"N": 0},
                {"N": 0},
                ("N",),
                max_arguments=None,
                max_order=None,
            )
        )


def test_a_histogram_buckets_observations_and_reads_back_totals() -> None:
    histogram = Histogram.of([3, 1, 3, 2, 3])

    assert histogram == Histogram({1: 1, 2: 1, 3: 3})
    assert hash(histogram) == hash(Histogram({3: 3, 2: 1, 1: 1}))
    assert list(histogram.counts) == [1, 2, 3]
    assert histogram.total == 5
    assert histogram.mean == 2.4
    assert histogram.variance == pytest.approx(0.64)
    assert Histogram({4: 3}).variance == 0.0
    assert histogram[3] == 3
    assert histogram[7] == 0
    assert repr(histogram) == "Histogram({1: 1, 2: 1, 3: 3})"
    assert histogram != Histogram({1: 1, 2: 1, 3: 2})
    assert histogram.__eq__({1: 1}) is NotImplemented


def test_an_empty_bucket_is_dropped_and_an_empty_histogram_has_no_mean() -> None:
    assert Histogram({2: 0, 5: 1}) == Histogram({5: 1})
    assert Histogram.of([]).total == 0

    with pytest.raises(ValueError, match="no mean"):
        _ = Histogram.of([]).mean

    with pytest.raises(ValueError, match="no mean"):
        _ = Histogram.of([]).variance


@pytest.mark.parametrize(
    ("build", "error", "fault"),
    [
        (lambda: Histogram({1: -1}), ValueError, "count of 1 is negative"),
        (lambda: Histogram({1: cast("int", 1.5)}), TypeError, "not an integer"),
        (lambda: Histogram({cast("int", 1.5): 1}), TypeError, "value is not"),
        (lambda: Histogram({cast("int", not 0): 1}), TypeError, "value is not"),
    ],
)
def test_a_histogram_refuses_non_integers_and_negative_counts(
    build: Callable[[], Histogram], error: type[Exception], fault: str
) -> None:
    with pytest.raises(error, match=fault):
        build()


def test_a_target_admits_measurements_within_its_band_inclusive() -> None:
    target = Target(3, 0.5)

    assert target.value == 3.0
    assert target.tolerance == 0.5
    assert target.admits(3)
    assert target.admits(2.5)
    assert target.admits(3.5)
    assert not target.admits(3.51)
    assert not target.admits(2.49)
    assert not target.admits(inf)
    assert not target.admits(nan)
    assert Target(3, 0).admits(3)
    assert not Target(3, 0).admits(4)
    assert target == Target(3.0, 0.5)
    assert hash(target) == hash(Target(3.0, 0.5))
    assert target != Target(3, 1)
    assert target != Target(2, 0.5)
    assert target.__eq__(3.0) is NotImplemented
    assert repr(target) == "Target(3.0, 0.5)"


def test_a_target_admits_a_measurement_on_an_edge_that_is_not_representable() -> None:
    assert abs(1.0 - 1.3) > 0.3
    assert Target(1.3, 0.3).admits(1.0)
    assert Target(1.3, 0.3).admits(1.6)
    assert 0.7 + 0.1 < 0.8
    assert Target(0.7, 0.1).admits(0.8)
    assert 0.8 - 0.1 > 0.7
    assert Target(0.8, 0.1).admits(0.7)
    assert not Target(1.3, 0.3).admits(0.999)
    assert not Target(1.3, 0.3).admits(1.601)
    assert not Target(1.3, 0.3).admits(1.0 - 1e-6)


@pytest.mark.parametrize(
    ("build", "error", "fault"),
    [
        (lambda: Target(3, -0.1), ValueError, "tolerance is negative"),
        (lambda: Target(inf, 1), ValueError, "value is not finite"),
        (lambda: Target(1, nan), ValueError, "tolerance is not finite"),
        (lambda: Target(cast("float", "3"), 1), TypeError, "not a number"),
    ],
)
def test_a_target_refuses_a_negative_tolerance_and_non_finite_numbers(
    build: Callable[[], Target], error: type[Exception], fault: str
) -> None:
    with pytest.raises(error, match=fault):
        build()


def test_an_ambiguity_target_holds_its_bands_and_budget() -> None:
    target = AmbiguityTarget(lexical=Target(2, 0), derivations=Target(3, 1), attempts=9)

    assert target.lexical == Target(2, 0)
    assert target.derivations == Target(3, 1)
    assert target.attempts == 9
    assert AmbiguityTarget(attempts=1).lexical is None
    assert AmbiguityTarget(attempts=1).derivations is None
    assert target == AmbiguityTarget(
        lexical=Target(2, 0), derivations=Target(3, 1), attempts=9
    )
    assert hash(target) == hash(
        AmbiguityTarget(lexical=Target(2, 0), derivations=Target(3, 1), attempts=9)
    )
    assert target != AmbiguityTarget(derivations=Target(3, 1), attempts=9)
    assert target != AmbiguityTarget(
        lexical=Target(2, 0), derivations=Target(3, 1), attempts=8
    )
    assert target.__eq__(9) is NotImplemented
    assert repr(target) == (
        "AmbiguityTarget(lexical=Target(2.0, 0.0), "
        "derivations=Target(3.0, 1.0), attempts=9)"
    )


@pytest.mark.parametrize(
    ("attempts", "error", "fault"),
    [
        (0, ValueError, "not positive"),
        (-1, ValueError, "negative"),
        (1.0, TypeError, "not an integer"),
    ],
)
def test_an_ambiguity_target_needs_a_positive_attempt_budget(
    attempts: int, error: type[Exception], fault: str
) -> None:
    with pytest.raises(error, match=fault):
        AmbiguityTarget(attempts=attempts)


def test_an_ambiguity_report_holds_what_was_measured_and_aimed_at() -> None:
    target = AmbiguityTarget(derivations=Target(3, 0), attempts=4)
    report = AmbiguityReport(
        Histogram({1: 4, 2: 2}), Histogram({3: 3}), rejected=5, target=target
    )

    assert report.lexical.mean == pytest.approx(4 / 3)
    assert report.derivations == Histogram({3: 3})
    assert report.rejected == 5
    assert report.target is target
    assert REPORT.target is None
    assert report == AmbiguityReport(
        Histogram({1: 4, 2: 2}), Histogram({3: 3}), rejected=5, target=target
    )
    assert hash(report) == hash(
        AmbiguityReport(
            Histogram({1: 4, 2: 2}), Histogram({3: 3}), rejected=5, target=target
        )
    )
    assert report != AmbiguityReport(
        Histogram({1: 4, 2: 2}), Histogram({3: 3}), rejected=4, target=target
    )
    assert report != AmbiguityReport(
        Histogram({1: 4, 2: 2}), Histogram({3: 3}), rejected=5, target=None
    )
    assert report != AmbiguityReport(
        Histogram({1: 6}), Histogram({3: 3}), rejected=5, target=target
    )
    assert report.__eq__(target) is NotImplemented
    assert repr(report) == (
        "AmbiguityReport(Histogram({1: 4, 2: 2}), Histogram({3: 3}), "
        f"rejected=5, target={target!r})"
    )

    with pytest.raises(ValueError, match="rejected is negative"):
        AmbiguityReport(Histogram({}), Histogram({}), rejected=-1, target=None)


def test_a_shape_target_admits_each_declared_measurement_band() -> None:
    target = ShapeTarget(
        depth=Target(4, 1),
        right_branching=Target(0.75, 0.25),
        slash_balance=Target(0.5, 0.1),
        arity=Target(2, 0),
        attempts=9,
    )
    admitted = ShapeMeasurement(4, 0.5, 0.6, 2.0)

    assert target.depth == Target(4, 1)
    assert target.right_branching == Target(0.75, 0.25)
    assert target.slash_balance == Target(0.5, 0.1)
    assert target.arity == Target(2, 0)
    assert target.attempts == 9
    assert target.admits(admitted)
    assert not target.admits(ShapeMeasurement(4, 0.5, 0.7, 2.0))
    assert hash(target) == hash(
        ShapeTarget(
            depth=Target(4, 1),
            right_branching=Target(0.75, 0.25),
            slash_balance=Target(0.5, 0.1),
            arity=Target(2, 0),
            attempts=9,
        )
    )


@pytest.mark.parametrize(
    ("build", "fault"),
    [
        (lambda: ShapeTarget(attempts=1), "names no measurement"),
        (
            lambda: ShapeTarget(right_branching=Target(1.1, 0), attempts=1),
            "not between zero and one",
        ),
        (
            lambda: ShapeTarget(slash_balance=Target(-0.1, 0), attempts=1),
            "not between zero and one",
        ),
        (lambda: ShapeTarget(depth=Target(0, 0), attempts=1), "not positive"),
        (lambda: ShapeTarget(arity=Target(0, 0), attempts=1), "less than one"),
    ],
)
def test_a_shape_target_refuses_an_invalid_band(
    build: Callable[[], ShapeTarget], fault: str
) -> None:
    with pytest.raises(ValueError, match=fault):
        build()


def test_a_shape_report_summarizes_accepted_derivations() -> None:
    target = ShapeTarget(depth=Target(3, 1), attempts=4)
    schedule = ShapeSchedule(target, {})
    measurements = (
        ShapeMeasurement(3, 0.0, 0.5, 2.0),
        ShapeMeasurement(4, 0.5, 1.0, 1.5),
    )
    report = ShapeReport(measurements, rejected=3, target=schedule)

    assert report.measurements == measurements
    assert report.rejected == 3
    assert report.target is schedule
    assert report.depths == Histogram({3: 1, 4: 1})
    assert report.right_branching == 0.25
    assert report.slash_balance == 0.75
    assert report.arity == 1.75
    assert report == ShapeReport(measurements, rejected=3, target=schedule)
    assert hash(report) == hash(ShapeReport(measurements, rejected=3, target=schedule))

    with pytest.raises(ValueError, match="rejected is negative"):
        ShapeReport((), rejected=-1, target=schedule)


def test_a_shape_schedule_selects_exact_split_targets_before_its_default() -> None:
    fallback = ShapeTarget(depth=Target(2, 0), attempts=4)
    deep = ShapeTarget(depth=Target(4, 0), attempts=8)
    schedule = ShapeSchedule(fallback, {"validation": deep})

    assert schedule.default is fallback
    assert dict(schedule.splits) == {"validation": deep}
    assert schedule.target(None) is fallback
    assert schedule.target("train") is fallback
    assert schedule.target("validation") is deep
    assert schedule == ShapeSchedule(fallback, {"validation": deep})
    assert hash(schedule) == hash(ShapeSchedule(fallback, {"validation": deep}))


def test_the_shape_refusal_is_a_sampling_exhaustion() -> None:
    assert issubclass(ShapeUnreachableError, SamplingExhaustedError)


def test_a_corruption_names_its_share_operators_and_budget() -> None:
    assert REQUEST.proportion == 0.5
    assert REQUEST.operator_names == ("swap", "deletion", "insertion", "substitution")
    assert REQUEST.attempts == 3
    assert REQUEST.draw == "exact"
    assert REQUEST.policy == NegativeDataPolicy("labeled-negatives")

    narrowed = _request("deletion", proportion=1, attempts=1)

    assert narrowed.proportion == 1.0
    assert narrowed.operator_names == ("deletion",)
    assert _request("swap", "deletion", "insertion", "substitution") == REQUEST
    assert hash(REQUEST) == hash(
        _request("swap", "deletion", "insertion", "substitution")
    )
    assert narrowed != REQUEST
    assert (
        _request("swap", "deletion", "insertion", "substitution", attempts=4) != REQUEST
    )
    assert (
        _request("swap", "deletion", "insertion", "substitution", proportion=0.4)
        != REQUEST
    )
    assert REQUEST != 0.5
    assert "operators=(WeightedOperator(_NamedOperator('deletion'), 1.0),)" in repr(
        narrowed
    )


def test_exact_and_bernoulli_corruption_counts_share_one_request_boundary() -> None:
    exact = _request("swap", proportion=0.25)
    stochastic = _request("swap", proportion=0.25, draw="bernoulli")

    assert exact.count(10, Random(0)) == 3
    draws = [stochastic.count(400, Random(seed)) for seed in range(20)]
    assert all(abs(count - 100) < 40 for count in draws)


def test_negative_data_policies_apply_the_three_current_regimes() -> None:
    labeled = NegativeDataPolicy("labeled-negatives")
    positive = NegativeDataPolicy("positive-only")
    noisy = NegativeDataPolicy("label-flips", noise=0.25)

    assert not labeled.flips(0.0)
    assert positive.flips(0.999)
    assert noisy.flips(0.24)
    assert not noisy.flips(0.25)
    assert noisy.regime == "label-flips"
    assert noisy.noise == 0.25
    assert repr(noisy) == "NegativeDataPolicy('label-flips', noise=0.25)"
    assert noisy == NegativeDataPolicy("label-flips", noise=0.25)
    assert hash(noisy) == hash(NegativeDataPolicy("label-flips", noise=0.25))


@pytest.mark.parametrize(
    ("build", "fault"),
    [
        (lambda: NegativeDataPolicy("positive-only", noise=0.1), "does not take"),
        (lambda: NegativeDataPolicy("label-flips", noise=-0.1), "between zero"),
        (lambda: NegativeDataPolicy("label-flips", noise=1.1), "between zero"),
        (lambda: NegativeDataPolicy("label-flips").flips(1.0), r"not in \[0, 1\)"),
    ],
)
def test_a_negative_data_policy_refuses_an_invalid_rate_or_draw(
    build: Callable[[], object], fault: str
) -> None:
    with pytest.raises(ValueError, match=fault):
        build()


@pytest.mark.parametrize(
    ("build", "error", "fault"),
    [
        (
            lambda: _request("swap", proportion=1.5, attempts=1),
            ValueError,
            "between zero and one",
        ),
        (
            lambda: _request("swap", proportion=-0.5, attempts=1),
            ValueError,
            "between zero and one",
        ),
        (
            lambda: _request("swap", proportion=nan, attempts=1),
            ValueError,
            "not finite",
        ),
        (
            lambda: _request("swap", proportion=cast("float", "1"), attempts=1),
            TypeError,
            "number",
        ),
        (
            lambda: _request("swap", attempts=0),
            ValueError,
            "not positive",
        ),
        (_request, ValueError, "no operator"),
        (
            lambda: _request("swap", "swap", attempts=1),
            ValueError,
            "twice",
        ),
    ],
)
def test_a_corruption_refuses_a_bad_share_operator_list_or_budget(
    build: Callable[[], Corruption[_NamedOperator]],
    error: type[Exception],
    fault: str,
) -> None:
    with pytest.raises(error, match=fault):
        build()


def test_a_certificate_names_the_grammar_and_the_budget_parsed_under() -> None:
    assert CERTIFICATE.declaration_digest == "d" * 64
    assert CERTIFICATE.theory_digest == "t" * 64
    assert CERTIFICATE.unary_rounds == 1
    assert NonDerivability("d" * 64, "t" * 64, unary_rounds=1) == CERTIFICATE
    assert hash(CERTIFICATE) == hash(
        NonDerivability("d" * 64, "t" * 64, unary_rounds=1)
    )
    assert NonDerivability("e" * 64, "t" * 64, unary_rounds=1) != CERTIFICATE
    assert NonDerivability("d" * 64, "u" * 64, unary_rounds=1) != CERTIFICATE
    assert NonDerivability("d" * 64, "t" * 64, unary_rounds=2) != CERTIFICATE
    assert CERTIFICATE.__eq__("d" * 64) is NotImplemented
    assert repr(CERTIFICATE) == (
        f"NonDerivability({'d' * 64!r}, {'t' * 64!r}, unary_rounds=1)"
    )


@pytest.mark.parametrize(
    ("build", "error", "fault"),
    [
        (lambda: NonDerivability("", "t", unary_rounds=1), ValueError, "declaration"),
        (lambda: NonDerivability("d", "", unary_rounds=1), ValueError, "theory"),
        (lambda: NonDerivability("d", "t", unary_rounds=-1), ValueError, "negative"),
        (
            lambda: NonDerivability("d", "t", unary_rounds=cast("int", 1.0)),
            TypeError,
            "not an integer",
        ),
    ],
)
def test_a_certificate_refuses_empty_digests_and_a_bad_budget(
    build: Callable[[], NonDerivability], error: type[Exception], fault: str
) -> None:
    with pytest.raises(error, match=fault):
        build()


def test_a_corrupted_record_pairs_the_item_with_its_source_and_certificate() -> None:
    record = _corrupted("b", "a")

    assert record.item.item_id == "!b a"
    assert record.kind == "swap"
    assert record.source == ("a", "b")
    assert record.certificate is CERTIFICATE
    assert record == _corrupted("b", "a")
    assert hash(record) == hash(_corrupted("b", "a"))
    assert record != _corrupted("b", "a", kind="deletion")
    assert record != _corrupted("a", "b")
    assert record.__eq__(CERTIFICATE) is NotImplemented
    assert repr(record) == (
        f"Corrupted({record.item!r}, 'swap', ('a', 'b'), {CERTIFICATE!r})"
    )


def test_a_corrupted_record_refuses_an_empty_operator_or_source() -> None:
    item = Item(ItemId("x"), (Token(0, "x"),))

    with pytest.raises(ValueError, match="operator is empty"):
        Corrupted(item, "", ("a",), CERTIFICATE)

    with pytest.raises(ValueError, match="'x' has no source string"):
        Corrupted(item, "swap", (), CERTIFICATE)


def test_a_corruption_report_counts_every_operator_and_derives_its_rates() -> None:
    report = CorruptionReport(
        REQUEST,
        accepted={"swap": 3, "deletion": 1},
        rejected={"swap": 1},
        inapplicable={"insertion": 2},
        sample_size=8,
    )

    assert report.requested is REQUEST
    assert dict(report.accepted) == {
        "swap": 3,
        "deletion": 1,
        "insertion": 0,
        "substitution": 0,
    }
    assert list(report.rejected) == list(REQUEST.operator_names)
    assert report.rejected["swap"] == 1
    assert report.inapplicable["insertion"] == 2
    assert report.sample_size == 8
    assert report.corrupted == 4
    assert report.achieved_proportion == 0.5
    assert report.rejection_rate == 0.2
    assert report == CorruptionReport(
        REQUEST,
        accepted={"deletion": 1, "swap": 3},
        rejected={"swap": 1},
        inapplicable={"insertion": 2},
        sample_size=8,
    )
    assert hash(report) == hash(
        CorruptionReport(
            REQUEST,
            accepted={"deletion": 1, "swap": 3},
            rejected={"swap": 1},
            inapplicable={"insertion": 2},
            sample_size=8,
        )
    )
    assert report != _report(4, 8)
    assert report != _report(3, 8)
    assert report.__eq__(REQUEST) is NotImplemented
    assert repr(_report(1, 2)) == (
        f"CorruptionReport({REQUEST!r}, "
        "accepted={'swap': 1, 'deletion': 0, 'insertion': 0, 'substitution': 0}, "
        "rejected={'swap': 0, 'deletion': 0, 'insertion': 0, 'substitution': 0}, "
        "inapplicable={'swap': 0, 'deletion': 0, 'insertion': 0, 'substitution': 0}, "
        "sample_size=2)"
    )


def test_a_corruption_report_with_nothing_tried_has_no_rate_and_no_share() -> None:
    empty = _report(0, 0)

    assert empty.achieved_proportion == 0.0

    with pytest.raises(ValueError, match="no rejection rate"):
        _ = empty.rejection_rate


@pytest.mark.parametrize(
    ("build", "error", "fault"),
    [
        (
            lambda: _report(3, 2),
            ValueError,
            "3 corrupted items outnumber a sample of 2",
        ),
        (lambda: _report(-1, 2), ValueError, "accepted\\['swap'\\] is negative"),
        (lambda: _report(1, -1), ValueError, "sample_size is negative"),
        (lambda: _report(cast("int", 1.0), 2), TypeError, "not an integer"),
        (
            lambda: CorruptionReport(
                REQUEST,
                accepted={},
                rejected={"shuffle": 1},
                inapplicable={},
                sample_size=2,
            ),
            ValueError,
            "rejected counts operators the request does not name: \\['shuffle'\\]",
        ),
    ],
)
def test_a_corruption_report_refuses_impossible_counts(
    build: Callable[[], CorruptionReport[_NamedOperator]],
    error: type[Exception],
    fault: str,
) -> None:
    with pytest.raises(error, match=fault):
        build()


def test_a_corpus_digests_the_bytes_it_holds() -> None:
    corpus = _corpus()

    assert corpus.declaration is DECLARATION
    assert corpus.theory is THEORY
    assert corpus.declaration_digest == sha256(DECLARATION).hexdigest()
    assert corpus.theory_digest == sha256(THEORY).hexdigest()
    assert corpus.seed == 7
    assert corpus.length is LENGTH
    assert corpus.min_length == 1
    assert corpus.max_length == 3
    assert corpus.knobs is KNOBS
    assert corpus.examples == (_example("a", "b"),)
    assert corpus.ambiguity is REPORT
    assert corpus.lexicon is LEXICON
    assert corpus.corrupted == ()
    assert corpus.corruption is None
    assert corpus.labels == (
        ObservedLabel(ItemId("a b"), "grammatical", "grammatical", flipped=False),
    )
    assert corpus.noise is None
    assert corpus.forms == {}
    assert len(corpus) == 1
    assert repr(corpus) == (
        f"SyntheticCorpus(declaration={corpus.declaration_digest[:12]!r}, "
        "seed=7, length=UniformLength(1, 3), examples=1, corrupted=0)"
    )


def test_a_corpus_counts_its_corrupted_items_beside_its_examples() -> None:
    corpus = _corpus(corrupted=(_corrupted("b", "a"),), corruption=_report(1, 2))

    assert corpus.corrupted == (_corrupted("b", "a"),)
    assert corpus.corruption == _report(1, 2)
    assert len(corpus) == 2
    assert repr(corpus).endswith("examples=1, corrupted=1)")


def test_a_corpus_is_equal_on_bytes_settings_items_and_reports() -> None:
    assert _corpus() == _corpus()
    assert hash(_corpus()) == hash(_corpus())
    assert _corpus() != _corpus(declaration=b'{"name":"other"}')
    assert _corpus() != _corpus(theory=b'{"ops":[1]}')
    assert _corpus() != _corpus(seed=8)
    assert _corpus() != _corpus(length=_length(2))
    assert _corpus() != _corpus(length=_length(max_length=4))
    assert _corpus() != _corpus(
        length=LengthReport(
            Histogram({2: 1}), resampled=1, distribution=UniformLength(1, 3)
        )
    )
    assert _corpus() != _corpus(knobs=GrammarKnobs(3, ("/", "\\"), ()))
    assert _corpus() != _corpus(examples=(), length=_length(achieved=Histogram({})))
    assert _corpus() != _corpus(
        ambiguity=AmbiguityReport(
            Histogram({1: 2}), Histogram({2: 1}), rejected=0, target=None
        )
    )
    assert _corpus() != _corpus(
        lexicon=LexiconReport(
            {"N": Histogram({1: 2})},
            Histogram({1: 2}),
            Histogram({1: 2}),
            None,
        ),
        inventory=InventoryReport(
            {"N": 0},
            {"N": 0},
            {"N": 0},
            ("N",),
            max_arguments=None,
            max_order=None,
        ),
    )
    assert _corpus() != _corpus(
        inventory=InventoryReport(
            {"NP": 0},
            {"NP": 0},
            {"NP": 0},
            (),
            max_arguments=None,
            max_order=None,
        )
    )
    assert _corpus() != _corpus(
        corrupted=(_corrupted("b", "a"),), corruption=_report(1, 2)
    )
    assert _corpus() != _corpus(
        labels=_labels((_example("a", "b"),), (), frozenset({"a b"})),
        noise=NoiseReport(
            LabelNoise(grammatical=0.5, ungrammatical=0.5),
            items={"grammatical": 1},
            flipped={"grammatical": 1},
        ),
    )
    assert _corpus() != _corpus(
        noise=NoiseReport(
            LabelNoise(grammatical=0.0, ungrammatical=0.0),
            items={"grammatical": 1},
            flipped={},
        )
    )
    assert _corpus() != _corpus(forms=(GoldForm(ItemId("a b"), "b(a)"),))
    assert _corpus().__eq__(DECLARATION) is NotImplemented


def test_a_corpus_holds_the_forms_of_its_items() -> None:
    examples = (_example("a", "b"), _example("c"))
    forms = (GoldForm(ItemId("a b"), "b(a)"), GoldForm(ItemId("c"), "c"))
    corpus = _corpus(
        examples=examples,
        length=_length(achieved=Histogram({1: 1, 2: 1})),
        forms=forms,
    )

    assert corpus.forms == {ItemId("a b"): forms[0], ItemId("c"): forms[1]}
    assert tuple(corpus.forms) == (ItemId("a b"), ItemId("c"))
    assert corpus == _corpus(
        examples=examples,
        length=_length(achieved=Histogram({1: 1, 2: 1})),
        forms=forms,
    )


def test_a_corpus_admits_forms_in_any_order_that_names_each_item_once() -> None:
    examples = (_example("a", "b"), _example("c"))
    forms = (GoldForm(ItemId("c"), "c"), GoldForm(ItemId("a b"), "b(a)"))
    corpus = _corpus(
        examples=examples,
        length=_length(achieved=Histogram({1: 1, 2: 1})),
        forms=forms,
    )

    assert tuple(corpus.forms) == (ItemId("c"), ItemId("a b"))
    assert corpus.forms[ItemId("a b")].text == "b(a)"


@pytest.mark.parametrize(
    ("build", "fault"),
    [
        (
            lambda: _corpus(forms=(GoldForm(ItemId("c"), "c"),)),
            "form 'c' names no grammatical item",
        ),
        (
            lambda: _corpus(
                forms=(GoldForm(ItemId("a b"), "b(a)"), GoldForm(ItemId("a b"), "x"))
            ),
            "item 'a b' bears two logical forms",
        ),
        (
            lambda: _corpus(
                examples=(_example("a", "b"), _example("c")),
                length=_length(achieved=Histogram({1: 1, 2: 1})),
                forms=(GoldForm(ItemId("c"), "c"),),
            ),
            "1 grammatical items bear no logical form",
        ),
        (
            lambda: _corpus(
                corrupted=(_corrupted("b", "a"),),
                corruption=_report(1, 2),
                forms=(
                    GoldForm(ItemId("a b"), "b(a)"),
                    GoldForm(ItemId("!b a"), "a(b)"),
                ),
            ),
            "form '!b a' names no grammatical item",
        ),
    ],
)
def test_a_corpus_refuses_forms_that_disagree_with_its_items(
    build: Callable[[], SyntheticCorpus], fault: str
) -> None:
    with pytest.raises(ValueError, match=fault):
        build()


@pytest.mark.parametrize(
    ("build", "error", "fault"),
    [
        (lambda: _corpus(declaration=b""), ValueError, "empty declaration"),
        (lambda: _corpus(theory=b""), ValueError, "empty theory"),
        (lambda: _corpus(seed=True), TypeError, "seed is not an integer"),
        (
            lambda: _corpus(length=_length(max_length=1)),
            ValueError,
            "has 2 tokens, outside the bounds 1 to 1",
        ),
        (
            lambda: _corpus(length=_length(3)),
            ValueError,
            "has 2 tokens, outside the bounds 3 to 3",
        ),
        (
            lambda: _corpus(length=_length(achieved=Histogram({2: 2}))),
            ValueError,
            r"holds Histogram\(\{2: 2\}\) but the grammatical items measure",
        ),
        (
            lambda: _corpus(
                corrupted=(_corrupted("b", "a", "c", "d"),), corruption=_report(1, 2)
            ),
            ValueError,
            "has 4 tokens, more than the bound 3",
        ),
        (
            lambda: _corpus(corrupted=(_corrupted("b", "a"),)),
            ValueError,
            "1 corrupted items are held without a report",
        ),
        (
            lambda: _corpus(
                corrupted=(_corrupted("b", "a"),), corruption=_report(1, 3)
            ),
            ValueError,
            "counts a sample of 3 but the corpus holds 2",
        ),
        (
            lambda: _corpus(labels=()),
            ValueError,
            "1 items have no label record",
        ),
        (
            lambda: _corpus(labels=_labels((_example("a", "b"), _example("c")), ())),
            ValueError,
            "label record 'c' names no item",
        ),
        (
            lambda: _corpus(labels=_labels((_example("a", "b"),) * 2, ())),
            ValueError,
            "item 'a b' is labeled twice",
        ),
        (
            lambda: _corpus(
                examples=(_example("a", "b"), _example("a", "b")),
                length=_length(achieved=Histogram({2: 2})),
                labels=_labels((_example("a", "b"),), ()),
            ),
            ValueError,
            "1 items share an identifier",
        ),
        (
            lambda: _corpus(
                labels=(
                    ObservedLabel.of(ItemId("a b"), "ungrammatical", flipped=False),
                )
            ),
            ValueError,
            "item 'a b' is grammatical but its record says 'ungrammatical'",
        ),
        (
            lambda: _corpus(
                labels=_labels((_example("a", "b"),), (), frozenset({"a b"}))
            ),
            ValueError,
            "1 labels are flipped without a noise report",
        ),
        (
            lambda: _corpus(
                noise=NoiseReport(
                    LabelNoise(grammatical=0.5, ungrammatical=0.5),
                    items={"grammatical": 1},
                    flipped={},
                ),
                labels=_labels((_example("a", "b"),), (), frozenset({"a b"})),
            ),
            ValueError,
            "the noise report holds .* but the labels measure",
        ),
    ],
)
def test_a_corpus_refuses_empty_bytes_crossed_bounds_and_unreported_items(
    build: Callable[[], SyntheticCorpus], error: type[Exception], fault: str
) -> None:
    with pytest.raises(error, match=fault):
        build()


def test_a_corpus_projects_every_item_onto_its_observed_label_in_draw_order() -> None:
    examples = (_example("a", "b"), _example("c"))
    corrupted = (_corrupted("b", "a"),)
    labels = _labels(examples, corrupted, frozenset({"c", "!b a"}))
    noise = NoiseReport(
        LabelNoise(grammatical=0.5, ungrammatical=0.5),
        items={"grammatical": 2, "ungrammatical": 1},
        flipped={"grammatical": 1, "ungrammatical": 1},
    )
    corpus = _corpus(
        examples=examples,
        length=_length(achieved=Histogram({1: 1, 2: 1})),
        corrupted=corrupted,
        corruption=_report(1, 3),
        labels=labels,
        noise=noise,
    )
    inventory = frozenset(GRAMMATICALITY_LABELS)

    assert corpus.labels is labels
    assert corpus.noise is noise
    assert corpus.noise == NoiseReport.of(
        LabelNoise(grammatical=0.5, ungrammatical=0.5), labels
    )
    assert corpus.observed() == (
        Example(examples[0].item, Categorical("grammatical", inventory)),
        Example(examples[1].item, Categorical("ungrammatical", inventory)),
        Example(corrupted[0].item, Categorical("grammatical", inventory)),
    )
    assert [label.true_label for label in corpus.labels] == [
        "grammatical",
        "grammatical",
        "ungrammatical",
    ]


def test_a_corpus_admits_the_labels_in_any_order_that_names_each_item_once() -> None:
    examples = (_example("a", "b"), _example("c"))
    labels = _labels(examples, ())
    corpus = _corpus(
        examples=examples,
        length=_length(achieved=Histogram({1: 1, 2: 1})),
        labels=labels[::-1],
    )

    assert corpus.labels == labels[::-1]
    assert [example.item.item_id for example in corpus.observed()] == ["c", "a b"]


def test_a_corpus_admits_a_corrupted_item_shorter_than_its_lower_bound() -> None:
    corpus = _corpus(
        length=_length(2),
        corrupted=(_corrupted("a"),),
        corruption=_report(1, 2),
    )

    assert len(corpus.corrupted[0].item.tokens) == 1


def test_a_length_report_holds_its_histogram_resamples_and_distribution() -> None:
    distribution = NegativeBinomialLength(2, 0.5, min_length=2, max_length=9)
    report = LengthReport(
        Histogram({2: 3, 4: 1}), resampled=2, distribution=distribution
    )

    assert report.achieved == Histogram({2: 3, 4: 1})
    assert report.resampled == 2
    assert report.distribution is distribution
    assert repr(report) == (
        "LengthReport(Histogram({2: 3, 4: 1}), resampled=2, "
        "distribution=NegativeBinomialLength(2.0, 0.5, min_length=2, max_length=9, "
        "out_of_support='resample'))"
    )


def test_a_length_report_is_equal_on_all_three_fields() -> None:
    assert _length() == LENGTH
    assert hash(_length()) == hash(LENGTH)
    assert _length(achieved=Histogram({3: 1})) != LENGTH
    assert _length(max_length=4) != LENGTH
    assert (
        LengthReport(Histogram({2: 1}), resampled=1, distribution=UniformLength(1, 3))
        != LENGTH
    )
    assert (
        LengthReport(
            Histogram({2: 1}),
            resampled=0,
            distribution=NegativeBinomialLength(1, 0.5, max_length=3),
        )
        != LENGTH
    )
    assert LENGTH.__eq__(REPORT) is NotImplemented


@pytest.mark.parametrize(
    ("resampled", "error", "fault"),
    [
        (-1, ValueError, "resampled is negative"),
        (cast("int", 1.5), TypeError, "resampled is not an integer"),
        (True, TypeError, "resampled is not an integer"),
    ],
)
def test_a_length_report_refuses_a_bad_resample_count(
    resampled: int, error: type[Exception], fault: str
) -> None:
    with pytest.raises(error, match=fault):
        LengthReport(
            Histogram({2: 1}), resampled=resampled, distribution=UniformLength(1, 3)
        )


def test_a_label_noise_request_holds_one_rate_per_true_label() -> None:
    noise = LabelNoise(grammatical=0.1, ungrammatical=0.3)

    assert noise.grammatical == 0.1
    assert noise.ungrammatical == 0.3
    assert noise.rate("grammatical") == 0.1
    assert noise.rate("ungrammatical") == 0.3
    assert noise == LabelNoise(grammatical=0.1, ungrammatical=0.3)
    assert hash(noise) == hash(LabelNoise(grammatical=0.1, ungrammatical=0.3))
    assert noise != LabelNoise(grammatical=0.3, ungrammatical=0.1)
    assert noise != LabelNoise(grammatical=0.1, ungrammatical=0.1)
    assert noise.__eq__(0.1) is NotImplemented
    assert repr(noise) == "LabelNoise(grammatical=0.1, ungrammatical=0.3)"
    assert LabelNoise(grammatical=0, ungrammatical=1).rate("grammatical") == 0.0
    assert LabelNoise(grammatical=0, ungrammatical=1).rate("ungrammatical") == 1.0


@pytest.mark.parametrize(
    ("rate", "error"), [(-0.1, ValueError), (1.5, ValueError), (nan, ValueError)]
)
def test_a_label_noise_request_refuses_a_rate_outside_the_unit_interval(
    rate: float, error: type[Exception]
) -> None:
    with pytest.raises(error, match="a grammatical label-noise rate"):
        LabelNoise(grammatical=rate, ungrammatical=0.0)

    with pytest.raises(error, match="an ungrammatical label-noise rate"):
        LabelNoise(grammatical=0.0, ungrammatical=rate)


def test_an_observed_label_holds_the_truth_the_observation_and_the_flip() -> None:
    kept = ObservedLabel(ItemId("x"), "grammatical", "grammatical", flipped=False)
    flipped = ObservedLabel(ItemId("x"), "grammatical", "ungrammatical", flipped=True)

    assert kept.item_id == "x"
    assert kept.true_label == "grammatical"
    assert kept.observed_label == "grammatical"
    assert kept.flipped is False
    assert flipped.observed_label == "ungrammatical"
    assert flipped.flipped is True
    assert kept == ObservedLabel.of(ItemId("x"), "grammatical", flipped=False)
    assert flipped == ObservedLabel.of(ItemId("x"), "grammatical", flipped=True)
    assert ObservedLabel.of(ItemId("x"), "ungrammatical", flipped=True) == (
        ObservedLabel(ItemId("x"), "ungrammatical", "grammatical", flipped=True)
    )
    assert hash(kept) == hash(
        ObservedLabel.of(ItemId("x"), "grammatical", flipped=False)
    )
    assert kept != flipped
    assert kept != ObservedLabel.of(ItemId("y"), "grammatical", flipped=False)
    assert kept != ObservedLabel.of(ItemId("x"), "ungrammatical", flipped=False)
    assert kept.__eq__("x") is NotImplemented
    assert repr(flipped) == (
        "ObservedLabel('x', 'grammatical', 'ungrammatical', flipped=True)"
    )


@pytest.mark.parametrize(
    ("true_label", "observed_label", "flipped", "fault"),
    [
        ("grammatical", "grammatical", True, "flip indicator is True"),
        ("grammatical", "ungrammatical", False, "flip indicator is False"),
        ("ungrammatical", "grammatical", False, "flip indicator is False"),
        ("acceptable", "grammatical", True, "true label of 'x' is not"),
        ("grammatical", "bad", True, "observed label of 'x' is not"),
    ],
)
def test_an_observed_label_refuses_a_disagreeing_triple_or_an_unknown_label(
    true_label: str, observed_label: str, *, flipped: bool, fault: str
) -> None:
    with pytest.raises(ValueError, match=fault):
        ObservedLabel(
            ItemId("x"),
            cast("GrammaticalityLabel", true_label),
            cast("GrammaticalityLabel", observed_label),
            flipped=flipped,
        )


def test_a_noise_report_counts_flips_by_true_label_and_derives_its_rates() -> None:
    report = NoiseReport(
        LabelNoise(grammatical=0.25, ungrammatical=0.25),
        items={"grammatical": 6, "ungrammatical": 2},
        flipped={"grammatical": 1, "ungrammatical": 1},
    )

    assert report.requested == LabelNoise(grammatical=0.25, ungrammatical=0.25)
    assert dict(report.items) == {"grammatical": 6, "ungrammatical": 2}
    assert dict(report.flipped) == {"grammatical": 1, "ungrammatical": 1}
    assert dict(report.observed) == {"grammatical": 6, "ungrammatical": 2}
    assert report.sample_size == 8
    assert report.flip_count == 2
    assert report.achieved_rate == 0.25
    assert dict(report.achieved_rates) == {"grammatical": 1 / 6, "ungrammatical": 0.5}
    assert tuple(report.items) == GRAMMATICALITY_LABELS
    assert report == NoiseReport(
        LabelNoise(grammatical=0.25, ungrammatical=0.25),
        items={"ungrammatical": 2, "grammatical": 6},
        flipped={"ungrammatical": 1, "grammatical": 1},
    )
    assert hash(report) == hash(
        NoiseReport(
            LabelNoise(grammatical=0.25, ungrammatical=0.25),
            items={"grammatical": 6, "ungrammatical": 2},
            flipped={"grammatical": 1, "ungrammatical": 1},
        )
    )
    assert report != NoiseReport(
        LabelNoise(grammatical=0.5, ungrammatical=0.5),
        items={"grammatical": 6, "ungrammatical": 2},
        flipped={"grammatical": 1, "ungrammatical": 1},
    )
    assert report != NoiseReport(
        LabelNoise(grammatical=0.25, ungrammatical=0.25),
        items={"grammatical": 6, "ungrammatical": 2},
        flipped={"grammatical": 2},
    )
    assert (
        report.__eq__(LabelNoise(grammatical=0.25, ungrammatical=0.25))
        is NotImplemented
    )
    assert repr(report) == (
        "NoiseReport(LabelNoise(grammatical=0.25, ungrammatical=0.25), "
        "items={'grammatical': 6, 'ungrammatical': 2}, "
        "flipped={'grammatical': 1, 'ungrammatical': 1})"
    )


def test_a_noise_report_measured_over_records_counts_what_they_hold() -> None:
    labels = (
        ObservedLabel.of(ItemId("a"), "grammatical", flipped=True),
        ObservedLabel.of(ItemId("b"), "grammatical", flipped=False),
        ObservedLabel.of(ItemId("c"), "ungrammatical", flipped=False),
        ObservedLabel.of(ItemId("d"), "grammatical", flipped=True),
    )
    report = NoiseReport.of(LabelNoise(grammatical=0.5, ungrammatical=0.5), labels)

    assert report == NoiseReport(
        LabelNoise(grammatical=0.5, ungrammatical=0.5),
        items={"grammatical": 3, "ungrammatical": 1},
        flipped={"grammatical": 2},
    )
    assert dict(report.observed) == {"grammatical": 1, "ungrammatical": 3}
    assert report.achieved_rate == 0.5

    empty = NoiseReport.of(LabelNoise(grammatical=0.5, ungrammatical=0.5), ())

    assert empty.sample_size == 0
    assert empty.flip_count == 0
    assert empty.achieved_rate == 0.0
    assert dict(empty.achieved_rates) == {"grammatical": 0.0, "ungrammatical": 0.0}


@pytest.mark.parametrize(
    ("items", "flipped", "error", "fault"),
    [
        ({"grammatical": 1}, {"grammatical": 2}, ValueError, "only 1 bear"),
        ({"maybe": 1}, {}, ValueError, "unknown labels"),
        ({}, {"grammatical": -1}, ValueError, "negative"),
        ({"grammatical": 1.5}, {}, TypeError, "not an integer"),
    ],
)
def test_a_noise_report_refuses_impossible_counts(
    items: dict[str, int],
    flipped: dict[str, int],
    error: type[Exception],
    fault: str,
) -> None:
    with pytest.raises(error, match=fault):
        NoiseReport(
            LabelNoise(grammatical=0.5, ungrammatical=0.5),
            items=cast("dict[GrammaticalityLabel, int]", items),
            flipped=cast("dict[GrammaticalityLabel, int]", flipped),
        )


def test_the_exhaustion_errors_are_sampling_exhaustions() -> None:
    assert issubclass(UnproductiveGrammarError, SamplingExhaustedError)
    assert issubclass(AmbiguityUnreachableError, SamplingExhaustedError)
    assert issubclass(CorruptionExhaustedError, SamplingExhaustedError)
    assert issubclass(UncertifiedDerivationError, RuntimeError)
    assert not issubclass(UncertifiedDerivationError, SamplingExhaustedError)
    assert issubclass(UninterpretableDerivationError, RuntimeError)
    assert issubclass(IllTypedFormError, RuntimeError)
    assert not issubclass(IllTypedFormError, SamplingExhaustedError)


def _assembly(name: str = "toy") -> GrammarAssembly:
    return GrammarIso().forward(
        GrammarDecl(
            name=name,
            signature=SignatureSpec(
                constructors=slashes(),
                functors=slash_views(),
                primitives=("S", "NP", "conj"),
                operators=(OperatorSpec(name="NP[conj]", sort="Cat", erases_to="NP"),),
                conjunctions=ConjunctionsSpec(
                    markers=("conj",),
                    pairs=(ConjunctionPairSpec(unmarked="NP", marked="NP[conj]"),),
                ),
            ),
            rules=RuleSetSpec(
                name="application",
                rules=(
                    ApplicationSpec(view="/"),
                    ApplicationSpec(view="\\"),
                ),
            ),
            lexicon=InlineLexiconSpec(
                entries=(
                    LexicalEntrySpec(form="a", category="NP", meaning="a"),
                    LexicalEntrySpec(form="b", category="S\\NP", meaning="\\x. b(x)"),
                )
            ),
            matching=FeatureErasureMatchingSpec(),
            budget=BudgetSpec(unary_rounds=1),
        ),
        registered_language("propositional"),
    )


def _interpretation() -> InterpretationSpec:
    return InterpretationSpec(
        erasure=SignatureMorphismErasureSpec(atoms={"S": "t", "NP": "e", "conj": "e"}),
        constants=(
            ConstantSpec(name="a", type="e", classes=("individual",)),
            ConstantSpec(name="b", type="e -> t", classes=("predicate",)),
        ),
        meanings=MeaningSpec(source=LexicalMeaningSourceSpec()),
    )


def _dataset(
    declaration: bytes = b'{"kind":"synthetic"}',
    *,
    assembly: GrammarAssembly | None = None,
    seed: int = 3,
    interpreted: bool = False,
) -> SyntheticDataset:
    held = _assembly() if assembly is None else assembly
    interpretation = None

    if interpreted:
        languages = LanguageIso()
        grammars = GrammarIso(languages)
        held, interpretation = lower_interpretation(
            GrammarIso().backward(held),
            registered_language("propositional"),
            _interpretation(),
            grammars,
            languages,
        )
    corpus = SyntheticGenerator(
        held,
        policy=ParsePolicy(Retain.FOREST, Reachability.ALL),
        interpretation=interpretation,
    ).sample(4, seed=seed, length=UniformLength(1, 2))

    return SyntheticDataset(
        declaration,
        held,
        corpus,
        source_version="decl-1",
        target_version="native-1",
        source_interpretation=interpretation,
    )


def test_a_dataset_holds_its_declaration_grammar_corpus_and_route() -> None:
    assembly = _assembly()
    dataset = _dataset(assembly=assembly)

    assert dataset.declaration == b'{"kind":"synthetic"}'
    assert dataset.fingerprint == sha256(b'{"kind":"synthetic"}').hexdigest()
    assert dataset.source_grammar is assembly
    assert dataset.corpus.declaration == assembly.declaration
    assert dataset.corpus.seed == 3
    assert dataset.source_version == "decl-1"
    assert dataset.target_version == "native-1"
    assert dataset.source_interpretation is None
    assert len(dataset) == len(dataset.corpus) == 4
    assert repr(dataset) == (
        f"SyntheticDataset(fingerprint={dataset.fingerprint[:12]!r}, "
        f"corpus={dataset.corpus!r})"
    )


def test_a_dataset_is_equal_on_its_declaration_and_corpus() -> None:
    assert _dataset() == _dataset()
    assert hash(_dataset()) == hash(_dataset())
    assert _dataset() != _dataset(b'{"kind":"synthetic","seed":1}')
    assert _dataset() != _dataset(seed=4)
    assert _dataset() != _dataset(assembly=_assembly("other"))


def test_a_dataset_holds_the_interpretation_its_forms_were_read_under() -> None:
    dataset = _dataset(interpreted=True)

    assert dataset.source_interpretation is not None
    assert len(dataset.corpus.forms) == 4
    assert dataset == _dataset(interpreted=True)
    assert dataset != _dataset()


def test_a_dataset_refuses_forms_against_its_interpretation() -> None:
    interpreted = _dataset(interpreted=True)
    plain = _dataset()

    with pytest.raises(ValueError, match="dataset holds no interpretation"):
        SyntheticDataset(
            b"{}",
            interpreted.source_grammar,
            interpreted.corpus,
            source_version="decl-1",
            target_version="native-1",
        )

    with pytest.raises(ValueError, match="corpus holds no forms"):
        SyntheticDataset(
            b"{}",
            plain.source_grammar,
            plain.corpus,
            source_version="decl-1",
            target_version="native-1",
            source_interpretation=interpreted.source_interpretation,
        )


def test_a_dataset_refuses_empty_bytes_or_a_corpus_of_another_grammar() -> None:
    with pytest.raises(ValueError, match="empty declaration bytes"):
        _dataset(b"")

    corpus = _dataset().corpus

    with pytest.raises(ValueError, match="drawn under grammar"):
        SyntheticDataset(
            b"{}",
            _assembly("other"),
            corpus,
            source_version="decl-1",
            target_version="native-1",
        )


SCOPE_TARGET: Final = ScopeTarget(readings=Target(2, 0), derivations=16, attempts=4)


def test_a_scope_target_holds_its_band_and_bounds() -> None:
    assert SCOPE_TARGET.readings == Target(2, 0)
    assert SCOPE_TARGET.derivations == 16
    assert SCOPE_TARGET.attempts == 4
    assert ScopeTarget(derivations=1, attempts=1).readings is None
    assert (
        ScopeTarget(readings=Target(2, 0), derivations=16, attempts=4) == SCOPE_TARGET
    )
    assert hash(SCOPE_TARGET) == hash(
        ScopeTarget(readings=Target(2, 0), derivations=16, attempts=4)
    )
    assert ScopeTarget(derivations=16, attempts=4) != SCOPE_TARGET
    assert ScopeTarget(readings=Target(2, 0), derivations=8, attempts=4) != SCOPE_TARGET
    assert SCOPE_TARGET.__eq__(16) is NotImplemented
    assert repr(SCOPE_TARGET) == (
        "ScopeTarget(readings=Target(2.0, 0.0), derivations=16, attempts=4, "
        "undecided='distinct')"
    )
    assert (
        ScopeTarget(
            readings=Target(2, 0),
            derivations=16,
            attempts=4,
            undecided="equivalent",
        )
        != SCOPE_TARGET
    )


@pytest.mark.parametrize(
    ("derivations", "attempts", "error", "fault"),
    [
        (0, 1, ValueError, "derivations is not positive"),
        (1, 0, ValueError, "attempts is not positive"),
        (1, 1.0, TypeError, "not an integer"),
    ],
)
def test_a_scope_target_refuses_a_bound_outside_its_range(
    derivations: int, attempts: int, error: type[Exception], fault: str
) -> None:
    with pytest.raises(error, match=fault):
        ScopeTarget(derivations=derivations, attempts=attempts)


def test_a_reading_count_holds_the_two_ambiguities_of_one_string() -> None:
    count = ReadingCount(ItemId("a"), derivations=6, readings=2)

    assert count.item_id == ItemId("a")
    assert count.derivations == 6
    assert count.readings == 2
    assert count == ReadingCount(ItemId("a"), derivations=6, readings=2)
    assert hash(count) == hash(ReadingCount(ItemId("a"), derivations=6, readings=2))
    assert count != ReadingCount(ItemId("b"), derivations=6, readings=2)
    assert count != ReadingCount(ItemId("a"), derivations=6, readings=1)
    assert count.__eq__(6) is NotImplemented
    assert repr(count) == "ReadingCount('a', derivations=6, readings=2)"


@pytest.mark.parametrize(
    ("derivations", "readings", "error", "fault"),
    [
        (0, 1, ValueError, "not positive"),
        (1, 0, ValueError, "not positive"),
        (2, 3, ValueError, "3 readings from 2 derivations"),
        (2.0, 1, TypeError, "not an integer"),
    ],
)
def test_a_reading_count_refuses_counts_that_cannot_be_measured(
    derivations: int, readings: int, error: type[Exception], fault: str
) -> None:
    with pytest.raises(error, match=fault):
        ReadingCount(ItemId("a"), derivations=derivations, readings=readings)


def test_a_scope_report_summarizes_the_counts_it_holds() -> None:
    counts = (
        ReadingCount(ItemId("a"), derivations=6, readings=2),
        ReadingCount(ItemId("b"), derivations=3, readings=1),
        ReadingCount(ItemId("c"), derivations=6, readings=2),
    )
    report = ScopeReport(
        counts, rejected=4, exceeded=1, undecided=2, target=SCOPE_TARGET
    )

    assert report.counts == counts
    assert report.readings == Histogram({1: 1, 2: 2})
    assert report.derivations == Histogram({3: 1, 6: 2})
    assert report.rejected == 4
    assert report.exceeded == 1
    assert report.undecided == 2
    assert report.target is SCOPE_TARGET
    assert report == ScopeReport(
        counts, rejected=4, exceeded=1, undecided=2, target=SCOPE_TARGET
    )
    assert hash(report) == hash(
        ScopeReport(counts, rejected=4, exceeded=1, undecided=2, target=SCOPE_TARGET)
    )
    assert report != ScopeReport(
        counts, rejected=4, exceeded=1, undecided=3, target=SCOPE_TARGET
    )
    assert report != ScopeReport(
        counts[:2], rejected=4, exceeded=1, undecided=2, target=SCOPE_TARGET
    )
    assert report.__eq__(counts) is NotImplemented
    assert repr(report) == (
        f"ScopeReport({counts!r}, rejected=4, exceeded=1, undecided=2, "
        f"target={SCOPE_TARGET!r})"
    )


def test_a_scope_report_refuses_a_repeated_item_or_a_negative_count() -> None:
    count = ReadingCount(ItemId("a"), derivations=2, readings=1)

    with pytest.raises(ValueError, match="counts an item twice"):
        ScopeReport(
            (count, count), rejected=0, exceeded=0, undecided=0, target=SCOPE_TARGET
        )

    with pytest.raises(ValueError, match="rejected is negative"):
        ScopeReport((), rejected=-1, exceeded=0, undecided=0, target=SCOPE_TARGET)


def test_a_corpus_refuses_a_scope_report_that_names_other_items() -> None:
    examples = (_example("a", "b"),)
    drawn = examples[0].item.item_id
    corpus = _corpus(
        examples=examples,
        scope=ScopeReport(
            (ReadingCount(drawn, derivations=2, readings=1),),
            rejected=0,
            exceeded=0,
            undecided=0,
            target=SCOPE_TARGET,
        ),
    )

    assert corpus.scope is not None
    assert corpus.scope.readings == Histogram({1: 1})
    assert _corpus(examples=examples).scope is None

    with pytest.raises(ValueError, match="scope report counts 0 items"):
        _corpus(
            examples=examples,
            scope=ScopeReport(
                (), rejected=0, exceeded=0, undecided=0, target=SCOPE_TARGET
            ),
        )


def test_a_corpus_holds_one_shape_measurement_per_grammatical_item() -> None:
    target = ShapeTarget(depth=Target(3, 1), attempts=4)
    schedule = ShapeSchedule(target, {})
    report = ShapeReport(
        (ShapeMeasurement(3, 0.0, 0.5, 2.0),), rejected=1, target=schedule
    )
    corpus = _corpus(shape=report)

    assert corpus.shape is report

    with pytest.raises(ValueError, match="shape report measures 0 items"):
        _corpus(shape=ShapeReport((), rejected=0, target=schedule))


def test_the_scope_refusal_is_a_sampling_exhaustion() -> None:
    assert issubclass(ScopeUnreachableError, SamplingExhaustedError)


def test_an_ill_typing_report_holds_one_injection_report_and_its_certificates() -> None:
    forms = (_ill_typed("a b"), _ill_typed("c d"))
    report = IllTypingReport.of(_injection(2, 8, rejected=1), forms)

    assert report.injection == _injection(2, 8, rejected=1)
    assert report.injection.achieved_proportion == 0.25
    assert report.injection.rejection_rate == 1 / 3
    assert dict(report.certificates) == {"unification": 2}
    assert dict(report.observed) == {"well-typed": 0, "ill-typed": 2}
    assert report.flipped == 0
    assert report.achieved_noise == 0.0


def test_an_ill_typing_report_counts_every_conflict_of_every_form() -> None:
    other = TypingFailure(
        (CONFLICT, TypingConflict("infinite_type", "e", "t", ())),
        language_digest="l" * 64,
    )
    report = IllTypingReport.of(
        _injection(2, 8),
        (
            _ill_typed(),
            IllTyped(
                ItemId("c"),
                "sort-mismatch",
                "c",
                "c(c)",
                other,
                observed_label="ill-typed",
            ),
        ),
    )

    assert dict(report.certificates) == {"infinite_type": 1, "unification": 2}


def test_an_ill_typing_report_is_equal_on_its_injection_and_certificates() -> None:
    report = IllTypingReport.of(_injection(1, 8), (_ill_typed(),))

    assert report == IllTypingReport.of(_injection(1, 8), (_ill_typed(),))
    assert hash(report) == hash(IllTypingReport.of(_injection(1, 8), (_ill_typed(),)))
    assert report != IllTypingReport.of(
        _injection(2, 8), (_ill_typed("a b"), _ill_typed("c d"))
    )
    assert report != IllTypingReport.of(
        _injection(1, 8), (_ill_typed(observed_label="well-typed"),)
    )
    assert report.__eq__(report.injection) is NotImplemented
    assert repr(report) == (
        f"IllTypingReport({report.injection!r}, certificates={{'unification': 1}}, "
        "observed={'well-typed': 0, 'ill-typed': 1})"
    )


@pytest.mark.parametrize(
    ("build", "error", "fault"),
    [
        (
            lambda: IllTypingReport(
                _injection(1, 8),
                certificates={"unification": 0},
                observed={"ill-typed": 1},
            ),
            ValueError,
            "certificates\\['unification'\\] is not positive",
        ),
        (
            lambda: IllTypingReport(
                _injection(1, 8),
                certificates={"unification": -1},
                observed={"ill-typed": 1},
            ),
            ValueError,
            "certificates\\['unification'\\] is negative",
        ),
        (
            lambda: IllTypingReport(
                _injection(1, 8),
                certificates={"unification": cast("int", 1.0)},
                observed={"ill-typed": 1},
            ),
            TypeError,
            "not an integer",
        ),
    ],
)
def test_an_ill_typing_report_refuses_a_count_that_is_not_a_positive_integer(
    build: Callable[[], IllTypingReport], error: type[Exception], fault: str
) -> None:
    with pytest.raises(error, match=fault):
        build()


def test_a_corpus_holds_the_injected_forms_and_what_the_injection_achieved() -> None:
    forms = (GoldForm(ItemId("a b"), "b(a)"),)
    injected = (_ill_typed(),)
    report = IllTypingReport.of(_injection(1, 1), injected)
    corpus = _corpus(forms=forms, ill_typed=injected, ill_typing=report)

    assert corpus.ill_typed == injected
    assert corpus.ill_typing == report
    assert report.injection.achieved_proportion == 1.0
    assert _corpus(forms=forms).ill_typed == ()
    assert _corpus(forms=forms).ill_typing is None
    assert corpus != _corpus(forms=forms)

    assert tuple(corpus.form_typing) == (ItemId("a b"),)
    assert tuple(example.text for example in corpus.form_typing[ItemId("a b")]) == (
        "b(a)",
        "a(b)",
    )
    assert tuple(
        example.observed_label for example in corpus.form_typing[ItemId("a b")]
    ) == ("well-typed", "ill-typed")


@pytest.mark.parametrize(
    ("build", "fault"),
    [
        (
            lambda: _corpus(ill_typed=(_ill_typed(),)),
            "bears no logical form",
        ),
        (
            lambda: _corpus(
                forms=(GoldForm(ItemId("a b"), "b(a)"),),
                ill_typed=(_ill_typed(),),
            ),
            "injected forms are held without a report",
        ),
        (
            lambda: _corpus(
                forms=(GoldForm(ItemId("a b"), "x(y)"),),
                ill_typed=(_ill_typed(),),
                ill_typing=IllTypingReport.of(_injection(1, 1), (_ill_typed(),)),
            ),
            "corrupts 'b\\(a\\)' but the item bears 'x\\(y\\)'",
        ),
        (
            lambda: _corpus(
                forms=(GoldForm(ItemId("a b"), "b(a)"),),
                ill_typed=(_ill_typed(),),
                ill_typing=IllTypingReport.of(_injection(1, 4), (_ill_typed(),)),
            ),
            "counts a sample of 4 but the corpus holds 1 grammatical",
        ),
        (
            lambda: _corpus(
                forms=(GoldForm(ItemId("a b"), "b(a)"),),
                ill_typed=(
                    IllTyped(
                        ItemId("a b"),
                        "sort-mismatch",
                        "b(a)",
                        "a(b)",
                        FAILURE,
                        observed_label="ill-typed",
                    ),
                ),
                ill_typing=IllTypingReport.of(_injection(1, 1), (_ill_typed(),)),
            ),
            "but the injected forms were drawn by",
        ),
        (
            lambda: _corpus(
                forms=(GoldForm(ItemId("a b"), "b(a)"),),
                ill_typed=(_ill_typed(),),
                ill_typing=IllTypingReport(
                    _injection(1, 1),
                    certificates={"infinite_type": 1},
                    observed={"ill-typed": 1},
                ),
            ),
            "but the injected forms measure",
        ),
    ],
)
def test_a_corpus_refuses_an_injection_its_forms_and_report_disagree_with(
    build: Callable[[], SyntheticCorpus], fault: str
) -> None:
    with pytest.raises(ValueError, match=fault):
        build()
