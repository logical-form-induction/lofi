from copy import copy, deepcopy
from math import log
from pickle import dumps
from typing import TYPE_CHECKING, Final, cast

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from tests.conformance.interpretation import InterpretationSuite
from tests.support.derivations import FormLexicon, axiom
from tests.support.derivations import step as applied

from lofi._native import PythonCarrier
from lofi.deduce import Derivation, Edge, ItemKey, Viterbi
from lofi.interpret import Interpretation, read
from lofi.interpret._readings import (
    CategoryReading,
    HeadDependencies,
    HeadLink,
    StringReading,
    WeightReading,
)
from lofi.interpret.laws import InterpretationFixture
from lofi.lexicon import Entry

if TYPE_CHECKING:
    from lofi.judge import Rule


HEADS = {"fa": 0, "ba": 1}


class _Rule:
    """Rule fields used by interpretation readings."""

    __slots__ = ("arity", "name")

    def __init__(self, name: str, arity: int) -> None:
        self.name = name
        self.arity = arity


UNARY: Final = cast("Rule[str]", _Rule("unary", 1))
BINARY: Final = cast("Rule[str]", _Rule("binary", 2))
TERNARY: Final = cast("Rule[str]", _Rule("ternary", 3))

ENTRIES: Final = (
    Entry[str]("the", "NP/N", weight=-0.1),
    Entry[str]("red", "N/N", weight=-0.2),
    Entry[str]("fox", "N", weight=-0.3),
    Entry[str]("runs", "S\\NP", weight=-0.4),
)
LEXICON: Final = FormLexicon[str]((UNARY, BINARY, TERNARY), ENTRIES)
THE: Final = axiom(ENTRIES[0])
RED: Final = axiom(ENTRIES[1])
FOX: Final = axiom(ENTRIES[2])
RUNS: Final = axiom(ENTRIES[3])

LEFT_BRANCH: Final = applied(
    BINARY,
    (applied(BINARY, (THE, RED), "NP/N", -0.5), FOX),
    "NP",
    -0.6,
)
RIGHT_BRANCH: Final = applied(
    BINARY,
    (THE, applied(BINARY, (RED, FOX), "N", -0.7)),
    "NP",
    -0.8,
)
FEATURE_PLAIN: Final = applied(UNARY, (FOX,), "NP", -0.9)
FEATURE_MARKED: Final = applied(UNARY, (FOX,), "NP[nb]", -0.9)
WEIGHT_LIGHT: Final = applied(BINARY, (THE, FOX), "NP", -0.1)
WEIGHT_HEAVY: Final = applied(BINARY, (THE, FOX), "NP", -3.0)

DERIVATION_CASES: Final = (
    THE,
    RED,
    FOX,
    RUNS,
    applied(UNARY, (FOX,), "NP", -0.1),
    applied(UNARY, (RUNS,), "S[dcl]\\NP", -0.2, 1),
    applied(BINARY, (THE, FOX), "NP", -0.3),
    applied(BINARY, (RED, FOX), "N", -0.4, 2),
    applied(TERNARY, (THE, RED, FOX), "NP", -0.5),
    LEFT_BRANCH,
    RIGHT_BRANCH,
    WEIGHT_LIGHT,
    WEIGHT_HEAVY,
    FEATURE_PLAIN,
    FEATURE_MARKED,
    applied(BINARY, (THE, FOX), "NP", -0.6, 1),
    applied(TERNARY, (THE, FOX, RUNS), "S", -0.7, 3),
    applied(
        UNARY,
        (applied(TERNARY, (THE, RED, FOX), "NP", -0.8),),
        "S/(S\\NP)",
        -0.9,
        4,
    ),
)
SAME_STRING_PAIRS: Final = (
    (LEFT_BRANCH, RIGHT_BRANCH),
    (FEATURE_PLAIN, FEATURE_MARKED),
    (WEIGHT_LIGHT, WEIGHT_HEAVY),
)

type Trace = frozenset[tuple[float, ...]]


def _trace(weight: float) -> Trace:
    """Lift a scalar into a singleton trace."""
    return frozenset(((weight,),))


class _TraceSemiring:
    """Free-language semiring over weight traces."""

    __slots__ = ()

    @property
    def name(self) -> str:
        """Return the semiring name."""
        return "trace"

    @property
    def log_probability(self) -> bool:
        """Report whether weights are log probabilities."""
        return False

    @property
    def zero(self) -> Trace:
        """Return the additive identity."""
        return frozenset()

    @property
    def one(self) -> Trace:
        """Return the multiplicative identity."""
        return frozenset(((),))

    @property
    def lowering(self) -> PythonCarrier:
        """Select Python-carried arithmetic."""
        return PythonCarrier()

    def plus(
        self,
        a: Trace,
        b: Trace,
        /,
    ) -> Trace:
        """Combine alternative traces."""
        return a | b

    def times(
        self,
        a: Trace,
        b: Trace,
        /,
    ) -> Trace:
        """Concatenate two traces."""
        return frozenset((*left, *right) for left in a for right in b)

    def is_zero(self, a: Trace, /) -> bool:
        """Return whether the trace is empty."""
        return not a


def key(span: tuple[int, int], category: str) -> ItemKey[str]:
    return ItemKey(span, category)


def leaf(span: tuple[int, int], category: str) -> Derivation[str, float]:
    return Derivation(key(span, category), Edge("lex", (), category, log(0.5)))


def step(
    span: tuple[int, int],
    category: str,
    rule: str,
    children: tuple[Derivation[str, float], ...],
) -> Derivation[str, float]:
    return Derivation(
        key(span, category),
        Edge(rule, tuple(child.key for child in children), category, log(0.8)),
        children,
    )


def sentence() -> Derivation[str, float]:
    subject = step((0, 2), "NP", "fa", (leaf((0, 1), "NP/N"), leaf((1, 2), "N")))
    predicate = step(
        (2, 4),
        "S\\NP",
        "ba",
        (leaf((2, 3), "S\\NP"), leaf((3, 4), "(S\\NP)\\(S\\NP)")),
    )

    return step((0, 4), "S", "ba", (subject, predicate))


def test_readings_cover_diverse_derivation_shapes() -> None:
    categories = CategoryReading[str]()
    strings = StringReading[str]()
    weights = WeightReading[str, float](Viterbi(), lambda weight: weight)

    assert {len(node.children) for node in DERIVATION_CASES if node.children} == {
        1,
        2,
        3,
    }

    for derivation in DERIVATION_CASES:
        assert None not in read(derivation, categories, LEXICON)
        assert None not in read(derivation, strings, LEXICON)
        assert None not in read(derivation, weights, LEXICON)

    assert read(LEFT_BRANCH, strings, LEXICON) == read(RIGHT_BRANCH, strings, LEXICON)
    assert read(FEATURE_PLAIN, categories, LEXICON) != read(
        FEATURE_MARKED, categories, LEXICON
    )
    assert read(WEIGHT_LIGHT, weights, LEXICON) != read(WEIGHT_HEAVY, weights, LEXICON)


def test_the_category_reading_uses_lexical_and_selected_exact_categories() -> None:
    reading: Interpretation[str, str] = CategoryReading()

    assert reading.name == "category"
    assert reading.refines_items
    assert isinstance(reading, Interpretation)
    assert read(THE, reading, LEXICON) == ("NP/N",)
    assert read(FEATURE_MARKED, reading, LEXICON) == ("NP[nb]",)


def test_the_string_reading_reconstructs_forms_in_physical_order() -> None:
    reading: Interpretation[str, tuple[str, ...]] = StringReading()
    derivation = applied(TERNARY, (FOX, THE, RED), "NP", 0.0, 7)

    assert reading.name == "string"
    assert not reading.refines_items
    assert isinstance(reading, Interpretation)
    assert read(derivation, reading, LEXICON) == (("fox", "the", "red"),)


def test_the_weight_reading_lifts_lexical_weights_and_multiplies_in_order() -> None:
    reading = WeightReading[str, Trace](_TraceSemiring(), _trace)
    entries = (
        Entry("a", "A", weight=1.0),
        Entry("b", "B", weight=2.0),
        Entry("c", "C", weight=3.0),
    )
    derivation = applied(
        TERNARY,
        tuple(
            axiom(entry, at, weight=frozenset[tuple[float, ...]]())
            for at, entry in enumerate(entries)
        ),
        "D",
        frozenset(((4.0,),)),
    )

    assert reading.name == "weight"
    assert reading.refines_items
    assert isinstance(reading, Interpretation)
    assert read(derivation, reading, FormLexicon((TERNARY,), entries)) == (
        frozenset(((4.0, 1.0, 2.0, 3.0),)),
    )


def test_readings_are_frozen_identity_copies_and_refuse_pickle() -> None:
    weight_reading = WeightReading[str, float](Viterbi(), lambda weight: weight)
    readings = (
        CategoryReading[str](),
        StringReading[str](),
        weight_reading,
    )

    for reading in readings:
        identity = (reading.name, reading.refines_items)

        assert copy(reading) is reading
        assert deepcopy(reading) is reading

        with pytest.raises(TypeError, match="cannot be pickled"):
            dumps(reading)

        with pytest.raises(AttributeError, match="immutable"):
            reading.__setattr__("_semiring", Viterbi())

        with pytest.raises(AttributeError, match="immutable"):
            reading.__delattr__("name")

        assert (reading.name, reading.refines_items) == identity

    with pytest.raises(AttributeError, match="immutable"):
        weight_reading.__setattr__("_lift", float)

    with pytest.raises(AttributeError, match="immutable"):
        weight_reading.__delattr__("_semiring")


@settings(max_examples=100, deadline=None)
@given(
    st.lists(
        st.text(
            alphabet=st.characters(categories=("Ll", "Lu")),
            min_size=1,
            max_size=8,
        ),
        min_size=3,
        max_size=12,
    )
)
def test_string_reading_agrees_across_binary_packings(forms: list[str]) -> None:
    entries = tuple(Entry(form, "X") for form in forms)
    leaves = tuple(axiom(entry, at) for at, entry in enumerate(entries))
    lexicon = FormLexicon((BINARY,), entries)
    left = leaves[0]

    for leaf_value in leaves[1:]:
        left = applied(BINARY, (left, leaf_value), "X", 0.0)

    right = leaves[-1]

    for leaf_value in reversed(leaves[:-1]):
        right = applied(BINARY, (leaf_value, right), "X", 0.0)

    reading = StringReading[str]()

    assert read(left, reading, lexicon) == (tuple(forms),)
    assert read(right, reading, lexicon) == (tuple(forms),)


@settings(max_examples=100, deadline=None)
@given(st.text(min_size=1), st.text(min_size=1))
def test_category_reading_preserves_every_selected_conclusion(
    lexical: str,
    conclusion: str,
) -> None:
    entry = Entry("x", lexical)
    derivation = applied(UNARY, (axiom(entry),), conclusion, 0.0)
    lexicon = FormLexicon((UNARY,), (entry,))

    assert read(derivation, CategoryReading[str](), lexicon) == (conclusion,)


@settings(max_examples=100, deadline=None)
@given(
    st.floats(allow_nan=False, allow_infinity=False, width=32),
    st.lists(
        st.floats(allow_nan=False, allow_infinity=False, width=32),
        min_size=3,
        max_size=3,
    ),
)
def test_weight_reading_preserves_local_then_physical_child_order(
    local: float,
    lexical: list[float],
) -> None:
    reading = WeightReading[str, Trace](_TraceSemiring(), _trace)
    entries = tuple(
        Entry(str(at), "X", weight=weight) for at, weight in enumerate(lexical)
    )
    children = tuple(
        axiom(entry, at, weight=frozenset[tuple[float, ...]]())
        for at, entry in enumerate(entries)
    )
    derivation = applied(
        TERNARY,
        children,
        "X",
        frozenset(((local,),)),
    )

    assert read(derivation, reading, FormLexicon((TERNARY,), entries)) == (
        frozenset(((local, *lexical),)),
    )


def _category_fixture() -> InterpretationFixture[str, str]:
    """Return the category-reading conformance fixture."""
    return InterpretationFixture(
        CategoryReading(),
        LEXICON,
        st.sampled_from(DERIVATION_CASES),
        st.sampled_from(SAME_STRING_PAIRS),
    )


def _string_fixture() -> InterpretationFixture[str, tuple[str, ...]]:
    """Return the string-reading conformance fixture."""
    return InterpretationFixture(
        StringReading(),
        LEXICON,
        st.sampled_from(DERIVATION_CASES),
        st.sampled_from(SAME_STRING_PAIRS),
    )


def _weight_fixture() -> InterpretationFixture[str, float]:
    """Return the weight-reading conformance fixture."""
    return InterpretationFixture(
        WeightReading(Viterbi(), lambda weight: weight),
        LEXICON,
        st.sampled_from(DERIVATION_CASES),
        st.sampled_from(SAME_STRING_PAIRS),
    )


class TestCategoryReading(InterpretationSuite):
    """Interpretation-law coverage for the category reading."""

    @pytest.fixture
    def interpretation_conformance(self) -> InterpretationFixture[str, str]:
        """Return the category-reading fixture."""
        return _category_fixture()


class TestStringReading(InterpretationSuite):
    """Interpretation-law coverage for the string reading."""

    @pytest.fixture
    def interpretation_conformance(
        self,
    ) -> InterpretationFixture[str, tuple[str, ...]]:
        """Return the string-reading fixture."""
        return _string_fixture()


class TestWeightReading(InterpretationSuite):
    """Interpretation-law coverage for the weight reading."""

    @pytest.fixture
    def interpretation_conformance(self) -> InterpretationFixture[str, float]:
        """Return the weight-reading fixture."""
        return _weight_fixture()


def test_every_premise_a_step_does_not_head_becomes_one_link() -> None:
    reduction = HeadDependencies(HEADS)

    assert reduction.reduce(sentence()) == (
        HeadLink(head=3, dependent=0, label="ba"),
        HeadLink(head=0, dependent=1, label="fa"),
        HeadLink(head=3, dependent=2, label="ba"),
    )


def test_a_derivation_of_one_token_licenses_no_link() -> None:
    assert HeadDependencies(HEADS).reduce(leaf((1, 2), "N")) == ()


def test_the_head_table_decides_which_side_of_a_step_heads_it() -> None:
    tree = step((0, 2), "S", "fa", (leaf((0, 1), "X"), leaf((1, 2), "Y")))

    forward = HeadDependencies({"fa": 0}).reduce(tree)
    backward = HeadDependencies({"fa": 1}).reduce(tree)

    assert forward == (HeadLink(head=0, dependent=1, label="fa"),)
    assert backward == (HeadLink(head=1, dependent=0, label="fa"),)


def test_a_chain_of_unary_steps_adds_no_link_and_changes_none() -> None:
    reduction = HeadDependencies(HEADS)
    plain = step((0, 2), "S", "fa", (leaf((0, 1), "X"), leaf((1, 2), "Y")))
    raised = step(
        (0, 2),
        "S[dcl]",
        "tr",
        (step((0, 2), "S", "tr", (plain,)),),
    )

    assert reduction.reduce(raised) == reduction.reduce(plain)
    assert reduction.reduce(raised) == (HeadLink(head=0, dependent=1, label="fa"),)


def test_a_link_runs_between_head_tokens_and_not_between_span_starts() -> None:
    # the right premise is headed by token 2, not its first token.
    inner = step((1, 3), "S\\NP", "ba", (leaf((1, 2), "A"), leaf((2, 3), "B")))
    outer = step((0, 3), "S", "ba", (leaf((0, 1), "NP"), inner))

    assert HeadDependencies(HEADS).reduce(outer) == (
        HeadLink(head=2, dependent=0, label="ba"),
        HeadLink(head=2, dependent=1, label="ba"),
    )


def test_a_binary_rule_the_head_table_does_not_name_is_a_reported_gap() -> None:
    reduction = HeadDependencies({"fa": 0})

    with pytest.raises(ValueError, match="names no premise of 'ba'"):
        reduction.reduce(sentence())


def test_a_head_the_rule_does_not_have_is_refused() -> None:
    reduction = HeadDependencies({"fa": 0, "ba": 4})

    with pytest.raises(ValueError, match="premise 4 of 'ba'"):
        reduction.reduce(sentence())


def test_a_negative_premise_position_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="heads\\['fa'\\] is negative"):
        HeadDependencies({"fa": -1})


def test_a_premise_position_that_is_not_an_integer_is_refused() -> None:
    with pytest.raises(TypeError, match="is not an integer"):
        HeadDependencies({"fa": True})


def test_the_table_is_copied_so_a_later_edit_changes_nothing() -> None:
    table = {"fa": 0, "ba": 1}
    reduction = HeadDependencies(table)

    table["ba"] = 0

    assert reduction.reduce(sentence())[0] == HeadLink(head=3, dependent=0, label="ba")


def test_the_reduction_names_itself_and_reports_the_rules_it_covers() -> None:
    reduction = HeadDependencies(HEADS)

    assert reduction.name == "head-dependencies"
    assert repr(reduction) == "HeadDependencies(heads=[ba, fa])"


def test_a_link_reads_as_a_named_triple() -> None:
    link = HeadLink(head=2, dependent=0, label="ba")

    assert (link.head, link.dependent, link.label) == (2, 0, "ba")
    assert tuple(link) == (2, 0, "ba")
