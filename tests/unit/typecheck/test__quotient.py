from importlib.metadata import EntryPoint
from math import fsum, isclose
from typing import TYPE_CHECKING, Final, cast

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from tests.language import language
from tests.support.derivations import FormLexicon, axiom, step
from tests.support.installed import install
from tests.support.rules import ccg_signature, declared_rules

from lofi._frozen import FrozenValue
from lofi.deduce import Cell, Derivation, SplitProbabilities
from lofi.interpret import (
    MeaningReading,
    MeaningSelection,
    compile_meaning_actions,
    read,
)
from lofi.judge import ProofSearch, Rule
from lofi.lexicon import Entry
from lofi.spec import (
    CompositionSpec,
    FixedOperatorSpec,
    NodePatternSpec,
    PatternRuleSpec,
    RuleSetSpec,
    TermPatternSpec,
    TypeRaisingSpec,
)
from lofi.spec.readouts import ProofTermMeaningSettings, TermsReadoutSpec
from lofi.terms import (
    Calculus,
    Normalization,
    Term,
    TermStore,
    normalize,
)
from lofi.typecheck import Erasure
from lofi.typecheck._quotient import (
    DISTINCT_MEANING_WEIGHTING,
    MULTIPLICITY_MEANING_WEIGHTING,
    LogicalFormChart,
    MeaningItem,
    MeaningSource,
    MeaningWeighting,
    MeaningWeightingPlugin,
    meaning_shares,
    meaning_weighting,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


VOCABULARY: Final = (
    "bill",
    "bo",
    "likes",
    "mary",
    "w0",
    "w1",
    "w2",
    "w3",
    "w4",
    "w5",
    "w6",
    "w7",
    "w8",
    "w9",
)

LANGUAGE: Final = language(VOCABULARY)
STORE: Final = LANGUAGE.store
CALCULUS: Final = LANGUAGE.calculus
NORMALIZATION: Final = Normalization(
    fuel=40,
    order="leftmost-outermost",
    eta="off",
    associative_commutative=(),
)
APPLICATION: Final = STORE.signature.by_name(CALCULUS.app)
TOLERANCE: Final = 1e-12


class _FinalMeaningWeighting(FrozenValue):
    """A third-party fixture that gives all mass to the final meaning."""

    __slots__ = ()

    _noun = "test meaning weighting"

    @property
    def name(self) -> str:
        return "test-final"

    def shares(self, reached: Sequence[Term], /) -> Mapping[Term, float]:
        distinct = tuple(dict.fromkeys(reached))
        return {meaning: float(meaning is distinct[-1]) for meaning in distinct}


TEST_WEIGHTING_PLUGIN: Final = MeaningWeightingPlugin(_FinalMeaningWeighting)


class _BrokenMeaningWeighting(FrozenValue):
    """A test policy returning a caller-supplied invalid share mapping."""

    __slots__ = ("_result",)

    _noun = "broken test meaning weighting"

    def __init__(self, result: Mapping[Term, float], /) -> None:
        object.__setattr__(self, "_result", tuple(result.items()))

    @property
    def name(self) -> str:
        return "broken"

    def shares(self, reached: Sequence[Term], /) -> Mapping[Term, float]:
        del reached
        return dict(self._result)


X_CATEGORY: Final = TermPatternSpec(name="X", sort="Cat")
Y_CATEGORY: Final = TermPatternSpec(name="Y", sort="Cat")
CATEGORY_RULES: Final = declared_rules(
    ccg_signature("NP", "S", name="quotient-erasure-language"),
    RuleSetSpec(
        name="quotient",
        rules=(
            PatternRuleSpec(
                name="forward-application",
                premises=(
                    NodePatternSpec(
                        operator=FixedOperatorSpec(name="/"),
                        arguments=(X_CATEGORY, Y_CATEGORY),
                    ),
                    Y_CATEGORY,
                ),
                conclusion=X_CATEGORY,
            ),
            PatternRuleSpec(
                name="backward-application",
                premises=(
                    Y_CATEGORY,
                    NodePatternSpec(
                        operator=FixedOperatorSpec(name="\\"),
                        arguments=(X_CATEGORY, Y_CATEGORY),
                    ),
                ),
                conclusion=X_CATEGORY,
            ),
            CompositionSpec(view="/", order=1, direction="unrestricted"),
            TypeRaisingSpec(view="/", targets=("S",), arguments=None),
        ),
    ),
)
CATEGORIES: Final = CATEGORY_RULES.context
ERASURE_STORE: Final = CATEGORIES.store
TYPE_ARROW_IMAGE: Final = ERASURE_STORE.op(
    ERASURE_STORE.signature.by_name("->"),
    ERASURE_STORE.fvar("Ty", "a1"),
    ERASURE_STORE.fvar("Ty", "a0"),
)
ERASURE: Final = Erasure(
    ERASURE_STORE,
    {"NP": ERASURE_STORE.const("Ty", "e"), "S": ERASURE_STORE.const("Ty", "t")},
    {"/": TYPE_ARROW_IMAGE, "\\": TYPE_ARROW_IMAGE},
    category_sort="Cat",
    function_type="->",
)


class LambdaCalculus:
    """The proof calculus over the logical-form store."""

    __slots__ = ()

    @property
    def store(self) -> TermStore:
        return STORE

    @property
    def operators(self) -> Calculus:
        return CALCULUS

    def hypothesis(self, index: int, /) -> Term:
        return STORE.fvar(LANGUAGE.term_sort, f"p{index}")

    def free_variable(self, name: str, /) -> Term:
        return STORE.fvar(LANGUAGE.term_sort, name)

    def bound_variable(self, index: int, /) -> Term:
        return STORE.bvar(LANGUAGE.term_sort, index)

    def application(self, function: Term, argument: Term, /) -> Term:
        return STORE.op(APPLICATION, function, argument)

    def abstraction(self, variable: str, body: Term, /) -> Term:
        return STORE.bind(CALCULUS.lam, variable, body)


PROOF: Final = LambdaCalculus()
FUNCTORS: Final = CATEGORIES.functors["/"][0]
NP: Final = CATEGORIES.syntax.parse(ERASURE_STORE, "NP")
S: Final = CATEGORIES.syntax.parse(ERASURE_STORE, "S")
VERB_PHRASE: Final = FUNCTORS.make(FUNCTORS.backward_slash, S, NP)
TRANSITIVE_VERB: Final = FUNCTORS.make(FUNCTORS.forward_slash, VERB_PHRASE, NP)
FORWARD_APPLICATION, BACKWARD_APPLICATION, COMPOSITION, RAISING = (
    CATEGORY_RULES.rule_set.rules
)
RULES: Final[tuple[Rule[Term], ...]] = (
    FORWARD_APPLICATION,
    BACKWARD_APPLICATION,
    COMPOSITION,
    RAISING,
)
READING: Final = MeaningReading[Term](
    compile_meaning_actions(
        RULES,
        erasure=ERASURE,
        calculus=PROOF,
        normalization=NORMALIZATION,
        search=ProofSearch(
            contraction=True, weakening=False, max_uses=1, max_inhabitants=64
        ),
        selection=MeaningSelection("unique"),
    ),
    normalization=NORMALIZATION,
    policy="eager",
)

LAZY_READING: Final = MeaningReading[Term](
    READING.actions,
    normalization=NORMALIZATION,
    policy="lazy",
)

LIKES: Final = STORE.const(LANGUAGE.term_sort, "likes")
BO: Final = STORE.const(LANGUAGE.term_sort, "bo")
MARY: Final = STORE.const(LANGUAGE.term_sort, "mary")
BILL: Final = STORE.const(LANGUAGE.term_sort, "bill")


def _apply(function: Term, argument: Term) -> Term:
    return STORE.op(APPLICATION, function, argument)


def _conjoin(left: Term, right: Term) -> Term:
    return LANGUAGE.apply(LANGUAGE.constant("and"), left, right)


def _identity(variable: str) -> Term:
    """Build the identity abstraction over a named variable."""
    return STORE.bind(
        CALCULUS.lam,
        variable,
        STORE.fvar(LANGUAGE.term_sort, variable),
    )


def _detour(value: Term, variable: str, depth: int) -> Term:
    """Wrap a value in a stack of identity redexes over a named variable."""
    wrapped = value

    for _ in range(depth):
        wrapped = _apply(_identity(variable), wrapped)

    return wrapped


def _uniform(spans: Sequence[Cell]) -> SplitProbabilities:
    """Declare a uniform split posterior over every split of every span."""
    rows: dict[Cell, dict[int, float]] = {}

    for start, end in spans:
        splits = range(start + 1, end)
        share = 1.0 / len(splits)
        rows[start, end] = dict.fromkeys(splits, share)

    return SplitProbabilities(rows)


def _binary_spans(n: int) -> tuple[Cell, ...]:
    """Return every span of at least two tokens, ascending."""
    return tuple((start, end) for start in range(n) for end in range(start + 2, n + 1))


def _conclude(
    rule: Rule[Term],
    children: tuple[Derivation[Term, float], ...],
) -> Derivation[Term, float]:
    """Apply a rule to child derivations, taking the one conclusion it licenses."""
    premises = tuple(child.edge.output for child in children)
    (conclusion,) = rule.apply(premises)

    return step(rule, children, conclusion, 0.0)


def _meaning(derivation: Derivation[Term, float]) -> Term:
    """Read a derivation's meaning off the derived actions."""
    (meaning,) = read(derivation, READING, LEXICON)

    assert meaning is not None

    return meaning


# the three derivations of "Bo likes Mary" as a sentence: subject application,
# type-raised subject applied to the verb phrase, and a composed subject-verb
# constituent applied to the object
ENTRIES: Final = (
    Entry("Bo", NP, BO),
    Entry("likes", TRANSITIVE_VERB, LIKES),
    Entry("Mary", NP, MARY),
)
LEXICON: Final = FormLexicon(
    (FORWARD_APPLICATION, BACKWARD_APPLICATION, RAISING, COMPOSITION), ENTRIES
)
BO_AXIOM: Final = axiom(ENTRIES[0])
LIKES_AXIOM: Final = axiom(ENTRIES[1], 1)
MARY_AXIOM: Final = axiom(ENTRIES[2], 2)
LIKES_MARY: Final = _conclude(FORWARD_APPLICATION, (LIKES_AXIOM, MARY_AXIOM))
RAISED_BO: Final = _conclude(RAISING, (BO_AXIOM,))
BO_LIKES: Final = _conclude(COMPOSITION, (RAISED_BO, LIKES_AXIOM))
BY_SUBJECT: Final = _conclude(BACKWARD_APPLICATION, (BO_AXIOM, LIKES_MARY))
BY_RAISED_SUBJECT: Final = _conclude(FORWARD_APPLICATION, (RAISED_BO, LIKES_MARY))
BY_COMPOSITION: Final = _conclude(FORWARD_APPLICATION, (BO_LIKES, MARY_AXIOM))
GOAL_DERIVATIONS: Final = (
    (1, BY_SUBJECT),
    (1, BY_RAISED_SUBJECT),
    (2, BY_COMPOSITION),
)

ROOT: Final[Cell] = (0, 3)
SUBJECT_SPAN: Final[Cell] = (0, 2)
OBJECT_SPAN: Final[Cell] = (1, 3)
THREE_WORD: Final = {
    ROOT: tuple(
        MeaningSource(split, _meaning(derivation))
        for split, derivation in GOAL_DERIVATIONS
    ),
    SUBJECT_SPAN: (MeaningSource(1, _meaning(BO_LIKES)),),
    OBJECT_SPAN: (MeaningSource(2, _meaning(LIKES_MARY)),),
}
THREE_WORD_ALPHA: Final = SplitProbabilities(
    {ROOT: {1: 0.75, 2: 0.25}, SUBJECT_SPAN: {1: 1.0}, OBJECT_SPAN: {2: 1.0}}
)

# likes(mary)(bo), the meaning every derivation of the three-word fixture reaches
TRANSITIVE: Final = _apply(_apply(LIKES, MARY), BO)
# the same meaning through a subject abstraction, under two different variable names
SUBJECT_X: Final = _apply(
    STORE.bind(
        CALCULUS.lam,
        "x",
        _apply(_apply(LIKES, MARY), STORE.fvar(LANGUAGE.term_sort, "x")),
    ),
    BO,
)
SUBJECT_Y: Final = _apply(
    STORE.bind(
        CALCULUS.lam,
        "y",
        _apply(_apply(LIKES, MARY), STORE.fvar(LANGUAGE.term_sort, "y")),
    ),
    BO,
)
OTHER: Final = _apply(_apply(LIKES, BILL), BO)
ROOT_ALPHA: Final = SplitProbabilities({ROOT: {1: 0.75, 2: 0.25}})


def _chart(
    contributions: Mapping[Cell, Sequence[MeaningSource]],
    alpha: SplitProbabilities,
    weighting: MeaningWeighting = DISTINCT_MEANING_WEIGHTING,
) -> LogicalFormChart:
    return LogicalFormChart(
        STORE,
        contributions,
        alpha,
        calculus=CALCULUS,
        normalization=NORMALIZATION,
        weighting=weighting,
    )


def test_the_three_derivations_of_the_goal_are_read_to_one_interned_meaning() -> None:
    meanings = tuple(_meaning(derivation) for _, derivation in GOAL_DERIVATIONS)

    assert len({derivation for _, derivation in GOAL_DERIVATIONS}) == 3
    assert {derivation.edge.rule_name for _, derivation in GOAL_DERIVATIONS} == {
        FORWARD_APPLICATION.name,
        BACKWARD_APPLICATION.name,
    }
    assert BY_COMPOSITION.children[0] is BO_LIKES
    assert BO_LIKES.edge.rule_name == COMPOSITION.name
    assert all(meaning is TRANSITIVE for meaning in meanings)
    assert _meaning(LIKES_MARY) is _apply(LIKES, MARY)
    assert _meaning(BO_LIKES) is not TRANSITIVE


def test_two_alpha_variant_derivations_are_one_meaning() -> None:
    assert SUBJECT_X is SUBJECT_Y

    chart = _chart(
        {ROOT: (MeaningSource(1, SUBJECT_X), MeaningSource(2, SUBJECT_Y))},
        ROOT_ALPHA,
    )

    assert len(chart.meanings(ROOT)) == 1
    assert chart.contribution_count(ROOT) == 2


def test_two_beta_convertible_derivations_are_one_meaning() -> None:
    assert SUBJECT_X is not TRANSITIVE

    chart = _chart(
        {ROOT: (MeaningSource(1, TRANSITIVE), MeaningSource(2, SUBJECT_X))},
        ROOT_ALPHA,
    )
    (item,) = chart.meanings(ROOT)

    assert item.meaning is TRANSITIVE
    assert item.splits == (1, 2)


def test_the_declared_eta_policy_changes_the_meaning_quotient() -> None:
    function = STORE.const(LANGUAGE.term_sort, "likes")
    expanded = STORE.bind(
        CALCULUS.lam,
        "x",
        _apply(function, STORE.fvar(LANGUAGE.term_sort, "x")),
    )
    contributions = {ROOT: (MeaningSource(1, function), MeaningSource(2, expanded))}
    without_eta = LogicalFormChart(
        STORE,
        contributions,
        ROOT_ALPHA,
        calculus=CALCULUS,
        normalization=NORMALIZATION,
        weighting=DISTINCT_MEANING_WEIGHTING,
    )
    with_eta = LogicalFormChart(
        STORE,
        contributions,
        ROOT_ALPHA,
        calculus=CALCULUS,
        normalization=Normalization(
            fuel=40,
            order="leftmost-outermost",
            eta="contract",
            associative_commutative=(),
        ),
        weighting=DISTINCT_MEANING_WEIGHTING,
    )

    assert len(without_eta.meanings(ROOT)) == 2
    assert len(with_eta.meanings(ROOT)) == 1


def test_declared_ac_symbols_change_the_meaning_quotient() -> None:
    left = _conjoin(BO, _conjoin(MARY, BILL))
    right = _conjoin(_conjoin(BILL, BO), MARY)
    contributions = {ROOT: (MeaningSource(1, left), MeaningSource(2, right))}
    without_ac = _chart(contributions, ROOT_ALPHA)
    with_ac = LogicalFormChart(
        STORE,
        contributions,
        ROOT_ALPHA,
        calculus=CALCULUS,
        normalization=Normalization(
            fuel=40,
            order="leftmost-outermost",
            eta="off",
            associative_commutative=("and",),
        ),
        weighting=DISTINCT_MEANING_WEIGHTING,
    )

    assert len(without_ac.meanings(ROOT)) == 2
    assert len(with_ac.meanings(ROOT)) == 1


def test_two_unrelated_derivations_stay_two_meanings() -> None:
    chart = _chart(
        {ROOT: (MeaningSource(1, TRANSITIVE), MeaningSource(2, OTHER))},
        ROOT_ALPHA,
    )
    meanings = chart.meanings(ROOT)

    assert [item.meaning for item in meanings] == [TRANSITIVE, OTHER]
    assert [item.weight for item in meanings] == [0.75, 0.25]


def _lazy_meaning(derivation: Derivation[Term, float]) -> Term:
    """Read a derivation's meaning under the lazy policy, redexes kept."""
    (meaning,) = read(derivation, LAZY_READING, LEXICON)

    assert meaning is not None

    return meaning


def test_lazily_read_derivations_quotient_to_the_eagerly_read_meanings() -> None:
    lazy = {
        ROOT: tuple(
            MeaningSource(split, _lazy_meaning(derivation))
            for split, derivation in GOAL_DERIVATIONS
        ),
        SUBJECT_SPAN: (MeaningSource(1, _lazy_meaning(BO_LIKES)),),
        OBJECT_SPAN: (MeaningSource(2, _lazy_meaning(LIKES_MARY)),),
    }
    lazily_read = {source.form for sources in lazy.values() for source in sources}
    eagerly_read = {
        source.form for sources in THREE_WORD.values() for source in sources
    }

    assert _lazy_meaning(BY_RAISED_SUBJECT) is not TRANSITIVE
    assert len(lazily_read) > len(eagerly_read)

    chart = _chart(lazy, THREE_WORD_ALPHA)
    eager = _chart(THREE_WORD, THREE_WORD_ALPHA)

    for span in (ROOT, SUBJECT_SPAN, OBJECT_SPAN):
        pairs = zip(chart.meanings(span), eager.meanings(span), strict=True)

        assert all(held.meaning is other.meaning for held, other in pairs)
        assert chart.weight_mass(span) == pytest.approx(eager.weight_mass(span))

    assert chart.contribution_total == eager.contribution_total
    assert chart.distinct_total == eager.distinct_total


def test_the_three_word_fixture_quotients_three_derivations_to_one_meaning() -> None:
    chart = _chart(THREE_WORD, THREE_WORD_ALPHA)
    (item,) = chart.meanings(ROOT)

    assert chart.contribution_count(ROOT) == 3
    assert item.meaning is TRANSITIVE
    assert item.multiplicity == 3
    assert item.splits == (1, 2)
    assert chart.distinct_total == 3
    assert chart.contribution_total == 5
    assert chart.multiplicity_ratio() == pytest.approx(5 / 3)


def test_the_three_word_quotient_keeps_the_alpha_mass_of_every_span() -> None:
    for weighting in (DISTINCT_MEANING_WEIGHTING, MULTIPLICITY_MEANING_WEIGHTING):
        chart = _chart(THREE_WORD, THREE_WORD_ALPHA, weighting)

        for span in chart.spans:
            assert chart.weight_mass(span) == pytest.approx(chart.alpha_mass(span))
            assert chart.weight_mass(span) == pytest.approx(1.0)

        assert chart.weight(ROOT, TRANSITIVE) == pytest.approx(0.75 + 0.25)


def test_the_multiplicity_weighting_gives_a_meaning_reached_twice_a_larger_share() -> (
    None
):
    contributions = {
        ROOT: (
            MeaningSource(1, TRANSITIVE),
            MeaningSource(1, OTHER),
            MeaningSource(1, SUBJECT_X),
            MeaningSource(2, TRANSITIVE),
        )
    }
    quotiented = _chart(contributions, ROOT_ALPHA)
    weighted = _chart(contributions, ROOT_ALPHA, MULTIPLICITY_MEANING_WEIGHTING)

    assert weighted.weighting is MULTIPLICITY_MEANING_WEIGHTING
    assert quotiented.weight(ROOT, TRANSITIVE) == pytest.approx(0.75 / 2 + 0.25)
    assert quotiented.weight(ROOT, OTHER) == pytest.approx(0.75 / 2)
    assert weighted.weight(ROOT, TRANSITIVE) == pytest.approx(0.75 * 2 / 3 + 0.25)
    assert weighted.weight(ROOT, OTHER) == pytest.approx(0.75 / 3)
    assert weighted.weight_mass(ROOT) == pytest.approx(quotiented.weight_mass(ROOT))
    assert weighted.weight_mass(ROOT) == pytest.approx(weighted.alpha_mass(ROOT))


def test_the_two_weightings_agree_when_no_split_reaches_two_meanings() -> None:
    quotiented = _chart(THREE_WORD, THREE_WORD_ALPHA)
    weighted = _chart(
        THREE_WORD,
        THREE_WORD_ALPHA,
        MULTIPLICITY_MEANING_WEIGHTING,
    )

    for span in quotiented.spans:
        for item in quotiented.meanings(span):
            assert weighted.weight(span, item.meaning) == item.weight


def test_a_document_selects_a_third_party_meaning_weighting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    point = EntryPoint(
        name="test-final",
        value=f"{__name__}:TEST_WEIGHTING_PLUGIN",
        group="lofi.meaning_weighting",
    )

    install(monkeypatch, "meaning_weighting", (point,))
    declared = TermsReadoutSpec.model_validate(
        {
            "kind": "terms",
            "space": "forms",
            "reads": "inside",
            "decoding": {"strategy": "greedy"},
            "meanings": {
                "kind": "proof-term",
                "weighting": "test-final",
                "residual_mass": 0.999,
            },
        }
    )
    settings = declared.settings().meanings
    assert isinstance(settings, ProofTermMeaningSettings)
    selected = meaning_weighting(settings.weighting)
    chart = _chart(
        {
            ROOT: (
                MeaningSource(1, TRANSITIVE),
                MeaningSource(1, OTHER),
            )
        },
        ROOT_ALPHA,
        selected,
    )

    assert selected.name == "test-final"
    assert chart.weight(ROOT, TRANSITIVE) == 0.0
    assert chart.weight(ROOT, OTHER) == pytest.approx(0.75)


def test_zero_meaning_shares_are_omitted_from_logarithmic_consumers() -> None:
    shares = meaning_shares(
        _FinalMeaningWeighting(),
        (TRANSITIVE, OTHER),
    )

    assert shares == {OTHER: 1.0}


@pytest.mark.parametrize(
    ("result", "message"),
    [
        ({TRANSITIVE: 1.0}, "exactly the reached"),
        (
            {TRANSITIVE: 0.5, OTHER: 0.5, MARY: 0.0},
            "exactly the reached",
        ),
        ({TRANSITIVE: -0.5, OTHER: 1.5}, "finite nonnegative"),
        ({TRANSITIVE: float("nan"), OTHER: 1.0}, "finite nonnegative"),
        ({TRANSITIVE: 0.25, OTHER: 0.25}, "sum to one"),
    ],
)
def test_invalid_meaning_weighting_results_are_refused(
    result: dict[Term, float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        meaning_shares(
            _BrokenMeaningWeighting(result),
            (TRANSITIVE, OTHER),
        )


def test_a_meaning_is_looked_up_by_span_and_normal_form() -> None:
    chart = _chart(THREE_WORD, THREE_WORD_ALPHA)

    assert chart.weight(ROOT, TRANSITIVE) == pytest.approx(1.0)

    with pytest.raises(KeyError):
        chart.weight(ROOT, MARY)

    with pytest.raises(KeyError):
        chart.weight(ROOT, SUBJECT_X)


def test_a_split_reaching_two_meanings_divides_its_mass_between_them() -> None:
    alpha = SplitProbabilities({SUBJECT_SPAN: {1: 1.0}})
    chart = _chart(
        {SUBJECT_SPAN: (MeaningSource(1, TRANSITIVE), MeaningSource(1, MARY))},
        alpha,
    )

    assert [item.weight for item in chart.meanings(SUBJECT_SPAN)] == [0.5, 0.5]
    assert chart.weight_mass(SUBJECT_SPAN) == pytest.approx(1.0)
    assert chart.alpha_mass(SUBJECT_SPAN) == pytest.approx(1.0)


def test_spans_come_back_ascending() -> None:
    chart = _chart(THREE_WORD, THREE_WORD_ALPHA)

    assert chart.spans == ((0, 2), (0, 3), (1, 3))


def test_a_nonpolicy_weighting_is_refused() -> None:
    with pytest.raises(TypeError, match="weighting policy"):
        LogicalFormChart(
            STORE,
            THREE_WORD,
            THREE_WORD_ALPHA,
            calculus=CALCULUS,
            normalization=NORMALIZATION,
            weighting=cast("MeaningWeighting", object()),
        )


def test_a_span_with_no_contribution_is_refused() -> None:
    with pytest.raises(ValueError, match="received no logical form"):
        _chart({ROOT: ()}, THREE_WORD_ALPHA)


def test_a_span_the_posterior_does_not_declare_is_refused() -> None:
    with pytest.raises(ValueError, match="no row for span"):
        _chart({(0, 2): (MeaningSource(1, MARY),)}, ROOT_ALPHA)


def test_a_split_the_posterior_does_not_score_is_refused() -> None:
    alpha = SplitProbabilities({ROOT: {1: 1.0}})

    with pytest.raises(ValueError, match="does not score split 2"):
        _chart({ROOT: (MeaningSource(2, TRANSITIVE),)}, alpha)


def test_a_chart_refuses_reassignment_and_deletion() -> None:
    chart = _chart(THREE_WORD, THREE_WORD_ALPHA)

    with pytest.raises(AttributeError, match="immutable"):
        chart.__setattr__("_weighting", MULTIPLICITY_MEANING_WEIGHTING)

    with pytest.raises(AttributeError, match="immutable"):
        chart.__delattr__("_items")


def test_the_rendering_names_the_weighting_and_the_two_totals() -> None:
    assert repr(_chart(THREE_WORD, THREE_WORD_ALPHA)) == (
        "LogicalFormChart(weighting='distinct', spans=3, distinct=3, contributions=5)"
    )


def test_an_empty_chart_has_no_multiplicity_ratio() -> None:
    chart = _chart({}, THREE_WORD_ALPHA)

    assert chart.spans == ()
    assert chart.distinct_total == 0

    with pytest.raises(ValueError, match="no multiplicity ratio"):
        chart.multiplicity_ratio()


def test_the_goal_span_alone_has_three_derivations_per_meaning() -> None:
    goal_only = _chart({ROOT: THREE_WORD[ROOT]}, THREE_WORD_ALPHA)

    assert goal_only.multiplicity_ratio() == 3.0


def _spurious_chart(n: int) -> LogicalFormChart:
    """Build the chart in which every split of a span reaches one meaning."""
    spans = _binary_spans(n)
    tokens = tuple(STORE.const(LANGUAGE.term_sort, f"w{at}") for at in range(n))
    contributions: dict[Cell, tuple[MeaningSource, ...]] = {}

    for start, end in spans:
        meaning = tokens[end - 1]

        for at in reversed(range(start, end - 1)):
            meaning = _conjoin(tokens[at], meaning)

        contributions[start, end] = tuple(
            MeaningSource(split, _detour(meaning, "u", split - start))
            for split in range(start + 1, end)
        )

    return LogicalFormChart(
        STORE,
        contributions,
        _uniform(spans),
        calculus=CALCULUS,
        normalization=NORMALIZATION,
        weighting=DISTINCT_MEANING_WEIGHTING,
    )


@pytest.mark.parametrize("n", [3, 5, 8])
def test_a_chart_whose_every_split_reaches_one_meaning_has_ratio_n_plus_one_over_three(
    n: int,
) -> None:
    chart = _spurious_chart(n)
    spans = _binary_spans(n)

    assert chart.distinct_total == len(spans)
    assert chart.contribution_total == sum(end - start - 1 for start, end in spans)
    assert chart.multiplicity_ratio() == pytest.approx((n + 1) / 3)

    for span in spans:
        assert chart.weight_mass(span) == pytest.approx(1.0)


def _pool() -> tuple[Term, ...]:
    """Return raw forms whose normal forms are three meanings, not six."""
    return (
        TRANSITIVE,
        SUBJECT_X,
        _detour(TRANSITIVE, "v", 2),
        MARY,
        _detour(MARY, "w", 1),
        OTHER,
    )


POOL: Final = _pool()


@st.composite
def _charts(
    draw: st.DrawFn,
) -> tuple[dict[Cell, tuple[MeaningSource, ...]], SplitProbabilities]:
    """Generate a chart over a short sentence with a normalized split posterior."""
    n = draw(st.integers(min_value=2, max_value=6))
    spans = _binary_spans(n)
    rows: dict[Cell, dict[int, float]] = {}
    contributions: dict[Cell, tuple[MeaningSource, ...]] = {}

    for start, end in spans:
        splits = tuple(range(start + 1, end))
        shares = draw(
            st.lists(
                st.integers(min_value=1, max_value=9),
                min_size=len(splits),
                max_size=len(splits),
            )
        )
        total = sum(shares)
        rows[start, end] = {
            split: share / total for split, share in zip(splits, shares, strict=True)
        }
        drawn = draw(
            st.lists(
                st.tuples(st.sampled_from(splits), st.sampled_from(POOL)),
                min_size=1,
                max_size=6,
            )
        )
        contributions[start, end] = tuple(
            MeaningSource(split, form) for split, form in drawn
        )

    return contributions, SplitProbabilities(rows)


def _by_split(
    sources: Sequence[MeaningSource],
) -> dict[int, list[Term]]:
    """Return the normal forms each split of a span produced, in order."""
    held: dict[int, list[Term]] = {}

    for source in sources:
        normal = normalize(
            STORE,
            source.form,
            CALCULUS,
            normalization=NORMALIZATION,
        )
        held.setdefault(source.split, []).append(normal)

    return held


def _expected_weights(
    sources: Sequence[MeaningSource],
    row: Mapping[int, float],
    weighting: MeaningWeighting,
) -> dict[Term, float]:
    """Compute each meaning's weight as its per-split shares, summed."""
    expected: dict[Term, float] = {}

    for split, forms in _by_split(sources).items():
        for form, proportion in weighting.shares(forms).items():
            share = row[split] * proportion
            expected[form] = expected.get(form, 0.0) + share

    return expected


@settings(max_examples=200, deadline=None)
@given(
    _charts(),
    st.sampled_from((DISTINCT_MEANING_WEIGHTING, MULTIPLICITY_MEANING_WEIGHTING)),
)
def test_the_quotient_neither_loses_nor_gains_alpha_mass(
    generated: tuple[dict[Cell, tuple[MeaningSource, ...]], SplitProbabilities],
    weighting: MeaningWeighting,
) -> None:
    contributions, alpha = generated
    chart = _chart(contributions, alpha, weighting)

    for span, sources in contributions.items():
        row = alpha.at(span)
        present = fsum(row[split] for split in sorted({s.split for s in sources}))
        expected = _expected_weights(sources, row, weighting)

        assert isclose(chart.weight_mass(span), present, abs_tol=TOLERANCE)
        assert isclose(chart.alpha_mass(span), present, abs_tol=TOLERANCE)
        assert isclose(fsum(expected.values()), present, abs_tol=TOLERANCE)
        assert {item.meaning for item in chart.meanings(span)} == set(expected)

        for meaning, weight in expected.items():
            assert isclose(chart.weight(span, meaning), weight, abs_tol=TOLERANCE)


@settings(max_examples=200, deadline=None)
@given(_charts())
def test_the_items_account_for_every_contribution(
    generated: tuple[dict[Cell, tuple[MeaningSource, ...]], SplitProbabilities],
) -> None:
    contributions, alpha = generated
    chart = _chart(contributions, alpha)

    for span, sources in contributions.items():
        items = chart.meanings(span)
        reached = _by_split(sources)

        assert sum(item.multiplicity for item in items) == len(sources)
        assert chart.contribution_count(span) == len(sources)
        assert [item.meaning for item in items] == list(
            dict.fromkeys(
                normalize(STORE, s.form, CALCULUS, normalization=NORMALIZATION)
                for s in sources
            )
        )

        for item in items:
            assert item.span == span
            assert item.splits == tuple(
                sorted(
                    split for split, forms in reached.items() if item.meaning in forms
                )
            )
            assert item.multiplicity == sum(
                forms.count(item.meaning) for forms in reached.values()
            )


def test_an_item_is_identified_by_its_span_and_normal_form_alone() -> None:
    uneven = {
        ROOT: (
            MeaningSource(1, TRANSITIVE),
            MeaningSource(1, OTHER),
            MeaningSource(1, SUBJECT_X),
        )
    }
    item, _ = _chart(uneven, ROOT_ALPHA).meanings(ROOT)
    weighted, _ = _chart(uneven, ROOT_ALPHA, MULTIPLICITY_MEANING_WEIGHTING).meanings(
        ROOT
    )
    (other,) = _chart({ROOT: (MeaningSource(1, TRANSITIVE),)}, ROOT_ALPHA).meanings(
        ROOT
    )

    assert isinstance(item, MeaningItem)
    assert item.weight != weighted.weight
    assert item == weighted
    assert hash(item) == hash(weighted)
    assert item == other
    assert len({item, weighted, other}) == 1
    assert item != MeaningItem(SUBJECT_SPAN, TRANSITIVE, 1.0, (1,), 1)
    assert item != MeaningItem(ROOT, MARY, 1.0, (1,), 1)
    assert item != (ROOT, TRANSITIVE)


def test_an_item_reports_its_fields_and_refuses_change() -> None:
    (item,) = _chart(THREE_WORD, THREE_WORD_ALPHA).meanings(ROOT)

    assert item.span == ROOT
    assert item.meaning is TRANSITIVE
    assert item.weight == pytest.approx(1.0)
    assert item.splits == (1, 2)
    assert item.multiplicity == 3
    assert repr(item) == f"MeaningItem((0, 3), {TRANSITIVE!r}, 1.0, (1, 2), 3)"
    assert MeaningSource(1, MARY).form is MARY

    with pytest.raises(AttributeError, match="immutable"):
        item.__setattr__("_weight", 0.0)

    with pytest.raises(AttributeError, match="immutable"):
        item.__delattr__("_weight")
