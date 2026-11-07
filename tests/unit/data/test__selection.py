import pickle
from collections import Counter
from random import Random
from typing import TYPE_CHECKING, Final, cast

import pytest
from tests.conformance.selection_policy import (
    SelectionPolicyConformance,
    SelectionPolicyFixture,
    the_same_mass_share_is_covered_at_two_sizes,
    the_selector_declares_no_absolute_count,
)

from lofi._checks import LawViolationError
from lofi.data import ClassRole, QuantilePopulation
from lofi.data._ports import AmbiguityRule
from lofi.data._selection import (
    CategoryBudget,
    CategoryClass,
    CategoryRole,
    CorpusProfile,
    FrequencyBand,
    FrequencyCount,
    FrequencyQuantile,
    MajorityRule,
    PerOccurrenceRule,
    RankBand,
    ScopeKind,
    SelectionPolicy,
    SelectionReport,
    UnionRule,
    WeightedRule,
    agreement,
)
from lofi.data._supervision import Weighted

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi.data._selection import Form, FormSelector

MASSES = {"S\\NP": 2.0, "N": 1.0}

CLOSED: Final = "D"
"""The closed class of the declared lexicon: the determiners."""

CLASS_SIZES: Final = {CLOSED: 5, "P": 15, "V": 50, "N": 200}
"""How many forms each declared category has.

The classes differ by an order of magnitude, so a positive coupling makes the
determiners the most frequent forms and a coupling of zero makes every form
equally frequent whatever its class.
"""

CLOSED_SHARE: Final = CLASS_SIZES[CLOSED] / sum(CLASS_SIZES.values())
"""The determiners' share of the declared forms."""

SMALL: Final = 2_000
LARGE: Final = 16_000
SEED: Final = 11
COUPLINGS: Final = (0.0, 0.1, 0.2, 0.3, 0.5, 1.0)
SHARE_TOLERANCE: Final = 0.06
CHANCE_TOLERANCE: Final = 0.1


def lexicon() -> dict[Form, str]:
    return {
        (f"{category.lower()}{index}",): category
        for category, size in CLASS_SIZES.items()
        for index in range(size)
    }


def drawn(tokens: int, *, coupling: float, seed: int = SEED) -> CorpusProfile[str]:
    declared = lexicon()
    weights = [
        float(CLASS_SIZES[category]) ** -coupling for category in declared.values()
    ]
    counts = Counter(Random(seed).choices(list(declared), weights=weights, k=tokens))

    return CorpusProfile(
        {form: counts[form] for form in declared},
        {form: frozenset({category}) for form, category in declared.items()},
    )


def test_a_frequency_band_admits_relative_frequencies_between_its_bounds() -> None:
    profile = CorpusProfile(
        {("a",): 1, ("b",): 3, ("c",): 6},
        {form: frozenset({"X"}) for form in (("a",), ("b",), ("c",))},
    )

    assert FrequencyBand(0.2, 0.3).admitted(profile) == {("b",)}
    assert FrequencyBand(0.3, 1.0).admitted(profile) == {("b",), ("c",)}
    assert FrequencyBand(0.0, 0.1).admitted(profile) == {("a",)}
    assert FrequencyBand(0.0, 1.0).admitted(profile) == frozenset(profile.counts)
    assert FrequencyBand(0.0, 1.0).size_invariant


def test_a_frequency_band_names_the_same_forms_in_a_corpus_twice_as_long() -> None:
    once = CorpusProfile(
        {("a",): 1, ("b",): 3}, {("a",): frozenset(), ("b",): frozenset()}
    )
    twice = CorpusProfile(
        {("a",): 2, ("b",): 6}, {("a",): frozenset(), ("b",): frozenset()}
    )

    assert FrequencyBand(0.5, 1.0).admitted(once) == FrequencyBand(0.5, 1.0).admitted(
        twice
    )
    assert FrequencyCount(2, None).admitted(once) != FrequencyCount(2, None).admitted(
        twice
    )


def test_a_frequency_band_refuses_bounds_outside_the_unit_interval() -> None:
    with pytest.raises(ValueError, match="admits none"):
        FrequencyBand(0.4, 0.2)

    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        FrequencyBand(1.5, 1.0)

    with pytest.raises(TypeError, match="not a number"):
        FrequencyBand(cast("float", "half"), 1.0)


def test_frequency_bands_are_equal_on_their_bounds() -> None:
    assert FrequencyBand(0.1, 0.3) == FrequencyBand(0.1, 0.3)
    assert hash(FrequencyBand(0.1, 0.3)) == hash(FrequencyBand(0.1, 0.3))
    assert FrequencyBand(0.1, 0.3) != FrequencyBand(0.1, 0.4)
    assert FrequencyBand(0.1, 1.0) != "band"
    assert repr(FrequencyBand(0.1, 0.3)) == "FrequencyBand(0.1, 0.3)"


def test_a_count_band_admits_counts_between_its_bounds_inclusive() -> None:
    profile = CorpusProfile(
        {(str(count),): count for count in range(1, 6)},
        {(str(count),): frozenset() for count in range(1, 6)},
    )

    assert FrequencyCount(2, 4).admitted(profile) == {("2",), ("3",), ("4",)}
    assert FrequencyCount(3, None).admitted(profile) == {("3",), ("4",), ("5",)}
    assert FrequencyCount(1, None).admitted(profile) == frozenset(profile.counts)
    assert FrequencyCount(3, None).at_most is None
    assert not FrequencyCount(1, None).size_invariant


def test_a_count_band_refuses_a_ceiling_below_its_floor_or_a_bad_bound() -> None:
    with pytest.raises(ValueError, match="admits none"):
        FrequencyCount(3, 2)

    with pytest.raises(ValueError, match="negative"):
        FrequencyCount(-1, None)

    with pytest.raises(TypeError, match="not an integer"):
        FrequencyCount(1, cast("int", 2.5))


def test_count_bands_are_equal_on_their_bounds() -> None:
    assert FrequencyCount(1, 3) == FrequencyCount(1, 3)
    assert hash(FrequencyCount(1, 3)) == hash(FrequencyCount(1, 3))
    assert FrequencyCount(1, 3) != FrequencyCount(1, 4)
    assert FrequencyCount(1, None) != FrequencyCount(1, 4)
    assert FrequencyCount(1, None) != "count"
    assert repr(FrequencyCount(1, 3)) == "FrequencyCount(1, 3)"


@pytest.mark.parametrize(
    ("rule", "scope"),
    [
        (UnionRule(), ScopeKind.FORM),
        (MajorityRule(), ScopeKind.FORM),
        (WeightedRule(), ScopeKind.FORM),
        (PerOccurrenceRule(), ScopeKind.OCCURRENCE),
    ],
)
def test_each_rule_declares_the_scope_it_binds_at(
    rule: AmbiguityRule[str, str], scope: ScopeKind
) -> None:
    assert rule.scope is scope
    assert isinstance(rule, AmbiguityRule)


def test_the_union_rule_permits_every_observed_type() -> None:
    assert UnionRule[str]().permitted(MASSES) == {"S\\NP", "N"}


def test_the_majority_rule_permits_the_heaviest_type_alone() -> None:
    assert MajorityRule[str]().permitted(MASSES) == {"S\\NP"}
    assert MajorityRule[str]().permitted({"N": 1.0, "S\\NP": 2.0}) == {"S\\NP"}


def test_a_majority_tie_goes_to_the_type_observed_first() -> None:
    assert MajorityRule[str]().permitted({"N": 1.0, "NP": 1.0}) == {"N"}
    assert MajorityRule[str]().permitted({"NP": 1.0, "N": 1.0}) == {"NP"}


def test_the_weighted_rule_shares_the_mass_out() -> None:
    permitted = WeightedRule[str]().permitted(MASSES)

    assert permitted == {Weighted("S\\NP", 2.0 / 3.0), Weighted("N", 1.0 / 3.0)}
    assert sum(label.weight for label in permitted) == pytest.approx(1.0)


def test_the_per_occurrence_rule_permits_what_one_occurrence_bore() -> None:
    assert PerOccurrenceRule[str]().permitted({"N": 1.0}) == {"N"}


@pytest.mark.parametrize(
    "rule", [UnionRule(), MajorityRule(), WeightedRule(), PerOccurrenceRule()]
)
def test_no_rule_permits_anything_for_a_form_observed_under_nothing(
    rule: AmbiguityRule[str, str],
) -> None:
    with pytest.raises(ValueError, match="observed under no type"):
        rule.permitted({})


@pytest.mark.parametrize(
    "rule", [UnionRule(), MajorityRule(), WeightedRule(), PerOccurrenceRule()]
)
def test_no_rule_counts_a_type_bearing_no_mass(
    rule: AmbiguityRule[str, str],
) -> None:
    with pytest.raises(ValueError, match="no observation"):
        rule.permitted({"N": 0.0})

    with pytest.raises(ValueError, match="no observation"):
        rule.permitted({"N": float("inf")})


def test_rules_are_equal_within_their_kind_and_hash_alike() -> None:
    assert UnionRule() == UnionRule()
    assert hash(UnionRule()) == hash(UnionRule())
    assert MajorityRule() == MajorityRule()
    assert WeightedRule() == WeightedRule()
    assert PerOccurrenceRule() == PerOccurrenceRule()
    assert UnionRule() != MajorityRule()
    assert WeightedRule() != PerOccurrenceRule()
    assert len({UnionRule(), MajorityRule(), WeightedRule(), PerOccurrenceRule()}) == 4


@pytest.mark.parametrize(
    "rule", [UnionRule(), MajorityRule(), WeightedRule(), PerOccurrenceRule()]
)
def test_a_rule_rebuilds_from_its_repr_and_pickles(
    rule: AmbiguityRule[str, str],
) -> None:
    assert repr(rule) == f"{type(rule).__name__}()"
    assert pickle.loads(pickle.dumps(rule)) == rule


def test_a_policy_defaults_to_full_coverage_of_every_form() -> None:
    policy = SelectionPolicy[str, str](UnionRule())

    assert policy.ambiguity == UnionRule()
    assert policy.scope is ScopeKind.FORM
    assert policy.selector == RankBand(0.0, 1.0)
    assert policy.coverage == 1.0
    assert policy.seed == 0


def test_a_policy_takes_the_scope_of_its_rule() -> None:
    assert SelectionPolicy[str, str](PerOccurrenceRule()).scope is ScopeKind.OCCURRENCE


def test_a_policy_holds_its_selector_coverage_and_seed() -> None:
    policy = SelectionPolicy[str, str](
        MajorityRule(), selector=FrequencyCount(2, 5), coverage=0.25, seed=7
    )

    assert policy.selector == FrequencyCount(2, 5)
    assert policy.coverage == 0.25
    assert policy.seed == 7


@pytest.mark.parametrize("coverage", [-0.1, 1.1, float("nan")])
def test_a_policy_refuses_a_coverage_outside_the_unit_interval(
    coverage: float,
) -> None:
    with pytest.raises(ValueError, match="fraction"):
        SelectionPolicy[str, str](UnionRule(), coverage=coverage)


def test_a_policy_refuses_a_coverage_that_is_not_a_number() -> None:
    with pytest.raises(TypeError, match="not a number"):
        SelectionPolicy[str, str](UnionRule(), coverage=cast("float", "half"))

    with pytest.raises(TypeError, match="not a number"):
        SelectionPolicy[str, str](UnionRule(), coverage=True)


def test_a_policy_admits_the_endpoints_of_coverage() -> None:
    assert SelectionPolicy[str, str](UnionRule(), coverage=0).coverage == 0.0
    assert SelectionPolicy[str, str](UnionRule(), coverage=1).coverage == 1.0


def test_a_policy_refuses_a_negative_seed() -> None:
    with pytest.raises(ValueError, match="negative"):
        SelectionPolicy[str, str](UnionRule(), seed=-1)


def test_reseeding_changes_the_seed_and_nothing_else() -> None:
    policy = SelectionPolicy[str, Weighted[str]](
        WeightedRule(),
        selector=FrequencyQuantile(0.2, over=QuantilePopulation.TYPES),
        coverage=0.5,
        seed=1,
    )
    reseeded = policy.with_seed(9)

    assert reseeded.seed == 9
    assert reseeded.ambiguity == policy.ambiguity
    assert reseeded.selector == policy.selector
    assert reseeded.coverage == policy.coverage
    assert reseeded != policy
    assert policy.with_seed(1) == policy


def test_policies_are_equal_on_all_their_fields() -> None:
    one = SelectionPolicy[str, str](UnionRule(), coverage=0.5, seed=2)

    assert one == SelectionPolicy[str, str](UnionRule(), coverage=0.5, seed=2)
    assert hash(one) == hash(
        SelectionPolicy[str, str](UnionRule(), coverage=0.5, seed=2)
    )
    assert one != SelectionPolicy[str, str](MajorityRule(), coverage=0.5, seed=2)
    assert one != SelectionPolicy[str, str](UnionRule(), coverage=0.6, seed=2)
    assert one != SelectionPolicy[str, str](UnionRule(), coverage=0.5, seed=3)
    assert one != SelectionPolicy[str, str](
        UnionRule(), selector=FrequencyCount(2, None), coverage=0.5, seed=2
    )
    assert one != "policy"


def test_a_policy_rebuilds_from_its_repr_and_pickles() -> None:
    policy = SelectionPolicy[str, str](
        MajorityRule(), selector=FrequencyCount(1, 3), coverage=0.5, seed=4
    )

    assert repr(policy) == (
        "SelectionPolicy(MajorityRule(), selector=FrequencyCount(1, 3), "
        "coverage=0.5, seed=4)"
    )
    assert pickle.loads(pickle.dumps(policy)) == policy


@pytest.mark.parametrize("field", ["coverage", "extra"])
def test_a_policy_is_frozen(field: str) -> None:
    policy = SelectionPolicy[str, str](UnionRule())

    with pytest.raises(AttributeError):
        setattr(policy, field, 0.5)


def test_a_profile_ranks_forms_by_count_with_ties_in_form_order() -> None:
    profile = CorpusProfile(
        {("a",): 3, ("b",): 7, ("c",): 3},
        {form: frozenset({"X"}) for form in (("a",), ("b",), ("c",))},
    )

    assert profile.ranked == (("b",), ("a",), ("c",))
    assert profile.types == 3
    assert profile.tokens == 13
    assert profile.share(profile.counts[("b",)]) == pytest.approx(7 / 13)
    assert profile.class_sizes == {"X": 3}


def test_a_profile_refuses_forms_it_counts_but_does_not_categorize() -> None:
    with pytest.raises(ValueError, match="one side only"):
        CorpusProfile({("a",): 1}, {})


def test_a_profile_refuses_a_negative_count() -> None:
    with pytest.raises(ValueError, match="negative"):
        CorpusProfile({("a",): -1}, {("a",): frozenset({"X"})})


def test_profiles_are_equal_on_their_counts_and_categories() -> None:
    one = CorpusProfile({("a",): 1}, {("a",): frozenset({"X"})})

    assert one == CorpusProfile({("a",): 1}, {("a",): frozenset({"X"})})
    assert hash(one) == hash(CorpusProfile({("a",): 1}, {("a",): frozenset({"X"})}))
    assert one != CorpusProfile({("a",): 2}, {("a",): frozenset({"X"})})
    assert one != CorpusProfile({("a",): 1}, {("a",): frozenset({"Y"})})
    assert one != "profile"
    assert repr(one) == "CorpusProfile({('a',): 1}, {('a',): frozenset({'X'})})"


def test_the_concordance_is_positive_when_the_small_class_is_the_frequent_one() -> None:
    coupled = drawn(SMALL, coupling=1.0).concordance
    flat = drawn(SMALL, coupling=0.0).concordance

    assert coupled is not None
    assert flat is not None
    assert coupled > flat
    assert coupled > 0.5
    assert abs(flat) < CHANCE_TOLERANCE


def test_a_profile_without_varying_counts_names_no_concordance() -> None:
    profile = CorpusProfile(
        {("a",): 2, ("b",): 2}, {("a",): frozenset({"X"}), ("b",): frozenset({"Y"})}
    )

    assert profile.concordance is None


def test_a_rank_band_takes_the_forms_at_those_shares_of_the_ranking() -> None:
    profile = drawn(SMALL, coupling=1.0)
    top = RankBand(0.0, CLOSED_SHARE).admitted(profile)

    assert top == frozenset(profile.ranked[: CLASS_SIZES[CLOSED]])
    assert RankBand(0.1, 0.2).admitted(profile) == frozenset(
        profile.ranked[round(0.1 * profile.types) : round(0.2 * profile.types)]
    )
    assert RankBand(0.0, 1.0).admitted(profile) == frozenset(profile.counts)
    assert RankBand(0.5, 0.5).admitted(profile) == frozenset()
    assert RankBand(0.0, 1.0).size_invariant


def test_a_rank_band_takes_the_same_share_of_a_larger_inventory() -> None:
    small = CorpusProfile(
        {(str(rank),): 100 - rank for rank in range(10)},
        {(str(rank),): frozenset() for rank in range(10)},
    )
    large = CorpusProfile(
        {(str(rank),): 100 - rank for rank in range(40)},
        {(str(rank),): frozenset() for rank in range(40)},
    )
    band = RankBand(0.0, 0.25)

    assert len(band.admitted(small)) / small.types == pytest.approx(0.2)
    assert len(band.admitted(large)) / large.types == pytest.approx(0.25)
    assert FrequencyCount(1, 1).admitted(small) == frozenset()


def test_a_rank_band_refuses_a_share_outside_the_unit_interval() -> None:
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        RankBand(-0.1, 1.0)

    with pytest.raises(ValueError, match="admits none"):
        RankBand(0.4, 0.2)

    with pytest.raises(TypeError, match="not a number"):
        RankBand(cast("float", "top"), 1.0)


def test_rank_bands_are_equal_on_their_shares() -> None:
    assert RankBand(0.0, 0.3) == RankBand(0.0, 0.3)
    assert hash(RankBand(0.0, 0.3)) == hash(RankBand(0.0, 0.3))
    assert RankBand(0.0, 0.3) != RankBand(0.0, 0.4)
    assert RankBand(0.0, 0.3) != "rank"
    assert repr(RankBand(0.0, 0.3)) == "RankBand(0.0, 0.3)"
    assert RankBand(0.0, 0.3).first == 0.0
    assert RankBand(0.0, 0.3).last == 0.3


def test_a_type_quantile_takes_that_share_of_the_forms() -> None:
    profile = drawn(SMALL, coupling=1.0)
    selected = FrequencyQuantile(0.1, over=QuantilePopulation.TYPES).admitted(profile)

    assert len(selected) == round(0.1 * profile.types)
    assert selected == frozenset(profile.ranked[: len(selected)])
    assert (
        FrequencyQuantile(0.0, over=QuantilePopulation.TYPES).admitted(profile)
        == frozenset()
    )
    assert FrequencyQuantile(1.0, over=QuantilePopulation.TYPES).admitted(
        profile
    ) == frozenset(profile.counts)


def test_a_token_quantile_takes_the_forms_holding_that_share_of_the_mass() -> None:
    profile = drawn(SMALL, coupling=1.0)
    wanted = 0.1
    selected = FrequencyQuantile(wanted, over=QuantilePopulation.TOKENS).admitted(
        profile
    )
    achieved = profile.share(sum(profile.counts[form] for form in selected))
    without = min(selected, key=lambda form: profile.counts[form])

    assert achieved >= wanted
    assert (
        profile.share(sum(profile.counts[form] for form in selected - {without}))
        < wanted
    )


def test_a_quantile_refuses_a_share_outside_the_unit_interval() -> None:
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        FrequencyQuantile(1.5, over=QuantilePopulation.TYPES)

    with pytest.raises(TypeError, match="not a number"):
        FrequencyQuantile(cast("float", "half"), over=QuantilePopulation.TYPES)


def test_quantiles_are_equal_on_their_share_and_their_scale() -> None:
    one = FrequencyQuantile(0.1, over=QuantilePopulation.TYPES)

    assert one == FrequencyQuantile(0.1, over=QuantilePopulation.TYPES)
    assert hash(one) == hash(FrequencyQuantile(0.1, over=QuantilePopulation.TYPES))
    assert one != FrequencyQuantile(0.2, over=QuantilePopulation.TYPES)
    assert one != FrequencyQuantile(0.1, over=QuantilePopulation.TOKENS)
    assert one != "quantile"
    assert (
        repr(one) == "FrequencyQuantile(0.1, over=<QuantilePopulation.TYPES: 'types'>)"
    )


def test_a_category_class_takes_every_form_bearing_a_named_category() -> None:
    profile = drawn(SMALL, coupling=1.0)
    selected = CategoryClass({CLOSED}).admitted(profile)

    assert selected == frozenset(
        form for form, category in lexicon().items() if category == CLOSED
    )
    assert len(selected) == CLASS_SIZES[CLOSED]
    assert CategoryClass({CLOSED, "P"}).admitted(profile) == selected | frozenset(
        form for form, category in lexicon().items() if category == "P"
    )


def test_a_category_class_names_at_least_one_category() -> None:
    with pytest.raises(ValueError, match="at least one category"):
        CategoryClass[str](())


def test_category_classes_are_equal_on_the_categories_they_name() -> None:
    one = CategoryClass({CLOSED})

    assert one == CategoryClass({CLOSED})
    assert hash(one) == hash(CategoryClass({CLOSED}))
    assert one != CategoryClass({"P"})
    assert one != "class"
    assert repr(one) == "CategoryClass({'D'})"


def test_a_closed_role_addresses_the_smallest_class() -> None:
    profile = drawn(SMALL, coupling=0.0)
    closed = CategoryRole[str](ClassRole.CLOSED, 0.25)

    assert closed.classes(profile) == {CLOSED}
    assert closed.admitted(profile) == CategoryClass({CLOSED}).admitted(profile)
    assert CategoryRole[str](ClassRole.CLOSED, 0.5).classes(profile) == {CLOSED, "P"}
    assert CategoryRole[str](ClassRole.OPEN, 0.25).classes(profile) == {"N"}
    assert CategoryRole[str](ClassRole.CLOSED, 0.0).admitted(profile) == frozenset()
    assert CategoryRole[str](ClassRole.OPEN, 1.0).admitted(profile) == frozenset(
        profile.counts
    )
    assert closed.size_invariant


def test_a_role_takes_every_category_tied_with_the_last_one_taken() -> None:
    profile = CorpusProfile(
        {("a",): 1, ("b",): 1, ("c",): 1, ("d",): 1},
        {
            ("a",): frozenset({"X"}),
            ("b",): frozenset({"Y"}),
            ("c",): frozenset({"Z"}),
            ("d",): frozenset({"Z"}),
        },
    )

    assert CategoryRole[str](ClassRole.CLOSED, 0.3).classes(profile) == {"X", "Y"}
    assert CategoryRole[str](ClassRole.OPEN, 0.3).classes(profile) == {"Z"}


def test_a_role_addresses_the_analogous_class_whatever_it_is_named() -> None:
    renamed = {CLOSED: "Det", "P": "Prep", "V": "Verb", "N": "Noun"}
    profile = drawn(SMALL, coupling=1.0)
    relabeled = CorpusProfile(
        profile.counts,
        {
            form: frozenset(renamed[category] for category in held)
            for form, held in profile.categories.items()
        },
    )
    closed = CategoryRole[str](ClassRole.CLOSED, 0.25)

    assert closed.classes(relabeled) == {"Det"}
    assert closed.admitted(relabeled) == closed.admitted(profile)


def test_a_role_refuses_a_share_outside_the_unit_interval() -> None:
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        CategoryRole[str](ClassRole.CLOSED, 1.5)


def test_roles_are_equal_on_their_end_and_share() -> None:
    one = CategoryRole[str](ClassRole.CLOSED, 0.25)

    assert one == CategoryRole[str](ClassRole.CLOSED, 0.25)
    assert hash(one) == hash(CategoryRole[str](ClassRole.CLOSED, 0.25))
    assert one != CategoryRole[str](ClassRole.OPEN, 0.25)
    assert one != CategoryRole[str](ClassRole.CLOSED, 0.5)
    assert one != "role"
    assert one.role is ClassRole.CLOSED
    assert one.share == 0.25
    assert repr(one) == "CategoryRole(<ClassRole.CLOSED: 'closed'>, 0.25)"


def test_a_rank_policy_and_a_class_policy_pick_the_same_closed_class() -> None:
    profile = drawn(LARGE, coupling=1.0)
    by_rank = RankBand(0.0, CLOSED_SHARE).admitted(profile)
    by_class = CategoryClass({CLOSED}).admitted(profile)

    assert by_rank == by_class
    assert agreement(by_rank, by_class, profile) == pytest.approx(1.0)


def test_a_token_quantile_covers_the_share_it_names_within_one_form() -> None:
    profile = drawn(LARGE, coupling=1.0)
    wanted = 0.1
    quantile = FrequencyQuantile(wanted, over=QuantilePopulation.TOKENS)
    selected = quantile.admitted(profile)
    report = SelectionReport.of(
        profile, {form: profile.counts[form] for form in selected}, quantile
    )
    granularity = profile.share(max(profile.counts.values()))

    assert report.mass_share >= wanted
    assert report.mass_share - wanted <= granularity
    assert report.types_pinned == len(selected)
    assert report.tokens_covered == sum(profile.counts[form] for form in selected)
    assert report.mass_share == pytest.approx(report.tokens_covered / profile.tokens)


def test_the_report_counts_types_tokens_mass_and_categories_from_the_corpus() -> None:
    profile = drawn(LARGE, coupling=1.0)
    declared = lexicon()
    selector = CategoryClass({CLOSED, "P"})
    selected = selector.admitted(profile)
    report = SelectionReport.of(
        profile, {form: profile.counts[form] for form in selected}, selector
    )

    assert report.types_pinned == CLASS_SIZES[CLOSED] + CLASS_SIZES["P"]
    assert report.tokens_covered == sum(profile.counts[form] for form in selected)
    assert report.mass_share == pytest.approx(report.tokens_covered / profile.tokens)
    assert set(report.by_category) == {CLOSED, "P"}

    for category in (CLOSED, "P"):
        borne = [form for form in selected if declared[form] == category]
        tokens = sum(profile.counts[form] for form in borne)

        assert report.by_category[category] == CategoryBudget(
            len(borne), tokens, tokens / profile.tokens
        )

    assert report.concordance == profile.concordance
    assert report.selector == selector
    assert report.size_invariant


def test_a_report_names_an_absolute_count_selector() -> None:
    profile = drawn(SMALL, coupling=1.0)
    policy = SelectionPolicy[str, str](UnionRule(), selector=FrequencyCount(3, None))
    selected = policy.selected(profile)
    report = policy.report(profile, {form: profile.counts[form] for form in selected})

    assert report.selector == FrequencyCount(3, None)
    assert not report.size_invariant
    assert "FrequencyCount(3, None)" in repr(report)


def test_the_report_refuses_a_form_the_corpus_does_not_hold() -> None:
    profile = CorpusProfile({("a",): 2}, {("a",): frozenset({"X"})})

    with pytest.raises(ValueError, match="does not hold"):
        SelectionReport.of(profile, {("b",): 1}, RankBand(0.0, 1.0))

    with pytest.raises(ValueError, match="covers 3 tokens"):
        SelectionReport.of(profile, {("a",): 3}, RankBand(0.0, 1.0))


def test_a_report_row_may_not_outrun_the_selection_it_belongs_to() -> None:
    with pytest.raises(ValueError, match="of the 1 forms"):
        SelectionReport(
            1, 4, 0.5, {"X": CategoryBudget(2, 4, 0.5)}, selector=RankBand(0.0, 1.0)
        )

    with pytest.raises(ValueError, match="of the 4 tokens"):
        SelectionReport(
            2, 4, 0.5, {"X": CategoryBudget(1, 5, 0.5)}, selector=RankBand(0.0, 1.0)
        )

    with pytest.raises(ValueError, match="is a fraction"):
        SelectionReport(1, 1, 1.5, {}, selector=RankBand(0.0, 1.0))


def test_reports_are_equal_on_every_quantity_they_name() -> None:
    rows = {"X": CategoryBudget(1, 4, 0.5)}
    one = SelectionReport(1, 4, 0.5, rows, 0.25, selector=RankBand(0.0, 1.0))

    assert one == SelectionReport(1, 4, 0.5, rows, 0.25, selector=RankBand(0.0, 1.0))
    assert hash(one) == hash(
        SelectionReport(1, 4, 0.5, rows, 0.25, selector=RankBand(0.0, 1.0))
    )
    assert one != SelectionReport(1, 4, 0.5, rows, 0.5, selector=RankBand(0.0, 1.0))
    assert one != SelectionReport(1, 4, 0.5, {}, 0.25, selector=RankBand(0.0, 1.0))
    assert one != SelectionReport(
        1, 4, 0.5, rows, 0.25, selector=FrequencyCount(1, None)
    )
    assert one != "report"
    assert repr(one) == (
        "SelectionReport(1, 4, 0.5, {'X': CategoryBudget(1, 4, 0.5)}, 0.25, "
        "selector=RankBand(0.0, 1.0))"
    )
    assert repr(CategoryBudget(1, 4, 0.5)) == "CategoryBudget(1, 4, 0.5)"
    assert CategoryBudget(1, 4, 0.5) != "budget"


def test_agreement_is_one_for_one_selection_and_chance_for_a_disjoint_pair() -> None:
    profile = drawn(SMALL, coupling=1.0)
    closed = CategoryClass({CLOSED}).admitted(profile)
    other = CategoryClass({"P"}).admitted(profile)

    disjoint = agreement(closed, other, profile)

    assert agreement(closed, closed, profile) == pytest.approx(1.0)
    assert disjoint is not None
    assert disjoint < 0.0
    assert agreement(frozenset(profile.counts), closed, profile) == pytest.approx(0.0)
    assert agreement(frozenset(), frozenset(), profile) is None
    assert (
        agreement(frozenset(profile.counts), frozenset(profile.counts), profile) is None
    )


def test_agreement_refuses_a_form_the_corpus_does_not_hold() -> None:
    profile = CorpusProfile({("a",): 1}, {("a",): frozenset({"X"})})

    with pytest.raises(ValueError, match="does not hold"):
        agreement(frozenset({("b",)}), frozenset(), profile)


def test_rank_and_class_agreement_rises_with_the_reported_concordance() -> None:
    points = []

    for coupling in COUPLINGS:
        profile = drawn(LARGE, coupling=coupling)
        rank = RankBand(0.0, CLOSED_SHARE)
        by_rank = rank.admitted(profile)
        by_class = CategoryClass({CLOSED}).admitted(profile)
        report = SelectionReport.of(
            profile, {form: profile.counts[form] for form in by_rank}, rank
        )
        measured = agreement(by_rank, by_class, profile)

        assert report.concordance is not None
        assert measured is not None

        points.append((report.concordance, measured))

    by_concordance = [measured for _, measured in sorted(points)]
    half = len(by_concordance) // 2
    lower = by_concordance[:half]
    upper = by_concordance[half:]

    assert by_concordance == sorted(by_concordance)
    assert sum(upper) / len(upper) > sum(lower) / len(lower)
    assert abs(points[0][1]) < CHANCE_TOLERANCE
    assert points[-1][1] == pytest.approx(1.0)


def test_a_policy_holds_its_selector() -> None:
    selector = RankBand(0.0, 0.25)
    policy = SelectionPolicy[str, str](UnionRule(), selector=selector)

    assert policy.selector == selector
    assert policy.with_seed(3).selector == selector
    assert policy != SelectionPolicy[str, str](UnionRule())
    assert repr(policy) == (
        "SelectionPolicy(UnionRule(), selector=RankBand(0.0, 0.25), "
        "coverage=1.0, seed=0)"
    )


def test_a_policy_selects_the_forms_its_selector_admits() -> None:
    profile = drawn(SMALL, coupling=1.0)

    for selector in (
        RankBand(0.0, CLOSED_SHARE),
        FrequencyBand(0.01, 1.0),
        FrequencyCount(4, None),
        CategoryRole[str](ClassRole.CLOSED, 0.25),
    ):
        policy = SelectionPolicy[str, str](UnionRule(), selector=selector)

        assert policy.selected(profile) == selector.admitted(profile)

    assert SelectionPolicy[str, str](UnionRule()).selected(profile) == frozenset(
        profile.counts
    )


def test_the_size_invariance_law_fails_for_an_absolute_count() -> None:
    profile = drawn(SMALL, coupling=1.0)
    floor = max(profile.counts.values())
    fixture = SelectionPolicyFixture(
        SelectionPolicy[str, str](UnionRule(), selector=FrequencyCount(floor, None)),
        profile,
        drawn(LARGE, coupling=1.0),
        SHARE_TOLERANCE,
    )

    with pytest.raises(LawViolationError, match="further apart"):
        the_same_mass_share_is_covered_at_two_sizes(fixture)

    with pytest.raises(LawViolationError, match="counts"):
        the_selector_declares_no_absolute_count(fixture)


@pytest.fixture(
    params=[
        RankBand(0.0, CLOSED_SHARE),
        FrequencyBand(0.005, 1.0),
        FrequencyQuantile(0.1, over=QuantilePopulation.TYPES),
        FrequencyQuantile(0.25, over=QuantilePopulation.TOKENS),
        CategoryClass({CLOSED}),
        CategoryRole(ClassRole.CLOSED, 0.25),
    ],
    ids=[
        "rank",
        "frequency",
        "quantile-over-types",
        "quantile-over-tokens",
        "category-class",
        "closed-role",
    ],
)
def selection_policy_fixture(
    request: pytest.FixtureRequest,
) -> SelectionPolicyFixture[str]:
    selector = cast("FormSelector[str]", request.param)

    return SelectionPolicyFixture(
        SelectionPolicy[str, str](UnionRule(), selector=selector),
        drawn(SMALL, coupling=1.0),
        drawn(LARGE, coupling=1.0),
        SHARE_TOLERANCE,
    )


class TestShippingSelectionPolicies(SelectionPolicyConformance):
    pass


type ConstantClass = tuple[str, int | None]

QUANTIFIER: Final[ConstantClass] = ("quantifier", None)
CONNECTIVE: Final[ConstantClass] = ("connective", None)
UNARY_PREDICATE: Final[ConstantClass] = ("predicate", 1)
BINARY_PREDICATE: Final[ConstantClass] = ("predicate", 2)

CONSTANT_CLASS_SIZES: Final[Mapping[ConstantClass, int]] = {
    QUANTIFIER: 2,
    CONNECTIVE: 10,
    UNARY_PREDICATE: 40,
    BINARY_PREDICATE: 160,
}
"""How many logical forms each constant class holds.

The classes differ by an order of magnitude, as the syntactic categories do, so
a positive coupling makes the quantifiers the most frequent forms.
"""

CONSTANT_CLASS_PREFIXES: Final[Mapping[ConstantClass, str]] = {
    QUANTIFIER: "q",
    CONNECTIVE: "c",
    UNARY_PREDICATE: "u",
    BINARY_PREDICATE: "b",
}


def semantic_lexicon() -> dict[Form, ConstantClass]:
    return {
        (f"{CONSTANT_CLASS_PREFIXES[held]}{index}",): held
        for held, size in CONSTANT_CLASS_SIZES.items()
        for index in range(size)
    }


def semantic_drawn(tokens: int, *, coupling: float) -> CorpusProfile[ConstantClass]:
    declared = semantic_lexicon()
    weights = [
        float(CONSTANT_CLASS_SIZES[held]) ** -coupling for held in declared.values()
    ]
    counts = Counter(Random(SEED).choices(list(declared), weights=weights, k=tokens))

    return CorpusProfile(
        {form: counts[form] for form in declared},
        {form: frozenset({held}) for form, held in declared.items()},
    )


def test_a_class_of_two_place_predicates_is_named_by_its_arity() -> None:
    profile = semantic_drawn(SMALL, coupling=1.0)
    declared = semantic_lexicon()
    selected = CategoryClass({BINARY_PREDICATE}).admitted(profile)

    assert selected == frozenset(
        form for form, held in declared.items() if held == BINARY_PREDICATE
    )
    assert CategoryClass({UNARY_PREDICATE}).admitted(profile) & selected == frozenset()


class TestShippingSemanticSelectionPolicies(SelectionPolicyConformance):
    @pytest.fixture(
        params=[
            RankBand(
                0.0,
                CONSTANT_CLASS_SIZES[QUANTIFIER] / sum(CONSTANT_CLASS_SIZES.values()),
            ),
            CategoryRole(ClassRole.CLOSED, 0.25),
            FrequencyQuantile(0.1, over=QuantilePopulation.TYPES),
            FrequencyQuantile(0.25, over=QuantilePopulation.TOKENS),
            CategoryClass({QUANTIFIER}),
            CategoryClass({BINARY_PREDICATE}),
        ],
        ids=[
            "rank",
            "closed-role",
            "quantile-over-types",
            "quantile-over-tokens",
            "quantifier-class",
            "two-place-predicate-class",
        ],
    )
    def selection_policy_fixture(
        self, request: pytest.FixtureRequest
    ) -> SelectionPolicyFixture[ConstantClass]:
        selector = cast("FormSelector[ConstantClass]", request.param)

        return SelectionPolicyFixture(
            SelectionPolicy[ConstantClass, ConstantClass](
                UnionRule(), selector=selector
            ),
            semantic_drawn(SMALL, coupling=1.0),
            semantic_drawn(LARGE, coupling=1.0),
            SHARE_TOLERANCE,
        )
