from math import exp, fsum, sqrt
from operator import itemgetter
from statistics import NormalDist, fmean
from typing import TYPE_CHECKING, cast

import pytest
from hypothesis import strategies as st

from lofi.stats import (
    NULL_FAMILIES,
    BootstrapMaxima,
    LabelPermutation,
    MatchedDraw,
    NullModel,
    PValue,
    Tail,
)
from lofi.stats.laws import NULL_MODEL_LAWS, NullModelFixture, check_null_model_laws

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

EXAMPLES = 20
LAW_REPLICATES = 19

BINS = 11
REPLICATES = 1099
DATASETS = 300
LEVEL = 0.05
TRIALS = 100
WIDTH = 4

STANDARD = NormalDist()

STRATA = ("small", "large")
PATTERN = ("small", "small", "small", "large", "large", "large")

INVENTORY = ("i0", "i1", "i2", "i3")
GOLD = ("g0", "g1")


def uniforms(count: int, *, seed: int) -> tuple[float, ...]:
    return tuple(STANDARD.cdf(draw) for draw in STANDARD.samples(count, seed=seed))


def chi_square_survival(statistic: float, freedom: int) -> float:
    terms = [1.0]
    running = 1.0

    for step in range(1, freedom // 2):
        running *= statistic / 2.0 / step
        terms.append(running)

    return exp(-statistic / 2.0) * fsum(terms)


def uniformity(pvalues: Sequence[float]) -> float:
    counts = [0] * BINS

    for pvalue in pvalues:
        counts[min(BINS - 1, int(pvalue * BINS))] += 1

    expected = len(pvalues) / BINS
    statistic = fsum((count - expected) ** 2 / expected for count in counts)

    return chi_square_survival(statistic, BINS - 1)


def half_difference(values: Sequence[float]) -> float:
    half = len(values) // 2

    return fmean(values[:half]) - fmean(values[half:])


def first(values: Sequence[float]) -> float:
    return values[0]


def total(units: Sequence[tuple[str, float]]) -> float:
    return fsum(value for _, value in units)


def stratum(unit: tuple[str, float]) -> str:
    return unit[0]


def value_of(unit: tuple[str, float]) -> float:
    return unit[1]


def coordinate(at: int) -> Callable[[Sequence[Sequence[float]]], float]:

    def measured(units: Sequence[Sequence[float]]) -> float:
        return fmean([unit[at] for unit in units])

    return measured


def gold(pair: tuple[str, str]) -> str:
    return pair[1]


def sent_to(target: str) -> Callable[[Sequence[tuple[str, str]]], float]:

    def measured(pairs: Sequence[tuple[str, str]]) -> float:
        return sum(1 for induced, _ in pairs if induced == target) / len(pairs)

    return measured


def units_of(count: int, *, seed: int) -> list[tuple[float, ...]]:
    flat = STANDARD.samples(count * WIDTH, seed=seed)

    return [tuple(flat[at * WIDTH : (at + 1) * WIDTH]) for at in range(count)]


def a_pool(*, seed: int) -> tuple[tuple[str, float], ...]:
    return tuple(
        (STRATA[at % len(STRATA)], value)
        for at, value in enumerate(STANDARD.samples(200, seed=seed))
    )


POOL = a_pool(seed=77)
MEMBERS = {key: tuple(unit for unit in POOL if unit[0] == key) for key in STRATA}
RELABELING = tuple((induced, label) for induced in INVENTORY for label in GOLD)

FAMILY = tuple(coordinate(at) for at in range(WIDTH))


class _JustTests:
    def test(
        self,
        _statistic: Callable[[Sequence[float]], float],
        _observed: Sequence[float],
        /,
        *,
        tail: Tail,
    ) -> str:
        return tail


VECTORS = st.lists(
    st.floats(min_value=-50.0, max_value=50.0),
    min_size=4,
    max_size=6,
).map(tuple)
UNIT_LISTS = st.lists(
    st.tuples(
        st.floats(min_value=-50.0, max_value=50.0),
        st.floats(min_value=-50.0, max_value=50.0),
    ),
    min_size=4,
    max_size=6,
).map(tuple)
MATCHED_UNITS = st.lists(st.sampled_from(POOL), min_size=3, max_size=6).map(tuple)

PERMUTING = NullModelFixture(
    LabelPermutation(replicates=LAW_REPLICATES, seed=1),
    half_difference,
    VECTORS,
    EXAMPLES,
)
MAXIMIZING = NullModelFixture(
    BootstrapMaxima(FAMILY[:2], replicates=LAW_REPLICATES, seed=2),
    FAMILY[0],
    UNIT_LISTS,
    EXAMPLES,
)
MATCHING = NullModelFixture(
    MatchedDraw("stratum", POOL, stratum, replicates=LAW_REPLICATES, seed=3),
    total,
    MATCHED_UNITS,
    EXAMPLES,
)
FIXTURES = (PERMUTING, MAXIMIZING, MATCHING)
FIXTURE_NAMES = [fixture.model.name for fixture in FIXTURES]


def test_each_null_is_one_by_type_and_by_shape() -> None:
    permuting: NullModel[Sequence[float]] = LabelPermutation(seed=1, replicates=999)
    maximizing: NullModel[Sequence[Sequence[float]]] = BootstrapMaxima(
        FAMILY, seed=2, replicates=999
    )
    matching: NullModel[Sequence[tuple[str, float]]] = MatchedDraw(
        "stratum", POOL, stratum, seed=3, replicates=999
    )

    assert isinstance(permuting, NullModel)
    assert isinstance(maximizing, NullModel)
    assert isinstance(matching, NullModel)


def test_a_type_with_only_a_test_method_is_not_a_null_model() -> None:
    assert not isinstance(_JustTests(), NullModel)


def test_the_three_nulls_cover_every_declared_family() -> None:
    covered = {fixture.model.family for fixture in FIXTURES}

    assert covered == set(NULL_FAMILIES)


def test_every_null_declares_the_budget_and_the_seed_its_p_values_record() -> None:
    values = STANDARD.samples(6, seed=30)
    units = units_of(6, seed=31)
    tested = (
        PERMUTING.model.test(half_difference, values, tail="greater"),
        MAXIMIZING.model.test(FAMILY[0], units, tail="greater"),
        MATCHING.model.test(total, MEMBERS["small"][:4], tail="greater"),
    )

    for fixture, pvalue in zip(FIXTURES, tested, strict=True):
        model = fixture.model

        assert isinstance(pvalue, PValue)
        assert model.replicates == LAW_REPLICATES
        assert pvalue.provenance.null == model.name
        assert pvalue.provenance.family == model.family
        assert pvalue.provenance.seed == model.seed
        assert pvalue.provenance.exact is False
        assert dict(pvalue.provenance.matching) == dict(model.matching)


def test_a_statistic_no_reordering_moves_is_never_extreme() -> None:
    values = STANDARD.samples(8, seed=12)
    tested = LabelPermutation(replicates=99, seed=4).test(fsum, values, tail="greater")

    assert tested.value == 1.0


def test_a_relabeling_reaches_each_position_as_often_as_any_other() -> None:
    # the largest of four values leads one arrangement in four
    tested = LabelPermutation(replicates=999, seed=5).test(
        first, (4.0, 1.0, 2.0, 3.0), tail="greater"
    )

    assert 0.2 < tested.value < 0.3


def test_a_relabeling_null_is_uniform_where_the_assignment_carries_nothing() -> None:
    pvalues = [
        LabelPermutation(replicates=REPLICATES, seed=10000 + index)
        .test(half_difference, STANDARD.samples(8, seed=index), tail="greater")
        .value
        for index in range(DATASETS)
    ]

    assert uniformity(pvalues) > 0.01
    assert min(pvalues) < LEVEL


def test_a_relabeling_null_finds_a_planted_difference() -> None:
    planted = [
        *NormalDist(2.0, 1.0).samples(6, seed=99),
        *STANDARD.samples(6, seed=98),
    ]
    tested = LabelPermutation(replicates=REPLICATES, seed=6).test(
        half_difference, planted, tail="greater"
    )

    assert tested.value < 0.01


def test_a_relabeling_null_refuses_data_it_cannot_rearrange() -> None:
    with pytest.raises(ValueError, match="needs at least 2 units"):
        LabelPermutation(seed=7, replicates=999).test(fsum, (1.0,), tail="greater")


def test_a_sampling_null_refuses_a_budget_or_a_seed_it_cannot_draw_under() -> None:
    with pytest.raises(ValueError, match="no replicate budget"):
        LabelPermutation(replicates=0, seed=8)

    with pytest.raises(ValueError, match="a null's seed is negative"):
        MatchedDraw("stratum", POOL, stratum, seed=-1, replicates=999)

    with pytest.raises(TypeError, match="a replicate budget is not an integer"):
        BootstrapMaxima(FAMILY, replicates=cast("int", 1.5), seed=1)


def test_a_maximum_over_a_larger_family_is_never_easier_to_beat() -> None:
    # one seed gives both nulls the same resampled units
    units = units_of(30, seed=13)
    alone = BootstrapMaxima(FAMILY[:1], replicates=199, seed=9)
    among = BootstrapMaxima(FAMILY, replicates=199, seed=9)
    raised = tuple(
        zip(
            among.reference(units, tail="greater"),
            alone.reference(units, tail="greater"),
            strict=True,
        )
    )
    narrow = alone.test(FAMILY[0], units, tail="greater")
    wide = among.test(FAMILY[0], units, tail="greater")

    assert all(wider >= narrower for wider, narrower in raised)
    assert any(wider > narrower for wider, narrower in raised)
    assert wide.value > narrow.value
    assert wide.provenance.matching["family"] == len(FAMILY)
    assert narrow.provenance.matching["family"] == 1


def test_a_two_sided_maximum_is_symmetric_and_twice_as_long() -> None:
    units = units_of(20, seed=14)
    null = BootstrapMaxima(FAMILY, replicates=50, seed=10)
    both = null.reference(units, tail="two-sided")
    upper = null.reference(units, tail="greater")

    assert len(both) == 2 * len(upper)
    assert sorted(both) == sorted(-value for value in both)
    assert null.test(FAMILY[0], units, tail="two-sided").provenance.replicates == 100


def test_a_family_wise_null_holds_its_level_under_no_signal() -> None:
    pvalues = []

    for index in range(TRIALS):
        units = units_of(30, seed=index)
        observed = [member(units) for member in FAMILY]
        top = FAMILY[observed.index(max(observed))]
        null = BootstrapMaxima(FAMILY, replicates=199, seed=10000 + index)

        pvalues.append(null.test(top, units, tail="two-sided").value)

    rejected = sum(1 for pvalue in pvalues if pvalue <= LEVEL) / TRIALS
    slack = 2.0 * sqrt(LEVEL * (1.0 - LEVEL) / TRIALS)

    assert rejected <= LEVEL + slack
    assert sum(1 for pvalue in pvalues if pvalue <= 0.5) / TRIALS > 0.1


def test_a_family_wise_null_finds_a_planted_effect_and_not_its_neighbors() -> None:
    flat = STANDARD.samples(WIDTH * 30, seed=4321)
    units = [
        (flat[at * WIDTH] + 2.0, *flat[at * WIDTH + 1 : (at + 1) * WIDTH])
        for at in range(30)
    ]
    null = BootstrapMaxima(FAMILY, replicates=999, seed=11)

    assert null.test(FAMILY[0], units, tail="two-sided").value < 0.01
    assert null.test(FAMILY[1], units, tail="two-sided").value > 0.5


def test_a_family_wise_null_refuses_a_statistic_it_never_took_a_maximum_over() -> None:
    units = units_of(10, seed=15)

    with pytest.raises(ValueError, match="is not one of the 4"):
        BootstrapMaxima(FAMILY, seed=12, replicates=999).test(
            coordinate(0), units, tail="greater"
        )


def test_a_family_wise_null_refuses_a_family_it_cannot_evaluate() -> None:
    empty: tuple[Callable[[Sequence[Sequence[float]]], float], ...] = ()

    with pytest.raises(ValueError, match="no family to take maxima over"):
        BootstrapMaxima(empty, seed=13, replicates=999)

    with pytest.raises(TypeError, match="is not callable"):
        BootstrapMaxima(
            cast("tuple[Callable[[Sequence[Sequence[float]]], float]]", (1.0,)),
            seed=14,
            replicates=999,
        )

    with pytest.raises(ValueError, match="not one of greater, less, two-sided"):
        BootstrapMaxima(FAMILY, seed=15, replicates=999).reference(
            units_of(4, seed=16), tail=cast("Tail", "both")
        )


def test_a_matched_draw_never_leaves_the_stratum_it_matched_on() -> None:
    pool = (("low", 0.0), ("low", 0.0), ("high", 1.0), ("high", 1.0))
    observed = (("low", 0.0), ("low", 0.0), ("high", 1.0))
    null = MatchedDraw("value band", pool, stratum, replicates=99, seed=16)

    assert set(null.reference(total, observed)) == {1.0}
    assert null.test(total, observed, tail="greater").value == 1.0
    assert null.name == "matched on value band"
    assert dict(null.matching) == {"criterion": "value band", "strata": 2, "pool": 4}


def test_a_matched_draw_over_the_gold_column_is_a_relabeling_floor() -> None:
    # each gold label is matched uniformly to four induced labels
    observed = tuple(
        (INVENTORY[at % len(INVENTORY)], GOLD[at % len(GOLD)]) for at in range(40)
    )
    null = MatchedDraw("gold label", RELABELING, gold, replicates=999, seed=17)
    floor = fmean(null.reference(sent_to(INVENTORY[0]), observed))

    assert abs(floor - 1.0 / len(INVENTORY)) < 0.01


def test_a_matched_null_is_uniform_where_the_observation_is_an_ordinary_member() -> (
    None
):
    pvalues = []

    for index in range(DATASETS):
        picks = uniforms(len(PATTERN), seed=index)
        observed = tuple(
            MEMBERS[key][min(len(MEMBERS[key]) - 1, int(pick * len(MEMBERS[key])))]
            for key, pick in zip(PATTERN, picks, strict=True)
        )
        null = MatchedDraw("stratum", POOL, stratum, replicates=REPLICATES, seed=index)

        pvalues.append(null.test(total, observed, tail="greater").value)

    assert uniformity(pvalues) > 0.01
    assert min(pvalues) < LEVEL


def test_a_matched_null_finds_an_observation_planted_at_the_top() -> None:
    planted = tuple(max(MEMBERS[key], key=itemgetter(1)) for key in PATTERN)
    tested = MatchedDraw("stratum", POOL, stratum, replicates=REPLICATES, seed=18).test(
        total, planted, tail="greater"
    )

    assert tested.value < 0.01


def test_a_matched_null_refuses_a_unit_its_pool_cannot_match() -> None:
    null = MatchedDraw("stratum", MEMBERS["small"], stratum, seed=19, replicates=999)

    with pytest.raises(ValueError, match="holding no 'large'"):
        null.test(total, (("large", 1.0),), tail="greater")


def test_a_matched_null_refuses_a_pool_and_a_criterion_it_cannot_use() -> None:
    empty: tuple[tuple[str, float], ...] = ()

    with pytest.raises(ValueError, match="an empty pool"):
        MatchedDraw("stratum", empty, stratum, seed=20, replicates=999)

    with pytest.raises(ValueError, match="a matching criterion is empty"):
        MatchedDraw("", POOL, stratum, seed=21, replicates=999)

    with pytest.raises(TypeError, match="a stratum is not a string"):
        MatchedDraw(
            "stratum",
            POOL,
            cast("Callable[[tuple[str, float]], str]", value_of),
            seed=22,
            replicates=999,
        )


@pytest.mark.conformance
@pytest.mark.parametrize("fixture", FIXTURES, ids=FIXTURE_NAMES)
def test_every_law_holds_of_every_null(
    fixture: NullModelFixture[Sequence[float]],
) -> None:
    check_null_model_laws(fixture)


@pytest.mark.conformance
@pytest.mark.parametrize("law", NULL_MODEL_LAWS, ids=lambda law: law.__name__)
def test_each_law_runs_against_the_matched_null_on_its_own(
    law: Callable[[NullModelFixture[Sequence[tuple[str, float]]]], None],
) -> None:
    law(MATCHING)
