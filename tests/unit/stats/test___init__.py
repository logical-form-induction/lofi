from math import fsum
from statistics import NormalDist, fmean
from typing import TYPE_CHECKING, cast

import pytest

import lofi.stats

if TYPE_CHECKING:
    from collections.abc import Sequence


def enumerated(observed: float, *, null: str) -> lofi.stats.PValue:
    return lofi.stats.p_value_against_replicates(
        observed,
        [float(step) for step in range(20)],
        tail="greater",
        null=null,
        family="permutation",
        exact=True,
        seed=None,
    )


def enumerated_under(*, family: str, tail: str) -> lofi.stats.PValue:
    return lofi.stats.p_value_against_replicates(
        19.0,
        [float(step) for step in range(20)],
        tail=cast("lofi.stats.Tail", tail),
        null="rank permutation",
        family=cast("lofi.stats.NullFamily", family),
        exact=True,
        seed=None,
    )


def test_a_family_is_tested_and_corrected_through_the_published_names() -> None:
    strong = enumerated(19.0, null="rank permutation")
    weak = enumerated(4.0, null="rank permutation")
    adjusted = tuple(min(1.0, 2 * result.value) for result in (strong, weak))
    decisions = lofi.stats.Decisions(
        "bonferroni",
        0.15,
        lofi.stats.Dependence.ARBITRARY,
        (strong, weak),
        adjusted,
    )

    assert strong.value == 0.05
    assert weak.value == 0.8
    assert decisions.adjusted == (0.1, 1.0)
    assert decisions.reject == (True, False)
    assert decisions.n_rejected == 1
    assert decisions.raw == (strong, weak)
    assert decisions.raw[0].provenance.null == "rank permutation"


def test_the_same_family_survives_nothing_at_a_stricter_level() -> None:
    strong = enumerated(19.0, null="rank permutation")
    decisions = lofi.stats.Decisions(
        "bonferroni",
        0.05,
        lofi.stats.Dependence.ARBITRARY,
        (strong,),
        (0.1,),
    )

    assert decisions.reject == (False,)


def test_a_test_runs_under_every_declared_level_and_under_no_other() -> None:
    for null_family in lofi.stats.NULL_FAMILIES:
        for tail in lofi.stats.TAILS:
            tested = lofi.stats.p_value_against_replicates(
                19.0,
                [float(step) for step in range(20)],
                tail=tail,
                null="rank permutation",
                family=null_family,
                exact=True,
                seed=None,
            )

            assert (tested.provenance.family, tested.provenance.tail) == (
                null_family,
                tail,
            )

    with pytest.raises(ValueError, match="a null's family is 'jackknife'"):
        enumerated_under(family="jackknife", tail="greater")

    with pytest.raises(ValueError, match="a null's tail is 'both'"):
        enumerated_under(family="permutation", tail="both")


def test_a_paired_comparison_runs_end_to_end_through_the_published_names() -> None:
    scores = NormalDist(0.6, 1.0).samples(12, seed=1)
    baseline = NormalDist().samples(12, seed=2)
    tested = lofi.stats.paired_permutation(
        scores, baseline, replicates=999, seed=4, tail="greater"
    )
    ranked = lofi.stats.paired_rank_inversion(
        scores, [3.0 * score for score in scores], replicates=999, seed=4, tail="less"
    )

    assert isinstance(tested, lofi.stats.PValue)
    assert isinstance(ranked, lofi.stats.PValue)
    assert tested.provenance.family == "permutation"
    assert 0.0 < tested.value <= 1.0


def test_a_quantity_is_bounded_at_two_units_through_the_published_names() -> None:
    samples = NormalDist(2.0, 1.0).samples(36, seed=3)
    labels = [f"p{index // 6}" for index in range(36)]
    over_items, over_people = lofi.stats.paired_bootstrap(
        samples,
        fmean,
        lofi.stats.Bootstrap(unit="item", seed=5, replicates=999, level=0.95),
        lofi.stats.Bootstrap(
            unit="participant", seed=5, groups=labels, replicates=999, level=0.95
        ),
    )

    assert isinstance(over_items, lofi.stats.BootstrapResult)
    assert isinstance(over_items.percentile, lofi.stats.Interval)
    assert over_items.units == 36
    assert over_people.units == 6
    assert over_items.observed == over_people.observed


def test_a_family_is_corrected_by_a_published_procedure_under_its_regime() -> None:
    family = tuple(enumerated(float(rank), null="rank permutation") for rank in (19, 4))
    corrected = lofi.stats.Bonferroni().apply(
        family, 0.15, lofi.stats.Dependence.ARBITRARY
    )
    routed = lofi.stats.false_discovery_correction(lofi.stats.Dependence.ARBITRARY)
    controlled = routed.apply(family, 0.15, lofi.stats.Dependence.ARBITRARY)

    assert corrected.adjusted == pytest.approx((0.1, 1.0))
    assert corrected.reject == (True, False)
    assert corrected.raw == family
    assert isinstance(routed, lofi.stats.BenjaminiYekutieli)
    assert controlled.correction == "benjamini-yekutieli"
    assert controlled.raw == family

    with pytest.raises(lofi.stats.DependenceError, match="benjamini-hochberg"):
        lofi.stats.BenjaminiHochberg().apply(
            family, 0.15, lofi.stats.Dependence.ARBITRARY
        )


def leading_gap(scores: Sequence[float]) -> float:
    return scores[0] - fmean(scores[1:])


def test_three_nulls_test_and_correct_through_the_published_names() -> None:
    scores = NormalDist(0.4, 1.0).samples(9, seed=6)
    strata = tuple(
        ("short" if index % 2 else "long", score)
        for index, score in enumerate(NormalDist().samples(60, seed=7))
    )
    permuting = lofi.stats.LabelPermutation(replicates=199, seed=8)
    matching = lofi.stats.MatchedDraw(
        "length", strata, lambda unit: unit[0], replicates=199, seed=9
    )
    maximizing = lofi.stats.BootstrapMaxima((fmean,), replicates=199, seed=10)
    family = (
        permuting.test(leading_gap, scores, tail="greater"),
        matching.test(
            lambda units: fsum(value for _, value in units), strata[:6], tail="greater"
        ),
        maximizing.test(fmean, scores, tail="two-sided"),
    )
    corrected = lofi.stats.Holm().apply(family, 0.05, lofi.stats.Dependence.ARBITRARY)

    assert {member.provenance.family for member in family} == set(
        lofi.stats.NULL_FAMILIES
    )
    assert all(member.provenance.seed is not None for member in family)
    assert all(member.provenance.exact is False for member in family)
    assert corrected.raw == family
    assert len(corrected.adjusted) == len(family)
