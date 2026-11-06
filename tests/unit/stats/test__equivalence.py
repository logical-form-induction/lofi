from itertools import product
from statistics import NormalDist, fmean
from typing import cast

import pytest

from lofi.stats import (
    Bootstrap,
    Equivalence,
    EquivalenceResult,
    MarginTest,
    PValue,
    Tail,
    Verdict,
    p_value_against_replicates,
    paired_permutation,
)

STANDARD = NormalDist()

# eight items give 256 exact reassignments
FEW = 8

MANY = 60


def drifting(
    count: int, *, by: float, spread: float, seed: int
) -> tuple[list[float], list[float]]:
    base = STANDARD.samples(count, seed=seed)
    step = NormalDist(by, spread).samples(count, seed=seed + 1)

    return ([one + other for one, other in zip(base, step, strict=True)], base)


def noisy() -> tuple[list[float], list[float]]:
    return drifting(FEW, by=0.1, spread=2.0, seed=21)


def one_sided(value: float, tail: Tail) -> PValue:
    return p_value_against_replicates(
        value,
        [float(step) for step in range(20)],
        tail=tail,
        null="paired sign flips",
        family="permutation",
        exact=True,
        seed=None,
    )


def test_a_difference_well_inside_a_generous_margin_is_equivalent() -> None:
    scores, base = drifting(MANY, by=0.02, spread=0.1, seed=11)

    result = Equivalence(margin=0.5, seed=7, replicates=999, level=0.05).compare(
        scores, base
    )

    assert result.verdict is Verdict.EQUIVALENT_WITHIN_MARGIN
    assert result.lower.inside.value <= result.alpha
    assert result.upper.inside.value <= result.alpha


def test_a_difference_beyond_the_margin_is_separated() -> None:
    scores, base = drifting(MANY, by=1.0, spread=0.2, seed=13)

    result = Equivalence(margin=0.3, seed=7, replicates=999, level=0.05).compare(
        scores, base
    )

    assert result.verdict is Verdict.SEPARATED
    assert result.upper.outside.value <= result.alpha
    assert result.observed > result.margin


def test_a_difference_beyond_the_margin_the_other_way_is_separated() -> None:
    scores, base = drifting(MANY, by=1.0, spread=0.2, seed=13)

    result = Equivalence(margin=0.3, seed=7, replicates=999, level=0.05).compare(
        base, scores
    )

    assert result.verdict is Verdict.SEPARATED
    assert result.lower.outside.value <= result.alpha
    assert result.observed < -result.margin


def test_a_difference_the_data_cannot_place_is_indeterminate() -> None:
    scores, base = noisy()
    margin = 0.3
    differences = [one - other for one, other in zip(scores, base, strict=True)]
    straddling = Bootstrap(unit="item", seed=2, replicates=999, level=0.95).interval(
        differences, fmean
    )

    result = Equivalence(margin=margin, seed=7, replicates=999, level=0.05).compare(
        scores, base
    )

    assert straddling.percentile.covers(-margin)
    assert straddling.percentile.covers(margin)
    assert result.verdict is Verdict.INDETERMINATE


def test_a_difference_that_does_not_test_significant_is_not_equivalence() -> None:
    scores, base = noisy()
    tested = paired_permutation(scores, base, replicates=999, seed=7, tail="two-sided")

    result = Equivalence(margin=0.3, seed=7, replicates=999, level=0.05).compare(
        scores, base
    )

    assert tested.value > 0.05
    assert result.verdict is not Verdict.EQUIVALENT_WITHIN_MARGIN


def test_the_boundary_p_values_are_the_tails_of_the_enumerated_null() -> None:
    scores, base = noisy()
    margin = 0.3
    shifted = [one + margin - other for one, other in zip(scores, base, strict=True)]
    observed = sum(shifted) / len(shifted)
    reference = [
        sum(sign * value for sign, value in zip(signs, shifted, strict=True))
        / len(shifted)
        for signs in product((1.0, -1.0), repeat=len(shifted))
    ]

    result = Equivalence(margin=margin, seed=7, replicates=999, level=0.05).compare(
        scores, base
    )

    assert len(reference) == 2**FEW
    assert result.lower.inside.value == pytest.approx(
        sum(1 for mean in reference if mean >= observed) / len(reference)
    )
    assert result.lower.outside.value == pytest.approx(
        sum(1 for mean in reference if mean <= observed) / len(reference)
    )
    assert result.lower.inside.provenance.exact is True


@pytest.mark.parametrize(
    ("group_totals", "tails", "verdict"),
    [
        (
            (0.0, 0.0, 0.0),
            (1 / 8, 1.0, 1 / 8, 1.0),
            Verdict.EQUIVALENT_WITHIN_MARGIN,
        ),
        ((4.0, 4.0, 4.0), (1 / 8, 1.0, 1.0, 1 / 8), Verdict.SEPARATED),
        ((-6.0, -6.0, -2.0), (1.0, 1 / 4, 1 / 8, 1.0), Verdict.INDETERMINATE),
    ],
)
def test_grouped_boundaries_produce_each_verdict(
    group_totals: tuple[float, ...],
    tails: tuple[float, float, float, float],
    verdict: Verdict,
) -> None:
    differences = [total / 2 for total in group_totals for _ in range(2)]
    groups = [label for label in ("a", "b", "c") for _ in range(2)]

    result = Equivalence(margin=1.0, seed=7, level=0.2, replicates=999).compare(
        differences, [0.0] * len(differences), groups=groups
    )

    assert (
        result.lower.inside.value,
        result.lower.outside.value,
        result.upper.inside.value,
        result.upper.outside.value,
    ) == pytest.approx(tails)
    assert result.lower.boundary == -1.0
    assert result.upper.boundary == 1.0
    assert result.observed == pytest.approx(sum(group_totals) / len(differences))
    assert result.lower.inside.provenance.replicates == 8
    assert result.upper.outside.provenance.replicates == 8
    assert result.verdict is verdict


def test_the_two_tails_of_a_boundary_cover_the_reference_distribution() -> None:
    scores, base = drifting(MANY, by=0.2, spread=0.5, seed=17)

    result = Equivalence(margin=0.4, seed=7, replicates=999, level=0.05).compare(
        scores, base
    )

    for boundary in (result.lower, result.upper):
        assert boundary.inside.value + boundary.outside.value >= 1.0


def test_a_difference_is_never_both_equivalent_and_separated() -> None:
    scores, base = drifting(MANY, by=0.3, spread=0.6, seed=19)

    for margin in (0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0):
        result = Equivalence(margin=margin, seed=7, replicates=999, level=0.05).compare(
            scores, base
        )
        equivalent = (
            result.lower.inside.value <= result.alpha
            and result.upper.inside.value <= result.alpha
        )
        separated = (
            result.lower.outside.value <= result.alpha
            or result.upper.outside.value <= result.alpha
        )

        assert not (equivalent and separated)
        assert (result.verdict is Verdict.EQUIVALENT_WITHIN_MARGIN) == equivalent
        assert (result.verdict is Verdict.SEPARATED) == separated


def test_widening_the_margin_walks_a_verdict_from_separated_to_equivalent() -> None:
    scores, base = drifting(MANY, by=0.5, spread=0.4, seed=23)
    verdicts = [
        Equivalence(margin=margin, seed=7, replicates=999, level=0.05)
        .compare(scores, base)
        .verdict
        for margin in (0.1, 0.3, 1.0)
    ]

    assert verdicts == [
        Verdict.SEPARATED,
        Verdict.SEPARATED,
        Verdict.EQUIVALENT_WITHIN_MARGIN,
    ]


def test_the_observed_difference_is_the_mean_per_item_difference() -> None:
    scores, base = drifting(MANY, by=0.4, spread=0.5, seed=29)

    result = Equivalence(margin=0.5, seed=7, replicates=999, level=0.05).compare(
        scores, base
    )

    assert result.observed == pytest.approx(
        fmean([one - other for one, other in zip(scores, base, strict=True)])
    )
    assert result.margin == 0.5
    assert result.lower.boundary == -0.5
    assert result.upper.boundary == 0.5


def test_one_seed_gives_one_result() -> None:
    scores, base = drifting(MANY, by=0.2, spread=0.9, seed=31)
    once = Equivalence(margin=0.4, seed=3, replicates=999, level=0.05).compare(
        scores, base
    )
    again = Equivalence(margin=0.4, seed=3, replicates=999, level=0.05).compare(
        scores, base
    )
    elsewhere = Equivalence(margin=0.4, seed=4, replicates=999, level=0.05).compare(
        scores, base
    )

    assert once == again
    assert hash(once) == hash(again)
    assert once != elsewhere


def test_a_test_with_no_margin_cannot_be_built() -> None:
    without: dict[str, int] = {"seed": 1}

    with pytest.raises(TypeError, match="margin"):
        Equivalence(**without)


def test_a_margin_that_is_not_positive_is_refused() -> None:
    with pytest.raises(ValueError, match="not positive"):
        Equivalence(margin=0.0, seed=1, replicates=999, level=0.05)


def test_a_level_at_or_above_one_half_is_refused() -> None:
    with pytest.raises(ValueError, match=r"not below 0\.5"):
        Equivalence(margin=0.3, seed=1, level=0.5, replicates=999)


def test_a_test_with_no_replicate_budget_is_refused() -> None:
    with pytest.raises(ValueError, match="no replicate budget"):
        Equivalence(margin=0.3, replicates=0, seed=1, level=0.05)


def test_a_negative_seed_is_refused() -> None:
    with pytest.raises(ValueError, match="negative"):
        Equivalence(margin=0.3, seed=-1, replicates=999, level=0.05)


def test_two_sequences_of_different_length_are_refused() -> None:
    with pytest.raises(ValueError, match="on one side"):
        Equivalence(margin=0.3, seed=1, replicates=999, level=0.05).compare(
            [1.0, 2.0], [1.0]
        )


def test_a_boundary_read_in_one_tail_twice_is_refused() -> None:
    once = one_sided(19.0, "greater")

    with pytest.raises(ValueError, match="opposite ones"):
        MarginTest(0.3, inside=once, outside=once)


def test_a_boundary_built_from_something_that_is_not_a_p_value_is_refused() -> None:
    with pytest.raises(TypeError, match="is not one"):
        MarginTest(
            0.3,
            inside=one_sided(19.0, "greater"),
            outside=cast("PValue", 0.5),
        )


def test_a_result_whose_boundaries_miss_the_margin_is_refused() -> None:
    boundary = MarginTest(
        0.3, inside=one_sided(19.0, "greater"), outside=one_sided(19.0, "less")
    )

    with pytest.raises(ValueError, match="was tested at the boundaries"):
        EquivalenceResult(0.1, margin=0.3, alpha=0.05, lower=boundary, upper=boundary)


def test_a_result_built_from_something_that_is_not_a_boundary_is_refused() -> None:
    boundary = MarginTest(
        -0.3, inside=one_sided(19.0, "greater"), outside=one_sided(19.0, "less")
    )

    with pytest.raises(TypeError, match="test is not one"):
        EquivalenceResult(
            0.1,
            margin=0.3,
            alpha=0.05,
            lower=boundary,
            upper=cast("MarginTest", "upper"),
        )


def test_the_declared_test_reports_what_it_was_built_with() -> None:
    test = Equivalence(margin=0.25, replicates=99, seed=6, level=0.01)

    assert (test.margin, test.replicates, test.seed, test.level) == (
        0.25,
        99,
        6,
        0.01,
    )
