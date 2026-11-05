from typing import cast

import pytest

from lofi.stats._values import (
    NULL_FAMILIES,
    TAILS,
    BootstrapResult,
    Decisions,
    Dependence,
    Interval,
    NullFamily,
    NullProvenance,
    PValue,
    Tail,
    _NullTest,
    p_value_against_replicates,
)


def sampled(
    observed: float, replicates: list[float], *, tail: Tail = "greater"
) -> PValue:
    return p_value_against_replicates(
        observed,
        replicates,
        tail=tail,
        null="shuffle",
        family="permutation",
        exact=False,
        seed=7,
    )


def family(values: list[float]) -> tuple[PValue, ...]:
    return tuple(sampled(value, [0.0, 1.0, 2.0, 3.0]) for value in values)


def test_a_sampled_tail_counts_the_observation_in_both_terms() -> None:
    # sampled correction: (3 + 1) / (9 + 1)
    replicates = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0]

    assert sampled(1.0, replicates).value == pytest.approx(0.4)


def test_a_sampled_tail_never_reaches_zero() -> None:
    result = sampled(100.0, [0.0, 0.0, 0.0])

    assert result.value == pytest.approx(0.25)
    assert result.value > 0.0


def test_an_exact_tail_is_the_plain_proportion() -> None:
    # exact tests do not add the observation
    result = p_value_against_replicates(
        2.0,
        [0.0, 1.0, 2.0, 3.0],
        tail="greater",
        null="sign flips",
        family="permutation",
        exact=True,
        seed=None,
    )

    assert result.value == pytest.approx(0.5)
    assert result.provenance.replicates == 4
    assert result.provenance.exact is True


def test_an_enumeration_without_the_observation_is_refused() -> None:
    with pytest.raises(ValueError, match="does not contain the observed"):
        p_value_against_replicates(
            9.0,
            [0.0, 1.0],
            tail="greater",
            null="sign flips",
            family="permutation",
            exact=True,
            seed=None,
        )


def test_the_lower_tail_counts_the_other_side() -> None:
    assert sampled(1.0, [0.0, 2.0, 2.0, 2.0], tail="less").value == pytest.approx(0.4)


def test_the_two_sided_tail_doubles_the_smaller_side() -> None:
    replicates = [0.0] * 9
    upper = sampled(1.0, replicates).value
    both = sampled(1.0, replicates, tail="two-sided").value

    assert upper == pytest.approx(0.1)
    assert both == pytest.approx(0.2)


def test_the_two_sided_tail_stops_at_one() -> None:
    assert sampled(1.0, [1.0, 1.0], tail="two-sided").value == 1.0


def test_the_replicate_count_is_read_off_the_reference_distribution() -> None:
    assert sampled(1.0, [0.0] * 31).provenance.replicates == 31


def test_a_null_that_drew_nothing_is_refused() -> None:
    with pytest.raises(ValueError, match="no replicates"):
        sampled(1.0, [])


def test_a_replicate_that_is_not_finite_is_refused() -> None:
    with pytest.raises(ValueError, match="not finite"):
        sampled(1.0, [0.0, float("nan")])


def test_a_p_value_has_no_constructor_of_its_own() -> None:
    provenance = NullProvenance(
        "shuffle", "permutation", 99, tail="greater", exact=False, seed=1
    )

    with pytest.raises(TypeError, match="no other constructor"):
        PValue(0.05, provenance, _NullTest())


def test_two_p_values_are_equal_when_null_and_probability_agree() -> None:
    one = sampled(1.0, [0.0, 2.0])
    same = sampled(1.0, [0.0, 2.0])
    elsewhere = p_value_against_replicates(
        1.0,
        [0.0, 2.0],
        tail="greater",
        null="bootstrap",
        family="bootstrap-maxima",
        exact=False,
        seed=7,
    )

    assert one == same
    assert hash(one) == hash(same)
    assert one.value == elsewhere.value
    assert one != elsewhere


def test_a_p_value_carries_the_provenance_of_the_test_that_made_it() -> None:
    result = sampled(1.0, [0.0, 2.0])

    assert result.provenance.null == "shuffle"
    assert result.provenance.family == "permutation"
    assert result.provenance.tail == "greater"
    assert result.provenance.seed == 7


def test_a_sampled_null_without_a_seed_is_refused() -> None:
    with pytest.raises(ValueError, match="no seed"):
        NullProvenance(
            "shuffle", "permutation", 99, tail="greater", exact=False, seed=None
        )


def test_an_enumerated_null_with_a_seed_is_refused() -> None:
    with pytest.raises(ValueError, match="reports a seed"):
        NullProvenance(
            "sign flips", "permutation", 8, tail="greater", exact=True, seed=3
        )


def test_a_family_outside_the_declared_ones_is_refused() -> None:
    with pytest.raises(ValueError, match="permutation, bootstrap-maxima, matched"):
        NullProvenance(
            "shuffle",
            cast("NullFamily", "resampling"),
            99,
            tail="greater",
            exact=False,
            seed=7,
        )


def test_a_tail_outside_the_declared_ones_is_refused() -> None:
    with pytest.raises(ValueError, match="greater, less, two-sided"):
        NullProvenance(
            "shuffle",
            "permutation",
            99,
            tail=cast("Tail", "two_sided"),
            exact=False,
            seed=7,
        )


def test_a_test_run_under_an_undeclared_tail_never_reaches_a_p_value() -> None:
    with pytest.raises(ValueError, match="a null's tail is 'upper'"):
        p_value_against_replicates(
            2.0,
            [0.0, 1.0],
            tail=cast("Tail", "upper"),
            null="shuffle",
            family="permutation",
            exact=False,
            seed=7,
        )


def test_every_declared_family_and_tail_builds_a_provenance() -> None:
    for name in NULL_FAMILIES:
        for direction in TAILS:
            provenance = NullProvenance(
                "shuffle", name, 99, tail=direction, exact=False, seed=7
            )

            assert (provenance.family, provenance.tail) == (name, direction)


def test_matching_criteria_are_copied_and_not_live() -> None:
    criteria: dict[str, str | int] = {"size": 4, "component": "encoder"}
    provenance = NullProvenance(
        "matched",
        "matched",
        99,
        tail="greater",
        exact=False,
        seed=1,
        matching=criteria,
    )
    criteria["size"] = 5

    assert provenance.matching == {"size": 4, "component": "encoder"}


def test_a_criterion_level_that_is_neither_a_name_nor_a_count_is_refused() -> None:
    with pytest.raises(TypeError, match="neither a name nor a count"):
        NullProvenance(
            "matched",
            "matched",
            99,
            tail="greater",
            exact=False,
            seed=1,
            matching=cast("dict[str, str | int]", {"size": 4.5}),
        )


def test_the_rejection_mask_is_derived_from_the_adjusted_values() -> None:
    raw = family([3.0, 0.5])
    decisions = Decisions("bonferroni", 0.05, Dependence.PRDS, raw, (0.04, 0.9))

    assert decisions.reject == (True, False)
    assert decisions.n_rejected == 1
    assert decisions.raw == raw
    assert decisions.adjusted == (0.04, 0.9)


def test_adjusted_p_values_do_not_stand_in_for_the_mask() -> None:
    raw = family([3.0, 0.5, 0.1])
    decisions = Decisions("fdr", 0.01, Dependence.PRDS, raw, (0.2, 0.6, 0.9))

    assert decisions.reject == (False, False, False)
    assert decisions.n_rejected == 0


def test_a_float_cannot_be_corrected_as_though_it_were_a_p_value() -> None:
    importances = cast("tuple[PValue, ...]", (0.91, 0.42))

    with pytest.raises(TypeError, match="not a p-value"):
        Decisions("holm", 0.05, Dependence.ARBITRARY, importances, (0.91, 0.42))


def test_a_correction_returning_the_wrong_number_of_values_is_refused() -> None:
    with pytest.raises(ValueError, match="adjusted values"):
        Decisions("holm", 0.05, Dependence.ARBITRARY, family([3.0, 0.5]), (0.1,))


def test_a_level_outside_the_unit_interval_is_refused() -> None:
    with pytest.raises(ValueError, match="not a proportion"):
        Decisions("holm", 1.0, Dependence.ARBITRARY, family([3.0]), (0.1,))


def test_an_adjusted_value_outside_the_unit_interval_is_refused() -> None:
    with pytest.raises(ValueError, match="outside the unit interval"):
        Decisions("holm", 0.05, Dependence.ARBITRARY, family([3.0]), (1.5,))


def test_a_dependence_regime_is_a_declared_value_and_not_a_string() -> None:
    with pytest.raises(TypeError, match="dependence regime is not one"):
        Decisions("holm", 0.05, cast("Dependence", "arbitrary"), family([3.0]), (0.1,))


def test_a_regime_is_recovered_from_the_name_a_configuration_spells() -> None:
    assert Dependence("prds") is Dependence.PRDS


def test_decisions_are_equal_on_what_they_decided() -> None:
    raw = family([3.0, 0.5])
    one = Decisions("holm", 0.05, Dependence.PRDS, raw, (0.04, 0.9))
    same = Decisions("holm", 0.05, Dependence.PRDS, raw, (0.04, 0.9))
    stricter = Decisions("holm", 0.01, Dependence.PRDS, raw, (0.04, 0.9))

    assert one == same
    assert hash(one) == hash(same)
    assert one != stricter
    assert one.reject != stricter.reject


def interval(low: float = 0.1, high: float = 0.9, level: float = 0.95) -> Interval:
    return Interval(low, high, level)


def bootstrapped(
    observed: float = 0.5, *, unit: str = "sentence", units: int = 12
) -> BootstrapResult:
    return BootstrapResult(
        observed,
        unit,
        units=units,
        replicates=999,
        seed=3,
        percentile=interval(),
        bca=interval(0.15, 0.95),
    )


def test_an_interval_carries_its_level_and_reports_its_width() -> None:
    around = interval(1.0, 3.0, 0.9)

    assert around.low == 1.0
    assert around.high == 3.0
    assert around.level == 0.9
    assert around.width == pytest.approx(2.0)


def test_an_interval_covers_its_own_bounds_and_nothing_outside_them() -> None:
    around = interval(1.0, 3.0)

    assert around.covers(1.0)
    assert around.covers(2.0)
    assert around.covers(3.0)
    assert not around.covers(3.5)


def test_an_interval_that_ends_before_it_starts_is_refused() -> None:
    with pytest.raises(ValueError, match="ends below where it starts"):
        Interval(3.0, 1.0, 0.95)


def test_an_interval_whose_level_is_not_a_proportion_is_refused() -> None:
    with pytest.raises(ValueError, match="not a proportion"):
        Interval(1.0, 3.0, 1.0)


def test_a_bound_that_is_not_finite_is_refused() -> None:
    with pytest.raises(ValueError, match="not finite"):
        Interval(1.0, float("inf"), 0.95)


def test_two_intervals_are_equal_on_their_bounds_and_their_level() -> None:
    one = interval(1.0, 3.0)
    same = interval(1.0, 3.0)
    looser = interval(1.0, 3.0, 0.9)

    assert one == same
    assert hash(one) == hash(same)
    assert one != looser


def test_a_bootstrap_result_names_the_unit_it_resampled() -> None:
    result = bootstrapped(unit="participant", units=9)

    assert result.unit == "participant"
    assert result.units == 9
    assert result.replicates == 999
    assert result.seed == 3
    assert result.level == 0.95


def test_a_bootstrap_result_carries_both_intervals_apart() -> None:
    result = bootstrapped()

    assert result.percentile == Interval(0.1, 0.9, 0.95)
    assert result.bca == Interval(0.15, 0.95, 0.95)
    assert result.percentile != result.bca


def test_a_bootstrap_over_no_units_is_refused() -> None:
    with pytest.raises(ValueError, match="over no 'sentence' at all"):
        bootstrapped(units=0)


def test_two_intervals_claiming_different_coverage_are_refused() -> None:
    with pytest.raises(ValueError, match="different coverage"):
        BootstrapResult(
            0.5,
            "sentence",
            units=12,
            replicates=999,
            seed=3,
            percentile=interval(level=0.95),
            bca=interval(level=0.9),
        )


def test_an_interval_that_is_not_one_is_refused() -> None:
    with pytest.raises(TypeError, match="interval is not one"):
        BootstrapResult(
            0.5,
            "sentence",
            units=12,
            replicates=999,
            seed=3,
            percentile=cast("Interval", (0.1, 0.9)),
            bca=interval(),
        )


def test_two_bootstrap_results_are_equal_on_what_they_estimated() -> None:
    one = bootstrapped()
    same = bootstrapped()
    elsewhere = bootstrapped(0.6)

    assert one == same
    assert hash(one) == hash(same)
    assert one != elsewhere


def test_null_provenance_names_every_field_as_json() -> None:
    provenance = NullProvenance(
        "label-permutation",
        "permutation",
        3,
        tail="greater",
        exact=False,
        seed=7,
        matching={"length": 4},
    )

    assert provenance.to_json() == {
        "name": "label-permutation",
        "family": "permutation",
        "replicates": 3,
        "tail": "greater",
        "exact": False,
        "seed": 7,
        "matching": {"length": 4},
    }
