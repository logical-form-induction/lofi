import math

import didactic.api as dx
import pytest

from lofi.judgment import (
    ConstraintCost,
    Measurement,
    ScaleProfile,
    constraint_cost,
    scale_profile,
)

SCALES = (0.5, 1.0, 2.0)


@pytest.mark.parametrize(
    ("log_posterior", "minimum_tail_drop", "peak", "drops", "ridge"),
    [
        ((-1.0, -1.0, -1.0), 1.0, -1.0, (0.0, 0.0), True),
        ((-5.0, 0.0, -4.0), 4.0, 0.0, (5.0, 4.0), False),
        ((-4.0, 0.0, -0.5), 1.0, 0.0, (4.0, 0.5), True),
        ((-2.0, 0.0, -2.0), 2.0, 0.0, (2.0, 2.0), False),
    ],
)
def test_a_scale_profile_derives_its_peak_tail_drops_and_ridge(
    log_posterior: tuple[float, ...],
    minimum_tail_drop: float,
    peak: float,
    drops: tuple[float, float],
    *,
    ridge: bool,
) -> None:
    profile = scale_profile(
        SCALES,
        log_posterior,
        minimum_tail_drop=minimum_tail_drop,
    )

    assert profile.peak == peak
    assert profile.peak_scale == SCALES[log_posterior.index(peak)]
    assert (profile.lower_tail_drop, profile.upper_tail_drop) == drops
    assert profile.ridge is ridge


@pytest.mark.parametrize(
    ("scales", "log_posterior", "minimum_tail_drop", "message"),
    [
        ((0.5, 2.0), (-1.0, -1.0), 1.0, "at least three"),
        (SCALES, (-1.0, -1.0), 1.0, "3 scale factors and 2"),
        ((0.0, 1.0, 2.0), (-1.0, 0.0, -1.0), 1.0, "not positive"),
        ((0.5, math.inf, 2.0), (-1.0, 0.0, -1.0), 1.0, "not finite"),
        ((0.5, 0.5, 2.0), (-1.0, 0.0, -1.0), 1.0, "strictly increasing"),
        ((0.5, 2.0, 1.0), (-1.0, 0.0, -1.0), 1.0, "strictly increasing"),
        ((1.0, 2.0, 3.0), (-1.0, 0.0, -1.0), 1.0, "do not bracket"),
        (SCALES, (-1.0, math.nan, -1.0), 1.0, "not finite"),
        (SCALES, (-1.0, 0.0, -1.0), 0.0, "not positive"),
        (SCALES, (-1.0, 0.0, -1.0), math.inf, "not finite"),
    ],
)
def test_a_malformed_scale_profile_is_refused(
    scales: tuple[float, ...],
    log_posterior: tuple[float, ...],
    minimum_tail_drop: float,
    message: str,
) -> None:
    with pytest.raises(dx.ValidationError, match=message):
        scale_profile(
            scales,
            log_posterior,
            minimum_tail_drop=minimum_tail_drop,
        )


@pytest.mark.parametrize(
    ("log_posterior", "tail"),
    [
        ((-1e308, 1e308, 0.0), "lower"),
        ((0.0, 1e308, -1e308), "upper"),
    ],
)
def test_a_scale_profile_refuses_an_overflowed_endpoint_drop(
    log_posterior: tuple[float, ...],
    tail: str,
) -> None:
    with pytest.raises(dx.ValidationError, match=rf"{tail} tail drop.*not finite"):
        scale_profile(SCALES, log_posterior, minimum_tail_drop=1.0)


@pytest.mark.parametrize(
    ("constrained_value", "unconstrained_value", "expected"),
    [(-3.0, -2.5, 0.5), (-2.5, -3.0, -0.5)],
)
def test_a_constraint_cost_derives_unconstrained_minus_constrained(
    constrained_value: float,
    unconstrained_value: float,
    expected: float,
) -> None:
    cost = constraint_cost(
        Measurement(value=constrained_value, n=20, unit="response"),
        Measurement(value=unconstrained_value, n=20, unit="response"),
    )

    assert cost.value == expected
    assert cost.as_measurement() == Measurement(
        value=expected,
        n=20,
        unit="response",
    )


@pytest.mark.parametrize(
    ("constrained", "unconstrained", "message"),
    [
        (
            Measurement(value=None, n=20, unit="response"),
            Measurement(value=-2.0, n=20, unit="response"),
            "two defined",
        ),
        (
            Measurement(value=-3.0, n=20, unit="response"),
            Measurement(value=None, n=20, unit="response"),
            "two defined",
        ),
        (
            Measurement(value=-3.0, n=20, unit="response"),
            Measurement(value=-2.0, n=19, unit="response"),
            "same units",
        ),
        (
            Measurement(value=-3.0, n=20, unit="response"),
            Measurement(value=-2.0, n=20, unit="item"),
            "same units",
        ),
    ],
)
def test_a_constraint_cost_requires_defined_matched_measurements(
    constrained: Measurement,
    unconstrained: Measurement,
    message: str,
) -> None:
    with pytest.raises(dx.ValidationError, match=message):
        constraint_cost(constrained, unconstrained)


@pytest.mark.parametrize(
    ("constrained_value", "unconstrained_value"),
    [(-1e308, 1e308), (1e308, -1e308)],
)
def test_a_constraint_cost_refuses_an_overflowed_signed_difference(
    constrained_value: float,
    unconstrained_value: float,
) -> None:
    with pytest.raises(dx.ValidationError, match=r"signed difference.*not finite"):
        constraint_cost(
            Measurement(value=constrained_value, n=20, unit="response"),
            Measurement(value=unconstrained_value, n=20, unit="response"),
        )


def test_identification_results_round_trip_with_only_computed_outputs() -> None:
    profile = scale_profile(SCALES, (-3.0, 0.0, -2.0), minimum_tail_drop=2.0)
    cost = constraint_cost(
        Measurement(value=-3.0, n=20, unit="response"),
        Measurement(value=-2.5, n=20, unit="response"),
    )

    restored_profile = ScaleProfile.model_validate_json(profile.model_dump_json())
    restored_cost = ConstraintCost.model_validate_json(cost.model_dump_json())

    assert restored_profile == profile
    assert restored_profile.peak == 0.0
    assert restored_profile.lower_tail_drop == 3.0
    assert restored_profile.upper_tail_drop == 2.0
    assert restored_profile.ridge is False
    assert restored_cost == cost
    assert restored_cost.value == 0.5
