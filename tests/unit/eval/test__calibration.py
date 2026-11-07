import json
from array import array
from math import fsum
from typing import TYPE_CHECKING, cast

import pytest

from lofi._types import ItemId
from lofi.eval._calibration import (
    CalibrationBin,
    Forecast,
    PitHistogram,
    ReliabilityCurve,
    brier_score,
    expected_calibration_error,
    pit_histogram,
    reliability_curve,
)
from lofi.eval._values import MetricResult, Resampling

if TYPE_CHECKING:
    from collections.abc import Sequence


def _forecast(name: str, probabilities: Sequence[float], outcome: int) -> Forecast:
    return Forecast(ItemId(name), probabilities, outcome)


def _binary(name: str, probability: float, *, occurred: bool) -> Forecast:
    return _forecast(name, (1.0 - probability, probability), int(occurred))


def _sharpened(probability: float) -> float:
    """Push a probability away from one half, keeping its order."""
    cubed = probability**3
    complement = (1.0 - probability) ** 3

    return cubed / (cubed + complement)


def _calibrated_binary() -> tuple[Forecast, ...]:
    """Return forecasts at a hundred probabilities, each with its outcome rate."""
    forecasts: list[Forecast] = []

    for level in range(100):
        forecasts.extend(
            _binary(
                f"item-{level:02d}-{repeat:02d}", level / 100, occurred=repeat < level
            )
            for repeat in range(100)
        )

    return tuple(forecasts)


def _overconfident_binary() -> tuple[Forecast, ...]:
    """Return the calibrated forecasts with their probabilities sharpened."""
    return tuple(
        _binary(
            forecast.item_id,
            _sharpened(forecast.probabilities[1]),
            occurred=forecast.outcome == 1,
        )
        for forecast in _calibrated_binary()
    )


WORKED = (
    _forecast("a", (0.7, 0.2, 0.1), 0),
    _forecast("b", (0.2, 0.5, 0.3), 2),
    _forecast("c", (0.1, 0.1, 0.8), 2),
    _forecast("d", (0.6, 0.3, 0.1), 1),
)
"""Four three-way forecasts whose calibration figures are worked by hand."""


# ---------------------------------------------------------------------------
# forecasts
# ---------------------------------------------------------------------------


def test_forecast_exposes_its_fields_and_derived_categories() -> None:
    forecast = _forecast("x", (0.2, 0.5, 0.3), 2)

    assert forecast.item_id == "x"
    assert forecast.probabilities == (0.2, 0.5, 0.3)
    assert forecast.outcome == 2
    assert forecast.categories == 3
    assert forecast.predicted == 1
    assert forecast.confidence == 0.5
    assert forecast.hit is False
    assert _forecast("y", (0.2, 0.5, 0.3), 1).hit is True


def test_forecast_breaks_a_confidence_tie_toward_the_lowest_category() -> None:
    forecast = _forecast("x", (0.4, 0.2, 0.4), 2)

    assert forecast.predicted == 0
    assert forecast.confidence == 0.4
    assert forecast.hit is False


def test_forecast_cumulative_mass_runs_from_nothing_to_everything() -> None:
    forecast = _forecast("x", (0.2, 0.3, 0.5), 1)

    assert forecast.cumulative(-1) == 0.0
    assert forecast.cumulative(0) == pytest.approx(0.2)
    assert forecast.cumulative(1) == pytest.approx(0.5)
    assert forecast.cumulative(2) == pytest.approx(1.0)


def test_forecast_accepts_single_precision_rounding_in_its_total() -> None:
    thirds = tuple(array("f", (1 / 3, 1 / 3, 1 / 3)))
    forecast = _forecast("x", thirds, 0)

    assert fsum(thirds) != 1.0
    assert forecast.probabilities == thirds
    assert _forecast("y", (0.3 + 9e-7, 0.7), 0).categories == 2

    with pytest.raises(ValueError, match="sum to"):
        _forecast("z", (0.3 + 2e-6, 0.7), 0)


@pytest.mark.parametrize(
    ("probabilities", "outcome", "error"),
    [
        ((1.0,), 0, ValueError),
        ((0.5, 0.6), 0, ValueError),
        ((0.5, 0.4), 0, ValueError),
        ((1.2, -0.2), 0, ValueError),
        ((0.5, 0.5), 2, ValueError),
        ((0.5, 0.5), -1, ValueError),
        ((0.5, "0.5"), 0, TypeError),
        ((0.5, 0.5), 0.0, TypeError),
        ((0.5, 0.5), True, TypeError),
    ],
)
def test_forecast_rejects_a_malformed_distribution_or_outcome(
    probabilities: tuple[float, ...], outcome: int, error: type[Exception]
) -> None:
    with pytest.raises(error):
        _forecast("x", probabilities, outcome)


def test_forecast_rejects_an_empty_item_id() -> None:
    with pytest.raises(ValueError, match="empty"):
        _forecast("", (0.5, 0.5), 0)


def test_forecast_equality_covers_every_field_and_the_repr_rebuilds_it() -> None:
    forecast = _forecast("x", (0.2, 0.8), 1)

    assert forecast == _forecast("x", (0.2, 0.8), 1)
    assert hash(forecast) == hash(_forecast("x", (0.2, 0.8), 1))
    assert forecast != _forecast("y", (0.2, 0.8), 1)
    assert forecast != _forecast("x", (0.8, 0.2), 1)
    assert forecast != _forecast("x", (0.2, 0.8), 0)
    assert forecast != (ItemId("x"), (0.2, 0.8), 1)
    assert repr(forecast) == "Forecast('x', [0.2, 0.8], 1)"


# ---------------------------------------------------------------------------
# reliability curve
# ---------------------------------------------------------------------------


def test_curve_bins_are_equal_mass_where_equal_width_bins_would_not_be() -> None:
    forecasts = tuple(
        _binary(f"item-{index:04d}", 0.9 + index / 10_030, occurred=index % 2 == 0)
        for index in range(1_003)
    )

    curve = reliability_curve(forecasts, category=1)
    counts = [entry.count for entry in curve.bins]

    assert len(counts) == 10
    assert sum(counts) == 1_003
    assert sorted(counts) == [100] * 7 + [101] * 3
    assert all(entry.low <= entry.high for entry in curve.bins)
    assert all(
        earlier.high <= later.low
        for earlier, later in zip(curve.bins, curve.bins[1:], strict=False)
    )
    assert all(entry.high < 1.0 for entry in curve.bins)


def test_curve_worked_by_hand_over_the_most_probable_category() -> None:
    curve = reliability_curve(WORKED, bins=2)

    assert curve.category is None
    assert curve.n == 4
    assert curve.bins == (
        (0.5, 0.6, 2, pytest.approx(0.55), 0.0),
        (0.7, 0.8, 2, pytest.approx(0.75), 1.0),
    )
    assert curve.expected_calibration_error == pytest.approx(0.4)


def test_curve_worked_by_hand_over_a_named_category() -> None:
    curve = reliability_curve(WORKED, category=2, bins=2)

    assert curve.category == 2
    assert curve.bins == (
        (0.1, 0.1, 2, pytest.approx(0.1), 0.0),
        (0.3, 0.8, 2, pytest.approx(0.55), 1.0),
    )
    assert curve.expected_calibration_error == pytest.approx(0.275)


def test_curve_weights_each_bin_gap_by_its_count() -> None:
    forecasts = (
        _binary("a", 0.2, occurred=False),
        _binary("b", 0.6, occurred=True),
        _binary("c", 0.9, occurred=True),
    )

    curve = reliability_curve(forecasts, category=1, bins=2)

    assert [entry.count for entry in curve.bins] == [1, 2]
    assert [entry.gap for entry in curve.bins] == [
        pytest.approx(0.2),
        pytest.approx(0.25),
    ]
    assert curve.expected_calibration_error == pytest.approx(0.7 / 3)


def test_curve_breaks_probability_ties_by_item_id_whatever_the_input_order() -> None:
    forecasts = (
        _binary("b", 0.5, occurred=True),
        _binary("a", 0.5, occurred=False),
        _binary("c", 0.5, occurred=True),
        _binary("d", 0.9, occurred=True),
    )

    forward = reliability_curve(forecasts, category=1, bins=2)
    backward = reliability_curve(tuple(reversed(forecasts)), category=1, bins=2)

    assert forward == backward
    assert [entry.mean_outcome for entry in forward.bins] == [0.5, 1.0]


def test_curve_over_no_forecasts_has_no_bins_and_no_error() -> None:
    curve = reliability_curve(())

    assert curve.bins == ()
    assert curve.n == 0
    assert curve.expected_calibration_error is None
    assert expected_calibration_error(()) == MetricResult(None, 0, Resampling("item"))


def test_curve_refuses_fewer_forecasts_than_bins() -> None:
    with pytest.raises(ValueError, match="cannot fill"):
        reliability_curve(WORKED, bins=5)


@pytest.mark.parametrize(
    ("bins", "error"), [(0, ValueError), (-1, ValueError), (True, TypeError)]
)
def test_curve_refuses_a_malformed_bin_count(bins: int, error: type[Exception]) -> None:
    with pytest.raises(error):
        reliability_curve(WORKED, bins=bins)


def test_curve_refuses_a_category_outside_the_forecasts() -> None:
    with pytest.raises(ValueError, match="outside"):
        reliability_curve(WORKED, category=3, bins=2)

    with pytest.raises(ValueError, match="negative"):
        reliability_curve(WORKED, category=-1, bins=2)


def test_curve_refuses_forecasts_over_different_category_counts() -> None:
    mixed = (*WORKED, _forecast("e", (0.5, 0.5), 0))

    with pytest.raises(ValueError, match="ranges over 2"):
        reliability_curve(mixed, bins=2)


def test_curve_refuses_a_value_that_is_not_a_forecast() -> None:
    with pytest.raises(TypeError, match="not a Forecast"):
        reliability_curve((*WORKED, cast("Forecast", (0.5, 0.5))), bins=2)


def test_curve_value_checks_its_bins() -> None:
    full = CalibrationBin(0.1, 0.2, 3, 0.15, 0.3)

    with pytest.raises(TypeError, match="not one"):
        ReliabilityCurve((full, cast("CalibrationBin", (0.3, 0.4, 3, 0.35, 0.3))))

    with pytest.raises(ValueError, match="empty"):
        ReliabilityCurve((full, CalibrationBin(0.3, 0.4, 0, 0.35, 0.3)))

    with pytest.raises(ValueError, match="not equal-mass"):
        ReliabilityCurve((full, CalibrationBin(0.3, 0.4, 5, 0.35, 0.3)))

    with pytest.raises(ValueError, match="negative"):
        ReliabilityCurve((full,), category=-1)


def test_curve_equality_covers_bins_and_category_and_serializes() -> None:
    curve = reliability_curve(WORKED, category=2, bins=2)

    assert curve == reliability_curve(WORKED, category=2, bins=2)
    assert hash(curve) == hash(reliability_curve(WORKED, category=2, bins=2))
    assert curve != reliability_curve(WORKED, category=1, bins=2)
    assert curve != reliability_curve(WORKED, category=2, bins=1)
    assert curve != curve.bins
    assert repr(curve).startswith("ReliabilityCurve([CalibrationBin(low=0.1, ")
    assert repr(curve).endswith("], 2)")

    spec = curve.to_spec()

    assert json.loads(json.dumps(spec)) == spec
    assert spec["category"] == 2
    assert spec["n"] == 4
    assert spec["expected_calibration_error"] == pytest.approx(0.275)
    assert spec["bins"] == [entry.to_spec() for entry in curve.bins]
    assert curve.bins[1].gap == pytest.approx(0.45)


# ---------------------------------------------------------------------------
# expected calibration error
# ---------------------------------------------------------------------------


def test_calibrated_predictor_scores_near_zero_and_sharpened_one_scores_high() -> None:
    calibrated = expected_calibration_error(_calibrated_binary(), category=1)
    overconfident = expected_calibration_error(_overconfident_binary(), category=1)

    assert calibrated.n == 10_000
    assert calibrated.resampling == Resampling("item")
    assert calibrated.value is not None
    assert calibrated.value < 0.01
    assert overconfident.n == 10_000
    assert overconfident.value is not None
    assert overconfident.value > 0.1


def test_calibration_error_uses_ten_equal_mass_bins_by_default() -> None:
    curve = reliability_curve(_overconfident_binary(), category=1)

    assert len(curve.bins) == 10
    assert {entry.count for entry in curve.bins} == {1_000}
    assert expected_calibration_error(
        _overconfident_binary(), category=1
    ).value == pytest.approx(curve.expected_calibration_error)


def test_calibration_error_over_the_most_probable_category_matches_the_table() -> None:
    result = expected_calibration_error(WORKED, bins=2)

    assert result.value == pytest.approx(0.4)
    assert result.n == 4
    assert result.resampling == Resampling("item")
    assert result.ci is None
    assert expected_calibration_error(WORKED, category=2, bins=2).value == (
        pytest.approx(0.275)
    )


# ---------------------------------------------------------------------------
# brier score
# ---------------------------------------------------------------------------


def test_brier_score_worked_by_hand() -> None:
    summed = brier_score(WORKED)
    single = brier_score(WORKED, category=2)

    assert summed.value == pytest.approx(0.46)
    assert single.value == pytest.approx(0.1375)
    assert (summed.n, single.n) == (4, 4)
    assert summed.resampling == single.resampling == Resampling("item")


def test_brier_score_over_two_categories_sums_to_twice_either_one() -> None:
    forecasts = _overconfident_binary()[:250]
    summed = brier_score(forecasts).value
    positive = brier_score(forecasts, category=1).value
    negative = brier_score(forecasts, category=0).value

    assert summed is not None
    assert positive is not None
    assert positive == pytest.approx(negative)
    assert summed == pytest.approx(2.0 * positive)
    assert positive == pytest.approx(
        fsum(
            (forecast.probabilities[1] - float(forecast.outcome == 1)) ** 2
            for forecast in forecasts
        )
        / len(forecasts)
    )


def test_brier_score_over_no_forecasts_has_no_value() -> None:
    assert brier_score(()) == MetricResult(None, 0, Resampling("item"))
    assert brier_score((), category=1) == MetricResult(None, 0, Resampling("item"))


def test_brier_score_refuses_a_category_outside_the_forecasts() -> None:
    with pytest.raises(ValueError, match="outside"):
        brier_score(WORKED, category=3)


# ---------------------------------------------------------------------------
# probability integral transform
# ---------------------------------------------------------------------------


def _ordinal(
    probabilities: Sequence[float], counts: Sequence[int]
) -> tuple[tuple[Forecast, ...], tuple[float, ...]]:
    """Return one fixed forecast per outcome with stratified draws."""
    forecasts: list[Forecast] = []
    draws: list[float] = []

    for category, count in enumerate(counts):
        for position in range(count):
            forecasts.append(
                _forecast(f"item-{category}-{position:03d}", probabilities, category)
            )
            draws.append((position + 0.5) / count)

    return tuple(forecasts), tuple(draws)


def test_transform_worked_by_hand() -> None:
    forecasts = (
        _forecast("a", (0.2, 0.3, 0.5), 1),
        _forecast("b", (0.2, 0.3, 0.5), 0),
        _forecast("c", (0.2, 0.3, 0.5), 2),
        _forecast("d", (0.2, 0.3, 0.5), 2),
    )

    histogram = pit_histogram(forecasts, (0.5, 0.0, 1.0, 0.5), bins=10)

    assert histogram.counts == (1, 0, 0, 1, 0, 0, 0, 1, 0, 1)
    assert histogram.n == 4
    assert histogram.bins == 10
    assert histogram.fractions == (0.25, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.25, 0.0, 0.25)
    assert histogram.max_deviation == pytest.approx(0.15)


def test_transform_of_a_calibrated_ordinal_model_is_uniform() -> None:
    forecasts, draws = _ordinal((0.2, 0.3, 0.5), (200, 300, 500))

    histogram = pit_histogram(forecasts, draws)

    assert histogram.n == 1_000
    assert histogram.counts == (100,) * 10
    assert histogram.max_deviation == 0.0


def test_transform_of_a_miscalibrated_ordinal_model_piles_up() -> None:
    forecasts, draws = _ordinal((0.5, 0.3, 0.2), (200, 300, 500))

    histogram = pit_histogram(forecasts, draws)

    assert histogram.counts == (40, 40, 40, 40, 40, 100, 100, 100, 250, 250)
    assert histogram.max_deviation == pytest.approx(0.15)


def test_transform_over_no_forecasts_counts_nothing() -> None:
    histogram = pit_histogram((), (), bins=4)

    assert histogram.counts == (0, 0, 0, 0)
    assert histogram.n == 0
    assert histogram.fractions is None
    assert histogram.max_deviation is None
    assert histogram.to_spec() == {
        "bins": 4,
        "n": 0,
        "counts": [0, 0, 0, 0],
        "fractions": None,
        "max_deviation": None,
    }


def test_transform_refuses_draws_that_do_not_match_the_forecasts() -> None:
    forecasts = WORKED

    with pytest.raises(ValueError, match="3 draws randomize 4"):
        pit_histogram(forecasts, (0.1, 0.2, 0.3))

    with pytest.raises(ValueError, match="outside the unit interval"):
        pit_histogram(forecasts, (0.1, 0.2, 0.3, 1.5))

    with pytest.raises(TypeError, match="not a number"):
        pit_histogram(forecasts, (0.1, 0.2, 0.3, cast("float", "0.4")))

    with pytest.raises(ValueError, match="not positive"):
        pit_histogram(forecasts, (0.1, 0.2, 0.3, 0.4), bins=0)


def test_transform_histogram_value_checks_its_counts_and_serializes() -> None:
    with pytest.raises(ValueError, match="no bins"):
        PitHistogram(())

    with pytest.raises(ValueError, match="negative"):
        PitHistogram((1, -1))

    with pytest.raises(TypeError, match="not an integer"):
        PitHistogram((1, cast("int", 1.0)))

    histogram = PitHistogram((1, 3))

    assert histogram == PitHistogram((1, 3))
    assert hash(histogram) == hash(PitHistogram((1, 3)))
    assert histogram != PitHistogram((3, 1))
    assert histogram != (1, 3)
    assert repr(histogram) == "PitHistogram([1, 3])"

    spec = histogram.to_spec()

    assert json.loads(json.dumps(spec)) == spec
    assert spec["fractions"] == [0.25, 0.75]
    assert spec["max_deviation"] == 0.25
