from math import exp, inf, log, nan
from typing import Final, cast

import pytest
import torch

from lofi.nn.sites import ComponentId, DTypeClass, Role, Site
from lofi.probe._dtypes import (
    LOG_ZERO,
    LogProbabilities,
    Probabilities,
    as_log_probabilities,
    as_probabilities,
    ensure_declared,
    entropy,
    exponentiate,
    logarithm,
    mass,
)
from lofi.probe._values import Observation

ENCODER: Final = ComponentId("toy-encoder")
SCORES: Final = Site(ENCODER, Role.SPLIT_DISTRIBUTION, (0,))
STREAM: Final = Site(ENCODER, Role.RESIDUAL_STREAM, (0,))
COVERED: Final = Site(ENCODER, Role.PARAMETER, ("mask",))
COUNTED: Final = Site(ENCODER, Role.PARAMETER, ("length",))

AXES: Final = ("item", "outcome")
LIVE: Final = (True, False)
WIDTH: Final = 3


def a_mask() -> torch.Tensor:
    return torch.tensor(LIVE)


def a_log_reading(
    first: tuple[float, ...] = (-0.5, -1.0, -2.0),
    second: tuple[float, ...] = (LOG_ZERO, LOG_ZERO, LOG_ZERO),
) -> Observation:
    return Observation(SCORES, torch.tensor([first, second]), axes=AXES, mask=a_mask())


def a_probability_reading(
    first: tuple[float, ...] = (0.5, 0.25, 0.25),
    second: tuple[float, ...] = (0.0, 0.0, 0.0),
) -> Observation:
    return Observation(SCORES, torch.tensor([first, second]), axes=AXES, mask=a_mask())


def test_a_masked_log_reading_at_the_log_of_zero_is_taken_into_the_domain() -> None:
    reading = a_log_reading()
    values = LogProbabilities(reading)

    assert values.reading is reading
    assert torch.equal(values.tensor, reading.tensor)
    assert values.axes == AXES
    assert values.mask is not None
    assert torch.equal(values.mask, a_mask())
    assert torch.equal(values.covering(), reading.covering())


def test_a_log_reading_above_zero_is_refused_with_the_count_of_positions() -> None:
    with pytest.raises(ValueError, match="1 live positions above zero"):
        LogProbabilities(a_log_reading(first=(0.5, -1.0, -2.0)))


def test_a_log_reading_a_hair_above_zero_is_within_its_class() -> None:
    values = LogProbabilities(a_log_reading(first=(1e-9, -1.0, -2.0)))

    assert float(values.tensor[0, 0]) == pytest.approx(1e-9)


def test_a_live_positive_infinity_is_refused_by_the_log_domain() -> None:
    with pytest.raises(ValueError, match="1 live positions above zero or not a number"):
        LogProbabilities(a_log_reading(first=(inf, -1.0, -2.0)))


def test_a_live_position_holding_no_number_is_refused_by_the_log_domain() -> None:
    with pytest.raises(ValueError, match="1 live positions above zero or not a number"):
        LogProbabilities(a_log_reading(first=(nan, -1.0, -2.0)))


def test_a_log_reading_stored_as_whole_numbers_is_refused() -> None:
    reading = Observation(SCORES, torch.zeros(2, WIDTH, dtype=torch.int64), axes=AXES)

    with pytest.raises(ValueError, match=r"torch\.int64 rather than floating-point"):
        LogProbabilities(reading)


def test_a_masked_off_position_above_zero_is_refused_only_for_its_fill() -> None:
    reading = a_log_reading(second=(7.0, 8.0, 9.0))

    with pytest.raises(ValueError, match=r"fills 3 invalid positions with 7\.0"):
        LogProbabilities(reading)


def test_a_log_reading_zero_filled_where_it_is_invalid_is_refused() -> None:
    with pytest.raises(ValueError, match=r"with 0\.0 rather than -inf"):
        LogProbabilities(a_log_reading(second=(0.0, 0.0, 0.0)))


def test_the_refusal_of_a_zero_filled_log_reading_counts_every_position() -> None:
    with pytest.raises(ValueError, match="fills 2 invalid positions"):
        LogProbabilities(a_log_reading(second=(0.0, LOG_ZERO, -3.0)))


def test_a_log_reading_names_the_point_it_was_read_at_when_it_is_refused() -> None:
    with pytest.raises(ValueError, match=r"SPLIT_DISTRIBUTION"):
        LogProbabilities(a_log_reading(second=(0.0, 0.0, 0.0)))


def test_a_domain_is_built_over_a_reading_and_nothing_else() -> None:
    with pytest.raises(TypeError, match="built over an observation"):
        LogProbabilities(cast("Observation", "tensor"))


def test_a_masked_probability_reading_at_zero_is_taken_into_the_domain() -> None:
    reading = a_probability_reading()
    values = Probabilities(reading)

    assert values.reading is reading
    assert torch.equal(values.tensor, reading.tensor)
    assert values.axes == AXES
    assert values.mask is not None
    assert torch.equal(values.mask, a_mask())
    assert torch.equal(values.covering(), reading.covering())


def test_a_probability_reading_above_one_is_refused() -> None:
    with pytest.raises(ValueError, match="outside the unit interval"):
        Probabilities(a_probability_reading(first=(1.5, 0.25, 0.25)))


def test_a_probability_reading_below_zero_is_refused() -> None:
    with pytest.raises(ValueError, match="outside the unit interval"):
        Probabilities(a_probability_reading(first=(-0.5, 0.25, 0.25)))


def test_the_refusal_of_a_probability_reading_counts_every_position() -> None:
    with pytest.raises(ValueError, match="2 live positions outside the unit interval"):
        Probabilities(a_probability_reading(first=(-0.5, 1.5, 0.25)))


@pytest.mark.parametrize("stray", [inf, -inf, nan])
def test_a_live_position_that_is_no_probability_is_refused(stray: float) -> None:
    with pytest.raises(ValueError, match="1 live positions outside the unit interval"):
        Probabilities(a_probability_reading(first=(stray, 0.5, 0.5)))


def test_a_probability_reading_stored_as_truth_values_is_refused() -> None:
    reading = Observation(SCORES, torch.ones(2, WIDTH, dtype=torch.bool), axes=AXES)

    with pytest.raises(ValueError, match=r"torch\.bool rather than floating-point"):
        Probabilities(reading)


def test_a_probability_that_rounds_below_zero_is_within_its_class() -> None:
    values = Probabilities(a_probability_reading(first=(-1e-7, 0.5, 0.5)))

    assert float(values.tensor[0, 0]) == pytest.approx(-1e-7)


def test_a_probability_reading_filled_off_zero_where_it_is_invalid_is_refused() -> None:
    with pytest.raises(ValueError, match=r"with 0\.25 rather than 0\.0"):
        Probabilities(a_probability_reading(second=(0.25, 0.0, 0.0)))


def test_a_point_declaring_log_probabilities_hands_back_the_log_domain() -> None:
    values = as_log_probabilities(a_log_reading(), DTypeClass.LOG_PROBABILITY)

    assert torch.equal(values.tensor, a_log_reading().tensor)


def test_a_point_declaring_probabilities_hands_back_the_probability_domain() -> None:
    values = as_probabilities(a_probability_reading(), DTypeClass.PROBABILITY)

    assert torch.equal(values.tensor, a_probability_reading().tensor)


def test_a_reading_declared_real_is_refused_by_the_log_domain() -> None:
    with pytest.raises(ValueError, match="declares real numbers"):
        as_log_probabilities(a_log_reading(), DTypeClass.REAL)


def test_a_reading_declared_log_probability_is_refused_by_the_probability_domain() -> (
    None
):
    with pytest.raises(ValueError, match="is not probability"):
        as_probabilities(a_probability_reading(), DTypeClass.LOG_PROBABILITY)


def test_a_declaration_that_is_not_a_dtype_class_is_refused() -> None:
    with pytest.raises(TypeError, match="not 'log-probability'"):
        as_log_probabilities(a_log_reading(), cast("DTypeClass", "log-probability"))


def test_a_point_declaring_probabilities_is_refused_by_the_log_domain() -> None:
    with pytest.raises(ValueError, match="declares probability numbers"):
        as_log_probabilities(a_log_reading(), DTypeClass.PROBABILITY)


def test_raising_a_log_reading_leaves_its_invalid_positions_at_zero() -> None:
    raised = exponentiate(LogProbabilities(a_log_reading()))

    assert torch.equal(raised.tensor[1], torch.zeros(WIDTH))
    assert raised.tensor[0].tolist() == pytest.approx([exp(-0.5), exp(-1.0), exp(-2.0)])
    assert raised.reading.site == SCORES
    assert raised.axes == AXES
    assert raised.mask is not None
    assert torch.equal(raised.mask, a_mask())


def test_raising_a_log_reading_that_rounds_above_zero_stays_in_the_unit_interval() -> (
    None
):
    reading = Observation(
        SCORES,
        torch.tensor([[1e-6, -1.0, -2.0], [LOG_ZERO] * WIDTH], dtype=torch.float64),
        axes=AXES,
        mask=a_mask(),
    )
    raised = exponentiate(LogProbabilities(reading))

    assert float(raised.tensor[0, 0]) == 1.0


def test_taking_the_logarithm_puts_the_log_of_zero_at_an_invalid_position() -> None:
    logged = logarithm(Probabilities(a_probability_reading()))

    assert torch.equal(logged.tensor[1], torch.full((WIDTH,), LOG_ZERO))
    assert logged.reading.site == SCORES
    assert logged.mask is not None
    assert torch.equal(logged.mask, a_mask())


def test_a_distribution_survives_a_round_trip_through_the_two_domains() -> None:
    values = Probabilities(a_probability_reading())
    returned = exponentiate(logarithm(values))

    assert returned.tensor == pytest.approx(values.tensor)


def test_the_logarithm_of_a_probability_that_rounds_below_zero_is_the_log_of_zero() -> (
    None
):
    logged = logarithm(Probabilities(a_probability_reading(first=(-1e-7, 0.5, 0.5))))

    assert float(logged.tensor[0, 0]) == LOG_ZERO
    assert exponentiate(logged).tensor[0].tolist() == pytest.approx([0.0, 0.5, 0.5])


def test_the_logarithm_of_a_probability_that_rounds_above_one_is_zero() -> None:
    reading = Observation(
        SCORES,
        torch.tensor([[1.0 + 1e-7, 0.0, 0.0], [0.0] * WIDTH], dtype=torch.float64),
        axes=AXES,
        mask=a_mask(),
    )
    logged = logarithm(Probabilities(reading))

    assert float(logged.tensor[0, 0]) == 0.0


def test_log_probabilities_are_refused_where_probabilities_are_raised() -> None:
    values = LogProbabilities(a_log_reading())

    with pytest.raises(TypeError, match="takes the logarithm of probabilities"):
        logarithm(cast("Probabilities", values))


def test_probabilities_are_refused_where_log_probabilities_are_raised() -> None:
    values = Probabilities(a_probability_reading())

    with pytest.raises(TypeError, match="raises log probabilities"):
        exponentiate(cast("LogProbabilities", values))


def test_the_mass_of_a_distribution_counts_only_the_positions_that_hold_data() -> None:
    measured = mass(Probabilities(a_probability_reading()), over="outcome")

    assert measured.tolist() == pytest.approx([1.0, 0.0])


def test_the_mass_of_an_unmasked_distribution_counts_every_position() -> None:
    reading = Observation(
        SCORES, torch.tensor([[0.5, 0.25, 0.25], [0.2, 0.3, 0.5]]), axes=AXES
    )
    measured = mass(Probabilities(reading), over="outcome")

    assert measured.tolist() == pytest.approx([1.0, 1.0])


def test_a_log_probability_reading_reaching_the_mass_helper_is_refused() -> None:
    values = LogProbabilities(a_log_reading())

    with pytest.raises(TypeError, match="reads probabilities"):
        mass(cast("Probabilities", values), over="outcome")


def test_an_axis_the_reading_does_not_declare_is_refused_by_the_mass_helper() -> None:
    with pytest.raises(KeyError, match="no axis 'split'"):
        mass(Probabilities(a_probability_reading()), over="split")


def test_the_entropy_of_a_uniform_distribution_is_the_log_of_its_width() -> None:
    even = (1.0 / WIDTH, 1.0 / WIDTH, 1.0 / WIDTH)
    measured = entropy(Probabilities(a_probability_reading(first=even)), over="outcome")

    assert float(measured[0]) == pytest.approx(log(WIDTH))


def test_an_outcome_of_probability_zero_contributes_nothing_to_the_entropy() -> None:
    measured = entropy(
        Probabilities(a_probability_reading(first=(1.0, 0.0, 0.0))), over="outcome"
    )

    assert float(measured[0]) == pytest.approx(0.0)


def test_a_masked_row_has_no_entropy_because_it_holds_no_data() -> None:
    measured = entropy(Probabilities(a_probability_reading()), over="outcome")

    assert float(measured[1]) == pytest.approx(0.0)


def test_a_probability_that_rounds_below_zero_is_a_number_in_the_entropy() -> None:
    values = Probabilities(a_probability_reading(first=(-1e-7, 0.5, 0.5)))
    measured = entropy(values, over="outcome")

    assert float(measured[0]) == pytest.approx(log(2))


def test_a_probability_that_rounds_above_one_is_a_number_in_the_mass() -> None:
    reading = Observation(
        SCORES,
        torch.tensor([[1.0 + 1e-7, 0.0, 0.0], [0.0] * WIDTH], dtype=torch.float64),
        axes=AXES,
        mask=a_mask(),
    )
    measured = mass(Probabilities(reading), over="outcome")

    assert float(measured[0]) == 1.0


def test_a_log_probability_reading_reaching_the_entropy_helper_is_refused() -> None:
    values = LogProbabilities(a_log_reading())

    with pytest.raises(TypeError, match="reads probabilities"):
        entropy(cast("Probabilities", values), over="outcome")


def test_a_reading_within_its_declared_class_is_held_to_it_without_complaint() -> None:
    ensure_declared(a_log_reading(), DTypeClass.LOG_PROBABILITY)
    ensure_declared(a_probability_reading(), DTypeClass.PROBABILITY)


def test_a_reading_of_any_numbers_is_held_to_a_point_declaring_real_ones() -> None:
    reading = Observation(
        STREAM, torch.tensor([[7.0, -3.0, 0.5], [0.0, 1.0, 2.0]]), axes=AXES
    )

    ensure_declared(reading, DTypeClass.REAL)


def test_a_point_declaring_truth_values_refuses_numbers() -> None:
    reading = Observation(COVERED, torch.ones(2, WIDTH), axes=AXES)

    with pytest.raises(ValueError, match="marks nothing"):
        ensure_declared(reading, DTypeClass.MASK)


def test_a_point_declaring_truth_values_accepts_them() -> None:
    reading = Observation(COVERED, torch.ones(2, WIDTH, dtype=torch.bool), axes=AXES)

    ensure_declared(reading, DTypeClass.MASK)


def test_a_point_declaring_whole_numbers_refuses_a_fraction() -> None:
    reading = Observation(
        COUNTED, torch.tensor([[1.0, 2.5, 3.0], [4.0, 5.0, 6.0]]), axes=AXES
    )

    with pytest.raises(ValueError, match="1 live positions that are not whole numbers"):
        ensure_declared(reading, DTypeClass.INTEGER)


def test_a_point_declaring_whole_numbers_refuses_an_infinite_count() -> None:
    reading = Observation(
        COUNTED, torch.tensor([[1.0, inf, 3.0], [4.0, 5.0, 6.0]]), axes=AXES
    )

    with pytest.raises(ValueError, match="1 live positions that are not whole numbers"):
        ensure_declared(reading, DTypeClass.INTEGER)


def test_a_point_declaring_whole_numbers_accepts_an_integer_reading() -> None:
    reading = Observation(COUNTED, torch.arange(6).reshape(2, WIDTH), axes=AXES)

    ensure_declared(reading, DTypeClass.INTEGER)


def test_a_point_declaring_whole_numbers_accepts_whole_floating_point_numbers() -> None:
    reading = Observation(COUNTED, torch.arange(6.0).reshape(2, WIDTH), axes=AXES)

    ensure_declared(reading, DTypeClass.INTEGER)


def test_the_log_of_zero_at_a_live_position_is_within_the_log_domain() -> None:
    values = LogProbabilities(a_log_reading(first=(LOG_ZERO, -1.0, -2.0)))

    assert float(values.tensor[0, 0]) == LOG_ZERO


def test_an_invalid_position_is_unconstrained_where_the_class_fixes_no_fill() -> None:
    reading = Observation(
        STREAM,
        torch.tensor([[1.0, 2.0, 3.0], [9.0, 9.0, 9.0]]),
        axes=AXES,
        mask=a_mask(),
    )

    ensure_declared(reading, DTypeClass.REAL)


def test_a_domain_value_reports_the_reading_it_was_built_over() -> None:
    values = LogProbabilities(a_log_reading())
    raised = exponentiate(values)

    assert repr(values) == f"LogProbabilities({values.reading!r})"
    assert repr(raised) == f"Probabilities({raised.reading!r})"
