from typing import Final

import pytest
import torch

from lofi.nn.sites import Access, ComponentId, DTypeClass, Role, Site, SiteSpec
from lofi.probe._diagnose import DiagnosticContext
from lofi.probe._finite import FiniteReadings
from lofi.probe._values import Observation, Trace

PARSER: Final = ComponentId("cky-parser")
CELL: Final = Site(PARSER, Role.CHART_CELL, (0, 2))
OUTSIDE: Final = Site(PARSER, Role.OUTSIDE_SCORE, (0, 2))
ELSEWHERE: Final = Site(PARSER, Role.CHART_CELL, (0, 4))
SPLITS: Final = Site(PARSER, Role.SPLIT_DISTRIBUTION, (0, 2))
GATED: Final = Site(PARSER, Role.GATE)
COVERED: Final = Site(PARSER, Role.PARAMETER, ("mask",))
COUNTED: Final = Site(PARSER, Role.PARAMETER, ("length",))


def a_spec(dtype_class: DTypeClass) -> SiteSpec:
    return SiteSpec(
        ("item", "width"),
        dtype_class,
        differentiable=True,
        access=Access.OBSERVE,
    )


REAL: Final = a_spec(DTypeClass.REAL)
LOGS: Final = a_spec(DTypeClass.LOG_PROBABILITY)
PROBABILITIES: Final = a_spec(DTypeClass.PROBABILITY)
MASKING: Final = a_spec(DTypeClass.MASK)
COUNTS: Final = a_spec(DTypeClass.INTEGER)

CONTEXT: Final = DiagnosticContext(
    {
        CELL: REAL,
        OUTSIDE: REAL,
        SPLITS: LOGS,
        GATED: PROBABILITIES,
        COVERED: MASKING,
        COUNTED: COUNTS,
    }
)


def a_trace(*readings: Observation) -> Trace:
    trace = Trace()

    for reading in readings:
        trace.record(reading)

    trace.close()

    return trace


def test_a_reading_of_numbers_reports_none_missing() -> None:
    reading = Observation(CELL, torch.arange(6.0).reshape(2, 3), axes=REAL.axes)
    result = FiniteReadings().run(a_trace(reading), context=CONTEXT)

    assert result.tag == "finite-readings"
    assert result.value(CELL, "live") == 6.0
    assert result.value(CELL, "nonfinite") == 0.0
    assert result.value(CELL, "nonfinite-share") == 0.0


def test_an_infinity_and_a_nan_are_both_counted() -> None:
    numbers = torch.tensor([[1.0, float("inf"), 3.0], [float("nan"), 5.0, -6.0]])
    result = FiniteReadings().run(
        a_trace(Observation(CELL, numbers, axes=REAL.axes)), context=CONTEXT
    )

    assert result.value(CELL, "live") == 6.0
    assert result.value(CELL, "nonfinite") == 2.0
    assert result.value(CELL, "nonfinite-share") == pytest.approx(1.0 / 3.0)


def test_a_padded_position_is_neither_live_nor_counted_against_the_model() -> None:
    numbers = torch.tensor([[1.0, 2.0, 3.0], [float("nan"), float("nan"), 6.0]])
    mask = torch.tensor([True, False])
    result = FiniteReadings().run(
        a_trace(Observation(CELL, numbers, axes=REAL.axes, mask=mask)),
        context=CONTEXT,
    )

    assert result.value(CELL, "live") == 3.0
    assert result.value(CELL, "nonfinite") == 0.0


def test_a_nan_the_mask_keeps_is_still_counted() -> None:
    numbers = torch.tensor([[1.0, float("nan"), 3.0], [4.0, 5.0, 6.0]])
    mask = torch.tensor([True, False])
    result = FiniteReadings().run(
        a_trace(Observation(CELL, numbers, axes=REAL.axes, mask=mask)),
        context=CONTEXT,
    )

    assert result.value(CELL, "live") == 3.0
    assert result.value(CELL, "nonfinite") == 1.0


def test_every_reading_of_a_trace_is_measured() -> None:
    numbers = torch.zeros(2, 3)
    result = FiniteReadings().run(
        a_trace(
            Observation(CELL, numbers, axes=REAL.axes),
            Observation(OUTSIDE, numbers, axes=REAL.axes),
        ),
        context=CONTEXT,
    )

    assert result.sites == (CELL, OUTSIDE)
    assert len(result.measurements) == 6


def test_a_reading_at_a_point_the_model_never_published_is_refused() -> None:
    reading = Observation(ELSEWHERE, torch.zeros(2, 3), axes=REAL.axes)

    with pytest.raises(KeyError, match="not published"):
        FiniteReadings().run(a_trace(reading), context=CONTEXT)


def test_a_scope_that_read_nothing_measures_nothing() -> None:
    result = FiniteReadings().run(a_trace(), context=CONTEXT)

    assert result.measurements == ()
    assert result.tag == "finite-readings"


def test_a_reading_the_mask_keeps_nothing_of_reports_zero_count_and_no_share() -> None:
    numbers = torch.tensor([[1.0, 2.0, 3.0], [float("nan"), 5.0, 6.0]])
    mask = torch.tensor([False, False])
    result = FiniteReadings().run(
        a_trace(Observation(CELL, numbers, axes=REAL.axes, mask=mask)),
        context=CONTEXT,
    )

    assert result.value(CELL, "live") == 0.0
    assert result.value(CELL, "nonfinite") == 0.0

    with pytest.raises(KeyError, match="nothing named 'nonfinite-share'"):
        result.value(CELL, "nonfinite-share")


def test_the_share_is_the_one_number_reported_on_the_unit_interval() -> None:
    numbers = torch.tensor([[float("inf"), float("nan")], [3.0, 4.0]])
    result = FiniteReadings().run(
        a_trace(Observation(CELL, numbers, axes=REAL.axes)), context=CONTEXT
    )

    assert FiniteReadings.scores == frozenset({"nonfinite-share"})

    for measurement in result.measurements:
        if measurement.name in FiniteReadings.scores:
            assert 0.0 <= measurement.value <= 1.0
        else:
            assert measurement.value >= 1.0


def test_the_diagnostic_requires_no_role_and_names_itself() -> None:
    assert FiniteReadings.requires == frozenset()
    assert FiniteReadings.tag == "finite-readings"
    assert repr(FiniteReadings()) == "FiniteReadings()"


def test_a_log_zero_is_the_declared_absent_weight_and_not_a_missing_number() -> None:
    numbers = torch.tensor([[0.0, -1.0, -float("inf")], [-2.0, -3.0, -float("inf")]])
    result = FiniteReadings().run(
        a_trace(Observation(SPLITS, numbers, axes=LOGS.axes)), context=CONTEXT
    )

    assert result.value(SPLITS, "live") == 6.0
    assert result.value(SPLITS, "log-zero") == 2.0
    assert result.value(SPLITS, "nonfinite") == 0.0
    assert result.value(SPLITS, "nonfinite-share") == 0.0


def test_a_nan_at_a_log_probability_point_is_still_a_missing_number() -> None:
    numbers = torch.tensor([[float("nan"), -1.0, -float("inf")], [-2.0, -3.0, -4.0]])
    result = FiniteReadings().run(
        a_trace(Observation(SPLITS, numbers, axes=LOGS.axes)), context=CONTEXT
    )

    assert result.value(SPLITS, "log-zero") == 1.0
    assert result.value(SPLITS, "nonfinite") == 1.0
    assert result.value(SPLITS, "nonfinite-share") == pytest.approx(1.0 / 6.0)


def test_a_positive_infinity_at_a_log_probability_point_is_a_missing_number() -> None:
    numbers = torch.tensor([[float("inf"), -1.0], [-2.0, -float("inf")]])
    result = FiniteReadings().run(
        a_trace(Observation(SPLITS, numbers, axes=LOGS.axes)), context=CONTEXT
    )

    assert result.value(SPLITS, "log-zero") == 1.0
    assert result.value(SPLITS, "nonfinite") == 1.0


def test_a_log_zero_the_mask_drops_is_counted_neither_way() -> None:
    numbers = torch.tensor([[-1.0, -2.0], [-float("inf"), -float("inf")]])
    result = FiniteReadings().run(
        a_trace(
            Observation(
                SPLITS, numbers, axes=LOGS.axes, mask=torch.tensor([True, False])
            )
        ),
        context=CONTEXT,
    )

    assert result.value(SPLITS, "live") == 2.0
    assert result.value(SPLITS, "log-zero") == 0.0
    assert result.value(SPLITS, "nonfinite") == 0.0


def test_a_negative_infinity_where_no_log_was_declared_is_a_missing_number() -> None:
    numbers = torch.tensor([[-float("inf"), -1.0], [-2.0, -3.0]])
    result = FiniteReadings().run(
        a_trace(Observation(CELL, numbers, axes=REAL.axes)), context=CONTEXT
    )

    assert result.value(CELL, "nonfinite") == 1.0

    with pytest.raises(KeyError, match="log-zero"):
        result.value(CELL, "log-zero")
