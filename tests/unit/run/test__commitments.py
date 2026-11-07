from datetime import date
from typing import TYPE_CHECKING

import pytest

from lofi.run import (
    DecisionState,
    Direction,
    ReportedRow,
    Statistic,
    StopRule,
    assess,
)
from lofi.run._commitments import (
    POINT_STATUSES,
    DecisionPoint,
    MetricBound,
    PointCount,
)

if TYPE_CHECKING:
    from lofi._types import JsonValue

RECOVERY = ReportedRow("held-out", "recovery")

STATUS: dict[str, JsonValue] = {
    "counts": {"completed": 3, "numerical-failure": 1, "infra-failed": 2}
}


def rows(*values: float | None, task: str = "held-out") -> list[dict[str, JsonValue]]:
    return [{"task": task, "metric": "recovery", "value": value} for value in values]


def bound(
    statistic: Statistic, direction: Direction, threshold: float, minimum: int = 1
) -> MetricBound:
    return MetricBound(
        RECOVERY,
        statistic=statistic,
        direction=direction,
        threshold=threshold,
        minimum_rows=minimum,
    )


def test_a_point_count_sums_the_counted_statuses() -> None:
    failures = ("numerical-failure", "infra-failed")

    assert PointCount(failures, 3).holds(STATUS, [])
    assert not PointCount(failures, 4).holds(STATUS, [])
    assert not PointCount(("killed",), 1).holds(STATUS, [])
    assert not PointCount(("completed",), 1).holds({}, [])


def test_a_point_count_refuses_a_status_no_point_takes() -> None:
    with pytest.raises(ValueError, match="'lost'"):
        PointCount(("lost",), 1)

    assert {"completed", "not-submitted", "numerical-failure"} <= POINT_STATUSES


@pytest.mark.parametrize(
    ("statistic", "direction", "threshold", "holds"),
    [
        (Statistic.MEAN, Direction.ABOVE, 0.5, True),
        (Statistic.MEAN, Direction.ABOVE, 0.6, False),
        (Statistic.MEAN, Direction.BELOW, 0.5, False),
        (Statistic.MEAN, Direction.BELOW, 0.6, True),
        (Statistic.MINIMUM, Direction.BELOW, 0.3, True),
        (Statistic.MINIMUM, Direction.BELOW, 0.2, False),
        (Statistic.MAXIMUM, Direction.ABOVE, 0.8, True),
        (Statistic.MAXIMUM, Direction.ABOVE, 0.9, False),
    ],
)
def test_a_metric_bound_reduces_the_rows_values_and_compares_strictly(
    statistic: Statistic, direction: Direction, threshold: float, *, holds: bool
) -> None:
    observed = rows(0.2, 0.6, 0.9, None, float("nan")) + rows(5.0, task="other")

    assert bound(statistic, direction, threshold).holds({}, observed) is holds


def test_a_metric_bound_does_not_hold_on_fewer_rows_than_it_reads() -> None:
    assert not bound(Statistic.MEAN, Direction.ABOVE, 0.0, minimum=3).holds(
        {}, rows(1.0, 1.0)
    )
    assert bound(Statistic.MEAN, Direction.ABOVE, 0.0, minimum=2).holds(
        {}, rows(1.0, 1.0)
    )


def test_a_decision_is_pending_before_its_date_and_read_from_its_signal_after() -> None:
    decision = DecisionPoint(
        date(2026, 12, 1), bound(Statistic.MEAN, Direction.ABOVE, 0.5), "fallback"
    )

    assert decision.state(date(2026, 11, 30), {}, rows(0.9)) is DecisionState.PENDING
    assert decision.state(date(2026, 12, 1), {}, rows(0.9)) is (
        DecisionState.SIGNAL_HELD
    )
    assert decision.state(date(2026, 12, 2), {}, rows(0.1)) is DecisionState.FALLBACK


def test_an_assessment_names_the_stop_rules_that_fire_in_declared_order() -> None:
    rules = (
        StopRule("failures", PointCount(("numerical-failure",), 1)),
        StopRule("collapse", bound(Statistic.MAXIMUM, Direction.BELOW, 0.1)),
        StopRule("infrastructure", PointCount(("infra-failed",), 2)),
    )
    decision = DecisionPoint(date(2030, 1, 1), PointCount(("completed",), 1), "f")

    assessment = assess(
        rules, decision, today=date(2026, 1, 1), status=STATUS, rows=rows(0.5)
    )

    assert assessment.fired == ("failures", "infrastructure")
    assert assessment.decision is DecisionState.PENDING
