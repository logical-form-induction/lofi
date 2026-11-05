"""Stop rules and decision points, read from an experiment's status and rows.

A predicate reads the status record a poll writes, whose ``counts`` give the
number of points in each status, and the report rows the completed points
recorded, each naming the evaluator role that reported it as its ``task`` and
its ``metric``. A stop rule fires when its predicate holds. A decision point is
pending before its date; from its date it reports whether its signal held or
the experiment falls back to its fallback claim.
"""

import enum
import math
import operator
from statistics import fmean
from types import MappingProxyType
from typing import TYPE_CHECKING, NamedTuple, Protocol, final, override

from lofi._frozen import FrozenValue
from lofi.run._boundary.experiment import Direction, Statistic
from lofi.run._boundary.manifest import (
    FailureKind,
)
from lofi.run._store import PointStatus

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence
    from datetime import date

    from lofi._types import JsonValue

__all__ = [
    "POINT_STATUSES",
    "Assessment",
    "DecisionPoint",
    "DecisionState",
    "MetricBound",
    "PointCount",
    "Predicate",
    "ReportedRow",
    "StopRule",
    "assess",
]

POINT_STATUSES: frozenset[str] = frozenset(
    {*(status.value for status in PointStatus), *(kind.value for kind in FailureKind)}
)
"""Every status an experiment's status record can give a point."""


_REDUCTIONS: Mapping[Statistic, Callable[[Sequence[float]], float]] = MappingProxyType(
    {Statistic.MEAN: fmean, Statistic.MINIMUM: min, Statistic.MAXIMUM: max}
)
"""The reduction each statistic names."""

_COMPARISONS: Mapping[Direction, Callable[[float, float], bool]] = MappingProxyType(
    {Direction.BELOW: operator.lt, Direction.ABOVE: operator.gt}
)
"""The strict comparison of a reduced value with its threshold each direction names."""


class ReportedRow(NamedTuple):
    """One report row named by the evaluator role that reports it and its metric."""

    evaluator: str
    metric: str

    @property
    def label(self) -> str:
        """Return the row as ``<evaluator>/<metric>``."""
        return f"{self.evaluator}/{self.metric}"


class Predicate(Protocol):
    """A condition over an experiment's status record and its report rows."""

    def holds(
        self,
        status: Mapping[str, JsonValue],
        rows: Sequence[Mapping[str, JsonValue]],
        /,
    ) -> bool:
        """Return whether the condition holds.

        Parameters
        ----------
        status : Mapping[str, JsonValue]
            The experiment's status record.
        rows : Sequence[Mapping[str, JsonValue]]
            The rows its completed points recorded.
        """
        ...


@final
class PointCount(FrozenValue):
    """Holds once enough points are in the counted statuses.

    Parameters
    ----------
    statuses : Iterable[str]
        The statuses counted, each one a status record gives a point.
    at_least : int
        The fewest counted points at which the condition holds.

    Raises
    ------
    ValueError
        If a status is not one a status record gives a point.
    """

    _noun = "point-count predicate"

    __slots__ = ("_at_least", "_statuses")

    def __init__(self, statuses: Iterable[str], at_least: int) -> None:
        counted = frozenset(statuses)
        unknown = sorted(counted - POINT_STATUSES)

        if unknown:
            msg = (
                f"a point-count predicate counts statuses no point takes: {unknown}; "
                f"a point takes one of {sorted(POINT_STATUSES)}"
            )
            raise ValueError(msg)

        object.__setattr__(self, "_statuses", counted)
        object.__setattr__(self, "_at_least", at_least)

    def holds(
        self,
        status: Mapping[str, JsonValue],
        rows: Sequence[Mapping[str, JsonValue]],
        /,
    ) -> bool:
        """Return whether the counted statuses hold enough points.

        Parameters
        ----------
        status : Mapping[str, JsonValue]
            The experiment's status record.
        rows : Sequence[Mapping[str, JsonValue]]
            The rows, which the count does not read.
        """
        del rows
        counts = status.get("counts")

        if not isinstance(counts, dict):
            return False

        counted = sum(
            count
            for name, count in counts.items()
            if name in self._statuses and isinstance(count, int)
        )

        return counted >= self._at_least

    @override
    def __repr__(self) -> str:
        return f"PointCount({sorted(self._statuses)!r}, at_least={self._at_least!r})"


@final
class MetricBound(FrozenValue):
    """Holds once one row's reduced values lie on the declared side of a threshold.

    Parameters
    ----------
    row : ReportedRow
        The row whose values are reduced.
    statistic : Statistic
        How the values are reduced.
    direction : Direction
        Which side of the threshold the reduced value lies on when it holds.
    threshold : float
        The threshold, compared strictly.
    minimum_rows : int
        The fewest values the condition is read from.
    """

    _noun = "metric predicate"

    __slots__ = ("_direction", "_minimum_rows", "_row", "_statistic", "_threshold")

    def __init__(
        self,
        row: ReportedRow,
        *,
        statistic: Statistic,
        direction: Direction,
        threshold: float,
        minimum_rows: int,
    ) -> None:
        object.__setattr__(self, "_row", row)
        object.__setattr__(self, "_statistic", statistic)
        object.__setattr__(self, "_direction", direction)
        object.__setattr__(self, "_threshold", threshold)
        object.__setattr__(self, "_minimum_rows", minimum_rows)

    @property
    def row(self) -> ReportedRow:
        """Return the row whose values are reduced."""
        return self._row

    def holds(
        self,
        status: Mapping[str, JsonValue],
        rows: Sequence[Mapping[str, JsonValue]],
        /,
    ) -> bool:
        """Return whether the row's reduced finite values pass the threshold.

        Parameters
        ----------
        status : Mapping[str, JsonValue]
            The status record, which the bound does not read.
        rows : Sequence[Mapping[str, JsonValue]]
            The rows its completed points recorded.
        """
        del status
        values = tuple(
            float(value)
            for row in rows
            if row.get("task") == self._row.evaluator
            and row.get("metric") == self._row.metric
            and isinstance(value := row.get("value"), int | float)
            and not isinstance(value, bool)
            and math.isfinite(value)
        )

        if len(values) < self._minimum_rows:
            return False

        reduced = _REDUCTIONS[self._statistic](values)

        return _COMPARISONS[self._direction](reduced, self._threshold)

    @override
    def __repr__(self) -> str:
        return (
            f"MetricBound({self._row.label!r}, {self._statistic.value} "
            f"{self._direction.value} {self._threshold!r})"
        )


class StopRule(NamedTuple):
    """A named condition under which an experiment halts early."""

    name: str
    predicate: Predicate


class DecisionState(enum.StrEnum):
    """Where an experiment stands against its decision point."""

    PENDING = "pending"
    SIGNAL_HELD = "signal-held"
    FALLBACK = "fallback"


class DecisionPoint(NamedTuple):
    """The date and signal an experiment is judged at, and its fallback claim."""

    decided_on: date
    signal: Predicate
    fallback: str

    def state(
        self,
        today: date,
        status: Mapping[str, JsonValue],
        rows: Sequence[Mapping[str, JsonValue]],
        /,
    ) -> DecisionState:
        """Return where the experiment stands on a given day.

        Parameters
        ----------
        today : date
            The day the decision is read on.
        status : Mapping[str, JsonValue]
            The experiment's status record.
        rows : Sequence[Mapping[str, JsonValue]]
            The rows its completed points recorded.
        """
        if today < self.decided_on:
            return DecisionState.PENDING

        if self.signal.holds(status, rows):
            return DecisionState.SIGNAL_HELD

        return DecisionState.FALLBACK


class Assessment(NamedTuple):
    """The stop rules that fired and the state of the decision point."""

    fired: tuple[str, ...]
    decision: DecisionState


def assess(
    stop_rules: Sequence[StopRule],
    decision: DecisionPoint,
    *,
    today: date,
    status: Mapping[str, JsonValue],
    rows: Sequence[Mapping[str, JsonValue]],
) -> Assessment:
    """Return which stop rules fire and where the decision point stands.

    Parameters
    ----------
    stop_rules : Sequence[StopRule]
        The experiment's stop rules, in declared order.
    decision : DecisionPoint
        The experiment's decision point.
    today : date
        The day the decision is read on.
    status : Mapping[str, JsonValue]
        The experiment's status record.
    rows : Sequence[Mapping[str, JsonValue]]
        The rows its completed points recorded.
    """
    return Assessment(
        tuple(rule.name for rule in stop_rules if rule.predicate.holds(status, rows)),
        decision.state(today, status, rows),
    )
