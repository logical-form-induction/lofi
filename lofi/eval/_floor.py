"""Empirical performance floors."""

from collections import Counter
from math import exp, expm1, floor, fsum, lgamma, log, log1p
from typing import TYPE_CHECKING, Final, final, override

from lofi._checks import ensure_name, ensure_non_negative
from lofi._frozen import FrozenValue
from lofi.eval._alignment import Contingency
from lofi.eval._values import MetricResult, Resampling
from lofi.stats import Bootstrap, Interval

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from lofi._types import JsonValue
    from lofi.stats import Statistic

__all__ = ["Floor", "FlooredScore", "random_relabeling_floor"]


_MINIMUM_INVENTORY: Final = 1
"""The smallest inventory available to a labeling."""

_MINIMUM_TOKENS: Final = 1
"""The smallest sample a random-labeling floor has a value over."""

_INTERVAL_TOKENS: Final = 2
"""The smallest sample that supports a token bootstrap interval."""

_MANY_TO_ONE: Final = "many-to-one"
"""The assignment metric with an exact random-labeling floor."""

type _MetricResultFields = tuple[
    float | None,
    int,
    tuple[str, int | None],
    tuple[tuple[str, float], ...],
    tuple[float, float, float | None] | None,
]
"""Fields needed to reconstruct a metric result."""

type _FloorFields = tuple[str, int, _MetricResultFields]
"""Fields needed to reconstruct a floor."""

type _FloorReduction = tuple[Callable[[_FloorFields], Floor], tuple[_FloorFields]]
"""Floor pickle reduction."""

type _FlooredScoreFields = tuple[_MetricResultFields, Floor]
"""Fields needed to reconstruct a floored score."""

type _FlooredScoreReduction = tuple[
    Callable[[_FlooredScoreFields], FlooredScore],
    tuple[_FlooredScoreFields],
]
"""Floored-score pickle reduction."""


def _positive_inventory(value: int) -> int:
    """Return a positive inventory size."""
    held = ensure_non_negative("a floor's inventory size", value)

    if held < _MINIMUM_INVENTORY:
        msg = f"a floor's inventory size is below one: {value!r}"
        raise ValueError(msg)

    return held


@final
class Floor(FrozenValue):
    """A random-labeling score and its interval.

    A floor over fewer than two tokens has no bootstrap interval, so its
    interval is absent; a floor over two or more tokens always has one.

    Attributes
    ----------
    metric : str
        Assignment metric name.
    inventory : int
        Induced-label inventory size.
    result : MetricResult
        Random-labeling metric result.
    value : float
        Random-labeling score.
    ci : Interval | None
        Interval around the score, absent over fewer than two tokens.

    Parameters
    ----------
    metric : str
        Assignment metric name.
    inventory : int
        Positive induced-label inventory size.
    result : MetricResult
        Random-labeling metric result.

    Raises
    ------
    TypeError
        If a field has an invalid type.
    ValueError
        If metadata or the metric result is invalid for a floor, or the
        result's interval is present over fewer than two tokens or absent over
        more.
    """

    _noun = "floor value"

    __slots__ = ("_inventory", "_metric", "_result_fields")

    def __init__(self, metric: str, inventory: int, result: MetricResult) -> None:
        if not isinstance(result, MetricResult):
            msg = f"a floor's result is not one: {result!r}"
            raise TypeError(msg)

        held_metric = ensure_name("a floor's metric", metric)
        held_inventory = _positive_inventory(inventory)

        if result.value is None:
            msg = f"a floor over {metric!r} reports no value"
            raise ValueError(msg)

        if result.ci is None and result.n >= _INTERVAL_TOKENS:
            msg = f"a floor over {metric!r} reports no interval over {result.n} tokens"
            raise ValueError(msg)

        if result.ci is not None and result.n < _INTERVAL_TOKENS:
            msg = (
                f"a floor over {metric!r} reports an interval over {result.n} "
                "token, which no bootstrap supports"
            )
            raise ValueError(msg)

        if result.resampling.unit != "token":
            msg = (
                "a random-labeling floor must resample 'token', not "
                f"{result.resampling.unit!r}"
            )
            raise ValueError(msg)

        object.__setattr__(self, "_metric", held_metric)
        object.__setattr__(self, "_inventory", held_inventory)
        object.__setattr__(self, "_result_fields", _metric_fields(result))

    @property
    def metric(self) -> str:
        """Return the assignment metric the floor is the chance value of."""
        return self._metric

    @property
    def inventory(self) -> int:
        """Return how many induced labels the random labeling drew from."""
        return self._inventory

    @property
    def result(self) -> MetricResult:
        """Return the value, the count of units behind it and its interval."""
        return _restore_metric_result(self._result_fields)

    @property
    def value(self) -> float:
        """Return the random-labeling score."""
        value = self._result_fields[0]

        if value is None:
            msg = "a constructed floor has no value"
            raise RuntimeError(msg)

        return value

    @property
    def ci(self) -> Interval | None:
        """Return the interval around it, absent over fewer than two tokens."""
        interval = self._result_fields[4]

        return None if interval is None else Interval(*interval)

    def to_spec(self) -> dict[str, JsonValue]:
        """Return the floor as report columns.

        Returns
        -------
        dict[str, JsonValue]
            Serialized floor, with null interval columns when it has none.
        """
        interval = self.ci

        return {
            "metric": self._metric,
            "inventory": self._inventory,
            "floor": self.value,
            "floor_ci_lo": None if interval is None else interval.low,
            "floor_ci_hi": None if interval is None else interval.high,
            "floor_ci_level": None if interval is None else interval.level,
        }

    def _fields(self) -> _FloorFields:
        """Return fields in reconstruction order."""
        return self._metric, self._inventory, self._result_fields

    @override
    def __reduce__(self) -> _FloorReduction:
        """Return the floor reconstruction call."""
        return _restore_floor, (self._fields(),)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value is the same floor at the same size."""
        if type(other) is not Floor:
            return NotImplemented

        return (
            self._metric == other._metric
            and self._inventory == other._inventory
            and self._result_fields == other._result_fields
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the metric, the inventory and the result."""
        return hash((Floor, self._metric, self._inventory, self._result_fields))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this floor."""
        return f"Floor({self._metric!r}, {self._inventory!r}, {self.result!r})"


def _restore_floor(fields: _FloorFields) -> Floor:
    """Reconstruct a floor from frozen fields."""
    metric, inventory, result = fields

    return Floor(metric, inventory, _restore_metric_result(result))


@final
class FlooredScore(FrozenValue):
    """An assignment score and its chance floor.

    Attributes
    ----------
    observed : MetricResult
        Observed assignment result.
    floor : Floor
        Matching random-labeling floor.
    excess : float | None
        Observed score minus the floor, if defined.
    beats_floor : bool | None
        Whether the score exceeds the floor interval, or the floor value when
        the floor has no interval, if defined.

    Parameters
    ----------
    metric : str
        Assignment metric name.
    observed : MetricResult
        Observed assignment result.
    floor : Floor
        Matching random-labeling floor.

    Raises
    ------
    TypeError
        If ``observed`` or ``floor`` has an invalid type.
    ValueError
        If the metric, sample count, unit, or inventory differs from the floor.
    """

    _noun = "floor value"

    __slots__ = ("_floor", "_observed_fields")

    def __init__(self, metric: str, observed: MetricResult, floor: Floor) -> None:
        if not isinstance(observed, MetricResult):
            msg = f"a floored score's observation is not a result: {observed!r}"
            raise TypeError(msg)

        if not isinstance(floor, Floor):
            msg = f"a floored score's floor is not one: {floor!r}"
            raise TypeError(msg)

        held_metric = ensure_name("a floored score's metric", metric)

        if held_metric != floor.metric:
            msg = (
                f"an observation uses metric {held_metric!r} but its floor uses "
                f"{floor.metric!r}"
            )
            raise ValueError(msg)

        if observed.n != floor.result.n:
            msg = (
                f"an observation counts {observed.n} tokens but its floor counts "
                f"{floor.result.n}"
            )
            raise ValueError(msg)

        if observed.resampling.unit != floor.result.resampling.unit:
            msg = (
                f"an observation counts {observed.resampling.unit!r} but its floor "
                f"counts {floor.result.resampling.unit!r}"
            )
            raise ValueError(msg)

        observed_inventory = observed.components.get("induced_labels")

        if observed_inventory is None:
            msg = "an observation does not report its induced-label inventory"
            raise ValueError(msg)

        if observed_inventory != floor.inventory:
            msg = (
                f"an observation uses {observed_inventory!r} induced labels but its "
                f"floor uses {floor.inventory}"
            )
            raise ValueError(msg)

        object.__setattr__(self, "_observed_fields", _metric_fields(observed))
        object.__setattr__(self, "_floor", floor)

    @property
    def observed(self) -> MetricResult:
        """Return the observed score."""
        return _restore_metric_result(self._observed_fields)

    @property
    def floor(self) -> Floor:
        """Return the matching random-labeling floor."""
        return self._floor

    @property
    def excess(self) -> float | None:
        """Return the observed score minus the floor, if available."""
        observed = self._observed_fields[0]

        return None if observed is None else observed - self._floor.value

    @property
    def beats_floor(self) -> bool | None:
        """Return whether the observation lies above the floor's interval.

        A floor without an interval is beaten by an observation above its value.
        """
        observed = self._observed_fields[0]

        if observed is None:
            return None

        interval = self._floor.ci
        bound = self._floor.value if interval is None else interval.high

        return observed > bound

    def to_spec(self) -> dict[str, JsonValue]:
        """Return the floored score as report columns.

        Returns
        -------
        dict[str, JsonValue]
            Serialized observation and floor.
        """
        return {
            "observed": self._observed_fields[0],
            **self._floor.to_spec(),
            "excess": self.excess,
            "beats_floor": self.beats_floor,
        }

    def _fields(self) -> _FlooredScoreFields:
        """Return fields in reconstruction order."""
        return self._observed_fields, self._floor

    @override
    def __reduce__(self) -> _FlooredScoreReduction:
        """Return the floored-score reconstruction call."""
        return _restore_floored_score, (self._fields(),)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value pairs the same observation and floor."""
        if type(other) is not FlooredScore:
            return NotImplemented

        return (
            self._observed_fields == other._observed_fields
            and self._floor == other._floor
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the observation and the floor."""
        return hash((FlooredScore, self._observed_fields, self._floor))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this row."""
        return (
            f"FlooredScore({self._floor.metric!r}, {self.observed!r}, {self._floor!r})"
        )


def _restore_floored_score(fields: _FlooredScoreFields) -> FlooredScore:
    """Reconstruct a floored score from frozen fields."""
    observed, floor = fields

    return FlooredScore(floor.metric, _restore_metric_result(observed), floor)


def _metric_fields(result: MetricResult) -> _MetricResultFields:
    """Return a metric result's pickle-safe fields."""
    resampling = result.resampling
    interval = result.ci

    return (
        result.value,
        result.n,
        (resampling.unit, resampling.seed),
        tuple(sorted(result.components.items())),
        None if interval is None else (interval.low, interval.high, interval.level),
    )


def _restore_metric_result(fields: _MetricResultFields) -> MetricResult:
    """Reconstruct a metric result from pickle-safe fields."""
    value, count, resampling, components, interval = fields

    return MetricResult(
        value,
        count,
        Resampling(*resampling),
        components=dict(components),
        ci=None if interval is None else Interval(*interval),
    )


def _binomial_survival(trials: int, probability: float) -> tuple[float, ...]:
    """Return the chance of reaching each positive count."""
    miss = 1.0 - probability
    mode = min(trials, floor((trials + 1) * probability))
    mass = exp(
        lgamma(trials + 1)
        - lgamma(mode + 1)
        - lgamma(trials - mode + 1)
        + mode * log(probability)
        + (trials - mode) * log1p(-probability)
    )
    masses = [0.0] * (trials + 1)
    masses[mode] = mass

    for count in range(mode, 0, -1):
        mass *= count * miss / ((trials - count + 1) * probability)
        masses[count - 1] = mass

    mass = masses[mode]

    for count in range(mode + 1, trials + 1):
        mass *= (trials - count + 1) * probability / (count * miss)
        masses[count] = mass

    reached = 0.0
    survival = [0.0] * trials

    for index in range(trials - 1, -1, -1):
        reached += masses[index + 1]
        survival[index] = min(1.0, reached)

    return tuple(survival)


def _expected_accuracy(marginals: Sequence[int], inventory: int) -> float:
    """Return the expected score of uniform random labeling.

    Parameters
    ----------
    marginals : Sequence[int]
        Gold-label counts.
    inventory : int
        Induced-label inventory size.

    Returns
    -------
    float
        Expected many-to-one accuracy.
    """
    positive = tuple(count for count in marginals if count > 0)
    total = sum(positive)

    if total == 0:
        return 0.0

    if len(positive) == 1:
        return 1.0

    if inventory == _MINIMUM_INVENTORY:
        return max(positive) / total

    probability = exp(-log(inventory))

    if probability == 0.0:
        return 1.0

    survivals = tuple(_binomial_survival(count, probability) for count in positive)
    largest = max(positive)
    reached: list[float] = []

    for threshold in range(largest):
        absent = []

        for survival in survivals:
            chance = survival[threshold] if threshold < len(survival) else 0.0

            if chance >= 1.0:
                absent = None
                break

            absent.append(log1p(-chance))

        reached.append(1.0 if absent is None else -expm1(fsum(absent)))

    return min(1.0, fsum(reached) / (total * probability))


def _statistic(inventory: int) -> Statistic[Sequence[str]]:
    """Return a floor statistic over resampled gold labels.

    Parameters
    ----------
    inventory : int
        Induced-label inventory size.

    Returns
    -------
    Statistic[Sequence[str]]
        Bootstrap statistic for the random-labeling floor.
    """

    def measured(sample: Sequence[str]) -> float:
        return _expected_accuracy(tuple(Counter(sample).values()), inventory)

    return measured


def random_relabeling_floor(
    contingency: Contingency,
    *,
    inventory: int,
    bootstrap: Bootstrap,
) -> Floor:
    """Return the exact random-labeling score with a bootstrap interval.

    The score is defined over one token; its interval needs two, so a floor
    over a single token reports the score with no interval.

    Parameters
    ----------
    contingency : Contingency
        Gold-label contingency.
    inventory : int
        Positive induced-label inventory size.
    bootstrap : Bootstrap
        Token-level bootstrap configuration.

    Returns
    -------
    Floor
        Random-labeling floor and interval.

    Raises
    ------
    TypeError
        If an input has an invalid type.
    ValueError
        If the inventory or bootstrap is invalid, or the table has no tokens.
    """
    if not isinstance(contingency, Contingency):
        msg = f"a floor's contingency is not one: {contingency!r}"
        raise TypeError(msg)

    if not isinstance(bootstrap, Bootstrap):
        msg = f"a floor's bootstrap is not one: {bootstrap!r}"
        raise TypeError(msg)

    size = _positive_inventory(inventory)

    if bootstrap.unit != "token":
        msg = f"a random-labeling floor must resample 'token', not {bootstrap.unit!r}"
        raise ValueError(msg)

    if bootstrap.groups is not None:
        msg = "a random-labeling floor must resample individual tokens without groups"
        raise ValueError(msg)

    totals: Counter[str] = Counter()

    for (_, gold), count in contingency.counts.items():
        totals[gold] += count

    labels = tuple(gold for gold in sorted(totals) for _ in range(totals[gold]))

    if len(labels) < _MINIMUM_TOKENS:
        msg = "a random-labeling floor needs at least one token, not 0"
        raise ValueError(msg)

    statistic = _statistic(size)
    value = statistic(labels)
    interval = None

    if len(labels) >= _INTERVAL_TOKENS:
        drawn = bootstrap.interval(labels, statistic)
        value = drawn.observed
        interval = Interval(drawn.bca.low, drawn.bca.high, drawn.bca.level)

    return Floor(
        _MANY_TO_ONE,
        size,
        MetricResult(
            value,
            len(labels),
            Resampling(bootstrap.unit, bootstrap.seed),
            components={
                "inventory": float(size),
                "gold_labels": float(len(contingency.gold_labels)),
            },
            ci=interval,
        ),
    )
