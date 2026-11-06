"""Calibration metrics over forecasts of ordered categories."""

from itertools import pairwise
from math import fsum
from typing import TYPE_CHECKING, NamedTuple, final, override

from lofi._checks import ensure_finite, ensure_name, ensure_non_negative
from lofi._types import ItemId
from lofi.eval._values import MetricResult, Resampling

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from lofi._types import JsonValue

__all__ = [
    "CalibrationBin",
    "Forecast",
    "PitHistogram",
    "ReliabilityCurve",
    "brier_score",
    "expected_calibration_error",
    "pit_histogram",
    "reliability_curve",
]


@final
class Forecast:
    """A predictive distribution over ordered categories and the category observed.

    Categories are indexed from zero in their order, so a cumulative mass is the
    mass at or below an index.

    Parameters
    ----------
    item_id : ItemId
        Item identifier.
    probabilities : Sequence[float]
        Predicted mass per category, in category order. At least two categories,
        each nonnegative, summing to one to within ``1e-6``.
    outcome : int
        Index of the observed category.

    Raises
    ------
    TypeError
        If a field has an invalid type.
    ValueError
        If the id is empty, a probability is negative or not finite, the masses do
        not sum to one, or the outcome is not a category index.
    """

    __slots__ = ("_item_id", "_outcome", "_probabilities")

    def __init__(
        self, item_id: ItemId, probabilities: Sequence[float], outcome: int
    ) -> None:
        self._item_id: ItemId = ItemId(ensure_name("a forecast's item id", item_id))
        self._probabilities = tuple(
            ensure_finite(f"forecast {item_id!r} probability {index}", mass)
            for index, mass in enumerate(probabilities)
        )
        self._outcome = ensure_non_negative(f"forecast {item_id!r} outcome", outcome)

        if len(self._probabilities) < 2:
            msg = f"forecast {item_id!r} has fewer than two categories"
            raise ValueError(msg)

        for index, mass in enumerate(self._probabilities):
            if mass < 0.0:
                msg = f"forecast {item_id!r} probability {index} is negative: {mass!r}"
                raise ValueError(msg)

        total = fsum(self._probabilities)

        if abs(total - 1.0) > 1e-6:
            msg = f"forecast {item_id!r} probabilities sum to {total!r}, not one"
            raise ValueError(msg)

        if self._outcome >= len(self._probabilities):
            msg = (
                f"forecast {item_id!r} observes category {outcome} of "
                f"{len(self._probabilities)}"
            )
            raise ValueError(msg)

    @property
    def item_id(self) -> ItemId:
        """Return the opaque handle the item is joined on."""
        return self._item_id

    @property
    def probabilities(self) -> tuple[float, ...]:
        """Return the predicted mass per category, in category order."""
        return self._probabilities

    @property
    def outcome(self) -> int:
        """Return the index of the observed category."""
        return self._outcome

    @property
    def categories(self) -> int:
        """Return how many categories the forecast ranges over."""
        return len(self._probabilities)

    @property
    def predicted(self) -> int:
        """Return the index of the most probable category, the lowest on a tie."""
        return max(
            range(len(self._probabilities)),
            key=lambda index: (self._probabilities[index], -index),
        )

    @property
    def confidence(self) -> float:
        """Return the mass on the most probable category."""
        return self._probabilities[self.predicted]

    @property
    def hit(self) -> bool:
        """Return whether the most probable category was observed."""
        return self.predicted == self._outcome

    def cumulative(self, category: int) -> float:
        """Return the mass at or below a category index.

        Parameters
        ----------
        category : int
            Category index. Minus one names the empty lower set.

        Returns
        -------
        float
            The predicted mass on categories up to and including ``category``.
        """
        return fsum(self._probabilities[: category + 1])

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value is a forecast with the same three fields."""
        if type(other) is not Forecast:
            return NotImplemented

        return (
            self._item_id == other._item_id
            and self._probabilities == other._probabilities
            and self._outcome == other._outcome
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the id, the masses and the outcome."""
        return hash((Forecast, self._item_id, self._probabilities, self._outcome))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this forecast."""
        return (
            f"Forecast({self._item_id!r}, {list(self._probabilities)!r}, "
            f"{self._outcome!r})"
        )


class CalibrationBin(NamedTuple):
    """One equal-mass bin of a reliability curve."""

    low: float
    high: float
    count: int
    mean_probability: float
    mean_outcome: float

    @property
    def gap(self) -> float:
        """Return the absolute difference between outcome rate and mean probability."""
        return abs(self.mean_outcome - self.mean_probability)

    def to_spec(self) -> dict[str, JsonValue]:
        """Return the bin as serializable report columns."""
        return {
            "low": self.low,
            "high": self.high,
            "count": self.count,
            "mean_probability": self.mean_probability,
            "mean_outcome": self.mean_outcome,
        }


type _Event = tuple[float, bool, ItemId]
"""A forecast probability, whether the event it predicts occurred, and the item."""


def _collected(forecasts: Iterable[Forecast]) -> tuple[Forecast, ...]:
    """Return the forecasts, checked to range over one category count."""
    collected = tuple(forecasts)

    for position, forecast in enumerate(collected):
        if not isinstance(forecast, Forecast):
            msg = f"forecast {position} is not a Forecast: {forecast!r}"
            raise TypeError(msg)

        if forecast.categories != collected[0].categories:
            msg = (
                f"forecast {forecast.item_id!r} ranges over {forecast.categories} "
                f"categories where the first ranges over {collected[0].categories}"
            )
            raise ValueError(msg)

    return collected


def _events(collected: Sequence[Forecast], category: int | None) -> tuple[_Event, ...]:
    """Return one predicted event per forecast.

    With a category, the event is that category being observed and the probability
    is its mass. Without one, the event is the most probable category being observed
    and the probability is its mass.
    """
    if category is None:
        return tuple(
            (forecast.confidence, forecast.hit, forecast.item_id)
            for forecast in collected
        )

    named = ensure_non_negative("a calibration category", category)

    if collected and named >= collected[0].categories:
        msg = (
            f"category {named} is outside the {collected[0].categories} "
            f"categories the forecasts range over"
        )
        raise ValueError(msg)

    return tuple(
        (
            forecast.probabilities[named],
            forecast.outcome == named,
            forecast.item_id,
        )
        for forecast in collected
    )


def _bin(members: Sequence[_Event]) -> CalibrationBin:
    """Summarize one bin of events ordered by probability."""
    count = len(members)
    probabilities = [probability for probability, _, _ in members]

    return CalibrationBin(
        low=probabilities[0],
        high=probabilities[-1],
        count=count,
        mean_probability=fsum(probabilities) / count,
        mean_outcome=sum(1 for _, occurred, _ in members if occurred) / count,
    )


@final
class ReliabilityCurve:
    """A reliability curve as a table of equal-mass bins.

    Events are sorted by probability, ties broken by item id, and cut into bins
    whose counts differ by at most one.

    Parameters
    ----------
    bins : Sequence[CalibrationBin]
        Bins in ascending probability order.
    category : int | None
        Category whose mass the curve is drawn over, or ``None`` for the most
        probable category of each forecast.

    Raises
    ------
    TypeError
        If a bin is not a ``CalibrationBin`` or ``category`` is not an integer.
    ValueError
        If a bin is empty, ``category`` is negative, or bin counts differ by more
        than one.
    """

    __slots__ = ("_bins", "_category")

    def __init__(
        self, bins: Sequence[CalibrationBin], category: int | None = None
    ) -> None:
        self._bins = tuple(bins)
        self._category = (
            None
            if category is None
            else ensure_non_negative("a reliability curve's category", category)
        )

        for entry in self._bins:
            if not isinstance(entry, CalibrationBin):
                msg = f"a reliability bin is not one: {entry!r}"
                raise TypeError(msg)

            if entry.count < 1:
                msg = f"a reliability bin is empty: {entry!r}"
                raise ValueError(msg)

        counts = {entry.count for entry in self._bins}

        if counts and max(counts) - min(counts) > 1:
            msg = f"reliability bins are not equal-mass: counts {sorted(counts)!r}"
            raise ValueError(msg)

    @property
    def bins(self) -> tuple[CalibrationBin, ...]:
        """Return the bins in ascending probability order."""
        return self._bins

    @property
    def category(self) -> int | None:
        """Return the category the curve is drawn over, if one was named."""
        return self._category

    @property
    def n(self) -> int:
        """Return how many forecasts the bins hold together."""
        return sum(entry.count for entry in self._bins)

    @property
    def expected_calibration_error(self) -> float | None:
        """Return the count-weighted mean gap across bins, if any were binned."""
        total = self.n

        if total == 0:
            return None

        return fsum(entry.count * entry.gap for entry in self._bins) / total

    def to_spec(self) -> dict[str, JsonValue]:
        """Return the curve as a serializable bin table.

        Returns
        -------
        dict[str, JsonValue]
            Serialized curve.
        """
        return {
            "category": self._category,
            "n": self.n,
            "expected_calibration_error": self.expected_calibration_error,
            "bins": [entry.to_spec() for entry in self._bins],
        }

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value is a curve with the same bins and category."""
        if type(other) is not ReliabilityCurve:
            return NotImplemented

        return self._bins == other._bins and self._category == other._category

    @override
    def __hash__(self) -> int:
        """Return a hash over the bins and the category."""
        return hash((ReliabilityCurve, self._bins, self._category))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this curve."""
        return f"ReliabilityCurve({list(self._bins)!r}, {self._category!r})"


def _bin_count(bins: int) -> int:
    """Return a validated bin count."""
    count = ensure_non_negative("a bin count", bins)

    if count < 1:
        msg = f"a bin count is not positive: {bins!r}"
        raise ValueError(msg)

    return count


def reliability_curve(
    forecasts: Iterable[Forecast], *, category: int | None = None, bins: int = 10
) -> ReliabilityCurve:
    """Bin forecasts into equal-mass probability bins and tabulate outcome rates.

    Parameters
    ----------
    forecasts : Iterable[Forecast]
        Forecasts, all over the same number of categories.
    category : int | None
        Category whose mass is binned against its occurrence, or ``None`` to bin
        each forecast's most probable category against its occurrence.
    bins : int
        Number of equal-mass bins.

    Returns
    -------
    ReliabilityCurve
        One bin per requested bin, or no bins when there are no forecasts.

    Raises
    ------
    TypeError
        If a forecast is not one, or ``bins`` is not an integer.
    ValueError
        If ``bins`` is not positive, ``category`` is outside the forecasts'
        categories, the forecasts differ in category count, or there are fewer
        forecasts than bins.
    """
    width = _bin_count(bins)
    events = sorted(
        _events(_collected(forecasts), category),
        key=lambda event: (event[0], event[2]),
    )
    count = len(events)

    if count == 0:
        return ReliabilityCurve((), category)

    if count < width:
        msg = f"{count} forecasts cannot fill {width} equal-mass bins"
        raise ValueError(msg)

    edges = [count * index // width for index in range(width + 1)]

    return ReliabilityCurve(
        tuple(_bin(events[start:stop]) for start, stop in pairwise(edges)),
        category,
    )


def expected_calibration_error(
    forecasts: Iterable[Forecast], *, category: int | None = None, bins: int = 10
) -> MetricResult:
    """Compute expected calibration error over equal-mass bins.

    Parameters
    ----------
    forecasts : Iterable[Forecast]
        Forecasts, all over the same number of categories.
    category : int | None
        Category whose mass is scored against its occurrence, or ``None`` to score
        each forecast's most probable category against its occurrence.
    bins : int
        Number of equal-mass bins.

    Returns
    -------
    MetricResult
        The count-weighted mean absolute gap between outcome rate and mean
        probability across bins, over items. No value when there are no forecasts.

    Raises
    ------
    TypeError
        If a forecast is not one, or ``bins`` is not an integer.
    ValueError
        If ``bins`` is not positive, ``category`` is outside the forecasts'
        categories, the forecasts differ in category count, or there are fewer
        forecasts than bins.
    """
    curve = reliability_curve(forecasts, category=category, bins=bins)

    return MetricResult(curve.expected_calibration_error, curve.n, Resampling("item"))


def brier_score(
    forecasts: Iterable[Forecast], *, category: int | None = None
) -> MetricResult:
    """Compute the mean squared distance between forecast and outcome.

    Parameters
    ----------
    forecasts : Iterable[Forecast]
        Forecasts, all over the same number of categories.
    category : int | None
        Category whose mass is scored against its occurrence, or ``None`` to sum
        the squared distance over every category. For two categories the summed
        score is twice the score of either category.

    Returns
    -------
    MetricResult
        The mean squared distance over items. No value when there are no
        forecasts.

    Raises
    ------
    TypeError
        If a forecast is not one.
    ValueError
        If ``category`` is outside the forecasts' categories, or the forecasts
        differ in category count.
    """
    collected = _collected(forecasts)

    if category is None:
        distances = [
            fsum(
                (mass - float(index == forecast.outcome)) ** 2
                for index, mass in enumerate(forecast.probabilities)
            )
            for forecast in collected
        ]
    else:
        distances = [
            (probability - float(occurred)) ** 2
            for probability, occurred, _ in _events(collected, category)
        ]

    if not distances:
        return MetricResult(None, 0, Resampling("item"))

    return MetricResult(
        fsum(distances) / len(distances), len(distances), Resampling("item")
    )


@final
class PitHistogram:
    """Counts of randomized probability integral transforms over equal-width bins.

    Parameters
    ----------
    counts : Sequence[int]
        Transforms per bin, in ascending order over the unit interval.

    Raises
    ------
    TypeError
        If a count is not an integer.
    ValueError
        If there are no bins or a count is negative.
    """

    __slots__ = ("_counts",)

    def __init__(self, counts: Sequence[int]) -> None:
        self._counts = tuple(
            ensure_non_negative(f"transform count {index}", count)
            for index, count in enumerate(counts)
        )

        if not self._counts:
            msg = "a transform histogram has no bins"
            raise ValueError(msg)

    @property
    def counts(self) -> tuple[int, ...]:
        """Return the transforms per bin."""
        return self._counts

    @property
    def bins(self) -> int:
        """Return how many equal-width bins the unit interval was cut into."""
        return len(self._counts)

    @property
    def n(self) -> int:
        """Return how many transforms were binned."""
        return sum(self._counts)

    @property
    def fractions(self) -> tuple[float, ...] | None:
        """Return the share of transforms per bin, if any were binned."""
        total = self.n

        if total == 0:
            return None

        return tuple(count / total for count in self._counts)

    @property
    def max_deviation(self) -> float | None:
        """Return the largest gap between a bin's share and the uniform share."""
        fractions = self.fractions

        if fractions is None:
            return None

        uniform = 1.0 / len(self._counts)

        return max(abs(fraction - uniform) for fraction in fractions)

    def to_spec(self) -> dict[str, JsonValue]:
        """Return the histogram as serializable report columns.

        Returns
        -------
        dict[str, JsonValue]
            Serialized histogram.
        """
        fractions = self.fractions

        return {
            "bins": self.bins,
            "n": self.n,
            "counts": list(self._counts),
            "fractions": None if fractions is None else list(fractions),
            "max_deviation": self.max_deviation,
        }

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value is a histogram with the same counts."""
        if type(other) is not PitHistogram:
            return NotImplemented

        return self._counts == other._counts

    @override
    def __hash__(self) -> int:
        """Return a hash over the counts."""
        return hash((PitHistogram, self._counts))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this histogram."""
        return f"PitHistogram({list(self._counts)!r})"


def pit_histogram(
    forecasts: Iterable[Forecast], uniforms: Sequence[float], *, bins: int = 10
) -> PitHistogram:
    """Histogram the randomized probability integral transform of each forecast.

    A forecast observing category ``k`` transforms to the cumulative mass below
    ``k`` plus a uniform draw's share of the mass on ``k``, which is uniform on the
    unit interval when the forecasts are calibrated.

    Parameters
    ----------
    forecasts : Iterable[Forecast]
        Forecasts, all over the same number of categories.
    uniforms : Sequence[float]
        One draw per forecast, each in the unit interval.
    bins : int
        Number of equal-width bins.

    Returns
    -------
    PitHistogram
        Transform counts per bin.

    Raises
    ------
    TypeError
        If a forecast is not one, ``bins`` is not an integer, or a draw is not a
        number.
    ValueError
        If ``bins`` is not positive, the forecasts differ in category count, the
        draws do not number the forecasts, or a draw is outside the unit interval.
    """
    width = _bin_count(bins)
    collected = _collected(forecasts)

    if len(uniforms) != len(collected):
        msg = f"{len(uniforms)} draws randomize {len(collected)} forecasts"
        raise ValueError(msg)

    counts = [0] * width

    for forecast, draw in zip(collected, uniforms, strict=True):
        share = ensure_finite(f"forecast {forecast.item_id!r} draw", draw)

        if not 0.0 <= share <= 1.0:
            msg = (
                f"forecast {forecast.item_id!r} draw is outside the unit interval: "
                f"{draw!r}"
            )
            raise ValueError(msg)

        below = forecast.cumulative(forecast.outcome - 1)
        transform = below + share * forecast.probabilities[forecast.outcome]
        counts[min(int(transform * width), width - 1)] += 1

    return PitHistogram(counts)
