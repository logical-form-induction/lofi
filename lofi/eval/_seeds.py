"""Seed requirements and summaries."""

from math import ceil
from statistics import NormalDist, quantiles, stdev
from typing import TYPE_CHECKING, Final, Literal, override

from lofi._checks import ensure_finite, ensure_non_negative
from lofi.eval._values import MINIMUM_SEEDS, MetricResult, seed_summary

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi._types import JsonValue

__all__ = [
    "SEED_TARGETS",
    "SeedBudget",
    "SeedTarget",
    "multi_seed_summary",
    "seed_budget",
    "seeds_required",
]


type SeedTarget = Literal["mean", "variance", "upper-decile"]
"""Defines the quantity a seed budget is being priced for."""

SEED_TARGETS: Final[tuple[SeedTarget, ...]] = ("mean", "variance", "upper-decile")
"""Budget targets in report order."""

_STANDARD_NORMAL: Final = NormalDist()
"""The reference distribution for seed requirements."""

_UPPER_DECILE: Final = 0.9
"""The upper quantile used by the extreme-value target."""

_DECILE_FACTOR: Final = (
    _UPPER_DECILE
    * (1.0 - _UPPER_DECILE)
    / _STANDARD_NORMAL.pdf(_STANDARD_NORMAL.inv_cdf(_UPPER_DECILE)) ** 2.0
)
"""The upper-decile difficulty relative to the mean."""


def _coverage(level: float) -> float:
    """Return the two-sided normal deviate for a coverage level.

    Parameters
    ----------
    level : float
        Coverage proportion.

    Returns
    -------
    float
        Standard-normal deviate.

    Raises
    ------
    TypeError
        If ``level`` is not numeric.
    ValueError
        If ``level`` is not finite or is outside $(0, 1)$.
    """
    coverage = ensure_finite("a coverage level", level)

    if not 0.0 < coverage < 1.0:
        msg = f"a coverage level is not a proportion: {level!r}"
        raise ValueError(msg)

    return _STANDARD_NORMAL.inv_cdf(1.0 - (1.0 - coverage) / 2.0)


def _upper_spread(pilot: Sequence[float], level: float) -> float:
    """Return an upper confidence limit on seed-noise deviation.

    Parameters
    ----------
    pilot : Sequence[float]
        Pilot scores.
    level : float
        Confidence level.

    Returns
    -------
    float
        Upper standard-deviation bound.

    Raises
    ------
    ValueError
        If the pilot cannot support the requested bound.
    """
    degrees = float(len(pilot) - 1)
    deviate = _STANDARD_NORMAL.inv_cdf(1.0 - level)
    # Wilson-Hilferty cube-root denominator
    shift = 9.0 * degrees
    base = 1.0 - 2.0 / shift + deviate * (2.0 / shift) ** 0.5

    if base <= 0.0:
        msg = (
            f"{len(pilot)} pilot seeds cannot bound their own spread at "
            f"{level!r} confidence"
        )
        raise ValueError(msg)

    return stdev(pilot) * (1.0 / base**3.0) ** 0.5


def seeds_required(
    pilot: Sequence[float],
    *,
    target: SeedTarget,
    precision: float,
    level: float,
    minimum: int,
) -> int:
    """Return the seed count needed for a target precision.

    Parameters
    ----------
    pilot : Sequence[float]
        Pilot scores from distinct seeds.
    target : SeedTarget
        Quantity whose precision is controlled.
    precision : float
        Positive relative half-width.
    level : float
        Confidence level.
    minimum : int
        The fewest seeds a reported cell rests on; the required count is
        never below it.

    Returns
    -------
    int
        Required seed count.

    Raises
    ------
    TypeError
        If a numeric input has an invalid type.
    ValueError
        If the target, precision, level, pilot, or pilot estimate is invalid.
    """
    if target not in SEED_TARGETS:
        msg = (
            f"a seed budget names a target that is not a declared one: "
            f"{target!r}, which is not in {SEED_TARGETS!r}"
        )
        raise ValueError(msg)

    width = ensure_finite("a target precision", precision)

    if width <= 0.0:
        msg = f"a target precision is not positive: {precision!r}"
        raise ValueError(msg)

    sample = tuple(ensure_finite("a pilot seed score", score) for score in pilot)
    floor = ensure_non_negative("a seed minimum", minimum)

    if len(sample) < 2:
        msg = (
            f"a seed budget needs at least 2 pilot seeds to measure their spread "
            f"and was handed {len(sample)}"
        )
        raise ValueError(msg)

    deviate = _coverage(level)

    # variance of the sample variance for normal draws
    if target == "variance":
        return max(floor, ceil(1.0 + 2.0 * (deviate / width) ** 2.0))

    spread = _upper_spread(sample, level)
    estimate = (
        sum(sample) / len(sample)
        if target == "mean"
        else quantiles(sample, n=10, method="inclusive")[-1]
    )

    if estimate == 0.0:
        msg = (
            f"the pilot estimate of {target!r} is zero, so no proportion "
            f"of it is a width"
        )
        raise ValueError(msg)

    scale = 1.0 if target == "mean" else _DECILE_FACTOR

    return max(
        floor,
        ceil(scale * (deviate * spread / (width * abs(estimate))) ** 2.0),
    )


class SeedBudget:
    """Stores required and observed seed counts.

    Attributes
    ----------
    target : SeedTarget
        Quantity whose precision is controlled.
    precision : float
        Target relative half-width.
    required : int
        Required seed count.
    achieved : int
        Observed seed count.
    is_sufficient : bool
        Whether the observed count meets the requirement.
    shortfall : int
        Additional seed count required.

    Parameters
    ----------
    target : SeedTarget
        Quantity whose precision is controlled.
    precision : float
        Positive relative half-width.
    required : int
        Required seed count.
    achieved : int
        Observed seed count.

    Raises
    ------
    TypeError
        If a numeric input has an invalid type.
    ValueError
        If the target is unknown, precision is invalid, or a count is negative.
    """

    __slots__ = ("_achieved", "_precision", "_required", "_target")

    def __init__(
        self, *, target: SeedTarget, precision: float, required: int, achieved: int
    ) -> None:
        if target not in SEED_TARGETS:
            msg = (
                f"a seed budget names a target that is not a declared one: "
                f"{target!r}, which is not in {SEED_TARGETS!r}"
            )
            raise ValueError(msg)

        self._target: SeedTarget = target
        self._precision = ensure_finite("a target precision", precision)
        self._required = ensure_non_negative("a required seed count", required)
        self._achieved = ensure_non_negative("an achieved seed count", achieved)

        if self._precision <= 0.0:
            msg = f"a target precision is not positive: {precision!r}"
            raise ValueError(msg)

    @property
    def target(self) -> SeedTarget:
        """Return the quantity the precision was stated about."""
        return self._target

    @property
    def precision(self) -> float:
        """Return the target relative half-width."""
        return self._precision

    @property
    def required(self) -> int:
        """Return how many seeds that precision needs."""
        return self._required

    @property
    def achieved(self) -> int:
        """Return how many seeds were actually run."""
        return self._achieved

    @property
    def is_sufficient(self) -> bool:
        """Return whether the run met the requirement."""
        return self._achieved >= self._required

    @property
    def shortfall(self) -> int:
        """Return the number of additional required seeds."""
        return max(0, self._required - self._achieved)

    def to_spec(self) -> dict[str, JsonValue]:
        """Return the budget as report columns.

        Returns
        -------
        dict[str, JsonValue]
            Serialized seed budget.
        """
        return {
            "target": self._target,
            "precision": self._precision,
            "seeds_required": self._required,
            "seeds_run": self._achieved,
            "is_sufficient": self.is_sufficient,
        }

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value prices the same claim the same way."""
        if type(other) is not SeedBudget:
            return NotImplemented

        return (
            self._target == other._target
            and self._precision == other._precision
            and self._required == other._required
            and self._achieved == other._achieved
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the four fields the budget is equal on."""
        return hash(
            (
                SeedBudget,
                self._target,
                self._precision,
                self._required,
                self._achieved,
            )
        )

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this budget."""
        return (
            f"SeedBudget(target={self._target!r}, precision={self._precision!r}, "
            f"required={self._required!r}, achieved={self._achieved!r})"
        )


def seed_budget(
    results: Sequence[MetricResult],
    *,
    target: SeedTarget,
    precision: float,
    level: float,
) -> SeedBudget:
    """Estimate a seed budget from observed results.

    Parameters
    ----------
    results : Sequence[MetricResult]
        Metric results from distinct seeds.
    target : SeedTarget
        Quantity whose precision is controlled.
    precision : float
        Positive relative half-width.
    level : float
        Confidence level.

    Returns
    -------
    SeedBudget
        Required and observed seed counts.

    Raises
    ------
    TypeError
        If an input has an invalid type.
    ValueError
        If results or budget settings are invalid.
    """
    scored = seed_summary(results)
    pilot = tuple(entry.value for entry in results if entry.value is not None)

    return SeedBudget(
        target=target,
        precision=precision,
        required=seeds_required(
            pilot,
            target=target,
            precision=precision,
            level=level,
            minimum=MINIMUM_SEEDS,
        ),
        achieved=int(scored.components["seeds"]),
    )


def multi_seed_summary(
    results: Sequence[MetricResult],
    *,
    target: SeedTarget,
    precision: float,
    level: float,
) -> MetricResult:
    """Summarize results with required and observed seed counts.

    Parameters
    ----------
    results : Sequence[MetricResult]
        Metric results from distinct seeds.
    target : SeedTarget
        Quantity whose precision is controlled.
    precision : float
        Positive relative half-width.
    level : float
        Confidence level.

    Returns
    -------
    MetricResult
        Multi-seed result with budget components.

    Raises
    ------
    TypeError
        If an input has an invalid type.
    ValueError
        If results or budget settings are invalid.
    """
    scored = seed_summary(results)
    budget = seed_budget(results, target=target, precision=precision, level=level)

    return MetricResult(
        scored.value,
        scored.n,
        scored.resampling,
        components={
            **scored.components,
            "seeds_required": float(budget.required),
        },
        ci=scored.ci,
    )
