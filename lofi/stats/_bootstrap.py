"""Nonparametric bootstrap intervals."""

from math import floor, fsum
from random import Random
from statistics import NormalDist, fmean
from typing import TYPE_CHECKING, Final

from lofi._checks import ensure_finite, ensure_name, ensure_non_negative
from lofi.stats._values import BootstrapResult, Interval

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi.stats._ports import Statistic

__all__ = ["Bootstrap", "CrossedBootstrap", "paired_bootstrap"]

_STANDARD_NORMAL: Final = NormalDist()
"""The distribution used to adjust accelerated-interval quantiles."""

_MINIMUM_UNITS: Final = 2
"""The minimum unit count for bootstrap resampling."""


def _quantile(ordered: Sequence[float], probability: float) -> float:
    """Return an empirical quantile of an ordered sample.

    Parameters
    ----------
    ordered : Sequence[float]
        The sample, sorted ascending.
    probability : float
        Where in the sample to read, between zero and one.

    Returns
    -------
    float
        The interpolated order statistic.
    """
    last = len(ordered) - 1

    if last == 0:
        return ordered[0]

    position = last * probability
    below = floor(position)
    above = min(below + 1, last)

    return ordered[below] + (position - below) * (ordered[above] - ordered[below])


def _percentile_interval(ordered: Sequence[float], level: float) -> Interval:
    """Return the interval the ordered replicates give directly.

    Parameters
    ----------
    ordered : Sequence[float]
        The replicate statistics, sorted ascending.
    level : float
        The coverage to aim at, as a proportion.

    Returns
    -------
    Interval
        The two quantiles leaving equal mass in each tail.
    """
    outside = (1.0 - level) / 2.0

    return Interval(
        _quantile(ordered, outside), _quantile(ordered, 1.0 - outside), level
    )


def _bias(ordered: Sequence[float], observed: float) -> float | None:
    """Return bootstrap bias in standard normal units.

    Parameters
    ----------
    ordered : Sequence[float]
        The replicate statistics, sorted ascending.
    observed : float
        The statistic on the data as it came in.

    Returns
    -------
    float | None
        The bias correction, or None where every replicate lies on one side of the
        observation and the correction is thus off the scale.
    """
    proportion = sum(1 for draw in ordered if draw < observed) / len(ordered)

    if not 0.0 < proportion < 1.0:
        return None

    return _STANDARD_NORMAL.inv_cdf(proportion)


def _acceleration(jackknife: Sequence[float]) -> float:
    """Return the skewness the delete-one jackknife finds in the statistic.

    Parameters
    ----------
    jackknife : Sequence[float]
        The statistic recomputed with each unit left out in turn.

    Returns
    -------
    float
        The acceleration, which is zero for a statistic the jackknife finds constant,
        since nothing then argues for moving the quantiles.
    """
    center = fmean(jackknife)
    deviations = [center - value for value in jackknife]
    spread = fsum(deviation * deviation for deviation in deviations)

    if spread <= 0.0:
        return 0.0

    return fsum(deviation**3 for deviation in deviations) / (6.0 * spread**1.5)


def _moved(probability: float, bias: float, acceleration: float) -> float | None:
    """Adjust a quantile for bias and acceleration.

    Parameters
    ----------
    probability : float
        The quantile the percentile interval would have read.
    bias : float
        How far the replicates sit off the observation, in normal units.
    acceleration : float
        The skewness the jackknife found.

    Returns
    -------
    float | None
        The quantile to read instead, or None where the correction sends it off the
        scale.
    """
    point = bias + _STANDARD_NORMAL.inv_cdf(probability)
    denominator = 1.0 - acceleration * point

    if denominator <= 0.0:
        return None

    return _STANDARD_NORMAL.cdf(bias + point / denominator)


def _accelerated_interval(
    ordered: Sequence[float],
    observed: float,
    jackknife: Sequence[float],
    level: float,
) -> Interval:
    """Return the bias-corrected and accelerated interval.

    Parameters
    ----------
    ordered : Sequence[float]
        The replicate statistics, sorted ascending.
    observed : float
        The statistic on the data as it came in.
    jackknife : Sequence[float]
        The statistic recomputed with each unit left out in turn.
    level : float
        The coverage to aim at, as a proportion.

    Returns
    -------
    Interval
        The moved quantiles, or the percentile interval where the correction is
        undefined.
    """
    bias = _bias(ordered, observed)

    if bias is None:
        return _percentile_interval(ordered, level)

    outside = (1.0 - level) / 2.0
    acceleration = _acceleration(jackknife)
    low = _moved(outside, bias, acceleration)
    high = _moved(1.0 - outside, bias, acceleration)

    if low is None or high is None or low > high:
        return _percentile_interval(ordered, level)

    return Interval(_quantile(ordered, low), _quantile(ordered, high), level)


class Bootstrap:
    """A nonparametric bootstrap over a declared unit.

    Parameters
    ----------
    unit : str
        What one resampled unit is: a sentence, a participant, an item.
    replicates : int
        How many resamples to draw.
    seed : int
        The seed the draw runs under, so a published interval can be recomputed.
    level : float
        The coverage the intervals aim at, as a proportion.
    groups : Sequence[str] | None
        The unit each sample belongs to, in the samples' own order, or None where each
        sample is a unit on its own.
    strata : Sequence[str] | None
        The stratum each sample belongs to, in the samples' own order. A stratified
        bootstrap independently resamples the observed number of samples within each
        stratum.

    Raises
    ------
    TypeError
        If a field is given a value of the wrong type.
    ValueError
        If the unit's name, a group name, or a stratum name is empty, if groups and
        strata are both given, if the replicate count is zero, if the seed is negative,
        or if the level does not lie strictly inside the unit interval.
    """

    __slots__ = ("_groups", "_level", "_replicates", "_seed", "_strata", "_unit")

    def __init__(
        self,
        *,
        unit: str,
        replicates: int,
        seed: int,
        level: float,
        groups: Sequence[str] | None = None,
        strata: Sequence[str] | None = None,
    ) -> None:
        self._unit = ensure_name("a resampling unit", unit)
        self._replicates = ensure_non_negative("a replicate budget", replicates)
        self._seed = ensure_non_negative("a bootstrap seed", seed)
        self._level = ensure_finite("a coverage level", level)
        self._groups = (
            None
            if groups is None
            else tuple(ensure_name(f"a {unit} label", label) for label in groups)
        )
        self._strata = (
            None
            if strata is None
            else tuple(ensure_name(f"a {unit} stratum", label) for label in strata)
        )

        if self._groups is not None and self._strata is not None:
            msg = f"a bootstrap over {unit!r} cannot declare both groups and strata"
            raise ValueError(msg)

        if self._replicates == 0:
            msg = f"a bootstrap over {unit!r} was given no replicate budget"
            raise ValueError(msg)

        if not 0.0 < self._level < 1.0:
            msg = f"a coverage level is not a proportion: {level!r}"
            raise ValueError(msg)

    @property
    def unit(self) -> str:
        """Return the resampling unit."""
        return self._unit

    @property
    def replicates(self) -> int:
        """Return how many resamples the bootstrap draws."""
        return self._replicates

    @property
    def seed(self) -> int:
        """Return the sampling seed."""
        return self._seed

    @property
    def level(self) -> float:
        """Return the coverage the intervals aim at."""
        return self._level

    @property
    def groups(self) -> tuple[str, ...] | None:
        """Return the unit each sample belongs to, absent where a sample is a unit."""
        return self._groups

    @property
    def strata(self) -> tuple[str, ...] | None:
        """Return the stratum each sample belongs to, if sampling is stratified."""
        return self._strata

    def _blocks[Unit](self, samples: Sequence[Unit]) -> tuple[tuple[Unit, ...], ...]:
        """Return the samples gathered into the units that are resampled.

        Parameters
        ----------
        samples : Sequence[Unit]
            The data, in the order the grouping labels it.

        Returns
        -------
        tuple[tuple[Unit, ...], ...]
            One block per unit, in first-appearance order, holding that unit's
            samples in their original order.

        Raises
        ------
        ValueError
            If the grouping or stratification does not label every sample exactly
            once, a stratum has fewer than two units, or fewer than two units stand
            behind the data.
        """
        if self._groups is None:
            gathered = tuple((sample,) for sample in samples)

        else:
            if len(self._groups) != len(samples):
                msg = (
                    f"a bootstrap over {self._unit!r} was given "
                    f"{len(self._groups)} labels for {len(samples)} samples"
                )
                raise ValueError(msg)

            members: dict[str, list[Unit]] = {}

            for label, sample in zip(self._groups, samples, strict=True):
                members.setdefault(label, []).append(sample)

            gathered = tuple(tuple(held) for held in members.values())

        if len(gathered) < _MINIMUM_UNITS:
            msg = (
                f"a bootstrap needs at least {_MINIMUM_UNITS} {self._unit!r} and "
                f"was given {len(gathered)}"
            )
            raise ValueError(msg)

        if self._strata is not None:
            if len(self._strata) != len(gathered):
                msg = (
                    f"a bootstrap over {self._unit!r} was given "
                    f"{len(self._strata)} strata for {len(gathered)} samples"
                )
                raise ValueError(msg)

            counts: dict[str, int] = {}
            for label in self._strata:
                counts[label] = counts.get(label, 0) + 1

            for label, count in counts.items():
                if count < _MINIMUM_UNITS:
                    msg = (
                        f"stratum {label!r} contains {count} {self._unit!r}; "
                        f"the accelerated interval needs at least {_MINIMUM_UNITS}"
                    )
                    raise ValueError(msg)

        return gathered

    def _draws[Unit](
        self,
        blocks: Sequence[tuple[Unit, ...]],
        statistic: Statistic[Sequence[Unit]],
    ) -> tuple[float, ...]:
        """Return the statistic over each resample, in the order drawn.

        Parameters
        ----------
        blocks : Sequence[tuple[Unit, ...]]
            One block per unit.
        statistic : Statistic[Sequence[Unit]]
            The summary recomputed over each resample.

        Returns
        -------
        tuple[float, ...]
            One statistic per replicate.

        Raises
        ------
        TypeError
            If the statistic returns something that is not a number.
        ValueError
            If the statistic returns a value that is not finite.
        """
        rng = Random(self._seed)
        units = len(blocks)

        if self._strata is not None:
            members: dict[str, list[Unit]] = {}

            for label, block in zip(self._strata, blocks, strict=True):
                members.setdefault(label, []).append(block[0])

            stratified = tuple(tuple(held) for held in members.values())

            return tuple(
                ensure_finite(
                    "a bootstrap replicate",
                    statistic(
                        [
                            sample
                            for stratum in stratified
                            for sample in rng.choices(stratum, k=len(stratum))
                        ]
                    ),
                )
                for _ in range(self._replicates)
            )

        if self._groups is None:
            flat = [block[0] for block in blocks]

            return tuple(
                ensure_finite(
                    "a bootstrap replicate", statistic(rng.choices(flat, k=units))
                )
                for _ in range(self._replicates)
            )

        return tuple(
            ensure_finite(
                "a bootstrap replicate",
                statistic(
                    [
                        sample
                        for block in rng.choices(blocks, k=units)
                        for sample in block
                    ]
                ),
            )
            for _ in range(self._replicates)
        )

    def _jackknife[Unit](
        self,
        blocks: Sequence[tuple[Unit, ...]],
        statistic: Statistic[Sequence[Unit]],
    ) -> tuple[float, ...]:
        """Return the statistic with each unit left out in turn.

        Parameters
        ----------
        blocks : Sequence[tuple[Unit, ...]]
            One block per unit.
        statistic : Statistic[Sequence[Unit]]
            The summary recomputed over each reduced sample.

        Returns
        -------
        tuple[float, ...]
            One statistic per unit left out, in the units' order.

        Raises
        ------
        TypeError
            If the statistic returns something that is not a number.
        ValueError
            If the statistic returns a value that is not finite.
        """
        return tuple(
            ensure_finite(
                "a jackknife statistic",
                statistic(
                    [
                        sample
                        for other, block in enumerate(blocks)
                        if other != left_out
                        for sample in block
                    ]
                ),
            )
            for left_out in range(len(blocks))
        )

    def interval[Unit](
        self, samples: Sequence[Unit], statistic: Statistic[Sequence[Unit]], /
    ) -> BootstrapResult:
        """Return the bootstrap intervals around a statistic of these samples.

        Parameters
        ----------
        samples : Sequence[Unit]
            The data, in the order any grouping labels it.
        statistic : Statistic[Sequence[Unit]]
            The summary being estimated.

        Returns
        -------
        BootstrapResult
            The observed statistic, unit data, sampling configuration, and
            percentile and accelerated intervals.

        Raises
        ------
        TypeError
            If the statistic returns something that is not a number.
        ValueError
            If the statistic returns a value that is not finite, if a grouping or
            stratification does not label every sample exactly once, if a stratum has
            fewer than two units, or if fewer than two units stand behind the data.
        """
        data = tuple(samples)
        blocks = self._blocks(data)
        observed = ensure_finite("an observed statistic", statistic(data))
        ordered = sorted(self._draws(blocks, statistic))
        jackknife = self._jackknife(blocks, statistic)

        return BootstrapResult(
            observed,
            self._unit,
            units=len(blocks),
            replicates=self._replicates,
            seed=self._seed,
            percentile=_percentile_interval(ordered, self._level),
            bca=_accelerated_interval(ordered, observed, jackknife, self._level),
        )


class CrossedBootstrap:
    """A two-way cluster bootstrap over two crossed resampling units.

    Every resample draws the clusters of both units independently and with
    replacement, as many of each as were observed, and holds each sample once
    for every pairing of a draw of its row cluster with a draw of its column
    cluster. A sample whose row cluster is drawn twice and whose column
    cluster is drawn three times therefore appears six times. The accelerated
    interval's acceleration is read off a jackknife that leaves out each row
    cluster and each column cluster in turn.

    Parameters
    ----------
    units : tuple[str, str]
        What one row cluster and one column cluster are, such as a participant
        and an item.
    rows : Sequence[str]
        The row cluster each sample belongs to, in the samples' own order.
    columns : Sequence[str]
        The column cluster each sample belongs to, in the samples' own order.
    replicates : int
        How many resamples to draw.
    seed : int
        The seed the draw runs under, so a published interval can be recomputed.
    level : float
        The coverage the intervals aim at, as a proportion.

    Raises
    ------
    TypeError
        If a field is given a value of the wrong type.
    ValueError
        If a unit's name or a cluster label is empty, if both units have one
        name, if the replicate count is zero, if the seed is negative, or if the
        level does not lie strictly inside the unit interval.
    """

    __slots__ = ("_columns", "_level", "_replicates", "_rows", "_seed", "_units")

    def __init__(
        self,
        *,
        units: tuple[str, str],
        rows: Sequence[str],
        columns: Sequence[str],
        replicates: int,
        seed: int,
        level: float,
    ) -> None:
        row, column = units
        self._units = (
            ensure_name("a resampling unit", row),
            ensure_name("a resampling unit", column),
        )
        self._rows = tuple(ensure_name(f"a {row} label", label) for label in rows)
        self._columns = tuple(
            ensure_name(f"a {column} label", label) for label in columns
        )
        self._replicates = ensure_non_negative("a replicate budget", replicates)
        self._seed = ensure_non_negative("a bootstrap seed", seed)
        self._level = ensure_finite("a coverage level", level)

        if row == column:
            msg = f"a crossed bootstrap resamples {row!r} on both sides"
            raise ValueError(msg)

        if self._replicates == 0:
            msg = (
                f"a bootstrap over {row!r} and {column!r} was given no replicate budget"
            )
            raise ValueError(msg)

        if not 0.0 < self._level < 1.0:
            msg = f"a coverage level is not a proportion: {level!r}"
            raise ValueError(msg)

    @property
    def units(self) -> tuple[str, str]:
        """Return the row and column resampling units."""
        return self._units

    @property
    def unit(self) -> str:
        """Return the name of the crossed resampling unit."""
        return "-by-".join(self._units)

    @property
    def replicates(self) -> int:
        """Return how many resamples the bootstrap draws."""
        return self._replicates

    @property
    def seed(self) -> int:
        """Return the sampling seed."""
        return self._seed

    @property
    def level(self) -> float:
        """Return the coverage the intervals aim at."""
        return self._level

    def _cells[Unit](
        self, samples: Sequence[Unit]
    ) -> tuple[tuple[str, ...], tuple[str, ...], dict[tuple[str, str], list[Unit]]]:
        """Return both units' clusters and the samples of each observed cell.

        Parameters
        ----------
        samples : Sequence[Unit]
            The data, in the order both labelings follow.

        Returns
        -------
        tuple[tuple[str, ...], tuple[str, ...], dict[tuple[str, str], list[Unit]]]
            The row clusters and column clusters in first-appearance order,
            and each observed row and column pairing's samples in their
            original order.

        Raises
        ------
        ValueError
            If either labeling does not label every sample exactly once, or
            either unit has fewer than two clusters.
        """
        row, column = self._units

        if not len(self._rows) == len(self._columns) == len(samples):
            msg = (
                f"a crossed bootstrap was given {len(self._rows)} {row} labels and "
                f"{len(self._columns)} {column} labels for {len(samples)} samples"
            )
            raise ValueError(msg)

        cells: dict[tuple[str, str], list[Unit]] = {}

        for key, sample in zip(
            zip(self._rows, self._columns, strict=True), samples, strict=True
        ):
            cells.setdefault(key, []).append(sample)

        clusters = (
            tuple(dict.fromkeys(self._rows)),
            tuple(dict.fromkeys(self._columns)),
        )

        for name, observed in zip(self._units, clusters, strict=True):
            if len(observed) < _MINIMUM_UNITS:
                msg = (
                    f"a bootstrap needs at least {_MINIMUM_UNITS} {name!r} and "
                    f"was given {len(observed)}"
                )
                raise ValueError(msg)

        return clusters[0], clusters[1], cells

    def _draws[Unit](
        self,
        rows: Sequence[str],
        columns: Sequence[str],
        cells: dict[tuple[str, str], list[Unit]],
        statistic: Statistic[Sequence[Unit]],
    ) -> tuple[float, ...]:
        """Return the statistic over each crossed resample, in the order drawn.

        Parameters
        ----------
        rows : Sequence[str]
            The observed row clusters.
        columns : Sequence[str]
            The observed column clusters.
        cells : dict[tuple[str, str], list[Unit]]
            The samples of each observed row and column pairing.
        statistic : Statistic[Sequence[Unit]]
            The summary recomputed over each resample.

        Returns
        -------
        tuple[float, ...]
            One statistic per replicate.

        Raises
        ------
        TypeError
            If the statistic returns something that is not a number.
        ValueError
            If the statistic returns a value that is not finite, or a resample
            pairs no drawn row cluster with a drawn column cluster it was
            observed with.
        """
        rng = Random(self._seed)
        statistics: list[float] = []

        for _ in range(self._replicates):
            drawn_rows = rng.choices(rows, k=len(rows))
            drawn_columns = rng.choices(columns, k=len(columns))
            resample = [
                sample
                for row in drawn_rows
                for column in drawn_columns
                for sample in cells.get((row, column), ())
            ]

            if not resample:
                msg = (
                    f"a crossed resample of {self.unit!r} drew no "
                    f"{self._units[0]} and {self._units[1]} observed together"
                )
                raise ValueError(msg)

            statistics.append(
                ensure_finite("a bootstrap replicate", statistic(resample))
            )

        return tuple(statistics)

    def _jackknife[Unit](
        self,
        samples: Sequence[Unit],
        rows: Sequence[str],
        columns: Sequence[str],
        statistic: Statistic[Sequence[Unit]],
    ) -> tuple[float, ...]:
        """Return the statistic with each row and each column cluster left out.

        Parameters
        ----------
        samples : Sequence[Unit]
            The data, in the order both labelings follow.
        rows : Sequence[str]
            The observed row clusters.
        columns : Sequence[str]
            The observed column clusters.
        statistic : Statistic[Sequence[Unit]]
            The summary recomputed over each reduced sample.

        Returns
        -------
        tuple[float, ...]
            One statistic per row cluster left out, then one per column cluster.

        Raises
        ------
        TypeError
            If the statistic returns something that is not a number.
        ValueError
            If the statistic returns a value that is not finite.
        """
        labeled = tuple(zip(self._rows, self._columns, samples, strict=True))
        without_row = (
            [sample for row, _, sample in labeled if row != left_out]
            for left_out in rows
        )
        without_column = (
            [sample for _, column, sample in labeled if column != left_out]
            for left_out in columns
        )

        return tuple(
            ensure_finite("a jackknife statistic", statistic(reduced))
            for reduced in (*without_row, *without_column)
        )

    def interval[Unit](
        self, samples: Sequence[Unit], statistic: Statistic[Sequence[Unit]], /
    ) -> BootstrapResult:
        """Return the crossed bootstrap intervals around a statistic of these samples.

        Parameters
        ----------
        samples : Sequence[Unit]
            The data, in the order both labelings follow.
        statistic : Statistic[Sequence[Unit]]
            The summary being estimated.

        Returns
        -------
        BootstrapResult
            The observed statistic, the crossed unit and its count of observed
            row and column pairings, the sampling configuration, and the
            percentile and accelerated intervals.

        Raises
        ------
        TypeError
            If the statistic returns something that is not a number.
        ValueError
            If the statistic returns a value that is not finite, if either
            labeling does not label every sample exactly once, if either unit
            has fewer than two clusters, or if a resample pairs no drawn
            clusters that were observed together.
        """
        data = tuple(samples)
        rows, columns, cells = self._cells(data)
        observed = ensure_finite("an observed statistic", statistic(data))
        ordered = sorted(self._draws(rows, columns, cells, statistic))
        jackknife = self._jackknife(data, rows, columns, statistic)

        return BootstrapResult(
            observed,
            self.unit,
            units=len(cells),
            replicates=self._replicates,
            seed=self._seed,
            percentile=_percentile_interval(ordered, self._level),
            bca=_accelerated_interval(ordered, observed, jackknife, self._level),
        )


def paired_bootstrap[Unit](
    samples: Sequence[Unit],
    statistic: Statistic[Sequence[Unit]],
    first: Bootstrap,
    second: Bootstrap,
    /,
) -> tuple[BootstrapResult, BootstrapResult]:
    """Return one quantity's intervals at two declared resampling units.

    Parameters
    ----------
    samples : Sequence[Unit]
        The data, in the order both groupings label it.
    statistic : Statistic[Sequence[Unit]]
        The summary being estimated, applied unchanged under both units.
    first : Bootstrap
        The bootstrap over one unit.
    second : Bootstrap
        The bootstrap over the other.

    Returns
    -------
    tuple[BootstrapResult, BootstrapResult]
        The two results, in the order the two bootstraps were given.

    Raises
    ------
    TypeError
        If either bootstrap is not one.
    ValueError
        If the two name the same unit, or aim at different coverage, or draw different
        numbers of replicates, since a pair of intervals that differ in any of those is
        not two readings of one quantity.
    """
    for side, given in (("first", first), ("second", second)):
        if not isinstance(given, Bootstrap):
            msg = f"the {side} bootstrap of a pair is not one: {given!r}"
            raise TypeError(msg)

    if first.unit == second.unit:
        msg = f"a paired bootstrap resamples {first.unit!r} on both sides"
        raise ValueError(msg)

    if first.level != second.level:
        msg = (
            f"a paired bootstrap aims at {first.level!r} coverage on one side "
            f"and {second.level!r} on the other"
        )
        raise ValueError(msg)

    if first.replicates != second.replicates:
        msg = (
            f"a paired bootstrap draws {first.replicates} replicates on one "
            f"side and {second.replicates} on the other"
        )
        raise ValueError(msg)

    return (first.interval(samples, statistic), second.interval(samples, statistic))
