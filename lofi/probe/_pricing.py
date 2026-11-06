"""Measured probe unit prices and sweep-grid cost estimates.

Unit prices use synchronized timed repeats after warmup. Grid estimates include all
declared units and reject units without a measured price.
"""

from enum import Enum
from math import isfinite, prod
from statistics import fmean
from time import perf_counter
from types import MappingProxyType
from typing import TYPE_CHECKING, override

import torch

from lofi import _checks
from lofi._checks import ensure_name, ensure_non_negative
from lofi.nn.policy import DeviceKind

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from lofi.nn.policy import DevicePolicy

__all__ = [
    "CostModel",
    "GridSpec",
    "ProbeUnit",
    "UnitPrice",
    "UnpricedUnitError",
    "price_unit",
]


class ProbeUnit(Enum):
    """One unit of work a sweep pays for."""

    #: One fixed-rank alignment fitted to the pairs of one cell.
    FIXED_ALIGNMENT_FIT = "fixed_alignment_fit"

    #: One fixed-rank refit under exchanged decisions, assessed.
    FIXED_PERMUTATION_REFIT = "fixed_permutation_refit"

    #: One boundless alignment fitted to the pairs of one cell.
    BOUNDLESS_ALIGNMENT_FIT = "boundless_alignment_fit"

    #: One boundless refit under exchanged decisions, assessed.
    BOUNDLESS_PERMUTATION_REFIT = "boundless_permutation_refit"

    #: One pass of the attribution estimator over a computation graph.
    ATTRIBUTION = "attribution"

    #: One pruning run over a computation graph.
    PRUNING = "pruning"


class UnpricedUnitError(Exception):
    """Reports units a grid needs that no measured price covers.

    Parameters
    ----------
    unpriced : tuple[ProbeUnit, ...]
        The units the grid needs and the model lacks, in declaration order.
    priced : tuple[ProbeUnit, ...]
        The units the model holds a price for, in declaration order.
    """

    def __init__(
        self, unpriced: tuple[ProbeUnit, ...], priced: tuple[ProbeUnit, ...]
    ) -> None:
        self.unpriced = unpriced
        self.priced = priced
        needed = ", ".join(unit.value for unit in unpriced)
        held = ", ".join(unit.value for unit in priced) or "nothing"

        super().__init__(
            f"refusing to cost a grid needing {needed}, which is unpriced "
            f"(priced: {held})"
        )


def _seconds(field: str, value: float) -> float:
    """Return a positive, finite duration.

    Parameters
    ----------
    field : str
        The name to report the value under.
    value : float
        The duration in seconds.

    Raises
    ------
    TypeError
        If the value is not a number.
    ValueError
        If it is not positive and finite.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"{field} is not a number: {value!r}"
        raise TypeError(msg)

    if not isfinite(value) or value <= 0.0:
        msg = f"{field} is not a positive duration: {value!r}"
        raise ValueError(msg)

    return float(value)


class UnitPrice:
    """What one unit of work cost when it was timed.

    Parameters
    ----------
    unit : ProbeUnit
        The unit that was timed.
    seconds_per_run : float
        The mean wall seconds of one timed run.
    seconds_fastest : float
        The wall seconds of the fastest timed run.
    warmup : int
        How many untimed runs preceded the timed ones.
    repeats : int
        How many runs were timed.
    device : str
        The device the unit ran on, index included.
    peak_allocated_bytes : int | None
        The most bytes the accelerator allocator held during the timed runs, or
        None on a device with no allocator to read.

    Raises
    ------
    TypeError
        If a duration is not a number, a count is not an integer, or the unit is
        not a probe unit.
    ValueError
        If a duration is not positive and finite, the fastest run is slower than
        the mean, fewer than one run was timed, the warmup is negative, the
        device is unnamed, or the peak is negative.
    """

    __slots__ = (
        "_device",
        "_peak_allocated_bytes",
        "_repeats",
        "_seconds_fastest",
        "_seconds_per_run",
        "_unit",
        "_warmup",
    )

    def __init__(
        self,
        unit: ProbeUnit,
        *,
        seconds_per_run: float,
        seconds_fastest: float,
        warmup: int,
        repeats: int,
        device: str,
        peak_allocated_bytes: int | None = None,
    ) -> None:
        if not isinstance(unit, ProbeUnit):
            msg = f"a price is for a ProbeUnit, not {unit!r}"
            raise TypeError(msg)

        self._unit = unit
        self._seconds_per_run = _seconds("a mean run time", seconds_per_run)
        self._seconds_fastest = _seconds("a fastest run time", seconds_fastest)

        if self._seconds_fastest > self._seconds_per_run:
            msg = (
                f"the fastest run took {seconds_fastest!r} seconds against a mean "
                f"of {seconds_per_run!r}"
            )
            raise ValueError(msg)

        self._warmup = ensure_non_negative("a warmup count", warmup)
        self._repeats = ensure_non_negative("a repeat count", repeats)

        if self._repeats == 0:
            msg = "a price is read off at least one timed run"
            raise ValueError(msg)

        self._device = ensure_name("a price's device", device)
        self._peak_allocated_bytes = (
            None
            if peak_allocated_bytes is None
            else ensure_non_negative("a peak allocation", peak_allocated_bytes)
        )

    @property
    def unit(self) -> ProbeUnit:
        """Return the unit that was timed."""
        return self._unit

    @property
    def seconds_per_run(self) -> float:
        """Return the mean wall seconds of one timed run."""
        return self._seconds_per_run

    @property
    def seconds_fastest(self) -> float:
        """Return the wall seconds of the fastest timed run."""
        return self._seconds_fastest

    @property
    def warmup(self) -> int:
        """Return how many untimed runs preceded the timed ones."""
        return self._warmup

    @property
    def repeats(self) -> int:
        """Return how many runs were timed."""
        return self._repeats

    @property
    def device(self) -> str:
        """Return the device the unit ran on."""
        return self._device

    @property
    def peak_allocated_bytes(self) -> int | None:
        """Return the allocator's peak over the timed runs, or None without one."""
        return self._peak_allocated_bytes

    def _identity(
        self,
    ) -> tuple[ProbeUnit, float, float, int, int, str, int | None]:
        """Return the seven fields a price is equal on."""
        return (
            self._unit,
            self._seconds_per_run,
            self._seconds_fastest,
            self._warmup,
            self._repeats,
            self._device,
            self._peak_allocated_bytes,
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value records the same measurement."""
        if type(other) is not UnitPrice:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the seven fields."""
        return hash((UnitPrice, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this price."""
        return (
            f"UnitPrice({self._unit!r}, seconds_per_run={self._seconds_per_run!r}, "
            f"seconds_fastest={self._seconds_fastest!r}, warmup={self._warmup!r}, "
            f"repeats={self._repeats!r}, device={self._device!r}, "
            f"peak_allocated_bytes={self._peak_allocated_bytes!r})"
        )


def _synchronize(policy: DevicePolicy) -> None:
    """Wait for the device's queued work, so a clock read closes the run.

    Parameters
    ----------
    policy : DevicePolicy
        The device policy.
    """
    if policy.kind is not DeviceKind.CPU:
        torch.accelerator.synchronize(policy.device)


def _reset_peak(policy: DevicePolicy) -> None:
    """Start the allocator's peak statistic afresh on a device that keeps one.

    Parameters
    ----------
    policy : DevicePolicy
        The device policy.
    """
    if policy.kind is not DeviceKind.CPU:
        torch.accelerator.reset_peak_memory_stats(policy.device)


def _peak(policy: DevicePolicy) -> int | None:
    """Return the allocator's peak allocation, or None where there is none to read.

    Parameters
    ----------
    policy : DevicePolicy
        The device policy.
    """
    if policy.kind is DeviceKind.CPU:
        return None

    return torch.accelerator.max_memory_allocated(policy.device)


def price_unit[R](
    unit: ProbeUnit,
    run: Callable[[], R],
    *,
    policy: DevicePolicy,
    warmup: int = 1,
    repeats: int = 5,
) -> UnitPrice:
    """Time one unit of work and read its cost off the timed runs.

    Parameters
    ----------
    unit : ProbeUnit
        The unit the body performs.
    run : Callable[[], R]
        The body, already bound to its inputs on the policy's device.
    policy : DevicePolicy
        Where the body's tensors live.
    warmup : int
        How many runs to make untimed before the first timed one.
    repeats : int
        How many runs to time.

    Returns
    -------
    UnitPrice
        The cost read off the timed runs.

    Raises
    ------
    ValueError
        If the policy names a meta device, on which nothing runs, or if fewer
        than one run would be timed or a negative number warmed up.
    RuntimeError
        If the debug checks are on, since a checked build times the checks.
    """
    if policy.kind is DeviceKind.META:
        msg = "a meta device computes nothing, so a unit on it has no price"
        raise ValueError(msg)

    if repeats < 1:
        msg = f"a price is read off at least one timed run, not {repeats}"
        raise ValueError(msg)

    if warmup < 0:
        msg = f"a warmup count is not negative: {warmup}"
        raise ValueError(msg)

    if _checks.CHECKS:
        msg = (
            "LOFI_CHECK is on, so a unit would be timed with the debug checks in "
            "it; price with LOFI_CHECK=0"
        )
        raise RuntimeError(msg)

    for _ in range(warmup):
        run()

    seconds: list[float] = []
    _synchronize(policy)
    _reset_peak(policy)

    for _ in range(repeats):
        start = perf_counter()
        run()
        _synchronize(policy)
        seconds.append(perf_counter() - start)

    return UnitPrice(
        unit,
        seconds_per_run=fmean(seconds),
        seconds_fastest=min(seconds),
        warmup=warmup,
        repeats=repeats,
        device=str(policy.device),
        peak_allocated_bytes=_peak(policy),
    )


class GridSpec:
    """The shape of a sweep: its cells and what each cell and the whole pay for.

    Parameters
    ----------
    axes : Iterable[tuple[str, int]]
        The name and size of each axis the grid crosses, in declaration order.
        The cells are the product of the sizes, one cell for a grid of no axes.
    replicates : int
        How many permutation refits each cell's alignment is assessed under.
    fit_unit : ProbeUnit
        The measured fitting unit used once per cell.
    refit_unit : ProbeUnit
        The measured refitting unit used once per replicate per cell.
    attribution_runs : int
        How many attribution passes the sweep adds.
    pruning_runs : int
        How many pruning runs the sweep adds.

    Raises
    ------
    TypeError
        If a size or a count is not an integer, or an axis name is not a string.
    ValueError
        If an axis is unnamed or named twice, a size is zero or negative, or a
        count is negative.
    """

    __slots__ = (
        "_attribution_runs",
        "_axes",
        "_fit_unit",
        "_pruning_runs",
        "_refit_unit",
        "_replicates",
    )

    def __init__(
        self,
        axes: Iterable[tuple[str, int]] = (),
        *,
        fit_unit: ProbeUnit,
        refit_unit: ProbeUnit,
        replicates: int = 0,
        attribution_runs: int = 0,
        pruning_runs: int = 0,
    ) -> None:
        held = tuple(
            (
                ensure_name("an axis name", name),
                ensure_non_negative(f"axis {name!r}", size),
            )
            for name, size in axes
        )

        for name, size in held:
            if size == 0:
                msg = f"axis {name!r} has no value, so the grid has no cell"
                raise ValueError(msg)

        names = [name for name, _ in held]

        if len(set(names)) != len(names):
            msg = f"an axis is named twice: {names}"
            raise ValueError(msg)

        pairs = {
            (
                ProbeUnit.FIXED_ALIGNMENT_FIT,
                ProbeUnit.FIXED_PERMUTATION_REFIT,
            ),
            (
                ProbeUnit.BOUNDLESS_ALIGNMENT_FIT,
                ProbeUnit.BOUNDLESS_PERMUTATION_REFIT,
            ),
        }

        if not isinstance(fit_unit, ProbeUnit):
            msg = f"a grid's fit unit is not a ProbeUnit: {fit_unit!r}"
            raise TypeError(msg)

        if not isinstance(refit_unit, ProbeUnit):
            msg = f"a grid's refit unit is not a ProbeUnit: {refit_unit!r}"
            raise TypeError(msg)

        if (fit_unit, refit_unit) not in pairs:
            msg = (
                f"fit unit {fit_unit.value!r} and refit unit {refit_unit.value!r} "
                "do not price one alignment strategy"
            )
            raise ValueError(msg)

        self._axes = held
        self._fit_unit = fit_unit
        self._refit_unit = refit_unit
        self._replicates = ensure_non_negative("a replicate count", replicates)
        self._attribution_runs = ensure_non_negative(
            "an attribution run count", attribution_runs
        )
        self._pruning_runs = ensure_non_negative("a pruning run count", pruning_runs)

    @property
    def axes(self) -> tuple[tuple[str, int], ...]:
        """Return the name and size of each axis, in declaration order."""
        return self._axes

    @property
    def cells(self) -> int:
        """Return how many cells the axes cross to."""
        return prod(size for _, size in self._axes)

    @property
    def replicates(self) -> int:
        """Return how many permutation refits each cell is assessed under."""
        return self._replicates

    @property
    def fit_unit(self) -> ProbeUnit:
        """Return the fitting unit paid once per cell."""
        return self._fit_unit

    @property
    def refit_unit(self) -> ProbeUnit:
        """Return the refitting unit paid once per replicate per cell."""
        return self._refit_unit

    @property
    def attribution_runs(self) -> int:
        """Return how many attribution passes the sweep adds."""
        return self._attribution_runs

    @property
    def pruning_runs(self) -> int:
        """Return how many pruning runs the sweep adds."""
        return self._pruning_runs

    def demand(self) -> Mapping[ProbeUnit, int]:
        """Return how many runs of each unit the grid needs, none of a unit left out."""
        counts = {
            self._fit_unit: self.cells,
            self._refit_unit: self.cells * self._replicates,
            ProbeUnit.ATTRIBUTION: self._attribution_runs,
            ProbeUnit.PRUNING: self._pruning_runs,
        }

        return MappingProxyType(
            {unit: count for unit, count in counts.items() if count}
        )

    def _identity(
        self,
    ) -> tuple[tuple[tuple[str, int], ...], ProbeUnit, ProbeUnit, int, int, int]:
        """Return the six fields a grid is equal on."""
        return (
            self._axes,
            self._fit_unit,
            self._refit_unit,
            self._replicates,
            self._attribution_runs,
            self._pruning_runs,
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value describes the same sweep."""
        if type(other) is not GridSpec:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the axes and the three counts."""
        return hash((GridSpec, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this grid."""
        return (
            f"GridSpec({self._axes!r}, fit_unit={self._fit_unit!r}, "
            f"refit_unit={self._refit_unit!r}, replicates={self._replicates!r}, "
            f"attribution_runs={self._attribution_runs!r}, "
            f"pruning_runs={self._pruning_runs!r})"
        )


class CostModel:
    """The accelerator time a grid needs, from prices measured on one device.

    Parameters
    ----------
    prices : Iterable[UnitPrice]
        One measured price per unit.

    Raises
    ------
    ValueError
        If two prices are for one unit, or the prices were measured on different
        devices.
    """

    __slots__ = ("_device", "_prices")

    def __init__(self, prices: Iterable[UnitPrice]) -> None:
        held: dict[ProbeUnit, UnitPrice] = {}

        for price in prices:
            if price.unit in held:
                msg = (
                    f"{price.unit.value} is priced twice: {held[price.unit]!r}, "
                    f"{price!r}"
                )
                raise ValueError(msg)

            held[price.unit] = price

        devices = sorted({price.device for price in held.values()})

        if len(devices) > 1:
            msg = f"a cost model prices one device, not {devices}"
            raise ValueError(msg)

        self._prices = MappingProxyType(dict(held))
        self._device = devices[0] if devices else None

    @property
    def prices(self) -> Mapping[ProbeUnit, UnitPrice]:
        """Return the measured price of each priced unit."""
        return self._prices

    @property
    def priced(self) -> tuple[ProbeUnit, ...]:
        """Return the priced units, in declaration order."""
        return tuple(unit for unit in ProbeUnit if unit in self._prices)

    @property
    def device(self) -> str | None:
        """Return the device the prices were measured on, or None with no price."""
        return self._device

    def seconds(self, grid: GridSpec) -> float:
        """Return the wall seconds the grid needs, run serially on the priced device.

        Parameters
        ----------
        grid : GridSpec
            The sweep to cost.

        Raises
        ------
        UnpricedUnitError
            If the grid needs a unit no price covers, naming every such unit.
        """
        demand = grid.demand()
        unpriced = tuple(
            unit for unit in ProbeUnit if unit in demand and unit not in self._prices
        )

        if unpriced:
            raise UnpricedUnitError(unpriced, self.priced)

        return sum(
            count * self._prices[unit].seconds_per_run for unit, count in demand.items()
        )

    def hours(self, grid: GridSpec) -> float:
        """Return the accelerator hours the grid needs.

        Parameters
        ----------
        grid : GridSpec
            The sweep to cost.

        Raises
        ------
        UnpricedUnitError
            If the grid needs a unit no price covers, naming every such unit.
        """
        return self.seconds(grid) / 3600.0

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this model."""
        return f"CostModel({list(self._prices.values())!r})"
