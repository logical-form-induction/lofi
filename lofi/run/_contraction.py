"""Frozen sweeps and the contraction rungs that transform them.

A validated sweep is held below its boundary as an :class:`ExpandedSweep` of
immutable axes. A contraction rung maps one such sweep to a smaller one: it
drops axis values that realize no preregistered level, reduces two-level axes
to a resolution-IV fraction, drops languages or corpus scales in a declared
order, reduces every cell's seeds to a floor, or leaves the points alone and
switches the experiment to its fallback claim. A contracted sweep declares no
point count, since its count is the one its expansion produces.
"""

import itertools
import json
from math import prod
from typing import TYPE_CHECKING, NamedTuple, Protocol, final, override

from lofi._frozen import FrozenValue

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from lofi.run._boundary.sweep import AxisValue, SweepDesignSpec

__all__ = [
    "Contracted",
    "ContractionError",
    "DropInOrder",
    "ExpandedAxis",
    "ExpandedSweep",
    "Fallback",
    "PreregisteredCells",
    "ResolutionIVFraction",
    "Rung",
    "SeedFloor",
    "point_key",
]


class ContractionError(ValueError):
    """Signals a rung that cannot be applied to the sweep it is given."""


def point_key(point: AxisValue, /) -> str:
    """Return text that is equal for two axis points exactly when they agree.

    Parameters
    ----------
    point : AxisValue
        The point, with each array as a list.
    """
    return json.dumps(point, allow_nan=False, separators=(",", ":"), sort_keys=True)


class _AxisSequence(NamedTuple):
    """One recursively frozen sequence-valued axis point."""

    items: tuple[_FrozenAxisValue, ...]


class _AxisMapping(NamedTuple):
    """One recursively frozen mapping-valued axis point."""

    items: tuple[tuple[str, _FrozenAxisValue], ...]


type _FrozenAxisValue = str | int | float | bool | _AxisSequence | _AxisMapping | None


def _freeze_axis_value(value: AxisValue, /) -> _FrozenAxisValue:
    """Return an immutable recursive sweep value."""
    if isinstance(value, list):
        return _AxisSequence(tuple(_freeze_axis_value(item) for item in value))

    if isinstance(value, dict):
        return _AxisMapping(
            tuple(
                (key, _freeze_axis_value(item)) for key, item in sorted(value.items())
            )
        )

    return value


def _thaw_axis_value(value: _FrozenAxisValue, /) -> AxisValue:
    """Return a fresh boundary value from an immutable sweep value."""
    if isinstance(value, _AxisSequence):
        return [_thaw_axis_value(item) for item in value.items]

    if isinstance(value, _AxisMapping):
        return {key: _thaw_axis_value(item) for key, item in value.items}

    return value


@final
class ExpandedAxis(FrozenValue):
    """One validated sweep axis retained as immutable interior values.

    Parameters
    ----------
    key : str
        The configuration path or group slot the axis assigns.
    values : tuple[AxisValue, ...]
        The axis points in declared order.
    selects_group : bool
        Whether the axis selects a group fragment rather than assigning a leaf.
    """

    _noun = "expanded sweep axis"

    __slots__ = ("_key", "_selects_group", "_values")

    def __init__(
        self, key: str, values: tuple[AxisValue, ...], *, selects_group: bool
    ) -> None:
        object.__setattr__(self, "_key", key)
        object.__setattr__(self, "_selects_group", selects_group)
        object.__setattr__(
            self, "_values", tuple(_freeze_axis_value(value) for value in values)
        )

    @property
    def key(self) -> str:
        """Return the configuration path or group slot the axis assigns."""
        return self._key

    @property
    def selects_group(self) -> bool:
        """Return whether the axis selects a group rather than assigning a leaf."""
        return self._selects_group

    @property
    def slot(self) -> tuple[str, ...]:
        """Return the segments of the dotted path or group slot."""
        return tuple(self._key.split("."))

    @property
    def points(self) -> tuple[AxisValue, ...]:
        """Return fresh boundary values for expansion."""
        return tuple(_thaw_axis_value(value) for value in self._values)

    def with_points(self, points: tuple[AxisValue, ...], /) -> ExpandedAxis:
        """Return the axis assigning other points to the same key.

        Parameters
        ----------
        points : tuple[AxisValue, ...]
            The points the returned axis assigns.
        """
        return ExpandedAxis(self._key, points, selects_group=self._selects_group)

    @override
    def __eq__(self, other: object) -> bool:
        if type(other) is not ExpandedAxis:
            return NotImplemented

        return (self._key, self._selects_group, self._values) == (
            other._key,
            other._selects_group,
            other._values,
        )

    @override
    def __hash__(self) -> int:
        return hash((ExpandedAxis, self._key, self._selects_group, self._values))

    @override
    def __repr__(self) -> str:
        return f"ExpandedAxis({self._key!r}, {self.points!r})"


@final
class ExpandedSweep(FrozenValue):
    """A validated sweep retained as immutable interior values.

    Parameters
    ----------
    grid : tuple[ExpandedAxis, ...]
        The axes crossed with one another.
    yoked : tuple[tuple[ExpandedAxis, ...], ...]
        The groups of axes whose values advance together.
    clauses : tuple[Mapping[str, AxisValue], ...]
        The exclusion clauses, each keyed by axis key.
    seeds : tuple[int, ...]
        The seeds crossed with every surviving assignment.
    expected_points : int | None
        The point count the expansion must produce, or None to accept any.
    """

    _noun = "expanded sweep"

    __slots__ = (
        "_clauses",
        "_design",
        "_expected_points",
        "_grid",
        "_seeds",
        "_yoked",
    )

    def __init__(
        self,
        *,
        design: SweepDesignSpec,
        grid: tuple[ExpandedAxis, ...],
        yoked: tuple[tuple[ExpandedAxis, ...], ...],
        clauses: tuple[Mapping[str, AxisValue], ...],
        seeds: tuple[int, ...],
        expected_points: int | None,
    ) -> None:
        object.__setattr__(self, "_design", design)
        object.__setattr__(self, "_grid", grid)
        object.__setattr__(self, "_yoked", yoked)
        object.__setattr__(
            self,
            "_clauses",
            tuple(
                tuple(
                    (path, _freeze_axis_value(value))
                    for path, value in sorted(clause.items())
                )
                for clause in clauses
            ),
        )
        object.__setattr__(self, "_seeds", seeds)
        object.__setattr__(self, "_expected_points", expected_points)

    @property
    def design(self) -> SweepDesignSpec:
        """Return the declared assignment design."""
        return self._design

    @property
    def grid(self) -> tuple[ExpandedAxis, ...]:
        """Return the axes crossed with one another."""
        return self._grid

    @property
    def yoked(self) -> tuple[tuple[ExpandedAxis, ...], ...]:
        """Return the groups of axes whose values advance together."""
        return self._yoked

    @property
    def clauses(self) -> tuple[Mapping[str, AxisValue], ...]:
        """Return fresh copies of the exclusion clauses."""
        return tuple(
            {path: _thaw_axis_value(value) for path, value in clause}
            for clause in self._clauses
        )

    @property
    def seeds(self) -> tuple[int, ...]:
        """Return the seeds crossed with every surviving assignment."""
        return self._seeds

    @property
    def expected_points(self) -> int | None:
        """Return the point count the expansion must produce, or None."""
        return self._expected_points

    def replaced(
        self,
        *,
        grid: tuple[ExpandedAxis, ...] | None = None,
        yoked: tuple[tuple[ExpandedAxis, ...], ...] | None = None,
        seeds: tuple[int, ...] | None = None,
        expected_points: int | None,
    ) -> ExpandedSweep:
        """Return the sweep with the named parts replaced.

        Parameters
        ----------
        grid : tuple[ExpandedAxis, ...] | None
            The crossed axes, or None to keep this sweep's.
        yoked : tuple[tuple[ExpandedAxis, ...], ...] | None
            The yoked groups, or None to keep this sweep's.
        seeds : tuple[int, ...] | None
            The seeds, or None to keep this sweep's.
        expected_points : int | None
            The point count the returned sweep declares, or None for none.
        """
        return ExpandedSweep(
            design=self._design,
            grid=self._grid if grid is None else grid,
            yoked=self._yoked if yoked is None else yoked,
            clauses=self.clauses,
            seeds=self._seeds if seeds is None else seeds,
            expected_points=expected_points,
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another sweep retains the same validated values."""
        if type(other) is not ExpandedSweep:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the retained sweep values."""
        return hash((ExpandedSweep, self._identity()))

    def _identity(
        self,
    ) -> tuple[
        tuple[ExpandedAxis, ...],
        tuple[tuple[ExpandedAxis, ...], ...],
        tuple[tuple[tuple[str, _FrozenAxisValue], ...], ...],
        tuple[int, ...],
        int | None,
    ]:
        """Return the five parts a sweep is equal on."""
        return (
            self._grid,
            self._yoked,
            self._clauses,
            self._seeds,
            self._expected_points,
        )


class Contracted(NamedTuple):
    """The sweep a rung produced and the axis values or seeds it removed."""

    sweep: ExpandedSweep
    dropped: tuple[AxisValue, ...]


class Rung(Protocol):
    """One rung of a contraction ladder, applied to a frozen sweep."""

    @property
    def step(self) -> str:
        """Return the ladder position the rung occupies."""
        ...

    def contract(self, sweep: ExpandedSweep, /) -> Contracted:
        """Return the sweep the rung reduces a sweep to.

        Parameters
        ----------
        sweep : ExpandedSweep
            The sweep in force.

        Raises
        ------
        ContractionError
            If the rung cannot reduce the sweep.
        """
        ...


def _restricted(
    sweep: ExpandedSweep, key: str, keep: Callable[[AxisValue], bool], /
) -> Contracted:
    """Return the sweep with one axis's rejected values removed.

    A grid axis loses the rejected values; a yoked group loses every position
    whose value on the axis is rejected.

    Raises
    ------
    ContractionError
        If no axis assigns the key, or every value of the axis is rejected.
    """
    for index, axis in enumerate(sweep.grid):
        if axis.key != key:
            continue

        kept = tuple(point for point in axis.points if keep(point))
        dropped = tuple(point for point in axis.points if not keep(point))

        if not kept:
            msg = f"the rung would remove every value of the axis {key!r}"
            raise ContractionError(msg)

        grid = (*sweep.grid[:index], axis.with_points(kept), *sweep.grid[index + 1 :])

        return Contracted(sweep.replaced(grid=grid, expected_points=None), dropped)

    for index, group in enumerate(sweep.yoked):
        keys = tuple(axis.key for axis in group)

        if key not in keys:
            continue

        column = keys.index(key)
        positions = tuple(zip(*(axis.points for axis in group), strict=True))
        kept_positions = tuple(row for row in positions if keep(row[column]))
        dropped = _distinct(
            tuple(row[column] for row in positions if not keep(row[column]))
        )

        if not kept_positions:
            msg = f"the rung would remove every position of the yoked axis {key!r}"
            raise ContractionError(msg)

        restricted = tuple(
            axis.with_points(tuple(row[place] for row in kept_positions))
            for place, axis in enumerate(group)
        )
        yoked = (*sweep.yoked[:index], restricted, *sweep.yoked[index + 1 :])

        return Contracted(sweep.replaced(yoked=yoked, expected_points=None), dropped)

    msg = f"no axis of the sweep assigns {key!r}"
    raise ContractionError(msg)


def _distinct(points: tuple[AxisValue, ...], /) -> tuple[AxisValue, ...]:
    """Return the points with repeats removed, in first-occurrence order."""
    seen: dict[str, AxisValue] = {}

    for point in points:
        seen.setdefault(point_key(point), point)

    return tuple(seen.values())


def _axis_points(sweep: ExpandedSweep, key: str, /) -> tuple[AxisValue, ...]:
    """Return the values one axis of the sweep currently assigns.

    Raises
    ------
    ContractionError
        If no axis assigns the key.
    """
    for axis in (*sweep.grid, *(axis for group in sweep.yoked for axis in group)):
        if axis.key == key:
            return _distinct(axis.points)

    msg = f"no axis of the sweep assigns {key!r}"
    raise ContractionError(msg)


@final
class PreregisteredCells(FrozenValue):
    """Drops every axis value that realizes no preregistered level.

    Parameters
    ----------
    realized : Mapping[str, frozenset[str]]
        For each axis that spans a preregistered factor, the keys of the
        values that realize the factor's levels, as :func:`point_key` writes
        them.
    """

    _noun = "preregistered-cells rung"

    __slots__ = ("_realized",)

    def __init__(self, realized: Mapping[str, frozenset[str]]) -> None:
        object.__setattr__(self, "_realized", tuple(sorted(realized.items())))

    @property
    def step(self) -> str:
        """Return the unpreregistered-cells position."""
        return "unpreregistered-cells"

    def contract(self, sweep: ExpandedSweep, /) -> Contracted:
        """Return the sweep restricted to values that realize preregistered levels.

        Parameters
        ----------
        sweep : ExpandedSweep
            The sweep in force.
        """
        dropped: list[AxisValue] = []

        for key, realized in self._realized:
            sweep, removed = _restricted(
                sweep, key, lambda point, kept=realized: point_key(point) in kept
            )
            dropped.extend(removed)

        return Contracted(sweep.replaced(expected_points=None), tuple(dropped))


@final
class ResolutionIVFraction(FrozenValue):
    """Reduces two-level grid axes to the fraction its generators define.

    The basic factors are the grid axes the words name; their full two-level
    factorial is enumerated with each axis's first value as the negative sign,
    and each generated factor takes the value whose sign is the product of its
    word's signs. The factors the generators name leave the grid and advance
    together as one yoked group, so every other axis stays crossed with the
    fraction.

    Parameters
    ----------
    generators : tuple[tuple[str, tuple[str, ...]], ...]
        Each generated axis key with the keys of the basic axes in its word.
    """

    _noun = "resolution-IV rung"

    __slots__ = ("_generators",)

    def __init__(self, generators: tuple[tuple[str, tuple[str, ...]], ...]) -> None:
        object.__setattr__(self, "_generators", generators)

    @property
    def step(self) -> str:
        """Return the resolution-IV position."""
        return "resolution-iv"

    def contract(self, sweep: ExpandedSweep, /) -> Contracted:
        """Return the sweep with the named axes reduced to the fraction.

        Parameters
        ----------
        sweep : ExpandedSweep
            The sweep in force.

        Raises
        ------
        ContractionError
            If a named axis is not a grid axis of exactly two values.
        """
        named = {key for factor, word in self._generators for key in (factor, *word)}
        grid = {axis.key: axis for axis in sweep.grid}
        outside = sorted(named - set(grid))

        if outside:
            msg = f"the resolution-IV rung names axes that are not grid axes: {outside}"
            raise ContractionError(msg)

        uneven = sorted(key for key in named if len(grid[key].points) != 2)

        if uneven:
            msg = (
                f"the resolution-IV rung names axes without exactly two values: "
                f"{uneven}"
            )
            raise ContractionError(msg)

        words = dict(self._generators)
        design = tuple(axis for axis in sweep.grid if axis.key in named)
        basics = tuple(axis.key for axis in design if axis.key not in words)
        runs: list[dict[str, int]] = []

        for signs in itertools.product((-1, 1), repeat=len(basics)):
            run = dict(zip(basics, signs, strict=True))
            run.update(
                {
                    factor: prod(run[key] for key in word)
                    for factor, word in words.items()
                }
            )
            runs.append(run)

        group = tuple(
            axis.with_points(
                tuple(axis.points[0 if run[axis.key] < 0 else 1] for run in runs)
            )
            for axis in design
        )

        return Contracted(
            sweep.replaced(
                grid=tuple(axis for axis in sweep.grid if axis.key not in named),
                yoked=(*sweep.yoked, group),
                expected_points=None,
            ),
            (),
        )


@final
class DropInOrder(FrozenValue):
    """Drops the next value of one axis in a declared order.

    Parameters
    ----------
    step : str
        The ladder position the rung occupies.
    axis : str
        The key of the axis values are dropped from.
    order : tuple[AxisValue, ...]
        The values in the order they are dropped.
    """

    _noun = "drop-order rung"

    __slots__ = ("_axis", "_order", "_step")

    def __init__(self, step: str, axis: str, order: tuple[AxisValue, ...]) -> None:
        object.__setattr__(self, "_step", step)
        object.__setattr__(self, "_axis", axis)
        object.__setattr__(self, "_order", tuple(point_key(value) for value in order))

    @property
    def step(self) -> str:
        """Return the ladder position the rung occupies."""
        return self._step

    def contract(self, sweep: ExpandedSweep, /) -> Contracted:
        """Return the sweep without the first listed value the axis still assigns.

        Parameters
        ----------
        sweep : ExpandedSweep
            The sweep in force.

        Raises
        ------
        ContractionError
            If the axis assigns no value left in the order, or the value is the
            axis's last.
        """
        present = {point_key(point) for point in _axis_points(sweep, self._axis)}
        remaining = tuple(key for key in self._order if key in present)

        if not remaining:
            msg = f"no value left in the drop order is assigned by {self._axis!r}"
            raise ContractionError(msg)

        target = remaining[0]

        return _restricted(sweep, self._axis, lambda point: point_key(point) != target)


@final
class SeedFloor(FrozenValue):
    """Reduces every cell's seeds to the first declared seeds, down to a floor.

    Parameters
    ----------
    floor : int
        The number of seeds every cell keeps.
    """

    _noun = "seed rung"

    __slots__ = ("_floor",)

    def __init__(self, floor: int) -> None:
        object.__setattr__(self, "_floor", floor)

    @property
    def floor(self) -> int:
        """Return the number of seeds every cell keeps."""
        return self._floor

    @property
    def step(self) -> str:
        """Return the seed position."""
        return "seeds"

    def contract(self, sweep: ExpandedSweep, /) -> Contracted:
        """Return the sweep keeping only its first ``floor`` seeds.

        Parameters
        ----------
        sweep : ExpandedSweep
            The sweep in force.

        Raises
        ------
        ContractionError
            If the sweep already declares no more seeds than the floor.
        """
        if len(sweep.seeds) <= self._floor:
            msg = (
                f"the sweep's {len(sweep.seeds)} seeds are already at the floor "
                f"of {self._floor}"
            )
            raise ContractionError(msg)

        return Contracted(
            sweep.replaced(seeds=sweep.seeds[: self._floor], expected_points=None),
            sweep.seeds[self._floor :],
        )


@final
class Fallback(FrozenValue):
    """Keeps the sweep and switches the experiment to its fallback claim."""

    _noun = "fallback rung"

    __slots__ = ()

    @property
    def step(self) -> str:
        """Return the fallback position."""
        return "fallback"

    def contract(self, sweep: ExpandedSweep, /) -> Contracted:
        """Return the sweep unchanged.

        Parameters
        ----------
        sweep : ExpandedSweep
            The sweep in force.
        """
        return Contracted(sweep, ())
