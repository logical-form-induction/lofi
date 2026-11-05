"""Declared sweep designs resolved through installed plugins."""

from __future__ import annotations

import itertools
import random
from math import floor
from typing import TYPE_CHECKING, Protocol, cast

from lofi._frozen import FrozenValue
from lofi.registry import resolve_plugin
from lofi.run._boundary.sweep import (
    AxisValue,
    FractionalFactorialDesignSpec,
    GridDesignSpec,
    LatinHypercubeDesignSpec,
    RandomDesignSpec,
    SweepDesignSpec,
)

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "FRACTIONAL_FACTORIAL_PLUGIN",
    "GRID_PLUGIN",
    "LATIN_HYPERCUBE_PLUGIN",
    "RANDOM_PLUGIN",
    "SweepAssignment",
    "SweepDesignPlan",
    "SweepDesignPlugin",
    "design_assignments",
]

type SweepAssignment = tuple[tuple[SweepAxis, AxisValue], ...]
"""Axis-value pairs assigned to one candidate, in declaration order."""


class SweepAxis(Protocol):
    """One immutable axis available to a sweep design."""

    @property
    def key(self) -> str: ...

    @property
    def selects_group(self) -> bool: ...

    @property
    def slot(self) -> tuple[str, ...]: ...

    @property
    def points(self) -> tuple[AxisValue, ...]: ...


class SweepDesignPlan(Protocol):
    """The axis inventory a sweep design selects assignments from."""

    @property
    def grid(self) -> tuple[SweepAxis, ...]: ...

    @property
    def yoked(self) -> tuple[tuple[SweepAxis, ...], ...]: ...


class SweepDesignPlugin[S: SweepDesignSpec](FrozenValue):
    """Expand assignments for one exact sweep-design declaration."""

    _noun = "sweep-design plugin"
    __slots__ = ("_assign", "_spec")

    def __init__(
        self,
        *,
        spec: type[S],
        assign: Callable[[S, SweepDesignPlan], tuple[SweepAssignment, ...]],
    ) -> None:
        object.__setattr__(self, "_spec", spec)
        object.__setattr__(self, "_assign", assign)

    @property
    def spec(self) -> type[S]:
        """Return the declaration variant this plugin consumes."""
        return self._spec

    def assignments(
        self, declaration: S, sweep: SweepDesignPlan, /
    ) -> tuple[SweepAssignment, ...]:
        """Return assignments selected by the declared design."""
        return self._assign(declaration, sweep)


def _plugin(declaration: SweepDesignSpec, /) -> SweepDesignPlugin[SweepDesignSpec]:
    return cast(
        "SweepDesignPlugin[SweepDesignSpec]",
        resolve_plugin(
            "sweep_design",
            cast("str", declaration.kind),
            SweepDesignPlugin,
            type(declaration),
        ),
    )


def design_assignments(
    declaration: SweepDesignSpec, sweep: SweepDesignPlan, /
) -> tuple[SweepAssignment, ...]:
    """Return assignments selected by an installed sweep-design plugin."""
    return _plugin(declaration).assignments(declaration, sweep)


def _dimensions(sweep: SweepDesignPlan, /) -> tuple[tuple[SweepAssignment, ...], ...]:
    dimensions: list[tuple[SweepAssignment, ...]] = [
        tuple(((axis, value),) for value in axis.points) for axis in sweep.grid
    ]

    dimensions.extend(
        tuple(
            tuple((axis, value) for axis, value in zip(group, position, strict=True))
            for position in zip(*(axis.points for axis in group), strict=True)
        )
        for group in sweep.yoked
    )

    return tuple(dimensions)


def _cross(
    dimensions: tuple[tuple[SweepAssignment, ...], ...], /
) -> tuple[SweepAssignment, ...]:
    return tuple(
        tuple(pair for chunk in chunks for pair in chunk)
        for chunks in itertools.product(*dimensions)
    )


def _grid(
    declaration: GridDesignSpec, sweep: SweepDesignPlan, /
) -> tuple[SweepAssignment, ...]:
    del declaration
    return _cross(_dimensions(sweep))


def _random(
    declaration: RandomDesignSpec, sweep: SweepDesignPlan, /
) -> tuple[SweepAssignment, ...]:
    assignments = _cross(_dimensions(sweep))

    if declaration.points > len(assignments):
        msg = (
            f"a random design asks for {declaration.points} points from a grid "
            f"of {len(assignments)}"
        )
        raise ValueError(msg)

    chosen = random.Random(declaration.seed).sample(
        range(len(assignments)), declaration.points
    )
    return tuple(assignments[index] for index in chosen)


def _latin_hypercube(
    declaration: LatinHypercubeDesignSpec, sweep: SweepDesignPlan, /
) -> tuple[SweepAssignment, ...]:
    dimensions = _dimensions(sweep)
    generator = random.Random(declaration.seed)
    columns: list[tuple[SweepAssignment, ...]] = []

    for dimension in dimensions:
        indices = [
            min(
                len(dimension) - 1,
                floor((position + 0.5) * len(dimension) / declaration.points),
            )
            for position in range(declaration.points)
        ]
        generator.shuffle(indices)
        columns.append(tuple(dimension[index] for index in indices))

    return tuple(
        tuple(pair for column in columns for pair in column[position])
        for position in range(declaration.points)
    )


def _fractional_factorial(
    declaration: FractionalFactorialDesignSpec,
    sweep: SweepDesignPlan,
    /,
) -> tuple[SweepAssignment, ...]:
    by_key = {axis.key: axis for axis in sweep.grid}
    generated = {
        generator.factor: generator.word for generator in declaration.generators
    }
    basic_keys = tuple(
        axis.key
        for axis in sweep.grid
        if any(axis.key in word for word in generated.values())
    )
    participating = frozenset((*basic_keys, *generated))
    unknown = sorted(participating - by_key.keys())

    if unknown:
        msg = f"fractional-factorial generators name undeclared grid axes: {unknown}"
        raise ValueError(msg)

    invalid = tuple(key for key in participating if len(by_key[key].points) != 2)

    if invalid:
        msg = f"fractional-factorial axes are not two-level: {invalid}"
        raise ValueError(msg)

    fraction: list[SweepAssignment] = []

    for signs in itertools.product((0, 1), repeat=len(basic_keys)):
        selected = dict(zip(basic_keys, signs, strict=True))

        for factor, word in generated.items():
            selected[factor] = sum(selected[key] for key in word) % 2

        fraction.append(
            tuple(
                (axis, axis.points[selected[axis.key]])
                for axis in sweep.grid
                if axis.key in participating
            )
        )

    dimensions: list[tuple[SweepAssignment, ...]] = []
    inserted = False

    for axis in sweep.grid:
        if axis.key in participating:
            if not inserted:
                dimensions.append(tuple(fraction))
                inserted = True
            continue

        dimensions.append(tuple(((axis, value),) for value in axis.points))

    dimensions.extend(_dimensions(_YokedOnly(sweep.yoked)))
    return _cross(tuple(dimensions))


class _YokedOnly:
    """Present only yoked dimensions to the common dimension builder."""

    __slots__ = ("_yoked",)

    def __init__(self, yoked: tuple[tuple[SweepAxis, ...], ...]) -> None:
        self._yoked = yoked

    @property
    def grid(self) -> tuple[SweepAxis, ...]:
        return ()

    @property
    def yoked(self) -> tuple[tuple[SweepAxis, ...], ...]:
        return self._yoked


GRID_PLUGIN = SweepDesignPlugin(spec=GridDesignSpec, assign=_grid)
RANDOM_PLUGIN = SweepDesignPlugin(spec=RandomDesignSpec, assign=_random)
LATIN_HYPERCUBE_PLUGIN = SweepDesignPlugin(
    spec=LatinHypercubeDesignSpec,
    assign=_latin_hypercube,
)
FRACTIONAL_FACTORIAL_PLUGIN = SweepDesignPlugin(
    spec=FractionalFactorialDesignSpec,
    assign=_fractional_factorial,
)
