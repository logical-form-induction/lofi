"""Grid and yoked sweep expansion into seeded points."""

import json
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Protocol

from lofi._types import JsonValue
from lofi.run._boundary.sweep import SweepDesignSpec, SweepPoint
from lofi.run._design import SweepAssignment, SweepAxis, design_assignments
from lofi.spec import ConfigSource

if TYPE_CHECKING:
    from lofi.run._boundary.sweep import AxisValue
    from lofi.spec import LeafOrigin, Provenance

__all__ = [
    "Composer",
    "GridSweepExpander",
    "ShadowedAxisError",
    "UnexpectedPointCountError",
    "leading_overrides",
    "unique_points",
]

type Composer = Callable[[tuple[str, ...], int], Mapping[str, JsonValue]]
"""Configuration composition from overrides and a run seed to a resolved mapping."""

type PointAugmenter = Callable[[SweepPoint], Sequence[SweepPoint]]
"""Post-composition expansion of one base point into ordinary trainable points."""

type ProvenanceResolver = Callable[[tuple[str, ...], int], Provenance]
"""Looks up the traced sources produced while composing one point."""

type SourceResolver = Callable[
    [tuple[str, ...], int], Mapping[str, tuple[LeafOrigin, ...]]
]
"""Looks up every source that set each path while composing one point."""

_DECLARING: frozenset[ConfigSource] = frozenset(
    source
    for source in ConfigSource
    if source.precedence > ConfigSource.GROUP.precedence
)
"""The sources that outrank every group fragment."""


class SweepPlan(Protocol):
    """The frozen values the grid expander reads from a validated sweep."""

    @property
    def design(self) -> SweepDesignSpec: ...

    @property
    def grid(self) -> tuple[SweepAxis, ...]: ...

    @property
    def yoked(self) -> tuple[tuple[SweepAxis, ...], ...]: ...

    @property
    def clauses(self) -> tuple[Mapping[str, AxisValue], ...]: ...

    @property
    def seeds(self) -> tuple[int, ...]: ...

    @property
    def expected_points(self) -> int | None: ...


class UnexpectedPointCountError(ValueError):
    """A point count other than the one the sweep declares.

    Attributes
    ----------
    expected : int
        The count the sweep declares.
    actual : int
        The count the expansion produced.
    """

    def __init__(self, expected: int, actual: int) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"the sweep declares {expected} points but expands to {actual}"
        )


class ShadowedAxisError(ValueError):
    """An axis sharing a path with another source of the composed document.

    The document body, a profile, an overlay or an override outranks a group
    fragment, so a setting under a group axis would replace the selected
    fragment's value at that leaf. A profile, an overlay or an override at a
    leaf axis's path would be silently discarded, and a document value under a
    mapping point would survive into it. Either way the point would not be the
    assignment the sweep declares.

    Attributes
    ----------
    axis : str
        The key of the axis.
    path : str
        The dotted path both sources set.
    source : ConfigSource
        The kind of the other source.
    layer : str
        Which member of that source set the path: the document or profile,
        or the override expression.
    """

    def __init__(self, axis: str, path: str, source: ConfigSource, layer: str) -> None:
        self.axis = axis
        self.path = path
        self.source = source
        self.layer = layer
        member = f" {layer}" if layer else ""
        super().__init__(
            f"the sweep axis {axis!r} and the {source.value}{member} both set "
            f"{path}; remove the setting from one of them"
        )


def unique_points(points: Sequence[SweepPoint]) -> tuple[SweepPoint, ...]:
    """Return the distinct points, each at its first position.

    Parameters
    ----------
    points : Sequence[SweepPoint]
        Points from one or several expansions.
    """
    return tuple(dict.fromkeys(points))


def _render(axis: SweepAxis, value: AxisValue) -> str:
    """Return the ``key=value`` override assigning one point of an axis.

    A group axis selects a fragment, written with the slot's segments joined
    by ``/`` and None as ``null``; a slot at the root of the document has no
    ``/`` to mark it as a selection, so it is written with a leading one. A
    leaf axis writes a string point as it stands, since the override text is
    the string itself, and every other point as JSON, so a list or mapping
    point arrives at override parsing as the value it is.
    """
    if axis.selects_group:
        name = "null" if value is None else value
        slot = "/".join(axis.slot)
        return f"{slot if len(axis.slot) > 1 else '/' + slot}={name}"

    match value:
        case str():
            text = value

        case _:
            text = json.dumps(value)

    return f"{axis.key}={text}"


def _agrees(assigned: AxisValue, expected: AxisValue) -> bool:
    """Return whether an assigned value equals a clause value in type and value.

    Two list points agree when they have the same length and agree element by
    element, so the type discrimination that separates 1 from 1.0 and from True
    holds at every level of a nested point rather than at the outer list alone.
    """
    if type(assigned) is not type(expected):
        return False

    if isinstance(assigned, list) and isinstance(expected, list):
        return len(assigned) == len(expected) and all(
            _agrees(item, other) for item, other in zip(assigned, expected, strict=True)
        )

    if isinstance(assigned, dict) and isinstance(expected, dict):
        return assigned.keys() == expected.keys() and all(
            _agrees(value, expected[key]) for key, value in assigned.items()
        )

    return assigned == expected


def _matches(
    clause: tuple[tuple[str, AxisValue], ...], assigned: Mapping[str, AxisValue]
) -> bool:
    """Return whether every pair in a clause agrees with an assignment."""
    return all(_agrees(assigned[key], value) for key, value in clause)


def _overlaps(path: str, key: str) -> bool:
    """Return whether one dotted path is, contains, or lies under another."""
    return path == key or path.startswith(f"{key}.") or key.startswith(f"{path}.")


def _shadows(
    axis: SweepAxis,
    path: str,
    origins: tuple[LeafOrigin, ...],
    own: frozenset[str],
) -> LeafOrigin | None:
    """Return the source that shares a path with an axis, if one does.

    A group axis selects a fragment, which every declaring source outranks, so
    any declaring source at, under or above its slot replaces part of the
    selection. A leaf axis is applied over the document it sweeps, so the
    document's own value at its path is what the axis varies; a profile, an
    overlay or another override there would be discarded unseen, and a
    document value surviving under a mapping point would be merged into it.

    Parameters
    ----------
    axis : SweepAxis
        The axis.
    path : str
        A dotted path that is, contains, or lies under the axis's key.
    origins : tuple[LeafOrigin, ...]
        Every source that set the path, lowest precedence first.
    own : frozenset[str]
        The overrides that render the point's assignments.
    """
    foreign = tuple(
        origin
        for origin in origins
        if origin.source in _DECLARING and origin.layer not in own
    )

    if axis.selects_group:
        return foreign[0] if foreign else None

    discarded = tuple(o for o in foreign if o.source is not ConfigSource.FILE)

    if discarded:
        return discarded[0]

    if path.startswith(f"{axis.key}.") and origins[-1] in foreign:
        return origins[-1]

    return None


def _refuse_shadowing(
    axes: Sequence[SweepAxis],
    overrides: tuple[str, ...],
    sources: Mapping[str, tuple[LeafOrigin, ...]],
) -> None:
    """Refuse a point in which another source shares a path with an axis.

    Parameters
    ----------
    axes : Sequence[SweepAxis]
        The axes the point assigns.
    overrides : tuple[str, ...]
        The overrides that render the point's assignments, which are the
        axes' own settings.
    sources : Mapping[str, tuple[LeafOrigin, ...]]
        Every source that set each path while the point was composed.

    Raises
    ------
    ShadowedAxisError
        If the document body, a profile, an overlay, or an override other
        than the point's own sets a path at, under or above a group axis's
        slot; if a profile, an overlay or another override sets a path a leaf
        axis assigns; or if a document value survives under a leaf axis's
        mapping point.
    """
    own = frozenset(overrides)

    for axis in axes:
        for path, origins in sources.items():
            if not _overlaps(path, axis.key):
                continue

            shadow = _shadows(axis, path, origins, own)

            if shadow is not None:
                raise ShadowedAxisError(axis.key, path, shadow.source, shadow.layer)


def _assignments(sweep: SweepPlan) -> tuple[SweepAssignment, ...]:
    """Return assignments selected by the sweep's declared design."""
    return design_assignments(sweep.design, sweep)


def leading_overrides(sweep: SweepPlan, /) -> tuple[str, ...]:
    """Return the overrides of a sweep's first assignment, before any exclusion.

    The first assignment takes every grid axis's first value and every yoked
    group's first position, so it composes the configuration every point's
    sections are drawn from.

    Parameters
    ----------
    sweep : SweepPlan
        The frozen values of a validated sweep declaration.
    """
    assignments = _assignments(sweep)

    return tuple(_render(axis, value) for axis, value in assignments[0])


class GridSweepExpander:
    """Expands a sweep by crossing grid axes, yoking groups, excluding, and seeding.

    Grid axes are crossed in declaration order with the last varying fastest,
    and each yoked group advances its axes together as one further dimension.
    A candidate survives when no exclusion clause agrees with it on every pair
    the clause names. Each survivor is rendered as one override per assigned
    axis and composed once per run seed; duplicate points keep their first
    position. When the composition reports its sources, a point
    in which another source overrules or is silently overruled by an axis is
    refused rather than composed into a point other than the one declared.
    """

    __slots__ = ("_augment", "_compose", "_provenance", "_sources")

    def __init__(
        self,
        compose: Composer,
        *,
        augment: PointAugmenter | None = None,
        provenance: ProvenanceResolver | None = None,
        sources: SourceResolver | None = None,
    ) -> None:
        self._compose = compose
        self._augment = augment
        self._provenance = provenance
        self._sources = sources

    def expand(self, sweep: SweepPlan, /) -> tuple[SweepPoint, ...]:
        """Return the unique points in deterministic order.

        Parameters
        ----------
        sweep : SweepPlan
            The frozen values of a validated sweep declaration.

        Raises
        ------
        ShadowedAxisError
            If a source the axes do not own sets a path an axis assigns.
        UnexpectedPointCountError
            If the sweep declares a point count the expansion does not produce.
        """
        clauses = tuple(tuple(clause.items()) for clause in sweep.clauses)
        seeds = tuple(sweep.seeds)
        points: list[SweepPoint] = []

        for assignment in _assignments(sweep):
            assigned = {axis.key: value for axis, value in assignment}

            if any(_matches(clause, assigned) for clause in clauses):
                continue

            overrides = tuple(_render(axis, value) for axis, value in assignment)

            for seed in seeds:
                config = self._compose(overrides, seed)

                if self._sources is not None:
                    _refuse_shadowing(
                        tuple(axis for axis, _ in assignment),
                        overrides,
                        self._sources(overrides, seed),
                    )
                provenance = (
                    None
                    if self._provenance is None
                    else self._provenance(overrides, seed)
                )
                point = SweepPoint(
                    config,
                    overrides=overrides,
                    provenance=provenance,
                    seed=seed,
                )
                points.extend(
                    (point,) if self._augment is None else self._augment(point)
                )

        unique = unique_points(points)

        if sweep.expected_points is not None and len(unique) != sweep.expected_points:
            raise UnexpectedPointCountError(sweep.expected_points, len(unique))

        return unique
