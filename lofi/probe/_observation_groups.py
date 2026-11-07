"""Caller-declared groups over flattened observation rows."""

from typing import TYPE_CHECKING, final, override

from lofi._checks import ensure_non_negative, ensure_positive
from lofi._frozen import FrozenValue

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = ["ObservationGrouping"]


def _row(value: int, rows: int, position: int, group: int) -> int:
    """Return one row index inside the grouping's extent.

    Parameters
    ----------
    value : int
        The row index to validate.
    rows : int
        The grouping's exclusive upper bound.
    position : int
        The row's position in its declared group.
    group : int
        The group's position in the declaration.

    Returns
    -------
    int
        The validated row index.

    Raises
    ------
    TypeError
        If the index is not an integer or is a boolean.
    ValueError
        If the index is negative or outside the grouping's extent.
    """
    field = f"row {position} of observation group {group}"
    checked = ensure_non_negative(field, value)

    if checked >= rows:
        msg = f"{field} is {checked}, beyond the grouping's {rows} rows"
        raise ValueError(msg)

    return checked


def _group(values: Iterable[int], rows: int, position: int) -> frozenset[int]:
    """Return one checked nonempty group of distinct rows.

    Parameters
    ----------
    values : Iterable[int]
        The rows grouped together.
    rows : int
        The grouping's row extent.
    position : int
        Which group is being checked.

    Returns
    -------
    frozenset[int]
        The immutable group.

    Raises
    ------
    TypeError
        If an index is not an integer or is a boolean.
    ValueError
        If the group is empty, repeats a row, or names an out-of-range row.
    """
    held: list[int] = []
    seen: set[int] = set()

    for offset, value in enumerate(values):
        checked = _row(value, rows, offset, position)

        if checked in seen:
            msg = f"observation group {position} repeats row {checked}"
            raise ValueError(msg)

        held.append(checked)
        seen.add(checked)

    if not held:
        msg = f"observation group {position} is empty"
        raise ValueError(msg)

    return frozenset(held)


@final
class ObservationGrouping(FrozenValue):
    """Declares disjoint groups over a flattened observation's rows.

    The positive row extent names the row-major flattening of every observation axis
    except the value axis selected by a diagnostic. Groups remain in caller order,
    while membership within each group is unordered. Rows may remain ungrouped.

    Parameters
    ----------
    rows : int
        The positive flat-row extent the memberships index.
    groups : Iterable[Iterable[int]]
        Caller-ordered nonempty groups of distinct rows, pairwise disjoint.

    Raises
    ------
    TypeError
        If the extent or an index is not an integer, or is a boolean.
    ValueError
        If the extent is not positive; a group is empty, repeated, or overlapping;
        or an index is negative or out of range.
    """

    __slots__ = ("_groups", "_rows")

    _noun = "observation grouping"

    def __init__(self, rows: int, groups: Iterable[Iterable[int]] = ()) -> None:
        extent = ensure_positive("an observation grouping's row extent", rows)
        held: list[frozenset[int]] = []
        occupied: set[int] = set()
        seen: set[frozenset[int]] = set()

        for position, values in enumerate(groups):
            group = _group(values, extent, position)

            if group in seen:
                msg = (
                    f"observation group {position} duplicates an earlier group with "
                    f"rows {sorted(group)}"
                )
                raise ValueError(msg)

            overlap = occupied.intersection(group)

            if overlap:
                msg = (
                    f"observation group {position} overlaps earlier groups at rows "
                    f"{sorted(overlap)}"
                )
                raise ValueError(msg)

            held.append(group)
            occupied.update(group)
            seen.add(group)

        object.__setattr__(self, "_rows", extent)
        object.__setattr__(self, "_groups", tuple(held))

    @property
    def rows(self) -> int:
        """Return the flat-row extent the memberships index."""
        return self._rows

    @property
    def groups(self) -> tuple[frozenset[int], ...]:
        """Return the groups in caller-declared order."""
        return self._groups

    def _identity(self) -> tuple[int, tuple[frozenset[int], ...]]:
        """Return the fields on which two grouping declarations agree."""
        return self._rows, self._groups

    @override
    def __reduce__(
        self,
    ) -> tuple[type[ObservationGrouping], tuple[int, tuple[frozenset[int], ...]]]:
        """Return the constructor call that rebuilds this grouping."""
        return ObservationGrouping, self._identity()

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another grouping has the same extent and groups."""
        if type(other) is not ObservationGrouping:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the row extent and declared groups."""
        return hash((ObservationGrouping, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this grouping."""
        groups = tuple(tuple(sorted(group)) for group in self._groups)

        return f"ObservationGrouping({self._rows!r}, {groups!r})"
