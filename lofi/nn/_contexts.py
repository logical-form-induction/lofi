"""Frontier contexts reconstructed from a teacher term's encoding."""

from types import MappingProxyType
from typing import TYPE_CHECKING

from lofi.terms import ChoiceIndex, Encoding, HoleContext

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi.terms import Path

__all__ = ["child_sides", "slot_paths", "teacher_contexts"]


def child_sides(row: Encoding, /) -> tuple[int, ...]:
    """Return each slot's argument position under its parent.

    A root or unparented slot takes position zero, and the children of a parent
    take consecutive positions in slot order.
    """
    seen: dict[int, int] = {}
    sides: list[int] = []

    for parent in row.parents:
        if parent == Encoding.NO_PARENT:
            sides.append(0)
            continue

        side = seen.get(parent, 0)
        sides.append(side)
        seen[parent] = side + 1

    return tuple(sides)


def slot_paths(row: Encoding, /) -> tuple[Path, ...]:
    """Return each slot's argument path from the root.

    Raises
    ------
    ValueError
        If a slot's parent does not precede it.
    """
    sides = child_sides(row)
    paths: list[Path] = []

    for slot, parent in enumerate(row.parents):
        if parent == Encoding.NO_PARENT:
            paths.append(())
        elif 0 <= parent < slot:
            paths.append((*paths[parent], sides[slot]))
        else:
            msg = f"slot {slot} has nonpreceding parent {parent}"
            raise ValueError(msg)

    return tuple(paths)


def teacher_contexts(index: ChoiceIndex, row: Encoding, /) -> tuple[HoleContext, ...]:
    """Return the frontier context frontier decoding reaches at every slot.

    Occupied slots are visited in the shallowest-leftmost hole order frontier
    decoding fills, so each slot's free-variable count and remaining budget are
    those of the skeleton decoded before it. An unoccupied slot is never scored
    and takes the root context, which keeps the rows rectangular.
    """
    codes = index.codes(row)
    sides = child_sides(row)
    paths = slot_paths(row)
    root = index.operator(next(code for code in codes if code != ChoiceIndex.PAD)).sort
    empty: Mapping[str, int] = MappingProxyType({})
    contexts = [HoleContext((), root, empty, empty, row.width - 1)] * row.width
    order = sorted(
        (slot for slot, held in enumerate(row.occupied) if held),
        key=lambda slot: (len(paths[slot]), paths[slot]),
    )
    free: dict[str, set[int]] = {}
    opened = 0

    for slot in order:
        parent = row.parents[slot]
        expected = (
            root
            if parent == Encoding.NO_PARENT
            else index.operator(codes[parent]).arg_sorts[sides[slot]]
        )
        in_scope: dict[str, int] = {}
        child = slot
        above = parent

        while above != Encoding.NO_PARENT:
            for site in index.operator(codes[above]).scope(sides[child]):
                in_scope[site.sort] = in_scope.get(site.sort, 0) + 1

            child = above
            above = row.parents[child]

        contexts[slot] = HoleContext(
            paths[slot],
            expected,
            MappingProxyType(in_scope),
            MappingProxyType({name: len(values) for name, values in free.items()}),
            row.width - 1 - opened,
        )
        operator = index.operator(codes[slot])
        payload = index.payload(codes[slot])

        if operator.is_free_variable and payload is not None:
            free.setdefault(operator.sort, set()).add(payload)

        opened += operator.arity

    return tuple(contexts)
