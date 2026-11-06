"""Test-only entry-point inventories that name in-memory components."""

import sys
from importlib.metadata import EntryPoint
from itertools import count
from types import ModuleType
from typing import TYPE_CHECKING, Final

from lofi.registry import Registry, entry_point_group
from lofi.registry import _registry as registry_module

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    import pytest

__all__ = ["entries", "install", "installed"]

_SERIALS: Final = count()
"""Distinguishes the module each inventory publishes its members from."""


def entries(family: str, members: Mapping[str, object]) -> tuple[EntryPoint, ...]:
    """Return one entry point per member, each naming the member it is given.

    Parameters
    ----------
    family : str
        The family whose group the entry points belong to.
    members : Mapping[str, object]
        Each tag's component.

    Returns
    -------
    tuple[EntryPoint, ...]
        The entry points, in the members' order.
    """
    module = ModuleType(f"lofi_installed_fixture_{next(_SERIALS)}")
    group = entry_point_group(family)
    points: list[EntryPoint] = []

    for position, (tag, member) in enumerate(members.items()):
        attribute = f"member_{position}"
        setattr(module, attribute, member)
        points.append(
            EntryPoint(name=tag, value=f"{module.__name__}:{attribute}", group=group)
        )

    sys.modules[module.__name__] = module

    return tuple(points)


def installed[A](family: str, members: Mapping[str, object]) -> Registry[A]:
    """Return a registry over a family whose only entries are the given members.

    Parameters
    ----------
    family : str
        The family.
    members : Mapping[str, object]
        Each tag's component, of the member type the caller's registry expects.

    Returns
    -------
    Registry[A]
        A registry resolving exactly those tags.
    """
    points = entries(family, members)

    return Registry(family, source=lambda: points)


def install(
    monkeypatch: pytest.MonkeyPatch, family: str, points: Iterable[EntryPoint]
) -> None:
    """Make the installed environment list exactly these points for one family.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Restores the installed environment after the test.
    family : str
        The family whose group is replaced.
    points : Iterable[EntryPoint]
        The group's entry points while the test runs.
    """
    current = registry_module.entry_points
    group = entry_point_group(family)
    held = tuple(points)

    def listed() -> tuple[EntryPoint, ...]:
        return (*(point for point in current() if point.group != group), *held)

    monkeypatch.setattr(registry_module, "entry_points", listed)
