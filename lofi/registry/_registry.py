"""Typed component resolution over installed entry points."""

import sys
from importlib.metadata import EntryPoint, entry_points
from itertools import pairwise
from typing import TYPE_CHECKING, Protocol, cast, override

from lofi._checks import ensure_name
from lofi.registry._errors import RegistryError, TagCollisionError, UnknownTagError

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = ["EntrySource", "Registry", "entry_point_group", "resolve_plugin"]


class EntrySource(Protocol):
    """Lists every installed entry point."""

    def __call__(self) -> Iterable[EntryPoint]:
        """Return the installed entry points of every group."""
        ...


_INVENTORIES: dict[tuple[EntrySource, tuple[str, ...]], tuple[EntryPoint, ...]] = {}
"""Installed inventories, keyed by their source and the import path read."""


def entry_point_group(family: str, /) -> str:
    """Return the entry-point group a component family owns.

    Parameters
    ----------
    family : str
        The family name.

    Returns
    -------
    str
        The family's entry-point group.

    Raises
    ------
    TypeError
        If the family is not a string.
    ValueError
        If the family is empty.
    """
    return f"lofi.{ensure_name('family', family)}"


def _inventory(source: EntrySource, /) -> tuple[EntryPoint, ...]:
    """Return every entry point one source lists under the current import path."""
    key = (source, tuple(sys.path))
    inventory = _INVENTORIES.get(key)

    if inventory is None:
        inventory = tuple(source())
        _INVENTORIES[key] = inventory

    return inventory


def _claimant(point: EntryPoint, /) -> str:
    """Return the distribution that installs an entry point, else its value."""
    distribution = point.dist

    return point.value if distribution is None else distribution.name


class Registry[A]:
    """Resolves one family's components from its entry-point group.

    Parameters
    ----------
    family : str
        The component family, whose group is ``lofi.<family>``.
    source : EntrySource | None
        What lists the entry points, or None for the installed environment's
        ``importlib.metadata.entry_points``.

    Raises
    ------
    TypeError
        If the family is not a string.
    ValueError
        If the family is empty.
    """

    __slots__ = ("_family", "_source")

    def __init__(self, family: str, /, *, source: EntrySource | None = None) -> None:
        self._family = ensure_name("family", family)
        self._source = source

    @property
    def family(self) -> str:
        """Return the component family."""
        return self._family

    @property
    def group(self) -> str:
        """Return the family's entry-point group."""
        return entry_point_group(self._family)

    def entries(self) -> tuple[EntryPoint, ...]:
        """Return the family's entry points, in tag order.

        Returns
        -------
        tuple[EntryPoint, ...]
            Every installed entry point of the family's group.

        Raises
        ------
        TagCollisionError
            If two entry points hold one tag.
        """
        group = self.group
        source = entry_points if self._source is None else self._source
        points = sorted(
            (point for point in _inventory(source) if point.group == group),
            key=lambda point: (point.name, point.value),
        )

        for first, second in pairwise(points):
            if first.name == second.name:
                raise TagCollisionError(
                    family=self._family,
                    tag=first.name,
                    distributions=tuple(
                        _claimant(point) for point in points if point.name == first.name
                    ),
                )

        return tuple(points)

    def available(self) -> tuple[str, ...]:
        """Return the installed tags in lexical order.

        Returns
        -------
        tuple[str, ...]
            Each installed tag once.

        Raises
        ------
        TagCollisionError
            If two entry points hold one tag.
        """
        return tuple(point.name for point in self.entries())

    def get(self, tag: str, /) -> A:
        """Return the component installed under one tag.

        Parameters
        ----------
        tag : str
            The tag to resolve.

        Returns
        -------
        A
            The object the tag's entry point names.

        Raises
        ------
        TypeError
            If the tag is not a string.
        ValueError
            If the tag is empty.
        UnknownTagError
            If no entry point holds the tag.
        TagCollisionError
            If two entry points of the family hold one tag.
        """
        named = ensure_name("tag", tag)
        entries = self.entries()

        for point in entries:
            if point.name == named:
                return cast("A", point.load())

        raise UnknownTagError(
            family=self._family,
            tag=named,
            available=tuple(point.name for point in entries),
        )

    def members(self) -> tuple[tuple[str, A], ...]:
        """Return every installed tag beside the component it names.

        Returns
        -------
        tuple[tuple[str, A], ...]
            Tag and component pairs in tag order.

        Raises
        ------
        TagCollisionError
            If two entry points hold one tag.
        """
        return tuple((tag, self.get(tag)) for tag in self.available())

    @override
    def __repr__(self) -> str:
        """Return the family and available tags."""
        return f"Registry(family={self._family!r}, available={self.available()!r})"


class _SpecPlugin(Protocol):
    """A plugin that names the declaration class it builds from."""

    @property
    def spec(self) -> type:
        """Return the declaration class."""
        ...


def resolve_plugin[P: _SpecPlugin](
    family: str, tag: str, plugin: type[P], declaration: type, /
) -> P:
    """Return the plugin one family installs under a declaration's tag.

    Parameters
    ----------
    family : str
        The plugin family.
    tag : str
        The declaration's tag within the family.
    plugin : type[P]
        The plugin class the family's entries are instances of.
    declaration : type
        The declaration class the plugin must build from.

    Returns
    -------
    P
        The installed plugin.

    Raises
    ------
    UnknownTagError
        If the family installs nothing under the tag.
    TagCollisionError
        If two entry points hold the tag.
    RegistryError
        If the entry names something other than a plugin of that class, or a
        plugin for another declaration class.
    """
    member = Registry[object](family).get(tag)

    if not isinstance(member, plugin):
        msg = f"the {family} entry {tag!r} names {member!r}, not a {plugin.__name__}"
        raise RegistryError(msg, family=family)

    if member.spec is not declaration:
        msg = (
            f"the {family} entry {tag!r} builds {member.spec.__name__}, "
            f"not {declaration.__name__}"
        )
        raise RegistryError(msg, family=family)

    return member
