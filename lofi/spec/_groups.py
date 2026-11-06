"""Search paths for configuration groups.

A dotted slot maps to nested directories under each root, with one document
per option. Roots are searched in project, package, then plugin order, so an
earlier option shadows later options of the same name.
"""

from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Final

from lofi.registry import Registry, entry_point_group
from lofi.spec._errors import ConfigDirectoryError

if TYPE_CHECKING:
    from importlib.metadata import EntryPoint

__all__ = [
    "CONFIG_DIRS_GROUP",
    "PACKAGED_CONFIG_ROOT",
    "config_search_path",
    "plugin_config_dirs",
]

CONFIG_DIRS_GROUP: Final = entry_point_group("config_dirs")
"""The entry point group a plugin distribution lists its configuration packages in."""

PACKAGED_CONFIG_ROOT: Final = Path(__file__).parent.parent / "configs"
"""The root of the configuration groups packaged with LoFI."""


def config_search_path(project_root: Path, /) -> tuple[Path, ...]:
    """Return the roots configuration groups are resolved against, in search order.

    Parameters
    ----------
    project_root : Path
        The directory whose ``configs`` subdirectory holds the project's own
        groups. It need not exist; a root without the group directory is
        skipped by the lookup and named among the paths tried when the lookup
        fails.

    Returns
    -------
    tuple[Path, ...]
        The project's ``configs`` directory, then the packaged root, then every
        plugin directory, in the order :func:`plugin_config_dirs` returns them.

    Raises
    ------
    ConfigDirectoryError
        If a plugin's entry point names no importable package.
    """
    return (project_root / "configs", PACKAGED_CONFIG_ROOT, *plugin_config_dirs())


def plugin_config_dirs() -> tuple[Path, ...]:
    """Return the configuration directories installed plugins contribute.

    Each entry point in the ``lofi.config_dirs`` group names a package whose
    directory holds configuration groups. A package spread over several
    directories contributes each of them in turn. Entry points are taken in
    their own order, by name and then by value, so the result does not depend
    on installation order.

    Returns
    -------
    tuple[Path, ...]
        One directory per package location, in entry point order.

    Raises
    ------
    ConfigDirectoryError
        If an entry point names an attribute rather than a package, names a
        module that is not a package, or names a package that cannot be
        imported.
    """
    directories: list[Path] = []

    for point in Registry[object]("config_dirs").entries():
        directories.extend(_package_directories(point))

    return tuple(directories)


def _package_directories(point: EntryPoint) -> tuple[Path, ...]:
    """Return the directories of the package one entry point names.

    Parameters
    ----------
    point : EntryPoint
        An entry point of the ``lofi.config_dirs`` group.

    Returns
    -------
    tuple[Path, ...]
        Every location of the named package, in the package's own order.

    Raises
    ------
    ConfigDirectoryError
        If the entry point names an attribute, a plain module, or a package
        that cannot be found or imported.
    """
    if point.attr:
        raise ConfigDirectoryError(
            entry=point.name,
            value=point.value,
            reason="names an attribute, and a configuration directory is a package",
        )

    try:
        spec = find_spec(point.module)

    except ModuleNotFoundError as missing:
        raise ConfigDirectoryError(
            entry=point.name,
            value=point.value,
            reason=f"cannot be imported: {missing}",
        ) from missing

    if spec is None:
        raise ConfigDirectoryError(
            entry=point.name,
            value=point.value,
            reason="is not an installed package",
        )

    if spec.submodule_search_locations is None:
        raise ConfigDirectoryError(
            entry=point.name,
            value=point.value,
            reason="is a module rather than a package, so it holds no directory",
        )

    return tuple(Path(location) for location in spec.submodule_search_locations)
