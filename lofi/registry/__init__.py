"""Component resolution over installed entry points."""

from lofi.registry._errors import RegistryError, TagCollisionError, UnknownTagError
from lofi.registry._plugin import Builder, ComponentPlugin, plugin_for
from lofi.registry._registry import (
    EntrySource,
    Registry,
    entry_point_group,
    resolve_plugin,
)

__all__ = [
    "Builder",
    "ComponentPlugin",
    "EntrySource",
    "Registry",
    "RegistryError",
    "TagCollisionError",
    "UnknownTagError",
    "entry_point_group",
    "plugin_for",
    "resolve_plugin",
]
