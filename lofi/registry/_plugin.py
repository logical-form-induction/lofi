"""The plugin a component variant registers under its family's entry point group."""

from typing import Protocol, cast, override

from lofi._frozen import FrozenValue
from lofi.registry._registry import resolve_plugin

__all__ = ["Builder", "ComponentPlugin", "plugin_for"]


class _ConfiguredSpec[V](Protocol):
    """The declaration capability a component plugin consumes."""

    @property
    def kind(self) -> str:
        """Return the installed variant tag."""
        ...

    @classmethod
    def family(cls) -> str:
        """Return the component family."""
        ...

    def settings(self) -> V:
        """Return the validated interior settings."""
        ...


class Builder[V, C, M](Protocol):
    """Builds one component from its resolved settings and its build context."""

    def __call__(self, settings: V, context: C, /) -> M:
        """Return the component the settings describe.

        Parameters
        ----------
        settings : V
            The resolved settings the variant's declaration returns.
        context : C
            What the run supplies to every component it builds.

        Returns
        -------
        M
            The built component.
        """
        ...


class ComponentPlugin[V, C, M](FrozenValue):
    """Pairs a variant's declaration class with the builder of its component.

    Parameters
    ----------
    spec : type[_ConfiguredSpec[V]]
        The declaration class, whose definition registers the variant in its
        family's tagged union.
    build : Builder[V, C, M]
        The builder taking the declaration's settings and a build context.
    """

    __slots__ = ("_build", "_spec")

    _noun = "component plugin"

    def __init__(
        self, *, spec: type[_ConfiguredSpec[V]], build: Builder[V, C, M]
    ) -> None:
        object.__setattr__(self, "_spec", spec)
        object.__setattr__(self, "_build", build)

    @property
    def spec(self) -> type[_ConfiguredSpec[V]]:
        """Return the declaration class."""
        return self._spec

    @property
    def build(self) -> Builder[V, C, M]:
        """Return the builder."""
        return self._build

    def construct(self, declaration: _ConfiguredSpec[V], context: C, /) -> M:
        """Build the component one declaration describes.

        Parameters
        ----------
        declaration : _ConfiguredSpec[V]
            A validated declaration of this plugin's variant.
        context : C
            What the run supplies to the component.

        Returns
        -------
        M
            The built component.
        """
        return self._build(declaration.settings(), context)

    @override
    def __repr__(self) -> str:
        """Return the declaration class the plugin builds from."""
        return f"ComponentPlugin(spec={self._spec.__name__})"


def plugin_for[V](
    declaration: _ConfiguredSpec[V], /
) -> ComponentPlugin[V, object, object]:
    """Return the plugin installed for one declaration's family and tag.

    Parameters
    ----------
    declaration : _ConfiguredSpec[V]
        A validated component declaration.

    Returns
    -------
    ComponentPlugin[V, object, object]
        The plugin the declaration's family lists under its tag.

    Raises
    ------
    UnknownTagError
        If the family lists no plugin under the declaration's tag.
    RegistryError
        If the entry names something other than a component plugin, or a plugin
        for another declaration class.
    """
    plugin = resolve_plugin(
        declaration.family(), declaration.kind, ComponentPlugin, type(declaration)
    )

    return cast("ComponentPlugin[V, object, object]", plugin)
