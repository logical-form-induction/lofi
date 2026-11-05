"""The parameter groups a component declares and the ones a run trains."""

from typing import TYPE_CHECKING, Never, final, override

from lofi._checks import ensure_name

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

__all__ = [
    "ActiveGroups",
    "GroupError",
    "ParameterGroup",
    "ParameterGroups",
    "qualified",
]


class GroupError(ValueError):
    """Reports a declaration that does not fit the model resolved against it."""


def qualified(prefix: str, name: str) -> str:
    """Return a name as it reads under a prefix.

    Parameters
    ----------
    prefix : str
        The name the enclosing model registers a component under, empty for the
        model itself.
    name : str
        The name the component uses, empty for the component itself.

    Returns
    -------
    str
        The dotted name, or whichever of the two is not empty.
    """
    if not prefix:
        return name

    if not name:
        return prefix

    return f"{prefix}.{name}"


@final
class ParameterGroup:
    """A named selection of the parameters a component holds.

    Parameters
    ----------
    name : str
        What the group is called, unique within a declaration.
    module : str
        The name the component registers the module under, dotted for a nested
        one, or the empty string for the component itself.
    layers : range | None
        Which of the module's layers to select, by the positions an indexed
        container registers its children under, or None for the module whole.

    Raises
    ------
    ValueError
        If the name is empty, or the range runs backward or starts before the
        first layer.
    """

    __slots__ = ("_layers", "_module", "_name")

    def __init__(self, name: str, module: str, layers: range | None = None) -> None:
        if layers is not None and (layers.start < 0 or layers.step <= 0):
            msg = f"a layer range runs forward from the first layer, not {layers!r}"

            raise ValueError(msg)

        object.__setattr__(self, "_name", ensure_name("a group name", name))
        object.__setattr__(self, "_module", module)
        object.__setattr__(self, "_layers", layers)

    @override
    def __setattr__[ValueT](self, name: str, value: ValueT) -> Never:
        """Raise AttributeError on field reassignment."""
        del name, value
        msg = "a parameter group is immutable"

        raise AttributeError(msg)

    @property
    def name(self) -> str:
        """Return what the group is called."""
        return self._name

    @property
    def module(self) -> str:
        """Return the registered name of the module the group selects from."""
        return self._module

    @property
    def layers(self) -> range | None:
        """Return the layers the group selects, or None for the module whole."""
        return self._layers

    def under(self, prefix: str) -> ParameterGroup:
        """Return the group as a component registered under a prefix declares it.

        Parameters
        ----------
        prefix : str
            The name the enclosing model registers the component under.

        Returns
        -------
        ParameterGroup
            The same selection, named and pointed from the enclosing model.
        """
        return ParameterGroup(
            qualified(prefix, self._name),
            qualified(prefix, self._module),
            self._layers,
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another group has the same name and selection."""
        if type(other) is not ParameterGroup:
            return NotImplemented

        return (
            self._name == other._name
            and self._module == other._module
            and self._layers == other._layers
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the name and the selection."""
        return hash((ParameterGroup, self._name, self._module, self._layers))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this group."""
        if self._layers is None:
            return f"ParameterGroup({self._name!r}, {self._module!r})"

        return f"ParameterGroup({self._name!r}, {self._module!r}, {self._layers!r})"


@final
class ParameterGroups:
    """The parameter groups a component declares over its own modules.

    Parameters
    ----------
    groups : Iterable[ParameterGroup]
        The groups, in any order.

    Raises
    ------
    GroupError
        If two of them share a name.
    """

    __slots__ = ("_groups",)

    def __init__(self, groups: Iterable[ParameterGroup] = ()) -> None:
        by_name: dict[str, ParameterGroup] = {}

        for group in groups:
            if group.name in by_name:
                msg = f"two groups are called {group.name!r}"

                raise GroupError(msg)

            by_name[group.name] = group

        ordered = tuple(by_name[name] for name in sorted(by_name))
        object.__setattr__(self, "_groups", ordered)

    @override
    def __setattr__[ValueT](self, name: str, value: ValueT) -> Never:
        """Raise AttributeError on field reassignment."""
        del name, value
        msg = "a parameter-group declaration is immutable"

        raise AttributeError(msg)

    @property
    def names(self) -> frozenset[str]:
        """Return the names of the declared groups."""
        return frozenset(group.name for group in self._groups)

    def under(self, prefix: str) -> ParameterGroups:
        """Return the declaration as the model registering it under a name reads it.

        Parameters
        ----------
        prefix : str
            The name the enclosing model registers the component under.

        Returns
        -------
        ParameterGroups
            The same groups, named and pointed from the enclosing model.
        """
        return ParameterGroups(group.under(prefix) for group in self._groups)

    def __iter__(self) -> Iterator[ParameterGroup]:
        """Yield the declared groups in name order."""
        return iter(self._groups)

    def __len__(self) -> int:
        """Return how many groups the declaration holds."""
        return len(self._groups)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another declaration holds the same groups."""
        if type(other) is not ParameterGroups:
            return NotImplemented

        return self._groups == other._groups

    @override
    def __hash__(self) -> int:
        """Return a hash over the declared groups."""
        return hash((ParameterGroups, self._groups))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this declaration."""
        listed = ", ".join(repr(group) for group in self._groups)

        return f"ParameterGroups([{listed}])"


@final
class ActiveGroups:
    """The declared parameter groups a run trains.

    Parameters
    ----------
    names : Iterable[str]
        The names of the groups whose parameters take a gradient. Every other
        declared group is frozen.

    Raises
    ------
    ValueError
        If one of the names is empty.
    """

    __slots__ = ("_names",)

    def __init__(self, names: Iterable[str] = ()) -> None:
        active = frozenset(ensure_name("an active group name", name) for name in names)
        object.__setattr__(self, "_names", active)

    @override
    def __setattr__[ValueT](self, name: str, value: ValueT) -> Never:
        """Raise AttributeError on field reassignment."""
        del name, value
        msg = "an active-group configuration is immutable"

        raise AttributeError(msg)

    @property
    def names(self) -> frozenset[str]:
        """Return the names of the groups a run trains."""
        return self._names

    def __contains__(self, name: str) -> bool:
        """Return whether a group is one this configuration trains.

        Parameters
        ----------
        name : str
            The name of a declared group.
        """
        return name in self._names

    def __iter__(self) -> Iterator[str]:
        """Yield the names of the groups a run trains, in order."""
        return iter(sorted(self._names))

    def __len__(self) -> int:
        """Return how many groups a run trains."""
        return len(self._names)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another configuration trains the same groups."""
        if type(other) is not ActiveGroups:
            return NotImplemented

        return self._names == other._names

    @override
    def __hash__(self) -> int:
        """Return a hash over the names of the groups a run trains."""
        return hash((ActiveGroups, self._names))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this configuration."""
        listed = ", ".join(repr(name) for name in self)

        return f"ActiveGroups([{listed}])"
