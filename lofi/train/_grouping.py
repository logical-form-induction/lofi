"""Resolution of declared parameter groups against a wired model."""

from itertools import combinations
from typing import TYPE_CHECKING, Never, final, override

from lofi.train._groups import GroupError, ParameterGroups, qualified

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from torch.nn import Module, Parameter

    from lofi.train._groups import ActiveGroups, ParameterGroup
    from lofi.train._ports import ParameterDeclaration

__all__ = ["GroupResolution", "declare", "resolve_groups"]


@final
class GroupResolution:
    """What a declaration and an active set select from a model.

    Parameters
    ----------
    parameters : Iterable[Parameter]
        The parameters an optimizer steps over, in registration order.
    membership : Mapping[str, frozenset[str]]
        The registered names of the parameters each declared group holds.
    active : ActiveGroups
        The groups the run trains.
    """

    __slots__ = ("_active", "_membership", "_parameters")

    def __init__(
        self,
        parameters: Iterable[Parameter],
        membership: Mapping[str, frozenset[str]],
        active: ActiveGroups,
    ) -> None:
        object.__setattr__(self, "_parameters", tuple(parameters))
        object.__setattr__(self, "_membership", dict(membership))
        object.__setattr__(self, "_active", active)

    @override
    def __setattr__[ValueT](self, name: str, value: ValueT) -> Never:
        """Raise AttributeError on field reassignment."""
        del name, value
        msg = "a group resolution is immutable"

        raise AttributeError(msg)

    @property
    def parameters(self) -> tuple[Parameter, ...]:
        """Return the parameters an optimizer steps over, each once."""
        return self._parameters

    @property
    def active(self) -> ActiveGroups:
        """Return the groups the run trains."""
        return self._active

    @property
    def trainable(self) -> frozenset[str]:
        """Return the registered names of the parameters that take a gradient."""
        return frozenset(
            name
            for group, held in self._membership.items()
            if group in self._active
            for name in held
        )

    @property
    def frozen(self) -> frozenset[str]:
        """Return the registered names of the parameters that take no gradient."""
        return frozenset(
            name
            for group, held in self._membership.items()
            if group not in self._active
            for name in held
        )

    def members(self, group: str) -> frozenset[str]:
        """Return the registered names of the parameters one group holds.

        A parameter that several groups select belongs to the most specific of
        them, so the groups hold disjoint sets.

        Parameters
        ----------
        group : str
            The name of a declared group.

        Returns
        -------
        frozenset[str]
            The names, empty for a group every more specific group covers.

        Raises
        ------
        GroupError
            If no group was declared under that name.
        """
        try:
            return self._membership[group]

        except KeyError as error:
            declared = ", ".join(sorted(self._membership))
            msg = f"no group named {group!r} was declared; declared: {declared}"

            raise GroupError(msg) from error

    @override
    def __repr__(self) -> str:
        """Return how many groups the resolution covers and what it left trainable."""
        return (
            f"GroupResolution(groups={len(self._membership)}, "
            f"trainable={len(self._parameters)}, frozen={len(self.frozen)})"
        )


def declare(components: Mapping[str, ParameterDeclaration]) -> ParameterGroups:
    """Return the groups a model's components declare, under the names it gives them.

    Parameters
    ----------
    components : Mapping[str, ParameterDeclaration]
        Each component of the model, keyed by the name the model registers it
        under.

    Returns
    -------
    ParameterGroups
        One declaration over the model, holding every component's groups.

    Raises
    ------
    GroupError
        If two components declare a group under the same qualified name.
    """
    return ParameterGroups(
        group
        for prefix in sorted(components)
        for group in components[prefix].parameter_groups.under(prefix)
    )


def _layer_of(module: Module, name: str, position: int) -> Module:
    """Return one layer of a module.

    Parameters
    ----------
    module : Module
        The module holding the layers.
    name : str
        Its registered name, for the message.
    position : int
        The position the layer is registered under.

    Returns
    -------
    Module
        The layer.

    Raises
    ------
    GroupError
        If the module registers no child under that position.
    """
    children = dict(module.named_children())

    try:
        return children[str(position)]

    except KeyError as error:
        registered = ", ".join(children) or "nothing"
        msg = (
            f"the module {name!r} registers no layer at position {position}; "
            f"it registers {registered}"
        )

        raise GroupError(msg) from error


def _selected(group: ParameterGroup, modules: Mapping[str, Module]) -> frozenset[str]:
    """Return the registered names of the parameters one group selects.

    Parameters
    ----------
    group : ParameterGroup
        The group to select for.
    modules : Mapping[str, Module]
        Every module of the model, by registered name.

    Returns
    -------
    frozenset[str]
        The names.

    Raises
    ------
    GroupError
        If the group names a module the model does not register, a layer the
        module does not register, or nothing that holds a parameter.
    """
    try:
        module = modules[group.module]

    except KeyError as error:
        msg = (
            f"the model registers no module named {group.module!r}; "
            f"it registers {sorted(modules)}"
        )

        raise GroupError(msg) from error

    if group.layers is None:
        roots = ((group.module, module),)
    else:
        roots = tuple(
            (
                qualified(group.module, str(position)),
                _layer_of(module, group.module, position),
            )
            for position in group.layers
        )

    selected = frozenset(
        qualified(prefix, name)
        for prefix, root in roots
        for name, _ in root.named_parameters()
    )

    if not selected:
        msg = f"the group {group.name!r} selects nothing that holds a parameter"

        raise GroupError(msg)

    return selected


def _nested(selection: Mapping[str, frozenset[str]]) -> None:
    """Refuse two groups neither of which is the other's specialization.

    Parameters
    ----------
    selection : Mapping[str, frozenset[str]]
        What each group selects, by group name.

    Raises
    ------
    GroupError
        If two groups select the same parameters, or share a parameter while
        each selects one the other does not.
    """
    for left, right in combinations(sorted(selection), 2):
        one, other = selection[left], selection[right]

        if not one & other:
            continue

        if one == other:
            msg = f"the groups {left!r} and {right!r} select the same parameters"

            raise GroupError(msg)

        if not (one <= other or other <= one):
            shared = ", ".join(sorted(one & other))
            msg = (
                f"the groups {left!r} and {right!r} share {shared} while neither "
                f"selects everything the other does"
            )

            raise GroupError(msg)


def _ownership(
    selection: Mapping[str, frozenset[str]], names: Iterable[str]
) -> dict[str, str]:
    """Return the group each parameter belongs to, by registered name.

    Parameters
    ----------
    selection : Mapping[str, frozenset[str]]
        What each group selects, by group name.
    names : Iterable[str]
        The registered names of the model's parameters.

    Returns
    -------
    dict[str, str]
        The name of the most specific group selecting each parameter.

    Raises
    ------
    GroupError
        If no group selects a parameter the model holds.
    """
    owners: dict[str, str] = {}

    for name in names:
        selecting = sorted(
            (len(selection[group]), group)
            for group in selection
            if name in selection[group]
        )

        if not selecting:
            msg = f"no declared group holds the parameter {name!r}"

            raise GroupError(msg)

        owners[name] = selecting[0][1]

    return owners


def resolve_groups(
    groups: ParameterGroups, active: ActiveGroups, model: Module
) -> GroupResolution:
    """Set a model's gradients from a declaration and return what stays trainable.

    Every parameter of every active group requires a gradient afterwards and
    every other parameter of the model requires none, whatever either required
    before, so resolving twice resolves to the same thing. A parameter several
    groups select belongs to the most specific of them, which is what lets a
    group hold the layers of an encoder while a group nested inside it holds an
    adapter those layers run.

    Parameters
    ----------
    groups : ParameterGroups
        What the model's components declare.
    active : ActiveGroups
        Which of the declared groups the run trains.
    model : Module
        The wired model.

    Returns
    -------
    GroupResolution
        The parameters an optimizer steps over and what each group holds.

    Raises
    ------
    GroupError
        If the configuration names an undeclared group, a group names something
        the model does not register or selects no parameter, two groups overlap
        without one specializing the other, or the model holds a parameter no
        group selects.
    """
    undeclared = sorted(active.names - groups.names)

    if undeclared:
        declared = ", ".join(sorted(groups.names))
        msg = (
            f"the configuration names undeclared groups {undeclared}; "
            f"declared: {declared}"
        )

        raise GroupError(msg)

    modules = dict(model.named_modules())
    selection = {group.name: _selected(group, modules) for group in groups}
    _nested(selection)

    parameters = dict(model.named_parameters())
    owners = _ownership(selection, parameters)
    membership = {
        group: frozenset(name for name, owner in owners.items() if owner == group)
        for group in selection
    }
    trainable = frozenset(
        name for name, owner in owners.items() if owner in active.names
    )

    for name, parameter in parameters.items():
        parameter.requires_grad = name in trainable

    stepped = tuple(
        parameter for name, parameter in parameters.items() if name in trainable
    )

    return GroupResolution(stepped, membership, active)
