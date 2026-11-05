"""Submodule ownership helpers for wired components."""

from typing import TYPE_CHECKING, NamedTuple

from lofi._checks import ensure_name
from lofi.run._ports import CountedParameter, FreezableModule, ModuleRegistrar

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "FrozenModuleReference",
    "ParameterCount",
    "parameter_count",
    "register_module_collection",
]


class FrozenModuleReference[M: FreezableModule]:
    """Holds a frozen module without registering it as a child module.

    Parameters
    ----------
    module : M
        Module whose registered parameters are frozen.
    """

    __slots__ = ("_module",)

    def __init__(self, module: M) -> None:
        self._module = module.requires_grad_(requires_grad=False)

    @property
    def module(self) -> M:
        """Return the frozen referenced module."""
        return self._module


class ParameterCount(NamedTuple):
    """How many parameter elements a model part holds, and how many train.

    Attributes
    ----------
    total : int
        The elements across its distinct parameters.
    trainable : int
        The elements across those of them that take a gradient.
    """

    total: int
    trainable: int


def parameter_count(parameters: Iterable[CountedParameter], /) -> ParameterCount:
    """Count the elements a collection of parameters holds, and those that train.

    A parameter listed twice, as one two modules share is, is counted once, so
    the total is the size of the model rather than the sum of its parts.

    Parameters
    ----------
    parameters : Iterable[CountedParameter]
        The parameters counted.

    Returns
    -------
    ParameterCount
        The elements across the distinct parameters, and across those of them
        that take a gradient.
    """
    distinct = {id(parameter): parameter for parameter in parameters}

    return ParameterCount(
        sum(parameter.numel() for parameter in distinct.values()),
        sum(
            parameter.numel()
            for parameter in distinct.values()
            if parameter.requires_grad
        ),
    )


def register_module_collection[M](
    owner: ModuleRegistrar[M],
    name: str,
    modules: Iterable[M],
    /,
) -> tuple[M, ...]:
    """Register and retain an ordered collection of child modules.

    Parameters
    ----------
    owner : ModuleRegistrar[M]
        Component that owns the child modules.
    name : str
        Prefix for the registered child names.
    modules : Iterable[M]
        Child modules in their runtime order.

    Returns
    -------
    tuple[M, ...]
        The registered modules in the same order.

    Raises
    ------
    TypeError
        If ``name`` is not a string.
    ValueError
        If ``name`` is empty.
    """
    prefix = ensure_name("module collection", name)
    held = tuple(modules)

    for index, module in enumerate(held):
        owner.add_module(f"{prefix}_{index}", module)

    return held
