"""Real parameter storage for matched-capacity model comparisons."""

from types import MappingProxyType
from typing import TYPE_CHECKING, Final, final

import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Mapping

CAPACITY_ROLE: Final = "capacity"
"""The checkpoint role that owns model-capacity padding."""

__all__ = ["CAPACITY_ROLE", "with_parameter_capacity"]


@final
class _ParameterCapacity(nn.Module):
    """Own frozen parameter elements that have no role in model computation."""

    def __init__(self, count: int, reference: nn.Parameter | None, /) -> None:
        super().__init__()
        values = (
            torch.zeros(count)
            if reference is None
            else reference.detach().new_zeros((count,))
        )
        self.padding = nn.Parameter(values, requires_grad=False)


def _parameters(modules: Mapping[str, nn.Module], /) -> tuple[nn.Parameter, ...]:
    """Return the distinct parameters held by the named module topology."""
    distinct: dict[int, nn.Parameter] = {}

    for module in modules.values():
        for parameter in module.parameters():
            distinct.setdefault(id(parameter), parameter)

    return tuple(distinct.values())


def with_parameter_capacity(
    modules: Mapping[str, nn.Module], target: int | None, /
) -> Mapping[str, nn.Module]:
    """Return modules with real frozen storage up to a total parameter target.

    Parameters
    ----------
    modules : Mapping[str, nn.Module]
        The fully built stateful model roles before capacity matching.
    target : int | None
        The desired total number of parameter elements, or None when the model
        keeps its natural capacity.

    Returns
    -------
    Mapping[str, nn.Module]
        The original roles, followed by one independently checkpointed capacity
        role when a target is declared.

    Raises
    ------
    ValueError
        If the capacity role is already occupied or the built model exceeds the
        declared target.

    Notes
    -----
    Padding is frozen because it has no path into the model function. Thus a
    matched target changes the measured total parameter count while leaving the
    measured trainable count equal to the natural model's count.
    """
    held = dict(modules)

    if target is None:
        return MappingProxyType(held)

    if CAPACITY_ROLE in held:
        msg = f"model component role {CAPACITY_ROLE!r} is reserved for capacity padding"
        raise ValueError(msg)

    parameters = _parameters(held)
    natural = sum(parameter.numel() for parameter in parameters)

    if target < natural:
        msg = (
            f"target parameter count {target} is below the built model's count "
            f"{natural}"
        )
        raise ValueError(msg)

    held[CAPACITY_ROLE] = _ParameterCapacity(
        target - natural,
        None if not parameters else parameters[0],
    )

    return MappingProxyType(held)
