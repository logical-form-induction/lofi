"""Optimizers over the parameters a run's group resolution selects."""

from typing import TYPE_CHECKING, Final

from torch.optim import SGD, Adam, AdamW

from lofi.registry import ComponentPlugin
from lofi.spec.optimizers.adam import AdamOptimizerSettings, AdamOptimizerSpec
from lofi.spec.optimizers.adamw import AdamWOptimizerSettings, AdamWOptimizerSpec
from lofi.spec.optimizers.sgd import SGDOptimizerSettings, SGDOptimizerSpec

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch.nn import Parameter

__all__ = ["ADAMW_PLUGIN", "ADAM_PLUGIN", "SGD_PLUGIN"]


def _build_adamw(
    settings: AdamWOptimizerSettings, parameters: Sequence[Parameter], /
) -> AdamW:
    """Build AdamW over exactly the parameters supplied, in their order.

    Parameters
    ----------
    settings : AdamWOptimizerSettings
        The step size, the moment decays, the stability term and the weight decay.
    parameters : Sequence[Parameter]
        The parameters the run's group resolution selected.
    """
    return AdamW(
        parameters,
        lr=settings.learning_rate,
        betas=settings.betas,
        eps=settings.epsilon,
        weight_decay=settings.weight_decay,
    )


ADAMW_PLUGIN: Final = ComponentPlugin(spec=AdamWOptimizerSpec, build=_build_adamw)
"""The AdamW optimizer's plugin."""


def _build_adam(
    settings: AdamOptimizerSettings, parameters: Sequence[Parameter], /
) -> Adam:
    """Build Adam over exactly the parameters supplied, in their order.

    Parameters
    ----------
    settings : AdamOptimizerSettings
        The step size, the moment decays, the stability term and the L2 penalty.
    parameters : Sequence[Parameter]
        The parameters the run's group resolution selected.
    """
    return Adam(
        parameters,
        lr=settings.learning_rate,
        betas=settings.betas,
        eps=settings.epsilon,
        weight_decay=settings.weight_decay,
    )


ADAM_PLUGIN: Final = ComponentPlugin(spec=AdamOptimizerSpec, build=_build_adam)
"""The Adam optimizer's plugin."""


def _build_sgd(
    settings: SGDOptimizerSettings, parameters: Sequence[Parameter], /
) -> SGD:
    """Build stochastic gradient descent over exactly the parameters supplied.

    Parameters
    ----------
    settings : SGDOptimizerSettings
        The step size, the momentum, its dampening, the L2 penalty and whether the
        momentum is Nesterov momentum.
    parameters : Sequence[Parameter]
        The parameters the run's group resolution selected, in their order.
    """
    return SGD(
        parameters,
        lr=settings.learning_rate,
        momentum=settings.momentum,
        dampening=settings.dampening,
        weight_decay=settings.weight_decay,
        nesterov=settings.nesterov,
    )


SGD_PLUGIN: Final = ComponentPlugin(spec=SGDOptimizerSpec, build=_build_sgd)
"""The stochastic gradient descent optimizer's plugin."""
