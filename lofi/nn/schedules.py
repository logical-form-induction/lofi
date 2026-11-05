"""Learning-rate schedules over a derived training horizon."""

from typing import TYPE_CHECKING, Final

from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    LinearLR,
    LRScheduler,
    SequentialLR,
)

from lofi._frozen import FrozenValue, Unpicklable
from lofi.registry import ComponentPlugin
from lofi.spec.schedulers.constant import (
    ConstantSchedulerSettings,
    ConstantSchedulerSpec,
)
from lofi.spec.schedulers.cosine import CosineSchedulerSettings, CosineSchedulerSpec
from lofi.spec.schedulers.linear import LinearSchedulerSettings, LinearSchedulerSpec

if TYPE_CHECKING:
    from torch.optim import Optimizer

__all__ = ["CONSTANT_PLUGIN", "COSINE_PLUGIN", "LINEAR_PLUGIN", "ScheduleContext"]


class ScheduleContext(FrozenValue, Unpicklable):
    """The optimizer a schedule drives and the training horizon it spans.

    The horizon is derived from the epoch count and the number of optimizer
    steps each epoch takes, so no separate step total can drift away from the
    loop it describes.

    Parameters
    ----------
    optimizer : Optimizer
        The optimizer whose learning rate is scheduled.
    epochs : int
        How many epochs the training loop takes.
    steps_per_epoch : int
        How many optimizer steps one epoch takes.
    """

    __slots__ = ("_epochs", "_optimizer", "_steps_per_epoch")

    _noun = "schedule context"
    _because = "holds a live optimizer"

    def __init__(
        self, optimizer: Optimizer, *, epochs: int, steps_per_epoch: int
    ) -> None:
        object.__setattr__(self, "_optimizer", optimizer)
        object.__setattr__(self, "_epochs", epochs)
        object.__setattr__(self, "_steps_per_epoch", steps_per_epoch)

    @property
    def optimizer(self) -> Optimizer:
        """Return the optimizer whose learning rate is scheduled."""
        return self._optimizer

    @property
    def total_steps(self) -> int:
        """Return how many optimizer steps the training loop takes."""
        return self._epochs * self._steps_per_epoch


def _build_cosine(
    settings: CosineSchedulerSettings, context: ScheduleContext, /
) -> LRScheduler:
    """Build an optional linear warmup followed by cosine annealing.

    The warmup starts from the declared fraction of the learning rate and the
    annealing reaches its minimum at the last step of the horizon.

    Parameters
    ----------
    settings : CosineSchedulerSettings
        How many optimizer steps the warmup takes and the fraction it starts from.
    context : ScheduleContext
        The optimizer and the horizon.

    Raises
    ------
    ValueError
        If the warmup leaves no step of the horizon for the annealing.
    """
    optimizer = context.optimizer
    warmup = settings.warmup_steps
    annealing = CosineAnnealingLR(
        optimizer, T_max=_remaining(warmup, context, "the cosine decay")
    )

    return _warmed(optimizer, warmup, settings.warmup_start_factor, annealing)


def _remaining(warmup: int, context: ScheduleContext, phase: str, /) -> int:
    """Return the horizon's steps left after a warmup, refusing an empty remainder.

    Parameters
    ----------
    warmup : int
        How many optimizer steps the warmup takes.
    context : ScheduleContext
        The optimizer and the horizon.
    phase : str
        What follows the warmup, for the message.

    Raises
    ------
    ValueError
        If the warmup leaves no step of the horizon.
    """
    total = context.total_steps

    if warmup >= total:
        msg = (
            f"a {warmup}-step warmup leaves no step of the derived {total}-step "
            f"training horizon for {phase}"
        )
        raise ValueError(msg)

    return total - warmup


def _warmed(
    optimizer: Optimizer, warmup: int, start: float, after: LRScheduler, /
) -> LRScheduler:
    """Return a schedule that warms linearly for some steps and then follows another.

    A warmup of no steps returns the following schedule itself.

    Parameters
    ----------
    optimizer : Optimizer
        The optimizer whose learning rate is scheduled.
    warmup : int
        How many optimizer steps the warmup takes.
    start : float
        The fraction of the learning rate the warmup starts from.
    after : LRScheduler
        The schedule the learning rate follows once the warmup is over.
    """
    if not warmup:
        return after

    warming = LinearLR(optimizer, start_factor=start, total_iters=warmup)

    return SequentialLR(optimizer, schedulers=[warming, after], milestones=[warmup])


COSINE_PLUGIN: Final = ComponentPlugin(spec=CosineSchedulerSpec, build=_build_cosine)
"""The cosine schedule's plugin."""


def _build_linear(
    settings: LinearSchedulerSettings, context: ScheduleContext, /
) -> LRScheduler:
    """Build an optional linear warmup followed by linear decay.

    The decay runs from the declared learning rate to its final fraction, reached
    at the last step of the horizon.

    Parameters
    ----------
    settings : LinearSchedulerSettings
        How many optimizer steps the warmup takes, the fraction it starts from
        and the final fraction.
    context : ScheduleContext
        The optimizer and the horizon.

    Raises
    ------
    ValueError
        If the warmup leaves no step of the horizon for the decay.
    """
    optimizer = context.optimizer
    warmup = settings.warmup_steps
    decay = LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=settings.final_factor,
        total_iters=_remaining(warmup, context, "the linear decay"),
    )

    return _warmed(optimizer, warmup, settings.warmup_start_factor, decay)


LINEAR_PLUGIN: Final = ComponentPlugin(spec=LinearSchedulerSpec, build=_build_linear)
"""The linear schedule's plugin."""


def _build_constant(
    settings: ConstantSchedulerSettings, context: ScheduleContext, /
) -> LRScheduler:
    """Build an optional linear warmup followed by the declared learning rate.

    Parameters
    ----------
    settings : ConstantSchedulerSettings
        How many optimizer steps the warmup takes and the fraction it starts from.
    context : ScheduleContext
        The optimizer and the horizon.

    Raises
    ------
    ValueError
        If the warmup leaves no step of the horizon at the declared rate.
    """
    optimizer = context.optimizer
    warmup = settings.warmup_steps
    _remaining(warmup, context, "the constant rate")

    return _warmed(
        optimizer,
        warmup,
        settings.warmup_start_factor,
        ConstantLR(optimizer, factor=1.0, total_iters=0),
    )


CONSTANT_PLUGIN: Final = ComponentPlugin(
    spec=ConstantSchedulerSpec, build=_build_constant
)
"""The constant schedule's plugin."""
