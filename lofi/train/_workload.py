"""The training steps of one prepared plan, run at chosen positions.

A fitting plan publishes the stack it trains as a timed stack: the steps one
unit of its declared length holds, the positions a step may run at, the shape
of the batches those positions read, and a call that runs the step at one
position. A plan counted in parser epochs runs its parser step over the
epoch's batches, or the task steps of its alternation; a plan counted in
generated-batch steps runs the step scheduled at each position of its horizon.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast, final

from lofi.data import BatchShape
from lofi.train._parsing import Featurized
from lofi.train._pretraining import Draw

if TYPE_CHECKING:
    from collections.abc import Callable

    from lofi.train._alternating import AlternatingStep
    from lofi.train._configured import ParserFittingPlan, StepFittingPlan
    from lofi.train._steps import StepOutcome, TrainResult

__all__ = [
    "AlternatingStack",
    "ParserStack",
    "StepStack",
    "TimedStack",
]


class TimedStack(Protocol):
    """Runs training steps of one prepared stack at chosen positions."""

    @property
    def steps_per_unit(self) -> int:
        """Return the steps one unit of the declared training length holds."""
        ...

    @property
    def positions(self) -> int:
        """Return how many distinct positions a step may be run at."""
        ...

    @property
    def shape(self) -> BatchShape | None:
        """Return the mean sentence length and batch size, or None without sentences."""
        ...

    def shape_at(self, position: int, /) -> BatchShape | None:
        """Return the shape of the batch one position reads, or None without one."""
        ...

    def run(self, position: int, /) -> None:
        """Run the training step at one position.

        Parameters
        ----------
        position : int
            The position, from zero.
        """
        ...


@final
class ParserStack:
    """Runs parser steps over chosen batches of one epoch."""

    __slots__ = ("_completed", "_plan")

    def __init__(self, plan: ParserFittingPlan) -> None:
        self._plan = plan
        self._completed = 0

    @property
    def steps_per_unit(self) -> int:
        """Return the batches one epoch holds."""
        return len(self._plan.batches)

    @property
    def positions(self) -> int:
        """Return the batches one epoch holds."""
        return len(self._plan.batches)

    @property
    def shape(self) -> BatchShape:
        """Return the mean sentence length and batch size over the epoch's batches."""
        batches = self._plan.batches
        lengths = [len(words) for batch in batches for words in batch.words]

        return BatchShape(sum(lengths) / len(lengths), len(lengths) / len(batches))

    def shape_at(self, position: int, /) -> BatchShape:
        """Return the mean sentence length and size of one batch of the epoch."""
        words = self._plan.batches[position].words
        lengths = [len(sentence) for sentence in words]

        return BatchShape(sum(lengths) / len(lengths), float(len(lengths)))

    def run(self, position: int, /) -> None:
        """Run the step over one batch of the epoch."""
        run_step = cast(
            "Callable[[object], StepOutcome[object, TrainResult]]",
            self._plan.step.run,
        )
        outcome = run_step(Featurized(self._plan.batches[position], self._completed))
        self._completed = outcome.result.step


@final
class StepStack:
    """Run the generated-batch step scheduled at chosen positions of the horizon."""

    __slots__ = ("_plan",)

    def __init__(self, plan: StepFittingPlan) -> None:
        self._plan = plan

    @property
    def steps_per_unit(self) -> int:
        """Return one, since a step-counted run declares its length in steps."""
        return 1

    @property
    def positions(self) -> int:
        """Return the declared optimizer-step horizon."""
        return self._plan.steps

    @property
    def shape(self) -> None:
        """Return None, since a generated batch has no sentence shape."""

    def shape_at(self, position: int, /) -> None:
        """Return None, since a generated batch has no sentence shape."""
        del position

    def run(self, position: int, /) -> None:
        """Run the step scheduled at one position over its draw."""
        plan = self._plan
        plan.trainer_at(position).run(
            Draw(
                plan.batch_seed(plan.stream_seed, position),
                plan.batch_size,
                step=position,
                horizon=plan.steps,
            )
        )


@final
class AlternatingStack:
    """Runs individual task steps at positions in one alternating epoch."""

    __slots__ = ("_completed", "_step")

    def __init__(self, step: AlternatingStep) -> None:
        self._step = step
        self._completed = 0

    @property
    def steps_per_unit(self) -> int:
        """Return the task steps in one alternating epoch."""
        return len(self._step.task_positions)

    @property
    def positions(self) -> int:
        """Return the task-step positions in one alternating epoch."""
        return len(self._step.task_positions)

    @property
    def shape(self) -> None:
        """Return None because alternating tasks need not consume sentences."""

    def shape_at(self, position: int, /) -> None:
        """Return None because alternating tasks need not consume sentences."""
        del position

    def run(self, position: int, /) -> None:
        """Run the one task step scheduled at an epoch position."""
        step = self._step
        task_position = step.task_positions[position]
        batch_position = (
            step.task_positions[:position].count(task_position)
            % step.tasks[task_position].batches
        )
        fitted = step.tasks[task_position].fit(batch_position, self._completed)
        result = fitted.result

        if result is not None and result.stepped:
            self._completed += 1
