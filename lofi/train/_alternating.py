"""Alternating fitting over several datasets under a declared interleaving."""

from __future__ import annotations

from math import fsum
from types import MappingProxyType
from typing import TYPE_CHECKING, NamedTuple, final, override

from lofi._checks import (
    ensure_name,
    ensure_non_negative,
)
from lofi.nn.loss._errors import MissingOutputError
from lofi.train._optimization import OptimizationResult
from lofi.train._optimizing import Optimizing
from lofi.train._pretraining import Trained
from lofi.train._steps import EvalResult, StepOutcome, TrainResult

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from torch import Tensor

    from lofi.mixing import Interleaving
    from lofi.nn.loss._ports import SurfaceItems
    from lofi.train._early_stopping import AccuracyStopping
    from lofi.train._losses import LossAssembly
    from lofi.train._ports import Forward, OptimizationPolicy, TrainStep
    from lofi.training_plugins.steps import (
        FittingOutcome,
        FittingTask,
        OptimizationOutcome,
    )

__all__ = [
    "AlternatingStep",
    "AlternatingTask",
    "DatasetTask",
    "Fitted",
    "StepTask",
]


class Fitted(NamedTuple):
    """What one step on one task's batch consumed and reported."""

    result: OptimizationOutcome | None
    items: int
    metrics: Mapping[str, float]


@final
class DatasetTask[BatchT: SurfaceItems]:
    """One dataset of an alternating fit, with the head and objectives it trains.

    A task owns the batches it is fitted on, the head that publishes what its
    objectives read, and the weights those objectives are assembled under, so two
    tasks stepping one optimizer still train different terms through different
    heads.

    Parameters
    ----------
    batches : Sequence[BatchT]
        One pass over the dataset, as the batches it is drawn in.
    forward : Forward[BatchT]
        This task's own head, which publishes what its objectives read.
    assembly : LossAssembly[BatchT]
        This task's own objectives, under its own weights.
    policy : OptimizationPolicy[Tensor]
        What runs the backward pass and the optimizer.
    name : str
        What the task is called, unique within an alternating step.

    Raises
    ------
    TypeError
        If the name is not a string.
    ValueError
        If the name is empty or the dataset holds no batch.
    MissingOutputError
        If an objective declares a name the head does not publish.
    """

    __slots__ = ("_batches", "_forward", "_name", "_optimizing")

    def __init__(
        self,
        batches: Sequence[BatchT],
        forward: Forward[BatchT],
        assembly: LossAssembly[BatchT],
        policy: OptimizationPolicy[Tensor],
        *,
        name: str,
    ) -> None:
        held = tuple(batches)

        if not held:
            msg = f"the task {name!r} holds no batch to fit on"

            raise ValueError(msg)

        for entry in assembly:
            missing = entry.objective.requires - forward.publishes

            if missing:
                raise MissingOutputError(
                    objective=entry.name,
                    missing=missing,
                    published=forward.publishes,
                )

        self._batches = held
        self._forward = forward
        self._optimizing: Optimizing[BatchT] = Optimizing(assembly, policy)
        self._name = ensure_name("name", name)

    @property
    def name(self) -> str:
        """Return what the task is called."""
        return self._name

    @property
    def batches(self) -> int:
        """Return how many batches one pass over the dataset takes."""
        return len(self._batches)

    @property
    def optimizer_steps(self) -> int:
        """Return how many nonempty batches take an optimizer step."""
        return sum(bool(batch.words) for batch in self._batches)

    @property
    def forward(self) -> Forward[BatchT]:
        """Return the head this task trains through."""
        return self._forward

    @property
    def assembly(self) -> LossAssembly[BatchT]:
        """Return the objectives this task fits under."""
        return self._optimizing.assembly

    def fit(self, position: int, step: int, /) -> Fitted:
        """Take one optimizer step on one batch of the dataset.

        Parameters
        ----------
        position : int
            Which batch to fit on, counted from the first.
        step : int
            The count of optimizer steps the run has taken.

        Returns
        -------
        Fitted
            What the policy reported, how many items the batch held, and the
            scalars of the loss it was fitted under; a batch of no item takes no
            step.

        Raises
        ------
        IndexError
            If the dataset holds no batch at that position.
        """
        if not 0 <= position < len(self._batches):
            msg = (
                f"the task {self._name!r} holds {len(self._batches)} batches and "
                f"was asked for the one at {position}"
            )

            raise IndexError(msg)

        batch = self._batches[position]
        items = len(batch.words)

        if not items:
            return Fitted(None, 0, MappingProxyType({}))

        reported, metrics = self._optimizing.run(
            self._forward.publish(batch), step=step
        )

        return Fitted(reported, items, MappingProxyType(metrics))

    @override
    def __repr__(self) -> str:
        """Return the name, the dataset's size, the head and the objectives."""
        return (
            f"DatasetTask(name={self._name!r}, batches={len(self._batches)}, "
            f"forward={self._forward!r}, assembly={self.assembly!r})"
        )


@final
class StepTask[InputsT, OutputsT]:
    """Adapt a declared low-level train step to one alternating dataset task."""

    __slots__ = (
        "_evaluate",
        "_inputs",
        "_name",
        "_optimizer_steps",
        "_position",
        "_step",
        "_stop",
        "_validation",
    )

    def __init__(
        self,
        inputs: Sequence[InputsT],
        step: TrainStep[InputsT, OutputsT],
        position: Callable[[InputsT, int], InputsT],
        /,
        *,
        name: str,
        optimizer_steps: int,
        validation: Sequence[InputsT] = (),
        evaluate: Callable[[InputsT], EvalResult] | None = None,
        stop: AccuracyStopping | None = None,
    ) -> None:
        held = tuple(inputs)

        if not held:
            msg = f"the task {name!r} holds no batch request to fit on"
            raise ValueError(msg)

        self._inputs = held
        self._step = step
        self._position = position
        self._name = ensure_name("name", name)
        expected = ensure_non_negative("optimizer_steps", optimizer_steps)

        if expected > len(held):
            msg = (
                f"the task {self._name!r} expects {expected} optimizer steps from "
                f"{len(held)} batch requests"
            )
            raise ValueError(msg)

        self._optimizer_steps = expected
        held_validation = tuple(validation)

        if bool(held_validation) != (evaluate is not None and stop is not None):
            msg = (
                "a targeted step task holds validation requests, an evaluator, "
                "and a stopping rule together"
            )
            raise ValueError(msg)

        self._validation = held_validation
        self._evaluate = evaluate
        self._stop = stop

    @property
    def name(self) -> str:
        """Return the declared task name."""
        return self._name

    @property
    def batches(self) -> int:
        """Return how many low-level requests one pass holds."""
        return len(self._inputs)

    @property
    def optimizer_steps(self) -> int:
        """Return how many requests are expected to take an optimizer step."""
        return self._optimizer_steps

    @property
    def stop(self) -> AccuracyStopping | None:
        """Return the held-out stopping rule, when this is a targeted task."""
        return self._stop

    def evaluate(self, step: int, /) -> EvalResult:
        """Evaluate the fixed held-out requests without fitting them."""
        evaluator = self._evaluate

        if evaluator is None:
            msg = f"the task {self._name!r} declares no held-out evaluation"
            raise ValueError(msg)

        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        items = 0

        for inputs in self._validation:
            result = evaluator(self._position(inputs, step))
            items += result.items

            for metric, value in result.metrics.items():
                totals[metric] = totals.get(metric, 0.0) + value * result.items
                counts[metric] = counts.get(metric, 0) + result.items

        return EvalResult(
            items,
            metrics={
                metric: total / counts[metric] for metric, total in totals.items()
            },
        )

    def fit(self, position: int, step: int, /) -> Fitted:
        """Run one positioned request and expose its optimizer outcome."""
        if not 0 <= position < len(self._inputs):
            msg = (
                f"the task {self._name!r} holds {len(self._inputs)} requests and "
                f"was asked for the one at {position}"
            )
            raise IndexError(msg)

        reported = self._step.run(self._position(self._inputs[position], step)).result

        if reported.step not in {step, step + 1}:
            msg = (
                f"the task {self._name!r} moved from step {step} to "
                f"{reported.step}, not by at most one optimizer step"
            )
            raise RuntimeError(msg)

        optimization = (
            None
            if reported.step == step
            else OptimizationResult(
                loss_finite=reported.metrics["optimizer/loss_finite"] == 1.0,
                stepped=True,
                scale=None,
            )
        )
        return Fitted(optimization, reported.items, reported.metrics)


@final
class _Tally:
    """Accumulates what one task's steps over an epoch consumed and reported."""

    __slots__ = ("_items", "_reported", "_steps")

    def __init__(self) -> None:
        self._steps = 0
        self._items = 0
        self._reported: dict[str, list[float]] = {}

    @property
    def steps(self) -> int:
        """Return how many steps the task has taken."""
        return self._steps

    @property
    def items(self) -> int:
        """Return how many items the task has consumed."""
        return self._items

    def take(self, fitted: FittingOutcome, /) -> None:
        """Take one step's outcome into the tally.

        Parameters
        ----------
        fitted : FittingOutcome
            What the step reported.
        """
        self._steps += 1
        self._items += fitted.items

        for name, value in fitted.metrics.items():
            self._reported.setdefault(name, []).append(value)

    def metrics(self, prefix: str, /) -> dict[str, float]:
        """Return the tally under one prefix, averaging what the steps reported.

        Parameters
        ----------
        prefix : str
            What each key is published under.

        Returns
        -------
        dict[str, float]
            The step and item counts, and the mean of each scalar the steps
            reported, which a scalar no step reported is absent from.
        """
        found = {
            f"{prefix}/steps": float(self._steps),
            f"{prefix}/items": float(self._items),
        }
        found.update(
            (f"{prefix}/{name}", fsum(values) / len(values))
            for name, values in self._reported.items()
        )

        return found

    @override
    def __repr__(self) -> str:
        """Return the step and item counts and the scalars taken so far."""
        return (
            f"_Tally(steps={self._steps}, items={self._items}, "
            f"reported={sorted(self._reported)!r})"
        )


@final
class AlternatingStep:
    """Fits several datasets over one epoch under a declared interleaving.

    One run is one epoch: every task takes the steps the interleaving gives it,
    in the order the interleaving puts them, and a task whose batches run out
    inside the epoch starts over at its first batch. How often each dataset is
    visited is therefore the interleaving's to say rather than a loop's, which is
    what lets two datasets of different sizes be fitted under a declared ratio
    instead of by alternating whole passes over each.

    Parameters
    ----------
    tasks : Sequence[FittingTask]
        The datasets to fit, each with its own head and objectives, in the order
        a ratio weights them.
    interleaving : Interleaving
        How many steps each task takes and in what order.
    role : str
        The role the trained state is checkpointed under.
    name : str | None
        The step's name, or None to name it after the role.

    Raises
    ------
    TypeError
        If the role or the name is not a string.
    ValueError
        If the role or the name is empty, if fewer than two tasks are given, if
        two tasks share a name, if a task holds no batch, or if a ratio weights
        another number of tasks than there are.
    """

    __slots__ = ("_counts", "_interleaving", "_name", "_role", "_schedule", "_tasks")

    def __init__(
        self,
        tasks: Sequence[FittingTask],
        interleaving: Interleaving,
        *,
        role: str,
        name: str | None = None,
    ) -> None:
        held = tuple(tasks)

        if len(held) < 2:
            msg = f"an alternating fit runs over at least two tasks, not {len(held)}"

            raise ValueError(msg)

        names = [task.name for task in held]
        repeated = sorted({found for found in names if names.count(found) > 1})

        if repeated:
            msg = f"two tasks are called {repeated}"

            raise ValueError(msg)

        sizes = [task.batches for task in held]

        self._tasks = held
        self._interleaving = interleaving
        self._counts = MappingProxyType(
            dict(zip(names, interleaving.counts(sizes), strict=True))
        )
        self._schedule = interleaving.schedule(sizes)
        self._role = ensure_name("role", role)
        self._name = ensure_name(
            "name", f"alternating:{role}" if name is None else name
        )

    @property
    def name(self) -> str:
        """Return the step name."""
        return self._name

    @property
    def role(self) -> str:
        """Return the checkpoint role."""
        return self._role

    @property
    def requires(self) -> frozenset[str]:
        """Return required value names."""
        return frozenset({"step"})

    @property
    def produces(self) -> frozenset[str]:
        """Return produced value names."""
        return frozenset({self._role})

    @property
    def tasks(self) -> tuple[FittingTask, ...]:
        """Return the datasets the step fits, in the order they were given."""
        return self._tasks

    @property
    def interleaving(self) -> Interleaving:
        """Return how many steps each task takes and in what order."""
        return self._interleaving

    @property
    def counts(self) -> Mapping[str, int]:
        """Return how many steps each task takes in one epoch, by task name."""
        return self._counts

    @property
    def schedule(self) -> tuple[str, ...]:
        """Return the task each step of one epoch fits, by name and in order."""
        return tuple(self._tasks[position].name for position in self._schedule)

    @property
    def task_positions(self) -> tuple[int, ...]:
        """Return each scheduled task's position in the declared task tuple."""
        return self._schedule

    def run(self, inputs: int, /) -> StepOutcome[Trained, TrainResult]:
        """Fit one epoch over the tasks and report what each of them took.

        Parameters
        ----------
        inputs : int
            The count of optimizer steps the run has taken before this epoch.

        Returns
        -------
        StepOutcome[Trained, TrainResult]
            The checkpoint role, and the result over every item the epoch
            consumed, at the count of optimizer steps the run has taken once the
            epoch is done.

        Raises
        ------
        TypeError
            If the count is not an integer.
        ValueError
            If the count is negative.
        """
        taken = ensure_non_negative("step", inputs)
        positions = [0] * len(self._tasks)
        tallies = [_Tally() for _ in self._tasks]
        stepped = 0

        for index in self._schedule:
            task = self._tasks[index]
            fitted = task.fit(positions[index], taken + stepped)
            positions[index] = (positions[index] + 1) % task.batches
            tallies[index].take(fitted)
            stepped += int(fitted.result is not None and fitted.result.stepped)

        metrics: dict[str, float] = {}

        for task, tally in zip(self._tasks, tallies, strict=True):
            metrics.update(tally.metrics(f"task/{task.name}"))

        metrics["optimizer/stepped"] = float(stepped)

        return StepOutcome(
            Trained(self._role),
            TrainResult(
                taken + stepped,
                sum(tally.items for tally in tallies),
                metrics=metrics,
            ),
        )

    @override
    def __repr__(self) -> str:
        """Return the name, the role, the interleaving and each task's steps."""
        return (
            f"AlternatingStep(name={self._name!r}, role={self._role!r}, "
            f"interleaving={self._interleaving!r}, counts={dict(self._counts)!r})"
        )


@final
class AlternatingTask:
    """Expose one nested alternating step as one flattened fitting task.

    Each position names one entry of the nested step's exact schedule. The child
    position is derived from earlier occurrences of that child, so repeated
    outer epochs reproduce the same finite nested epoch without mutable cursors.
    """

    __slots__ = ("_name", "_step")

    def __init__(self, step: AlternatingStep, /, *, name: str) -> None:
        self._step = step
        self._name = ensure_name("name", name)

    @property
    def name(self) -> str:
        """Return the declared nested-task name."""
        return self._name

    @property
    def batches(self) -> int:
        """Return the number of single-child steps in one nested epoch."""
        return len(self._step.task_positions)

    @property
    def optimizer_steps(self) -> int:
        """Return optimizer steps in one complete nested schedule."""
        expected = 0

        for task in self._step.tasks:
            if task.optimizer_steps not in (0, task.batches):
                msg = (
                    f"nested task {task.name!r} expects {task.optimizer_steps} "
                    f"optimizer steps from {task.batches} requests"
                )
                raise ValueError(msg)

        for child_position in self._step.task_positions:
            task = self._step.tasks[child_position]
            expected += int(task.optimizer_steps == task.batches)

        return expected

    def fit(self, position: int, step: int, /) -> Fitted:
        """Fit the child selected at one exact nested schedule position."""
        schedule = self._step.task_positions

        if not 0 <= position < len(schedule):
            msg = (
                f"the task {self._name!r} holds {len(schedule)} scheduled steps "
                f"and was asked for the one at {position}"
            )
            raise IndexError(msg)

        child_position = schedule[position]
        child = self._step.tasks[child_position]
        occurrence = schedule[:position].count(child_position) % child.batches
        fitted = child.fit(occurrence, step)
        metrics = MappingProxyType(
            {
                f"task/{child.name}/{name}": value
                for name, value in fitted.metrics.items()
            }
        )

        return Fitted(fitted.result, fitted.items, metrics)
