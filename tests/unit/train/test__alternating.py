import re
from typing import TYPE_CHECKING, Final, NamedTuple

import pytest
import torch

from lofi.mixing import Interleaving
from lofi.nn.loss._errors import MissingOutputError
from lofi.nn.loss._ports import LossTerm
from lofi.train import LossAssembly, OptimizationResult, WeightedObjective
from lofi.train._alternating import (
    AlternatingStep,
    DatasetTask,
    Fitted,
)
from lofi.train._optimizing import over_no_span
from lofi.train._ports import TrainStep
from lofi.training_plugins.steps import FittingTask

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from torch import Tensor

    from lofi.nn.loss._ports import PublishedOutputs
    from lofi.nn.loss._values import ObjectiveOutputs

STEPPED: Final = OptimizationResult(loss_finite=True, stepped=True, scale=None)

HELD: Final = OptimizationResult(loss_finite=True, stepped=False, scale=None)


class Words(NamedTuple):
    words: tuple[tuple[str, ...], ...]


class Recording:
    """A task that records which batch of its dataset each step was taken on."""

    def __init__(self, name: str, batches: int, *, steps: bool = True) -> None:
        self._name = name
        self._batches = batches
        self._steps = steps
        self.visited: list[int] = []
        self.steps: list[int] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def batches(self) -> int:
        return self._batches

    @property
    def optimizer_steps(self) -> int:
        return self._batches if self._steps else 0

    def fit(self, position: int, step: int, /) -> Fitted:
        self.visited.append(position)
        self.steps.append(step)

        return Fitted(STEPPED if self._steps else HELD, 2, {"loss": float(position)})


class Squared:
    """Contributes the square of one task's own parameter."""

    def __init__(self, value: float, name: str = "squared") -> None:
        self.parameter = torch.tensor(value, dtype=torch.float64).requires_grad_()
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def requires(self) -> frozenset[str]:
        return frozenset({"batch"})

    def evaluate(self, outputs: PublishedOutputs[Words], /) -> LossTerm:
        return LossTerm(self.parameter.pow(2), len(outputs.batch.words))


class Chartless:
    """A head that publishes the batch over no span."""

    def __init__(self, publishes: frozenset[str] = frozenset({"batch"})) -> None:
        self._publishes = publishes

    @property
    def publishes(self) -> frozenset[str]:
        return self._publishes

    def publish(self, batch: Words, /) -> ObjectiveOutputs[Words]:
        return over_no_span(batch, len(batch.words))


class Descent:
    """A policy that counts the steps it takes on one parameter."""

    def __init__(self, parameter: Tensor, rate: float = 0.1) -> None:
        self.parameter = parameter
        self.rate = rate
        self.steps = 0

    def step(self, forward: Callable[[], Tensor], /) -> OptimizationResult:
        self.parameter.grad = None
        loss = forward()
        loss.backward()
        gradient = self.parameter.grad

        if gradient is not None:
            with torch.no_grad():
                self.parameter.sub_(self.rate * gradient)

        self.steps += 1

        return STEPPED


class Fitting(NamedTuple):
    task: DatasetTask[Words]
    objective: Squared
    policy: Descent


def batches(sizes: Sequence[int]) -> tuple[Words, ...]:
    return tuple(Words((("a",),) * size) for size in sizes)


def task(name: str, sizes: Sequence[int], value: float = 2.0) -> Fitting:
    objective = Squared(value, name)
    assembly = LossAssembly[Words]((WeightedObjective(name, 1.0, objective),))
    policy = Descent(objective.parameter)

    return Fitting(
        DatasetTask(batches(sizes), Chartless(), assembly, policy, name=name),
        objective,
        policy,
    )


def alternating(
    tasks: Sequence[FittingTask], interleaving: Interleaving
) -> AlternatingStep:
    return AlternatingStep(tasks, interleaving, role="parser")


@pytest.mark.parametrize(
    ("interleaving", "expected"),
    [
        (Interleaving.round_robin(), {"judgments": 5, "types": 5}),
        (Interleaving.proportional(), {"judgments": 3, "types": 5}),
        (Interleaving.ratio([2, 1]), {"judgments": 10, "types": 5}),
    ],
)
def test_each_dataset_takes_the_steps_the_declared_policy_gives_it(
    interleaving: Interleaving, expected: dict[str, int]
) -> None:
    tasks = [Recording("judgments", 3), Recording("types", 5)]
    step = alternating(tasks, interleaving)

    outcome = step.run(0)

    assert dict(step.counts) == expected
    assert {held.name: len(held.visited) for held in tasks} == expected
    assert outcome.result.metrics["task/judgments/steps"] == expected["judgments"]
    assert outcome.result.metrics["task/types/steps"] == expected["types"]


def test_a_task_whose_batches_run_out_inside_an_epoch_starts_over_at_its_first() -> (
    None
):
    tasks = [Recording("small", 2), Recording("large", 5)]

    alternating(tasks, Interleaving.round_robin()).run(0)

    assert tasks[0].visited == [0, 1, 0, 1, 0]
    assert tasks[1].visited == [0, 1, 2, 3, 4]


def test_each_task_is_fitted_at_the_global_count_of_steps_taken_before_it() -> None:
    tasks = [Recording("small", 2), Recording("held", 2, steps=False)]

    alternating(tasks, Interleaving.round_robin()).run(7)

    assert tasks[0].steps == [7, 8]
    assert tasks[1].steps == [8, 9]


def test_proportional_alternation_visits_every_batch_of_every_dataset_once() -> None:
    tasks = [Recording("small", 3), Recording("large", 4)]

    alternating(tasks, Interleaving.proportional()).run(0)

    assert tasks[0].visited == [0, 1, 2]
    assert tasks[1].visited == [0, 1, 2, 3]


def test_the_schedule_names_the_task_each_step_of_the_epoch_fits() -> None:
    step = alternating(
        [Recording("small", 1), Recording("large", 3)], Interleaving.round_robin()
    )

    assert step.schedule == ("small", "large", "small", "large", "small", "large")
    assert len(step.schedule) == sum(step.counts.values())


def test_the_epoch_counts_one_optimizer_step_for_each_task_step_that_stepped() -> None:
    tasks = [Recording("taking", 2), Recording("holding", 2, steps=False)]

    outcome = alternating(tasks, Interleaving.proportional()).run(6)

    assert outcome.result.step == 8
    assert outcome.result.items == 8
    assert outcome.outputs.role == "parser"


def test_each_task_reports_its_own_scalars_under_its_own_name() -> None:
    tasks = [Recording("judgments", 2), Recording("types", 4)]

    outcome = alternating(tasks, Interleaving.proportional()).run(0)
    metrics = outcome.result.metrics

    assert metrics["task/judgments/loss"] == pytest.approx(0.5)
    assert metrics["task/types/loss"] == pytest.approx(1.5)
    assert metrics["task/judgments/items"] == 4.0
    assert metrics["task/types/items"] == 8.0


def test_an_alternating_step_declares_what_it_reads_and_the_role_it_writes() -> None:
    step = alternating(
        [Recording("a", 1), Recording("b", 1)], Interleaving.round_robin()
    )

    assert isinstance(step, TrainStep)
    assert step.name == "alternating:parser"
    assert step.role == "parser"
    assert step.requires == {"step"}
    assert step.produces == {"parser"}
    assert step.interleaving == Interleaving.round_robin()
    assert len(step.tasks) == 2
    assert repr(step).startswith("AlternatingStep(name='alternating:parser'")


def test_an_alternating_step_may_be_named_apart_from_its_role() -> None:
    step = AlternatingStep(
        [Recording("a", 1), Recording("b", 1)],
        Interleaving.round_robin(),
        role="parser",
        name="stage-one",
    )

    assert step.name == "stage-one"


def test_an_alternating_fit_runs_over_at_least_two_tasks() -> None:
    with pytest.raises(ValueError, match="at least two tasks, not 1"):
        alternating([Recording("only", 2)], Interleaving.round_robin())


def test_two_tasks_may_not_share_a_name() -> None:
    with pytest.raises(ValueError, match=re.escape("two tasks are called ['same']")):
        alternating(
            [Recording("same", 1), Recording("same", 2)], Interleaving.round_robin()
        )


def test_a_task_holding_no_batch_is_refused() -> None:
    with pytest.raises(ValueError, match="task 1 holds no batch"):
        alternating([Recording("a", 2), Recording("b", 0)], Interleaving.proportional())


def test_a_ratio_weights_every_task_or_none_of_them() -> None:
    with pytest.raises(ValueError, match="over 3 tasks does not weight 2"):
        alternating(
            [Recording("a", 1), Recording("b", 1)], Interleaving.ratio([1, 1, 1])
        )


def test_an_epoch_starts_at_a_nonnegative_step_count() -> None:
    step = alternating(
        [Recording("a", 1), Recording("b", 1)], Interleaving.round_robin()
    )

    with pytest.raises(ValueError, match="step is negative"):
        step.run(-1)


def test_two_datasets_step_their_own_policies_as_often_as_the_policy_says() -> None:
    judgments = task("judgments", [1, 2, 3])
    types = task("types", [1, 2, 3, 4, 5])

    outcome = alternating(
        [judgments.task, types.task], Interleaving.proportional()
    ).run(0)

    assert (judgments.policy.steps, types.policy.steps) == (3, 5)
    assert outcome.result.step == 8
    assert outcome.result.items == 6 + 15
    assert outcome.result.metrics["task/judgments/steps"] == 3.0
    assert outcome.result.metrics["task/types/steps"] == 5.0
    assert outcome.result.metrics["task/judgments/loss/judgments"] > 0.0
    assert outcome.result.metrics["task/types/loss/types"] > 0.0


def test_each_dataset_moves_its_own_parameter_and_no_other_task_s() -> None:
    judgments = task("judgments", [1, 2], value=2.0)
    types = task("types", [1, 2, 3, 4], value=3.0)

    alternating([judgments.task, types.task], Interleaving.ratio([1, 2])).run(0)

    assert (judgments.policy.steps, types.policy.steps) == (2, 4)
    assert judgments.objective.parameter.item() == pytest.approx(2.0 * 0.8**2)
    assert types.objective.parameter.item() == pytest.approx(3.0 * 0.8**4)


def test_a_dataset_task_names_its_size_its_head_and_its_objectives() -> None:
    held = task("judgments", [1, 2, 3]).task

    assert isinstance(held, FittingTask)
    assert held.name == "judgments"
    assert held.batches == 3
    assert held.forward.publishes == {"batch"}
    assert held.assembly.names == ("judgments",)
    assert repr(held).startswith("DatasetTask(name='judgments', batches=3")


def test_a_dataset_task_refuses_a_batch_it_does_not_hold() -> None:
    held = task("judgments", [1, 2]).task

    with pytest.raises(IndexError, match="holds 2 batches and was asked for"):
        held.fit(2, 0)

    with pytest.raises(IndexError, match="holds 2 batches and was asked for"):
        held.fit(-1, 0)


def test_a_dataset_task_holds_at_least_one_batch() -> None:
    objective = Squared(2.0)
    assembly = LossAssembly[Words]((WeightedObjective("squared", 1.0, objective),))

    with pytest.raises(ValueError, match="holds no batch to fit on"):
        DatasetTask((), Chartless(), assembly, Descent(objective.parameter), name="e")


def test_a_batch_of_no_item_takes_no_step() -> None:
    objective = Squared(2.0)
    assembly = LossAssembly[Words]((WeightedObjective("squared", 1.0, objective),))
    policy = Descent(objective.parameter)
    held = DatasetTask((Words(()),), Chartless(), assembly, policy, name="empty")

    fitted = held.fit(0, 0)

    assert held.batches == 1
    assert held.optimizer_steps == 0
    assert fitted == Fitted(None, 0, {})
    assert policy.steps == 0


def test_a_dataset_task_derives_steps_separately_from_batch_traversal() -> None:
    held = task("mixed", [1, 0, 2]).task

    assert held.batches == 3
    assert held.optimizer_steps == 2


def test_a_task_whose_head_does_not_publish_what_its_objectives_read_is_refused() -> (
    None
):
    objective = Squared(2.0)
    assembly = LossAssembly[Words]((WeightedObjective("squared", 1.0, objective),))

    with pytest.raises(MissingOutputError, match="squared"):
        DatasetTask(
            batches([1]),
            Chartless(frozenset({"chart"})),
            assembly,
            Descent(objective.parameter),
            name="unpublished",
        )
