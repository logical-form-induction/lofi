from typing import TYPE_CHECKING, Final, NamedTuple

import pytest
import torch

from lofi.nn.loss._ports import LossTerm
from lofi.train import LossAssembly, OptimizationResult, WeightedObjective
from lofi.train._optimizing import Optimizing, over_no_span, stepped_result

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

    from lofi.nn.loss._ports import PublishedOutputs


class Counted(NamedTuple):
    rows: int


class Squared:
    """Contributes the square of a parameter, with a diagnostic beside it."""

    def __init__(self, value: float, *, finite: bool = True) -> None:
        self.parameter = torch.tensor(value, dtype=torch.float64).requires_grad_()
        self.finite = finite

    @property
    def name(self) -> str:
        return "squared"

    @property
    def requires(self) -> frozenset[str]:
        return frozenset({"batch"})

    def evaluate(self, outputs: PublishedOutputs[Counted], /) -> LossTerm:
        value = self.parameter.pow(2)

        if not self.finite:
            value = value * torch.inf

        return LossTerm(
            value,
            outputs.batch.rows,
            diagnostics={
                "twice": (2 * self.parameter).detach(),
                "nan": torch.tensor(torch.nan, dtype=torch.float64),
            },
        )


class Descent:
    def __init__(self, parameter: Tensor, rate: float = 0.5) -> None:
        self.parameter = parameter
        self.rate = rate

    def step(self, forward: Callable[[], Tensor], /) -> OptimizationResult:
        self.parameter.grad = None
        loss = forward()

        if not bool(torch.isfinite(loss)):
            return OptimizationResult(loss_finite=False, stepped=False, scale=None)

        loss.backward()
        gradient = self.parameter.grad

        if gradient is not None:
            with torch.no_grad():
                self.parameter.sub_(self.rate * gradient)

        return OptimizationResult(loss_finite=True, stepped=True, scale=None)


class Silent:
    def step(self, forward: Callable[[], Tensor], /) -> OptimizationResult:
        del forward

        return OptimizationResult(loss_finite=True, stepped=False, scale=None)


STEPPED: Final = OptimizationResult(loss_finite=True, stepped=True, scale=None)
HELD: Final = OptimizationResult(loss_finite=True, stepped=False, scale=None)


def optimizing(objective: Squared, policy: Descent | Silent) -> Optimizing[Counted]:
    assembly = LossAssembly[Counted](
        (WeightedObjective(objective.name, 2.0, objective),)
    )

    return Optimizing(assembly, policy)


def test_the_outputs_over_no_span_hold_the_batch_and_an_empty_chart() -> None:
    outputs = over_no_span(Counted(4), 4)

    assert outputs.batch == Counted(4)
    assert outputs.spans.n == 0
    assert outputs.chart.inside.tensor.shape == (4, 0)
    assert outputs.chart.outside is None


def test_one_step_moves_the_parameter_and_reports_the_finite_scalars() -> None:
    objective = Squared(3.0)
    built = optimizing(objective, Descent(objective.parameter))

    reported, metrics = built.run(over_no_span(Counted(2), 2), step=0)

    assert reported == STEPPED
    assert objective.parameter.item() == pytest.approx(3.0 - 0.5 * 2.0 * 6.0)
    assert metrics["optimizer/stepped"] == 1.0
    assert metrics["optimizer/loss_finite"] == 1.0
    assert metrics["loss/squared"] == pytest.approx(9.0)
    assert metrics["loss/squared/twice"] == pytest.approx(6.0)
    assert "loss/squared/nan" not in metrics
    assert built.assembly.names == ("squared",)


def test_a_policy_that_never_evaluates_reports_only_its_own_outcome() -> None:
    objective = Squared(3.0)
    built = optimizing(objective, Silent())

    reported, metrics = built.run(over_no_span(Counted(2), 2), step=0)

    assert reported == HELD
    assert metrics == {"optimizer/stepped": 0.0, "optimizer/loss_finite": 1.0}


def test_a_nonfinite_loss_is_reported_and_its_value_withheld() -> None:
    objective = Squared(3.0, finite=False)
    built = optimizing(objective, Descent(objective.parameter))

    reported, metrics = built.run(over_no_span(Counted(2), 2), step=0)

    assert reported == OptimizationResult(loss_finite=False, stepped=False, scale=None)
    assert metrics["optimizer/loss_finite"] == 0.0
    assert "loss/squared" not in metrics
    assert metrics["loss/squared/twice"] == pytest.approx(6.0)


def test_the_result_counts_the_step_only_when_the_policy_stepped() -> None:
    assert stepped_result(4, 3, STEPPED, {"a": 1.0}).step == 5
    assert stepped_result(4, 3, HELD, {}).step == 4
    assert stepped_result(4, 0, None, {}).step == 4

    reported = stepped_result(2, 5, STEPPED, {"a": 1.0}, dropped=2)

    assert (reported.items, reported.dropped, reported.metrics) == (5, 2, {"a": 1.0})
