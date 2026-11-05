"""One optimizer step over an assembled loss, and the result a step reports for it."""

from math import isfinite
from typing import TYPE_CHECKING, final

import torch

from lofi.deduce import SpanIndex
from lofi.nn._values import LogDistribution
from lofi.nn.loss._parser import SPAN_AXES
from lofi.nn.loss._values import (
    CellChart,
    ObjectiveOutputs,
)
from lofi.train._steps import TrainResult

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import Tensor

    from lofi.train._losses import LossAssembly, LossResult
    from lofi.train._optimization import OptimizationResult
    from lofi.train._ports import OptimizationPolicy

__all__ = ["Optimizing", "over_no_span", "stepped_result"]


def over_no_span[BatchT](batch: BatchT, rows: int) -> ObjectiveOutputs[BatchT]:
    """Return the outputs a chartless step publishes, which is the batch over no span.

    Parameters
    ----------
    batch : BatchT
        The batch the step drew.
    rows : int
        How many items the batch holds.
    """
    inside = LogDistribution(torch.empty(rows, 0), SPAN_AXES)

    return ObjectiveOutputs(spans=SpanIndex(0), chart=CellChart(inside), batch=batch)


@final
class Optimizing[BatchT]:
    """Evaluates the assembled loss on published outputs and takes one optimizer step.

    Parameters
    ----------
    assembly : LossAssembly[BatchT]
        The weighted objectives the step trains under.
    policy : OptimizationPolicy[Tensor]
        What runs the backward pass and the optimizer on the total.
    """

    __slots__ = ("_assembly", "_policy")

    def __init__(
        self, assembly: LossAssembly[BatchT], policy: OptimizationPolicy[Tensor]
    ) -> None:
        self._assembly = assembly
        self._policy = policy

    @property
    def assembly(self) -> LossAssembly[BatchT]:
        """Return the weighted objectives."""
        return self._assembly

    def run(
        self, outputs: ObjectiveOutputs[BatchT], /, *, step: int
    ) -> tuple[OptimizationResult, dict[str, float]]:
        """Take one optimizer step on published outputs and return its outcome.

        Parameters
        ----------
        outputs : ObjectiveOutputs[BatchT]
            What the graph published for the step.
        step : int
            The count of optimizer steps the run has taken, which sets the weight
            each scheduled term takes.

        Returns
        -------
        tuple[OptimizationResult, dict[str, float]]
            What the policy reported, and the finite scalars of the loss it optimized
            beside whether it stepped and whether the loss was finite.
        """
        evaluated: list[LossResult] = []

        def forward() -> Tensor:
            result = self._assembly.evaluate(outputs, step=step)
            evaluated.append(result)

            return result.total

        reported = self._policy.step(forward)
        metrics = {
            "optimizer/stepped": float(reported.stepped),
            "optimizer/loss_finite": float(reported.loss_finite),
        }

        if evaluated:
            metrics.update(
                (key, value)
                for key, value in evaluated[-1].metrics.items()
                if isfinite(value)
            )

        return reported, metrics


def stepped_result(
    step: int,
    items: int,
    reported: OptimizationResult | None,
    metrics: Mapping[str, float],
    *,
    dropped: int = 0,
) -> TrainResult:
    """Return the result a step reports once its optimizer step is done.

    Parameters
    ----------
    step : int
        The count of optimizer steps the run had taken before this one.
    items : int
        How many items the step consumed.
    reported : OptimizationResult | None
        What the policy reported, or None where nothing was there to optimize on.
    metrics : Mapping[str, float]
        The scalars the step reports.
    dropped : int
        How many items the step declined to consume.

    Returns
    -------
    TrainResult
        The result, at the count of optimizer steps the run has taken once this one
        is done.
    """
    stepped = reported is not None and reported.stepped

    return TrainResult(step + int(stepped), items, dropped, metrics=metrics)
