"""Tests for item-weighted retained-partition validation."""

from typing import TYPE_CHECKING, cast

import torch
from torch import nn

from lofi.nn.policy import DevicePolicy
from lofi.train._validation import validate_epoch

if TYPE_CHECKING:
    from lofi.data.synthetic.training import SyntheticTrainingBatch
    from lofi.train._losses import LossAssembly
    from lofi.train._parsing import ChartForward


class _Batch:
    """A sized validation batch with one reported metric."""

    __slots__ = ("metric", "size")

    def __init__(self, size: int, metric: float) -> None:
        self.size = size
        self.metric = metric

    def __len__(self) -> int:
        """Return the item occurrence count."""
        return self.size


class _Result:
    """A validation loss result exposing finite metrics."""

    __slots__ = ("metrics",)

    def __init__(self, metric: float) -> None:
        self.metrics = {"loss/example": metric}


class _Forward:
    """A validation forward that publishes its input batch."""

    def publish(self, batch: _Batch, /) -> _Batch:
        """Return the supplied batch."""
        return batch


class _Assembly:
    """A validation assembly that reports the batch's metric."""

    def __init__(self) -> None:
        self.steps: list[int] = []

    def evaluate(self, outputs: _Batch, /, *, step: int) -> _Result:
        """Return the metric assigned to the published batch."""
        self.steps.append(step)

        return _Result(outputs.metric)


def test_validation_weights_metrics_by_item_and_restores_module_modes() -> None:
    parent = nn.Sequential(nn.Linear(1, 1), nn.Dropout())
    parent.train()
    parent[0].eval()
    linear = cast("nn.Linear", parent[0])
    parameter_before = linear.weight.detach().clone()
    batches = cast(
        "tuple[SyntheticTrainingBatch[object], ...]",
        (_Batch(1, 2.0), _Batch(3, 6.0)),
    )

    assembly = _Assembly()
    result = validate_epoch(
        cast("ChartForward[SyntheticTrainingBatch[object]]", _Forward()),
        cast("LossAssembly[SyntheticTrainingBatch[object]]", assembly),
        batches,
        {"model": parent},
        DevicePolicy(torch.device("cpu"), torch.float32),
        step=6,
    )

    assert assembly.steps == [6, 6]
    assert result.items == 4
    assert result.metrics == {"loss/example": 5.0}
    assert parent.training
    assert not parent[0].training
    assert parent[1].training
    assert torch.equal(linear.weight, parameter_before)
