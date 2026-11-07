import math

import pytest
import torch

from lofi.nn.parsers._ports import LogWeightReduction
from lofi.nn.parsers._reduction import LogSumExpReduction, MaximumReduction

REDUCTIONS: tuple[LogWeightReduction, ...] = (
    LogSumExpReduction(),
    MaximumReduction(),
)


@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_every_shipped_reduction_satisfies_the_port(
    reduction: LogWeightReduction,
) -> None:
    assert isinstance(reduction, LogWeightReduction)


@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_the_port_is_enough_to_run_a_reduction(reduction: LogWeightReduction) -> None:
    weights = torch.tensor([[math.log(0.5), math.log(0.25), 0.0]])
    mask = torch.tensor([[True, True, False]])

    combined = reduction.reduce(weights + weights, mask, dim=1)

    assert combined.shape == (1,)
    assert bool(torch.isfinite(combined).all())
