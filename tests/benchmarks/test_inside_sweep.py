"""Benchmarks the neural chart's inside and outside sweeps with their backward pass."""

from typing import TYPE_CHECKING, Final, NamedTuple

import pytest
import torch
from tests._controllers import mlp_controller

from lofi.judge import Action, ActionSet
from lofi.nn._values import Encoded
from lofi.nn.parsers import (
    DeepInsideOutside,
    LogSumExpReduction,
    MarginalSplitWeigher,
    MeanFieldComposition,
    MlpSpanFunction,
)
from lofi.nn.psi import RuleScorer
from lofi.nn.sites import ComponentId

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture
    from tests.benchmarks.conftest import WallClockBudget
    from torch import Tensor

# the working point the budget names: forty tokens over a batch of thirty-two,
# with cells sixty-four wide.
TOKENS: Final = 40
ITEMS: Final = 32
WIDTH: Final = 64

KEEP: Final = Action("keep")
ACTIONS: Final = ActionSet(
    (Action("apply", "application"), Action("compose", "composition"), KEEP), KEEP
)


class Chart(NamedTuple):
    """Stores the scorer, the leaves it is timed over and where both live.

    Attributes
    ----------
    scorer
        The chart scorer at the working point's width.
    leaves
        Rank 3 over the axes ``("item", "word", "width")``, float32, every item full.
    device
        Where the sweep computes, which the budget states as one accelerator.
    """

    scorer: DeepInsideOutside
    leaves: Encoded
    device: torch.device


def resolve_device() -> torch.device:
    """Return the accelerator this machine offers, or the processor where it has none.

    Returns
    -------
    torch.device
        The device the sweep is timed on.
    """
    resolved = torch.accelerator.current_accelerator()

    if resolved is None or not torch.accelerator.is_available():
        return torch.device("cpu")

    return resolved


def wait(device: torch.device) -> None:
    """Wait for the work already queued on a device.

    Parameters
    ----------
    device : torch.device
        The device to wait on, which the processor never queues work on.
    """
    if device.type != "cpu":
        torch.accelerator.synchronize(device)


@pytest.fixture(scope="module")
def chart() -> Chart:
    """Return the scorer and the batch of leaves for the benchmark.

    Returns
    -------
    Chart
        The scorer, the leaves and the device both were placed on.
    """
    device = resolve_device()
    torch.manual_seed(3)
    controller = mlp_controller(
        ACTIONS, width=WIDTH, component=ComponentId("controller"), hidden=64
    )
    scorer = DeepInsideOutside(
        RuleScorer(controller, width=WIDTH),
        composition=MeanFieldComposition(
            MlpSpanFunction((WIDTH,), WIDTH, layers=1, nonlinearity="tanh"),
            MlpSpanFunction((WIDTH, WIDTH), WIDTH, layers=1, nonlinearity="tanh"),
            width=WIDTH,
        ),
        split_left=MlpSpanFunction(
            (WIDTH, WIDTH), WIDTH, layers=1, nonlinearity="tanh"
        ),
        split_right=MlpSpanFunction(
            (WIDTH, WIDTH), WIDTH, layers=1, nonlinearity="tanh"
        ),
        outside_weights="contextual",
        root_outside="learned",
        root_init_gain=1.0,
        component=ComponentId("chart"),
        reduction=LogSumExpReduction(),
        weigher=MarginalSplitWeigher(),
        checkpoint_from_width=None,
    )
    scorer.to(device)
    leaves = torch.randn(ITEMS, TOKENS, WIDTH, device=device)

    return Chart(scorer, Encoded(leaves, ("item", "word", "width")), device)


def parse(chart: Chart) -> tuple[Tensor, Tensor]:
    """Return the log partition and the constituent marginals after one backward pass.

    Parameters
    ----------
    chart
        The scorer and the batch to run it over.

    Returns
    -------
    tuple of torch.Tensor
        The log partition per item, of shape (item,), and the constituent marginal per
        span, of shape (item, span), both detached.
    """
    chart.scorer.zero_grad()
    readouts = chart.scorer.readouts(chart.leaves)
    log_z = readouts.log_z
    mu = readouts.mu
    cells = readouts.cells

    if log_z is None or mu is None or cells is None:
        pytest.fail("the scorer ran and left a readout it publishes unfilled")

    (log_z.tensor.sum() + mu.tensor.sum() + cells.tensor.sum()).backward()
    wait(chart.device)

    return log_z.tensor.detach(), mu.tensor.detach()


def test_the_sweep_sums_over_every_derivation_of_the_batch(chart: Chart) -> None:
    log_z, mu = parse(chart)

    marginals = mu.cpu()

    assert log_z.shape == (ITEMS,)
    # every binary derivation brackets its n leaves and n - 1 wider spans
    assert torch.allclose(marginals[:, :TOKENS], torch.ones(ITEMS, TOKENS), atol=1e-4)
    assert torch.allclose(
        marginals[:, TOKENS:].sum(dim=-1),
        torch.full((ITEMS,), float(TOKENS - 1)),
        atol=1e-3,
    )


def test_a_batch_is_parsed_and_differentiated_within_budget(
    benchmark: BenchmarkFixture, chart: Chart, wall_clock_budget: WallClockBudget
) -> None:
    benchmark(parse, chart)

    measured = benchmark.stats

    if measured is None:
        pytest.fail("the sweep ran but pytest-benchmark timed nothing")

    # use the median to reduce sensitivity to scheduler delays
    per_batch_ms = measured.stats.median * 1e3
    benchmark.extra_info["tokens"] = TOKENS
    benchmark.extra_info["items"] = ITEMS
    benchmark.extra_info["width"] = WIDTH
    benchmark.extra_info["device"] = str(chart.device)
    benchmark.extra_info["ms_per_batch"] = per_batch_ms
    benchmark.extra_info["mean_ms_per_batch"] = measured.stats.mean * 1e3
    benchmark.extra_info["fastest_ms_per_batch"] = measured.stats.min * 1e3
    benchmark.extra_info["budget"] = "inside-sweep"

    # enforce the 2,500 ms per-batch budget, five times the quiet median measured
    # on an M1 Max under mps
    if wall_clock_budget.enforced:
        assert per_batch_ms <= 2500
