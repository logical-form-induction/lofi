"""Benchmarks a differentiable outside sweep."""

from typing import TYPE_CHECKING, Final, NamedTuple

import pytest
import torch

from lofi.deduce import SpanIndex
from lofi.nn.parsers import InsideKernel, LogSumExpReduction, OutsideKernel

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture
    from torch import Tensor

# the working point the budget names: forty tokens over a batch of thirty-two.
TOKENS: Final = 40
ITEMS: Final = 32


class Chart(NamedTuple):
    """Stores an inside chart, split scores, and their span index.

    Attributes
    ----------
    inside
        Rank 2 over the axes ``("item", "span")``, float32, holding no gradient of its
        own, so a backward pass over the outside sweep stops at the split scores rather
        than running the inside sweep a second time.
    split
        Rank 3 over the axes ``("item", "span", "split")``, float32, taking a gradient.
    index
        The span addressing both tensors are laid out under.
    """

    inside: Tensor
    split: Tensor
    index: SpanIndex


@pytest.fixture(scope="module")
def chart() -> Chart:
    """Return an inside chart for the benchmark batch.

    Returns
    -------
    Chart
        The inside chart, the split scores it was swept from, and their index.
    """
    torch.manual_seed(5)
    index = SpanIndex(TOKENS)
    leaf = torch.randn(ITEMS, TOKENS)
    split = torch.randn(ITEMS, index.num_spans, index.max_splits, requires_grad=True)
    inside = InsideKernel(LogSumExpReduction()).inside(leaf, split.detach(), index)

    return Chart(inside, split, index)


def surround(chart: Chart) -> tuple[Tensor, Tensor]:
    """Return the outside chart and split gradient for one sweep.

    Parameters
    ----------
    chart
        The inside chart to sweep back over.

    Returns
    -------
    tuple of torch.Tensor
        The outside chart, of shape (item, span), and the gradient of its summed weight
        with respect to the split scores, of shape (item, span, split).
    """
    outside = OutsideKernel(LogSumExpReduction()).outside(
        chart.inside, chart.split, chart.index
    )
    outside.exp().sum().backward()

    splits = chart.split.grad

    if splits is None:
        pytest.fail("the sweep ran but no gradient reached the scores")

    chart.split.grad = None

    return (outside.detach(), splits)


def test_the_two_sweeps_agree_on_what_a_derivation_holds(chart: Chart) -> None:
    outside, _ = surround(chart)
    log_z = chart.inside[:, chart.index.span_id(0, TOKENS)].unsqueeze(1)
    marginal = (chart.inside + outside - log_z).exp()

    # every binary derivation has n leaves and n - 1 internal nodes
    assert torch.allclose(
        marginal[:, :TOKENS], torch.ones_like(marginal[:, :TOKENS]), atol=1e-4
    )
    assert torch.allclose(
        marginal.sum(-1), torch.full((ITEMS,), float(2 * TOKENS - 1)), atol=1e-2
    )
    assert outside.shape == (ITEMS, chart.index.num_spans)


def test_a_chart_is_swept_back_over_and_differentiated_at_the_working_point(
    benchmark: BenchmarkFixture, chart: Chart
) -> None:
    benchmark(surround, chart)

    measured = benchmark.stats

    if measured is None:
        pytest.fail("the sweep ran but pytest-benchmark timed nothing")

    benchmark.extra_info["tokens"] = TOKENS
    benchmark.extra_info["items"] = ITEMS
    benchmark.extra_info["median_ms_per_batch"] = measured.stats.median * 1e3
    benchmark.extra_info["mean_ms_per_batch"] = measured.stats.mean * 1e3
    benchmark.extra_info["max_ms_per_batch"] = measured.stats.max * 1e3
