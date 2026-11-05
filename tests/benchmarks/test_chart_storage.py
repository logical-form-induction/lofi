"""Benchmarks differentiable chart storage."""

from typing import TYPE_CHECKING, override

import torch
from torch.overrides import TorchFunctionMode

from lofi.deduce import SpanIndex
from lofi.nn.parsers import TensorChart

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from pytest_benchmark.fixture import BenchmarkFixture


class Allocated(TorchFunctionMode):
    """Tracks tensor bytes allocated within a torch function mode."""

    def __init__(self) -> None:
        super().__init__()
        self.total = 0

    @override
    def __torch_function__[Argument, Result](
        self,
        func: Callable[..., Result],
        types: Sequence[type],
        args: Sequence[Argument] = (),
        kwargs: Mapping[str, Argument] | None = None,
    ) -> Result:
        """Run the call and add what it returned to the running total."""
        result = func(*args, **(kwargs if kwargs is not None else {}))

        if isinstance(result, torch.Tensor):
            self.total += result.nbytes

        return result


def _allocate_charts(
    index: SpanIndex, like: torch.Tensor, allocated: Allocated
) -> tuple[TensorChart, TensorChart, torch.Tensor]:
    """Allocate the inside, outside and split charts while counting their bytes."""
    with allocated:
        inside = TensorChart(index, like, batch=32, width=64)
        outside = TensorChart(index, like, batch=32, width=64)
        splits = like.new_zeros((32, index.num_spans, index.max_splits))

    return inside, outside, splits


def test_allocated_charts_store_both_span_boundaries() -> None:
    index = SpanIndex(40)
    like = torch.zeros((), dtype=torch.float32)
    allocated = Allocated()

    inside, outside, splits = _allocate_charts(index, like, allocated)
    root = torch.full((inside.batch, inside.width), 2.0)
    leaf = torch.full((inside.batch, inside.width), 3.0)
    inside.write(0, index.n, root)
    inside.write(index.n - 1, index.n, leaf)

    assert torch.equal(inside.read(0, index.n), root)
    assert torch.equal(inside.read(index.n - 1, index.n), leaf)
    assert outside.read(0, index.n).shape == root.shape
    assert splits.shape == (32, index.num_spans, index.max_splits)
    assert allocated.total >= splits.nbytes


def test_a_chart_at_the_working_point_fits_its_storage_budget(
    benchmark: BenchmarkFixture,
) -> None:
    # the working point: forty tokens, thirty-two items, sixty-four wide cells.
    index = SpanIndex(40)
    like = torch.zeros((), dtype=torch.float32)
    allocated = Allocated()
    benchmark.pedantic(
        _allocate_charts,
        args=(index, like, allocated),
        rounds=1,
        iterations=1,
    )
    benchmark.extra_info["bytes"] = allocated.total
    benchmark.extra_info["budget"] = "P-20"

    # enforce the 40 MB storage budget
    assert allocated.total <= 40_000_000
