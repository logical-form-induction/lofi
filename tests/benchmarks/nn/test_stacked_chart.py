"""Benchmarks recurrent neural-chart activation memory by declared depth."""

from typing import TYPE_CHECKING, Final, override

import torch
from tests._controllers import mlp_controller
from torch import nn

from lofi.judge import Action, ActionSet
from lofi.nn._values import Encoded
from lofi.nn.parsers import (
    DeepInsideOutside,
    LogSumExpReduction,
    MarginalSplitWeigher,
    SpanFunctionModule,
    SpanTransition,
    StackedInsideOutside,
)
from lofi.nn.parsers.composition import MeanFieldComposition
from lofi.nn.psi import RuleScorer
from lofi.nn.sites import ComponentId

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture

INPUT_WIDTH: Final = 8
WIDTH: Final = 8
DEPTHS: Final = (1, 2, 3, 4)
ACTIONS: Final = ActionSet(
    (Action("apply", "application"), Action("keep")),
    Action("keep"),
)


class BenchmarkSpanFunction(SpanFunctionModule):
    """Small direct function independent of registry discovery."""

    def __init__(self, inputs: tuple[int, ...], output: int) -> None:
        super().__init__(inputs, output)
        self._linear = nn.Linear(sum(inputs), output)

    @override
    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        """Project the concatenated inputs through tanh."""
        return torch.tanh(self._linear(self.joined(inputs)))


def function(*inputs: int, output: int = WIDTH) -> BenchmarkSpanFunction:
    """Return one small span function for the benchmark stack."""
    return BenchmarkSpanFunction(inputs, output)


def layer(seed: int) -> DeepInsideOutside:
    """Return one independent inside-outside layer."""
    torch.manual_seed(seed)
    controller = mlp_controller(
        ACTIONS,
        width=WIDTH,
        component=ComponentId(f"controller-{seed}"),
        hidden=WIDTH,
    )

    return DeepInsideOutside(
        RuleScorer(controller, width=WIDTH),
        composition=MeanFieldComposition(
            function(INPUT_WIDTH), function(WIDTH, WIDTH), width=WIDTH
        ),
        split_left=function(WIDTH, WIDTH),
        split_right=function(WIDTH, WIDTH),
        outside_weights="contextual",
        root_outside="learned",
        root_init_gain=1.0,
        component=ComponentId(f"chart-{seed}"),
        reduction=LogSumExpReduction(),
        weigher=MarginalSplitWeigher(),
        checkpoint_from_width=None,
    )


def transition(seed: int) -> SpanTransition:
    """Return one independent completed-span transition."""
    torch.manual_seed(seed)

    return SpanTransition(function(2 * WIDTH), function(2 * WIDTH))


def stack(depth: int) -> StackedInsideOutside:
    """Return a recurrent chart at one measured depth."""
    return StackedInsideOutside(
        tuple(layer(10 + position) for position in range(depth)),
        tuple(transition(30 + position) for position in range(depth - 1)),
    )


def held_activation_bytes(chart: StackedInsideOutside, leaves: Encoded) -> int:
    """Return bytes autograd retains for one forward pass."""
    held = 0

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        nonlocal held
        held += tensor.numel() * tensor.element_size()

        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        chart.readouts(leaves)

    return held


def measured(
    stacks: tuple[StackedInsideOutside, ...], leaves: Encoded
) -> tuple[int, ...]:
    """Return retained activation bytes in depth order."""
    return tuple(held_activation_bytes(chart, leaves) for chart in stacks)


def test_saved_activation_memory_grows_linearly_with_chart_depth(
    benchmark: BenchmarkFixture,
) -> None:
    stacks = tuple(stack(depth) for depth in DEPTHS)
    leaves = Encoded(
        torch.randn(2, 8, INPUT_WIDTH),
        ("item", "word", "width"),
    )
    rows = benchmark.pedantic(
        measured,
        args=(stacks, leaves),
        rounds=1,
        iterations=1,
    )
    one, two, _, four = rows
    per_layer = two - one
    linear = one + per_layer * (DEPTHS[-1] - 1)
    quadratic = one + per_layer * DEPTHS[-1] * (DEPTHS[-1] - 1) // 2
    benchmark.extra_info["activation_bytes_by_depth"] = dict(
        zip(DEPTHS, rows, strict=True)
    )

    assert per_layer > 0
    assert four <= linear * 1.1
    assert four < (linear + quadratic) / 2
