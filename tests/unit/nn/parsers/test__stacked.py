"""Tests for recurrent neural inside-outside chart layers."""

from typing import Final, override

import pytest
import torch
from tests._controllers import mlp_controller
from torch import nn

from lofi.deduce import SpanIndex
from lofi.judge import Action, ActionSet
from lofi.nn._values import ChartReadouts, Encoded
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

INPUT_WIDTH: Final = 5
WIDTH: Final = 6
CHART: Final = ComponentId("chart")
CONTROLLER: Final = ComponentId("controller")
ACTIONS: Final = ActionSet(
    (Action("apply", "application"), Action("keep")),
    Action("keep"),
)


class FixtureSpanFunction(SpanFunctionModule):
    """Small direct function independent of the plugin registry."""

    def __init__(self, inputs: tuple[int, ...], output: int) -> None:
        super().__init__(inputs, output)
        self._linear = nn.Linear(sum(inputs), output)

    @override
    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        """Project the concatenated inputs through tanh."""
        return torch.tanh(self._linear(self.joined(inputs)))


def function(*inputs: int, output: int = WIDTH) -> FixtureSpanFunction:
    """Return the deterministic span function used by a layer."""
    return FixtureSpanFunction(inputs, output)


def layer(seed: int) -> DeepInsideOutside:
    """Return one independent chart layer."""
    torch.manual_seed(seed)
    controller = mlp_controller(ACTIONS, width=WIDTH, component=CONTROLLER, hidden=7)

    return DeepInsideOutside(
        RuleScorer(controller, width=WIDTH),
        composition=MeanFieldComposition(
            function(INPUT_WIDTH),
            function(WIDTH, WIDTH),
            width=WIDTH,
        ),
        split_left=function(WIDTH, WIDTH),
        split_right=function(WIDTH, WIDTH),
        outside_weights="contextual",
        root_outside="learned",
        root_init_gain=1.0,
        component=CHART,
        reduction=LogSumExpReduction(),
        weigher=MarginalSplitWeigher(),
        checkpoint_from_width=None,
    ).to(torch.float64)


def transition(seed: int) -> SpanTransition:
    """Return one independent completed-span transition."""
    torch.manual_seed(seed)

    return SpanTransition(
        function(2 * WIDTH, output=INPUT_WIDTH),
        function(2 * WIDTH, output=WIDTH),
    ).to(torch.float64)


def leaves(seed: int = 3) -> Encoded:
    """Return one fully live four-word item."""
    values = torch.randn(
        1,
        4,
        INPUT_WIDTH,
        generator=torch.Generator().manual_seed(seed),
        dtype=torch.float64,
    )
    return Encoded(values, ("item", "word", "width"))


def required(readouts: ChartReadouts, name: str) -> torch.Tensor:
    """Return one required tensor readout."""
    value = getattr(readouts, name)

    if value is None:
        pytest.fail(f"the stacked chart left {name!r} unpublished")

    if name == "actions":
        return value.scores.tensor

    return value.tensor


def test_depth_one_is_the_single_pass_bit_for_bit() -> None:
    base = layer(11)
    stacked = StackedInsideOutside((base,), ())
    values = leaves()
    expected = base.readouts(values)
    found = stacked.readouts(values)

    assert stacked.depth == 1
    assert stacked.layers == (base,)
    assert stacked.publishes == base.publishes

    for name in stacked.publishes:
        assert torch.equal(required(found, name), required(expected, name))


@pytest.mark.parametrize("depth", [2, 3])
def test_deeper_charts_train_through_every_layer(depth: int) -> None:
    layers = tuple(layer(seed) for seed in range(11, 11 + depth))
    transitions = tuple(transition(seed) for seed in range(31, 30 + depth))
    stacked = StackedInsideOutside(layers, transitions)
    result = stacked.readouts(leaves())
    loss = -required(result, "log_z").mean()

    loss.backward()

    assert stacked.depth == depth
    assert all(
        any(parameter.grad is not None for parameter in module.parameters())
        for module in (*layers, *transitions)
    )


def test_each_layer_consumes_only_the_completed_span_tensor_below_it() -> None:
    lower = layer(11)
    upper = layer(12)
    bridge = transition(31)
    stacked = StackedInsideOutside((lower, upper), (bridge,))
    values = leaves()
    lower_result, upper_result = stacked.layer_readouts(values)
    spans = required(lower_result, "spans")
    shifted = spans.clone()
    target = SpanIndex(4).span_id(0, 3)
    shifted[:, target] += 1.0
    changed = stacked.refine(upper, bridge, lower_result, replacement=shifted)

    assert torch.equal(
        required(lower_result, "outside_cells"),
        required(stacked.layer_readouts(values)[0], "outside_cells"),
    )
    assert not torch.equal(required(upper_result, "log_z"), required(changed, "log_z"))


def test_shared_and_independent_layers_are_distinct_parameterizations() -> None:
    shared_layer = layer(11)
    shared_transition = transition(31)
    shared = StackedInsideOutside(
        (shared_layer, shared_layer, shared_layer),
        (shared_transition, shared_transition),
    )
    independent = StackedInsideOutside(
        (layer(11), layer(12), layer(13)),
        (transition(31), transition(32)),
    )

    assert shared.layers[0] is shared.layers[1] is shared.layers[2]
    assert shared.transitions[0] is shared.transitions[1]
    assert independent.layers[0] is not independent.layers[1]
    assert independent.transitions[0] is not independent.transitions[1]
    assert len(tuple(shared.parameters())) < len(tuple(independent.parameters()))


@pytest.mark.parametrize(
    ("layers", "transitions", "message"),
    [
        ((), (), "at least one"),
        ((layer(11), layer(12)), (), "one transition"),
    ],
)
def test_an_invalid_layer_stack_is_refused(
    layers: tuple[DeepInsideOutside, ...],
    transitions: tuple[SpanTransition, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        StackedInsideOutside(layers, transitions)
