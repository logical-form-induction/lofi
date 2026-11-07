from typing import TYPE_CHECKING

import pytest
import torch
from torch import nn

from lofi.nn._layers import (
    CellStep,
    Gate,
    RecurrentCellPlugin,
    activation,
    feedforward,
    recurrence,
    recurrent_step,
)
from lofi.nn.sites import DTypeClass
from lofi.registry import RegistryError
from lofi.spec.layers import (
    BUILTIN_RECURRENT_CELLS,
    Activation,
    FeedForwardLayer,
    RecurrentCell,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

WIDTH = 3
STATE = 5


@pytest.mark.parametrize(
    ("cell", "kind", "nonlinearity"),
    [
        ("gru", nn.GRU, None),
        ("lstm", nn.LSTM, None),
        ("elman-tanh", nn.RNN, "tanh"),
        ("elman-relu", nn.RNN, "relu"),
    ],
)
@pytest.mark.parametrize("bidirectional", [False, True])
def test_a_recurrence_is_the_named_cell_at_the_given_depth_and_direction(
    cell: RecurrentCell,
    kind: type[nn.RNNBase],
    nonlinearity: str | None,
    *,
    bidirectional: bool,
) -> None:
    built = recurrence(
        cell, inputs=WIDTH, width=STATE, layers=2, bidirectional=bidirectional
    )
    output, _ = built(torch.randn(4, 6, WIDTH))

    assert type(built) is kind
    assert built.num_layers == 2
    assert built.batch_first
    assert (built.input_size, built.hidden_size) == (WIDTH, STATE)
    assert output.shape == (4, 6, STATE * (2 if bidirectional else 1))

    if nonlinearity is not None:
        assert built.nonlinearity == nonlinearity


def _reference(cell: RecurrentCell) -> nn.RNNCellBase:
    """Return torch's single-step cell for a shipped recurrent-cell name."""
    match cell:
        case "gru":
            return nn.GRUCell(WIDTH, STATE)
        case "lstm":
            return nn.LSTMCell(WIDTH, STATE)
        case "elman-relu":
            return nn.RNNCell(WIDTH, STATE, nonlinearity="relu")
        case _:
            return nn.RNNCell(WIDTH, STATE, nonlinearity="tanh")


@pytest.mark.parametrize("cell", BUILTIN_RECURRENT_CELLS)
def test_a_step_follows_the_named_cell_from_its_parameters(
    cell: RecurrentCell,
) -> None:
    torch.manual_seed(5)
    reference = _reference(cell)
    built = recurrent_step(cell, inputs=WIDTH, width=STATE)
    built.load_state_dict(reference.state_dict())
    inputs = torch.randn(4, WIDTH)
    previous = torch.randn(4, STATE)
    memory = torch.randn(4, STATE)
    step = built(inputs, previous, memory)

    if built.carries_memory:
        expected_hidden, expected_memory = reference(inputs, (previous, memory))
    else:
        expected_hidden = reference(inputs, previous)
        expected_memory = expected_hidden

    assert isinstance(step, CellStep)
    assert built.carries_memory is (cell == "lstm")
    assert torch.allclose(step.hidden, expected_hidden, atol=1e-6)
    assert torch.allclose(step.memory, expected_memory, atol=1e-6)
    assert len(step.gates) == len(built.gates)
    assert all(gate.shape == (4, STATE) for gate in step.gates)


@pytest.mark.parametrize(
    ("cell", "gates"),
    [
        (
            "gru",
            (
                Gate("reset", DTypeClass.PROBABILITY),
                Gate("update", DTypeClass.PROBABILITY),
                Gate("candidate", DTypeClass.REAL),
            ),
        ),
        (
            "lstm",
            (
                Gate("input", DTypeClass.PROBABILITY),
                Gate("forget", DTypeClass.PROBABILITY),
                Gate("candidate", DTypeClass.REAL),
                Gate("output", DTypeClass.PROBABILITY),
            ),
        ),
        ("elman-tanh", (Gate("candidate", DTypeClass.REAL),)),
        ("elman-relu", (Gate("candidate", DTypeClass.REAL),)),
    ],
)
def test_a_step_names_the_activations_it_computes_its_state_from(
    cell: RecurrentCell, gates: tuple[Gate, ...]
) -> None:
    built = recurrent_step(cell, inputs=WIDTH, width=STATE)
    step = built(torch.randn(2, WIDTH), torch.randn(2, STATE), torch.zeros(2, STATE))

    assert built.gates == gates
    assert (built.inputs, built.width) == (WIDTH, STATE)
    for gate, values in zip(gates, step.gates, strict=True):
        if gate.dtype is DTypeClass.PROBABILITY:
            assert bool(values.ge(0.0).all())
            assert bool(values.le(1.0).all())


def test_the_relu_elman_step_reads_no_negative_candidate() -> None:
    built = recurrent_step("elman-relu", inputs=WIDTH, width=STATE)
    step = built(torch.randn(8, WIDTH), torch.randn(8, STATE), torch.zeros(8, STATE))

    (candidate,) = step.gates
    assert bool(candidate.ge(0.0).all())
    assert torch.equal(step.hidden, candidate)


def test_an_unregistered_cell_is_refused_in_both_roles() -> None:
    with pytest.raises(RegistryError, match="not-a-cell"):
        recurrent_step("not-a-cell", inputs=WIDTH, width=STATE)
    with pytest.raises(RegistryError, match="not-a-cell"):
        recurrence(
            "not-a-cell", inputs=WIDTH, width=STATE, layers=1, bidirectional=False
        )


def _hold(
    inputs: tuple[Tensor, ...],
    hidden: tuple[Tensor, ...],
    previous: Tensor,
    memory: Tensor,
    /,
) -> CellStep:
    del hidden, memory
    return CellStep(previous, previous, (inputs[0],))


@pytest.mark.parametrize(
    ("gates", "detail"),
    [
        ((), "at least one gate"),
        (
            (Gate("same", DTypeClass.REAL), Gate("same", DTypeClass.REAL)),
            "'same' twice",
        ),
    ],
)
def test_a_plugin_without_distinct_gates_is_refused(
    gates: tuple[Gate, ...], detail: str
) -> None:
    with pytest.raises(ValueError, match=detail):
        RecurrentCellPlugin(nn.GRU, _hold, gates=gates, memory=False)


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("tanh", torch.tanh),
        ("relu", torch.relu),
        ("gelu", nn.functional.gelu),
        ("silu", nn.functional.silu),
    ],
)
def test_an_activation_applies_the_named_nonlinearity(
    kind: Activation, expected: Callable[[Tensor], Tensor]
) -> None:
    values = torch.linspace(-3.0, 3.0, 13)

    assert torch.allclose(activation(kind)(values), expected(values))


def test_a_feedforward_stacks_one_hidden_layer_per_activation_then_the_output() -> None:
    built = feedforward(
        (
            FeedForwardLayer.checked(STATE, "relu", 0.1),
            FeedForwardLayer.checked(4, "tanh", 0.0),
        ),
        inputs=WIDTH,
        output=2,
        bias=False,
    )
    layers = list(built)

    assert [type(layer) for layer in layers] == [
        nn.Linear,
        nn.ReLU,
        nn.Dropout,
        nn.Linear,
        nn.Tanh,
        nn.Dropout,
        nn.Linear,
    ]
    assert [
        tuple(layer.weight.shape) for layer in layers if isinstance(layer, nn.Linear)
    ] == [
        (STATE, WIDTH),
        (4, STATE),
        (2, 4),
    ]
    assert isinstance(layers[2], nn.Dropout)
    assert layers[2].p == 0.1
    assert layers[0].bias is not None
    assert layers[-1].bias is None


def test_a_feedforward_without_hidden_layers_is_one_affine_map() -> None:
    (layer,) = feedforward((), inputs=WIDTH, output=5, bias=True)

    assert isinstance(layer, nn.Linear)
    assert layer.weight.shape == (5, WIDTH)
    assert layer.bias is not None
