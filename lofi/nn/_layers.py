"""Neural layers resolved through their installed plugin families."""

import math
from typing import TYPE_CHECKING, NamedTuple, Protocol, cast, final, override

import torch
from torch import nn
from torch.nn import functional

from lofi._checks import ensure_name, ensure_positive
from lofi._frozen import FrozenValue
from lofi.nn.sites import DTypeClass
from lofi.registry import Registry, RegistryError

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

    from lofi.spec.layers import Activation, FeedForwardLayer, RecurrentCell

__all__ = [
    "ELMAN_RELU",
    "ELMAN_TANH",
    "GELU",
    "GRU",
    "LSTM",
    "RELU",
    "SILU",
    "TANH",
    "ActivationPlugin",
    "CellStep",
    "Gate",
    "RecurrentCellPlugin",
    "RecurrentStep",
    "activation",
    "feedforward",
    "recurrence",
    "recurrent_step",
]


class Gate(NamedTuple):
    """One named activation a single recurrent step computes its state from.

    Attributes
    ----------
    name : str
        The name the activation is published under.
    dtype : DTypeClass
        The value class of the activation, a probability for a sigmoid gate.
    """

    name: str
    dtype: DTypeClass


class CellStep(NamedTuple):
    """The state one recurrent step reaches and the activations it read.

    Attributes
    ----------
    hidden : Tensor
        The hidden state reached, over the row and width axes.
    memory : Tensor
        The memory reached, of the same shape; the hidden state itself for a cell
        that carries no separate memory.
    gates : tuple[Tensor, ...]
        One activation per declared gate, in declaration order, each of the
        hidden state's shape.
    """

    hidden: Tensor
    memory: Tensor
    gates: tuple[Tensor, ...]


class _Transition(Protocol):
    """Advance one recurrent state from its gate pre-activations."""

    def __call__(
        self,
        inputs: tuple[Tensor, ...],
        hidden: tuple[Tensor, ...],
        previous: Tensor,
        memory: Tensor,
        /,
    ) -> CellStep:
        """Return the next state from the input and hidden pre-activations.

        Parameters
        ----------
        inputs : tuple[Tensor, ...]
            The input's affine pre-activation for each declared gate.
        hidden : tuple[Tensor, ...]
            The previous hidden state's affine pre-activation for each gate.
        previous : Tensor
            The previous hidden state.
        memory : Tensor
            The previous memory.
        """
        ...


class _RecurrentBuilder(Protocol):
    """Build one batch-first recurrent stack."""

    def __call__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        num_layers: int,
        batch_first: bool,
        bidirectional: bool,
    ) -> nn.RNNBase:
        """Return the recurrent stack."""
        ...


class ActivationPlugin(FrozenValue):
    """Build one installed activation module."""

    __slots__ = ("_build",)
    _noun = "activation plugin"

    def __init__(self, build: Callable[[], nn.Module], /) -> None:
        object.__setattr__(self, "_build", build)

    def construct(self) -> nn.Module:
        """Return a fresh activation module."""
        return self._build()


@final
class RecurrentStep(nn.Module):
    """One recurrent layer advanced a single step at a time.

    Each declared gate reads one affine block of the input and one of the
    previous hidden state, so the parameters are those of the matching torch
    cell: ``weight_ih`` and ``weight_hh`` stack one width-row block per gate, and
    ``bias_ih`` and ``bias_hh`` likewise.

    Parameters
    ----------
    inputs : int
        The width of an input row.
    width : int
        The width of the hidden state.
    transition : _Transition
        The map from gate pre-activations to the next state.
    gates : tuple[Gate, ...]
        The activations the transition returns, in order.
    memory : bool
        Whether the transition carries a memory apart from its hidden state.
    """

    weight_ih: nn.Parameter
    weight_hh: nn.Parameter
    bias_ih: nn.Parameter
    bias_hh: nn.Parameter

    def __init__(
        self,
        *,
        inputs: int,
        width: int,
        transition: _Transition,
        gates: tuple[Gate, ...],
        memory: bool,
    ) -> None:
        super().__init__()
        blocks = len(gates)
        self.weight_ih = nn.Parameter(torch.empty(blocks * width, inputs))
        self.weight_hh = nn.Parameter(torch.empty(blocks * width, width))
        self.bias_ih = nn.Parameter(torch.empty(blocks * width))
        self.bias_hh = nn.Parameter(torch.empty(blocks * width))
        self._inputs = inputs
        self._width = width
        self._transition = transition
        self._gates = gates
        self._memory = memory
        self.reset_parameters()

    @property
    def inputs(self) -> int:
        """Return the width of an input row."""
        return self._inputs

    @property
    def width(self) -> int:
        """Return the width of the hidden state."""
        return self._width

    @property
    def gates(self) -> tuple[Gate, ...]:
        """Return the activations each step publishes, in order."""
        return self._gates

    @property
    def carries_memory(self) -> bool:
        """Return whether the state includes a memory apart from the hidden state."""
        return self._memory

    def reset_parameters(self) -> None:
        """Initialize every affine parameter from the hidden-width scale."""
        bound = 1.0 / math.sqrt(self._width)

        for parameter in self.parameters():
            nn.init.uniform_(parameter, -bound, bound)

    @override
    def forward(self, inputs: Tensor, previous: Tensor, memory: Tensor, /) -> CellStep:
        """Return the state reached from one input row and the previous state.

        Parameters
        ----------
        inputs : Tensor
            The input rows, over the row and input-width axes.
        previous : Tensor
            The previous hidden rows, over the row and width axes.
        memory : Tensor
            The previous memory rows, of the hidden rows' shape.
        """
        blocks = len(self._gates)

        return self._transition(
            functional.linear(inputs, self.weight_ih, self.bias_ih).chunk(
                blocks, dim=-1
            ),
            functional.linear(previous, self.weight_hh, self.bias_hh).chunk(
                blocks, dim=-1
            ),
            previous,
            memory,
        )


class RecurrentCellPlugin(FrozenValue):
    """Build one installed recurrent cell as a stack or as a single step.

    Parameters
    ----------
    stack : _RecurrentBuilder
        The builder of a batch-first multi-layer stack.
    transition : _Transition
        The single-step map from gate pre-activations to the next state.
    gates : tuple[Gate, ...]
        The activations the transition returns, one affine block each.
    memory : bool
        Whether the transition carries a memory apart from its hidden state.

    Raises
    ------
    ValueError
        If no gate is declared or two gates share a name.
    """

    __slots__ = ("_gates", "_memory", "_stack", "_transition")
    _noun = "recurrent-cell plugin"

    def __init__(
        self,
        stack: _RecurrentBuilder,
        transition: _Transition,
        /,
        *,
        gates: tuple[Gate, ...],
        memory: bool,
    ) -> None:
        if not gates:
            msg = "a recurrent cell declares at least one gate"
            raise ValueError(msg)

        seen: set[str] = set()

        for gate in gates:
            name = ensure_name("a gate name", gate.name)

            if name in seen:
                msg = f"a recurrent cell declares the gate {name!r} twice"
                raise ValueError(msg)

            seen.add(name)

        object.__setattr__(self, "_stack", stack)
        object.__setattr__(self, "_transition", transition)
        object.__setattr__(self, "_gates", gates)
        object.__setattr__(self, "_memory", memory)

    def construct(
        self,
        *,
        inputs: int,
        width: int,
        layers: int,
        bidirectional: bool,
    ) -> nn.RNNBase:
        """Return a batch-first recurrent stack."""
        return self._stack(
            inputs,
            width,
            num_layers=layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

    def step(self, *, inputs: int, width: int) -> RecurrentStep:
        """Return one freshly initialized single-step layer."""
        return RecurrentStep(
            inputs=ensure_positive("a recurrent step's input width", inputs),
            width=ensure_positive("a recurrent step's width", width),
            transition=self._transition,
            gates=self._gates,
            memory=self._memory,
        )


def activation(kind: Activation, /) -> nn.Module:
    """Return a fresh module from the named activation plugin."""
    member = Registry[object]("activation").get(kind)

    if not isinstance(member, ActivationPlugin):
        msg = (
            f"the activation entry {kind!r} names {member!r}, not an activation plugin"
        )
        raise RegistryError(msg, family="activation")

    return member.construct()


def _recurrent_cell(cell: RecurrentCell, /) -> RecurrentCellPlugin:
    """Return the installed recurrent-cell plugin of one name."""
    member = Registry[object]("recurrent_cell").get(cell)

    if not isinstance(member, RecurrentCellPlugin):
        msg = (
            f"the recurrent_cell entry {cell!r} names {member!r}, "
            "not a recurrent-cell plugin"
        )
        raise RegistryError(msg, family="recurrent_cell")

    return member


def recurrence(
    cell: RecurrentCell, *, inputs: int, width: int, layers: int, bidirectional: bool
) -> nn.RNNBase:
    """Return a recurrent stack from the named recurrent-cell plugin."""
    return _recurrent_cell(cell).construct(
        inputs=inputs,
        width=width,
        layers=layers,
        bidirectional=bidirectional,
    )


def recurrent_step(cell: RecurrentCell, *, inputs: int, width: int) -> RecurrentStep:
    """Return a single-step layer from the named recurrent-cell plugin."""
    return _recurrent_cell(cell).step(inputs=inputs, width=width)


def feedforward(
    hidden: tuple[FeedForwardLayer, ...], *, inputs: int, output: int, bias: bool
) -> nn.Sequential:
    """Return declared hidden affine layers followed by an output map."""
    modules: list[nn.Module] = []
    reads = inputs

    for layer in hidden:
        modules.extend(
            (
                nn.Linear(reads, layer.width),
                activation(layer.activation),
                nn.Dropout(layer.dropout),
            )
        )
        reads = layer.width

    modules.append(nn.Linear(reads, output, bias=bias))

    return nn.Sequential(*modules)


def _elman_tanh(
    input_size: int,
    hidden_size: int,
    *,
    num_layers: int,
    batch_first: bool,
    bidirectional: bool,
) -> nn.RNN:
    """Build the tanh Elman cell stack."""
    return nn.RNN(
        input_size,
        hidden_size,
        num_layers=num_layers,
        nonlinearity="tanh",
        batch_first=batch_first,
        bidirectional=bidirectional,
    )


def _elman_relu(
    input_size: int,
    hidden_size: int,
    *,
    num_layers: int,
    batch_first: bool,
    bidirectional: bool,
) -> nn.RNN:
    """Build the ReLU Elman cell stack."""
    return nn.RNN(
        input_size,
        hidden_size,
        num_layers=num_layers,
        nonlinearity="relu",
        batch_first=batch_first,
        bidirectional=bidirectional,
    )


def _gru(
    inputs: tuple[Tensor, ...],
    hidden: tuple[Tensor, ...],
    previous: Tensor,
    memory: Tensor,
    /,
) -> CellStep:
    """Advance a gated recurrent unit, its reset gate scaling the hidden candidate."""
    del memory
    input_reset, input_update, input_candidate = inputs
    hidden_reset, hidden_update, hidden_candidate = hidden
    reset = torch.sigmoid(input_reset + hidden_reset)
    update = torch.sigmoid(input_update + hidden_update)
    candidate = torch.tanh(input_candidate + reset * hidden_candidate)
    reached = (1.0 - update) * candidate + update * previous

    return CellStep(reached, reached, (reset, update, candidate))


def _lstm(
    inputs: tuple[Tensor, ...],
    hidden: tuple[Tensor, ...],
    previous: Tensor,
    memory: Tensor,
    /,
) -> CellStep:
    """Advance a long short-term memory through its input, forget and output gates."""
    del previous
    input_gate, forget, candidate, output = (
        read + held for read, held in zip(inputs, hidden, strict=True)
    )
    input_gate = torch.sigmoid(input_gate)
    forget = torch.sigmoid(forget)
    candidate = torch.tanh(candidate)
    output = torch.sigmoid(output)
    remembered = forget * memory + input_gate * candidate

    return CellStep(
        output * torch.tanh(remembered),
        remembered,
        (input_gate, forget, candidate, output),
    )


def _elman_tanh_step(
    inputs: tuple[Tensor, ...],
    hidden: tuple[Tensor, ...],
    previous: Tensor,
    memory: Tensor,
    /,
) -> CellStep:
    """Advance an ungated tanh recurrence."""
    del previous, memory
    reached = torch.tanh(inputs[0] + hidden[0])

    return CellStep(reached, reached, (reached,))


def _elman_relu_step(
    inputs: tuple[Tensor, ...],
    hidden: tuple[Tensor, ...],
    previous: Tensor,
    memory: Tensor,
    /,
) -> CellStep:
    """Advance an ungated ReLU recurrence."""
    del previous, memory
    reached = torch.relu(inputs[0] + hidden[0])

    return CellStep(reached, reached, (reached,))


_CANDIDATE = Gate("candidate", DTypeClass.REAL)

TANH = ActivationPlugin(nn.Tanh)
RELU = ActivationPlugin(nn.ReLU)
GELU = ActivationPlugin(nn.GELU)
SILU = ActivationPlugin(nn.SiLU)

GRU = RecurrentCellPlugin(
    cast("_RecurrentBuilder", nn.GRU),
    _gru,
    gates=(
        Gate("reset", DTypeClass.PROBABILITY),
        Gate("update", DTypeClass.PROBABILITY),
        _CANDIDATE,
    ),
    memory=False,
)
LSTM = RecurrentCellPlugin(
    cast("_RecurrentBuilder", nn.LSTM),
    _lstm,
    gates=(
        Gate("input", DTypeClass.PROBABILITY),
        Gate("forget", DTypeClass.PROBABILITY),
        _CANDIDATE,
        Gate("output", DTypeClass.PROBABILITY),
    ),
    memory=True,
)
ELMAN_TANH = RecurrentCellPlugin(
    _elman_tanh, _elman_tanh_step, gates=(_CANDIDATE,), memory=False
)
ELMAN_RELU = RecurrentCellPlugin(
    _elman_relu, _elman_relu_step, gates=(_CANDIDATE,), memory=False
)
