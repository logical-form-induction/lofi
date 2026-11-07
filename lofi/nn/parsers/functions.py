"""Registered vector functions used at neural-chart span slots."""

from __future__ import annotations

from math import sqrt
from typing import TYPE_CHECKING, Protocol, final, override, runtime_checkable

import torch
from torch import nn

from lofi._checks import ensure_positive
from lofi._frozen import FrozenValue
from lofi.nn._layers import activation
from lofi.registry import Registry, RegistryError
from lofi.spec.span_functions import (
    BilinearSpanFunctionSettings,
    CpBilinearSpanFunctionSettings,
    GatedMlpSpanFunctionSettings,
    HrrSpanFunctionSettings,
    MlpSpanFunctionSettings,
    SpanFunctionDeclaration,
    TransformerSpanFunctionSettings,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

    from lofi.spec.layers import Activation, NormPlacement

__all__ = [
    "BILINEAR_PLUGIN",
    "CP_BILINEAR_PLUGIN",
    "GATED_MLP_PLUGIN",
    "HRR_PLUGIN",
    "MLP_PLUGIN",
    "TRANSFORMER_PLUGIN",
    "BilinearSpanFunction",
    "CpBilinearSpanFunction",
    "GatedMlpSpanFunction",
    "HrrSpanFunction",
    "MlpSpanFunction",
    "SpanFunction",
    "SpanFunctionModule",
    "SpanFunctionPlugin",
    "TransformerSpanFunction",
    "build_span_function",
]


@runtime_checkable
class SpanFunction(Protocol):
    """Map one or more span vectors to one output vector."""

    @property
    def input_widths(self) -> tuple[int, ...]:
        """Return the innermost width read from each positional input."""
        ...

    @property
    def output_width(self) -> int:
        """Return the innermost width of the output."""
        ...

    def __call__(self, *inputs: Tensor) -> Tensor:
        """Return one output vector for each broadcast input row."""
        ...


class SpanFunctionModule(nn.Module):
    """Hold and check one span function's declared widths."""

    def __init__(self, input_widths: tuple[int, ...], output_width: int) -> None:
        super().__init__()

        if not input_widths:
            msg = "a span function reads at least one input"
            raise ValueError(msg)

        self._input_widths = tuple(
            ensure_positive("a span function's input width", width)
            for width in input_widths
        )
        self._output_width = ensure_positive(
            "a span function's output width", output_width
        )

    @property
    def input_widths(self) -> tuple[int, ...]:
        """Return the innermost width read from each positional input."""
        return self._input_widths

    @property
    def output_width(self) -> int:
        """Return the innermost width of the output."""
        return self._output_width

    def joined(self, inputs: tuple[Tensor, ...], /) -> Tensor:
        """Validate and concatenate positional inputs over their feature axis."""
        if len(inputs) != len(self._input_widths):
            msg = (
                f"this span function reads {len(self._input_widths)} inputs, "
                f"not {len(inputs)}"
            )
            raise ValueError(msg)

        for position, (value, width) in enumerate(
            zip(inputs, self._input_widths, strict=True)
        ):
            if not value.ndim or int(value.shape[-1]) != width:
                received = None if not value.ndim else int(value.shape[-1])
                msg = (
                    f"span-function input {position} has width {received}, not {width}"
                )
                raise ValueError(msg)

        return torch.cat(inputs, dim=-1)


@final
class MlpSpanFunction(SpanFunctionModule):
    """Apply affine maps and one activation per layer to concatenated inputs."""

    def __init__(
        self,
        input_widths: tuple[int, ...],
        output_width: int,
        *,
        layers: int,
        nonlinearity: Activation,
    ) -> None:
        super().__init__(input_widths, output_width)
        count = ensure_positive("an MLP span function's layer count", layers)
        modules: list[nn.Module] = []
        reads = sum(self.input_widths)

        for _ in range(count):
            modules.extend((nn.Linear(reads, output_width), activation(nonlinearity)))
            reads = output_width

        self._network = nn.Sequential(*modules)

    @override
    def forward(self, *inputs: Tensor) -> Tensor:
        """Return the multilayer projection of the concatenated inputs."""
        return self._network(self.joined(inputs))


@final
class BilinearSpanFunction(SpanFunctionModule):
    """Apply a full bilinear map to two inputs."""

    def __init__(self, input_widths: tuple[int, ...], output_width: int) -> None:
        super().__init__(input_widths, output_width)
        _require_binary("a bilinear span function", self.input_widths)
        self._bilinear = nn.Bilinear(
            self.input_widths[0], self.input_widths[1], output_width
        )

    @override
    def forward(self, *inputs: Tensor) -> Tensor:
        """Return the bilinear projection of two inputs."""
        self.joined(inputs)
        return self._bilinear(inputs[0], inputs[1])


@final
class CpBilinearSpanFunction(SpanFunctionModule):
    """Apply a CP-factorized bilinear map to two inputs."""

    def __init__(
        self, input_widths: tuple[int, ...], output_width: int, *, rank: int
    ) -> None:
        super().__init__(input_widths, output_width)
        _require_binary("a CP-bilinear span function", self.input_widths)
        factors = ensure_positive("a CP-bilinear span function's rank", rank)
        self._left = nn.Linear(input_widths[0], factors, bias=False)
        self._right = nn.Linear(input_widths[1], factors, bias=False)
        self._output = nn.Linear(factors, output_width)

    @override
    def forward(self, *inputs: Tensor) -> Tensor:
        """Return the factorized bilinear projection of two inputs."""
        self.joined(inputs)
        return self._output(self._left(inputs[0]) * self._right(inputs[1]))


@final
class GatedMlpSpanFunction(SpanFunctionModule):
    """Interpolate a multilayer candidate and a residual projection."""

    def __init__(
        self,
        input_widths: tuple[int, ...],
        output_width: int,
        *,
        layers: int,
        nonlinearity: Activation,
    ) -> None:
        super().__init__(input_widths, output_width)
        count = ensure_positive("a gated MLP span function's layer count", layers)
        joined_width = sum(input_widths)
        hidden: list[nn.Module] = []
        reads = joined_width

        for _ in range(count - 1):
            hidden.extend((nn.Linear(reads, output_width), activation(nonlinearity)))
            reads = output_width

        self._hidden = nn.Sequential(*hidden)
        self._candidate = nn.Sequential(
            nn.Linear(reads, output_width), activation(nonlinearity)
        )
        self._gate = nn.Linear(joined_width, output_width)
        self._residual = nn.Linear(joined_width, output_width)

    @override
    def forward(self, *inputs: Tensor) -> Tensor:
        """Return a learned mixture of candidate and residual projections."""
        joined = self.joined(inputs)
        candidate = self._candidate(self._hidden(joined))
        residual = self._residual(joined)
        gate = torch.sigmoid(self._gate(joined))
        return gate * candidate + (1.0 - gate) * residual


@final
class HrrSpanFunction(SpanFunctionModule):
    """Bind two equal-width vectors by Hadamard or circular convolution."""

    def __init__(
        self,
        input_widths: tuple[int, ...],
        output_width: int,
        *,
        binding: str,
    ) -> None:
        super().__init__(input_widths, output_width)
        _require_binary("a holographic span function", self.input_widths)

        if input_widths[0] != input_widths[1] or input_widths[0] != output_width:
            msg = (
                "a holographic span function requires two input widths and an "
                f"output width that agree, not {(*input_widths, output_width)!r}"
            )
            raise ValueError(msg)

        if binding not in {"hadamard", "circular"}:
            msg = f"a holographic binding is 'hadamard' or 'circular', not {binding!r}"
            raise ValueError(msg)

        self._binding = binding

    @override
    def forward(self, *inputs: Tensor) -> Tensor:
        """Return the declared holographic binding of two inputs."""
        self.joined(inputs)
        left, right = inputs

        if self._binding == "hadamard":
            return left * right

        transformed = torch.fft.rfft(left) * torch.fft.rfft(right)
        return torch.fft.irfft(transformed, n=self.output_width)


@final
class TransformerSpanFunction(SpanFunctionModule):
    """Compose projected inputs with learned positions and self-attention."""

    def __init__(
        self,
        input_widths: tuple[int, ...],
        output_width: int,
        *,
        layers: int,
        heads: int,
        feedforward_width: int,
        nonlinearity: Activation,
        dropout: float,
        norm_placement: NormPlacement,
    ) -> None:
        super().__init__(input_widths, output_width)
        depth = ensure_positive("a transformer span function's layer count", layers)
        head_count = ensure_positive("a transformer span function's head count", heads)

        if output_width % head_count:
            msg = (
                f"a transformer span width {output_width} is not divisible by "
                f"its {head_count} heads"
            )
            raise ValueError(msg)

        feedforward = ensure_positive(
            "a transformer span function's feed-forward width", feedforward_width
        )
        self._projections = nn.ModuleList(
            nn.Linear(width, output_width) for width in input_widths
        )
        self._positions = nn.Parameter(
            torch.empty(len(input_widths), output_width).normal_(
                std=1.0 / sqrt(output_width)
            )
        )
        layer = nn.TransformerEncoderLayer(
            output_width,
            head_count,
            dim_feedforward=feedforward,
            dropout=dropout,
            activation=activation(nonlinearity),
            batch_first=True,
            norm_first=norm_placement == "pre",
        )
        self._encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self._output = nn.Linear(output_width * len(input_widths), output_width)

    @override
    def forward(self, *inputs: Tensor) -> Tensor:
        """Return the projection of the contextualized input sequence."""
        self.joined(inputs)
        shape = inputs[0].shape[:-1]
        rows = int(inputs[0].numel() // inputs[0].shape[-1])
        tokens = torch.stack(
            tuple(
                projection(value.reshape(rows, width))
                for projection, value, width in zip(
                    self._projections, inputs, self.input_widths, strict=True
                )
            ),
            dim=1,
        )
        contextual = self._encoder(tokens + self._positions.unsqueeze(0))
        return self._output(contextual.flatten(1)).reshape(*shape, self.output_width)


def _require_binary(noun: str, widths: tuple[int, ...], /) -> None:
    """Refuse a function that does not have exactly two input widths."""
    if len(widths) != 2:
        msg = f"{noun} reads exactly two inputs, not {len(widths)}"
        raise ValueError(msg)


class SpanFunctionPlugin[S](FrozenValue):
    """Build one span function from its resolved settings and declared widths."""

    __slots__ = ("_build", "_settings")
    _noun = "span-function plugin"

    def __init__(
        self,
        *,
        settings: type[S],
        build: Callable[[S, tuple[int, ...], int], SpanFunctionModule],
    ) -> None:
        object.__setattr__(self, "_settings", settings)
        object.__setattr__(self, "_build", build)

    def construct(
        self, settings: object, input_widths: tuple[int, ...], output_width: int, /
    ) -> SpanFunctionModule:
        """Build a function after checking the registered settings type."""
        if not isinstance(settings, self._settings):
            msg = (
                f"a span-function plugin reads {self._settings.__name__}, "
                f"not {type(settings).__name__}"
            )
            raise TypeError(msg)

        return self._build(settings, input_widths, output_width)


def build_span_function(
    declaration: SpanFunctionDeclaration,
    input_widths: tuple[int, ...],
    output_width: int,
    /,
) -> SpanFunctionModule:
    """Build one span function through its installed plugin."""
    member = Registry[object]("span_function").get(declaration.kind)

    if not isinstance(member, SpanFunctionPlugin):
        msg = (
            f"the span_function entry {declaration.kind!r} names {member!r}, "
            "not a span-function plugin"
        )
        raise RegistryError(msg, family="span_function")

    return member.construct(declaration.settings, input_widths, output_width)


def _mlp(
    settings: MlpSpanFunctionSettings,
    inputs: tuple[int, ...],
    output: int,
    /,
) -> MlpSpanFunction:
    """Build a multilayer perceptron span function."""
    return MlpSpanFunction(
        inputs,
        output,
        layers=settings.layers,
        nonlinearity=settings.activation,
    )


def _bilinear(
    settings: BilinearSpanFunctionSettings,
    inputs: tuple[int, ...],
    output: int,
    /,
) -> BilinearSpanFunction:
    """Build a full bilinear span function."""
    del settings
    return BilinearSpanFunction(inputs, output)


def _cp_bilinear(
    settings: CpBilinearSpanFunctionSettings,
    inputs: tuple[int, ...],
    output: int,
    /,
) -> CpBilinearSpanFunction:
    """Build a CP-factorized bilinear span function."""
    return CpBilinearSpanFunction(inputs, output, rank=settings.rank)


def _gated_mlp(
    settings: GatedMlpSpanFunctionSettings,
    inputs: tuple[int, ...],
    output: int,
    /,
) -> GatedMlpSpanFunction:
    """Build a gated multilayer span function."""
    return GatedMlpSpanFunction(
        inputs,
        output,
        layers=settings.layers,
        nonlinearity=settings.activation,
    )


def _hrr(
    settings: HrrSpanFunctionSettings,
    inputs: tuple[int, ...],
    output: int,
    /,
) -> HrrSpanFunction:
    """Build a holographic span function."""
    return HrrSpanFunction(inputs, output, binding=settings.binding)


def _transformer(
    settings: TransformerSpanFunctionSettings,
    inputs: tuple[int, ...],
    output: int,
    /,
) -> TransformerSpanFunction:
    """Build a self-attentive span function."""
    return TransformerSpanFunction(
        inputs,
        output,
        layers=settings.layers,
        heads=settings.heads,
        feedforward_width=settings.feedforward_width,
        nonlinearity=settings.activation,
        dropout=settings.dropout,
        norm_placement=settings.norm_placement,
    )


MLP_PLUGIN = SpanFunctionPlugin(settings=MlpSpanFunctionSettings, build=_mlp)
BILINEAR_PLUGIN = SpanFunctionPlugin(
    settings=BilinearSpanFunctionSettings, build=_bilinear
)
CP_BILINEAR_PLUGIN = SpanFunctionPlugin(
    settings=CpBilinearSpanFunctionSettings, build=_cp_bilinear
)
GATED_MLP_PLUGIN = SpanFunctionPlugin(
    settings=GatedMlpSpanFunctionSettings, build=_gated_mlp
)
HRR_PLUGIN = SpanFunctionPlugin(settings=HrrSpanFunctionSettings, build=_hrr)
TRANSFORMER_PLUGIN = SpanFunctionPlugin(
    settings=TransformerSpanFunctionSettings, build=_transformer
)
