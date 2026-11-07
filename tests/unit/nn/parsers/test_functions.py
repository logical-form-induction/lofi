from collections.abc import Callable
from typing import Literal

import pytest
import torch

from lofi.nn.parsers.functions import (
    BILINEAR_PLUGIN,
    CP_BILINEAR_PLUGIN,
    GATED_MLP_PLUGIN,
    HRR_PLUGIN,
    MLP_PLUGIN,
    TRANSFORMER_PLUGIN,
    BilinearSpanFunction,
    HrrSpanFunction,
    SpanFunction,
    SpanFunctionPlugin,
)
from lofi.spec.span_functions import (
    BilinearSpanFunctionSettings,
    CpBilinearSpanFunctionSettings,
    GatedMlpSpanFunctionSettings,
    HrrSpanFunctionSettings,
    MlpSpanFunctionSettings,
    TransformerSpanFunctionSettings,
)

Build = Callable[[], SpanFunction]


@pytest.mark.parametrize(
    "build",
    [
        lambda: MLP_PLUGIN.construct(MlpSpanFunctionSettings(2, "gelu"), (3, 5), 4),
        lambda: BILINEAR_PLUGIN.construct(BilinearSpanFunctionSettings(), (3, 5), 4),
        lambda: CP_BILINEAR_PLUGIN.construct(
            CpBilinearSpanFunctionSettings(2), (3, 5), 4
        ),
        lambda: GATED_MLP_PLUGIN.construct(
            GatedMlpSpanFunctionSettings(2, "silu"), (3, 5), 4
        ),
        lambda: TRANSFORMER_PLUGIN.construct(
            TransformerSpanFunctionSettings(1, 2, 8, "gelu", 0.0, "post"),
            (3, 5),
            4,
        ),
    ],
)
def test_learned_span_functions_have_declared_shapes_and_gradients(
    build: Build,
) -> None:
    function = build()
    left = torch.randn(2, 7, 3, requires_grad=True)
    right = torch.randn(2, 7, 5, requires_grad=True)

    result = function(left, right)
    result.square().sum().backward()

    assert isinstance(function, SpanFunction)
    assert function.input_widths == (3, 5)
    assert function.output_width == 4
    assert result.shape == (2, 7, 4)
    assert left.grad is not None
    assert right.grad is not None


def test_unary_mlp_reads_one_input() -> None:
    function = MLP_PLUGIN.construct(MlpSpanFunctionSettings(1, "tanh"), (3,), 4)

    assert function(torch.randn(2, 3)).shape == (2, 4)


def test_full_bilinear_function_requires_two_inputs() -> None:
    with pytest.raises(ValueError, match="exactly two"):
        BilinearSpanFunction((3,), 4)


def test_input_count_and_width_are_checked_at_the_function_boundary() -> None:
    function = MLP_PLUGIN.construct(MlpSpanFunctionSettings(1, "tanh"), (3, 5), 4)

    with pytest.raises(ValueError, match="reads 2 inputs"):
        function(torch.randn(2, 3))

    with pytest.raises(ValueError, match="input 1 has width 6"):
        function(torch.randn(2, 3), torch.randn(2, 6))


def test_holographic_bindings_are_declared_and_differentiable() -> None:
    left = torch.randn(3, 4, requires_grad=True)
    right = torch.randn(3, 4, requires_grad=True)
    hadamard = HRR_PLUGIN.construct(HrrSpanFunctionSettings("hadamard"), (4, 4), 4)
    circular = HRR_PLUGIN.construct(HrrSpanFunctionSettings("circular"), (4, 4), 4)

    expected = torch.fft.irfft(torch.fft.rfft(left) * torch.fft.rfft(right), n=4)

    torch.testing.assert_close(hadamard(left, right), left * right)
    torch.testing.assert_close(circular(left, right), expected)
    (hadamard(left, right) + circular(left, right)).sum().backward()
    assert left.grad is not None
    assert right.grad is not None


def test_holographic_function_requires_equal_widths() -> None:
    with pytest.raises(ValueError, match="agree"):
        HrrSpanFunction((3, 4), 4, binding="hadamard")


def test_transformer_refuses_an_indivisible_head_count() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        TRANSFORMER_PLUGIN.construct(
            TransformerSpanFunctionSettings(1, 3, 8, "relu", 0.0, "post"),
            (4, 4),
            4,
        )


def test_transformer_uses_an_installed_activation() -> None:
    function = TRANSFORMER_PLUGIN.construct(
        TransformerSpanFunctionSettings(1, 2, 8, "tanh", 0.0, "post"),
        (4, 4),
        4,
    )

    assert function(torch.randn(3, 4), torch.randn(3, 4)).shape == (3, 4)


def test_transformer_reads_its_declared_norm_placement() -> None:
    inputs = (torch.randn(3, 4), torch.randn(3, 4))

    def built(placement: Literal["pre", "post"]) -> torch.Tensor:
        torch.manual_seed(0)
        function = TRANSFORMER_PLUGIN.construct(
            TransformerSpanFunctionSettings(1, 2, 8, "gelu", 0.0, placement),
            (4, 4),
            4,
        )
        return function(*inputs)

    assert not torch.allclose(built("pre"), built("post"))


def test_transformer_reads_its_declared_dropout() -> None:
    inputs = (torch.randn(3, 4), torch.randn(3, 4))
    function = TRANSFORMER_PLUGIN.construct(
        TransformerSpanFunctionSettings(1, 2, 8, "gelu", 0.5, "pre"),
        (4, 4),
        4,
    )

    assert not torch.equal(function(*inputs), function(*inputs))

    function.eval()

    assert torch.equal(function(*inputs), function(*inputs))


def test_plugin_checks_its_settings_type() -> None:
    with pytest.raises(TypeError, match="MlpSpanFunctionSettings"):
        MLP_PLUGIN.construct(BilinearSpanFunctionSettings(), (2, 2), 2)


def test_registered_values_are_span_function_plugins() -> None:
    assert isinstance(MLP_PLUGIN, SpanFunctionPlugin)
    assert isinstance(BILINEAR_PLUGIN, SpanFunctionPlugin)
    assert isinstance(CP_BILINEAR_PLUGIN, SpanFunctionPlugin)
    assert isinstance(GATED_MLP_PLUGIN, SpanFunctionPlugin)
    assert isinstance(HRR_PLUGIN, SpanFunctionPlugin)
    assert isinstance(TRANSFORMER_PLUGIN, SpanFunctionPlugin)
