import pytest

from lofi.spec.layers import FeedForwardLayer


def test_a_hidden_layer_keeps_each_declared_setting() -> None:
    assert FeedForwardLayer.checked(64, "test-activation", 0.1) == (
        64,
        "test-activation",
        0.1,
    )


@pytest.mark.parametrize("width", [0, -1])
def test_a_hidden_layer_requires_a_positive_width(width: int) -> None:
    with pytest.raises(ValueError, match="width"):
        FeedForwardLayer.checked(width, "tanh", 0.0)


@pytest.mark.parametrize("dropout", [-0.1, 1.0, float("inf")])
def test_a_hidden_layer_requires_a_dropout_rate(dropout: float) -> None:
    with pytest.raises(ValueError, match="dropout"):
        FeedForwardLayer.checked(3, "tanh", dropout)


def test_a_hidden_layer_accepts_an_external_activation_name() -> None:
    assert FeedForwardLayer.checked(3, "test-package-cube", 0.0).activation == (
        "test-package-cube"
    )
