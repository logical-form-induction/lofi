"""Tests for declared decoder forward-mode names."""

from typing import get_args

from lofi.spec.forward import ForwardModeName


def test_forward_modes_name_each_supported_decoder_pass() -> None:
    assert get_args(ForwardModeName.__value__) == (
        "observe",
        "sample",
        "marginalize",
    )
