from __future__ import annotations

import pytest
import torch
from tests.conformance.code_tensorizer import (
    CodeTensorizerConformance,
    CodeTensorizerFixture,
)

from lofi.data import CodeTensorizer
from lofi.nn._tensorize import TorchCodeTensorizer


class TestTorchCodeTensorizer(CodeTensorizerConformance):
    """Conformance tests for torch-backed tensorization."""

    @pytest.fixture
    def code_tensorizer_fixture(self) -> CodeTensorizerFixture:
        """Build a tensorizer fixture with ragged and empty rows."""
        return CodeTensorizerFixture(
            TorchCodeTensorizer,
            [(4, 1), (7,), ()],
            9,
            ((4, 1), (7, 9), (9, 9)),
            ((True, True), (True, False), (False, False)),
        )


def test_the_tensorizer_pads_integer_rows_and_marks_only_live_codes() -> None:
    vectors, present = TorchCodeTensorizer()(((4, 1), (7,), ()), padding=9)

    assert torch.equal(vectors, torch.tensor([[4, 1], [7, 9], [9, 9]]))
    assert vectors.dtype is torch.int64
    assert torch.equal(
        present,
        torch.tensor([[True, True], [True, False], [False, False]]),
    )
    assert present.dtype is torch.bool


def test_empty_rows_have_two_empty_axes() -> None:
    vectors, present = TorchCodeTensorizer()((), padding=3)

    assert vectors.shape == (0, 0)
    assert present.shape == (0, 0)


def test_the_tensorizer_satisfies_the_public_code_tensorizer_port() -> None:
    assert isinstance(TorchCodeTensorizer(), CodeTensorizer)
