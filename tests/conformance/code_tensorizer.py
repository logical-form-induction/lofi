"""Reusable laws for integer-code tensorizers."""

from typing import TYPE_CHECKING, ClassVar, NamedTuple

import pytest
import torch

if TYPE_CHECKING:
    from collections.abc import Callable

    from lofi.data import CodeTensorizer

__all__ = ["CodeTensorizerConformance", "CodeTensorizerFixture"]


class CodeTensorizerFixture(NamedTuple):
    """Construction and expected values for a code tensorizer.

    Attributes
    ----------
    build : Callable[[], CodeTensorizer]
        Return a fresh tensorizer.
    rows : list[tuple[int, ...]]
        Ragged integer rows supplied to the tensorizer.
    padding : int
        Code used outside each row's live extent.
    vectors : tuple[tuple[int, ...], ...]
        Expected padded integer values.
    present : tuple[tuple[bool, ...], ...]
        Expected live-position mask.
    """

    build: Callable[[], CodeTensorizer]
    rows: list[tuple[int, ...]]
    padding: int
    vectors: tuple[tuple[int, ...], ...]
    present: tuple[tuple[bool, ...], ...]


class CodeTensorizerConformance:
    """Reusable code-tensorizer conformance laws."""

    pytestmark: ClassVar[list[pytest.MarkDecorator]] = [pytest.mark.conformance]

    def test_values_masks_dtypes_repeat_and_leave_input_unchanged(
        self,
        code_tensorizer_fixture: CodeTensorizerFixture,
    ) -> None:
        """Run the value, mask, dtype, repeatability and nonmutation law."""
        fixture = code_tensorizer_fixture
        tensorizer = fixture.build()
        before = tuple(fixture.rows)
        first_vectors, first_present = tensorizer(
            fixture.rows,
            padding=fixture.padding,
        )
        second_vectors, second_present = tensorizer(
            fixture.rows,
            padding=fixture.padding,
        )

        assert tuple(fixture.rows) == before
        assert first_vectors.dtype is torch.int64
        assert first_present.dtype is torch.bool
        assert first_vectors.tolist() == [list(row) for row in fixture.vectors]
        assert first_present.tolist() == [list(row) for row in fixture.present]
        assert torch.equal(second_vectors, first_vectors)
        assert torch.equal(second_present, first_present)
