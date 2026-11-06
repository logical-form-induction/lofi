"""Control CPU parallelism for forward-equivalent timing comparisons."""

from typing import TYPE_CHECKING

import pytest
import torch

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def single_threaded_cpu() -> Iterator[None]:
    """Time the model and probe with one intra-op thread, then restore the pool.

    Matrix multiplication and probe reductions scale differently with the
    thread count. Fixing it makes their ratio comparable across runner sizes.
    """
    previous = torch.get_num_threads()
    torch.set_num_threads(1)

    try:
        yield
    finally:
        torch.set_num_threads(previous)
