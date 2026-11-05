"""Acceptance tests for recurrent neural chart documents."""

from typing import TYPE_CHECKING, Literal

import pytest
from tests.acceptance.driver import train

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("depth", "sharing"),
    [(2, "independent"), (3, "shared")],
)
def test_a_stacked_chart_document_trains_a_tiny_point(
    tmp_path: Path,
    depth: int,
    sharing: Literal["independent", "shared"],
) -> None:
    point = train(
        "slash-point.yaml",
        tmp_path,
        f"model.parser.depth={depth}",
        f"model.parser.layer_sharing={sharing}",
    )

    assert point.resolved
    assert any((point.directory / "checkpoints").iterdir())
