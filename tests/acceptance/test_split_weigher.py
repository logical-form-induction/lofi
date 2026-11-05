"""The declared split weigher reaches a fitted neural chart."""

from typing import TYPE_CHECKING

import pytest
from tests.acceptance.driver import leaf, train

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("weigher", ["attend", "marginal"])
def test_a_span_readout_trains_with_each_split_weigher(
    weigher: str, tmp_path: Path
) -> None:
    point = train(
        "attend-chart.yaml",
        tmp_path / weigher,
        f"model.parser.attend.kind={weigher}",
    )

    assert leaf(point.resolved, "model.parser.attend.kind") == weigher
    assert leaf(point.resolved, "model.readouts.types.reads") == "span"
