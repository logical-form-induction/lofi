"""Benchmarks the pooled-encoder objective's forward pass over fixed encoder rows."""

from typing import TYPE_CHECKING, Final

import pytest
import torch

from lofi.data import Alignment
from lofi.deduce import SpanIndex
from lofi.judgment import (
    DesignCodec,
    GroupedJudgmentSet,
    JudgmentRow,
    OrdinalHead,
    PooledEncoderObjective,
    RowColumns,
)
from lofi.nn._values import Encoded, LogDistribution, TokenEncoding
from lofi.nn.loss._parser import SPAN_AXES
from lofi.nn.loss._values import (
    CellChart,
    ObjectiveOutputs,
)
from lofi.nn.sites import ComponentId, SiteMap

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pytest_benchmark.fixture import BenchmarkFixture
    from tests.benchmarks.conftest import WallClockBudget

BATCH: Final = 512
LEVELS: Final = 7
WIDTH: Final = 768
ROWS: Final = 32
DTYPE: Final = torch.float32
AXES: Final = ("item", "row", "width")


class FixedRows:
    """An encoder standing on rows drawn once, so the timing is the rung's own."""

    def __init__(self, encoding: TokenEncoding) -> None:
        self._encoding = encoding

    @property
    def output_width(self) -> int:
        return self._encoding.vectors.size("width")

    def encode_words(self, items: Sequence[Sequence[str]], /) -> TokenEncoding:
        del items

        return self._encoding

    def site_map(self) -> SiteMap:
        return SiteMap.custom(ComponentId("fixed-rows"), {})


class Scored:
    """The objective over one batch of drawn rows and judgments."""

    def __init__(self, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        lengths = torch.randint(1, ROWS + 1, (BATCH,), generator=generator)
        mask = torch.arange(ROWS).unsqueeze(0) < lengths.unsqueeze(1)
        rows = torch.randn(BATCH, ROWS, WIDTH, generator=generator, dtype=DTYPE)
        alignments = tuple(
            Alignment(tuple((row, row + 1) for row in range(int(length))))
            for length in lengths
        )
        words = tuple(tuple("w" for _ in range(int(length))) for length in lengths)
        responses = torch.randint(0, LEVELS, (BATCH,), generator=generator)
        encoding = TokenEncoding(Encoded(rows, AXES, mask), alignments, words=words)
        self.objective = PooledEncoderObjective(
            FixedRows(encoding), OrdinalHead(LEVELS)
        )
        item_ids = tuple(f"item-{item}" for item in range(BATCH))
        design = DesignCodec(LEVELS).encode(
            tuple(
                JudgmentRow(
                    "participant",
                    item_id,
                    int(response),
                    RowColumns({}, {}),
                )
                for item_id, response in zip(item_ids, responses, strict=True)
            )
        )
        batch = GroupedJudgmentSet(design, item_ids, words).select((0,))
        live = torch.ones(BATCH, SpanIndex(1).num_spans, dtype=torch.bool)
        self.outputs = ObjectiveOutputs(
            spans=SpanIndex(1),
            chart=CellChart(LogDistribution(torch.zeros(live.shape), SPAN_AXES, live)),
            batch=batch,
        )

    def forward(self) -> torch.Tensor:
        """Run one forward pass over the batch."""
        return self.objective.evaluate(self.outputs).value.detach()


def test_the_benchmark_batch_scores_every_judgment() -> None:
    scored = Scored(seed=3)

    term = scored.objective.evaluate(scored.outputs)
    value = scored.forward()

    assert term.count == BATCH
    assert torch.isfinite(term.value)
    assert torch.isfinite(value)
    assert set(term.diagnostics) == {"log_density", "accuracy", "rank_correlation"}


def test_the_rung_runs_a_forward_pass_over_a_batch_of_512(
    benchmark: BenchmarkFixture,
    wall_clock_budget: WallClockBudget,
) -> None:
    scored = Scored(seed=3)
    value = benchmark.pedantic(
        scored.forward,
        rounds=20,
        warmup_rounds=2,
        iterations=1,
    )
    measured = benchmark.stats

    if measured is None:
        pytest.fail("the rung ran but pytest-benchmark timed nothing")

    benchmark.extra_info["batch"] = BATCH
    benchmark.extra_info["levels"] = LEVELS
    benchmark.extra_info["width"] = WIDTH
    benchmark.extra_info["rows"] = ROWS
    benchmark.extra_info["dtype"] = str(DTYPE)
    benchmark.extra_info["device"] = value.device.type
    median_ms = measured.stats.median * 1e3
    benchmark.extra_info["median_ms_per_forward"] = median_ms
    benchmark.extra_info["mean_ms_per_forward"] = measured.stats.mean * 1e3
    benchmark.extra_info["budget"] = "P-16"

    if wall_clock_budget.enforced:
        assert median_ms <= 25.0
