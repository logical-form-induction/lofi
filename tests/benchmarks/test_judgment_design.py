"""Benchmarks judgment-design encoding."""

from typing import TYPE_CHECKING, Final

import pytest

from lofi.judgment import DesignCodec, JudgmentRow, RowColumns

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture
    from tests.benchmarks.conftest import WallClockBudget

ROWS: Final = 250_000
PARTICIPANTS: Final = 500
ITEMS: Final = 50_000
LEVELS: Final = 7
FACTORS: Final = ("frame", "verb")
READOUTS: Final = ("p_gram",)


@pytest.fixture(scope="module")
def judgment_rows() -> tuple[JudgmentRow, ...]:
    """Return the benchmark judgment rows."""
    return tuple(
        JudgmentRow(
            participant=f"participant-{index % PARTICIPANTS}",
            item=f"item-{index % ITEMS}",
            response=index % LEVELS,
            columns=RowColumns(
                factors={
                    "frame": f"frame-{index % 40}",
                    "verb": f"verb-{index % 1_000}",
                },
                readouts={"p_gram": (index % 100) / 100.0},
            ),
        )
        for index in range(ROWS)
    )


def test_the_design_codec_preserves_the_benchmark_row_count(
    judgment_rows: tuple[JudgmentRow, ...],
) -> None:
    design = DesignCodec(LEVELS, FACTORS, READOUTS).encode(judgment_rows)

    assert design.n_rows == ROWS
    assert len(design.participant_levels) == PARTICIPANTS
    assert len(design.item_levels) == ITEMS


def test_the_design_codec_encodes_250000_rows(
    benchmark: BenchmarkFixture,
    judgment_rows: tuple[JudgmentRow, ...],
    wall_clock_budget: WallClockBudget,
) -> None:
    codec = DesignCodec(LEVELS, FACTORS, READOUTS)
    design = benchmark.pedantic(
        codec.encode,
        args=(judgment_rows,),
        rounds=1,
        iterations=1,
    )

    measured = benchmark.stats

    if measured is None:
        pytest.fail("the design codec ran but pytest-benchmark timed nothing")

    benchmark.extra_info["rows"] = ROWS
    benchmark.extra_info["participants"] = PARTICIPANTS
    benchmark.extra_info["items"] = ITEMS
    benchmark.extra_info["levels"] = LEVELS
    benchmark.extra_info["factors"] = list(FACTORS)
    benchmark.extra_info["readouts"] = list(READOUTS)
    benchmark.extra_info["dtype"] = str(design.responses.dtype)
    benchmark.extra_info["device"] = design.responses.device.type
    benchmark.extra_info["budget"] = "P-50"
    benchmark.extra_info["median_ms_total"] = measured.stats.median * 1e3
    mean_ms = measured.stats.mean * 1e3
    benchmark.extra_info["mean_ms_total"] = mean_ms

    if wall_clock_budget.enforced:
        assert mean_ms <= 3_000.0
