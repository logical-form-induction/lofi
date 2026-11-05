from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final

import pytest

from lofi._types import ItemId
from lofi.data import LicenseRef, Licensing, Provenance, Reproducibility
from lofi.eval._matched import InducedBracketing
from lofi.eval._publish import (
    LayersPredictionLens,
    PredictedAnalysis,
)

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture
    from tests.benchmarks.conftest import WallClockBudget

# the P-36 budget: one B-7 crossing, which publishes one artifact
BUDGET_MS: Final = 1000.0

ITEM_ID: Final = ItemId("at://did:plc:lofi/pub.layers.expression.expression/3k2")
TOKENIZATION: Final = "1f0a6d1e-0d3c-4a1b-9f77-0c4b9c2f1d55"
PRODUCED_AT: Final = datetime(2026, 9, 22, 14, 30, tzinfo=UTC)
LENGTH: Final = 40


def _provenance() -> Provenance:
    return Provenance(
        "converted-corrected",
        Licensing(
            "CC-BY-4.0 AND LicenseRef-LDC-User-Agreement",
            (
                LicenseRef("CC-BY-4.0", "annotations"),
                LicenseRef("LicenseRef-LDC-User-Agreement", "underlying-text"),
            ),
        ),
        ("at://did:plc:lofi/pub.layers.eprint.eprint/1",),
        Reproducibility(
            "https://github.com/logical-form-induction/lofi",
            "0f1e2d3c4b5a",
            "lofi run parse",
            "uv-3.14",
            17,
        ),
    )


def _right_branching(length: int) -> InducedBracketing:
    """Return the right-branching analysis of a sentence of one length.

    Parameters
    ----------
    length : int
        The sentence length in tokens.
    """
    spans = {(start, length): "S" for start in range(length - 1)}
    spans |= {(start, start + 1): "NP" for start in range(length)}

    return InducedBracketing(length, spans)


def _median_ms(benchmark: BenchmarkFixture) -> float:
    """Return the measured median in milliseconds, recorded beside its budget.

    Parameters
    ----------
    benchmark : BenchmarkFixture
        The fixture the publish ran through.
    """
    measured = benchmark.stats

    if measured is None:
        pytest.fail("the publish ran but pytest-benchmark timed nothing")

    median_ms = measured.stats.median * 1e3
    benchmark.extra_info["budget"] = "P-36"
    benchmark.extra_info["budget_cell"] = "layers-prediction-publish"
    benchmark.extra_info["budget_ms"] = BUDGET_MS
    benchmark.extra_info["median_ms"] = median_ms

    return median_ms


def test_publishing_a_prediction_record_meets_its_budget(
    benchmark: BenchmarkFixture,
    wall_clock_budget: WallClockBudget,
) -> None:
    lens = LayersPredictionLens("lofi", "ccg")
    analysis = PredictedAnalysis(
        ITEM_ID,
        TOKENIZATION,
        _right_branching(LENGTH),
        _provenance(),
        PRODUCED_AT,
    )
    benchmark.pedantic(
        lens.render, args=(analysis,), rounds=30, warmup_rounds=2, iterations=1
    )

    if wall_clock_budget.enforced:
        assert _median_ms(benchmark) <= BUDGET_MS


def test_publishing_a_prediction_record_marks_it_as_automatic() -> None:
    lens = LayersPredictionLens("lofi", "ccg")
    analysis = PredictedAnalysis(
        ITEM_ID,
        TOKENIZATION,
        _right_branching(LENGTH),
        _provenance(),
        PRODUCED_AT,
    )

    assert '"sourceMethod": "automatic"' in lens.render(analysis)
