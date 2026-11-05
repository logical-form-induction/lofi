from typing import TYPE_CHECKING, Final, cast

import pytest

from lofi.eval._diagnostic_reducers import ENCODER_ABLATION_PLUGIN
from lofi.eval._reducers import ExperimentReducer, ReductionContext
from lofi.spec.reducers import EncoderAblationReducerSpec

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_benchmark.fixture import BenchmarkFixture
    from tests.benchmarks.conftest import WallClockBudget

    from lofi._types import JsonValue
    from lofi.registry import ComponentPlugin

BUDGET_MS: Final = 25.0
"""Budget for reducing 64 seeded metric rows into 40 summaries."""


def _rows() -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        {
            "checkpoint": checkpoint,
            "freeze_state": freeze,
            "metric": metric,
            "seed": seed,
            "value": base + 0.01 * seed,
            "weights": "pretrained",
        }
        for checkpoint, base in (
            ("mt5", 0.60),
            ("xlm-roberta", 0.62),
            ("mmbert", 0.64),
            ("eurobert", 0.66),
        )
        for freeze in ("frozen", "fine-tuned")
        for metric in ("tagger-accuracy", "sentence-accuracy")
        for seed in range(4)
    )


def _reducer(root: Path) -> ExperimentReducer:
    declaration = EncoderAblationReducerSpec.model_validate(
        {
            "kind": "encoder-ablation",
            "checkpoints": ["mt5", "xlm-roberta", "mmbert", "eurobert"],
            "freeze_states": ["frozen", "fine-tuned"],
            "metrics": ["tagger-accuracy", "sentence-accuracy"],
            "minimum_seeds": 4,
            "checkpoint_column": "checkpoint",
            "freeze_column": "freeze_state",
            "metric_column": "metric",
            "seed_column": "seed",
            "value_column": "value",
            "weights_column": "weights",
        }
    )
    plugin = cast(
        "ComponentPlugin[object, ReductionContext, ExperimentReducer]",
        ENCODER_ABLATION_PLUGIN,
    )
    return plugin.build(declaration.settings(), ReductionContext(root, seed=13))


def test_the_reducer_budget_fixture_has_the_declared_matrix(tmp_path: Path) -> None:
    rows = _rows()
    reduced = _reducer(tmp_path).reduce(rows).rows

    assert len(rows) == 64
    assert all(row["weights"] == "pretrained" for row in rows)
    assert sum(row["kind"] == "cell" for row in reduced) == 16
    assert sum(row["kind"] == "difference" for row in reduced) == 24


@pytest.mark.benchmark
def test_the_registered_encoder_ablation_reducer_meets_its_budget(
    benchmark: BenchmarkFixture,
    tmp_path: Path,
    wall_clock_budget: WallClockBudget,
) -> None:
    reducer = _reducer(tmp_path)
    rows = _rows()
    benchmark.pedantic(reducer.reduce, args=(rows,), rounds=20, iterations=10)
    measured = benchmark.stats

    if measured is None:
        pytest.fail("the reducer ran but pytest-benchmark timed nothing")

    median_ms = measured.stats.median * 1e3
    benchmark.extra_info["budget"] = "P-52"
    benchmark.extra_info["budget_cell"] = "encoder-ablation-reducer"
    benchmark.extra_info["budget_ms"] = BUDGET_MS
    benchmark.extra_info["median_ms"] = median_ms

    if wall_clock_budget.enforced:
        assert median_ms <= BUDGET_MS
