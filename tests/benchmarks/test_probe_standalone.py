from typing import TYPE_CHECKING, Final

import pytest
from tests.independence.test_probe_standalone import run_probe_smoke

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture
    from tests.benchmarks.conftest import WallClockBudget

FRESH_INTERPRETER_SECONDS: Final = 10.0


def test_the_standalone_probe_succeeds() -> None:
    result = run_probe_smoke()

    assert result.returncode == 0, result.stderr


@pytest.mark.benchmark
def test_the_standalone_probe_starts_within_budget(
    benchmark: BenchmarkFixture,
    wall_clock_budget: WallClockBudget,
) -> None:
    benchmark.pedantic(run_probe_smoke, rounds=1, iterations=1)
    measured = benchmark.stats

    if measured is None:
        pytest.fail("the standalone probe ran but pytest-benchmark timed nothing")

    elapsed = measured.stats.median
    benchmark.extra_info["median_seconds"] = elapsed
    benchmark.extra_info["budget_seconds"] = FRESH_INTERPRETER_SECONDS
    benchmark.extra_info["processes"] = 1

    if wall_clock_budget.enforced:
        assert elapsed < FRESH_INTERPRETER_SECONDS
