"""Benchmarks importing the run package in a fresh process."""

import subprocess
import sys
from statistics import median
from typing import TYPE_CHECKING, Final

import pytest

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture
    from tests.benchmarks.conftest import WallClockBudget

SOURCE: Final = (
    "import sys, time\n"
    "start = time.perf_counter()\n"
    "import lofi.run\n"
    "print(time.perf_counter() - start, 'torch' in sys.modules)"
)


def import_in_a_fresh_process() -> tuple[float, bool]:
    completed = subprocess.run(
        [sys.executable, "-c", SOURCE],
        check=True,
        capture_output=True,
        encoding="utf-8",
    )
    seconds, imported_torch = completed.stdout.split()

    return float(seconds), imported_torch == "True"


def _observe(imports: list[float], torch_imported: list[bool]) -> None:
    seconds, imported_torch = import_in_a_fresh_process()
    imports.append(seconds)
    torch_imported.append(imported_torch)


def test_importing_the_run_package_does_not_import_torch() -> None:
    imports: list[float] = []
    torch_imported: list[bool] = []

    _observe(imports, torch_imported)

    assert not any(torch_imported)


def test_importing_the_run_package_in_a_fresh_process_meets_its_budget(
    benchmark: BenchmarkFixture,
    wall_clock_budget: WallClockBudget,
) -> None:
    imports: list[float] = []
    torch_imported: list[bool] = []

    benchmark.pedantic(_observe, args=(imports, torch_imported), rounds=5)
    measured = benchmark.stats

    if measured is None:
        pytest.fail("the import ran but pytest-benchmark timed nothing")

    import_s = median(imports)
    benchmark.extra_info["rounds"] = len(imports)
    benchmark.extra_info["median_import_s"] = import_s
    benchmark.extra_info["max_import_s"] = max(imports)
    benchmark.extra_info["median_process_s"] = measured.stats.median
    benchmark.extra_info["budget"] = "package-import"

    if wall_clock_budget.enforced:
        assert import_s <= 2
