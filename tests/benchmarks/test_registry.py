import sys
from typing import TYPE_CHECKING, Final

import pytest

from lofi.registry import Registry

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_benchmark.fixture import BenchmarkFixture
    from tests.benchmarks.conftest import WallClockBudget

FAMILY: Final = "registry_benchmark"
PLUGIN_COUNT: Final = 40


@pytest.fixture
def plugin_modules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[str, ...]:
    modules = tuple(f"lofi_registry_benchmark_{index}" for index in range(PLUGIN_COUNT))

    for index, module in enumerate(modules):
        (tmp_path / f"{module}.py").write_text(
            f"COMPONENT = 'component-{index}'\n",
            encoding="utf-8",
            newline="\n",
        )

    metadata = tmp_path / "lofi_registry_benchmark-1.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: lofi-registry-benchmark\nVersion: 1.0\n",
        encoding="utf-8",
        newline="\n",
    )
    entries = "\n".join(
        f"component-{index} = {module}:COMPONENT"
        for index, module in enumerate(modules)
    )
    (metadata / "entry_points.txt").write_text(
        f"[lofi.{FAMILY}]\n{entries}\n",
        encoding="utf-8",
        newline="\n",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    return modules


def _discover(plugin_modules: tuple[str, ...]) -> int:
    for module in plugin_modules:
        sys.modules.pop(module, None)

    return len(Registry[str](FAMILY).members())


def test_plugin_discovery_returns_every_component(
    plugin_modules: tuple[str, ...],
) -> None:
    assert _discover(plugin_modules) == PLUGIN_COUNT


def test_plugin_discovery_meets_its_budgets(
    benchmark: BenchmarkFixture,
    plugin_modules: tuple[str, ...],
    wall_clock_budget: WallClockBudget,
) -> None:
    benchmark(_discover, plugin_modules)
    measured = benchmark.stats

    if measured is None:
        pytest.fail("the registry ran but pytest-benchmark timed nothing")

    total_ms = measured.stats.median * 1e3
    per_plugin_ms = total_ms / PLUGIN_COUNT
    benchmark.extra_info["plugins"] = PLUGIN_COUNT
    benchmark.extra_info["median_ms_total"] = total_ms
    benchmark.extra_info["median_ms_per_plugin"] = per_plugin_ms
    benchmark.extra_info["budget"] = "component-registry"

    # enforce the 200 ms total discovery budget and the 5 ms per-plugin budget
    if wall_clock_budget.enforced:
        assert total_ms <= 200
        assert per_plugin_ms <= 5
