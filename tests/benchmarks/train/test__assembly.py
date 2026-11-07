"""Benchmark the current component-wiring boundary."""

from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

import pytest
from tests.conformance.component_plugin import plugin_context

from lofi.spec import PACKAGED_CONFIG_ROOT, RunSpecCodec
from lofi.train._assembly import ComponentGraph, wire_components

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pytest_benchmark.fixture import BenchmarkFixture
    from tests.benchmarks.conftest import WallClockBudget

    from lofi.spec import ComponentSpec


CONFIG: Final = (
    Path(__file__).resolve().parents[3] / "lofi/configs/run/synthetic-parser.yaml"
)
CODEC: Final = RunSpecCodec(search_path=(PACKAGED_CONFIG_ROOT,))
ROLES: Final = ("token_encoder", "subword_pooler")


def _front() -> Mapping[str, ComponentSpec]:
    """Return the current production declarations wired in one graph call."""
    declared = CODEC(CONFIG).model.base_components()

    return MappingProxyType({role: declared[role] for role in ROLES})


FRONT: Final = _front()
CONTEXT: Final = plugin_context(component="model")


def _wire() -> ComponentGraph:
    """Wire the current token front from fresh component instances."""
    return wire_components(
        FRONT, CONTEXT, pretrained_resources={}, built={}, derived={}
    )


def test_the_wiring_fixture_is_the_current_production_token_front() -> None:
    graph = _wire()

    assert tuple(FRONT) == ROLES
    assert graph.order == ROLES
    assert tuple(graph.components) == ROLES
    assert (
        graph.dependency("subword_pooler", "encoder")
        is graph.components["token_encoder"]
    )


@pytest.mark.benchmark
def test_the_current_production_token_front_wires_within_budget(
    benchmark: BenchmarkFixture,
    wall_clock_budget: WallClockBudget,
) -> None:
    graph: ComponentGraph = benchmark.pedantic(
        _wire,
        rounds=10,
        warmup_rounds=2,
        iterations=1,
    )
    measured = benchmark.stats

    if measured is None:
        pytest.fail("the component graph wired but pytest-benchmark timed nothing")

    median_ms = measured.stats.median * 1e3
    benchmark.extra_info["budget"] = "training-assembly"
    benchmark.extra_info["components"] = len(graph.components)
    benchmark.extra_info["roles"] = list(graph.order)
    benchmark.extra_info["median_ms"] = median_ms

    if wall_clock_budget.enforced:
        assert median_ms <= 200
