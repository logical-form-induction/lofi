"""Wall-clock cost of the five-call determinism diagnostic."""

from typing import TYPE_CHECKING, Final, NamedTuple

import pytest
import torch
from torch import nn

from lofi.nn.sites import (
    Access,
    ComponentId,
    DTypeClass,
    Role,
    Site,
    SiteSpec,
)
from lofi.probe import ExternalModel, ExternalSite, ExternalSiteMap, determinism

if TYPE_CHECKING:
    from collections.abc import Callable

    from pytest_benchmark.fixture import BenchmarkFixture
    from tests.benchmarks.conftest import WallClockBudget
    from torch import Tensor

WIDTH: Final = 8
COMPONENT: Final = ComponentId("determinism-benchmark")
OUTPUT: Final = Site(COMPONENT, Role.RESIDUAL_STREAM, (0,))


class Subject(NamedTuple):
    """The observed module, its fixed call, and the requested point."""

    model: ExternalModel
    forward: Callable[[], Tensor]
    site: Site


@pytest.fixture(scope="module")
def subject() -> Subject:
    """Return a small real neural component behind the external adapter."""
    module = nn.Sequential(nn.Linear(WIDTH, WIDTH), nn.Tanh())
    given = torch.linspace(-1.0, 1.0, 4 * WIDTH).reshape(4, WIDTH)
    declared = ExternalSiteMap(
        COMPONENT,
        (
            ExternalSite(
                "0",
                Role.RESIDUAL_STREAM,
                0,
                SiteSpec(
                    ("item", "width"),
                    DTypeClass.REAL,
                    differentiable=True,
                    access=Access.OBSERVE,
                ),
            ),
        ),
    )

    return Subject(ExternalModel(module, declared), lambda: module(given), OUTPUT)


def test_five_call_determinism_stays_under_one_second(
    benchmark: BenchmarkFixture,
    subject: Subject,
    wall_clock_budget: WallClockBudget,
) -> None:
    report = benchmark(
        determinism,
        subject.model,
        subject.forward,
        sites=frozenset({subject.site}),
        seed=0,
    )
    measured = benchmark.stats

    if measured is None:
        pytest.fail("the determinism diagnostic ran but pytest-benchmark timed nothing")

    benchmark.extra_info["calls_per_diagnostic"] = 5
    benchmark.extra_info["sites"] = len(report.sites)
    benchmark.extra_info["mean_ms_per_diagnostic"] = measured.stats.mean * 1e3
    benchmark.extra_info["budget"] = "under one second"

    if wall_clock_budget.enforced:
        assert measured.stats.mean < 1.0
