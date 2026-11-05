"""Benchmarks the pinned reference judgment fit through genuine NUTS."""

from typing import TYPE_CHECKING

import pytest
from tests.conformance.posterior_fitter import reference_program

from lofi.judgment import REFERENCE_NUTS_PLAN, QuiversNutsFitter

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture
    from tests.benchmarks.conftest import WallClockBudget


def test_the_reference_posterior_fit_runs_the_pinned_nuts_plan(
    benchmark: BenchmarkFixture,
    wall_clock_budget: WallClockBudget,
) -> None:
    program = reference_program()
    fitter = QuiversNutsFitter(REFERENCE_NUTS_PLAN)
    fitted = benchmark.pedantic(
        fitter.fit,
        args=(program,),
        rounds=1,
        warmup_rounds=0,
        iterations=1,
    )
    measured = benchmark.stats

    if measured is None:
        pytest.fail(
            "the reference posterior fit ran but pytest-benchmark timed nothing"
        )

    diagnostics = fitted.diagnostics
    benchmark.extra_info["budget"] = "posterior-fit"
    benchmark.extra_info["engine"] = type(fitter).__name__
    benchmark.extra_info["rows"] = program.rows
    benchmark.extra_info["participants"] = program.sizes["Participant"]
    benchmark.extra_info["items"] = program.sizes["Item"]
    benchmark.extra_info["levels"] = program.levels
    benchmark.extra_info["readouts"] = list(program.program.readouts)
    benchmark.extra_info["random_effects"] = list(program.program.groups)
    benchmark.extra_info["sites"] = [slot.site for slot in program.layout]
    benchmark.extra_info["parameters"] = program.size
    benchmark.extra_info["warmup"] = fitter.plan.warmup
    benchmark.extra_info["retained"] = fitter.plan.retained
    benchmark.extra_info["chains"] = fitter.plan.chains
    benchmark.extra_info["seed"] = fitter.plan.seed
    benchmark.extra_info["device"] = fitted.log_densities.device.type
    benchmark.extra_info["divergences"] = diagnostics.divergences
    benchmark.extra_info["max_rhat"] = diagnostics.max_rhat
    benchmark.extra_info["min_ess"] = diagnostics.min_ess
    benchmark.extra_info["median_ms_per_fit"] = measured.stats.median * 1e3
    mean_ms = measured.stats.mean * 1e3
    benchmark.extra_info["mean_ms_per_fit"] = mean_ms

    if wall_clock_budget.enforced:
        assert mean_ms <= 6_000_000.0
