from typing import TYPE_CHECKING, Final

import pytest
from hypothesis import given, seed, settings
from tests.conformance.lf._naive_check import TermDeclaration, type_of
from tests.conformance.lf.strategies import SIGNATURE, TRUTH, well_typed

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture
    from tests.benchmarks.conftest import WallClockBudget

EXAMPLES: Final = 1_000
BUDGET_SECONDS: Final = 5.0
SEED: Final = 20260912


def _draw_and_check() -> int:
    """Generate and check one thousand typed declarations."""
    checked = 0

    @seed(SEED)
    @settings(max_examples=EXAMPLES, deadline=None, database=None)
    @given(term=well_typed(SIGNATURE, TRUTH, depth=3))
    def check(term: TermDeclaration) -> None:
        nonlocal checked

        checked += int(type_of(SIGNATURE, term) == TRUTH)

    check()

    return checked


def test_a_thousand_typed_draws_are_checked() -> None:
    assert _draw_and_check() == EXAMPLES


def test_a_thousand_typed_draws_complete_under_five_seconds(
    benchmark: BenchmarkFixture,
    wall_clock_budget: WallClockBudget,
) -> None:
    checked = benchmark.pedantic(_draw_and_check, rounds=1, iterations=1)
    measured = benchmark.stats

    if measured is None:
        pytest.fail("the generator ran but pytest-benchmark timed nothing")

    elapsed = measured.stats.median
    benchmark.extra_info["draws"] = checked
    benchmark.extra_info["seconds"] = elapsed
    benchmark.extra_info["budget_seconds"] = BUDGET_SECONDS

    if wall_clock_budget.enforced:
        assert elapsed < BUDGET_SECONDS
