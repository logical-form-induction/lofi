"""Fixtures shared by the benchmark suite.

``pinned_chart`` compiles the recorded grammar pack and checks the pinned
sentence file against its fingerprint before anything is timed.
``harness_floor`` is the cost of one empty call through the benchmark runner's
loop, measured once per session, which a chart benchmark subtracts from its
timing before comparing against its budget. ``wall_clock_budget`` says whether
the session holds a benchmark to its absolute wall-clock budget, as the
``wall_clock_budgets`` ini option declares. An autouse fixture refuses to time
anything while the debug checks are on, since a checked build times the checks
rather than the operation.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Final, NamedTuple

import pytest
from pytest_benchmark.utils import load_timer

from lofi import _checks
from lofi.run._benchmark_fixture import PinnedChart, load_pinned_chart

if TYPE_CHECKING:
    from collections.abc import Callable


#: The directory the pinned fixtures are written under.
FIXTURES: Final = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _unchecked_build(request: pytest.FixtureRequest) -> None:
    """Refuse to run a benchmark while the debug checks are on.

    A benchmark that ``--benchmark-skip`` skips is skipped at collection and
    never reaches this fixture.
    """
    if "benchmark" not in request.fixturenames:
        return

    if _checks.CHECKS:
        pytest.fail(
            "LOFI_CHECK is on, so this benchmark would time the debug checks "
            "rather than the operation; run the benchmarks with LOFI_CHECK=0",
            pytrace=False,
        )


def _empty() -> None:
    """Do nothing."""


def _round(
    timer: Callable[[], float], call: Callable[[], None], iterations: int
) -> float:
    """Return what one round of calls takes, timed as the runner times it.

    Parameters
    ----------
    timer : Callable[[], float]
        The clock, read once before the calls and once after them.
    call : Callable[[], None]
        What the round calls.
    iterations : int
        How many calls the round makes.
    """
    start = timer()

    for _ in range(iterations):
        call()

    end = timer()

    return end - start


def measure_floor(
    timer: Callable[[], float],
    call: Callable[[], None],
    *,
    rounds: int,
    iterations: int,
) -> float:
    """Return the seconds one call costs under the runner's loop.

    Parameters
    ----------
    timer : Callable[[], float]
        The clock the benchmarks read, called once before and once after each
        round.
    call : Callable[[], None]
        What to time.
    rounds : int
        How many rounds to time. The fastest is the floor, since anything
        slower is interference.
    iterations : int
        How many calls each round makes.
    """
    fastest = min(_round(timer, call, iterations) for _ in range(rounds))

    return fastest / iterations


@pytest.fixture(scope="session")
def harness_floor(pytestconfig: pytest.Config) -> float:
    """Return the seconds one empty call costs, measured once per session.

    The clock is the one the session's benchmarks read.
    """
    timer = load_timer(pytestconfig.getoption("benchmark_timer")).target

    return measure_floor(timer, _empty, rounds=7, iterations=20_000)


class WallClockBudget(NamedTuple):
    """Whether a session holds benchmarks to their absolute wall-clock budgets.

    An absolute budget is a number of seconds that only the fixed benchmark
    machine is held to. A session that compares a commit with its parent on
    the same machine reads timings relative to each other instead, and turns
    the absolute budgets off.
    """

    enforced: bool


@pytest.fixture(scope="session")
def wall_clock_budget(pytestconfig: pytest.Config) -> WallClockBudget:
    """Return whether this session enforces absolute wall-clock budgets."""
    return WallClockBudget(enforced=bool(pytestconfig.getini("wall_clock_budgets")))


@pytest.fixture(scope="session")
def pinned_chart() -> PinnedChart:
    """Return the pinned chart fixture, or skip the test when it is not on disk.

    Returns
    -------
    PinnedChart
        The two hundred pinned sentences with the grammar they parse under.
    """
    sentences_path = FIXTURES / "chart_200.jsonl"
    fingerprint_path = FIXTURES / "chart_200.fingerprint"
    absent = [path for path in (sentences_path, fingerprint_path) if not path.is_file()]

    if absent:
        named = ", ".join(str(path) for path in absent)
        pytest.skip(
            f"the pinned chart fixture is absent ({named}); "
            "scripts/fixtures/make_chart_fixture.py writes it"
        )

    return load_pinned_chart(sentences_path, fingerprint_path)
