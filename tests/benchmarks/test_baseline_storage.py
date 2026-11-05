"""What a benchmark comparison reads, and what it can fail on."""

import shutil
from pathlib import Path
from typing import Final

import pytest
from pytest_benchmark.utils import get_machine_id

#: The directory the committed baselines are saved under.
BASELINES: Final = Path(__file__).parent / "baselines"

TIMED: Final = """
import time


def test_sleeps(benchmark):
    benchmark(time.sleep, {duration})
"""

#: Enough rounds for a mean, few enough that the suite does not wait on them.
QUICK: Final = (
    "--benchmark-only",
    "--benchmark-min-rounds=5",
    "--benchmark-max-time=0.01",
)

#: Multiplied against the baseline sleep to make a regression no clock can miss.
REGRESSION: Final = 10

BASELINE_SLEEP: Final = 0.001

#: Every saved run under every machine directory, in the tool's own spelling.
EVERY_SAVED_RUN: Final = "*/[0-9][0-9][0-9][0-9]_"


def _committed() -> tuple[Path, ...]:
    """Return the JSON files committed under a machine directory of the baselines."""
    return tuple(sorted(BASELINES.glob("*/*.json")))


def _timed(
    pytester: pytest.Pytester, duration: float, *options: str
) -> pytest.RunResult:
    """Time one sleep of ``duration`` seconds under the given options."""
    pytester.makepyfile(TIMED.format(duration=duration))

    return pytester.runpytest_subprocess("-p", "no:cacheprovider", *QUICK, *options)


def _saved(pytester: pytest.Pytester, store: Path, name: str) -> None:
    """Save a run of the baseline sleep into ``store`` under ``name``."""
    result = _timed(
        pytester,
        BASELINE_SLEEP,
        f"--benchmark-storage={store}",
        f"--benchmark-save={name}",
    )

    result.assert_outcomes(passed=1)


def _reported(result: pytest.RunResult) -> str:
    """Return what the benchmark plugin wrote, which is the run's standard error."""
    return result.stderr.str()


def _compared(result: pytest.RunResult) -> tuple[str, ...]:
    """Return the stored runs the run reports comparing against, in the order given."""
    marker = "Comparing against benchmarks from:"

    return tuple(
        line.split(marker, 1)[1].strip()
        for line in result.stderr.lines
        if marker in line
    )


def test_the_configured_storage_names_the_committed_baselines_from_the_repository_root(
    pytestconfig: pytest.Config,
) -> None:
    configured = Path(pytestconfig.getoption("benchmark_storage"))

    assert (pytestconfig.rootpath / configured).resolve() == BASELINES.resolve()


def test_a_saved_run_lands_under_this_machine_id(pytester: pytest.Pytester) -> None:
    store = pytester.path / "store"
    _saved(pytester, store, "baseline")

    assert sorted(path.relative_to(store) for path in store.rglob("*.json")) == [
        Path(get_machine_id()) / "0001_baseline.json"
    ]


def test_every_committed_baseline_is_a_run_a_comparison_reads(
    pytester: pytest.Pytester,
) -> None:
    committed = _committed()
    if not committed:
        pytest.skip(f"no run is committed under {BASELINES}")
    store = pytester.path / "store"
    shutil.copytree(BASELINES, store)

    result = _timed(
        pytester,
        BASELINE_SLEEP,
        f"--benchmark-storage={store}",
        f"--benchmark-compare={EVERY_SAVED_RUN}",
    )

    assert sorted(_compared(result)) == sorted(
        str(path.relative_to(BASELINES)) for path in committed
    )


def test_a_regression_against_a_stored_run_fails_the_comparison(
    pytester: pytest.Pytester,
) -> None:
    store = pytester.path / "store"
    _saved(pytester, store, "baseline")

    result = _timed(
        pytester,
        BASELINE_SLEEP * REGRESSION,
        f"--benchmark-storage={store}",
        "--benchmark-compare",
        "--benchmark-compare-fail=mean:10%",
    )

    assert result.ret != 0
    assert "Performance has regressed" in _reported(result)


def test_the_same_regression_passes_against_an_empty_store(
    pytester: pytest.Pytester,
) -> None:
    store = pytester.path / "empty"

    result = _timed(
        pytester,
        BASELINE_SLEEP * REGRESSION,
        f"--benchmark-storage={store}",
        "--benchmark-compare",
    )

    result.assert_outcomes(passed=1)
    assert result.ret == 0
    assert "Can't compare" in _reported(result)


def test_a_failure_threshold_over_an_empty_store_is_refused(
    pytester: pytest.Pytester,
) -> None:
    store = pytester.path / "empty"

    result = _timed(
        pytester,
        BASELINE_SLEEP * REGRESSION,
        f"--benchmark-storage={store}",
        "--benchmark-compare",
        "--benchmark-compare-fail=mean:10%",
    )

    assert result.ret == pytest.ExitCode.USAGE_ERROR
    assert "--benchmark-compare-fail requires valid --benchmark-compare" in _reported(
        result
    )


def test_a_comparison_carrying_no_threshold_passes_the_regression_it_prints(
    pytester: pytest.Pytester,
) -> None:
    store = pytester.path / "store"
    _saved(pytester, store, "baseline")

    result = _timed(
        pytester,
        BASELINE_SLEEP * REGRESSION,
        f"--benchmark-storage={store}",
        "--benchmark-compare",
    )

    result.assert_outcomes(passed=1)
    assert result.ret == 0
    assert _compared(result)


def test_a_comparison_with_no_argument_reads_only_the_last_stored_run(
    pytester: pytest.Pytester,
) -> None:
    store = pytester.path / "store"
    _saved(pytester, store, "first")
    _saved(pytester, store, "second")

    result = _timed(
        pytester,
        BASELINE_SLEEP,
        f"--benchmark-storage={store}",
        "--benchmark-compare",
    )
    named = _compared(result)

    assert len(named) == 1
    assert named[0].endswith("0002_second.json")


def test_a_comparison_naming_a_prefix_reads_every_stored_run_it_matches(
    pytester: pytest.Pytester,
) -> None:
    store = pytester.path / "store"
    _saved(pytester, store, "first")
    _saved(pytester, store, "second")

    result = _timed(
        pytester,
        BASELINE_SLEEP,
        f"--benchmark-storage={store}",
        "--benchmark-compare=000",
    )
    named = _compared(result)

    assert len(named) == 2
    assert named[0].endswith("0001_first.json")
    assert named[1].endswith("0002_second.json")
