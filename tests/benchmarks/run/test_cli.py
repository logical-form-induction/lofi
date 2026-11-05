"""Benchmark the command line's process entry."""

import json
import subprocess
import sys
from typing import TYPE_CHECKING, Final, NamedTuple, cast

import pytest

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_benchmark.fixture import BenchmarkFixture
    from tests.benchmarks.conftest import WallClockBudget

    from lofi._types import JsonValue

COLD_BUDGET_MS: Final = 2500.0
PROCESS_TIMEOUT_SECONDS: Final = 30.0

PROBE: Final = r"""\
import io
import json
import sys
import time
from typing import Literal

from lofi.run import main
from lofi.spec import (
    DatasetSpec,
    FeaturizerSpec,
    LossSpec,
    OptimizerSpec,
    ParserSpec,
    SinkSpec,
    TrainerSpec,
)


class BenchParser(ParserSpec):
    kind: Literal["cli-parser"] = "cli-parser"


class BenchLoss(LossSpec):
    kind: Literal["cli-loss"] = "cli-loss"


class BenchDataset(DatasetSpec):
    kind: Literal["cli-dataset"] = "cli-dataset"


class BenchFeaturizer(FeaturizerSpec):
    kind: Literal["cli-featurizer"] = "cli-featurizer"


class BenchTrainer(TrainerSpec):
    kind: Literal["cli-trainer"] = "cli-trainer"
    epochs: int


class BenchOptimizer(OptimizerSpec):
    kind: Literal["cli-optimizer"] = "cli-optimizer"
    rate: float


class BenchSink(SinkSpec):
    kind: Literal["cli-sink"] = "cli-sink"


stdout = io.StringIO()
started = time.perf_counter_ns()
status = main(
    [
        "show-config",
        "--config",
        sys.argv[1],
        "--profile",
        "fast",
        "--extra",
        sys.argv[2],
        "--set",
        "optimizer.rate=0.25",
    ],
    stdout=stdout,
    stderr=sys.stderr,
)
cold_ms = (time.perf_counter_ns() - started) / 1_000_000
lines = stdout.getvalue().splitlines()

print(
    json.dumps(
        {
            "cold_ms": cold_ms,
            "status": status,
            "epochs_line": next(line.strip() for line in lines if "epochs:" in line),
            "rate_line": next(line.strip() for line in lines if "rate:" in line),
        }
    )
)
"""


class ProbeResult(NamedTuple):
    """Timing and result from one fresh interpreter."""

    cold_ms: float
    status: int
    epochs_line: str
    rate_line: str


def _run_probe(document: Path, overlay: Path) -> ProbeResult:
    """Run the command line once in a fresh interpreter and time the command.

    Parameters
    ----------
    document : Path
        Configuration document to show.
    overlay : Path
        Extra overlay document to apply.

    Returns
    -------
    ProbeResult
        Command time, exit status, and two rendered lines.
    """
    completed = subprocess.run(
        [sys.executable, "-c", PROBE, str(document), str(overlay)],
        check=True,
        capture_output=True,
        encoding="utf-8",
        timeout=PROCESS_TIMEOUT_SECONDS,
    )
    payload = cast("dict[str, JsonValue]", json.loads(completed.stdout))
    cold_ms = payload["cold_ms"]
    status = payload["status"]
    epochs_line = payload["epochs_line"]
    rate_line = payload["rate_line"]

    if not isinstance(cold_ms, int | float) or isinstance(cold_ms, bool):
        msg = "cold_ms is not numeric"

        raise TypeError(msg)

    if not isinstance(status, int) or isinstance(status, bool):
        msg = "status is not an integer"

        raise TypeError(msg)

    if not isinstance(epochs_line, str) or not isinstance(rate_line, str):
        msg = "a rendered line is not text"

        raise TypeError(msg)

    return ProbeResult(float(cold_ms), status, epochs_line, rate_line)


@pytest.fixture
def run_documents(tmp_path: Path) -> tuple[Path, Path]:
    """Write a run document with a profile beside it, and an overlay."""
    document = {
        "parse": {"retain": "forest", "reachability": "all"},
        "model": {
            "language": {"from": "data"},
            "parser": {"kind": "cli-parser"},
        },
        "loss": {"terms": [{"objective": {"kind": "cli-loss"}}]},
        "data": {
            "datasets": {
                "train": {
                    "dataset": {"kind": "cli-dataset"},
                    "featurizer": {"kind": "cli-featurizer"},
                }
            }
        },
        "trainer": {"kind": "cli-trainer", "epochs": 1},
        "release": {"checkpoint": "resume-latest"},
        "optimizer": {"kind": "cli-optimizer", "rate": 0.1},
        "device_policy": {
            "device": "cpu",
            "compute_dtype": "float32",
            "deterministic": True,
            "matmul_precision": "highest",
            "activation_memory_bytes": 8_000_000_000,
        },
        "sinks": {"log": {"kind": "cli-sink"}},
    }
    path = tmp_path / "run.json"
    overlay = tmp_path / "overlay.json"
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    path.write_text(json.dumps(document), encoding="utf-8", newline="\n")
    (profiles / "fast.json").write_text(
        json.dumps({"trainer": {"epochs": 2}}), encoding="utf-8", newline="\n"
    )
    overlay.write_text(
        json.dumps({"device_policy": {"deterministic": False}}),
        encoding="utf-8",
        newline="\n",
    )

    return path, overlay


@pytest.mark.benchmark
def test_showing_a_configuration_from_a_fresh_process_meets_p30(
    benchmark: BenchmarkFixture,
    run_documents: tuple[Path, Path],
    wall_clock_budget: WallClockBudget,
) -> None:
    result = benchmark.pedantic(
        _run_probe, args=run_documents, rounds=5, warmup_rounds=0, iterations=1
    )
    measured = benchmark.stats

    if measured is None:
        pytest.fail("the command ran but pytest-benchmark timed nothing")

    process_ms = measured.stats.median * 1e3
    benchmark.extra_info["budget"] = "command-line-entry"
    benchmark.extra_info["budget_cell"] = "cli-show-config"
    benchmark.extra_info["cold_ms"] = result.cold_ms
    benchmark.extra_info["process_ms"] = process_ms

    if wall_clock_budget.enforced:
        assert result.cold_ms <= COLD_BUDGET_MS
        assert process_ms <= COLD_BUDGET_MS


def test_showing_a_configuration_reports_the_merged_sources(
    run_documents: tuple[Path, Path],
) -> None:
    result = _run_probe(*run_documents)

    assert result.status == 0
    assert result.epochs_line == "epochs: 2  # profile fast"
    assert result.rate_line == "rate: 0.25  # override optimizer.rate=0.25"
