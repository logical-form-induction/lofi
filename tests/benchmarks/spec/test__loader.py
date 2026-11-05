"""The B-1 crossing benchmark for run specifications."""

import json
import subprocess
import sys
from typing import TYPE_CHECKING, Final, NamedTuple, cast

import pytest

if TYPE_CHECKING:
    from pathlib import Path

    from tests.benchmarks.conftest import WallClockBudget

    from lofi._types import JsonValue

COLD_BUDGET_MS: Final = 2500.0
WARM_BUDGET_MS: Final = 50.0
PROCESS_TIMEOUT_SECONDS: Final = 30.0

PROBE: Final = r"""\
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Literal

from lofi.spec import (
    DatasetSpec,
    FeaturizerSpec,
    LossSpec,
    OptimizerSpec,
    ParserSpec,
    RunSpecCodec,
    SinkSpec,
    TrainerSpec,
    TermEncoderSpec,
)


class BenchParser(ParserSpec):
    kind: Literal["p30-parser"] = "p30-parser"


class BenchEncoder(TermEncoderSpec):
    kind: Literal["p30-encoder"] = "p30-encoder"
    width: int


class BenchLoss(LossSpec):
    kind: Literal["p30-loss"] = "p30-loss"


class BenchDataset(DatasetSpec):
    kind: Literal["p30-dataset"] = "p30-dataset"


class BenchFeaturizer(FeaturizerSpec):
    kind: Literal["p30-featurizer"] = "p30-featurizer"


class BenchTrainer(TrainerSpec):
    kind: Literal["p30-trainer"] = "p30-trainer"
    epochs: int
    batch_size: int


class BenchOptimizer(OptimizerSpec):
    kind: Literal["p30-optimizer"] = "p30-optimizer"
    rate: float


class BenchSink(SinkSpec):
    kind: Literal["p30-sink"] = "p30-sink"
    every: int


codec = RunSpecCodec()
document = Path(sys.argv[1])
started = time.perf_counter_ns()
spec = codec(
    document,
    profile="fast",
    extra={"trainer.batch_size": 16},
    overrides=("trainer.epochs=3", "optimizer.rate=0.25"),
)
cold_ms = (time.perf_counter_ns() - started) / 1_000_000

warm = []
for _ in range(25):
    started = time.perf_counter_ns()
    codec(
        document,
        profile="fast",
        extra={"trainer.batch_size": 16},
        overrides=("trainer.epochs=3", "optimizer.rate=0.25"),
    )
    warm.append((time.perf_counter_ns() - started) / 1_000_000)

print(
    json.dumps(
        {
            "cold_ms": cold_ms,
            "warm_ms": statistics.median(warm),
            "panproto_imported": "panproto" in sys.modules,
            "lairs_imported": "lairs" in sys.modules,
            "telemetry_imported": any(
                name in sys.modules for name in ("wandb", "trackio", "mlflow")
            ),
            "epochs": spec.trainer.epochs,
            "batch_size": spec.trainer.batch_size,
            "encoder_width": spec.model.encoders["terms"].width,
            "sink_every": spec.sinks["log"].every,
        }
    )
)
"""


class ProbeResult(NamedTuple):
    """Timing and result from one fresh interpreter."""

    cold_ms: float
    warm_ms: float
    panproto_imported: bool
    lairs_imported: bool
    telemetry_imported: bool
    epochs: int
    batch_size: int
    encoder_width: int
    sink_every: int


def _number(payload: dict[str, JsonValue], key: str) -> float:
    """Return a numeric probe field.

    Parameters
    ----------
    payload : dict[str, JsonValue]
        Decoded probe result.
    key : str
        Field to read.

    Raises
    ------
    TypeError
        If the field is not numeric.
    """
    value = payload[key]

    if not isinstance(value, int | float):
        msg = f"{key} is not numeric"

        raise TypeError(msg)

    return float(value)


def _integer(payload: dict[str, JsonValue], key: str) -> int:
    """Return an integer probe field.

    Parameters
    ----------
    payload : dict[str, JsonValue]
        Decoded probe result.
    key : str
        Field to read.

    Raises
    ------
    TypeError
        If the field is not an integer.
    """
    value = payload[key]

    if not isinstance(value, int) or isinstance(value, bool):
        msg = f"{key} is not an integer"

        raise TypeError(msg)

    return value


def _boolean(payload: dict[str, JsonValue], key: str) -> bool:
    """Return a Boolean probe field.

    Parameters
    ----------
    payload : dict[str, JsonValue]
        Decoded probe result.
    key : str
        Field to read.

    Raises
    ------
    TypeError
        If the field is not Boolean.
    """
    value = payload[key]

    if not isinstance(value, bool):
        msg = f"{key} is not Boolean"

        raise TypeError(msg)

    return value


def _run_probe(document: Path) -> ProbeResult:
    """Measure cold and warm composition in a fresh interpreter.

    Parameters
    ----------
    document : Path
        Configuration document to compose.

    Returns
    -------
    ProbeResult
        Crossing times, import state, and selected result fields.
    """
    completed = subprocess.run(
        [sys.executable, "-c", PROBE, str(document)],
        check=True,
        capture_output=True,
        encoding="utf-8",
        timeout=PROCESS_TIMEOUT_SECONDS,
    )
    payload = cast("dict[str, JsonValue]", json.loads(completed.stdout))
    return ProbeResult(
        cold_ms=_number(payload, "cold_ms"),
        warm_ms=_number(payload, "warm_ms"),
        panproto_imported=_boolean(payload, "panproto_imported"),
        lairs_imported=_boolean(payload, "lairs_imported"),
        telemetry_imported=_boolean(payload, "telemetry_imported"),
        epochs=_integer(payload, "epochs"),
        batch_size=_integer(payload, "batch_size"),
        encoder_width=_integer(payload, "encoder_width"),
        sink_every=_integer(payload, "sink_every"),
    )


@pytest.fixture
def run_document(tmp_path: Path) -> Path:
    """Write a run assembled from a document, fragment, and profile."""
    document = {
        "defaults": [{"model.encoders.terms": "wide"}],
        "parse": {"retain": "forest", "reachability": "all"},
        "model": {
            "language": {"from": "data"},
            "parser": {"kind": "p30-parser"},
        },
        "loss": {"terms": [{"objective": {"kind": "p30-loss"}}]},
        "data": {
            "datasets": {
                "train": {
                    "dataset": {"kind": "p30-dataset"},
                    "featurizer": {"kind": "p30-featurizer"},
                }
            }
        },
        "trainer": {
            "kind": "p30-trainer",
            "epochs": 1,
            "batch_size": 8,
        },
        "release": {"checkpoint": "resume-latest"},
        "optimizer": {"kind": "p30-optimizer", "rate": 0.1},
        "device_policy": {
            "device": "cpu",
            "compute_dtype": "float32",
            "deterministic": True,
            "matmul_precision": "highest",
            "activation_memory_bytes": 8_000_000_000,
        },
        "sinks": {"log": {"kind": "p30-sink", "every": "${trainer.epochs}"}},
    }
    profile = {"trainer": {"epochs": 2}}
    fragment = {"kind": "p30-encoder", "width": 64}
    path = tmp_path / "run.json"
    profiles = tmp_path / "profiles"
    fragments = tmp_path / "model" / "encoders" / "terms"
    profiles.mkdir()
    fragments.mkdir(parents=True)
    path.write_text(json.dumps(document), encoding="utf-8", newline="\n")
    (profiles / "fast.json").write_text(
        json.dumps(profile), encoding="utf-8", newline="\n"
    )
    (fragments / "wide.json").write_text(
        json.dumps(fragment), encoding="utf-8", newline="\n"
    )

    return path


@pytest.mark.benchmark
def test_the_run_spec_crossing_meets_p30(
    run_document: Path, wall_clock_budget: WallClockBudget
) -> None:
    result = _run_probe(run_document)

    if wall_clock_budget.enforced:
        assert result.cold_ms <= COLD_BUDGET_MS
        assert result.warm_ms <= WARM_BUDGET_MS


def test_the_run_spec_crossing_loads_without_theory_or_telemetry_imports(
    run_document: Path,
) -> None:
    result = _run_probe(run_document)

    assert result.panproto_imported is False
    assert result.lairs_imported is False
    assert result.telemetry_imported is False
    assert result.epochs == 3
    assert result.batch_size == 16
    assert result.encoder_width == 64
    assert result.sink_every == 3
