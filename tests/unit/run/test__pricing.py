import json
from typing import TYPE_CHECKING, Final

import numpy as np
import pytest
from tests.unit.run.conftest import (
    PACKAGED_RUNS,
    a_packaged_run,
    write_priced_experiment,
)
from tests.unit.run.test__experiment import (
    a_judgment_experiment_document,
    write_experiment,
)
from tests.unit.run.test_baselines import baseline_experiment

from lofi import _checks
from lofi.data import BatchShape
from lofi.run import (
    COST_MODEL,
    SECONDS_PER_HOUR,
    UNPRICED,
    ExperimentPrice,
    GridCell,
    GridTiming,
    HardwareDescriptor,
    StackTiming,
    StepTimer,
    UndeclaredTrainingLengthError,
    UnpriceableExperimentError,
    expand_experiment,
    price_experiment,
    read_experiment,
    training_length,
)
from lofi.run._boundary.manifest import PriceIso
from lofi.spec import RunSpec, RunSpecCodec
from lofi.spec.trainers.distance import DistanceTrainerSpec
from lofi.spec.trainers.parser import ParserTrainerSpec
from lofi.training_plugins import TrainingLength, TrainingUnit

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from lofi._types import JsonValue
    from lofi.run import PointSeed


SEEDS: Final = [0, 1, 2]
PACKAGED_CODEC: Final = RunSpecCodec(search_path=(PACKAGED_RUNS, PACKAGED_RUNS.parent))


@pytest.fixture(autouse=True)
def unchecked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Price with the debug checks off, as a timed build requires."""
    monkeypatch.setattr(_checks, "CHECKS", False)


class Recording:
    """A step timer that answers declared timings and records what it built."""

    def __init__(
        self,
        seconds: Sequence[Sequence[float]],
        *,
        steps_per_unit: int,
        peak: int | None = None,
    ) -> None:
        self._seconds = tuple(tuple(run) for run in seconds)
        self._steps_per_unit = steps_per_unit
        self._peak = peak
        self.timed: list[tuple[RunSpec, PointSeed, str, int, int]] = []
        self.grids: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def time(
        self,
        spec: RunSpec,
        point: PointSeed,
        /,
        *,
        device: str,
        warmup: int,
        repeats: int,
    ) -> StackTiming:
        seconds = self._seconds[len(self.timed) % len(self._seconds)][:repeats]
        self.timed.append((spec, point, device, warmup, repeats))

        return StackTiming(
            steps_per_unit=self._steps_per_unit,
            positions=tuple(range(warmup, warmup + len(seconds))),
            seconds=seconds,
            peak_allocated_bytes=self._peak,
            shape=BatchShape(12.0, 4.0),
        )

    def time_grid(
        self,
        stacks: Mapping[tuple[int, int], RunSpec],
        point: PointSeed,
        /,
        *,
        device: str,
        warmup: int,
        repeats: int,
    ) -> GridTiming:
        del point, device, warmup
        lengths = tuple(dict.fromkeys(length for length, _ in stacks))
        batches = tuple(dict.fromkeys(batch for _, batch in stacks))
        self.grids.append((lengths, batches))

        return GridTiming(
            [
                GridCell(
                    length, batch, tuple(length * batch / 1_000 for _ in range(repeats))
                )
                for length, batch in stacks
            ]
        )

    def hardware(self, device: str, /) -> HardwareDescriptor:
        return HardwareDescriptor(
            device_name=device, driver=None, cuda=None, torch="2.8.0", host_cpu="arm"
        )


def a_grid(path: str, values: list[JsonValue]) -> dict[str, JsonValue]:
    return {
        "grid": [{"path": path, "values": values}],
        "seeds": list(SEEDS),
        "expected_points": len(values) * len(SEEDS),
    }


def priced(
    tmp_path: Path,
    run: dict[str, JsonValue],
    sweep: dict[str, JsonValue],
    timer: StepTimer,
    *,
    repeats: int = 3,
) -> ExperimentPrice:
    experiment = read_experiment(write_priced_experiment(tmp_path, run, sweep))

    return price_experiment(
        experiment, tmp_path, timer, device="cpu", warmup=1, repeats=repeats
    )


def test_an_experiment_builds_each_distinct_stack_its_points_declare_once(
    tmp_path: Path,
) -> None:
    timer = Recording([(0.5, 0.6, 0.7)], steps_per_unit=2)
    sweep = a_grid("optimizer.learning_rate", [0.001, 0.0005])

    price = priced(tmp_path, a_packaged_run("synthetic-parser"), sweep, timer)
    points = expand_experiment(
        read_experiment(tmp_path / "experiments/pricing.yaml"), tmp_path
    )

    assert {spec.optimizer.learning_rate for spec, *_ in timer.timed} == {
        0.0005,
        0.001,
    }
    assert [
        (device, warmup, repeats) for *_, device, warmup, repeats in timer.timed
    ] == [
        ("cpu", 1, 3),
        ("cpu", 1, 3),
    ]
    assert len(price.stacks) == 2
    assert sorted(point for stack in price.stacks for point in stack.points) == sorted(
        point.point_id for point in points
    )
    assert all(len(stack.points) == len(SEEDS) for stack in price.stacks)
    assert price.device == "cpu"
    assert price.cost_model == COST_MODEL
    assert price.experiment == "pricing"
    assert read_experiment(tmp_path / "experiments/pricing.yaml").compute is UNPRICED
    assert price.freeze_id == "freeze/1"


def test_each_stack_is_read_against_its_own_device_policys_budget(
    tmp_path: Path,
) -> None:
    timer = Recording([(0.5, 0.6, 0.7)], steps_per_unit=2, peak=2_000)
    sweep = a_grid("device_policy.activation_memory_bytes", [1_000, 4_000])

    price = priced(tmp_path, a_packaged_run("synthetic-parser"), sweep, timer)

    assert sorted(
        (stack.budget.max_bytes, stack.over_budget) for stack in price.stacks
    ) == [(1_000, True), (4_000, False)]
    assert {spec.device_policy.activation_memory_bytes for spec, *_ in timer.timed} == {
        1_000,
        4_000,
    }


def test_each_point_trains_for_the_length_its_run_document_declares(
    tmp_path: Path,
) -> None:
    timer = Recording([(0.25, 0.25)], steps_per_unit=5)

    price = priced(
        tmp_path,
        a_packaged_run("synthetic-parser"),
        a_grid("trainer.epochs", [1, 3]),
        timer,
        repeats=2,
    )

    assert sorted((stack.length, stack.steps) for stack in price.stacks) == [
        (TrainingLength(1, TrainingUnit.EPOCHS), 5),
        (TrainingLength(3, TrainingUnit.EPOCHS), 15),
    ]


def test_a_distance_trainer_is_priced_at_its_declared_steps() -> None:
    spec = RunSpecCodec()(PACKAGED_RUNS / "synthetic-distance.yaml")

    assert isinstance(spec.trainer, DistanceTrainerSpec)
    assert training_length(spec) == TrainingLength(
        spec.trainer.steps, TrainingUnit.STEPS
    )


def test_the_experiment_figures_match_a_recomputation_from_the_raw_timings(
    tmp_path: Path,
) -> None:
    timer = Recording(
        [(0.31, 0.29, 0.45, 0.30, 0.38), (1.2, 0.9, 1.1, 1.6, 1.0)],
        steps_per_unit=4,
    )

    price = priced(
        tmp_path,
        a_packaged_run("synthetic-parser"),
        a_grid("trainer.epochs", [2, 5]),
        timer,
        repeats=5,
    )
    record = json.loads(PriceIso().forward(price))
    resamples = record["resamples"]
    generator = np.random.default_rng(record["bootstrap_seed"])
    totals = np.zeros(resamples)

    for stack in record["stacks"]:
        seconds = np.asarray(stack["seconds"])
        draws = generator.integers(0, seconds.size, size=(resamples, seconds.size))
        steps = stack["length"]["count"] * stack["steps_per_unit"]
        totals += (
            len(stack["points"]) * steps * seconds[draws].mean(axis=1)
        ) / SECONDS_PER_HOUR

    p50, p90 = np.quantile(totals, [0.5, 0.9])
    per_step_deciles = sum(
        len(stack["points"])
        * stack["length"]["count"]
        * stack["steps_per_unit"]
        * float(np.percentile(stack["seconds"], 90))
        / SECONDS_PER_HOUR
        for stack in record["stacks"]
    )

    assert [stack["seconds"] for stack in record["stacks"]] == [
        [0.31, 0.29, 0.45, 0.30, 0.38],
        [1.2, 0.9, 1.1, 1.6, 1.0],
    ]
    assert resamples == (
        read_experiment(tmp_path / "experiments/pricing.yaml").pricing.resamples
    )
    assert price.p50_hours == pytest.approx(float(p50))
    assert price.p90_hours == pytest.approx(float(p90))
    assert price.p50_hours < price.p90_hours < per_step_deciles


def test_an_experiment_declares_the_resamples_its_price_is_read_off(
    tmp_path: Path,
) -> None:
    path = write_priced_experiment(
        tmp_path, a_packaged_run("synthetic-parser"), a_grid("trainer.epochs", [1])
    )
    document = json.loads(path.read_text(encoding="utf-8"))
    document["pricing"] = {"kind": "stack-bootstrap", "resamples": 50}
    path.write_text(json.dumps(document), encoding="utf-8", newline="\n")
    experiment = read_experiment(path)

    price = price_experiment(
        experiment,
        tmp_path,
        Recording([(0.5, 0.6)], steps_per_unit=2),
        device="cpu",
        warmup=1,
        repeats=2,
    )

    assert experiment.pricing.resamples == 50
    assert price.resamples == 50


def test_the_bootstrap_is_seeded_by_the_experiment_root(tmp_path: Path) -> None:
    timer = Recording([(0.5, 0.6, 0.7)], steps_per_unit=2)

    price = priced(
        tmp_path,
        a_packaged_run("synthetic-parser"),
        {"grid": [], "seeds": list(SEEDS), "expected_points": len(SEEDS)},
        timer,
    )
    experiment = read_experiment(tmp_path / "experiments/pricing.yaml")

    assert price.bootstrap_seed == experiment.seed_policy.root


@pytest.mark.usefixtures("open_ended_trainer")
def test_a_trainer_that_declares_no_length_is_refused_before_anything_is_timed(
    tmp_path: Path,
) -> None:
    run = a_packaged_run("synthetic-parser")
    run["trainer"] = {"kind": "open-ended-unit-trainer", "patience": 2}
    timer = Recording([(1.0, 1.0)], steps_per_unit=1)

    with pytest.raises(
        UndeclaredTrainingLengthError, match="open-ended-unit-trainer"
    ) as refused:
        priced(tmp_path, run, a_grid("trainer.patience", [2, 3]), timer)

    assert refused.value.trainer == "open-ended-unit-trainer"
    assert timer.timed == []


def test_a_parser_trainer_is_priced_at_the_epochs_that_cap_its_early_stopping() -> None:
    spec = PACKAGED_CODEC(PACKAGED_RUNS / "synthetic-parser.yaml")

    assert isinstance(spec.trainer, ParserTrainerSpec)
    assert training_length(spec) == TrainingLength(
        spec.trainer.epochs, TrainingUnit.EPOCHS
    )


def test_a_checked_build_is_not_timed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_checks, "CHECKS", True)
    timer = Recording([(1.0, 1.0)], steps_per_unit=1)

    with pytest.raises(RuntimeError, match="LOFI_CHECK=0"):
        priced(
            tmp_path,
            a_packaged_run("synthetic-parser"),
            {"grid": [], "seeds": list(SEEDS), "expected_points": len(SEEDS)},
            timer,
        )

    assert timer.timed == []


@pytest.mark.parametrize(
    ("warmup", "repeats", "message"),
    [(1, 1, "at least two timed runs"), (-1, 3, "warmup")],
    ids=("one-repeat", "negative-warmup"),
)
def test_a_price_needs_repeated_timed_runs_and_a_real_warmup(
    tmp_path: Path, warmup: int, repeats: int, message: str
) -> None:
    experiment = read_experiment(
        write_priced_experiment(
            tmp_path,
            a_packaged_run("synthetic-parser"),
            {"grid": [], "seeds": list(SEEDS), "expected_points": len(SEEDS)},
        )
    )
    timer = Recording([(1.0, 1.0)], steps_per_unit=1)

    with pytest.raises(ValueError, match=message):
        price_experiment(
            experiment, tmp_path, timer, device="cpu", warmup=warmup, repeats=repeats
        )

    assert timer.timed == []


def test_each_stack_is_built_under_the_run_seed_its_first_point_derives(
    tmp_path: Path,
) -> None:
    timer = Recording([(0.5, 0.6)], steps_per_unit=1)
    sweep = a_grid("optimizer.learning_rate", [0.001, 0.0005])

    priced(tmp_path, a_packaged_run("synthetic-parser"), sweep, timer, repeats=2)
    experiment = read_experiment(tmp_path / "experiments/pricing.yaml")
    points = expand_experiment(experiment, tmp_path)
    first = {str(point.config["optimizer"]): point for point in reversed(points)}

    assert sorted(point.root for _, point, *_ in timer.timed) == sorted(
        experiment.seed_policy.point(point.point_id, point.seed).root
        for point in first.values()
    )


def test_a_judgment_fit_experiment_is_refused_rather_than_priced(
    tmp_path: Path,
) -> None:
    experiment = read_experiment(
        write_experiment(tmp_path, a_judgment_experiment_document(), name="fits")
    )
    timer = Recording([(1.0, 1.0)], steps_per_unit=1)

    with pytest.raises(UnpriceableExperimentError, match="judgment-fit") as refused:
        price_experiment(experiment, tmp_path, timer, device="cpu", warmup=0, repeats=2)

    assert refused.value.experiment == "fits"
    assert timer.timed == []


def test_a_baseline_fit_experiment_is_refused_rather_than_priced(
    tmp_path: Path,
) -> None:
    experiment = baseline_experiment(tmp_path)
    timer = Recording([(1.0, 1.0)], steps_per_unit=1)

    with pytest.raises(UnpriceableExperimentError, match="baseline-fit") as refused:
        price_experiment(experiment, tmp_path, timer, device="cpu", warmup=0, repeats=2)

    assert refused.value.experiment == "baseline"
    assert timer.timed == []


def test_an_experiment_whose_configuration_trains_no_timed_step_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = read_experiment(
        write_priced_experiment(
            tmp_path,
            a_packaged_run("synthetic-parser"),
            a_grid("optimizer.learning_rate", [0.001]),
        )
    )

    def untimed(configuration: object) -> bool:
        del configuration
        return False

    monkeypatch.setattr(type(experiment.config), "timed", property(untimed))
    timer = Recording([(1.0, 1.0)], steps_per_unit=1)

    with pytest.raises(UnpriceableExperimentError, match="run document") as refused:
        price_experiment(experiment, tmp_path, timer, device="cpu", warmup=0, repeats=2)

    assert refused.value.kind == experiment.config.kind == "run"
    assert timer.timed == []
