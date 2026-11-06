"""Exercise the known-grammar lexical-induction documents end to end."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from lofi.data import BatchShape
from lofi.run import (
    BaselineExecutor,
    ExecutionEnvironment,
    ExperimentSeed,
    ExperimentStore,
    GridCell,
    GridTiming,
    HardwareDescriptor,
    ManifestStatus,
    PointExecutor,
    RegisteredBaselines,
    RunStore,
    SerialLauncher,
    StackTiming,
    expand_experiment,
    price_experiment,
    read_experiment,
    read_rows,
)
from lofi.run.operations import TRAINING_OPERATIONS
from lofi.spec import RunSpecCodec

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi.run import PointSeed
    from lofi.spec import RunSpec


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = Path(__file__).resolve().parent
DIAGNOSTIC_ROWS = frozenset(
    {
        ("lexical", "supertag-accuracy"),
        ("structural", "dependency:labeled-directed:zero:micro-f-score"),
        ("types", "lexical-type-f1"),
        ("types", "primitive-one-to-one-accuracy"),
        ("types", "primitive-random-relabeling-floor"),
        ("types", "constructor-one-to-one-accuracy"),
        ("types", "constructor-random-relabeling-floor"),
        ("lexical", "supertag-accuracy-frequency-unseen"),
        ("lexical", "supertag-accuracy-frequency-rare"),
        ("lexical", "supertag-accuracy-frequency-frequent"),
        ("lexical", "supertag-accuracy-ambiguity-unseen"),
        ("lexical", "supertag-accuracy-ambiguity-unambiguous"),
        ("lexical", "supertag-accuracy-ambiguity-ambiguous"),
    }
)
BASELINE_METRICS = (
    "supertagging-length-10",
    "dependencies-length-10",
    "supertagging-length-20",
    "dependencies-length-20",
    "supertagging-all-lengths",
    "dependencies-all-lengths",
)
REDUCED_TREATMENTS = (
    "induced",
    "labeled-DIORA",
    "tag-dictionary",
    "supervised",
    "Prange-style",
)


def _check(condition: object, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _mapping(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        msg = f"{field} is not a mapping: {value!r}"
        raise TypeError(msg)
    return cast("dict[str, object]", value)


def _treatment(config: dict[str, object]) -> str:
    """Return the learner treatment encoded by one resolved S1 point."""
    model = _mapping(config["model"], "model")
    parser = _mapping(model["parser"], "model.parser")
    attend = _mapping(parser["attend"], "model.parser.attend")
    decoders = _mapping(model["decoders"], "model.decoders")
    identity = _mapping(decoders["identity"], "model.decoders.identity")
    data = _mapping(config["data"], "data")
    datasets = _mapping(data["datasets"], "data.datasets")
    train = _mapping(datasets["train"], "data.datasets.train")
    policies = _mapping(train["constraint_policy"], "constraint_policy")
    syntactic = _mapping(policies["syntactic"], "constraint_policy.syntactic")

    if identity["kind"] == "linear-rnn":
        return "Prange-style"
    if attend["kind"] == "attend":
        return "labeled-DIORA"
    if syntactic["coverage"] == 0.0:
        return "induced"
    if syntactic["ambiguity"] == "union":
        return "tag-dictionary"
    return "supervised"


class _RecordingTimer:
    """Price exact stacks without running accelerator kernels."""

    def __init__(self) -> None:
        self.timed: list[RunSpec] = []
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
        del point, device
        self.timed.append(spec)
        return StackTiming(
            steps_per_unit=1,
            positions=tuple(range(warmup, warmup + repeats)),
            seconds=tuple(0.01 for _ in range(repeats)),
            peak_allocated_bytes=1,
            shape=BatchShape(4.0, 1.0),
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
            tuple(
                GridCell(length, batch, tuple(0.01 for _ in range(repeats)))
                for length, batch in stacks
            )
        )

    def hardware(self, device: str, /) -> HardwareDescriptor:
        return HardwareDescriptor(
            device_name=device,
            driver=None,
            cuda=None,
            torch="test",
            host_cpu="test",
        )


@pytest.mark.integration
def test_production_documents_expand_every_declared_cell() -> None:
    """The production documents retain their exact crossed designs."""
    ccgbank = read_experiment(EXPERIMENTS / "ccgbank.yaml")
    synthetic = read_experiment(EXPERIMENTS / "synthetic.yaml")
    baseline = read_experiment(EXPERIMENTS / "bisk-hockenmaier.yaml")
    reduced = read_experiment(EXPERIMENTS / "reduced-freeze.yaml")
    reduced_baseline = read_experiment(EXPERIMENTS / "bisk-hockenmaier-reduced.yaml")

    ccgbank_points = expand_experiment(ccgbank, ROOT)
    synthetic_points = expand_experiment(synthetic, ROOT)
    baseline_points = expand_experiment(baseline, ROOT)
    reduced_points = expand_experiment(reduced, ROOT)
    reduced_baseline_points = expand_experiment(reduced_baseline, ROOT)

    _check(len(ccgbank_points) == 120, "the CCGbank grid does not contain 120 points")
    _check(
        len(synthetic_points) == 144,
        "the synthetic grid does not contain 144 points",
    )
    _check(len(baseline_points) == 4, "the published baseline omits a seed")
    _check(len(reduced_points) == 5, "the reduced neural grid omits a treatment")
    _check(
        len(reduced_baseline_points) == 1,
        "the reduced published baseline is not one point",
    )
    _check(
        Counter(
            _treatment(cast("dict[str, object]", point.config))
            for point in ccgbank_points
        )
        == {
            "induced": 24,
            "labeled-DIORA": 24,
            "tag-dictionary": 24,
            "supervised": 24,
            "Prange-style": 24,
        },
        "the CCGbank grid does not contain every declared learner treatment",
    )
    _check(
        Counter(
            _treatment(cast("dict[str, object]", point.config))
            for point in reduced_points
        )
        == {
            "induced": 1,
            "labeled-DIORA": 1,
            "tag-dictionary": 1,
            "supervised": 1,
            "Prange-style": 1,
        },
        "the reduced grid does not contain every declared learner treatment",
    )
    _check(
        {(row.evaluator, row.metric) for row in ccgbank.rows} == DIAGNOSTIC_ROWS,
        "the CCGbank document omits a required diagnostic row",
    )
    _check(
        {(row.evaluator, row.metric) for row in synthetic.rows} == DIAGNOSTIC_ROWS,
        "the synthetic document omits a required diagnostic row",
    )

    synthetic_cells: Counter[tuple[int, str, str, int, int]] = Counter()
    for point in synthetic_points:
        data = _mapping(point.config["data"], "data")
        datasets = _mapping(data["datasets"], "data.datasets")
        train = _mapping(datasets["train"], "data.datasets.train")
        dataset = _mapping(train["dataset"], "data.datasets.train.dataset")
        length = _mapping(dataset["length"], "data.datasets.train.dataset.length")
        model = _mapping(point.config["model"], "model")
        encoders = _mapping(model["encoders"], "model.encoders")
        terms = _mapping(encoders["terms"], "model.encoders.terms")
        capacity = _mapping(model["capacity"], "model.capacity")
        synthetic_cells[
            (
                cast("int", dataset["size"]),
                cast("str", length["kind"]),
                cast("str", terms["kind"]),
                cast("int", capacity["target"]),
                point.seed,
            )
        ] += 1
    _check(
        len(synthetic_cells) == 144 and set(synthetic_cells.values()) == {1},
        "a synthetic factorial cell was omitted or duplicated",
    )


@pytest.mark.integration
def test_production_run_documents_price_every_distinct_stack() -> None:
    """Pricing reaches every CCGbank and synthetic stack exactly once."""
    ccgbank = read_experiment(EXPERIMENTS / "ccgbank.yaml")
    synthetic = read_experiment(EXPERIMENTS / "synthetic.yaml")
    ccgbank_timer = _RecordingTimer()
    synthetic_timer = _RecordingTimer()

    ccgbank_price = price_experiment(
        ccgbank,
        ROOT,
        ccgbank_timer,
        device="cpu",
        warmup=0,
        repeats=2,
    )
    synthetic_price = price_experiment(
        synthetic,
        ROOT,
        synthetic_timer,
        device="cpu",
        warmup=0,
        repeats=2,
    )

    _check(len(ccgbank_price.stacks) == 30, "CCGbank did not price 30 stacks")
    _check(len(ccgbank_timer.timed) == 30, "a CCGbank stack was not timed once")
    _check(len(synthetic_price.stacks) == 36, "synthetic did not price 36 stacks")
    _check(len(synthetic_timer.timed) == 36, "a synthetic stack was not timed once")
    _check(
        synthetic_timer.grids == [((4, 8, 16), (4, 8, 16))],
        "the synthetic length-by-batch timing grid changed",
    )
    _check(
        sum(len(stack.points) for stack in ccgbank_price.stacks) == 120,
        "the CCGbank price omitted a point",
    )
    _check(
        sum(len(stack.points) for stack in synthetic_price.stacks) == 144,
        "the synthetic price omitted a point",
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    ("point_index", "expected_treatment"),
    tuple(enumerate(REDUCED_TREATMENTS)),
    ids=REDUCED_TREATMENTS,
)
def test_reduced_neural_freeze_trains_and_ingests_every_declared_row(
    tmp_path: Path,
    point_index: int,
    expected_treatment: str,
) -> None:
    """One reduced neural treatment closes with all declared evidence."""
    experiment = read_experiment(EXPERIMENTS / "reduced-freeze.yaml")
    points = expand_experiment(experiment, ROOT)
    _check(len(points) == 5, "the reduced freeze does not declare five treatments")
    point = points[point_index]
    _check(
        _treatment(cast("dict[str, object]", point.config)) == expected_treatment,
        f"reduced point {point_index} is not the {expected_treatment} treatment",
    )
    store = RunStore(
        tmp_path / "runs",
        freeze_id=None,
        seed=ExperimentSeed(experiment.seed_policy.root),
    )
    executor = PointExecutor(
        store,
        codec=RunSpecCodec(),
        operations=TRAINING_OPERATIONS,
        infrastructure_attempts=1,
        failure_policy=experiment.failure,
    )
    handles = SerialLauncher(store, execute=executor).submit((point,))

    _check(len(handles) == 1, "the launcher omitted the reduced neural treatment")
    _check(
        store.is_completed(point),
        f"the reduced {expected_treatment} treatment did not complete",
    )
    recorded = read_rows(store.directory(point))
    if recorded is None:
        msg = f"completed point {point.point_id!r} has no evaluator rows"
        raise AssertionError(msg)
    present = {
        (cast("str", row["task"]), cast("str", row["metric"])) for row in recorded.rows
    }
    _check(
        present >= DIAGNOSTIC_ROWS,
        f"completed point {point.point_id!r} omitted declared rows",
    )


@pytest.mark.integration
def test_reduced_bisk_hockenmaier_fit_records_fixture_measurements(
    tmp_path: Path,
) -> None:
    """The published baseline boundary fits and scores the repository fixture."""
    experiment = read_experiment(EXPERIMENTS / "bisk-hockenmaier-reduced.yaml")
    [point] = expand_experiment(experiment, ROOT)
    store = ExperimentStore(tmp_path / "runs", experiment)
    executor = BaselineExecutor(
        store,
        operation=RegisteredBaselines(),
        environment=ExecutionEnvironment({"lofi": "test"}, "test", "cpu"),
        infrastructure_attempts=1,
    )

    outcome = executor(point)
    recorded = read_rows(store.directory(point))

    _check(
        (outcome.status, outcome.cached) == (ManifestStatus.COMPLETED, False),
        "the reduced published baseline did not complete",
    )
    if recorded is None:
        msg = "the reduced published baseline emitted no rows"
        raise AssertionError(msg)
    _check(
        tuple(row["metric"] for row in recorded.rows) == BASELINE_METRICS,
        "the reduced published baseline omitted a declared metric",
    )
    _check(
        tuple(row["value"] for row in recorded.rows) == (1.0, 0.0) * 3,
        "the reduced published baseline changed its fixture-derived measurements",
    )
    _check(
        all(row["n"] == 1 for row in recorded.rows),
        "the reduced published baseline did not score its one fixture sentence",
    )
