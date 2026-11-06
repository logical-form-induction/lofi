"""Exercise the source-grammar comparison documents through the run boundary."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, cast

import pytest

from lofi.eval import ReductionContext, reduce_experiment_rows
from lofi.run import (
    ExperimentSeed,
    PointExecutor,
    RunStore,
    SerialLauncher,
    comparison_points,
    expand_experiment,
    metrics_row,
    read_experiment,
    read_metrics,
    read_rows,
)
from lofi.run.operations import TRAINING_OPERATIONS
from lofi.spec import RunSpecCodec

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lofi._types import JsonValue
    from lofi.data.synthetic import SyntheticDataset
    from lofi.nn.policy import DevicePolicy
    from lofi.run import (
        HeldOutScoring,
        MetricSink,
        PointRecorder,
        TrainingBlueprint,
    )
    from lofi.run._ports import TrainingOperations
    from lofi.train import RestorableRandomStreams


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = Path(__file__).resolve().parent
GRAMMARS = frozenset(
    {
        "slashes-application-k2-depth1",
        "arrow-composition-k4-depth2",
        "slashes-raised-k8-depth3",
        "arrow-generalized-composition-k16-depth1",
        "arrow-application-k8-depth2",
        "slashes-composition-k16-depth3",
        "arrow-raised-k2-depth1",
        "slashes-generalized-composition-k4-depth2",
    }
)


class _ObservedCorpus(NamedTuple):
    """Hold one source declaration, its realized strings, and emitted labels."""

    revision: str
    examples: tuple[str, ...]
    labels: tuple[str, ...]


class _RecordingTraining:
    """Record realized source data before delegating to production training."""

    def __init__(self, delegate: TrainingOperations) -> None:
        self._delegate = delegate
        self.corpora: dict[str, set[_ObservedCorpus]] = {}

    def execute(
        self,
        blueprint: TrainingBlueprint,
        directory: Path,
        streams: RestorableRandomStreams,
        policy: DevicePolicy,
        /,
        *,
        scoring: HeldOutScoring | None,
        sinks: Sequence[MetricSink],
        recorder: PointRecorder,
    ) -> None:
        comparison = blueprint.run_spec.comparison
        if comparison is None:
            msg = "a source-comparison point has no comparison identity"
            raise TypeError(msg)
        dataset = cast("SyntheticDataset", blueprint.lower_dataset("train"))
        corpus = dataset.corpus
        observed = _ObservedCorpus(
            blueprint.corpus_revision,
            tuple(example.item.text or "" for example in corpus.examples),
            tuple(
                f"{label.item_id}:{label.true_label}:{label.observed_label}"
                for label in corpus.labels
            ),
        )
        self.corpora.setdefault(comparison.condition, set()).add(observed)
        self._delegate.execute(
            blueprint,
            directory,
            streams,
            policy,
            scoring=scoring,
            sinks=sinks,
            recorder=recorder,
        )


def _rows(
    rows: Sequence[Mapping[str, JsonValue]], reducer: str
) -> tuple[Mapping[str, JsonValue], ...]:
    return tuple(row for row in rows if row["reducer"] == reducer)


def _check(condition: object, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_source_comparison_documents_load_the_complete_factor_grid() -> None:
    """The production and reduced documents retain their declared designs."""
    production = read_experiment(EXPERIMENTS / "source-complexity.yaml")
    reduced = read_experiment(EXPERIMENTS / "reduced-freeze.yaml")
    reduced_points = expand_experiment(reduced, ROOT)

    _check(
        production.sweep.expected_points == 4080,
        "the production sweep does not declare every crossed point",
    )
    _check(
        reduced.sweep.expected_points == 10,
        "the reduced freeze does not declare its ten fitted arms",
    )
    _check(
        production.factors
        == {
            "source-grammar": "data.datasets.train.dataset",
            "candidate-grammar": "model.language",
            "data-size": "data.datasets.train.dataset.size",
        },
        "the production experiment does not bind all three declared factors",
    )
    _check(len(reduced_points) == 10, "the reduced document does not expand")
    _check(
        set(production.preregistration.grid["source-grammar"]) == GRAMMARS,
        "the production experiment omits a source grammar",
    )
    _check(
        set(production.preregistration.grid["candidate-grammar"]) == GRAMMARS,
        "the production experiment omits a candidate grammar",
    )
    _check(
        production.preregistration.grid["data-size"] == (16, 64, 256),
        "the production experiment omits a data-size level",
    )


@pytest.mark.integration
def test_reduced_freeze_trains_and_reduces_real_evaluator_rows(
    tmp_path: Path,
) -> None:
    """A reduced run emits verdict and source-by-winner rows from real fits."""
    experiment = read_experiment(EXPERIMENTS / "reduced-freeze.yaml")
    points = expand_experiment(experiment, ROOT)
    conditions = Counter(
        cast("dict[str, JsonValue]", point.config["comparison"])["condition"]
        for point in points
    )
    twins = Counter(
        cast("dict[str, JsonValue]", point.config["comparison"])["twin"]
        for point in points
    )

    _check(len(points) == 10, "the reduced freeze did not expand ten points")
    _check(
        conditions
        == {
            "primary": 8,
            "reversed-ground-truth": 1,
            "lexicon-scrambled": 1,
        },
        "the reduced freeze did not retain both validity controls",
    )
    _check(
        twins
        == {
            "observed": 4,
            "twin-001": 2,
            "twin-002": 2,
            "twin-003": 2,
        },
        "the reduced freeze did not expand only the primary candidates",
    )

    operations = _RecordingTraining(TRAINING_OPERATIONS)
    store = RunStore(
        tmp_path / "runs",
        freeze_id=None,
        seed=ExperimentSeed(experiment.seed_policy.root),
    )
    executor = PointExecutor(
        store,
        codec=RunSpecCodec(),
        operations=operations,
        infrastructure_attempts=1,
        failure_policy=experiment.failure,
    )
    handles = SerialLauncher(store, execute=executor).submit(points)

    _check(len(handles) == len(points), "the launcher omitted a reduced point")
    _check(
        all(store.is_completed(point) for point in points),
        "a reduced point did not complete",
    )
    primary = operations.corpora["primary"]
    [reversed_corpus] = operations.corpora["reversed-ground-truth"]
    [scrambled_corpus] = operations.corpora["lexicon-scrambled"]
    _check(len(primary) == 1, "primary candidates did not share one source corpus")
    _check(
        reversed_corpus not in primary,
        "the reversed control reproduced the primary corpus and labels",
    )
    _check(
        scrambled_corpus not in primary,
        "the scrambled control reproduced the primary corpus and labels",
    )
    _check(
        reversed_corpus != scrambled_corpus,
        "the two validity controls produced the same corpus and labels",
    )

    result_rows: list[Mapping[str, JsonValue]] = []
    manifests = []
    measured: list[Mapping[str, JsonValue]] = []
    for point in points:
        directory = store.directory(point)
        recorded = read_rows(directory)
        record = read_metrics(directory)
        if recorded is None or record is None:
            msg = f"completed point {point.point_id!r} has no results"
            raise AssertionError(msg)
        result_rows.extend(recorded.rows)
        measured.append(
            {
                "point_id": point.point_id,
                **metrics_row(record),
            }
        )
        manifests.append(
            experiment.config.manifest_point((directory / "manifest.json").read_bytes())
        )

    reduction = reduce_experiment_rows(
        result_rows,
        experiment.reducers,
        ReductionContext(
            EXPERIMENTS,
            seed=experiment.seed_policy.root,
            points=comparison_points(manifests, measured),
            preregistration=experiment.preregistration,
            alpha=experiment.evaluation.alpha,
            correction=experiment.evaluation.correction,
            dependence=experiment.evaluation.dependence,
        ),
    )
    verdicts = _rows(reduction.rows, "source-comparison-verdict")
    matrix = _rows(reduction.rows, "source-by-winner")

    _check(len(verdicts) == 1, "the reduced freeze emitted no unique verdict")
    _check(
        verdicts[0]["source"] == "slashes-application-k2-depth1",
        "the reduced verdict names the wrong source grammar",
    )
    _check(
        verdicts[0]["winner"]
        in {
            "slashes-application-k2-depth1",
            "arrow-composition-k4-depth2",
        },
        "the reduced verdict names an undeclared winner",
    )
    _check(len(matrix) == 1, "the reduced freeze emitted no unique matrix cell")
    _check(
        matrix[0]["source"] == "slashes-application-k2-depth1",
        "the source-by-winner matrix names the wrong source",
    )
    _check(
        matrix[0]["winner"] == verdicts[0]["winner"],
        "the matrix and verdict disagree on the winner",
    )
    _check(matrix[0]["value"] == 1, "the matrix does not count its one source")
