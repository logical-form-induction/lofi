from typing import TYPE_CHECKING

import pytest
from tests.unit.run.conftest import a_packaged_run, a_point, write_priced_experiment

from lofi.run import (
    MANIFEST_NAME,
    ExecutionContext,
    ExecutionEnvironment,
    ExperimentStore,
    RunStore,
    expand_experiment,
    installed_packages,
    read_experiment,
)
from lofi.run import _run_configuration as run_configuration
from lofi.run._run_configuration import RunConfiguration
from lofi.run.operations import TRAINING_OPERATIONS

if TYPE_CHECKING:
    from pathlib import Path

    from lofi._types import JsonValue
    from lofi.run import Experiment, FailurePolicySpec, TrainingOperations
    from lofi.spec import RunSpecCodec

SWEEP: dict[str, JsonValue] = {
    "grid": [{"path": "optimizer.learning_rate", "values": [0.001, 0.0005]}],
    "seeds": [0],
    "expected_points": 2,
}


def an_experiment(root: Path) -> Experiment:
    return read_experiment(
        write_priced_experiment(root, a_packaged_run("synthetic-parser"), SWEEP)
    )


def test_a_run_trains_a_timed_step_in_tasks_as_well_as_in_process(
    tmp_path: Path,
) -> None:
    configuration = an_experiment(tmp_path).config

    assert configuration.kind == "run"
    assert configuration.manifest_name == MANIFEST_NAME
    assert configuration.enumerates
    assert configuration.timed
    assert not configuration.in_process


def test_the_executor_trains_each_point_in_the_runs_of_the_experiments_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = an_experiment(tmp_path)
    store = ExperimentStore(tmp_path / "store", experiment)
    environment = ExecutionEnvironment(installed_packages(), "0" * 40, "cpu")
    built: list[tuple[RunStore, int, FailurePolicySpec, ExecutionEnvironment]] = []

    class Built:
        def __init__(
            self,
            runs: RunStore,
            *,
            codec: RunSpecCodec,
            operations: TrainingOperations,
            infrastructure_attempts: int,
            failure_policy: FailurePolicySpec,
            environment: ExecutionEnvironment,
        ) -> None:
            del codec, operations
            built.append((runs, infrastructure_attempts, failure_policy, environment))

    monkeypatch.setattr(run_configuration, "PointExecutor", Built)
    context = ExecutionContext(
        store, environment, experiment.failure, TRAINING_OPERATIONS
    )

    executor = experiment.config.executor(
        context, expand_experiment(experiment, tmp_path)
    )
    ((runs, attempts, policy, recorded),) = built

    assert isinstance(executor, Built)
    assert runs.root == store.runs.root
    assert runs.freeze_id == experiment.freeze_id
    assert attempts == experiment.failure.infrastructure_attempts
    assert policy == experiment.failure
    assert recorded is environment


def test_identities_are_what_training_plans_for_every_point(tmp_path: Path) -> None:
    experiment = an_experiment(tmp_path)
    points = expand_experiment(experiment, tmp_path)

    identities = experiment.config.identities(points)

    assert set(identities) == {point.point_id for point in points}
    assert all(
        set(identity) >= {"grammar_fingerprint", "codec_fingerprint"}
        for identity in identities.values()
        if isinstance(identity, dict)
    )


def test_a_point_training_cannot_plan_is_refused_naming_it(tmp_path: Path) -> None:
    configuration = RunConfiguration("run", tmp_path / "run.json")
    point = a_point(1)

    with pytest.raises(ValueError, match="cannot be planned for training") as refused:
        configuration.identities((point,))

    assert point.point_id in str(refused.value)


def test_the_rows_a_run_reports_are_the_rows_its_points_recorded(
    tmp_path: Path,
) -> None:
    experiment = an_experiment(tmp_path)
    points = expand_experiment(experiment, tmp_path)
    rows: tuple[dict[str, JsonValue], ...] = (
        {"metric": "unlabeled-bracketing-f1", "value": 0.5},
    )

    reported = experiment.config.result_rows(
        rows, points, ExperimentStore(tmp_path, experiment)
    )

    assert reported == rows
