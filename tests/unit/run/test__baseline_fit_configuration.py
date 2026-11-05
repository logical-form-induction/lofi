from typing import TYPE_CHECKING

from tests.unit.run.test_baselines import baseline_experiment

from lofi.run import (
    ExecutionContext,
    ExecutionEnvironment,
    ExperimentStore,
    expand_experiment,
    installed_packages,
)
from lofi.run._boundary.baselines import BASELINE_FIT_MANIFEST_NAME, BASELINE_ROLE
from lofi.run.baselines import BaselineExecutor
from lofi.run.operations import TRAINING_OPERATIONS

if TYPE_CHECKING:
    from pathlib import Path

    from lofi._types import JsonValue


def test_a_baseline_fit_executes_in_process_and_is_priced_by_no_step_timer(
    tmp_path: Path,
) -> None:
    configuration = baseline_experiment(tmp_path).config

    assert configuration.kind == "baseline-fit"
    assert configuration.manifest_name == BASELINE_FIT_MANIFEST_NAME
    assert configuration.in_process
    assert not configuration.timed
    assert not configuration.enumerates
    assert configuration.roles({}) == frozenset({BASELINE_ROLE})


def test_the_executor_checks_every_fit_and_executes_none(tmp_path: Path) -> None:
    experiment = baseline_experiment(tmp_path)
    context = ExecutionContext(
        ExperimentStore(tmp_path / "store", experiment),
        ExecutionEnvironment(installed_packages(), "0" * 40, "cpu"),
        experiment.failure,
        TRAINING_OPERATIONS,
    )

    executor = experiment.config.executor(
        context, expand_experiment(experiment, tmp_path)
    )

    assert isinstance(executor, BaselineExecutor)
    assert not (tmp_path / "store").exists()


def test_a_baseline_fit_freezes_no_identity_and_reports_its_rows(
    tmp_path: Path,
) -> None:
    experiment = baseline_experiment(tmp_path)
    points = expand_experiment(experiment, tmp_path)
    rows: tuple[dict[str, JsonValue], ...] = ({"metric": "f1", "value": 0.5},)

    assert experiment.config.identities(points) == {}
    assert (
        experiment.config.result_rows(
            rows, points, ExperimentStore(tmp_path / "store", experiment)
        )
        == rows
    )
