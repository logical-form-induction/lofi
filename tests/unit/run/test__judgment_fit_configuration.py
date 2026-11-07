from typing import TYPE_CHECKING

from tests.unit.run.test__experiment import (
    a_judgment_experiment_document,
    write_experiment,
)

from lofi.run import (
    FIT_MANIFEST_NAME,
    ExperimentStore,
    expand_experiment,
    read_experiment,
)
from lofi.run._experiment import FIT_METRICS, FIT_ROLE
from lofi.run.fits import HELD_OUT_PREDICTIVE_METRIC

if TYPE_CHECKING:
    from pathlib import Path

    from lofi._types import JsonValue
    from lofi.run import Experiment


def a_fit_experiment(root: Path) -> Experiment:
    return read_experiment(
        write_experiment(root, a_judgment_experiment_document(), name="fits")
    )


def test_a_fit_executes_in_process_and_is_priced_by_no_step_timer(
    tmp_path: Path,
) -> None:
    configuration = a_fit_experiment(tmp_path).config

    assert configuration.kind == "judgment-fit"
    assert configuration.manifest_name == FIT_MANIFEST_NAME
    assert configuration.in_process
    assert not configuration.timed
    assert not configuration.enumerates
    assert configuration.evaluators((tmp_path,), (), 0) == {
        FIT_ROLE: frozenset(FIT_METRICS)
    }
    assert configuration.roles({}) == frozenset({FIT_ROLE})


def test_a_fit_freezes_no_identity(tmp_path: Path) -> None:
    experiment = a_fit_experiment(tmp_path)

    assert experiment.config.identities(expand_experiment(experiment, tmp_path)) == {}


def test_a_fits_predictive_density_is_reported_through_its_comparisons(
    tmp_path: Path,
) -> None:
    experiment = a_fit_experiment(tmp_path)
    points = expand_experiment(experiment, tmp_path)
    kept: dict[str, JsonValue] = {"metric": "max-rhat", "value": 1.0}
    density: dict[str, JsonValue] = {
        "metric": HELD_OUT_PREDICTIVE_METRIC,
        "value": -3.0,
    }

    reported = experiment.config.result_rows(
        (kept, density), points, ExperimentStore(tmp_path / "store", experiment)
    )

    assert reported == (kept,)
