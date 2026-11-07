import didactic.api as dx
import pytest

from lofi.spec.sinks.mlflow import MlflowSinkSettings, MlflowSinkSpec


def test_the_spec_validates_every_supplied_name() -> None:
    spec = MlflowSinkSpec(
        experiment_name="grammar",
        run_name="trial",
        tracking_uri="https://tracker.example",
    )

    assert spec.experiment_name == "grammar"
    assert spec.run_name == "trial"
    assert spec.tracking_uri == "https://tracker.example"

    with pytest.raises(dx.ValidationError, match="is empty"):
        MlflowSinkSpec(experiment_name="")

    with pytest.raises(dx.ValidationError, match="is empty"):
        MlflowSinkSpec(experiment_name="grammar", run_name="")

    with pytest.raises(dx.ValidationError, match="is empty"):
        MlflowSinkSpec(experiment_name="grammar", tracking_uri="")


def test_the_spec_resolves_to_its_settings_in_the_sink_family() -> None:
    spec = MlflowSinkSpec(experiment_name="grammar", tracking_uri="sqlite:///runs.db")

    assert spec.family() == "sink"
    assert spec.settings() == MlflowSinkSettings(
        experiment_name="grammar", run_name=None, tracking_uri="sqlite:///runs.db"
    )
