import didactic.api as dx
import pytest

from lofi.spec.sinks.wandb import WandbSinkSettings, WandbSinkSpec


def test_the_spec_validates_every_supplied_name() -> None:
    spec = WandbSinkSpec(project="grammar", entity="lab", run_name="trial")

    assert spec.model_dump() | {} == {
        "kind": "wandb",
        "project": "grammar",
        "entity": "lab",
        "run_name": "trial",
        "declared_output": None,
    }

    with pytest.raises(dx.ValidationError, match="is empty"):
        WandbSinkSpec(project="")

    with pytest.raises(dx.ValidationError, match="is empty"):
        WandbSinkSpec(project="grammar", entity="")

    with pytest.raises(dx.ValidationError, match="is empty"):
        WandbSinkSpec(project="grammar", run_name="")


def test_the_spec_resolves_to_its_settings_in_the_sink_family() -> None:
    spec = WandbSinkSpec(project="grammar", entity="lab")

    assert spec.family() == "sink"
    assert spec.settings() == WandbSinkSettings(
        project="grammar", entity="lab", run_name=None
    )
