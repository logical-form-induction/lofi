import didactic.api as dx
import pytest

from lofi.spec.sinks.trackio import TrackioSinkSettings, TrackioSinkSpec


def test_the_spec_validates_names_and_one_remote_target() -> None:
    spec = TrackioSinkSpec(
        project="grammar",
        run_name="trial",
        space_id="lab/grammar",
    )

    assert spec.project == "grammar"
    assert spec.run_name == "trial"
    assert spec.space_id == "lab/grammar"
    assert spec.server_url is None

    with pytest.raises(dx.ValidationError, match="is empty"):
        TrackioSinkSpec(project="")

    with pytest.raises(dx.ValidationError, match="is empty"):
        TrackioSinkSpec(project="grammar", run_name="")

    with pytest.raises(dx.ValidationError, match="is empty"):
        TrackioSinkSpec(project="grammar", space_id="")

    with pytest.raises(dx.ValidationError, match="is empty"):
        TrackioSinkSpec(project="grammar", server_url="")

    with pytest.raises(dx.ValidationError, match="either space_id or server_url"):
        TrackioSinkSpec(
            project="grammar",
            space_id="lab/grammar",
            server_url="https://tracker.example",
        )


def test_the_spec_resolves_to_its_settings_in_the_sink_family() -> None:
    spec = TrackioSinkSpec(project="grammar", space_id="lab/grammar")

    assert spec.family() == "sink"
    assert spec.settings() == TrackioSinkSettings(
        project="grammar", run_name=None, space_id="lab/grammar", server_url=None
    )
