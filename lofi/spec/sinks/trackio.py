"""Trackio sink declaration."""

from typing import Literal, NamedTuple, Self

from lofi._checks import ensure_name
from lofi.spec.components.run import SinkSpec
from lofi.spec.sinks import _model_validator, _validates

__all__ = ["TrackioSinkSettings", "TrackioSinkSpec"]


class TrackioSinkSettings(NamedTuple):
    """The settings a Trackio sink is built from.

    Attributes
    ----------
    project : str
        The project the run is filed under.
    run_name : str | None
        The run name, or None to name the run after its point.
    space_id : str | None
        The hosted Space the run is written to, or None.
    server_url : str | None
        The self-hosted server the run is written to, or None.
    """

    project: str
    run_name: str | None
    space_id: str | None
    server_url: str | None


class TrackioSinkSpec(SinkSpec):
    """Declare a Trackio metric sink."""

    kind: Literal["trackio"] = "trackio"
    project: str
    run_name: str | None = None
    space_id: str | None = None
    server_url: str | None = None

    @_validates("project", "run_name", "space_id", "server_url")
    @classmethod
    def _names_are_present(cls, value: str | None) -> str | None:
        """Validate a supplied sink name."""
        if value is None:
            return None

        return ensure_name("Trackio sink name", value)

    @_model_validator()
    def _remote_target_is_unambiguous(self) -> Self:
        """Reject simultaneous hosted and self-hosted targets."""
        if self.space_id is not None and self.server_url is not None:
            msg = "a Trackio sink accepts either space_id or server_url, not both"

            raise ValueError(msg)

        return self

    def settings(self) -> TrackioSinkSettings:
        """Return the settings the plugin builds this sink from."""
        return TrackioSinkSettings(
            project=self.project,
            run_name=self.run_name,
            space_id=self.space_id,
            server_url=self.server_url,
        )
