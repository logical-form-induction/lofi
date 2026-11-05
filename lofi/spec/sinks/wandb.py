"""W&B sink declaration."""

from typing import Literal, NamedTuple

from lofi._checks import ensure_name
from lofi.spec.components.run import SinkSpec
from lofi.spec.sinks import _validates

__all__ = ["WandbSinkSettings", "WandbSinkSpec"]


class WandbSinkSettings(NamedTuple):
    """The settings a W&B sink is built from.

    Attributes
    ----------
    project : str
        The project the run is filed under.
    entity : str | None
        The team or user owning the project, or None for the client's default.
    run_name : str | None
        The run name, or None to name the run after its point.
    """

    project: str
    entity: str | None
    run_name: str | None


class WandbSinkSpec(SinkSpec):
    """Declare a W&B metric sink."""

    kind: Literal["wandb"] = "wandb"
    project: str
    entity: str | None = None
    run_name: str | None = None

    @_validates("project", "entity", "run_name")
    @classmethod
    def _names_are_present(cls, value: str | None) -> str | None:
        """Validate a supplied sink name."""
        if value is None:
            return None

        return ensure_name("W&B sink name", value)

    def settings(self) -> WandbSinkSettings:
        """Return the settings the plugin builds this sink from."""
        return WandbSinkSettings(
            project=self.project,
            entity=self.entity,
            run_name=self.run_name,
        )
