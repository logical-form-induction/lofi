"""MLflow sink declaration."""

from typing import Literal, NamedTuple

from lofi._checks import ensure_name
from lofi.spec.components.run import SinkSpec
from lofi.spec.sinks import _validates

__all__ = ["MlflowSinkSettings", "MlflowSinkSpec"]


class MlflowSinkSettings(NamedTuple):
    """The settings an MLflow sink is built from.

    Attributes
    ----------
    experiment_name : str
        The experiment the run is filed under.
    run_name : str | None
        The run name, or None to name the run after its point.
    tracking_uri : str | None
        The tracking store, or None for the client's configured store.
    """

    experiment_name: str
    run_name: str | None
    tracking_uri: str | None


class MlflowSinkSpec(SinkSpec):
    """Declare an MLflow metric sink."""

    kind: Literal["mlflow"] = "mlflow"
    experiment_name: str
    run_name: str | None = None
    tracking_uri: str | None = None

    @_validates("experiment_name", "run_name", "tracking_uri")
    @classmethod
    def _names_are_present(cls, value: str | None) -> str | None:
        """Validate a supplied sink name."""
        if value is None:
            return None

        return ensure_name("MLflow sink name", value)

    def settings(self) -> MlflowSinkSettings:
        """Return the settings the plugin builds this sink from."""
        return MlflowSinkSettings(
            experiment_name=self.experiment_name,
            run_name=self.run_name,
            tracking_uri=self.tracking_uri,
        )
