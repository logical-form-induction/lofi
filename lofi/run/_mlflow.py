"""MLflow metric sink."""

import importlib
import time
from typing import TYPE_CHECKING, Final, Protocol, cast

from lofi.registry import ComponentPlugin
from lofi.run._sinks import MissingTelemetryBackendError
from lofi.spec.sinks.mlflow import MlflowSinkSettings, MlflowSinkSpec

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lofi._types import JsonValue
    from lofi.run._sinks import SinkContext

__all__ = ["PLUGIN", "MlflowSink"]


class _MlflowExperiment(Protocol):
    """An MLflow experiment record."""

    @property
    def experiment_id(self) -> str:
        """Return the experiment's identifier."""
        ...


class _MlflowRunInfo(Protocol):
    """The identifying part of an MLflow run record."""

    @property
    def run_id(self) -> str:
        """Return the run's identifier."""
        ...


class _MlflowRun(Protocol):
    """An MLflow run record."""

    @property
    def info(self) -> _MlflowRunInfo:
        """Return the run's identifying record."""
        ...


class _MlflowMetric(Protocol):
    """One logged MLflow metric entry."""


class _MlflowMetricFactory[E](Protocol):
    """The MLflow metric-entry constructor."""

    def __call__(self, key: str, value: float, timestamp: int, step: int) -> E:
        """Return one metric entry."""
        ...


class _MlflowClient[E](Protocol):
    """The MLflow tracking-client operations the sink uses."""

    def get_experiment_by_name(self, name: str) -> _MlflowExperiment | None:
        """Return the named experiment, or None when the store holds none."""
        ...

    def create_experiment(self, name: str) -> str:
        """Create an experiment and return its identifier."""
        ...

    def create_run(self, experiment_id: str, *, run_name: str | None) -> _MlflowRun:
        """Create and return a run under an experiment."""
        ...

    def log_batch(self, run_id: str, *, metrics: Sequence[E]) -> None:
        """Log several metric entries at once."""
        ...

    def log_table(
        self,
        run_id: str,
        data: Mapping[str, Sequence[JsonValue]],
        artifact_file: str,
    ) -> None:
        """Log a table artifact."""
        ...

    def set_terminated(self, run_id: str) -> None:
        """Mark the run finished."""
        ...


class _MlflowClientFactory(Protocol):
    """The MLflow tracking-client constructor."""

    def __call__(self, *, tracking_uri: str | None) -> _MlflowClient[_MlflowMetric]:
        """Return a client bound to a tracking store."""
        ...


class MlflowSink[E]:
    """Metric sink for one MLflow run.

    Parameters
    ----------
    client : _MlflowClient[E]
        The tracking client bound to the run's store.
    metric : _MlflowMetricFactory[E]
        The constructor of the entries a step's scalars are logged as.
    run_id : str
        The run every metric and table is logged to.
    """

    __slots__ = ("_client", "_metric", "_run_id")

    def __init__(
        self, client: _MlflowClient[E], metric: _MlflowMetricFactory[E], run_id: str
    ) -> None:
        self._client = client
        self._metric = metric
        self._run_id = run_id

    def log_scalars(self, mapping: Mapping[str, float], step: int) -> None:
        """Log scalar values at one step in one batch, one history per name."""
        timestamp = time.time_ns() // 1_000_000
        entries = tuple(
            self._metric(key, value, timestamp, step) for key, value in mapping.items()
        )
        self._client.log_batch(self._run_id, metrics=entries)

    def log_table(
        self,
        name: str,
        rows: Sequence[Mapping[str, JsonValue]],
    ) -> None:
        """Log rows as a table artifact."""
        columns = tuple(sorted({key for row in rows for key in row}))
        data = {column: tuple(row.get(column) for row in rows) for column in columns}
        self._client.log_table(self._run_id, data, f"{name}.json")

    def close(self) -> None:
        """Mark the run finished."""
        self._client.set_terminated(self._run_id)


def _build_mlflow(
    settings: MlflowSinkSettings, context: SinkContext, /
) -> MlflowSink[_MlflowMetric]:
    """Open an MLflow run for one point.

    The run is filed under the declared experiment, which is created when the
    store holds none of that name, and is named after the point unless the
    declaration names it.

    Parameters
    ----------
    settings : MlflowSinkSettings
        The experiment, run name and tracking store.
    context : SinkContext
        The point the run records.

    Returns
    -------
    MlflowSink[_MlflowMetric]
        The sink logging to the opened run.

    Raises
    ------
    MissingTelemetryBackendError
        If the ``mlflow`` package is not installed.
    """
    backend = "mlflow"

    try:
        loaded = importlib.import_module(backend)

    except ModuleNotFoundError as error:
        raise MissingTelemetryBackendError(backend) from error

    factory = cast("_MlflowClientFactory", loaded.__dict__["MlflowClient"])
    metric = cast(
        "_MlflowMetricFactory[_MlflowMetric]",
        importlib.import_module(f"{backend}.entities").__dict__["Metric"],
    )
    client = factory(tracking_uri=settings.tracking_uri)
    experiment = client.get_experiment_by_name(settings.experiment_name)
    experiment_id = (
        client.create_experiment(settings.experiment_name)
        if experiment is None
        else experiment.experiment_id
    )
    run_name = context.point_id if settings.run_name is None else settings.run_name
    run = client.create_run(experiment_id, run_name=run_name)

    return MlflowSink(client, metric, run.info.run_id)


PLUGIN: Final = ComponentPlugin(spec=MlflowSinkSpec, build=_build_mlflow)
"""The MLflow sink's plugin."""
