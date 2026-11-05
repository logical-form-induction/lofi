"""Trackio metric sink."""

import importlib
from typing import TYPE_CHECKING, Final, Protocol, cast

from lofi.registry import ComponentPlugin
from lofi.run._sinks import MissingTelemetryBackendError
from lofi.spec.sinks.trackio import TrackioSinkSettings, TrackioSinkSpec

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lofi._types import JsonValue
    from lofi.run._sinks import SinkContext

__all__ = ["PLUGIN", "TrackioSink"]


class _TrackioTable(Protocol):
    """Trackio table interface."""


class _TrackioTableFactory(Protocol):
    """Trackio table factory."""

    def __call__(
        self,
        *,
        columns: Sequence[str],
        data: Sequence[Sequence[JsonValue]],
    ) -> _TrackioTable:
        """Return a table from columns and rows."""
        ...


class _TrackioClient(Protocol):
    """Trackio operations used by the sink."""

    def init(
        self,
        *,
        project: str,
        name: str | None,
        space_id: str | None,
        server_url: str | None,
    ) -> None:
        """Start a run."""
        ...

    def log(
        self,
        data: Mapping[str, float | _TrackioTable],
        *,
        step: int | None = None,
    ) -> None:
        """Log values to the run."""
        ...

    def finish(self) -> None:
        """Finish the run."""
        ...


class TrackioSink:
    """Metric sink for one Trackio run."""

    __slots__ = ("_client", "_table_factory")

    def __init__(
        self,
        client: _TrackioClient,
        table_factory: _TrackioTableFactory,
    ) -> None:
        self._client = client
        self._table_factory = table_factory

    def log_scalars(self, mapping: Mapping[str, float], step: int) -> None:
        """Log scalar values at one step."""
        self._client.log(mapping, step=step)

    def log_table(
        self,
        name: str,
        rows: Sequence[Mapping[str, JsonValue]],
    ) -> None:
        """Log rows as a table."""
        columns = tuple(sorted({key for row in rows for key in row}))
        data = tuple(tuple(row.get(column) for column in columns) for row in rows)
        table = self._table_factory(columns=columns, data=data)
        self._client.log({name: table})

    def close(self) -> None:
        """Finish the run."""
        self._client.finish()


def _build_trackio(
    settings: TrackioSinkSettings, context: SinkContext, /
) -> TrackioSink:
    """Start a Trackio run for one point.

    The run is named after the point unless the declaration names it.

    Parameters
    ----------
    settings : TrackioSinkSettings
        The project, run name and remote target.
    context : SinkContext
        The point the run records.

    Returns
    -------
    TrackioSink
        The sink logging to the started run.

    Raises
    ------
    MissingTelemetryBackendError
        If the ``trackio`` package is not installed.
    """
    backend = "trackio"

    try:
        loaded = importlib.import_module(backend)

    except ModuleNotFoundError as error:
        raise MissingTelemetryBackendError(backend) from error

    client = cast("_TrackioClient", loaded)
    table_factory = cast("_TrackioTableFactory", loaded.__dict__["Table"])
    client.init(
        project=settings.project,
        name=context.point_id if settings.run_name is None else settings.run_name,
        space_id=settings.space_id,
        server_url=settings.server_url,
    )

    return TrackioSink(client, table_factory)


PLUGIN: Final = ComponentPlugin(spec=TrackioSinkSpec, build=_build_trackio)
"""The Trackio sink's plugin."""
