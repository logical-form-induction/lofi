"""W&B metric sink."""

import importlib
from typing import TYPE_CHECKING, Final, Protocol, cast

from lofi.registry import ComponentPlugin
from lofi.run._sinks import MissingTelemetryBackendError
from lofi.spec.sinks.wandb import WandbSinkSettings, WandbSinkSpec

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lofi._types import JsonValue
    from lofi.run._sinks import SinkContext

__all__ = ["PLUGIN", "WandbSink"]


class _WandbTable(Protocol):
    """W&B table interface."""


class _WandbTableFactory(Protocol):
    """W&B table factory."""

    def __call__(
        self,
        *,
        columns: Sequence[str],
        data: Sequence[Sequence[JsonValue]],
    ) -> _WandbTable:
        """Return a table from columns and rows."""
        ...


class _WandbRun(Protocol):
    """W&B run operations used by the sink."""

    def log(
        self,
        data: Mapping[str, float | _WandbTable],
        *,
        step: int | None = None,
    ) -> None:
        """Log values to the run."""
        ...

    def finish(self) -> None:
        """Finish the run."""
        ...


class _WandbClient(Protocol):
    """W&B module operations used by the factory."""

    def init(
        self,
        *,
        project: str,
        entity: str | None,
        name: str | None,
    ) -> _WandbRun:
        """Start and return a run."""
        ...


class WandbSink:
    """Metric sink for one W&B run."""

    __slots__ = ("_run", "_table_factory")

    def __init__(
        self,
        run: _WandbRun,
        table_factory: _WandbTableFactory,
    ) -> None:
        self._run = run
        self._table_factory = table_factory

    def log_scalars(self, mapping: Mapping[str, float], step: int) -> None:
        """Log scalar values at one step."""
        self._run.log(mapping, step=step)

    def log_table(
        self,
        name: str,
        rows: Sequence[Mapping[str, JsonValue]],
    ) -> None:
        """Log rows as a table."""
        columns = tuple(sorted({key for row in rows for key in row}))
        data = tuple(tuple(row.get(column) for column in columns) for row in rows)
        table = self._table_factory(columns=columns, data=data)
        self._run.log({name: table})

    def close(self) -> None:
        """Finish the run."""
        self._run.finish()


def _build_wandb(settings: WandbSinkSettings, context: SinkContext, /) -> WandbSink:
    """Start a W&B run for one point.

    The run is named after the point unless the declaration names it.

    Parameters
    ----------
    settings : WandbSinkSettings
        The project, entity and run name.
    context : SinkContext
        The point the run records.

    Returns
    -------
    WandbSink
        The sink logging to the started run.

    Raises
    ------
    MissingTelemetryBackendError
        If the ``wandb`` package is not installed.
    """
    backend = "wandb"

    try:
        loaded = importlib.import_module(backend)

    except ModuleNotFoundError as error:
        raise MissingTelemetryBackendError(backend) from error

    client = cast("_WandbClient", loaded)
    table_factory = cast("_WandbTableFactory", loaded.__dict__["Table"])
    run = client.init(
        project=settings.project,
        entity=settings.entity,
        name=context.point_id if settings.run_name is None else settings.run_name,
    )

    return WandbSink(run, table_factory)


PLUGIN: Final = ComponentPlugin(spec=WandbSinkSpec, build=_build_wandb)
"""The W&B sink's plugin."""
