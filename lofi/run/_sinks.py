"""Metric sinks, held-out rows, and evaluator artifacts for one point."""

from pathlib import PurePath, PureWindowsPath
from typing import TYPE_CHECKING, NamedTuple, override

from lofi._checks import ensure_name
from lofi.run._manifest import replace_file

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from lofi._types import JsonValue
    from lofi.eval import EvaluationArtifact, Report
    from lofi.run._ports import MetricSink

__all__ = [
    "ARTIFACT_DIRECTORY",
    "HELD_OUT_TABLE",
    "MissingTelemetryBackendError",
    "NullSink",
    "PointReportSink",
    "SinkContext",
    "log_held_out_rows",
    "write_artifacts",
]

ARTIFACT_DIRECTORY = "evaluation-artifacts"
"""The point-directory child holding evaluator artifacts by role."""

HELD_OUT_TABLE = "held-out"
"""The table name under which a point's held-out rows reach a metric sink."""


class SinkContext(NamedTuple):
    """What the run supplies to every metric sink it builds for one point.

    Attributes
    ----------
    point_id : str
        The identifier of the point whose metrics the sink receives, which
        names the sink's run when its declaration names none.
    """

    point_id: str


class MissingTelemetryBackendError(ImportError):
    """Unavailable selected telemetry client."""

    def __init__(self, backend: str) -> None:
        self.backend = backend
        message = (
            f"the selected {backend} sink requires the {backend!r} package; "
            "install lofi[tracking]"
        )
        super().__init__(message)


class NullSink:
    """Discards reports and metric events.

    Attributes
    ----------
    name : str
        Sink name, ``"null"``.
    """

    __slots__ = ()

    @property
    def name(self) -> str:
        """Return ``"null"``."""
        return "null"

    def emit(self, report: Report) -> None:
        """Accept a report and write nothing.

        Parameters
        ----------
        report : Report
            The evaluation report.
        """

    def log_scalars(self, mapping: Mapping[str, float], step: int) -> None:
        """Accept scalar metrics and write nothing.

        Parameters
        ----------
        mapping : Mapping[str, float]
            The scalar metrics to log.
        step : int
            The training step.
        """

    def log_table(self, name: str, rows: Sequence[Mapping[str, JsonValue]]) -> None:
        """Accept table rows and write nothing.

        Parameters
        ----------
        name : str
            The name used in validation or reporting.
        rows : Sequence[Mapping[str, JsonValue]]
            The table rows.
        """

    def close(self) -> None:
        """Close the sink without side effects."""

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this sink."""
        return "NullSink()"


class PointReportSink:
    """Collects the rows and evaluator artifacts one point reports.

    A row is serialized as its report columns, the budget and residual
    columns of the approximation it was scored under, and the identifier of
    the point that reported it, so the row leads back to the point's run
    directory and manifest.

    Parameters
    ----------
    point_id : str
        The identifier of the point whose rows are collected.
    """

    __slots__ = ("_artifacts", "_point_id", "_rows")

    def __init__(self, point_id: str) -> None:
        self._point_id = point_id
        self._rows: list[Mapping[str, JsonValue]] = []
        self._artifacts: list[tuple[str, EvaluationArtifact]] = []

    @property
    def rows(self) -> tuple[Mapping[str, JsonValue], ...]:
        """Return the serialized rows in emission order."""
        return tuple(self._rows)

    @property
    def artifacts(self) -> tuple[tuple[str, EvaluationArtifact], ...]:
        """Return evaluator artifacts in emission order."""
        return tuple(self._artifacts)

    def emit(self, report: Report, /) -> None:
        """Collect a report's rows.

        Parameters
        ----------
        report : Report
            Completed report.
        """
        self._rows.extend(
            {
                **row.to_spec(),
                "point_id": self._point_id,
            }
            for row in report
        )

    def emit_artifact(self, role: str, artifact: EvaluationArtifact, /) -> None:
        """Collect one artifact under its evaluator's declared role."""
        named = ensure_name("an evaluator role", role)

        if (
            PurePath(named).name != named
            or PureWindowsPath(named).name != named
            or named in {".", ".."}
        ):
            msg = f"an evaluator role is not one path segment: {named!r}"
            raise ValueError(msg)

        key = named, artifact.name
        if any((held_role, held.name) == key for held_role, held in self._artifacts):
            msg = f"evaluator artifact {named!r}/{artifact.name!r} was emitted twice"
            raise ValueError(msg)

        self._artifacts.append((named, artifact))

    @override
    def __repr__(self) -> str:
        """Return the point and how many rows it has reported."""
        return f"PointReportSink({self._point_id!r}, rows={len(self._rows)})"


def log_held_out_rows(
    sinks: Sequence[MetricSink],
    rows: Sequence[Mapping[str, JsonValue]],
    /,
) -> None:
    """Log a point's serialized held-out rows to every declared metric sink.

    Parameters
    ----------
    sinks : Sequence[MetricSink]
        The metric sinks the point's run declares.
    rows : Sequence[Mapping[str, JsonValue]]
        The rows the point's held-out evaluators reported, as its rows record
        holds them.
    """
    for sink in sinks:
        sink.log_table(HELD_OUT_TABLE, rows)


def write_artifacts(
    directory: Path,
    artifacts: Sequence[tuple[str, EvaluationArtifact]],
    /,
) -> tuple[Path, ...]:
    """Atomically persist evaluator artifacts by role beneath one point."""
    written = []

    for role, artifact in artifacts:
        target = directory / ARTIFACT_DIRECTORY / role / artifact.name
        target.parent.mkdir(parents=True, exist_ok=True)
        replace_file(target, artifact.content)
        written.append(target)

    return tuple(written)
