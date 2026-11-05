"""Atomic report-row storage and deterministic table rendering."""

import os
import sys
from collections import Counter
from math import isnan
from pathlib import Path
from tempfile import mkstemp
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, cast, override

import pyarrow as pa
import pyarrow.parquet as pq

from lofi._checks import ensure_name, ensure_non_negative
from lofi._files import replace_path
from lofi._frozen import FrozenValue
from lofi.eval import ReportRow
from lofi.report import NAN_RATE, REPORT_FIELDS
from lofi.run._experiment import read_experiment
from lofi.run._manifest import replace_file
from lofi.run.results import ROWS_NAME, pilot_directory, results_directory
from lofi.spec import HeldOutParticipantPolicy

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from lofi._types import JsonValue
    from lofi.eval import Split
    from lofi.report import ReportRenderer

__all__ = [
    "REPORT_SCHEMA",
    "DuplicateReportRowError",
    "ParquetReportSink",
    "ReportSelector",
    "RequiredReportRowsError",
    "render_experiment_reports",
    "render_table",
    "select_required_rows",
]

_ARROW_TYPES: Final = MappingProxyType(
    {
        "string": pa.string(),
        "float": pa.float64(),
        "integer": pa.int64(),
        "boolean": pa.bool_(),
    }
)

REPORT_SCHEMA: Final = pa.schema(
    tuple(
        pa.field(field.name, _ARROW_TYPES[field.type], nullable=field.nullable)
        for field in REPORT_FIELDS
    )
)
"""The exact Parquet schema of a serialized canonical report row."""

_KEY_COLUMNS: Final = (
    "freeze_id",
    "config_hash",
    "grammar_fingerprint",
    "convention_hash",
    "suite",
    "task",
    "dataset",
    "split",
    "method",
    "metric",
    "seed",
    "normal_form",
    "specification_hash",
    "fold",
    "rung",
    "held_out_convention",
)
type _KeyValue = str | int | None
type _ModelKey = tuple[str | None, str | None, str | None, str | None]


class ReportSelector(FrozenValue):
    """The complete identity columns selecting one experiment report row."""

    _noun = "report selector"
    __slots__ = ("_values",)

    def __init__(
        self,
        *,
        freeze_id: str,
        config_hash: str,
        grammar_fingerprint: str,
        convention_hash: str | None,
        suite: str,
        task: str,
        dataset: str,
        split: Split,
        method: str,
        metric: str,
        seed: int,
        normal_form: str | None = None,
        specification_hash: str | None = None,
        fold: int | None = None,
        rung: str | None = None,
        held_out_convention: str | None = None,
    ) -> None:
        if held_out_convention is not None and held_out_convention not in {
            policy.value for policy in HeldOutParticipantPolicy
        }:
            msg = (
                "a report selector has an unknown held-out convention: "
                f"{held_out_convention!r}"
            )
            raise ValueError(msg)

        named = (
            freeze_id,
            config_hash,
            grammar_fingerprint,
            suite,
            task,
            dataset,
        )
        prefix = tuple(ensure_name("a selector field", value) for value in named)
        convention = (
            None
            if convention_hash is None
            else ensure_name("a selector convention hash", convention_hash)
        )
        suffix: tuple[_KeyValue, ...] = (
            ensure_name("a selector split", split),
            ensure_name("a selector method", method),
            ensure_name("a selector metric", metric),
            ensure_non_negative("a selector seed", seed),
            None
            if normal_form is None
            else ensure_name("a selector normal form", normal_form),
            None
            if specification_hash is None
            else ensure_name("a selector specification hash", specification_hash),
            None if fold is None else ensure_non_negative("a selector fold", fold),
            None if rung is None else ensure_name("a selector rung", rung),
            held_out_convention,
        )
        object.__setattr__(
            self, "_values", (*prefix[:3], convention, *prefix[3:], *suffix)
        )

    @property
    def key(self) -> tuple[_KeyValue, ...]:
        """Return the hashable complete key."""
        return self._values

    def to_record(self) -> dict[str, JsonValue]:
        """Return the selector in canonical key-column order."""
        return dict(zip(_KEY_COLUMNS, self._values, strict=True))

    @override
    def __eq__(self, other: object) -> bool:
        if type(other) is not ReportSelector:
            return NotImplemented
        return self._values == other._values

    @override
    def __hash__(self) -> int:
        return hash((ReportSelector, self._values))

    @override
    def __repr__(self) -> str:
        fields = ", ".join(
            f"{name}={value!r}" for name, value in self.to_record().items()
        )
        return f"ReportSelector({fields})"


def _record_key(record: Mapping[str, JsonValue]) -> tuple[_KeyValue, ...]:
    return cast(
        "tuple[_KeyValue, ...]", tuple(record[column] for column in _KEY_COLUMNS)
    )


class DuplicateReportRowError(ValueError):
    """A report append whose complete selector is already present."""

    def __init__(self, selectors: Sequence[ReportSelector]) -> None:
        self.selectors = tuple(dict.fromkeys(selectors))
        super().__init__(
            "report rows duplicate complete selectors: "
            + ", ".join(repr(selector) for selector in self.selectors)
        )


class RequiredReportRowsError(LookupError):
    """Required report rows that are absent or ambiguous."""

    def __init__(
        self,
        *,
        missing: Sequence[ReportSelector],
        ambiguous: Sequence[ReportSelector],
    ) -> None:
        self.missing = tuple(missing)
        self.ambiguous = tuple(ambiguous)
        parts: list[str] = []
        if self.missing:
            parts.append("missing " + ", ".join(repr(key) for key in self.missing))
        if self.ambiguous:
            parts.append("ambiguous " + ", ".join(repr(key) for key in self.ambiguous))
        super().__init__("required report rows are " + "; ".join(parts))


def _selector(row: ReportRow) -> ReportSelector:
    context = (row.suite, row.task, row.dataset, row.split, row.seed)
    if any(value is None for value in context):
        msg = f"a stored report row has no experiment context: {row!r}"
        raise ValueError(msg)
    return ReportSelector(
        freeze_id=row.freeze_id,
        config_hash=row.config_hash,
        grammar_fingerprint=row.grammar_fingerprint,
        convention_hash=row.convention_hash,
        suite=cast("str", row.suite),
        task=cast("str", row.task),
        dataset=cast("str", row.dataset),
        split=cast("Split", row.split),
        method=row.method,
        metric=row.metric,
        seed=cast("int", row.seed),
        normal_form=row.normal_form,
        specification_hash=row.specification_hash,
        fold=row.fold,
        rung=row.rung,
        held_out_convention=row.held_out_convention,
    )


class ParquetReportSink:
    """An atomic append-only sink for canonical report rows."""

    __slots__ = ("_path",)

    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def name(self) -> str:
        """Return ``"parquet"``."""
        return "parquet"

    @property
    def path(self) -> Path:
        """Return the report file."""
        return self._path

    def emit(self, rows: Iterable[ReportRow]) -> None:
        """Append rows atomically after refusing duplicate complete keys."""
        incoming = tuple(rows)
        if not incoming:
            return
        for row in incoming:
            if not isinstance(row, ReportRow):
                msg = f"a report sink row is not canonical: {row!r}"
                raise TypeError(msg)
        selectors = tuple(_selector(row) for row in incoming)
        existing = _read_table(self._path)
        stored = tuple(
            cast("Mapping[str, JsonValue]", row) for row in existing.to_pylist()
        )
        seen = {_record_key(row) for row in stored}
        duplicates: list[ReportSelector] = []
        for selector in selectors:
            if selector.key in seen:
                duplicates.append(selector)
            else:
                seen.add(selector.key)
        if duplicates:
            raise DuplicateReportRowError(duplicates)
        _replace_table(self._path, pa.concat_tables((existing, _table(incoming))))

    @override
    def __repr__(self) -> str:
        return f"ParquetReportSink({self._path!r})"


def _empty_table() -> pa.Table:
    return pa.Table.from_pylist([], schema=REPORT_SCHEMA)


def _table(rows: Sequence[ReportRow]) -> pa.Table:
    return pa.Table.from_pylist([row.to_spec() for row in rows], schema=REPORT_SCHEMA)


def _read_table(path: Path) -> pa.Table:
    if not path.exists():
        return _empty_table()
    table = pq.read_table(path)
    if not table.schema.equals(REPORT_SCHEMA, check_metadata=True):
        msg = f"report {path} has schema {table.schema}, expected {REPORT_SCHEMA}"
        raise ValueError(msg)
    return table


def _replace_table(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = mkstemp(
        prefix=f".{path.name}.", suffix=".staged", dir=path.parent
    )
    os.close(descriptor)
    staged = Path(name)
    try:
        pq.write_table(table, staged)
        with staged.open("rb") as stream:
            os.fsync(stream.fileno())
        replace_path(staged, path)
        _sync_directory(path.parent)
    finally:
        staged.unlink(missing_ok=True)


def _sync_directory(directory: Path) -> None:
    """Flush a directory's entries to disk.

    Windows cannot open a directory to flush it, so there an entry is as
    durable as the file system records it.
    """
    if sys.platform == "win32":
        return

    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _records(path: Path) -> tuple[Mapping[str, JsonValue], ...]:
    return tuple(
        cast("Mapping[str, JsonValue]", row) for row in _read_table(path).to_pylist()
    )


def select_required_rows(
    path: Path, selectors: Sequence[ReportSelector]
) -> tuple[Mapping[str, JsonValue], ...]:
    """Return exactly one serialized row per selector, in selector order."""
    rows = _records(path)
    counts = Counter(_record_key(row) for row in rows)
    missing = tuple(key for key in selectors if counts[key.key] == 0)
    ambiguous = tuple(key for key in selectors if counts[key.key] > 1)
    if missing or ambiguous:
        raise RequiredReportRowsError(missing=missing, ambiguous=ambiguous)
    by_key = {_record_key(row): row for row in rows}
    return tuple(by_key[key.key] for key in selectors)


def render_table(
    renderer: ReportRenderer,
    rows: Sequence[Mapping[str, JsonValue]],
    /,
    *,
    columns: Sequence[str],
    labels: Sequence[str],
    precision: int | None,
) -> bytes:
    """Render the declared columns of stored report rows as one table.

    Parameters
    ----------
    renderer : ReportRenderer
        The installed renderer that writes the table.
    rows : Sequence[Mapping[str, JsonValue]]
        The stored rows, in table order, each holding every declared column.
    columns : Sequence[str]
        The columns shown, in table order.
    labels : Sequence[str]
        One heading per column.
    precision : int | None
        Decimal places for floating values, or None to write them in full.

    Returns
    -------
    bytes
        The rendered table.

    Raises
    ------
    ValueError
        If the columns and labels differ in number, or a row lacks a column.
    """
    held_columns = tuple(columns)
    held_labels = tuple(labels)

    if not held_columns or len(held_columns) != len(held_labels):
        msg = (
            f"a report table has {len(held_columns)} columns but "
            f"{len(held_labels)} labels"
        )
        raise ValueError(msg)

    missing = tuple(
        column for column in held_columns if any(column not in row for row in rows)
    )

    if missing:
        msg = f"stored report rows lack declared columns: {missing!r}"
        raise ValueError(msg)

    return renderer.render(
        held_labels,
        tuple(tuple(row[column] for column in held_columns) for row in rows),
        precision=precision,
    )


def _nan_rate(rows: Sequence[Mapping[str, JsonValue]]) -> float:
    """Return the fraction of rows whose value is absent or NaN."""
    if not rows:
        return 1.0
    missing = sum(
        value is None or (isinstance(value, float) and isnan(value))
        for value in (row.get("value") for row in rows)
    )
    return missing / len(rows)


def render_experiment_reports(
    document: Path, root: Path, /, *, pilot: bool
) -> tuple[Path, ...]:
    """Render every table an experiment declares from its ``rows.parquet``.

    Parameters
    ----------
    document : Path
        The experiment document.
    root : Path
        The repository root the experiment's results lie under.
    pilot : bool
        Whether the tables are rendered from the experiment's pilot results,
        under ``<root>/pilot/results/<experiment>/``, rather than from its
        results under its freeze.

    Returns
    -------
    tuple[Path, ...]
        The tables written beside the rows they were rendered from.

    Raises
    ------
    ValueError
        If the rows cannot be read or a declared table selects none of them.
    """
    experiment = read_experiment(document)
    if not experiment.reports:
        return ()
    directory = (
        pilot_directory(root, experiment)
        if pilot
        else results_directory(root, experiment)
    )
    source = directory / ROWS_NAME
    try:
        records = tuple(
            cast("Mapping[str, JsonValue]", row)
            for row in pq.read_table(source).to_pylist()
        )
    except (OSError, pa.ArrowException) as refused:
        msg = f"experiment report rows cannot be read from {source}: {refused}"
        raise ValueError(msg) from refused

    declared = {row.label: row for row in experiment.rows}
    written: list[Path] = []
    for layout in experiment.reports:
        selected: list[Mapping[str, JsonValue]] = []
        for label in layout.rows:
            row = declared[label]
            matching = tuple(
                record
                for record in records
                if record.get("task") == row.evaluator
                and record.get("metric") == row.metric
            )
            grouped: dict[_ModelKey, list[Mapping[str, JsonValue]]] = {}
            for record in matching:
                key = (
                    cast("str | None", record.get("config_hash")),
                    cast("str | None", record.get("grammar_fingerprint")),
                    cast("str | None", record.get("method")),
                    cast("str | None", record.get("metric")),
                )
                grouped.setdefault(key, []).append(record)
            for records in grouped.values():
                rate = _nan_rate(records)
                selected.extend({**record, NAN_RATE: rate} for record in records)
        if not selected:
            msg = f"report {layout.name!r} selects no stored rows"
            raise ValueError(msg)
        renderer = layout.renderer()
        output = directory / f"{layout.name}{renderer.suffix}"
        replace_file(
            output,
            render_table(
                renderer,
                selected,
                columns=layout.columns,
                labels=layout.labels,
                precision=layout.precision,
            ),
        )
        written.append(output)
    return tuple(written)
