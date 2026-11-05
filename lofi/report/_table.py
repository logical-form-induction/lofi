"""The columns a report table may show and the renderers that write it."""

from pathlib import PurePosixPath, PureWindowsPath
from typing import (
    TYPE_CHECKING,
    Final,
    Literal,
    NamedTuple,
    Protocol,
    runtime_checkable,
)

from lofi._checks import ensure_name
from lofi.eval import RESIDUAL_SOURCES
from lofi.registry import Registry, RegistryError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi._types import JsonValue

__all__ = [
    "DERIVED_COLUMNS",
    "NAN_RATE",
    "REPORT_FIELDS",
    "REPORT_FORMAT_FAMILY",
    "ReportField",
    "ReportRenderer",
    "format_cell",
    "report_columns",
    "report_renderer",
    "report_stem",
]

type ReportFieldType = Literal["string", "float", "integer", "boolean"]
"""The scalar type a stored report column holds."""


class ReportField(NamedTuple):
    """One stored report column: its name, scalar type and whether it may be null."""

    name: str
    type: ReportFieldType
    nullable: bool


REPORT_FIELDS: Final = (
    ReportField("freeze_id", "string", nullable=False),
    ReportField("config_hash", "string", nullable=False),
    ReportField("grammar_fingerprint", "string", nullable=False),
    ReportField("convention_hash", "string", nullable=True),
    ReportField("suite", "string", nullable=True),
    ReportField("task", "string", nullable=True),
    ReportField("dataset", "string", nullable=True),
    ReportField("split", "string", nullable=True),
    ReportField("method", "string", nullable=False),
    ReportField("metric", "string", nullable=False),
    ReportField("value", "float", nullable=True),
    ReportField("n", "integer", nullable=False),
    ReportField("resampling_unit", "string", nullable=False),
    ReportField("resampling_seed", "integer", nullable=True),
    ReportField("seed", "integer", nullable=True),
    ReportField("ci_low", "float", nullable=True),
    ReportField("ci_high", "float", nullable=True),
    ReportField("ci_level", "float", nullable=True),
    ReportField("normal_form", "string", nullable=True),
    ReportField("specification_hash", "string", nullable=True),
    ReportField("fold", "integer", nullable=True),
    ReportField("rung", "string", nullable=True),
    ReportField("held_out_convention", "string", nullable=True),
    ReportField("required_control", "string", nullable=True),
    ReportField("budget_kind", "string", nullable=False),
    ReportField("budget_depth", "integer", nullable=True),
    ReportField("budget_beam", "integer", nullable=True),
    ReportField("budget_mass", "float", nullable=True),
    ReportField("residual_measured", "boolean", nullable=False),
    *(
        ReportField(f"residual_{source}", "float", nullable=True)
        for source in RESIDUAL_SOURCES
    ),
)
"""Every column of a stored canonical report row, in storage order."""

NAN_RATE: Final = "nan_rate"
"""The derived column holding, for each row, the fraction of the rows sharing its
configuration, grammar, method and metric whose value is missing or NaN."""

DERIVED_COLUMNS: Final = (NAN_RATE,)
"""Columns a report computes from the stored rows rather than reads from them."""

REPORT_FORMAT_FAMILY: Final = "report_format"
"""The family whose entry-point group lists the installed table renderers."""


@runtime_checkable
class ReportRenderer(Protocol):
    """Writes one table of report cells in one output format."""

    @property
    def suffix(self) -> str:
        """Return the filename suffix of a rendered table, its dot included."""
        ...

    def render(
        self,
        labels: Sequence[str],
        rows: Sequence[Sequence[JsonValue]],
        /,
        *,
        precision: int | None,
    ) -> bytes:
        """Return the table's bytes.

        Parameters
        ----------
        labels : Sequence[str]
            One heading per column, in column order.
        rows : Sequence[Sequence[JsonValue]]
            One sequence of scalar cells per table row, in column order.
        precision : int | None
            Decimal places for floating cells, or None to write them in full.
        """
        ...


def report_columns(columns: Sequence[str], /) -> tuple[str, ...]:
    """Return columns a report may show, refusing any it cannot.

    A report may show every column of ``REPORT_FIELDS`` and every column of
    ``DERIVED_COLUMNS``.

    Parameters
    ----------
    columns : Sequence[str]
        The declared columns, in table order.

    Raises
    ------
    ValueError
        If a column is neither stored nor derived.
    """
    held = tuple(columns)
    known = (*(field.name for field in REPORT_FIELDS), *DERIVED_COLUMNS)
    unknown = tuple(column for column in held if column not in known)

    if unknown:
        msg = f"a report names unknown columns: {unknown!r}"
        raise ValueError(msg)

    return held


def report_stem(name: str, /) -> str:
    """Return a report name that is a bare filename stem.

    Parameters
    ----------
    name : str
        The declared report name.

    Raises
    ------
    ValueError
        If the name is blank, or would leave the directory it is written to.
    """
    named = ensure_name("a report name", name)

    if (
        not named.strip()
        or PurePosixPath(named).name != named
        or PureWindowsPath(named).name != named
        or named in {".", ".."}
    ):
        msg = f"a report name is not a filename stem: {named!r}"
        raise ValueError(msg)

    return named


def report_renderer(name: str, /) -> ReportRenderer:
    """Return the renderer installed under one format name.

    Parameters
    ----------
    name : str
        The format's entry-point name in the ``lofi.report_format`` group.

    Raises
    ------
    ValueError
        If no renderer is installed under the name.
    TypeError
        If the entry names something other than a renderer.
    """
    try:
        member = Registry[object](REPORT_FORMAT_FAMILY).get(name)

    except RegistryError as refused:
        raise ValueError(str(refused)) from refused

    if not isinstance(member, ReportRenderer):
        msg = f"the report format {name!r} names {member!r}, not a report renderer"
        raise TypeError(msg)

    return member


def format_cell(value: JsonValue, precision: int | None, /) -> str:
    """Return one scalar cell as unescaped text.

    Parameters
    ----------
    value : JsonValue
        The cell; null is written as empty text.
    precision : int | None
        Decimal places for a floating cell, or None to write it in full.

    Raises
    ------
    TypeError
        If the cell is not a scalar.
    """
    if value is None:
        return ""

    if isinstance(value, float) and precision is not None:
        return f"{value:.{precision}f}"

    if isinstance(value, str | int | float):
        return str(value)

    msg = f"a report cell is not scalar: {value!r}"
    raise TypeError(msg)
