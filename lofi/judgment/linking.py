"""Evaluation of a linking over the readout columns of a design."""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Never, final, override

from torch import Tensor

from lofi._checks import CHECKS, ensure_name

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from lofi.judgment._ports import Linking, ReadoutColumns

__all__ = ["RecordedColumns", "UndeclaredColumnError", "latent_contribution"]


class UndeclaredColumnError(LookupError):
    """Reports a read of a column a linking did not declare.

    Parameters
    ----------
    column : str
        Column that was read.
    admitted : Iterable[str]
        Readout columns the linking declared.
    """

    def __init__(self, column: str, admitted: Iterable[str]) -> None:
        self.column = column
        self.admitted = tuple(sorted(admitted))

        super().__init__(
            f"a linking read the column {column!r}, and its declared readouts are "
            f"{self.admitted!r}"
        )


@final
class _AdmittedReadouts(Mapping[str, Tensor]):
    """The admitted readout columns, recording each one read by name.

    Every read of a column, whether by subscript, ``get``, ``items`` or
    ``values``, goes through one lookup and is recorded there.

    Parameters
    ----------
    source : Mapping[str, Tensor]
        Readout columns being read through.
    admitted : frozenset[str]
        Names a read may name.
    touched : set[str]
        Where each read by name is recorded.
    """

    __slots__ = ("_admitted", "_columns", "_touched")

    def __init__(
        self,
        source: Mapping[str, Tensor],
        admitted: frozenset[str],
        touched: set[str],
    ) -> None:
        self._columns = {
            name: column for name, column in source.items() if name in admitted
        }
        self._admitted = admitted
        self._touched = touched

    @override
    def __getitem__(self, name: str) -> Tensor:
        """Return one admitted column, recording the read.

        Parameters
        ----------
        name : str
            Column to read.

        Raises
        ------
        UndeclaredColumnError
            If the column is not among the admitted names.
        """
        if name not in self._admitted:
            raise UndeclaredColumnError(name, self._admitted)

        self._touched.add(name)

        return self._columns[name]

    @override
    def __contains__(self, name: object) -> bool:
        """Return whether a column is admitted and present, recording nothing."""
        return name in self._columns

    @override
    def __iter__(self) -> Iterator[str]:
        """Return the admitted column names in the source's order."""
        return iter(self._columns)

    @override
    def __len__(self) -> int:
        """Return the number of admitted columns present."""
        return len(self._columns)


@final
class RecordedColumns:
    """A view of readout columns that admits declared names and records reads.

    The view exposes only the source row count and admitted readouts. Access to
    recorded responses, other design columns, or undeclared readouts fails.

    Parameters
    ----------
    source : ReadoutColumns
        Columns being read through.
    admitted : Iterable[str]
        Readout names the linking declared.

    Raises
    ------
    TypeError
        If an admitted name is not a string.
    ValueError
        If an admitted name is empty.
    """

    __slots__ = ("_admitted", "_source", "_touched")

    def __init__(self, source: ReadoutColumns, admitted: Iterable[str]) -> None:
        self._source = source
        self._admitted = frozenset(
            ensure_name("an admitted readout", name) for name in admitted
        )
        self._touched: set[str] = set()

    @property
    def n_rows(self) -> int:
        """Return the number of rows each column has."""
        return self._source.n_rows

    @property
    def readouts(self) -> Mapping[str, Tensor]:
        """Return the admitted readout columns, recording each read by name."""
        return _AdmittedReadouts(self._source.readouts, self._admitted, self._touched)

    @property
    def touched(self) -> frozenset[str]:
        """Return the readout columns read by name through this view so far."""
        return frozenset(self._touched)

    def __getattr__(self, name: str) -> Never:
        """Refuse every column beyond the readouts.

        Parameters
        ----------
        name : str
            Column that was read.

        Raises
        ------
        AttributeError
            If the name is private, so that a slot left unset stays an
            ordinary missing attribute.
        UndeclaredColumnError
            Otherwise.
        """
        if name.startswith("_"):
            raise AttributeError(name)

        raise UndeclaredColumnError(name, self._admitted)

    @override
    def __repr__(self) -> str:
        """Return the admitted names and the rows they cover."""
        return (
            f"RecordedColumns(rows={self.n_rows}, admitted={sorted(self._admitted)!r})"
        )


def latent_contribution(
    linking: Linking,
    columns: ReadoutColumns,
    parameters: Tensor,
    /,
) -> Tensor:
    """Return a linking's contribution to the latent score of each row.

    Under the checked build the linking reads through a view admitting only
    the readouts its domain declares, the parameter vector is checked against
    the declared parameters, and the result is checked to be one value per row.

    Parameters
    ----------
    linking : Linking
        Linking to evaluate.
    columns : ReadoutColumns
        Readout columns to evaluate it over.
    parameters : Tensor
        Fitted parameter vector, in the order the linking's domain declares.

    Raises
    ------
    UndeclaredColumnError
        If the checked build sees the linking read a column it did not declare.
    ValueError
        If the checked build sees a parameter vector of another length, or a
        result that is not one value per row.
    """
    if not CHECKS:
        return linking.contribution(columns, parameters)

    declared = linking.domain.parameter_names

    if parameters.ndim != 1 or parameters.numel() != len(declared):
        msg = (
            f"a linking over the parameters {declared!r} was given a vector of "
            f"shape {tuple(parameters.shape)}"
        )

        raise ValueError(msg)

    view = RecordedColumns(columns, linking.domain.readouts)
    result = linking.contribution(view, parameters)

    if result.shape != (columns.n_rows,):
        msg = (
            f"a linking over {columns.n_rows} rows returned a contribution of "
            f"shape {tuple(result.shape)}"
        )

        raise ValueError(msg)

    return result
