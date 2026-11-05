"""Approximation residuals and summaries.

A reported number was computed under an enumeration budget, which bounds the
depth, beam and omitted mass of the search that produced it, or under no
enumeration budget at all, as a posterior diagnostic drawn by a sampler is.
The second is a declared value, ``NO_ENUMERATION_BUDGET``, not a missing one:
it states that no budget applies and so no enumeration residual can be
measured, and a summary counts such rows apart from measured and unmeasured
ones.
"""

from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, cast, final, get_args, override

from lofi._checks import ensure_finite, ensure_non_negative
from lofi.eval._ports import Budget

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from lofi._types import JsonValue

__all__ = [
    "NO_ENUMERATION_BUDGET",
    "RESIDUAL_SOURCES",
    "Approximation",
    "BudgetKind",
    "NoEnumerationBudget",
    "Residual",
    "ResidualSource",
    "ResidualSummary",
    "residual_summary",
]


type ResidualSource = Literal[
    "beam-truncation", "chart-cap", "enumeration-budget", "prover-timeout"
]
"""Defines one place a run stops short of the quantity it is computing."""

RESIDUAL_SOURCES: Final = cast(
    "tuple[ResidualSource, ...]", get_args(ResidualSource.__value__)
)
"""Approximation sources in report order, as ``ResidualSource`` declares them."""

type BudgetKind = Literal["enumeration", "none"]
"""Names whether an enumeration budget applies to a reported number."""


@final
class NoEnumerationBudget:
    """Declares that no enumeration budget applies to a reported number.

    A number drawn without enumerating anything, such as a posterior
    diagnostic, has no depth, beam or mass bound to state and no enumeration
    residual to measure. Every instance is equal to every other.
    """

    __slots__ = ()

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value also declares no enumeration budget."""
        if type(other) is not NoEnumerationBudget:
            return NotImplemented

        return True

    @override
    def __hash__(self) -> int:
        """Return the one hash every declaration shares."""
        return hash(NoEnumerationBudget)

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this declaration."""
        return "NoEnumerationBudget()"


NO_ENUMERATION_BUDGET: Final = NoEnumerationBudget()
"""The declaration that no enumeration budget applies."""


class Residual:
    """Stores measured approximation costs.

    Attributes
    ----------
    measurements : Mapping[ResidualSource, float]
        Measured costs by source.
    unmeasured : tuple[ResidualSource, ...]
        Sources without measurements.
    is_measured : bool
        Whether every source was measured.

    Parameters
    ----------
    measurements : Mapping[ResidualSource, float]
        Measured nonnegative costs by source.

    Raises
    ------
    TypeError
        If a source is unknown or a cost is not numeric.
    ValueError
        If a cost is not finite or is negative.
    """

    __slots__ = ("_measurements",)

    def __init__(self, measurements: Mapping[ResidualSource, float]) -> None:
        checked: dict[ResidualSource, float] = {}

        for source, cost in measurements.items():
            if source not in RESIDUAL_SOURCES:
                msg = (
                    f"a residual names a source that is not a declared one: "
                    f"{source!r}, which is not in {RESIDUAL_SOURCES!r}"
                )
                raise TypeError(msg)

            measured = ensure_finite(f"the residual for {source!r}", cost)

            if measured < 0.0:
                msg = f"the residual for {source!r} is negative: {cost!r}"
                raise ValueError(msg)

            checked[source] = measured

        self._measurements: Mapping[ResidualSource, float] = MappingProxyType(checked)

    @property
    def measurements(self) -> Mapping[ResidualSource, float]:
        """Return measured costs as a read-only view."""
        return self._measurements

    @property
    def unmeasured(self) -> tuple[ResidualSource, ...]:
        """Return the sources nobody measured, in the order a report lists them."""
        return tuple(
            source for source in RESIDUAL_SOURCES if source not in self._measurements
        )

    @property
    def is_measured(self) -> bool:
        """Return whether every source was measured."""
        return not self.unmeasured

    def to_spec(self) -> dict[str, JsonValue]:
        """Return the residual as one column per source.

        Returns
        -------
        dict[str, JsonValue]
            Serialized residual measurements.
        """
        return {source: self._measurements.get(source) for source in RESIDUAL_SOURCES}

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value measured the same sources at the same cost."""
        if type(other) is not Residual:
            return NotImplemented

        return dict(self._measurements) == dict(other._measurements)

    @override
    def __hash__(self) -> int:
        """Return a hash over the measured sources and their costs."""
        return hash((Residual, tuple(sorted(self._measurements.items()))))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this residual."""
        return f"Residual({dict(self._measurements)!r})"


class Approximation:
    """Records a computation budget and its measured residual.

    Attributes
    ----------
    budget : Budget | NoEnumerationBudget
        Computation budget, or the declaration that none applies.
    residual : Residual | None
        Measured residual, if available.
    applies : bool
        Whether an enumeration budget applies.
    is_measured : bool
        Whether every residual source was measured.

    Parameters
    ----------
    budget : Budget | NoEnumerationBudget
        Computation budget, or the declaration that none applies.
    residual : Residual | None
        Measured residual, if available; always None where no budget applies.

    Raises
    ------
    TypeError
        If ``budget`` lacks the required protocol or ``residual`` is invalid.
    ValueError
        If a budget bound is negative or nonfinite, or a residual is given
        where no enumeration budget applies.
    """

    __slots__ = ("_beam", "_budget", "_depth", "_mass", "_residual")

    def __init__(
        self, budget: Budget | NoEnumerationBudget, *, residual: Residual | None
    ) -> None:
        if residual is not None and not isinstance(residual, Residual):
            msg = f"an approximation's residual is not one: {residual!r}"
            raise TypeError(msg)

        self._budget = budget
        self._residual = residual

        if type(budget) is NoEnumerationBudget:
            if residual is not None:
                msg = (
                    "a residual is measured against an enumeration budget, and "
                    "no enumeration budget applies to this number"
                )
                raise ValueError(msg)

            self._depth: int | None = None
            self._beam: int | None = None
            self._mass: float | None = None

            return

        if not isinstance(budget, Budget):
            msg = (
                f"an approximation's budget publishes no depth, beam and mass: "
                f"{budget!r}"
            )
            raise TypeError(msg)

        self._depth = ensure_non_negative("a budget's depth", budget.depth)
        self._beam = ensure_non_negative("a budget's beam", budget.beam)
        self._mass = (
            None
            if budget.mass is None
            else ensure_finite("a budget's mass bound", budget.mass)
        )

    @property
    def budget(self) -> Budget | NoEnumerationBudget:
        """Return the budget the number was computed under, or that none applies."""
        return self._budget

    @property
    def residual(self) -> Residual | None:
        """Return measured approximation cost, if available."""
        return self._residual

    @property
    def applies(self) -> bool:
        """Return whether an enumeration budget applies to the number."""
        return self._depth is not None

    @property
    def is_measured(self) -> bool:
        """Return whether a residual is present and every source in it was measured."""
        return self._residual is not None and self._residual.is_measured

    def to_spec(self) -> dict[str, JsonValue]:
        """Return the budget and residual as report columns.

        ``budget_kind`` is ``enumeration`` for a number computed under a
        budget and ``none`` for one no enumeration budget applies to, whose
        bounds and residuals are null.

        Returns
        -------
        dict[str, JsonValue]
            Serialized approximation metadata.
        """
        measured: Mapping[ResidualSource, float] = (
            {} if self._residual is None else self._residual.measurements
        )
        kind: BudgetKind = "enumeration" if self.applies else "none"

        return {
            "budget_kind": kind,
            "budget_depth": self._depth,
            "budget_beam": self._beam,
            "budget_mass": self._mass,
            "residual_measured": self.is_measured,
            **{
                f"residual_{source}": measured.get(source)
                for source in RESIDUAL_SOURCES
            },
        }

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value bought the same accuracy and cost the same."""
        if type(other) is not Approximation:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over whether a budget applies, its bounds and the residual."""
        return hash((Approximation, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this approximation."""
        return f"Approximation({self._budget!r}, residual={self._residual!r})"

    def _identity(
        self,
    ) -> tuple[bool, int | None, int | None, float | None, Residual | None]:
        """Return the five fields an approximation is equal on."""
        return (self.applies, self._depth, self._beam, self._mass, self._residual)


class ResidualSummary:
    """Stores report counts by cost-measurement status.

    Attributes
    ----------
    rows : int
        Number of summarized rows.
    measured : int
        Number of fully measured rows.
    unmeasured : int
        Number of incompletely measured rows under an enumeration budget.
    not_applicable : int
        Number of rows no enumeration budget applies to.
    by_source : Mapping[ResidualSource, int]
        Missing-measurement counts by source.

    Parameters
    ----------
    rows : int
        Number of summarized rows.
    measured : int
        Number of fully measured rows.
    by_source : Mapping[ResidualSource, int]
        Missing-measurement counts by source.
    not_applicable : int
        Number of rows no enumeration budget applies to.

    Raises
    ------
    TypeError
        If a count is not an integer.
    ValueError
        If a count is negative or the measured and not-applicable rows exceed
        ``rows``.
    """

    __slots__ = ("_by_source", "_measured", "_not_applicable", "_rows")

    def __init__(
        self,
        rows: int,
        measured: int,
        by_source: Mapping[ResidualSource, int],
        *,
        not_applicable: int,
    ) -> None:
        self._rows = ensure_non_negative("a summary's row count", rows)
        self._measured = ensure_non_negative("a summary's measured count", measured)
        self._not_applicable = ensure_non_negative(
            "a summary's not-applicable count", not_applicable
        )
        counted = {
            source: ensure_non_negative(f"the count for {source!r}", count)
            for source, count in by_source.items()
        }
        self._by_source: Mapping[ResidualSource, int] = MappingProxyType(
            {source: count for source, count in counted.items() if count}
        )

        if self._measured + self._not_applicable > self._rows:
            msg = (
                f"a summary counts {measured} measured rows and {not_applicable} "
                f"rows no budget applies to among {rows} rows in all"
            )
            raise ValueError(msg)

    @property
    def rows(self) -> int:
        """Return how many rows were summarized."""
        return self._rows

    @property
    def measured(self) -> int:
        """Return how many rows measured every source."""
        return self._measured

    @property
    def unmeasured(self) -> int:
        """Return how many budgeted rows left at least one source unmeasured."""
        return self._rows - self._measured - self._not_applicable

    @property
    def not_applicable(self) -> int:
        """Return how many rows no enumeration budget applies to."""
        return self._not_applicable

    @property
    def by_source(self) -> Mapping[ResidualSource, int]:
        """Return missing-measurement counts by source.

        Returns
        -------
        Mapping[ResidualSource, int]
            Read-only source counts.
        """
        return self._by_source

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value is a summary with the same counts."""
        if type(other) is not ResidualSummary:
            return NotImplemented

        return (
            self._rows == other._rows
            and self._measured == other._measured
            and self._not_applicable == other._not_applicable
            and dict(self._by_source) == dict(other._by_source)
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the counts and the per-source counts."""
        return hash(
            (
                ResidualSummary,
                self._rows,
                self._measured,
                self._not_applicable,
                tuple(sorted(self._by_source.items())),
            )
        )

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this summary."""
        return (
            f"ResidualSummary({self._rows!r}, {self._measured!r}, "
            f"{dict(self._by_source)!r}, not_applicable={self._not_applicable!r})"
        )


def residual_summary(approximations: Iterable[Approximation]) -> ResidualSummary:
    """Summarize report rows by cost-measurement status.

    Parameters
    ----------
    approximations : Iterable[Approximation]
        Approximation metadata for report rows.

    Returns
    -------
    ResidualSummary
        Counts of measured and unmeasured residuals, with the rows no
        enumeration budget applies to counted apart from both.

    Raises
    ------
    TypeError
        If an input is not an approximation.
    """
    rows = 0
    measured = 0
    not_applicable = 0
    by_source = dict.fromkeys(RESIDUAL_SOURCES, 0)

    for approximation in approximations:
        if not isinstance(approximation, Approximation):
            msg = f"a summarized approximation is not one: {approximation!r}"
            raise TypeError(msg)

        rows += 1

        if not approximation.applies:
            not_applicable += 1
            continue

        residual = approximation.residual
        missing = RESIDUAL_SOURCES if residual is None else residual.unmeasured

        if not missing:
            measured += 1

        for source in missing:
            by_source[source] += 1

    return ResidualSummary(rows, measured, by_source, not_applicable=not_applicable)
