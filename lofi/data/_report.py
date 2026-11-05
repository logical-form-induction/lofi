"""Corpus load reports."""

from types import MappingProxyType
from typing import TYPE_CHECKING, override

from lofi._checks import ensure_name, ensure_non_negative

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping

    from lofi.data._reachability import ReachabilityReport

__all__ = ["AnalysisCounts", "LoadReport"]


class AnalysisCounts:
    """Gold-analysis validation counts.

    Parameters
    ----------
    checked : int
        Number of checked analyses.
    reachable : int
        Number of reachable analyses.
    in_normal_form : int
        Number of analyses in normal form.

    Raises
    ------
    ValueError
        If a count is negative, or if more analyses held than were checked.
    """

    __slots__ = ("_checked", "_in_normal_form", "_reachable")

    def __init__(
        self,
        checked: int,
        reachable: int = 0,
        in_normal_form: int = 0,
    ) -> None:
        self._checked = ensure_non_negative("a checked analysis count", checked)
        self._reachable = ensure_non_negative("a reachable analysis count", reachable)
        self._in_normal_form = ensure_non_negative(
            "a normal-form analysis count",
            in_normal_form,
        )

        for name, count in (
            ("reachable", self._reachable),
            ("in normal form", self._in_normal_form),
        ):
            if count > self._checked:
                msg = (
                    f"{count} analyses are {name} out of {self._checked} checked, "
                    "which counts an analysis the load never looked at"
                )
                raise ValueError(msg)

    @property
    def checked(self) -> int:
        """Return how many gold analyses the load checked."""
        return self._checked

    @property
    def reachable(self) -> int:
        """Return how many of those the deduction system can derive."""
        return self._reachable

    @property
    def in_normal_form(self) -> int:
        """Return how many of those are already in their formalism's normal form."""
        return self._in_normal_form

    @property
    def reachability(self) -> float | None:
        """Return the derivable share, or None if no analyses were checked."""
        return None if self._checked == 0 else self._reachable / self._checked

    @property
    def normal_form_conformance(self) -> float | None:
        """Return the normal-form share, or None if no analyses were checked."""
        return None if self._checked == 0 else self._in_normal_form / self._checked

    def _identity(self) -> tuple[int, int, int]:
        """Return the three counts a set of analysis counts is equal on."""
        return (self._checked, self._reachable, self._in_normal_form)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value counts the same three quantities."""
        if type(other) is not AnalysisCounts:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the three counts."""
        return hash((AnalysisCounts, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds these counts."""
        return (
            f"AnalysisCounts({self._checked}, {self._reachable}, "
            f"{self._in_normal_form})"
        )


class LoadReport[L: Hashable = Hashable]:
    """A summary of one corpus load.

    Parameters
    ----------
    records_seen : int
        Number of records read.
    emitted : int
        Number of emitted items.
    skips_by_reason : Mapping[str, int]
        Skipped-record counts by reason.
    analyses : AnalysisCounts | None
        Optional gold-analysis validation counts.
    reachability_report : ReachabilityReport[L] | None
        Optional per-derivation reachability report.

    Raises
    ------
    ValueError
        If counts are negative or inconsistent, or a skip reason is empty.
    """

    __slots__ = (
        "_analyses",
        "_emitted",
        "_reachability_report",
        "_records_seen",
        "_skips_by_reason",
    )

    def __init__(
        self,
        records_seen: int,
        emitted: int,
        skips_by_reason: Mapping[str, int] = MappingProxyType({}),
        analyses: AnalysisCounts | None = None,
        reachability_report: ReachabilityReport[L] | None = None,
    ) -> None:
        self._records_seen = ensure_non_negative("a record count", records_seen)
        self._emitted = ensure_non_negative("an emitted count", emitted)
        self._skips_by_reason = MappingProxyType(
            {
                ensure_name("a skip reason", reason): ensure_non_negative(
                    f"the count skipped as {reason!r}",
                    count,
                )
                for reason, count in skips_by_reason.items()
            },
        )

        accounted = self._emitted + sum(self._skips_by_reason.values())

        if accounted != self._records_seen:
            msg = (
                f"a load saw {self._records_seen} records and accounts for "
                f"{accounted} of them: {self._emitted} emitted and "
                f"{sum(self._skips_by_reason.values())} skipped"
            )
            raise ValueError(msg)

        if analyses is not None and analyses.checked > self._emitted:
            msg = (
                f"a load checked {analyses.checked} analyses against "
                f"{self._emitted} emitted items, so it checked an analysis of "
                "an item it did not emit"
            )
            raise ValueError(msg)

        if reachability_report is not None:
            if analyses is None:
                msg = "a per-derivation reachability report requires analysis counts"
                raise ValueError(msg)

            if (
                reachability_report.checked_derivations != analyses.checked
                or reachability_report.reachable_derivations != analyses.reachable
            ):
                msg = "per-derivation reachability does not match the analysis counts"
                raise ValueError(msg)

        self._analyses = analyses
        self._reachability_report = reachability_report

    @property
    def records_seen(self) -> int:
        """Return how many records the pass read from the corpus."""
        return self._records_seen

    @property
    def emitted(self) -> int:
        """Return how many of those became items."""
        return self._emitted

    @property
    def skips_by_reason(self) -> Mapping[str, int]:
        """Return skipped-record counts by reason."""
        return self._skips_by_reason

    @property
    def skipped(self) -> int:
        """Return how many records were set aside, over every reason."""
        return sum(self._skips_by_reason.values())

    @property
    def analyses(self) -> AnalysisCounts | None:
        """Return gold-analysis counts, if checked."""
        return self._analyses

    @property
    def reachability_report(self) -> ReachabilityReport[L] | None:
        """Return per-derivation reachability outcomes, if checked."""
        return self._reachability_report

    @property
    def reachability(self) -> float | None:
        """Return the derivable share, or None if no analyses were checked."""
        return None if self._analyses is None else self._analyses.reachability

    @property
    def normal_form_conformance(self) -> float | None:
        """Return the normal-form share, or None if no analyses were checked."""
        return (
            None if self._analyses is None else self._analyses.normal_form_conformance
        )

    def _identity(
        self,
    ) -> tuple[
        int,
        int,
        tuple[tuple[str, int], ...],
        AnalysisCounts | None,
        ReachabilityReport[L] | None,
    ]:
        """Return the fields a report is equal on."""
        return (
            self._records_seen,
            self._emitted,
            tuple(sorted(self._skips_by_reason.items())),
            self._analyses,
            self._reachability_report,
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value reports the same load the same way."""
        if type(other) is not LoadReport:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over all report fields."""
        return hash((LoadReport, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this report."""
        prefix = (
            f"LoadReport({self._records_seen}, {self._emitted}, "
            f"{dict(self._skips_by_reason)!r}, {self._analyses!r})"
        )
        if self._reachability_report is None:
            return prefix

        return prefix[:-1] + f", reachability_report={self._reachability_report!r})"
