"""Reusable laws for item-stratified target rows."""

from typing import TYPE_CHECKING, ClassVar, NamedTuple, NoReturn

import pytest

from lofi._checks import LawViolationError

if TYPE_CHECKING:
    from lofi.judgment import AgreementBin, StratifiedTargets

__all__ = [
    "StratifiedTargetsSubject",
    "StratifiedTargetsSuite",
    "rows_name_declared_items_and_strata",
]


class StratifiedTargetsSubject(NamedTuple):
    """Expected dimensions of one stratified-target implementation."""

    targets: StratifiedTargets
    items: int
    rows: int
    strata: frozenset[AgreementBin]


def _fail(detail: str) -> NoReturn:
    """Raise a stratified-target law violation.

    Parameters
    ----------
    detail : str
        Description of the failed condition.

    Raises
    ------
    LawViolationError
        Always raised with the law name and detail.
    """
    msg = f"rows_name_declared_items_and_strata: {detail}"
    raise LawViolationError(msg)


def rows_name_declared_items_and_strata(subject: StratifiedTargetsSubject, /) -> None:
    """Check that every row names one declared item and agreement stratum.

    Parameters
    ----------
    subject : StratifiedTargetsSubject
        Targets and their expected dimensions and strata.

    Raises
    ------
    LawViolationError
        If the item or row count differs, a row names an absent item, or the
        observed strata differ from the expected set.
    """
    targets = subject.targets

    if len(targets.items) != subject.items:
        _fail(f"found {len(targets.items)} items, expected {subject.items}")

    if len(targets.rows) != subject.rows:
        _fail(f"found {len(targets.rows)} rows, expected {subject.rows}")

    absent = tuple(row.item for row in targets.rows if row.item not in targets.items)

    if absent:
        _fail(f"rows name absent items {absent!r}")

    miskeyed = tuple(
        (name, target.item)
        for name, target in targets.items.items()
        if name != target.item
    )

    if miskeyed:
        _fail(f"item keys disagree with their targets {miskeyed!r}")

    inconsistent = tuple(
        (row.item, row.stratum, targets.items[row.item].stratum)
        for row in targets.rows
        if row.item in targets.items
        and row.stratum is not targets.items[row.item].stratum
    )

    if inconsistent:
        _fail(f"rows disagree with their item strata {inconsistent!r}")

    observed = frozenset(row.stratum for row in targets.rows)

    if observed != subject.strata:
        _fail(f"found strata {observed!r}, expected {subject.strata!r}")


class StratifiedTargetsSuite:
    """Reusable item-stratification conformance law."""

    pytestmark: ClassVar[list[pytest.MarkDecorator]] = [pytest.mark.conformance]

    def test_rows_name_declared_items_and_strata(
        self, stratified_targets_subject: StratifiedTargetsSubject
    ) -> None:
        """Run the item and stratum law.

        Parameters
        ----------
        stratified_targets_subject : StratifiedTargetsSubject
            Targets and their expected dimensions and strata.
        """
        rows_name_declared_items_and_strata(stratified_targets_subject)
