"""Combination-matrix report generation."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

CONSTRUCTED = "constructed"
"""Identifies a constructed component pair."""

DECLARED_INCOMPATIBLE = "declared_incompatible"
"""Identifies a declared incompatible component pair."""


class CombinationCounts(NamedTuple):
    """Counts for evaluated component pairs by outcome."""

    constructed: int
    declared_incompatible: int
    total: int


def count_outcomes(outcomes: Iterable[str]) -> CombinationCounts:
    """Count the supplied combination outcomes.

    Parameters
    ----------
    outcomes : Iterable[str]
        Evaluated outcomes from a composability matrix.

    Returns
    -------
    CombinationCounts
        Constructed, declared-incompatible, and total counts.

    Raises
    ------
    ValueError
        If an outcome is not recognized.
    """
    constructed = 0
    declared_incompatible = 0
    total = 0

    for outcome in outcomes:
        total += 1

        if outcome == CONSTRUCTED:
            constructed += 1

        elif outcome == DECLARED_INCOMPATIBLE:
            declared_incompatible += 1

        else:
            msg = f"unknown combination outcome: {outcome!r}"

            raise ValueError(msg)

    return CombinationCounts(constructed, declared_incompatible, total)


def render_report(counts: CombinationCounts) -> str:
    """Return the combination-count report.

    Parameters
    ----------
    counts : CombinationCounts
        Counts computed from evaluated outcomes.

    Returns
    -------
    str
        One line containing every count.
    """
    return (
        f"{counts.constructed} of {counts.total} combinations construct; "
        f"{counts.declared_incompatible} declared incompatible"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Write an outcome report to standard output and the configured CI summary.

    Parameters
    ----------
    argv : Sequence[str] | None
        Outcomes without the program name. If ``None``, read them from
        ``sys.argv[1:]``.

    Returns
    -------
    int
        Zero after writing the report.

    Raises
    ------
    ValueError
        If an outcome is not recognized.
    """
    outcomes = sys.argv[1:] if argv is None else argv
    line = render_report(count_outcomes(outcomes))
    sys.stdout.write(f"{line}\n")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")

    if summary is not None:
        with Path(summary).open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(f"{line}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
