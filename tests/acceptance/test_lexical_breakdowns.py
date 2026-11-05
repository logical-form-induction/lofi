"""A trained parser reports supertag accuracy by training evidence."""

from typing import TYPE_CHECKING

from tests.acceptance.driver import train

if TYPE_CHECKING:
    from pathlib import Path

    from tests.acceptance.driver import Row


def _count(row: Row, /) -> int:
    """Return one emitted integer sample count."""
    count = row["n"]
    if isinstance(count, bool) or not isinstance(count, int):
        msg = f"a lexical diagnostic row has noninteger count {count!r}"
        raise TypeError(msg)
    return count


def test_a_trained_point_reports_frequency_and_gold_ambiguity_strata(
    tmp_path: Path,
) -> None:
    trained = train("lexical-breakdowns.yaml", tmp_path / "experiment")
    rows = {row["metric"]: row for row in trained.rows}
    total = _count(rows["supertag-accuracy"])
    frequency = tuple(
        rows[f"supertag-accuracy-frequency-{name}"]
        for name in ("unseen", "rare", "frequent")
    )
    ambiguity = tuple(
        rows[f"supertag-accuracy-ambiguity-{name}"]
        for name in ("unseen", "unambiguous", "ambiguous")
    )

    assert total > 0
    assert sum(map(_count, frequency)) == total
    assert sum(map(_count, ambiguity)) == total
    assert any(_count(row) > 0 for row in frequency)
    assert any(_count(row) > 0 for row in ambiguity)
