from typing import TYPE_CHECKING, Final

import pytest
from scripts.report_combinations import (
    CombinationCounts,
    count_outcomes,
    main,
    render_report,
)

if TYPE_CHECKING:
    from pathlib import Path

CONSTRUCTED: Final = "constructed"
INCOMPATIBLE: Final = "declared_incompatible"
OUTCOMES: Final = (CONSTRUCTED, INCOMPATIBLE, CONSTRUCTED, CONSTRUCTED)


def test_every_supplied_outcome_contributes_to_the_computed_total() -> None:
    counts = count_outcomes(OUTCOMES)

    assert counts == CombinationCounts(3, 1, len(OUTCOMES))
    assert counts.constructed + counts.declared_incompatible == counts.total


def test_changing_one_outcome_changes_the_counts_but_not_the_total() -> None:
    original = count_outcomes(OUTCOMES)
    changed = count_outcomes((INCOMPATIBLE, *OUTCOMES[1:]))

    assert changed.constructed == original.constructed - 1
    assert changed.declared_incompatible == original.declared_incompatible + 1
    assert changed.total == original.total


def test_an_unknown_outcome_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown combination outcome"):
        count_outcomes((*OUTCOMES, "unknown"))


def test_the_report_contains_the_exact_computed_line() -> None:
    assert render_report(count_outcomes(OUTCOMES)) == (
        f"3 of {len(OUTCOMES)} combinations construct; 1 declared incompatible"
    )


def test_main_writes_the_exact_computed_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(OUTCOMES) == 0
    assert capsys.readouterr().out == (
        f"3 of {len(OUTCOMES)} combinations construct; 1 declared incompatible\n"
    )


def test_main_appends_the_same_line_to_the_ci_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    summary = tmp_path / "summary.md"
    summary.write_text("existing\n", encoding="utf-8", newline="\n")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))

    assert main(OUTCOMES) == 0

    line = capsys.readouterr().out
    assert summary.read_text(encoding="utf-8") == f"existing\n{line}"
