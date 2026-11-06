from types import MappingProxyType
from typing import TYPE_CHECKING, Final, override

import pytest
from tests.conformance.judgment_source import (
    JudgmentSourceSubject,
    JudgmentSourceSuite,
    repeated_measures_are_preserved,
)

from lofi._checks import LawViolationError
from lofi.judgment import JudgmentRow, RowColumns

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

ITEM: Final = "the dog barked"
OTHER: Final = "birds sing"
ANNOTATORS: Final = ("p1", "p2", "p3", "p4", "p5")


class Rows:
    """Rows from (participant, item, response) triples, skipping a negative response."""

    def __init__(self, table: Iterable[tuple[str, str, int]]) -> None:
        self._rows: list[JudgmentRow] = []
        skipped = 0

        for participant, item, response in table:
            if response < 0:
                skipped += 1

                continue

            self._rows.append(
                JudgmentRow(participant, item, response, RowColumns({}, {}))
            )

        self._skips: Mapping[str, int] = MappingProxyType(
            {"a negative response": skipped} if skipped else {}
        )

    @property
    def response_levels(self) -> int:
        return 7

    @property
    def factors(self) -> tuple[str, ...]:
        return ()

    @property
    def skips(self) -> Mapping[str, int]:
        return self._skips

    def rows(self) -> tuple[JudgmentRow, ...]:
        return tuple(self._rows)


class LastResponseWins(Rows):
    """Keeps one row per item, as a container keyed by sentence would."""

    @override
    def rows(self) -> tuple[JudgmentRow, ...]:
        kept = {row.item: row for row in super().rows()}

        return tuple(kept.values())


class OneParticipant(Rows):
    """Attributes every row to the first participant."""

    @override
    def rows(self) -> tuple[JudgmentRow, ...]:
        return tuple(
            JudgmentRow(ANNOTATORS[0], row.item, row.response, RowColumns({}, {}))
            for row in super().rows()
        )


class SplitItem(Rows):
    """Gives each participant's copy of the item its own identity."""

    @override
    def rows(self) -> tuple[JudgmentRow, ...]:
        return tuple(
            JudgmentRow(
                row.participant,
                f"{row.item}#{row.participant}",
                row.response,
                RowColumns({}, {}),
            )
            for row in super().rows()
        )


class Uncounted(Rows):
    """Drops a response below zero without counting it."""

    @property
    @override
    def skips(self) -> Mapping[str, int]:
        return {}


def table() -> tuple[tuple[str, str, int], ...]:
    rated = tuple(
        (annotator, ITEM, level) for level, annotator in enumerate(ANNOTATORS)
    )

    return (*rated, ("p1", OTHER, 6), ("p2", OTHER, -1))


def subject_of(source: Rows) -> JudgmentSourceSubject:
    return JudgmentSourceSubject(source, len(table()), ITEM, 5, 5)


class TestRows(JudgmentSourceSuite):
    @pytest.fixture
    def judgment_source_subject(self) -> JudgmentSourceSubject:
        return subject_of(Rows(table()))


def test_a_source_keeping_the_last_response_per_item_fails_the_law() -> None:
    with pytest.raises(LawViolationError, match="became 2 rows"):
        repeated_measures_are_preserved(subject_of(LastResponseWins(table())))


def test_a_source_merging_the_participants_fails_the_law() -> None:
    with pytest.raises(LawViolationError, match="judged by 1 participant"):
        repeated_measures_are_preserved(subject_of(OneParticipant(table())))


def test_a_source_splitting_the_item_per_participant_fails_the_law() -> None:
    with pytest.raises(LawViolationError, match="has the item indices \\[\\]"):
        repeated_measures_are_preserved(subject_of(SplitItem(table())))


def test_a_source_dropping_a_judgment_without_counting_it_fails_the_law() -> None:
    with pytest.raises(LawViolationError, match="became 6 rows and 0 skips"):
        repeated_measures_are_preserved(subject_of(Uncounted(table())))


def test_a_subject_stating_the_wrong_row_count_is_caught() -> None:
    subject = JudgmentSourceSubject(Rows(table()), len(table()), ITEM, 4, 5)

    with pytest.raises(LawViolationError, match="has 5 rows and the subject states 4"):
        repeated_measures_are_preserved(subject)


def test_a_subject_over_a_single_measure_is_refused() -> None:
    subject = JudgmentSourceSubject(Rows(table()), len(table()), OTHER, 1, 1)

    with pytest.raises(LawViolationError, match="at least two rows"):
        repeated_measures_are_preserved(subject)
