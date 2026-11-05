"""Reusable laws for sources of judgment rows."""

from typing import TYPE_CHECKING, ClassVar, NamedTuple, NoReturn

import pytest

from lofi._checks import LawViolationError
from lofi.judgment import design_of

if TYPE_CHECKING:
    from lofi.judgment import JudgmentSource

__all__ = [
    "JudgmentSourceSubject",
    "JudgmentSourceSuite",
    "repeated_measures_are_preserved",
]


class JudgmentSourceSubject(NamedTuple):
    """Inputs for the judgment-source conformance laws.

    Attributes
    ----------
    source : JudgmentSource
        The source under test.
    judgments : int
        How many judgments the source was given.
    item : str
        An item at least two judgments were given to.
    rows : int
        How many of that item's judgments have a response and a participant.
    participants : int
        How many distinct participants gave those.
    """

    source: JudgmentSource
    judgments: int
    item: str
    rows: int
    participants: int


def _fail(detail: str, *, law: str) -> NoReturn:
    """Raise a judgment-source law violation.

    Parameters
    ----------
    detail : str
        Description of the failed condition.
    law : str
        Name of the failed law.

    Raises
    ------
    LawViolationError
        Always raised with the law name and detail.
    """
    message = f"{law}: {detail}"

    raise LawViolationError(message)


def repeated_measures_are_preserved(subject: JudgmentSourceSubject, /) -> None:
    """Check that every judgment of one item stays one row of that item.

    The design's rows and the source's skips together account for the
    judgments, and the subject's item has one index, one row per judgment
    that became a row, and one participant index per distinct participant.

    Parameters
    ----------
    subject : JudgmentSourceSubject
        Source and the item it is checked over.

    Raises
    ------
    LawViolationError
        If rows and skips do not account for the judgments, the item is
        absent or repeated among the item levels, or the item's rows or
        participants are not what the subject states.
    """
    law = "repeated_measures_are_preserved"

    if subject.rows < 2:
        _fail(
            f"the subject's item {subject.item!r} needs at least two rows to "
            f"repeat a measure, and states {subject.rows}",
            law=law,
        )

    design = design_of(subject.source)
    skipped = sum(subject.source.skips.values())

    if design.n_rows + skipped != subject.judgments:
        _fail(
            f"{subject.judgments} judgments became {design.n_rows} rows and "
            f"{skipped} skips",
            law=law,
        )

    indices = [
        index for index, level in enumerate(design.item_levels) if level == subject.item
    ]

    if len(indices) != 1:
        _fail(
            f"the item {subject.item!r} has the item indices {indices!r}",
            law=law,
        )

    rows = design.items == indices[0]
    found = int(rows.sum())

    if found != subject.rows:
        _fail(
            f"the item {subject.item!r} has {found} rows and the subject states "
            f"{subject.rows}",
            law=law,
        )

    participants = int(design.participants[rows].unique().numel())

    if participants != subject.participants:
        _fail(
            f"the item {subject.item!r} was judged by {participants} participant "
            f"indices and the subject states {subject.participants}",
            law=law,
        )


class JudgmentSourceSuite:
    """Reusable judgment-source conformance laws."""

    pytestmark: ClassVar[list[pytest.MarkDecorator]] = [pytest.mark.conformance]

    def test_repeated_measures_are_preserved(
        self, judgment_source_subject: JudgmentSourceSubject
    ) -> None:
        """Run the repeated-measures law.

        Parameters
        ----------
        judgment_source_subject : JudgmentSourceSubject
            Source and the item it is checked over.
        """
        repeated_measures_are_preserved(judgment_source_subject)
