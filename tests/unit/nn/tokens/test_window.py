from typing import cast

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from lofi.nn._values import TruncationReport
from lofi.nn.tokens.window import (
    REFUSE,
    REPORT,
    ContextOverflowError,
    ContextWindow,
    OverflowPlugin,
)

WORDS = ("the", "quick", "brown", "fox", "jumps")


def words() -> st.SearchStrategy[tuple[str, ...]]:
    return st.lists(
        st.text(alphabet="abc", min_size=1, max_size=3), min_size=0, max_size=8
    ).map(tuple)


def test_a_window_reports_the_suffix_it_drops() -> None:
    window = ContextWindow(3, overflow=REPORT)

    report = window.fit(WORDS)

    assert report == TruncationReport((3, 4), ("fox", "jumps"))
    assert report.words == WORDS[3:]
    assert window.rows == 3
    assert window.overflow is REPORT


def test_a_window_that_refuses_overflow_raises_and_names_the_item() -> None:
    window = ContextWindow(3, overflow=REFUSE)

    with pytest.raises(ContextOverflowError, match="5 words needs 5 rows") as caught:
        window.fit(WORDS)

    assert caught.value.words == WORDS
    assert caught.value.rows == 3
    assert caught.value.needed == 5


def test_an_item_that_fits_drops_nothing_under_either_overflow() -> None:
    for overflow in (REFUSE, REPORT):
        report = ContextWindow(5, overflow=overflow).fit(WORDS)

        assert report == TruncationReport((), ())
        assert not report


def test_a_word_is_kept_or_dropped_whole_by_its_row_cost() -> None:
    window = ContextWindow(4, overflow=REPORT)

    # the third word would push the total to five rows, so it and its successors go
    report = window.fit(("ab", "c", "def", "g"), costs=(2, 1, 3, 1))

    assert report == TruncationReport((2, 3), ("def", "g"))
    assert window.fit(("ab", "c", "d"), costs=(2, 1, 1)) == TruncationReport((), ())


def test_a_window_refuses_by_row_cost_rather_than_by_word_count() -> None:
    window = ContextWindow(4, overflow=REFUSE)

    with pytest.raises(ContextOverflowError, match="2 words needs 5 rows"):
        window.fit(("ab", "cde"), costs=(2, 3))


def test_every_word_has_one_positive_cost() -> None:
    window = ContextWindow(4, overflow=REPORT)

    with pytest.raises(ValueError, match="2 words was given 1 row costs"):
        window.fit(("ab", "c"), costs=(2,))

    with pytest.raises(ValueError, match="at least one row"):
        window.fit(("ab", "c"), costs=(2, 0))


def test_a_window_holds_at_least_one_row_and_a_declared_overflow() -> None:
    with pytest.raises(ValueError, match="no rows"):
        ContextWindow(0, overflow=REPORT)

    with pytest.raises(ValueError, match="negative"):
        ContextWindow(-1, overflow=REPORT)

    with pytest.raises(TypeError, match="not an integer"):
        ContextWindow(cast("int", 2.5), overflow=REPORT)

    with pytest.raises(TypeError, match="by an OverflowPlugin"):
        ContextWindow(2, overflow=cast("OverflowPlugin", "report"))


def test_windows_of_one_size_and_overflow_are_one_window() -> None:
    window = ContextWindow(3, overflow=REFUSE)

    assert window == ContextWindow(3, overflow=REFUSE)
    assert hash(window) == hash(ContextWindow(3, overflow=REFUSE))
    assert window != ContextWindow(3, overflow=REPORT)
    assert window != ContextWindow(4, overflow=REFUSE)
    assert window != 3
    assert repr(window).startswith("ContextWindow(3, overflow=")


@settings(max_examples=50)
@given(item=words(), rows=st.integers(min_value=1, max_value=6))
def test_the_dropped_words_are_exactly_the_suffix_past_the_window(
    item: tuple[str, ...], rows: int
) -> None:
    report = ContextWindow(rows, overflow=REPORT).fit(item)
    kept = len(item) - len(report)

    assert kept == min(len(item), rows)
    assert report.words == item[kept:]
    assert report.positions == tuple(range(kept, len(item)))


@settings(max_examples=50)
@given(item=words(), rows=st.integers(min_value=1, max_value=6))
def test_refusing_and_reporting_agree_about_which_items_fit(
    item: tuple[str, ...], rows: int
) -> None:
    reported = ContextWindow(rows, overflow=REPORT).fit(item)
    refusing = ContextWindow(rows, overflow=REFUSE)

    if reported:
        with pytest.raises(ContextOverflowError):
            refusing.fit(item)
    else:
        assert refusing.fit(item) == reported
