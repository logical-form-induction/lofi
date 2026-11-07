from typing import Final, cast

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from lofi.nn.tokens.embedder import ReservedTokens, TokenVocabulary
from lofi.nn.tokens.transfer import (
    AmbiguityPolicy,
    TransferReport,
    initialize_from_types,
)

WIDTH: Final = 2
RESERVED: Final = ReservedTokens({"<unk>": 0})
VOCABULARY: Final = TokenVocabulary(("dog", "cat"), reserved=RESERVED, unknown="<unk>")
TABLE: Final = torch.zeros(3, WIDTH)

WALKS: Final = torch.tensor([1.0, 2.0])
BARKS: Final = torch.tensor([3.0, 6.0])
MEOWS: Final = torch.tensor([-1.0, -4.0])


def vocabulary_over(*tokens: str) -> TokenVocabulary:
    return TokenVocabulary(tokens, reserved=RESERVED, unknown="<unk>")


def rows(table: torch.Tensor) -> list[list[float]]:
    return table.tolist()


def test_the_mean_policy_averages_the_vectors_a_key_was_given() -> None:
    written, report = initialize_from_types(
        TABLE,
        {"dog": (WALKS, BARKS), "cat": (MEOWS,)},
        vocabulary=VOCABULARY,
        policy=AmbiguityPolicy.MEAN,
    )

    assert rows(written)[VOCABULARY.index_of("dog")] == [2.0, 4.0]
    assert rows(written)[VOCABULARY.index_of("cat")] == [-1.0, -4.0]
    assert report == TransferReport(
        rows_considered=2, rows_initialized=2, rows_skipped=0, rows_unmatched=0
    )


def test_the_first_policy_takes_the_vector_that_came_in_first() -> None:
    written, report = initialize_from_types(
        TABLE,
        {"dog": (WALKS, BARKS), "cat": (MEOWS,)},
        vocabulary=VOCABULARY,
        policy=AmbiguityPolicy.FIRST,
    )

    assert rows(written)[VOCABULARY.index_of("dog")] == [1.0, 2.0]
    assert report.rows_initialized == 2
    assert report.rows_skipped == 0


def test_the_skip_policy_leaves_an_ambiguous_row_as_it_found_it() -> None:
    start = torch.full((3, WIDTH), 7.0)

    written, report = initialize_from_types(
        start,
        {"dog": (WALKS, BARKS), "cat": (MEOWS,)},
        vocabulary=VOCABULARY,
        policy=AmbiguityPolicy.SKIP,
    )

    assert rows(written)[VOCABULARY.index_of("dog")] == [7.0, 7.0]
    assert rows(written)[VOCABULARY.index_of("cat")] == [-1.0, -4.0]
    assert report == TransferReport(
        rows_considered=2, rows_initialized=1, rows_skipped=1, rows_unmatched=0
    )


def test_a_key_the_vocabulary_holds_no_row_for_reaches_no_row_at_all() -> None:
    written, report = initialize_from_types(
        TABLE,
        {"zzz": (WALKS,), "<unk>": (BARKS,)},
        vocabulary=VOCABULARY,
        policy=AmbiguityPolicy.MEAN,
    )

    assert rows(written) == rows(TABLE)
    assert report == TransferReport(
        rows_considered=2, rows_initialized=0, rows_skipped=0, rows_unmatched=2
    )


def test_the_callers_key_casing_reaches_the_row_that_spelling_names() -> None:
    cased = vocabulary_over("The", "the")

    written, report = initialize_from_types(
        torch.zeros(3, WIDTH),
        {"The": (WALKS,), "the": (BARKS,), "THE": (MEOWS,)},
        vocabulary=cased,
        policy=AmbiguityPolicy.MEAN,
    )

    assert rows(written)[cased.index_of("The")] == [1.0, 2.0]
    assert rows(written)[cased.index_of("the")] == [3.0, 6.0]
    assert cased.index_of("THE") == cased.unknown_index
    assert rows(written)[cased.unknown_index] == [0.0, 0.0]
    assert report == TransferReport(
        rows_considered=3, rows_initialized=2, rows_skipped=0, rows_unmatched=1
    )


def test_a_width_mismatch_at_the_last_vector_is_caught_before_any_write() -> None:
    start = torch.zeros(3, WIDTH)
    narrow = torch.tensor([5.0])

    with pytest.raises(ValueError, match=r"vector 2 of 'cat' is 1 wide"):
        initialize_from_types(
            start,
            {"dog": (WALKS,), "cat": (MEOWS, BARKS, narrow)},
            vocabulary=VOCABULARY,
            policy=AmbiguityPolicy.MEAN,
        )

    assert rows(start) == rows(torch.zeros(3, WIDTH))


def test_a_width_mismatch_is_caught_wherever_it_sits() -> None:
    wide = torch.tensor([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match=r"vector 0 of 'dog' is 3 wide"):
        initialize_from_types(
            TABLE,
            {"dog": (wide,)},
            vocabulary=VOCABULARY,
            policy=AmbiguityPolicy.FIRST,
        )


def test_the_table_a_caller_handed_in_is_not_the_table_that_comes_back() -> None:
    start = torch.zeros(3, WIDTH)

    written, _ = initialize_from_types(
        start,
        {"dog": (WALKS,)},
        vocabulary=VOCABULARY,
        policy=AmbiguityPolicy.FIRST,
    )

    assert written is not start
    assert rows(start) == [[0.0, 0.0]] * 3
    assert rows(written)[VOCABULARY.index_of("dog")] == [1.0, 2.0]


@settings(max_examples=50)
@given(
    offered=st.dictionaries(
        st.sampled_from(("dog", "cat", "zzz", "<unk>", "Dog")),
        st.integers(min_value=1, max_value=3),
        min_size=0,
        max_size=5,
    )
)
def test_the_three_counts_partition_the_keys_under_every_policy(
    offered: dict[str, int],
) -> None:
    vectors = {
        key: tuple(torch.full((WIDTH,), float(at)) for at in range(count))
        for key, count in offered.items()
    }
    reports = {
        policy: initialize_from_types(
            TABLE, vectors, vocabulary=VOCABULARY, policy=policy
        )[1]
        for policy in AmbiguityPolicy
    }

    for policy, report in reports.items():
        counted = report.rows_initialized + report.rows_skipped + report.rows_unmatched

        assert report.rows_considered == len(offered)
        assert counted == len(offered)

        if policy is not AmbiguityPolicy.SKIP:
            assert report.rows_skipped == 0

    assert len({report.rows_unmatched for report in reports.values()}) == 1
    assert reports[AmbiguityPolicy.SKIP].rows_skipped == sum(
        1 for key, count in offered.items() if count > 1 and VOCABULARY.holds(key)
    )


def test_a_policy_outside_the_closed_set_is_refused_where_it_is_named() -> None:
    assert {policy.value for policy in AmbiguityPolicy} == {"mean", "first", "skip"}

    with pytest.raises(ValueError, match="not a valid AmbiguityPolicy"):
        AmbiguityPolicy("average")

    with pytest.raises(TypeError, match="runs under an AmbiguityPolicy"):
        initialize_from_types(
            TABLE,
            {"dog": (WALKS,)},
            vocabulary=VOCABULARY,
            policy=cast("AmbiguityPolicy", "mean"),
        )


def test_a_key_offered_no_vector_is_refused_rather_than_counted() -> None:
    with pytest.raises(ValueError, match="'dog' was offered no vector"):
        initialize_from_types(
            TABLE,
            {"dog": ()},
            vocabulary=VOCABULARY,
            policy=AmbiguityPolicy.MEAN,
        )

    with pytest.raises(ValueError, match="a transfer key is empty"):
        initialize_from_types(
            TABLE,
            {"": (WALKS,)},
            vocabulary=VOCABULARY,
            policy=AmbiguityPolicy.MEAN,
        )


def test_a_table_no_row_could_be_written_into_is_refused() -> None:
    with pytest.raises(TypeError, match="holds real numbers"):
        initialize_from_types(
            torch.zeros(3, WIDTH, dtype=torch.int64),
            {"dog": (WALKS,)},
            vocabulary=VOCABULARY,
            policy=AmbiguityPolicy.MEAN,
        )

    with pytest.raises(ValueError, match="so it is rank 2, not 3"):
        initialize_from_types(
            torch.zeros(3, WIDTH, 1),
            {"dog": (WALKS,)},
            vocabulary=VOCABULARY,
            policy=AmbiguityPolicy.MEAN,
        )

    with pytest.raises(ValueError, match="at least one column"):
        initialize_from_types(
            torch.zeros(3, 0),
            {"dog": (WALKS,)},
            vocabulary=VOCABULARY,
            policy=AmbiguityPolicy.MEAN,
        )

    with pytest.raises(
        ValueError, match="a vocabulary of 3 rows was given a table of 4"
    ):
        initialize_from_types(
            torch.zeros(4, WIDTH),
            {"dog": (WALKS,)},
            vocabulary=VOCABULARY,
            policy=AmbiguityPolicy.MEAN,
        )


def test_a_vector_no_row_could_take_is_refused() -> None:
    with pytest.raises(TypeError, match=r"vector 0 of 'dog' holds real numbers"):
        initialize_from_types(
            TABLE,
            {"dog": (torch.zeros(WIDTH, dtype=torch.int64),)},
            vocabulary=VOCABULARY,
            policy=AmbiguityPolicy.MEAN,
        )

    with pytest.raises(
        ValueError, match=r"vector 0 of 'dog' is one number per feature"
    ):
        initialize_from_types(
            TABLE,
            {"dog": (torch.zeros(1, WIDTH),)},
            vocabulary=VOCABULARY,
            policy=AmbiguityPolicy.MEAN,
        )


def test_a_report_whose_counts_do_not_account_for_the_keys_is_refused() -> None:
    with pytest.raises(ValueError, match="do not account for them"):
        TransferReport(
            rows_considered=5, rows_initialized=1, rows_skipped=1, rows_unmatched=1
        )

    with pytest.raises(ValueError, match="is negative"):
        TransferReport(
            rows_considered=0, rows_initialized=-1, rows_skipped=1, rows_unmatched=0
        )

    with pytest.raises(TypeError, match="not an integer"):
        TransferReport(
            rows_considered=cast("int", 1.0),
            rows_initialized=1,
            rows_skipped=0,
            rows_unmatched=0,
        )


def test_two_reports_of_the_same_transfer_are_one_value() -> None:
    report = TransferReport(
        rows_considered=4, rows_initialized=2, rows_skipped=1, rows_unmatched=1
    )
    same = TransferReport(
        rows_considered=4, rows_initialized=2, rows_skipped=1, rows_unmatched=1
    )

    assert report == same
    assert hash(report) == hash(same)
    assert report.rows_considered == 4
    assert report.rows_initialized == 2
    assert report.rows_skipped == 1
    assert report.rows_unmatched == 1
    assert report != TransferReport(
        rows_considered=4, rows_initialized=1, rows_skipped=2, rows_unmatched=1
    )
    assert report.__eq__(4) is NotImplemented
    assert repr(report) == (
        "TransferReport(rows_considered=4, rows_initialized=2, "
        "rows_skipped=1, rows_unmatched=1)"
    )
