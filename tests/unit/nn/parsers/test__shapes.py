import pytest
import torch

from lofi.deduce import SpanIndex
from lofi.nn.parsers._shapes import (
    checked_leaf_scores,
    checked_scalar_chart,
    checked_split_scores,
)


@pytest.mark.parametrize("n", [1, 2, 5])
@pytest.mark.parametrize("batch", [1, 3])
def test_leaf_scores_of_the_shape_the_index_calls_for_report_their_batch(
    n: int, batch: int
) -> None:
    assert checked_leaf_scores(torch.zeros(batch, n), SpanIndex(n)) == batch


@pytest.mark.parametrize("n", [1, 2, 5])
@pytest.mark.parametrize("batch", [1, 3])
def test_a_scalar_chart_of_the_shape_the_index_calls_for_reports_its_batch(
    n: int, batch: int
) -> None:
    index = SpanIndex(n)

    assert (
        checked_scalar_chart("a chart", torch.zeros(batch, index.num_spans), index)
        == batch
    )


@pytest.mark.parametrize("n", [1, 2, 5])
@pytest.mark.parametrize("batch", [1, 3])
def test_split_scores_of_the_shape_the_index_calls_for_pass(n: int, batch: int) -> None:
    index = SpanIndex(n)

    assert (
        checked_split_scores(
            torch.zeros(batch, index.num_spans, index.max_splits), index, batch
        )
        is None
    )


def test_leaf_scores_are_refused_unless_they_hold_log_weights() -> None:
    with pytest.raises(TypeError, match="leaf scores are log weights"):
        checked_leaf_scores(torch.zeros(1, 3, dtype=torch.long), SpanIndex(3))


def test_leaf_scores_are_refused_unless_they_run_over_item_and_token() -> None:
    with pytest.raises(ValueError, match="rank 2"):
        checked_leaf_scores(torch.zeros(3), SpanIndex(3))

    with pytest.raises(ValueError, match="rank 2"):
        checked_leaf_scores(torch.zeros(1, 1, 3), SpanIndex(3))


def test_leaf_scores_are_refused_unless_there_is_one_per_token() -> None:
    with pytest.raises(ValueError, match="leaf scores of shape"):
        checked_leaf_scores(torch.zeros(1, 4), SpanIndex(3))


def test_a_scalar_chart_is_refused_unless_it_holds_log_weights() -> None:
    index = SpanIndex(3)

    with pytest.raises(TypeError, match="a chart holds log weights"):
        checked_scalar_chart(
            "a chart", torch.zeros(1, index.num_spans, dtype=torch.long), index
        )


def test_a_scalar_chart_is_refused_unless_it_runs_over_item_and_span() -> None:
    index = SpanIndex(3)

    with pytest.raises(ValueError, match="a chart runs over item and span"):
        checked_scalar_chart("a chart", torch.zeros(index.num_spans), index)


def test_a_scalar_chart_is_refused_unless_there_is_one_cell_per_span() -> None:
    index = SpanIndex(3)

    with pytest.raises(ValueError, match="a chart of shape"):
        checked_scalar_chart("a chart", torch.zeros(1, index.num_spans - 1), index)


def test_split_scores_are_refused_unless_they_hold_log_weights() -> None:
    index = SpanIndex(3)

    with pytest.raises(TypeError, match="split scores are log weights"):
        checked_split_scores(
            torch.zeros(1, index.num_spans, index.max_splits, dtype=torch.long),
            index,
            1,
        )


def test_split_scores_are_refused_unless_they_run_over_item_span_and_split() -> None:
    index = SpanIndex(3)

    with pytest.raises(ValueError, match="rank 3"):
        checked_split_scores(torch.zeros(index.num_spans, index.max_splits), index, 1)


def test_split_scores_are_refused_unless_there_is_one_per_split_of_every_span() -> None:
    index = SpanIndex(3)

    with pytest.raises(ValueError, match="split scores of shape"):
        checked_split_scores(
            torch.zeros(1, index.num_spans - 1, index.max_splits), index, 1
        )


def test_split_scores_are_refused_unless_they_cover_the_batch_they_accompany() -> None:
    index = SpanIndex(3)

    with pytest.raises(ValueError, match="split scores of shape"):
        checked_split_scores(
            torch.zeros(2, index.num_spans, index.max_splits), index, 3
        )


def test_the_name_a_chart_is_refused_under_is_the_one_the_caller_gave() -> None:
    index = SpanIndex(3)

    with pytest.raises(TypeError, match="an outside chart holds log weights"):
        checked_scalar_chart(
            "an outside chart", torch.zeros(1, index.num_spans).long(), index
        )
