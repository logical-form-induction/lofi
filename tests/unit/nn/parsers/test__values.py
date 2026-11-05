import pytest
import torch

from lofi.deduce import SpanIndex
from lofi.nn.parsers._values import (
    BracketedTree,
    ChanneledChart,
    ScoredChart,
    TaggedTree,
)

WIDTH = 3


def charted(
    n: int, batch: int = 1
) -> tuple[SpanIndex, torch.Tensor, torch.Tensor, torch.Tensor]:
    index = SpanIndex(n)

    return (
        index,
        torch.zeros(batch, index.num_spans),
        torch.zeros(batch, index.num_spans, index.max_splits),
        torch.zeros(index.num_spans, batch, WIDTH),
    )


def test_a_chart_reports_the_three_tensors_it_was_built_from() -> None:
    index, inside, alpha, cells = charted(4, batch=2)

    chart = ChanneledChart(inside, alpha, cells, index)

    assert chart.index is index
    assert chart.inside is inside
    assert chart.alpha is alpha
    assert chart.cells is cells


def test_a_chart_reads_its_batch_and_its_width_off_the_tensors() -> None:
    index, inside, alpha, cells = charted(5, batch=7)

    chart = ChanneledChart(inside, alpha, cells, index)

    assert chart.batch == 7
    assert chart.width == WIDTH
    assert not hasattr(chart, "__dict__")


def test_a_chart_says_what_it_covers_when_printed() -> None:
    index, inside, alpha, cells = charted(4, batch=2)

    printed = repr(ChanneledChart(inside, alpha, cells, index))

    assert printed == f"ChanneledChart(n=4, batch=2, width={WIDTH})"


def test_a_one_token_input_carries_an_empty_posterior() -> None:
    index, inside, alpha, cells = charted(1)

    chart = ChanneledChart(inside, alpha, cells, index)

    assert chart.alpha.shape == (1, 1, 0)
    assert chart.cells.shape == (1, 1, WIDTH)


def test_an_integer_chart_is_refused() -> None:
    index, inside, alpha, cells = charted(3)

    with pytest.raises(TypeError, match="an inside chart accumulates"):
        ChanneledChart(inside.long(), alpha, cells, index)

    with pytest.raises(TypeError, match="a split posterior accumulates"):
        ChanneledChart(inside, alpha.long(), cells, index)

    with pytest.raises(TypeError, match="a vector chart accumulates"):
        ChanneledChart(inside, alpha, cells.long(), index)


def test_a_chart_of_the_wrong_rank_is_refused() -> None:
    index, inside, alpha, cells = charted(3)

    with pytest.raises(ValueError, match="rank 2"):
        ChanneledChart(inside[0], alpha, cells, index)

    with pytest.raises(ValueError, match="rank 3"):
        ChanneledChart(inside, alpha[0], cells, index)

    with pytest.raises(ValueError, match="rank 3"):
        ChanneledChart(inside, alpha, cells[0], index)


def test_a_chart_the_index_does_not_fit_is_refused() -> None:
    index, inside, alpha, cells = charted(3)

    with pytest.raises(ValueError, match="an inside chart has shape"):
        ChanneledChart(inside[:, :-1], alpha, cells, index)

    with pytest.raises(ValueError, match="a split posterior has shape"):
        ChanneledChart(inside, alpha[:, :-1], cells, index)

    with pytest.raises(ValueError, match="before its width axis"):
        ChanneledChart(inside, alpha, cells[:-1], index)


def test_charts_that_cover_different_batches_are_refused() -> None:
    index, inside, alpha, cells = charted(3, batch=2)

    with pytest.raises(ValueError, match="a split posterior has shape"):
        ChanneledChart(inside, alpha[:1], cells, index)

    with pytest.raises(ValueError, match="before its width axis"):
        ChanneledChart(inside, alpha, cells[:, :1], index)


def test_a_bracketing_is_identified_by_its_constituents_and_not_by_its_total() -> None:
    one = BracketedTree(((0, 2), (0, 3)), 1.5)
    same = BracketedTree(((0, 2), (0, 3)), -4.0)
    other = BracketedTree(((1, 3), (0, 3)), 1.5)

    assert one == same
    assert hash(one) == hash(same)
    assert one != other
    assert one != ((0, 2), (0, 3))
    assert len({one, same, other}) == 2
    assert repr(one) == "BracketedTree(((0, 2), (0, 3)), 1.5)"
    assert one.constituents == ((0, 2), (0, 3))
    assert one.total == pytest.approx(1.5)


def test_a_bracketing_holds_no_constituent_over_one_token() -> None:
    leaf = BracketedTree((), 0.25)

    assert leaf.constituents == ()
    assert leaf.total == pytest.approx(0.25)
    assert leaf == BracketedTree((), -1.0)


def leaf(start: int, category: int, tag: int, total: float) -> TaggedTree:
    return TaggedTree((start, start + 1), (category, tag), None, (), total)


def test_a_derivation_reports_the_span_the_state_and_the_rule_it_ends_with() -> None:
    left = leaf(0, 0, 2, 1.0)
    right = leaf(1, 0, 2, 2.0)
    joined = TaggedTree((0, 2), (1, 3), 5, (left, right), 4.0)

    assert joined.span == (0, 2)
    assert joined.category == 1
    assert joined.tag == 3
    assert joined.rule == 5
    assert joined.children == (left, right)
    assert joined.total == 4.0
    assert left.rule is None


def test_a_derivation_brackets_every_span_it_builds_from_two_premises() -> None:
    inner = TaggedTree((1, 3), (0, 0), 1, (leaf(1, 0, 0, 0.0), leaf(2, 0, 0, 0.0)), 0.0)
    raised = TaggedTree((1, 3), (1, 1), 0, (inner,), 0.0)
    whole = TaggedTree((0, 3), (1, 1), 1, (leaf(0, 1, 1, 0.0), raised), 0.0)

    assert whole.constituents() == ((0, 3), (1, 3))
    assert raised.constituents() == ((1, 3),)
    assert leaf(0, 0, 0, 0.0).constituents() == ()


def test_two_derivations_over_the_same_steps_are_one_value() -> None:
    def built(rule: int, total: float) -> TaggedTree:
        return TaggedTree(
            (0, 2), (1, 3), rule, (leaf(0, 0, 2, total), leaf(1, 0, 2, 0.0)), total
        )

    assert built(5, 1.0) == built(5, 9.0)
    assert hash(built(5, 1.0)) == hash(built(5, 9.0))
    assert built(5, 1.0) != built(6, 1.0)
    assert built(5, 1.0) != ((0, 2), (1, 3), 5)


def test_two_derivations_differing_only_in_a_premise_are_different() -> None:
    def built(tag: int) -> TaggedTree:
        return TaggedTree(
            (0, 2), (1, 3), 5, (leaf(0, 0, tag, 0.0), leaf(1, 0, 2, 0.0)), 0.0
        )

    assert built(2) != built(3)
    assert len({built(2), built(3)}) == 2


def test_a_derivation_refuses_reassignment_and_deletion() -> None:
    joined = TaggedTree((0, 2), (1, 3), 5, (leaf(0, 0, 2, 0.0),), 4.0)
    before = hash(joined)
    field = "_span"

    with pytest.raises(AttributeError, match=r"tagged derivation is immutable"):
        setattr(joined, field, (9, 9))

    with pytest.raises(AttributeError, match=r"tagged derivation is immutable"):
        delattr(joined, field)

    assert joined.span == (0, 2)
    assert hash(joined) == before


def test_a_derivation_says_what_it_ends_with_when_printed() -> None:
    joined = TaggedTree((0, 2), (1, 3), 5, (leaf(0, 0, 2, 0.0),), 4.0)

    assert repr(joined) == "TaggedTree((0, 2), (1, 3), 5, 1 premises, 4.0)"


def test_a_scored_chart_pairs_the_chart_with_the_split_scores_it_read() -> None:
    index, inside, alpha, cells = charted(3)
    chart = ChanneledChart(inside, alpha, cells, index)

    scored = ScoredChart(chart, alpha)

    assert scored.chart is chart
    assert scored.split_scores is alpha
    assert tuple(scored) == (chart, alpha)
