import math
from typing import TYPE_CHECKING

import pytest
import torch

from lofi.deduce import SpanIndex
from lofi.nn.parsers import _inside
from lofi.nn.parsers._chart import TensorChart
from lofi.nn.parsers._inside import InsideKernel
from lofi.nn.parsers._reduction import LogSumExpReduction, MaximumReduction

if TYPE_CHECKING:
    from collections.abc import Callable

NEG_INF = float("-inf")


def catalan(trees_over: int) -> int:
    internal = trees_over - 1

    return math.comb(2 * internal, internal) // (internal + 1)


def scored(
    index: SpanIndex, seed: int, batch: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    leaves = torch.randn(batch, index.n, generator=generator, dtype=torch.float64)
    splits = torch.randn(
        batch,
        index.num_spans,
        index.max_splits,
        generator=generator,
        dtype=torch.float64,
    )

    return (leaves, splits)


def tree_scores(
    index: SpanIndex, leaves: torch.Tensor, splits: torch.Tensor, i: int, j: int
) -> list[float]:
    if j - i == 1:
        return [leaves[i].item()]

    scores = []

    for k in range(i + 1, j):
        edge = splits[index.span_id(i, j), k - i - 1].item()
        scores += [
            left + right + edge
            for left in tree_scores(index, leaves, splits, i, k)
            for right in tree_scores(index, leaves, splits, k, j)
        ]

    return scores


def total_score(scores: list[float]) -> float:
    return math.log(sum(math.exp(score) for score in scores))


def zero_scores(index: SpanIndex) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.zeros(1, index.n, dtype=torch.float64),
        torch.zeros(1, index.num_spans, index.max_splits, dtype=torch.float64),
    )


def test_a_two_token_input_has_one_derivation_worth_its_three_scores() -> None:
    index = SpanIndex(2)
    leaves = torch.tensor([[0.5, 1.5]])
    splits = torch.zeros(1, index.num_spans, index.max_splits)
    splits[0, index.span_id(0, 2), 0] = 0.25

    partition = InsideKernel(LogSumExpReduction()).log_partition(leaves, splits, index)

    assert partition.item() == pytest.approx(0.5 + 1.5 + 0.25)


def test_a_three_token_input_sums_its_two_derivations() -> None:
    index = SpanIndex(3)
    leaves = torch.tensor([[0.5, 1.5, 2.5]], dtype=torch.float64)
    splits = torch.zeros(1, index.num_spans, index.max_splits, dtype=torch.float64)
    splits[0, index.span_id(0, 2), 0] = 0.25
    splits[0, index.span_id(1, 3), 0] = 0.75
    splits[0, index.span_id(0, 3), 0] = 1.25
    splits[0, index.span_id(0, 3), 1] = 1.75

    partition = InsideKernel(LogSumExpReduction()).log_partition(leaves, splits, index)

    cut_after_one = 0.5 + (1.5 + 2.5 + 0.75) + 1.25
    cut_after_two = (0.5 + 1.5 + 0.25) + 2.5 + 1.75

    assert partition.item() == pytest.approx(
        math.log(math.exp(cut_after_one) + math.exp(cut_after_two))
    )


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
def test_the_log_partition_matches_enumeration_over_every_binary_tree(n: int) -> None:
    index = SpanIndex(n)
    leaves, splits = scored(index, seed=n)

    partition = InsideKernel(LogSumExpReduction()).log_partition(leaves, splits, index)
    trees = tree_scores(index, leaves[0], splits[0], 0, n)

    assert len(trees) == catalan(n)
    assert partition.item() == pytest.approx(total_score(trees))


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
def test_viterbi_matches_the_best_tree_found_by_enumeration(n: int) -> None:
    index = SpanIndex(n)
    leaves, splits = scored(index, seed=n)

    partition = InsideKernel(MaximumReduction()).log_partition(leaves, splits, index)

    assert partition.item() == pytest.approx(
        max(tree_scores(index, leaves[0], splits[0], 0, n))
    )


def test_the_reduction_is_the_whole_difference_between_the_two_sweeps() -> None:
    index = SpanIndex(5)
    leaves, splits = scored(index, seed=11)

    summed = InsideKernel(LogSumExpReduction()).log_partition(leaves, splits, index)
    best = InsideKernel(MaximumReduction()).log_partition(leaves, splits, index)

    assert best.item() < summed.item()


def test_every_span_of_the_chart_matches_enumeration_over_that_span() -> None:
    index = SpanIndex(5)
    leaves, splits = scored(index, seed=3)

    chart = InsideKernel(LogSumExpReduction()).inside(leaves, splits, index)

    assert chart.shape == (1, index.num_spans)

    for i, j in index.spans_by_length():
        trees = tree_scores(index, leaves[0], splits[0], i, j)
        assert chart[0, index.span_id(i, j)].item() == pytest.approx(total_score(trees))


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7, 8])
def test_uniform_scores_count_the_binary_trees_of_the_input(n: int) -> None:
    index = SpanIndex(n)
    leaves, splits = zero_scores(index)

    partition = InsideKernel(LogSumExpReduction()).log_partition(leaves, splits, index)

    assert partition.exp().item() == pytest.approx(catalan(n))


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7, 8])
def test_uniform_scores_leave_viterbi_with_one_derivation(n: int) -> None:
    index = SpanIndex(n)
    leaves, splits = zero_scores(index)

    partition = InsideKernel(MaximumReduction()).log_partition(leaves, splits, index)

    assert partition.exp().item() == pytest.approx(1.0)


def test_the_gradient_of_the_log_partition_is_an_expected_count() -> None:
    index = SpanIndex(5)
    leaves, splits = scored(index, seed=7)
    leaves.requires_grad_()
    splits.requires_grad_()

    InsideKernel(LogSumExpReduction()).log_partition(
        leaves, splits, index
    ).sum().backward()

    assert leaves.grad is not None
    assert splits.grad is not None
    assert leaves.grad.reshape(-1).tolist() == pytest.approx([1.0] * index.n)
    assert splits.grad.sum().item() == pytest.approx(index.n - 1)


def test_the_gradient_of_the_viterbi_weight_selects_the_best_derivation() -> None:
    index = SpanIndex(5)
    leaves, splits = scored(index, seed=7)
    leaves.requires_grad_()
    splits.requires_grad_()

    InsideKernel(MaximumReduction()).log_partition(
        leaves, splits, index
    ).sum().backward()

    assert leaves.grad is not None
    assert splits.grad is not None
    assert leaves.grad.reshape(-1).tolist() == pytest.approx([1.0] * index.n)
    assert sorted(set(splits.grad.reshape(-1).tolist())) == [0.0, 1.0]
    assert splits.grad.sum().item() == pytest.approx(index.n - 1)


def test_a_gradient_reaches_whatever_produced_the_scores() -> None:
    index = SpanIndex(4)
    source = torch.zeros(1, index.num_spans, index.max_splits, requires_grad=True)
    leaves = torch.zeros(1, index.n)

    InsideKernel(LogSumExpReduction()).log_partition(
        leaves, source * 2.0, index
    ).backward()

    assert source.grad is not None
    assert source.grad.abs().sum().item() > 0.0


@pytest.mark.parametrize("reduction", [LogSumExpReduction(), MaximumReduction()])
def test_an_impossible_token_leaves_the_partition_at_zero_without_a_nan(
    reduction: LogSumExpReduction | MaximumReduction,
) -> None:
    index = SpanIndex(3)
    leaves = torch.tensor([[0.0, NEG_INF, 0.0]], requires_grad=True)
    splits = torch.zeros(1, index.num_spans, index.max_splits, requires_grad=True)

    partition = InsideKernel(reduction).log_partition(leaves, splits, index)
    partition.sum().backward()

    assert partition.item() == NEG_INF
    assert leaves.grad is not None
    assert splits.grad is not None
    assert bool(torch.isfinite(leaves.grad).all())
    assert bool(torch.isfinite(splits.grad).all())


def test_one_impossible_item_leaves_the_rest_of_the_batch_alone() -> None:
    index = SpanIndex(3)
    leaves, splits = scored(index, seed=5, batch=2)
    leaves[0, 1] = NEG_INF

    partition = InsideKernel(LogSumExpReduction()).log_partition(leaves, splits, index)

    assert partition[0].item() == NEG_INF
    assert partition[1].item() == pytest.approx(
        total_score(tree_scores(index, leaves[1], splits[1], 0, 3))
    )


def test_a_shorter_item_reads_its_goal_from_the_span_it_actually_covers() -> None:
    index = SpanIndex(4)
    leaves, splits = scored(index, seed=13, batch=2)
    kernel = InsideKernel(LogSumExpReduction())

    ragged = kernel.log_partition(leaves, splits, index, lengths=torch.tensor([3, 4]))

    assert ragged[0].item() == pytest.approx(
        total_score(tree_scores(index, leaves[0], splits[0], 0, 3))
    )
    assert ragged[1].item() == pytest.approx(
        total_score(tree_scores(index, leaves[1], splits[1], 0, 4))
    )


def test_naming_every_item_its_full_length_is_the_default() -> None:
    index = SpanIndex(4)
    leaves, splits = scored(index, seed=17, batch=3)
    kernel = InsideKernel(LogSumExpReduction())

    assert torch.equal(
        kernel.log_partition(leaves, splits, index),
        kernel.log_partition(leaves, splits, index, lengths=torch.tensor([4, 4, 4])),
    )


def test_the_kernel_reports_its_reduction_and_holds_nothing_else() -> None:
    reduction = MaximumReduction()
    kernel = InsideKernel(reduction)

    assert kernel.reduction is reduction
    assert not hasattr(kernel, "__dict__")


def test_the_kernel_publishes_a_sweep_and_nothing_grammatical() -> None:
    published = {name for name in dir(InsideKernel) if not name.startswith("_")}

    assert published == {"inside", "join", "log_partition", "reduction"}


def test_an_input_with_no_spans_is_refused() -> None:
    with pytest.raises(ValueError, match="length at least one"):
        InsideKernel(LogSumExpReduction()).log_partition(
            torch.zeros(1, 0), torch.zeros(1, 0, 0), SpanIndex(0)
        )


def test_scores_of_the_wrong_rank_are_refused() -> None:
    index = SpanIndex(3)
    leaves, splits = scored(index, seed=1)
    kernel = InsideKernel(LogSumExpReduction())

    with pytest.raises(ValueError, match="rank 2"):
        kernel.log_partition(leaves[0], splits, index)

    with pytest.raises(ValueError, match="rank 3"):
        kernel.log_partition(leaves, splits[0], index)


def test_integer_scores_are_refused() -> None:
    index = SpanIndex(3)
    kernel = InsideKernel(LogSumExpReduction())
    leaves = torch.zeros(1, index.n)
    splits = torch.zeros(1, index.num_spans, index.max_splits)

    with pytest.raises(TypeError, match="leaf scores are log weights"):
        kernel.log_partition(leaves.long(), splits, index)

    with pytest.raises(TypeError, match="split scores are log weights"):
        kernel.log_partition(leaves, splits.long(), index)


def test_scores_the_index_does_not_fit_are_refused() -> None:
    index = SpanIndex(3)
    kernel = InsideKernel(LogSumExpReduction())
    leaves = torch.zeros(1, index.n)
    splits = torch.zeros(1, index.num_spans, index.max_splits)

    with pytest.raises(ValueError, match="leaf scores of shape"):
        kernel.log_partition(torch.zeros(1, 4), splits, index)

    with pytest.raises(ValueError, match="split scores of shape"):
        kernel.log_partition(leaves, torch.zeros(1, 5, 2), index)


def test_lengths_that_are_not_whole_numbers_of_tokens_are_refused() -> None:
    index = SpanIndex(3)
    leaves, splits = scored(index, seed=1)
    kernel = InsideKernel(LogSumExpReduction())

    with pytest.raises(TypeError, match="whole number of tokens"):
        kernel.log_partition(leaves, splits, index, lengths=torch.tensor([2.5]))

    with pytest.raises(TypeError, match="not a boolean"):
        kernel.log_partition(leaves, splits, index, lengths=torch.tensor([True]))


def test_lengths_that_are_not_one_per_item_are_refused() -> None:
    index = SpanIndex(3)
    leaves, splits = scored(index, seed=1)
    kernel = InsideKernel(LogSumExpReduction())

    with pytest.raises(ValueError, match="lengths of shape"):
        kernel.log_partition(leaves, splits, index, lengths=torch.tensor([1, 2]))

    with pytest.raises(ValueError, match="lengths of shape"):
        kernel.log_partition(leaves, splits, index, lengths=torch.tensor([[3]]))


def test_the_debug_checks_reject_a_length_the_index_does_not_cover(
    reload_with_checks: Callable[..., None],
) -> None:
    reload_with_checks(_inside, on=True)
    index = SpanIndex(3)
    leaves, splits = scored(index, seed=1)
    kernel = _inside.InsideKernel(LogSumExpReduction())

    with pytest.raises(ValueError, match=r"outside range\(1, 4\)"):
        kernel.log_partition(leaves, splits, index, lengths=torch.tensor([9]))

    with pytest.raises(ValueError, match=r"outside range\(1, 4\)"):
        kernel.log_partition(leaves, splits, index, lengths=torch.tensor([0]))


def test_without_the_debug_checks_a_length_is_taken_as_given(
    reload_with_checks: Callable[..., None],
) -> None:
    reload_with_checks(_inside, on=False)
    index = SpanIndex(3)
    leaves, splits = scored(index, seed=1)
    kernel = _inside.InsideKernel(LogSumExpReduction())

    with pytest.raises(IndexError):
        kernel.log_partition(leaves, splits, index, lengths=torch.tensor([9]))


def test_the_gathers_hold_the_index_beside_its_ids_and_its_split_mask() -> None:
    index = SpanIndex(4)
    gathers = _inside.ChildGathers(index, torch.zeros(1, dtype=torch.float64))

    assert gathers.index is index
    assert not hasattr(gathers, "__dict__")

    for width in range(1, 5):
        start, stop = gathers.bounds(width)

        assert (start, stop) == (
            index.span_id(0, width),
            index.span_id(0, width) + 5 - width,
        )
        assert gathers.left(width).tolist() == [
            list(row) for row in index.left_child_ids[start:stop]
        ]
        assert gathers.right(width).tolist() == [
            list(row) for row in index.right_child_ids[start:stop]
        ]
        assert gathers.real(width).shape == (1, stop - start, index.max_splits)
        assert gathers.real(width)[0].tolist() == [
            list(row) for row in index.split_mask[start:stop]
        ]


@pytest.mark.parametrize("n", [2, 3, 5])
def test_joining_one_width_and_reducing_it_is_that_slab_of_the_sweep(n: int) -> None:
    index = SpanIndex(n)
    leaves, splits = scored(index, seed=n + 40)
    kernel = InsideKernel(LogSumExpReduction())
    chart = kernel.inside(leaves, splits, index)
    packed = TensorChart(index, chart, batch=1, width=1)
    gathers = _inside.ChildGathers(index, leaves)

    for i, j in index.spans_by_length():
        packed.write(i, j, chart[:, index.span_id(i, j)].unsqueeze(-1))

    for width in range(2, n + 1):
        start, stop = gathers.bounds(width)
        joined = kernel.join(packed, splits[:, start:stop], width=width)

        assert joined.shape == (1, stop - start, index.max_splits)
        assert torch.allclose(
            LogSumExpReduction().reduce(joined, gathers.real(width), dim=-1),
            chart[:, start:stop],
        )
