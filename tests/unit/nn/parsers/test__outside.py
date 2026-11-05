import math
from typing import Literal

import pytest
import torch

from lofi.deduce import LogInside as SymbolicLogInside
from lofi.deduce import SpanIndex
from lofi.nn.parsers._channel import log_split_posterior
from lofi.nn.parsers._inside import InsideKernel
from lofi.nn.parsers._outside import (
    OutsideKernel,
    VectorOutsideKernel,
    _parent_slabs,
    log_parent_posterior,
    parent_contexts,
)
from lofi.nn.parsers._reduction import LogSumExpReduction, MaximumReduction
from lofi.nn.parsers.functions import MlpSpanFunction

NEG_INF = float("-inf")

#: Absolute tolerance for log weights.
TOLERANCE = 1e-6

#: Minimum measurable log-weight change.
SEPARATION = 0.05


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


def vector_outside(
    *,
    weighting: Literal["reuse", "contextual"] = "contextual",
    root: Literal["zero", "learned"] = "learned",
    seed: int = 5,
) -> VectorOutsideKernel:
    """Return a deterministic vector outside pass for a three-wide fixture."""
    torch.manual_seed(seed)
    left = MlpSpanFunction((3, 3), 3, layers=1, nonlinearity="tanh")
    right = MlpSpanFunction((3, 3), 3, layers=1, nonlinearity="tanh")
    return VectorOutsideKernel(
        left,
        right,
        width=3,
        weighting=weighting,
        root=root,
        root_init_gain=1.0,
    ).to(torch.float64)


def symbolic_total(scores: list[float]) -> float:
    reduction = SymbolicLogInside()
    total = reduction.zero

    for score in scores:
        total = reduction.plus(total, score)

    return total


def derivations(
    index: SpanIndex, leaves: torch.Tensor, splits: torch.Tensor, i: int, j: int
) -> list[tuple[float, frozenset[tuple[int, int]]]]:
    if j - i == 1:
        return [(leaves[i].item(), frozenset({(i, j)}))]

    found = []

    for k in index.splits(i, j):
        edge = splits[index.span_id(i, j), k - i - 1].item()
        found += [
            (left + right + edge, frozenset({(i, j)}) | held | rest)
            for left, held in derivations(index, leaves, splits, i, k)
            for right, rest in derivations(index, leaves, splits, k, j)
        ]

    return found


def reference_outside(
    index: SpanIndex, leaves: torch.Tensor, splits: torch.Tensor, i: int, j: int
) -> float:
    whole = derivations(index, leaves, splits, 0, index.n)
    holding = symbolic_total([score for score, spans in whole if (i, j) in spans])
    inside = symbolic_total(
        [score for score, _ in derivations(index, leaves, splits, i, j)]
    )

    return holding - inside


def leaky_outside(
    index: SpanIndex, inside: torch.Tensor, splits: torch.Tensor
) -> dict[tuple[int, int], float]:
    alpha = log_split_posterior(inside, splits, index)[0]
    found: dict[tuple[int, int], list[float]] = {(0, index.n): [0.0]}

    for i, j in index.spans_by_length_desc(min_length=2):
        parent = symbolic_total(found[i, j])

        for column, k in enumerate(index.splits(i, j)):
            weight = parent + alpha[index.span_id(i, j), column].item()
            found.setdefault((i, k), []).append(weight)
            found.setdefault((k, j), []).append(weight)

    return {span: symbolic_total(scores) for span, scores in found.items()}


def perturbed(
    index: SpanIndex,
    leaves: torch.Tensor,
    splits: torch.Tensor,
    span: tuple[int, int],
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    i, j = span
    generator = torch.Generator().manual_seed(seed)
    held = [
        index.span_id(a, b) for a, b in index.spans_by_length() if i <= a and b <= j
    ]
    shifted_leaves = leaves.clone()
    shifted_splits = splits.clone()
    shifted_leaves[:, i:j] += torch.randn(
        leaves.shape[0], j - i, generator=generator, dtype=leaves.dtype
    )
    shifted_splits[:, held, :] += torch.randn(
        splits.shape[0],
        len(held),
        index.max_splits,
        generator=generator,
        dtype=splits.dtype,
    )

    return (shifted_leaves, shifted_splits)


def moves(index: SpanIndex, seed: int) -> dict[tuple[int, int], tuple[float, float]]:
    leaves, splits = scored(index, seed=seed)
    kernel = InsideKernel(LogSumExpReduction())
    sweep = OutsideKernel(LogSumExpReduction())
    inside = kernel.inside(leaves, splits, index)
    before = sweep.outside(inside, splits, index)
    leaky_before = leaky_outside(index, inside, splits)
    found = {}

    for span in index.spans_by_length():
        other_leaves, other_splits = perturbed(
            index, leaves, splits, span, seed=seed + 1
        )
        other_inside = kernel.inside(other_leaves, other_splits, index)
        after = sweep.outside(other_inside, other_splits, index)
        leaky_after = leaky_outside(index, other_inside, other_splits)
        cell = index.span_id(*span)
        found[span] = (
            abs(after[0, cell].item() - before[0, cell].item()),
            abs(leaky_after[span] - leaky_before[span]),
        )

    return found


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_the_outside_chart_reproduces_the_rule_marginals_of_the_inside_pass(
    n: int,
) -> None:
    index = SpanIndex(n)
    batch = 2
    leaves, splits = scored(index, seed=n, batch=batch)
    splits.requires_grad_()

    kernel = InsideKernel(LogSumExpReduction())
    kernel.log_partition(leaves, splits, index).sum().backward()
    scores = splits.detach()
    inside = kernel.inside(leaves, scores, index)
    log_z = kernel.log_partition(leaves, scores, index)
    outside = OutsideKernel(LogSumExpReduction()).outside(inside, scores, index)

    assert splits.grad is not None

    for item in range(batch):
        for i, j in index.spans_by_length(min_length=2):
            cell = index.span_id(i, j)

            for column, k in enumerate(index.splits(i, j)):
                weight = (
                    outside[item, cell].item()
                    + scores[item, cell, column].item()
                    + inside[item, index.span_id(i, k)].item()
                    + inside[item, index.span_id(k, j)].item()
                    - log_z[item].item()
                )

                assert math.exp(weight) == pytest.approx(
                    splits.grad[item, cell, column].item(), abs=TOLERANCE
                )


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
def test_every_token_is_a_constituent_of_every_derivation(n: int) -> None:
    index = SpanIndex(n)
    leaves, splits = scored(index, seed=n + 40)

    kernel = InsideKernel(LogSumExpReduction())
    inside = kernel.inside(leaves, splits, index)
    log_z = kernel.log_partition(leaves, splits, index)
    outside = OutsideKernel(LogSumExpReduction()).outside(inside, splits, index)

    for i, j in index.leaves():
        cell = index.span_id(i, j)
        weight = inside[0, cell].item() + outside[0, cell].item() - log_z[0].item()

        assert math.exp(weight) == pytest.approx(1.0, abs=TOLERANCE)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
def test_the_outside_of_a_token_is_the_gradient_of_the_partition_at_its_score(
    n: int,
) -> None:
    index = SpanIndex(n)
    batch = 2
    leaves, splits = scored(index, seed=n + 60, batch=batch)
    leaves.requires_grad_()

    kernel = InsideKernel(LogSumExpReduction())
    kernel.log_partition(leaves, splits, index).sum().backward()
    scores = leaves.detach()
    inside = kernel.inside(scores, splits, index)
    log_z = kernel.log_partition(scores, splits, index)
    outside = OutsideKernel(LogSumExpReduction()).outside(inside, splits, index)

    assert leaves.grad is not None

    for item in range(batch):
        for i, j in index.leaves():
            weight = (
                outside[item, index.span_id(i, j)].item()
                + scores[item, i].item()
                - log_z[item].item()
            )

            assert math.exp(weight) == pytest.approx(
                leaves.grad[item, i].item(), abs=TOLERANCE
            )


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_the_outside_chart_matches_enumeration_over_every_binary_tree(n: int) -> None:
    index = SpanIndex(n)
    leaves, splits = scored(index, seed=n + 80)

    inside = InsideKernel(LogSumExpReduction()).inside(leaves, splits, index)
    outside = OutsideKernel(LogSumExpReduction()).outside(inside, splits, index)

    for i, j in index.spans_by_length():
        assert outside[0, index.span_id(i, j)].item() == pytest.approx(
            reference_outside(index, leaves[0], splits[0], i, j), abs=TOLERANCE
        )


@pytest.mark.parametrize("n", [3, 4, 5])
def test_the_outside_of_a_span_ignores_a_perturbation_of_that_span(n: int) -> None:
    found = moves(SpanIndex(n), seed=n + 120)

    assert max(shift for shift, _ in found.values()) < TOLERANCE


@pytest.mark.parametrize("n", [3, 4, 5])
def test_reusing_the_inside_posterior_still_leaks_the_span_into_its_own_outside(
    n: int,
) -> None:
    found = moves(SpanIndex(n), seed=n + 120)
    dividing = {span for span in found if span != (0, n) and span[1] - span[0] > 1}
    leaked = {span for span, (_, shift) in found.items() if shift > TOLERANCE}

    assert leaked == dividing
    assert max(shift for _, shift in found.values()) > SEPARATION


def test_the_goal_span_weighs_one_and_is_the_only_seeded_cell() -> None:
    index = SpanIndex(4)
    leaves, splits = scored(index, seed=11)

    inside = InsideKernel(LogSumExpReduction()).inside(leaves, splits, index)
    outside = OutsideKernel(LogSumExpReduction()).outside(inside, splits, index)

    assert outside[0, index.span_id(0, 4)].item() == 0.0
    assert all(
        outside[0, index.span_id(i, j)].item() != 0.0
        for i, j in index.spans_by_length()
        if (i, j) != (0, 4)
    )


def test_a_ragged_batch_reads_each_item_from_its_own_goal_span() -> None:
    index = SpanIndex(5)
    batch = 3
    lengths = torch.tensor([2, 4, 5])
    leaves, splits = scored(index, seed=13, batch=batch)
    splits.requires_grad_()

    kernel = InsideKernel(LogSumExpReduction())
    kernel.log_partition(leaves, splits, index, lengths=lengths).sum().backward()
    scores = splits.detach()
    inside = kernel.inside(leaves, scores, index)
    log_z = kernel.log_partition(leaves, scores, index, lengths=lengths)
    outside = OutsideKernel(LogSumExpReduction()).outside(
        inside, scores, index, lengths=lengths
    )

    assert splits.grad is not None

    for item in range(batch):
        length = int(lengths[item])

        assert outside[item, index.span_id(0, length)].item() == 0.0

        for i, j in index.spans_by_length():
            cell = index.span_id(i, j)

            if j > length:
                assert outside[item, cell].item() == NEG_INF

            for column, k in enumerate(index.splits(i, j)):
                weight = (
                    outside[item, cell].item()
                    + scores[item, cell, column].item()
                    + inside[item, index.span_id(i, k)].item()
                    + inside[item, index.span_id(k, j)].item()
                    - log_z[item].item()
                )

                assert math.exp(weight) == pytest.approx(
                    splits.grad[item, cell, column].item(), abs=TOLERANCE
                )


def test_one_item_of_a_batch_does_not_reach_another() -> None:
    index = SpanIndex(4)
    leaves, splits = scored(index, seed=17, batch=2)
    kernel = InsideKernel(LogSumExpReduction())
    sweep = OutsideKernel(LogSumExpReduction())

    alone = sweep.outside(
        kernel.inside(leaves[:1], splits[:1], index), splits[:1], index
    )
    together = sweep.outside(kernel.inside(leaves, splits, index), splits, index)

    assert together[0].tolist() == pytest.approx(alone[0].tolist(), abs=TOLERANCE)


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_the_best_derivation_is_the_spans_whose_two_weights_reach_the_goal(
    n: int,
) -> None:
    index = SpanIndex(n)
    leaves, splits = scored(index, seed=n + 200)
    splits.requires_grad_()

    kernel = InsideKernel(MaximumReduction())
    kernel.log_partition(leaves, splits, index).sum().backward()
    scores = splits.detach()
    inside = kernel.inside(leaves, scores, index)
    best = kernel.log_partition(leaves, scores, index)
    outside = OutsideKernel(MaximumReduction()).outside(inside, scores, index)

    assert splits.grad is not None

    reached = {
        (i, j)
        for i, j in index.spans_by_length(min_length=2)
        if inside[0, index.span_id(i, j)].item()
        + outside[0, index.span_id(i, j)].item()
        > best[0].item() - TOLERANCE
    }
    chosen = {
        (i, j)
        for i, j in index.spans_by_length(min_length=2)
        if splits.grad[0, index.span_id(i, j)].sum().item() > 0.0
    }

    assert reached == chosen
    assert len(reached) == n - 1


def test_a_chart_nothing_derives_has_no_outside_and_no_nan_gradient() -> None:
    index = SpanIndex(3)
    leaves = torch.zeros(1, index.n, dtype=torch.float64)
    splits = torch.full(
        (1, index.num_spans, index.max_splits),
        NEG_INF,
        dtype=torch.float64,
        requires_grad=True,
    )

    inside = InsideKernel(LogSumExpReduction()).inside(leaves, splits, index)
    outside = OutsideKernel(LogSumExpReduction()).outside(inside, splits, index)
    outside.exp().sum().backward()

    assert outside[0, index.span_id(0, 3)].item() == 0.0
    assert outside[0, index.span_id(0, 1)].item() == NEG_INF
    assert splits.grad is not None
    assert bool(torch.isfinite(splits.grad).all())


def test_a_one_token_input_is_its_own_goal() -> None:
    index = SpanIndex(1)
    leaves, splits = scored(index, seed=3)

    inside = InsideKernel(LogSumExpReduction()).inside(leaves, splits, index)
    outside = OutsideKernel(LogSumExpReduction()).outside(inside, splits, index)

    assert outside.tolist() == [[0.0]]


def test_the_kernel_publishes_the_reduction_it_uses() -> None:
    reduction = MaximumReduction()

    assert OutsideKernel(reduction).reduction is reduction


def test_the_outside_sweep_refuses_a_chart_or_scores_it_cannot_read() -> None:
    index = SpanIndex(3)
    leaves, splits = scored(index, seed=5)
    inside = InsideKernel(LogSumExpReduction()).inside(leaves, splits, index)
    sweep = OutsideKernel(LogSumExpReduction())

    with pytest.raises(TypeError, match="an inside chart holds log weights"):
        sweep.outside(inside.long(), splits, index)

    with pytest.raises(TypeError, match="split scores are log weights"):
        sweep.outside(inside, splits.long(), index)

    with pytest.raises(ValueError, match="rank 2"):
        sweep.outside(inside[0], splits, index)

    with pytest.raises(ValueError, match="rank 2"):
        sweep.outside(inside.unsqueeze(0), splits, index)

    with pytest.raises(ValueError, match="rank 3"):
        sweep.outside(inside, splits[0], index)

    with pytest.raises(ValueError, match="rank 3"):
        sweep.outside(inside, splits.unsqueeze(0), index)

    with pytest.raises(ValueError, match="an inside chart of shape"):
        sweep.outside(inside[:, :-1], splits, index)

    with pytest.raises(ValueError, match="split scores of shape"):
        sweep.outside(inside, splits[:, :-1], index)


def test_the_outside_sweep_refuses_lengths_it_cannot_read() -> None:
    index = SpanIndex(3)
    leaves, splits = scored(index, seed=7)
    inside = InsideKernel(LogSumExpReduction()).inside(leaves, splits, index)
    sweep = OutsideKernel(LogSumExpReduction())

    with pytest.raises(TypeError, match="whole number of tokens"):
        sweep.outside(inside, splits, index, lengths=torch.tensor([2.5]))

    with pytest.raises(ValueError, match="lengths of shape"):
        sweep.outside(inside, splits, index, lengths=torch.tensor([1, 2]))


def test_an_input_of_length_zero_has_no_spans_to_sweep() -> None:
    index = SpanIndex(0)
    empty = torch.zeros(1, 0, dtype=torch.float64)

    with pytest.raises(ValueError, match="length at least one"):
        OutsideKernel(LogSumExpReduction()).outside(
            empty, torch.zeros(1, 0, 0, dtype=torch.float64), index
        )


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_the_parent_tables_are_the_transpose_of_the_index_child_tables(n: int) -> None:
    index = SpanIndex(n)
    found = set()

    for width, (parents, siblings, columns) in enumerate(
        _parent_slabs(index, torch.zeros(1)), start=1
    ):
        assert tuple(parents.shape) == (n - width + 1, n - width)
        assert tuple(siblings.shape) == (n - width + 1, n - width)
        assert tuple(columns.shape) == (n - width + 1, n - width)

        for row in range(n - width + 1):
            ids = parents[row].tolist()

            assert ids == sorted(ids)

            found |= {
                (
                    parent,
                    int(columns[row, column]),
                    index.span_id(0, width) + row,
                    int(siblings[row, column]),
                )
                for column, parent in enumerate(ids)
            }

    wanted = {
        pair
        for parent in range(index.num_spans)
        for column, real in enumerate(index.split_mask[parent])
        if real
        for pair in (
            (
                parent,
                column,
                index.left_child_ids[parent][column],
                index.right_child_ids[parent][column],
            ),
            (
                parent,
                column,
                index.right_child_ids[parent][column],
                index.left_child_ids[parent][column],
            ),
        )
    }

    assert found == wanted


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_the_published_contexts_are_the_transpose_of_the_index_child_tables(
    n: int,
) -> None:
    index = SpanIndex(n)
    contexts = parent_contexts(index)
    found = {
        (parent, contexts.columns[child][t], child, contexts.siblings[child][t])
        for child in range(index.num_spans)
        for t, parent in enumerate(contexts.parents[child])
    }
    wanted = {
        pair
        for parent in range(index.num_spans)
        for column, real in enumerate(index.split_mask[parent])
        if real
        for pair in (
            (
                parent,
                column,
                index.left_child_ids[parent][column],
                index.right_child_ids[parent][column],
            ),
            (
                parent,
                column,
                index.right_child_ids[parent][column],
                index.left_child_ids[parent][column],
            ),
        )
    }

    assert found == wanted

    for i, j in index.spans_by_length():
        child = index.span_id(i, j)
        ids = list(contexts.parents[child])

        assert len(ids) == n - (j - i)
        assert ids == sorted(ids)
        assert len(contexts.siblings[child]) == len(ids)
        assert len(contexts.columns[child]) == len(ids)


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_a_published_context_names_the_column_of_the_parent_posterior(n: int) -> None:
    index = SpanIndex(n)
    leaves, splits = scored(index, seed=n + 420)
    contexts = parent_contexts(index)

    inside = InsideKernel(LogSumExpReduction()).inside(leaves, splits, index)
    outside = OutsideKernel(LogSumExpReduction()).outside(inside, splits, index)
    posterior = log_parent_posterior(inside, outside, splits, index)

    for i, j in index.spans_by_length():
        child = index.span_id(i, j)
        weights = [
            outside[0, parent].item()
            + splits[0, parent, contexts.columns[child][t]].item()
            + inside[0, contexts.siblings[child][t]].item()
            for t, parent in enumerate(contexts.parents[child])
        ]
        total = symbolic_total(weights)

        assert posterior[0, child, : len(weights)].tolist() == pytest.approx(
            [weight - total for weight in weights], abs=TOLERANCE
        )

        for t, parent in enumerate(contexts.parents[child]):
            column = contexts.columns[child][t]
            sibling = contexts.siblings[child][t]
            side = (
                index.left_child_ids[parent][column],
                index.right_child_ids[parent][column],
            )

            assert side in {(child, sibling), (sibling, child)}


def test_the_parent_posterior_of_a_chart_nothing_derives_is_a_row_of_zeros() -> None:
    index = SpanIndex(3)
    leaves = torch.zeros(1, index.n, dtype=torch.float64)
    splits = torch.full(
        (1, index.num_spans, index.max_splits),
        NEG_INF,
        dtype=torch.float64,
        requires_grad=True,
    )

    inside = InsideKernel(LogSumExpReduction()).inside(leaves, splits, index)
    outside = OutsideKernel(LogSumExpReduction()).outside(inside, splits, index)
    posterior = log_parent_posterior(inside, outside, splits, index)

    posterior.exp().sum().backward()

    assert posterior.exp().sum().item() == 0.0
    assert splits.grad is not None
    assert bool(torch.isfinite(splits.grad).all())


def test_a_ragged_batch_needs_no_lengths_in_the_parent_posterior() -> None:
    index = SpanIndex(5)
    batch = 3
    lengths = torch.tensor([2, 4, 5])
    leaves, splits = scored(index, seed=19, batch=batch)

    inside = InsideKernel(LogSumExpReduction()).inside(leaves, splits, index)
    outside = OutsideKernel(LogSumExpReduction()).outside(
        inside, splits, index, lengths=lengths
    )
    posterior = log_parent_posterior(inside, outside, splits, index).exp()

    for item in range(batch):
        length = int(lengths[item])

        for i, j in index.spans_by_length():
            row = posterior[item, index.span_id(i, j)]
            real = index.n - (j - i)
            reached = j <= length and (i, j) != (0, length)

            assert row[:real].sum().item() == pytest.approx(
                1.0 if reached else 0.0, abs=TOLERANCE
            )
            assert row[real:].tolist() == [0.0] * (index.n - 1 - real)


def test_the_parent_posterior_has_no_span_to_read_at_length_zero() -> None:
    index = SpanIndex(0)
    empty = torch.zeros(1, 0, dtype=torch.float64)
    scores = torch.zeros(1, 0, 0, dtype=torch.float64)

    with pytest.raises(ValueError, match="length at least one"):
        log_parent_posterior(empty, empty, scores, index)


def test_the_parent_posterior_refuses_two_charts_of_different_batches() -> None:
    index = SpanIndex(3)
    leaves, splits = scored(index, seed=31, batch=2)
    inside = InsideKernel(LogSumExpReduction()).inside(leaves, splits, index)
    outside = OutsideKernel(LogSumExpReduction()).outside(inside, splits, index)

    with pytest.raises(ValueError, match="an outside chart of the same"):
        log_parent_posterior(inside, outside[:1], splits, index)


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_the_parent_posterior_sums_to_one_over_the_contexts_of_every_span(
    n: int,
) -> None:
    index = SpanIndex(n)
    leaves, splits = scored(index, seed=n + 300)

    inside = InsideKernel(LogSumExpReduction()).inside(leaves, splits, index)
    outside = OutsideKernel(LogSumExpReduction()).outside(inside, splits, index)
    posterior = log_parent_posterior(inside, outside, splits, index).exp()

    for i, j in index.spans_by_length():
        row = posterior[0, index.span_id(i, j)]
        real = index.n - (j - i)

        assert row[:real].sum().item() == pytest.approx(
            0.0 if (i, j) == (0, n) else 1.0, abs=TOLERANCE
        )
        assert row[real:].tolist() == [0.0] * (index.n - 1 - real)


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_the_parent_posterior_is_the_outside_weight_read_the_other_way(n: int) -> None:
    index = SpanIndex(n)
    leaves, splits = scored(index, seed=n + 340)
    splits.requires_grad_()

    kernel = InsideKernel(LogSumExpReduction())
    kernel.log_partition(leaves, splits, index).sum().backward()
    scores = splits.detach()
    inside = kernel.inside(leaves, scores, index)
    log_z = kernel.log_partition(leaves, scores, index)
    outside = OutsideKernel(LogSumExpReduction()).outside(inside, scores, index)
    posterior = log_parent_posterior(inside, outside, scores, index).exp()

    assert splits.grad is not None

    for i, j in index.spans_by_length():
        cell = index.span_id(i, j)
        mu = math.exp(
            inside[0, cell].item() + outside[0, cell].item() - log_z[0].item()
        )
        contexts = sorted(
            [(index.span_id(i, m), j - i - 1) for m in range(j + 1, n + 1)]
            + [(index.span_id(m, j), i - m - 1) for m in range(i)]
        )

        assert posterior[0, cell, : n - (j - i)].tolist() == pytest.approx(
            [splits.grad[0, parent, column].item() / mu for parent, column in contexts],
            abs=TOLERANCE,
        )


@pytest.mark.parametrize("n", [3, 4, 5])
def test_the_parent_posterior_of_a_span_ignores_a_perturbation_of_that_span(
    n: int,
) -> None:
    index = SpanIndex(n)
    leaves, splits = scored(index, seed=n + 380)
    kernel = InsideKernel(LogSumExpReduction())
    sweep = OutsideKernel(LogSumExpReduction())
    inside = kernel.inside(leaves, splits, index)
    before = log_parent_posterior(
        inside, sweep.outside(inside, splits, index), splits, index
    )

    for span in index.spans_by_length():
        other_leaves, other_splits = perturbed(index, leaves, splits, span, seed=n)
        other_inside = kernel.inside(other_leaves, other_splits, index)
        after = log_parent_posterior(
            other_inside,
            sweep.outside(other_inside, other_splits, index),
            other_splits,
            index,
        )
        cell = index.span_id(*span)

        assert after[0, cell].tolist() == pytest.approx(
            before[0, cell].tolist(), abs=TOLERANCE
        )


def test_a_one_token_input_has_an_empty_parent_posterior() -> None:
    index = SpanIndex(1)
    leaves, splits = scored(index, seed=23)

    inside = InsideKernel(LogSumExpReduction()).inside(leaves, splits, index)
    outside = OutsideKernel(LogSumExpReduction()).outside(inside, splits, index)

    assert log_parent_posterior(inside, outside, splits, index).shape == (1, 1, 0)


def test_the_parent_posterior_refuses_a_chart_or_scores_it_cannot_read() -> None:
    index = SpanIndex(3)
    leaves, splits = scored(index, seed=29)
    inside = InsideKernel(LogSumExpReduction()).inside(leaves, splits, index)
    outside = OutsideKernel(LogSumExpReduction()).outside(inside, splits, index)

    with pytest.raises(TypeError, match="an inside chart holds log weights"):
        log_parent_posterior(inside.long(), outside, splits, index)

    with pytest.raises(TypeError, match="an outside chart holds log weights"):
        log_parent_posterior(inside, outside.long(), splits, index)

    with pytest.raises(ValueError, match="an inside chart runs over item and span"):
        log_parent_posterior(inside.unsqueeze(0), outside, splits, index)

    with pytest.raises(ValueError, match="an outside chart runs over item and span"):
        log_parent_posterior(inside, outside.unsqueeze(0), splits, index)

    with pytest.raises(ValueError, match="split scores run over item, span and split"):
        log_parent_posterior(inside, outside, splits.unsqueeze(0), index)

    with pytest.raises(ValueError, match="an outside chart of shape"):
        log_parent_posterior(inside, outside[:, :-1], splits, index)

    with pytest.raises(ValueError, match="split scores of shape"):
        log_parent_posterior(inside, outside, splits[:, :-1], index)


@pytest.mark.parametrize("weighting", ["contextual", "reuse"])
def test_vector_outside_pass_has_one_differentiable_vector_per_span(
    weighting: Literal["reuse", "contextual"],
) -> None:
    index = SpanIndex(4)
    leaves, splits = scored(index, seed=701, batch=2)
    inside = InsideKernel(LogSumExpReduction()).inside(leaves, splits, index)
    outside = OutsideKernel(LogSumExpReduction()).outside(inside, splits, index)
    alpha = log_split_posterior(inside, splits, index)
    cells = torch.randn(index.num_spans, 2, 3, dtype=torch.float64, requires_grad=True)
    kernel = vector_outside(weighting=weighting)

    vectors = kernel(cells, inside, outside, splits, alpha, index)
    vectors.square().sum().backward()

    assert vectors.shape == cells.shape
    assert cells.grad is not None
    assert all(parameter.grad is not None for parameter in kernel.parameters())


def test_contextual_outside_of_a_span_does_not_read_its_own_inside_vector() -> None:
    index = SpanIndex(4)
    leaves, splits = scored(index, seed=709)
    inside = InsideKernel(LogSumExpReduction()).inside(leaves, splits, index)
    outside = OutsideKernel(LogSumExpReduction()).outside(inside, splits, index)
    alpha = log_split_posterior(inside, splits, index)
    cells = torch.randn(index.num_spans, 1, 3, dtype=torch.float64)
    kernel = vector_outside(weighting="contextual")
    child = index.span_id(1, 3)

    before = kernel(cells, inside, outside, splits, alpha, index)
    perturbed_cells = cells.clone()
    perturbed_cells[child] += 19.0
    after = kernel(perturbed_cells, inside, outside, splits, alpha, index)

    torch.testing.assert_close(after[child], before[child])


def test_zero_and_learned_root_boundaries_are_distinct() -> None:
    index = SpanIndex(3)
    leaves, splits = scored(index, seed=719)
    inside = InsideKernel(LogSumExpReduction()).inside(leaves, splits, index)
    outside = OutsideKernel(LogSumExpReduction()).outside(inside, splits, index)
    alpha = log_split_posterior(inside, splits, index)
    cells = torch.randn(index.num_spans, 1, 3, dtype=torch.float64)
    zero = vector_outside(root="zero", seed=17)
    learned = vector_outside(root="learned", seed=17)
    root = index.span_id(0, index.n)

    zero_vectors = zero(cells, inside, outside, splits, alpha, index)
    learned_vectors = learned(cells, inside, outside, splits, alpha, index)

    assert zero_vectors[root].tolist() == [[0.0, 0.0, 0.0]]
    assert learned_vectors[root].abs().sum().item() > 0.0


def test_vector_outside_pass_handles_each_items_own_ragged_root() -> None:
    index = SpanIndex(4)
    batch = 2
    lengths = torch.tensor([2, 4])
    leaves, splits = scored(index, seed=727, batch=batch)
    inside = InsideKernel(LogSumExpReduction()).inside(leaves, splits, index)
    outside = OutsideKernel(LogSumExpReduction()).outside(
        inside, splits, index, lengths=lengths
    )
    alpha = log_split_posterior(inside, splits, index)
    cells = torch.randn(index.num_spans, batch, 3, dtype=torch.float64)
    kernel = vector_outside(weighting="contextual", root="learned")

    vectors = kernel(cells, inside, outside, splits, alpha, index, lengths=lengths)

    short_root = index.span_id(0, 2)
    full_root = index.span_id(0, 4)
    torch.testing.assert_close(vectors[short_root, 0], vectors[full_root, 1])

    for span, (_, end) in enumerate(index.spans_by_length()):
        if end > 2:
            assert vectors[span, 0].tolist() == [0.0, 0.0, 0.0]
