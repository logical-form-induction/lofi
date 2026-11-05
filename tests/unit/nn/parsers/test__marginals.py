import math

import pytest
import torch

from lofi.deduce import SpanIndex
from lofi.nn.parsers._channel import log_split_posterior
from lofi.nn.parsers._inside import InsideKernel
from lofi.nn.parsers._marginals import (
    _checked_charts,
    derivation_entropy,
    expected_rule_loss,
    log_constituent_marginals,
    log_rule_marginals,
    log_split_posterior_by_gradient,
    rule_marginals_by_gradient,
)
from lofi.nn.parsers._outside import OutsideKernel
from lofi.nn.parsers._reduction import LogSumExpReduction

NEG_INF = float("-inf")

#: Absolute tolerance for probabilities.
TOLERANCE = 1e-5

#: Number of parameter draws used for the alpha identity.
DRAWS = 50

#: Input length used for the alpha identity.
LENGTH = 12

#: Input lengths used for constituent-loss comparisons.
REPORTED_LENGTHS = (10, 20, 40)


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


def charts(
    leaves: torch.Tensor,
    splits: torch.Tensor,
    index: SpanIndex,
    lengths: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    inside = InsideKernel(LogSumExpReduction()).inside(leaves, splits, index)
    outside = OutsideKernel(LogSumExpReduction()).outside(
        inside, splits, index, lengths=lengths
    )

    return (inside, outside)


def derivations(
    index: SpanIndex, leaves: torch.Tensor, splits: torch.Tensor, i: int, j: int
) -> list[tuple[float, frozenset[tuple[int, int]]]]:
    if j - i == 1:
        return [(leaves[i].item(), frozenset())]

    found = []

    for k in index.splits(i, j):
        edge = splits[index.span_id(i, j), k - i - 1].item()
        found += [
            (left + right + edge, frozenset({(i, j)}) | held | rest)
            for left, held in derivations(index, leaves, splits, i, k)
            for right, rest in derivations(index, leaves, splits, k, j)
        ]

    return found


def posterior(
    index: SpanIndex, leaves: torch.Tensor, splits: torch.Tensor
) -> list[tuple[float, frozenset[tuple[int, int]]]]:
    found = derivations(index, leaves, splits, 0, index.n)
    partition = math.log(sum(math.exp(score) for score, _ in found))

    return [(math.exp(score - partition), held) for score, held in found]


def catalan(trees_over: int) -> int:
    internal = trees_over - 1

    return math.comb(2 * internal, internal) // (internal + 1)


def divisible(index: SpanIndex) -> int:
    return sum(1 for _ in index.spans_by_length(min_length=2))


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
def test_every_token_is_a_constituent_of_every_derivation(n: int) -> None:
    index = SpanIndex(n)
    leaves, splits = scored(index, seed=n)
    inside, outside = charts(leaves, splits, index)

    marginals = log_constituent_marginals(inside, outside, index).exp()

    assert marginals[0, : index.n].tolist() == pytest.approx([1.0] * n, abs=TOLERANCE)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7])
def test_the_constituent_marginals_sum_to_one_less_than_the_length(n: int) -> None:
    index = SpanIndex(n)
    leaves, splits = scored(index, seed=n)
    inside, outside = charts(leaves, splits, index)

    marginals = log_constituent_marginals(inside, outside, index).exp()

    assert marginals[0, index.n :].sum().item() == pytest.approx(n - 1, abs=TOLERANCE)


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_a_constituent_marginal_is_the_share_of_trees_holding_that_span(
    n: int,
) -> None:
    index = SpanIndex(n)
    leaves, splits = scored(index, seed=n + 1)
    inside, outside = charts(leaves, splits, index)
    trees = posterior(index, leaves[0], splits[0])

    marginals = log_constituent_marginals(inside, outside, index).exp()

    for i, j in index.spans_by_length(min_length=2):
        share = sum(weight for weight, held in trees if (i, j) in held)

        assert marginals[0, index.span_id(i, j)].item() == pytest.approx(
            share, abs=TOLERANCE
        )


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_a_rule_marginal_is_the_share_of_trees_dividing_there(n: int) -> None:
    index = SpanIndex(n)
    leaves, splits = scored(index, seed=n + 2)
    inside, outside = charts(leaves, splits, index)
    trees = posterior(index, leaves[0], splits[0])

    marginals = log_rule_marginals(inside, outside, splits, index).exp()

    for i, j in index.spans_by_length(min_length=2):
        for k in index.splits(i, j):
            share = sum(
                weight
                for weight, held in trees
                if (i, j) in held and ((i, k) in held or k - i == 1)
                if (k, j) in held or j - k == 1
            )

            assert marginals[0, index.span_id(i, j), k - i - 1].item() == pytest.approx(
                share, abs=TOLERANCE
            )


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
def test_the_rule_marginals_of_a_span_sum_to_its_constituent_marginal(n: int) -> None:
    index = SpanIndex(n)
    leaves, splits = scored(index, seed=n + 3)
    inside, outside = charts(leaves, splits, index)

    spans = log_constituent_marginals(inside, outside, index).exp()
    rules = log_rule_marginals(inside, outside, splits, index).exp()

    assert rules.sum(dim=-1)[0, index.n :].tolist() == pytest.approx(
        spans[0, index.n :].tolist(), abs=TOLERANCE
    )


def test_rule_losses_use_expected_counts_by_default() -> None:
    index = SpanIndex(4)
    leaves, splits = scored(index, seed=41, batch=2)
    inside, outside = charts(leaves, splits, index)
    weights = log_rule_marginals(inside, outside, splits, index).exp()
    losses = torch.arange(weights.numel(), dtype=weights.dtype).reshape(weights.shape)
    valid = torch.tensor(index.split_mask).unsqueeze(0)
    wanted = (torch.where(valid, losses, 0.0) * weights).sum(dim=(1, 2))

    assert expected_rule_loss(losses, weights, index).tolist() == pytest.approx(
        wanted.tolist(), abs=TOLERANCE
    )


def test_uniform_span_weighting_is_an_explicit_ablation() -> None:
    n = 5
    index = SpanIndex(n)
    leaves, splits = scored(index, seed=43)
    inside, outside = charts(leaves, splits, index)
    weights = log_rule_marginals(inside, outside, splits, index).exp()
    losses = torch.ones_like(weights)

    corrected = expected_rule_loss(losses, weights, index)
    ablation = expected_rule_loss(losses, weights, index, uniform=True)

    assert corrected.item() == pytest.approx(n - 1, abs=TOLERANCE)
    assert ablation.item() == pytest.approx(divisible(index), abs=TOLERANCE)


def test_rule_loss_respects_item_lengths() -> None:
    index = SpanIndex(5)
    leaves, splits = scored(index, seed=47, batch=3)
    lengths = torch.tensor([5, 3, 1])
    inside, outside = charts(leaves, splits, index, lengths)
    weights = log_rule_marginals(inside, outside, splits, index, lengths=lengths).exp()
    losses = torch.ones_like(weights)

    corrected = expected_rule_loss(losses, weights, index, lengths=lengths)
    ablation = expected_rule_loss(losses, weights, index, uniform=True, lengths=lengths)

    assert corrected.tolist() == pytest.approx([4.0, 2.0, 0.0], abs=TOLERANCE)
    assert ablation.tolist() == pytest.approx([10.0, 3.0, 0.0], abs=TOLERANCE)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
def test_the_rule_marginal_is_the_gradient_of_the_partition_at_its_score(
    n: int,
) -> None:
    index = SpanIndex(n)
    leaves, splits = scored(index, seed=n + 4)
    inside, outside = charts(leaves, splits, index)

    counts = rule_marginals_by_gradient(leaves, splits, index)
    closed = log_rule_marginals(inside, outside, splits, index).exp()

    assert counts.shape == closed.shape
    assert counts.flatten().tolist() == pytest.approx(
        closed.flatten().tolist(), abs=TOLERANCE
    )


def test_alpha_off_the_gradient_is_the_closed_form_softmax() -> None:
    index = SpanIndex(LENGTH)
    real = torch.tensor(index.split_mask).unsqueeze(0)
    worst = 0.0

    for draw in range(DRAWS):
        leaves, splits = scored(index, seed=draw)

        closed = log_split_posterior(
            InsideKernel(LogSumExpReduction()).inside(leaves, splits, index),
            splits,
            index,
        )
        gradient = log_split_posterior_by_gradient(leaves, splits, index)
        gap = (closed.exp() - gradient.exp()).abs().masked_fill(~real, 0.0)
        worst = max(worst, gap.max().item())

    assert worst < TOLERANCE


def test_the_two_readings_of_alpha_part_company_where_one_divides_by_zero() -> None:
    index = SpanIndex(3)
    leaves = torch.zeros(1, index.n, dtype=torch.float64)
    splits = torch.zeros(1, index.num_spans, index.max_splits, dtype=torch.float64)
    # the final two-token span has inside weight but no constituent mass
    splits[0, index.span_id(0, 3), 0] = NEG_INF
    stranded = index.span_id(1, 3)

    closed = log_split_posterior(
        InsideKernel(LogSumExpReduction()).inside(leaves, splits, index), splits, index
    )
    gradient = log_split_posterior_by_gradient(leaves, splits, index)

    assert closed[0, stranded, 0].exp().item() == pytest.approx(1.0)
    assert gradient[0, stranded, 0].item() == NEG_INF
    assert closed[0, index.span_id(0, 2), 0].exp().item() == pytest.approx(1.0)
    assert gradient[0, index.span_id(0, 2), 0].exp().item() == pytest.approx(1.0)


def test_the_gradient_read_out_needs_no_gradient_of_its_own() -> None:
    index = SpanIndex(4)
    leaves, splits = scored(index, seed=17)
    leaves.requires_grad_()
    splits.requires_grad_()

    with torch.no_grad():
        counts = rule_marginals_by_gradient(leaves, splits, index)

    assert not counts.requires_grad
    assert splits.grad is None


def test_the_closed_form_marginal_stays_attached_to_the_scores() -> None:
    index = SpanIndex(4)
    leaves, splits = scored(index, seed=19)
    splits.requires_grad_()
    inside, outside = charts(leaves, splits, index)

    log_rule_marginals(inside, outside, splits, index).exp().sum().backward()

    assert splits.grad is not None
    assert bool(torch.isfinite(splits.grad).all())


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
def test_the_entropy_is_the_entropy_of_the_distribution_over_trees(n: int) -> None:
    index = SpanIndex(n)
    leaves, splits = scored(index, seed=n + 5)
    inside, outside = charts(leaves, splits, index)
    trees = posterior(index, leaves[0], splits[0])

    entropy = derivation_entropy(inside, outside, splits, index)

    assert entropy.item() == pytest.approx(
        -sum(weight * math.log(weight) for weight, _ in trees), abs=1e-9
    )


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7])
def test_a_chart_that_prefers_nothing_is_as_uncertain_as_it_can_be(n: int) -> None:
    index = SpanIndex(n)
    leaves = torch.zeros(1, index.n, dtype=torch.float64)
    splits = torch.zeros(1, index.num_spans, index.max_splits, dtype=torch.float64)
    inside, outside = charts(leaves, splits, index)

    entropy = derivation_entropy(inside, outside, splits, index)

    assert entropy.item() == pytest.approx(math.log(catalan(n)), abs=1e-9)


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_an_unambiguous_chart_has_no_entropy(n: int) -> None:
    index = SpanIndex(n)
    leaves, splits = scored(index, seed=n + 6)
    # left-only split points admit exactly one finite tree
    splits = splits.masked_fill(
        torch.tensor(index.split_mask).unsqueeze(0), NEG_INF
    ).index_put(
        (torch.tensor([0]), torch.arange(index.num_spans), torch.zeros(1).long()),
        torch.zeros(1, dtype=torch.float64),
    )
    inside, outside = charts(leaves, splits, index)

    entropy = derivation_entropy(inside, outside, splits, index)

    assert entropy.item() == pytest.approx(0.0, abs=1e-9)


def test_a_ragged_batch_reads_every_marginal_from_its_own_goal_span() -> None:
    index = SpanIndex(5)
    leaves, splits = scored(index, seed=23, batch=3)
    lengths = torch.tensor([5, 3, 1])
    inside, outside = charts(leaves, splits, index, lengths)

    marginals = log_constituent_marginals(inside, outside, index, lengths=lengths).exp()

    for item, length in enumerate(lengths.tolist()):
        inner = sum(
            marginals[item, index.span_id(i, j)].item()
            for i, j in index.spans_by_length(min_length=2)
            if j <= length
        )

        assert inner == pytest.approx(length - 1, abs=TOLERANCE)
        assert marginals[item, :length].tolist() == pytest.approx([1.0] * length)
        assert marginals[item, length : index.n].tolist() == pytest.approx(
            [0.0] * (index.n - length)
        )


def test_a_ragged_batch_reads_every_entropy_from_its_own_goal_span() -> None:
    index = SpanIndex(4)
    leaves = torch.zeros(2, index.n, dtype=torch.float64)
    splits = torch.zeros(2, index.num_spans, index.max_splits, dtype=torch.float64)
    lengths = torch.tensor([4, 2])
    inside, outside = charts(leaves, splits, index, lengths)

    entropy = derivation_entropy(inside, outside, splits, index, lengths=lengths)

    assert entropy.tolist() == pytest.approx(
        [math.log(catalan(4)), math.log(catalan(2))], abs=1e-9
    )


def test_a_chart_nothing_derives_has_no_marginal_and_no_nan() -> None:
    index = SpanIndex(4)
    leaves = torch.zeros(1, index.n, dtype=torch.float64)
    splits = torch.full(
        (1, index.num_spans, index.max_splits), NEG_INF, dtype=torch.float64
    )
    inside, outside = charts(leaves, splits, index)

    spans = log_constituent_marginals(inside, outside, index)
    rules = log_rule_marginals(inside, outside, splits, index)
    entropy = derivation_entropy(inside, outside, splits, index)
    counts = rule_marginals_by_gradient(leaves, splits, index)

    assert not bool(torch.isnan(spans).any())
    assert not bool(torch.isnan(rules).any())
    assert spans.max().item() == NEG_INF
    assert entropy.item() == pytest.approx(0.0)
    assert not bool(torch.isnan(counts).any())
    assert counts.abs().max().item() == pytest.approx(0.0)


def believed_share(n: int) -> float:
    index = SpanIndex(n)
    leaves = torch.zeros(1, index.n, dtype=torch.float64)
    splits = torch.zeros(1, index.num_spans, index.max_splits, dtype=torch.float64)
    inside, outside = charts(leaves, splits, index)
    marginals = log_constituent_marginals(inside, outside, index).exp()

    return marginals[0, index.n :].sum().item() / divisible(index)


@pytest.mark.parametrize("n", REPORTED_LENGTHS)
def test_weighting_every_span_alike_spends_most_of_the_loss_off_the_tree(
    n: int,
) -> None:
    index = SpanIndex(n)

    assert divisible(index) == index.num_spans - n
    assert divisible(index) == n * (n - 1) // 2
    # a binary tree has n - 1 constituents among n(n - 1)/2 non-leaf spans
    assert believed_share(n) == pytest.approx(2 / n, abs=TOLERANCE)


def test_the_share_of_a_uniform_weighting_falls_away_with_length() -> None:
    shares = [believed_share(n) for n in REPORTED_LENGTHS]

    assert shares == sorted(shares, reverse=True)
    assert [share * n for share, n in zip(shares, REPORTED_LENGTHS, strict=True)] == (
        pytest.approx([shares[0] * REPORTED_LENGTHS[0]] * len(shares), abs=TOLERANCE)
    )


def test_the_marginals_refuse_charts_they_cannot_read() -> None:
    index = SpanIndex(3)
    leaves, splits = scored(index, seed=29)
    inside, outside = charts(leaves, splits, index)

    with pytest.raises(TypeError, match="log weights"):
        log_constituent_marginals(inside.long(), outside, index)

    with pytest.raises(ValueError, match="rank"):
        log_constituent_marginals(inside.unsqueeze(0), outside, index)

    with pytest.raises(ValueError, match="an outside chart of the same"):
        log_constituent_marginals(inside, outside.expand(2, index.num_spans), index)

    with pytest.raises(ValueError, match="split scores"):
        log_rule_marginals(inside, outside, splits[:, :, :-1], index)

    with pytest.raises(ValueError, match="split scores"):
        derivation_entropy(inside, outside, splits[:, :, :-1], index)


def test_the_chart_pair_reports_the_batch_the_two_agree_on() -> None:
    index = SpanIndex(3)
    leaves, splits = scored(index, seed=31, batch=4)
    inside, outside = charts(leaves, splits, index)

    assert _checked_charts(inside, outside, index) == 4
