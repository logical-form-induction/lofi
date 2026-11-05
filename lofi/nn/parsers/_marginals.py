"""Chart marginals and derivation entropy."""

from math import inf
from typing import TYPE_CHECKING

import torch

from lofi.nn.parsers._channel import log_split_posterior
from lofi.nn.parsers._inside import InsideKernel, _root_ids
from lofi.nn.parsers._normalize import masked_log_softmax
from lofi.nn.parsers._reduction import LogSumExpReduction
from lofi.nn.parsers._shapes import (
    SPLIT_RANK,
    checked_scalar_chart,
    checked_split_scores,
)

if TYPE_CHECKING:
    from torch import Tensor

    from lofi.deduce import SpanIndex

__all__ = [
    "derivation_entropy",
    "expected_rule_loss",
    "log_constituent_marginals",
    "log_rule_marginals",
    "log_split_posterior_by_gradient",
    "rule_marginals_by_gradient",
]


def _valid_splits(
    values: Tensor,
    index: SpanIndex,
    lengths: Tensor | None,
) -> Tensor:
    """Return the split mask for each item.

    Parameters
    ----------
    values : Tensor
        The values to inspect.
    index : SpanIndex
        The addressing used by the computation.
    lengths : Tensor | None
        The item lengths, or None when all items are full length.
    """
    valid = values.new_tensor(index.split_mask, dtype=torch.bool).unsqueeze(0)
    valid = valid.expand(int(values.shape[0]), -1, -1)

    if lengths is None:
        return valid

    _root_ids(values.sum(dim=-1), index, lengths)
    ends = [0] * index.num_spans

    for start, end in index.spans_by_length():
        ends[index.span_id(start, end)] = end

    final = values.new_tensor(ends, dtype=torch.long)

    active = final.unsqueeze(0) <= lengths.to(values.device).unsqueeze(1)

    return valid & active.unsqueeze(-1)


def expected_rule_loss(
    losses: Tensor,
    rule_marginals: Tensor,
    index: SpanIndex,
    /,
    *,
    uniform: bool = False,
    lengths: Tensor | None = None,
) -> Tensor:
    """Weight per-rule losses by their expected counts.

    Parameters
    ----------
    losses : Tensor
        The loss assigned to each rule.
    rule_marginals : Tensor
        The expected count of each rule.
    index : SpanIndex
        The addressing used by the computation.
    uniform : bool
        Whether to weight valid rule positions uniformly.
    lengths : Tensor | None
        The item lengths, or None when all items are full length.

    Raises
    ------
    TypeError
        If a score tensor is not floating point or ``lengths`` is not integral.
    ValueError
        If a tensor has an incompatible rank or shape, or a length is invalid.
    """
    if losses.ndim != SPLIT_RANK:
        msg = f"rule losses run over item, span and split, not rank {losses.ndim}"
        raise ValueError(msg)

    batch = int(losses.shape[0])
    checked_split_scores(losses, index, batch)
    checked_split_scores(rule_marginals, index, batch)
    valid = _valid_splits(losses, index, lengths)
    held = torch.where(valid, losses, losses.new_zeros(()))

    if uniform:
        counts = valid.sum(dim=-1, keepdim=True).clamp_min(1)
        weights = valid.to(losses.dtype) / counts
    else:
        weights = torch.where(
            valid,
            rule_marginals,
            rule_marginals.new_zeros(()),
        )

    return (held * weights).sum(dim=(1, 2))


def _checked_charts(inside: Tensor, outside: Tensor, index: SpanIndex, /) -> int:
    """Return the batch size of an inside and an outside chart.

    Parameters
    ----------
    inside : Tensor
        The log inside weight of every span, of rank 2.
    outside : Tensor
        The log outside weight of every span, of rank 2.
    index : SpanIndex
        The span addressing both are laid out under.

    Returns
    -------
    int
        How many items the two charts agree on.

    Raises
    ------
    TypeError
        If either chart is not of a floating point dtype.
    ValueError
        If either has the wrong rank or an extent the index does not call for, or if the
        two cover different numbers of items.
    """
    batch = checked_scalar_chart("an inside chart", inside, index)
    surrounding = checked_scalar_chart("an outside chart", outside, index)

    if surrounding != batch:
        msg = (
            f"an inside chart of {batch} items needs an outside chart of the "
            f"same, not {surrounding}"
        )
        raise ValueError(msg)

    return batch


def _normalizer(inside: Tensor, index: SpanIndex, lengths: Tensor | None) -> Tensor:
    """Return the log partition a chart divides by, shaped to broadcast over spans.

    Parameters
    ----------
    inside : Tensor
        Rank-2 log inside weights over items and spans.
    index : SpanIndex
        The chart's span index.
    lengths : Tensor | None
        Token counts by item, or ``None`` when every item fills the index.

    Returns
    -------
    Tensor
        A rank-2 ``(item, 1)`` partition, with zero for underived items.
    """
    roots = _root_ids(inside, index, lengths)
    partition = inside.gather(1, roots.unsqueeze(1))

    return partition.masked_fill(partition == -inf, 0.0)


def log_constituent_marginals(
    inside: Tensor,
    outside: Tensor,
    index: SpanIndex,
    /,
    *,
    lengths: Tensor | None = None,
) -> Tensor:
    """Return the log probability that each span is a constituent.

    Parameters
    ----------
    inside : Tensor
        A rank-2 floating point tensor of shape (item, span) holding each span's log
        inside weight, as ``InsideKernel.inside`` returns it under log-sum-exp
        reduction.
    outside : Tensor
        A rank-2 floating point tensor of shape (item, span) holding each span's log
        outside weight, as ``OutsideKernel.outside`` returns it under log-sum-exp
        reduction.
    index : SpanIndex
        The span addressing both charts are laid out under.
    lengths : Tensor | None
        A rank-1 integer tensor of shape (item,) holding how many tokens each item has,
        or None where every item runs the full length.

    Returns
    -------
    Tensor
        A rank-2 tensor of shape (item, span) in the charts' floating point dtype,
        holding the log probability that each span is a constituent.

    Raises
    ------
    TypeError
        If a chart is not of a floating point dtype, or if the lengths are not of an
        integer dtype.
    ValueError
        If chart ranks, extents, item counts, or length shapes disagree with the index,
        or a checked length is outside its range.
    """
    _checked_charts(inside, outside, index)

    return inside + outside - _normalizer(inside, index, lengths)


def log_rule_marginals(
    inside: Tensor,
    outside: Tensor,
    split_scores: Tensor,
    index: SpanIndex,
    /,
    *,
    lengths: Tensor | None = None,
) -> Tensor:
    """Return the log expected count of every span divided at every cut point.

    Parameters
    ----------
    inside : Tensor
        A rank-2 floating point tensor of shape (item, span) holding each span's log
        inside weight.
    outside : Tensor
        A rank-2 floating point tensor of shape (item, span) holding each span's log
        outside weight.
    split_scores : Tensor
        A rank-3 floating point tensor of shape (item, span, split) holding the log
        weight of building a span from its two children at each of its splits.
    index : SpanIndex
        The span addressing the three tensors are laid out under.
    lengths : Tensor | None
        A rank-1 integer tensor of shape (item,) holding how many tokens each item has,
        or None where every item runs the full length.

    Returns
    -------
    Tensor
        A rank-3 tensor of shape (item, span, split) in the scores' floating point
        dtype.

    Raises
    ------
    TypeError
        If a tensor is not of a floating point dtype, or if the lengths are not of an
        integer dtype.
    ValueError
        If tensor ranks, extents, item counts, or length shapes disagree with the index,
        or a checked length is outside its range.
    """
    batch = _checked_charts(inside, outside, index)
    checked_split_scores(split_scores, index, batch)

    if index.max_splits == 0:
        # one-token inputs have no decomposition
        return split_scores.clone()

    shape = (batch, index.num_spans, index.max_splits)
    left_ids = inside.new_tensor(index.left_child_ids, dtype=torch.long).reshape(-1)
    right_ids = inside.new_tensor(index.right_child_ids, dtype=torch.long).reshape(-1)
    real = inside.new_tensor(index.split_mask, dtype=torch.bool).unsqueeze(0)
    joined = (
        inside.index_select(1, left_ids).reshape(shape)
        + inside.index_select(1, right_ids).reshape(shape)
        + split_scores
        + outside.unsqueeze(-1)
        - _normalizer(inside, index, lengths).unsqueeze(-1)
    )

    return joined.masked_fill(~real, -inf)


def rule_marginals_by_gradient(
    leaf_scores: Tensor,
    split_scores: Tensor,
    index: SpanIndex,
    /,
    *,
    lengths: Tensor | None = None,
) -> Tensor:
    """Return the expected count of every decomposition, by differentiating log Z.

    Parameters
    ----------
    leaf_scores : Tensor
        A rank-2 floating point tensor of shape (item, token) holding the log weight of
        each token as a one-token span.
    split_scores : Tensor
        A rank-3 floating point tensor of shape (item, span, split) holding the log
        weight of building a span from its two children at each of its splits.
    index : SpanIndex
        The span addressing both tensors are laid out under.
    lengths : Tensor | None
        A rank-1 integer tensor of shape (item,) holding how many tokens each item has,
        or None where every item runs the full length.

    Returns
    -------
    Tensor
        A rank-3 tensor of shape (item, span, split) in the scores' floating point
        dtype, holding the expected number of times a derivation drawn from the chart
        divides that span there.

    Raises
    ------
    TypeError
        If a score tensor is not of a floating point dtype, or if the lengths are not of
        an integer dtype.
    ValueError
        If the index is empty, score or length shapes disagree with it, or a checked
        length is outside its range.
    """
    leaves = leaf_scores.detach().clone()
    splits = split_scores.detach().clone().requires_grad_()

    with torch.enable_grad():
        partition = InsideKernel(LogSumExpReduction()).log_partition(
            leaves, splits, index, lengths=lengths
        )

        if index.max_splits == 0:
            # one-token inputs have no split-score gradient
            return splits.detach()

        (counts,) = torch.autograd.grad(partition.sum(), (splits,))

    return counts


def log_split_posterior_by_gradient(
    leaf_scores: Tensor,
    split_scores: Tensor,
    index: SpanIndex,
    /,
    *,
    lengths: Tensor | None = None,
) -> Tensor:
    """Return the log split posterior, read off the gradient of the log partition.

    Parameters
    ----------
    leaf_scores : Tensor
        A rank-2 floating point tensor of shape (item, token) holding the log weight of
        each token as a one-token span.
    split_scores : Tensor
        A rank-3 floating point tensor of shape (item, span, split) holding the log
        weight of building a span from its two children at each of its splits.
    index : SpanIndex
        The span addressing both tensors are laid out under.
    lengths : Tensor | None
        A rank-1 integer tensor of shape (item,) holding how many tokens each item has,
        or None where every item runs the full length.

    Returns
    -------
    Tensor
        A rank-3 tensor of shape (item, span, split) in the scores' floating point
        dtype, holding the log posterior over each span's splits.

    Raises
    ------
    TypeError
        If a score tensor is not of a floating point dtype, or if the lengths are not of
        an integer dtype.
    ValueError
        If the index is empty, score or length shapes disagree with it, or a checked
        length is outside its range.
    """
    counts = rule_marginals_by_gradient(
        leaf_scores, split_scores, index, lengths=lengths
    )

    if index.max_splits == 0:
        return counts

    real = counts.new_tensor(index.split_mask, dtype=torch.bool).unsqueeze(0)
    # clamp negative rounding error before taking the logarithm
    return masked_log_softmax(counts.clamp_min(0.0).log(), real)


def derivation_entropy(
    inside: Tensor,
    outside: Tensor,
    split_scores: Tensor,
    index: SpanIndex,
    /,
    *,
    lengths: Tensor | None = None,
) -> Tensor:
    """Return the entropy of each item's distribution over binary trees, in nats.

    Parameters
    ----------
    inside : Tensor
        A rank-2 floating point tensor of shape (item, span) holding each span's log
        inside weight.
    outside : Tensor
        A rank-2 floating point tensor of shape (item, span) holding each span's log
        outside weight.
    split_scores : Tensor
        A rank-3 floating point tensor of shape (item, span, split) holding the log
        weight of building a span from its two children at each of its splits.
    index : SpanIndex
        The span addressing the three tensors are laid out under.
    lengths : Tensor | None
        A rank-1 integer tensor of shape (item,) holding how many tokens each item has,
        or None where every item runs the full length.

    Returns
    -------
    Tensor
        A rank-1 tensor of shape (item,) in the scores' floating point dtype, holding
        each item's entropy in nats.

    Raises
    ------
    TypeError
        If a tensor is not of a floating point dtype, or if the lengths are not of an
        integer dtype.
    ValueError
        If tensor ranks, extents, item counts, or length shapes disagree with the index,
        or a checked length is outside its range.
    """
    batch = _checked_charts(inside, outside, index)
    checked_split_scores(split_scores, index, batch)
    alpha = log_split_posterior(inside, split_scores, index)
    weight = log_constituent_marginals(inside, outside, index, lengths=lengths).exp()
    # Replace negative infinity before multiplying. Masking the product afterwards
    # leaves a hidden ``0 * -inf`` branch whose backward pass is not finite.
    finite = alpha.isfinite()
    safe_alpha = torch.where(finite, alpha, alpha.new_zeros(()))
    shares = torch.where(finite, safe_alpha.exp() * safe_alpha, alpha.new_zeros(()))

    return -(weight * shares.sum(dim=-1)).sum(dim=-1)
