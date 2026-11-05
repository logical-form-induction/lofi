"""Maximum-weight, k-best, and MBR tree decoding from charts."""

from math import inf
from typing import TYPE_CHECKING

import torch

from lofi.nn.parsers._inside import _checked_scores, _root_ids
from lofi.nn.parsers._marginals import (
    log_constituent_marginals,
    log_rule_marginals,
)
from lofi.nn.parsers._values import BracketedTree

if TYPE_CHECKING:
    from torch import Tensor

    from lofi.deduce import Cell, SpanIndex

__all__ = ["k_best", "mbr_trees"]


def _k_best_charts(
    leaf_scores: Tensor, split_scores: Tensor, index: SpanIndex, k: int
) -> tuple[Tensor, Tensor]:
    """Return the best few weights of every span, and the choices behind them.

    Parameters
    ----------
    leaf_scores : Tensor
        The log weight of each token as a one-token span, of rank 2 and detached.
    split_scores : Tensor
        The log weight of each split of each span, of rank 3 and detached.
    index : SpanIndex
        The span addressing both are laid out under.
    k : int
        How many derivations a span keeps.

    Returns
    -------
    tuple[Tensor, Tensor]
        Descending weights of shape ``(item, span, rank)`` and int64 choices of shape
        ``(item, span, rank, 3)`` containing each cutpoint and child rank.
    """
    batch = int(leaf_scores.shape[0])
    weights = leaf_scores.new_full((batch, index.num_spans, k), -inf)
    weights[:, : index.n, 0] = leaf_scores
    choices = torch.zeros(
        (batch, index.num_spans, k, 3), dtype=torch.long, device=leaf_scores.device
    )
    left_ids = leaf_scores.new_tensor(index.left_child_ids, dtype=torch.long)
    right_ids = leaf_scores.new_tensor(index.right_child_ids, dtype=torch.long)
    real = leaf_scores.new_tensor(index.split_mask, dtype=torch.bool).unsqueeze(0)

    for width in range(2, index.n + 1):
        start = index.span_id(0, width)
        stop = start + index.n - width + 1
        shape = (batch, stop - start, index.max_splits, k)
        left = weights.index_select(1, left_ids[start:stop].reshape(-1)).reshape(shape)
        right = weights.index_select(1, right_ids[start:stop].reshape(-1)).reshape(
            shape
        )
        joined = (
            left.unsqueeze(-1)
            + right.unsqueeze(-2)
            + split_scores[:, start:stop, :].unsqueeze(-1).unsqueeze(-1)
        ).masked_fill(~real[:, start:stop, :, None, None], -inf)
        best = joined.reshape(batch, stop - start, -1).topk(k, dim=-1)
        weights[:, start:stop, :] = best.values
        choices[:, start:stop, :, 0] = best.indices // (k * k)
        choices[:, start:stop, :, 1] = best.indices.remainder(k * k) // k
        choices[:, start:stop, :, 2] = best.indices.remainder(k)

    return (weights, choices)


def _unfold(
    index: SpanIndex, choices: list[list[list[int]]], span: int, rank: int
) -> list[Cell]:
    """Return the spans of two tokens or more one ranked derivation holds.

    Parameters
    ----------
    index : SpanIndex
        The span addressing the choices are laid out under.
    choices : list[list[list[int]]]
        One item's choices, read by span id and then by rank, each the cut point's
        column and the rank taken from the left and the right child.
    span : int
        The span to unfold.
    rank : int
        Which of that span's derivations to unfold.

    Returns
    -------
    list[Cell]
        The spans of two tokens or more beneath and including this one, in the order the
        unfolding reached them.
    """
    i, j = index.span_of(span)

    if j - i == 1:
        return []

    column, left_rank, right_rank = choices[span][rank]
    cut = i + column + 1

    return [
        (i, j),
        *_unfold(index, choices, index.span_id(i, cut), left_rank),
        *_unfold(index, choices, index.span_id(cut, j), right_rank),
    ]


def _tree(
    index: SpanIndex, choices: list[list[list[int]]], span: int, rank: int, total: float
) -> BracketedTree:
    """Return one ranked derivation as a tree.

    Parameters
    ----------
    index : SpanIndex
        The span addressing the choices are laid out under.
    choices : list[list[list[int]]]
        One item's choices, read by span id and then by rank.
    span : int
        The span the derivation covers.
    rank : int
        Which of that span's derivations to read.
    total : float
        What the decode maximized at that rank.

    Returns
    -------
    BracketedTree
        The derivation's constituents and the total.
    """
    found = _unfold(index, choices, span, rank)

    return BracketedTree(
        tuple(sorted(found, key=lambda cell: index.span_id(cell[0], cell[1]))), total
    )


def k_best(
    leaf_scores: Tensor,
    split_scores: Tensor,
    index: SpanIndex,
    /,
    *,
    k: int,
    lengths: Tensor | None = None,
) -> tuple[tuple[BracketedTree, ...], ...]:
    """Return each item's best few derivations, heaviest first.

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
    k : int
        The number of derivations to keep per span and return per item.
    lengths : Tensor | None
        A rank-1 integer tensor of shape (item,) holding how many tokens each item has,
        or None where every item runs the full length.

    Returns
    -------
    tuple[tuple[BracketedTree, ...], ...]
        One tuple per item, holding that item's derivations in descending order of log
        weight.

    Raises
    ------
    TypeError
        If a score tensor is not of a floating point dtype, or if the lengths are not of
        an integer dtype.
    ValueError
        If ``k`` is nonpositive, the index is empty, score or length shapes are invalid,
        or a checked length is outside the index.
    """
    _checked_scores(leaf_scores, split_scores, index)

    if k < 1:
        msg = f"a k-best decode keeps at least one derivation, not {k}"
        raise ValueError(msg)

    leaves = leaf_scores.detach()
    weights, choices = _k_best_charts(leaves, split_scores.detach(), index, k)
    roots = _root_ids(leaves, index, lengths).tolist()
    totals = weights.tolist()
    table = choices.tolist()

    return tuple(
        tuple(
            _tree(index, table[item], root, rank, total)
            for rank, total in enumerate(totals[item][root])
            if total != -inf
        )
        for item, root in enumerate(roots)
    )


def mbr_trees(
    inside: Tensor,
    outside: Tensor,
    split_scores: Tensor,
    index: SpanIndex,
    /,
    *,
    lengths: Tensor | None = None,
) -> tuple[BracketedTree | None, ...]:
    """Return the tree of each item that holds the most constituent mass.

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
    tuple[BracketedTree | None, ...]
        One entry per item, holding the tree and the expected count of constituents it
        gets right, or None where the item has no derivation.

    Raises
    ------
    TypeError
        If a tensor is not of a floating point dtype, or if the lengths are not of an
        integer dtype.
    ValueError
        If tensor ranks, extents, item counts, or length shapes disagree with the index,
        or a checked length is outside its range.
    """
    rules = log_rule_marginals(inside, outside, split_scores, index, lengths=lengths)
    spans = log_constituent_marginals(inside, outside, index, lengths=lengths)
    gain = spans.exp().unsqueeze(-1).expand_as(rules)
    # uniform leaf weights rank trees only by constituent weights
    leaves = inside.new_zeros((int(inside.shape[0]), index.n))
    supported = torch.where(rules > -inf, gain, gain.new_full((), -inf))
    found = k_best(leaves, supported.detach(), index, k=1, lengths=lengths)

    return tuple(trees[0] if trees else None for trees in found)
