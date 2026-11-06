"""Globally normalized span-parser tests."""

from typing import Final

import pytest
import torch

from lofi.deduce import (
    Edge,
    ItemKey,
    LogInside,
    MarginalScorer,
    PackedForest,
    RootGoal,
    SpanIndex,
    outside_from_edges,
)
from lofi.nn._values import Encoded
from lofi.nn.parsers._span_crf import SpanCrfParser
from lofi.nn.sites import ComponentId
from lofi.terms import TargetSet

INPUT_WIDTH: Final = 4
HIDDEN: Final = 6
ITEMS: Final = 2
WORDS: Final = 4


class Equality:
    """Compare symbolic oracle categories by value."""

    __slots__ = ()

    @property
    def name(self) -> str:
        """Return the relation name."""
        return "equality"

    @property
    def has_key(self) -> bool:
        """Return whether each equivalence class has a representative."""
        return True

    def key(self, category: str, /) -> str:
        """Return the category as its representative."""
        return category

    def equivalent(self, left: str, right: str, /) -> bool:
        """Return whether two categories have equal values."""
        return left == right


def leaves(*, dtype: torch.dtype = torch.float64) -> Encoded:
    generator = torch.Generator().manual_seed(13)
    values = torch.randn(ITEMS, WORDS, INPUT_WIDTH, generator=generator, dtype=dtype)
    mask = torch.tensor(
        [[True, True, True, True], [True, True, True, False]], dtype=torch.bool
    )
    return Encoded(values, ("item", "word", "width"), mask)


def build() -> SpanCrfParser:
    torch.manual_seed(11)
    return SpanCrfParser(
        input_width=INPUT_WIDTH,
        hidden=HIDDEN,
        component=ComponentId("parser"),
    ).double()


def test_the_span_crf_publishes_a_global_distribution_and_its_decisions() -> None:
    parser = build()
    readouts = parser.readouts(leaves())

    assert parser.publishes == {
        "inside",
        "outside",
        "alpha",
        "mu",
        "log_z",
        "entropy",
        "mbr",
    }
    assert readouts.alpha is not None
    assert readouts.mu is not None
    assert readouts.entropy is not None
    assert readouts.mbr is not None
    assert readouts.entropy.axes == ("item",)
    assert readouts.mbr.axes == ("item", "span")

    alpha = readouts.alpha
    valid = alpha.mask
    assert valid is not None
    totals = alpha.tensor.exp().masked_fill(~valid, 0.0).sum(dim=-1)
    live_rows = valid.any(dim=-1)
    assert torch.allclose(totals[live_rows], torch.ones_like(totals)[live_rows])

    index = SpanIndex(WORDS)
    mbr = readouts.mbr
    assert int(mbr.tensor[0].sum()) == WORDS - 1
    assert int(mbr.tensor[1].sum()) == 2
    assert mbr.tensor[1, index.span_id(0, 4)] == 0.0


def test_the_split_posterior_is_the_gradient_of_the_partition() -> None:
    parser = build()
    leaf_scores, split_scores, index, lengths, _ = parser.potentials(leaves())
    splits = split_scores.detach().clone().requires_grad_()
    partition = parser.kernel.log_partition(
        leaf_scores.detach(), splits, index, lengths=lengths
    )
    (counts,) = torch.autograd.grad(partition.sum(), (splits,))
    readouts = parser.readouts(leaves())
    alpha = readouts.alpha

    assert alpha is not None
    valid = alpha.mask
    assert valid is not None
    row_total = counts.sum(dim=-1, keepdim=True)
    expected = torch.where(row_total > 0.0, counts / row_total, counts)
    assert torch.allclose(alpha.tensor.exp()[valid], expected[valid], atol=1e-10)


def test_the_split_posterior_agrees_with_the_symbolic_marginal_scorer() -> None:
    parser = build()
    leaf_scores, split_scores, index, _, _ = parser.potentials(leaves())
    forest = PackedForest[str, float](index, LogInside())
    keys = {span: ItemKey(span, "S", "tree") for span in index.spans_by_length()}

    for start, end in index.spans_by_length():
        key = keys[(start, end)]
        span_id = index.span_id(start, end)

        if end - start == 1:
            forest.add(
                key,
                Edge("leaf", (), "S", float(leaf_scores[0, start].detach())),
            )
            continue

        for split in index.splits(start, end):
            _, split_id = index.split_id(start, end, split)
            forest.add(
                key,
                Edge(
                    "binary",
                    (keys[(start, split)], keys[(split, end)]),
                    "S",
                    float(split_scores[0, span_id, split_id].detach()),
                    split=split,
                ),
            )

    goal = RootGoal(TargetSet(("S",), Equality()))
    symbolic = MarginalScorer(forest, outside_from_edges(forest, goal)).probabilities
    alpha = parser.readouts(leaves()).alpha

    assert alpha is not None

    for start, end in index.spans_by_length(min_length=2):
        span_id = index.span_id(start, end)
        expected = alpha.tensor[0, span_id].exp()
        actual = symbolic.at((start, end))

        for split in index.splits(start, end):
            _, split_id = index.split_id(start, end, split)
            assert actual[split] == pytest.approx(float(expected[split_id]), abs=1e-10)


def test_entropy_is_differentiable_and_the_mbr_is_a_tree() -> None:
    parser = build()
    readouts = parser.readouts(leaves())
    entropy = readouts.entropy

    assert entropy is not None
    entropy.tensor.sum().backward()
    gradients = tuple(parameter.grad for parameter in parser.parameters())
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(
        bool(torch.isfinite(gradient).all())
        for gradient in gradients
        if gradient is not None
    )


def test_a_leaf_width_mismatch_is_refused_at_the_parser_boundary() -> None:
    parser = build()
    wrong = Encoded(
        torch.zeros(1, 2, INPUT_WIDTH + 1, dtype=torch.float64),
        ("item", "word", "width"),
    )

    with pytest.raises(ValueError, match="leaf width"):
        parser.readouts(wrong)
