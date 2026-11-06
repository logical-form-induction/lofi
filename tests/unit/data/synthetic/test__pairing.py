"""Tests for declared premise--hypothesis pairing strategies."""

from random import Random

from lofi.data.synthetic._complexity import FormComplexity
from lofi.data.synthetic._pairing import (
    IndependentPairing,
    OverlapMatchedPairing,
)


def _form(*predicates: str) -> FormComplexity:
    """Return a measured form with the requested predicate vocabulary."""
    return FormComplexity({}, predicates=predicates)


def test_independent_pairing_draws_each_side_independently() -> None:
    premises = (_form("left"), _form("shared"))
    hypotheses = (_form("right"), _form("shared"))
    random = Random(3)

    draws = tuple(
        IndependentPairing().choose(random, premises, hypotheses) for _ in range(8)
    )

    assert len({left for left, _ in draws}) == 2
    assert len({right for _, right in draws}) == 2


def test_overlap_matched_pairing_draws_only_maximum_overlap_pairs() -> None:
    premises = (_form("a"), _form("b", "shared"))
    hypotheses = (_form("c"), _form("shared", "b"))

    assert OverlapMatchedPairing().choose(Random(9), premises, hypotheses) == (1, 1)
