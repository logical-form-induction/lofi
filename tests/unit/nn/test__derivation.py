"""Tests for the derivation view consumed by neural scorers."""

from typing import TYPE_CHECKING

from lofi.data import Derivation, DerivationNode, TokenSpan

if TYPE_CHECKING:
    from collections.abc import Hashable

    from lofi.nn._derivation import DerivationTree


def _root_label[T: Hashable](derivation: DerivationTree[T], /) -> T:
    """Return a label through only the neural derivation view."""
    return derivation.node(derivation.root).label


def test_a_data_derivation_supplies_the_neural_scoring_view() -> None:
    derivation = Derivation(
        2,
        {
            0: DerivationNode(TokenSpan(0, 1), "NP"),
            1: DerivationNode(TokenSpan(1, 2), "S\\NP"),
            2: DerivationNode(
                TokenSpan(0, 2),
                "S",
                rule="backward-application",
                children=(0, 1),
            ),
        },
    )

    assert _root_label(derivation) == "S"
    assert derivation.preorder == (2, 0, 1)
