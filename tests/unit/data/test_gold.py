"""Tests for repeated-response aggregation."""

import pytest

from lofi._types import ItemId
from lofi.data import Example, Item, Relational, Token, Unlabeled
from lofi.data.gold import MajorityGold


def _item(name: str) -> Item:
    """Return one single-token item."""
    return Item(ItemId(name), (Token(0, name),), text=name)


def test_majority_gold_collapses_repeated_pairs_in_first_use_order() -> None:
    """Emit one anonymous response for each pair and retain unrelated rows."""
    anchor = _item("anchor")
    other = _item("other")
    rows = (
        Example(anchor, Relational(other.item_id, "entails")),
        Example(other, Unlabeled()),
        Example(anchor, Relational(other.item_id, "neutral")),
        Example(anchor, Relational(other.item_id, "entails")),
    )

    aggregated = MajorityGold()(rows)

    assert aggregated == (
        Example(anchor, Relational(other.item_id, "entails")),
        Example(other, Unlabeled()),
    )


def test_majority_gold_refuses_a_tied_pair() -> None:
    """Require a unique empirical mode."""
    anchor = _item("anchor")
    other = _item("other")

    with pytest.raises(ValueError, match="no unique majority"):
        MajorityGold()(
            (
                Example(anchor, Relational(other.item_id, "entails")),
                Example(anchor, Relational(other.item_id, "neutral")),
            )
        )
