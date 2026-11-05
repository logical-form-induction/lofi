import pickle
from typing import Final

import pytest

from lofi._types import ItemId
from lofi.data._join import (
    DuplicateItemIdError,
    ItemLookup,
    ItemMismatchError,
    MissingItemError,
    PairAnchorError,
    prejoin,
)
from lofi.data._supervision import (
    Example,
    ForcedChoice,
    Relational,
    RelationalScalar,
    Scalar,
    Scale,
    Unlabeled,
)
from lofi.data._values import Item, Token

SCALE: Final = Scale("ordinal", 1.0, 7.0)


def item(name: str) -> Item:
    return Item(ItemId(name), (Token(0, name),))


def items() -> tuple[Item, ...]:
    return (item("first"), item("second"), item("third"))


def test_a_lookup_resolves_reader_assigned_ids() -> None:
    held = items()
    lookup = ItemLookup(held)

    assert len(lookup) == 3
    assert lookup.resolve(ItemId("second")) is held[1]
    assert tuple(lookup.items) == tuple(one.item_id for one in held)


def test_a_lookup_survives_a_pickle_round_trip() -> None:
    held = items()

    restored = pickle.loads(pickle.dumps(ItemLookup(held)))

    assert restored.items == ItemLookup(held).items
    assert restored.resolve(held[1].item_id) == held[1]


def test_a_lookup_refuses_duplicate_ids_even_when_the_items_differ() -> None:
    duplicate = Item(ItemId("first"), (Token(0, "different"),))

    with pytest.raises(DuplicateItemIdError) as raised:
        ItemLookup((item("first"), duplicate))

    assert raised.value.item_id == ItemId("first")


def test_a_lookup_reports_a_missing_reference_by_id() -> None:
    lookup = ItemLookup(items())

    with pytest.raises(MissingItemError) as raised:
        lookup.resolve(ItemId("absent"))

    assert raised.value.item_id == ItemId("absent")


def test_a_relational_example_joins_its_anchor_and_other_item() -> None:
    held = items()
    example = Example(held[0], Relational(held[2].item_id, "contrast"))

    joined = prejoin((example,), ItemLookup(held))[0]

    assert joined.example is example
    assert joined.items == (held[0], held[2])


def test_a_relational_scalar_joins_the_scored_pair() -> None:
    held = items()
    supervision = RelationalScalar(
        held[1].item_id,
        "similarity",
        Scalar(6.0, SCALE),
    )

    joined = prejoin((Example(held[0], supervision),), ItemLookup(held))[0]

    assert joined.items == held[:2]
    assert joined.example.supervision is supervision


def test_a_choice_joins_alternatives_in_presentation_order() -> None:
    held = items()
    supervision = ForcedChoice(
        (held[0].item_id, held[2].item_id, held[1].item_id),
        frozenset({held[2].item_id}),
    )

    joined = prejoin((Example(held[0], supervision),), ItemLookup(held))[0]

    assert joined.items == (held[0], held[2], held[1])


def test_a_choice_must_be_anchored_on_its_first_alternative() -> None:
    held = items()
    supervision = ForcedChoice(
        (held[0].item_id, held[1].item_id),
        frozenset({held[0].item_id}),
    )

    with pytest.raises(PairAnchorError) as raised:
        prejoin((Example(held[2], supervision),), ItemLookup(held))

    assert raised.value.anchor == held[2].item_id
    assert raised.value.expected == held[0].item_id


def test_prejoining_reports_a_missing_pair_member() -> None:
    held = items()
    example = Example(held[0], Relational(ItemId("absent"), "contrast"))

    with pytest.raises(MissingItemError, match="absent"):
        prejoin((example,), ItemLookup(held))


def test_prejoining_reports_a_missing_anchor() -> None:
    held = items()
    example = Example(item("absent"), Unlabeled())

    with pytest.raises(MissingItemError) as raised:
        prejoin((example,), ItemLookup(held))

    assert raised.value.item_id == example.item.item_id


def test_prejoining_refuses_equal_ids_with_different_item_content() -> None:
    held = items()
    changed = Item(ItemId("first"), (Token(0, "changed"),))

    with pytest.raises(ItemMismatchError) as raised:
        prejoin((Example(changed, Unlabeled()),), ItemLookup(held))

    assert raised.value.item_id == changed.item_id


def test_prejoining_accepts_equal_items_from_separate_reader_calls() -> None:
    held = items()
    equal_copy = item("first")

    joined = prejoin((Example(equal_copy, Unlabeled()),), ItemLookup(held))[0]

    assert joined.items[0] is held[0]
    assert joined.items[0] == equal_copy


def test_a_nonrelational_example_keeps_only_its_anchor() -> None:
    held = items()
    example = Example(held[1], Unlabeled())

    joined = prejoin((example,), ItemLookup(held))[0]

    assert joined.items == (held[1],)
