from typing import Final

import pytest

from lofi.data._inventory import (
    DuplicateSymbolError,
    Inventory,
    UnknownSymbolError,
)

RESERVED: Final = ("<unknown>", "not")

ORDINARY: Final = ("the", "child", "sleeps")

RESERVED_COUNT: Final = 2

DECLARED: Final = 5


def built(name: str = "surfaces") -> Inventory:
    return Inventory(name, ORDINARY, RESERVED)


def test_the_reserved_symbols_take_the_prefix_of_the_index_range() -> None:
    inventory = built()

    assert [inventory.index(symbol) for symbol in RESERVED] == [0, 1]
    assert [inventory.index(symbol) for symbol in ORDINARY] == [2, 3, 4]


def test_an_inventory_declaring_nothing_reserved_starts_its_codes_at_zero() -> None:
    inventory = Inventory("surfaces", ORDINARY)

    assert inventory.index("the") == 0
    assert inventory.reserved == ()
    assert inventory.size == len(ORDINARY)


def test_a_code_reads_back_as_the_symbol_it_stands_for() -> None:
    inventory = built()

    for symbol in (*RESERVED, *ORDINARY):
        assert inventory.symbol(inventory.index(symbol)) == symbol


def test_a_code_outside_the_inventory_is_refused() -> None:
    inventory = built()

    with pytest.raises(IndexError, match="was asked for"):
        inventory.symbol(inventory.size)

    with pytest.raises(IndexError):
        inventory.symbol(-1)


def test_the_reserved_half_is_the_one_the_prefix_holds() -> None:
    inventory = built()

    assert inventory.is_reserved("not")
    assert not inventory.is_reserved("child")


def test_a_symbol_the_inventory_does_not_declare_is_refused() -> None:
    inventory = built()

    with pytest.raises(UnknownSymbolError) as raised:
        inventory.index("kangaroo")

    assert raised.value.symbol == "kangaroo"
    assert raised.value.inventory == "surfaces"
    assert "kangaroo" not in inventory
    assert inventory.size == DECLARED


def test_a_symbol_declared_twice_is_refused() -> None:
    with pytest.raises(DuplicateSymbolError, match="twice"):
        Inventory("surfaces", ("the", "the"))


def test_a_symbol_declared_in_both_halves_is_refused() -> None:
    with pytest.raises(DuplicateSymbolError) as raised:
        Inventory("surfaces", ("not",), ("not",))

    assert raised.value.symbol == "not"


def test_an_empty_symbol_is_refused() -> None:
    with pytest.raises(ValueError, match="ordinary symbol"):
        Inventory("surfaces", ("",))

    with pytest.raises(ValueError, match="reserved symbol"):
        Inventory("surfaces", (), ("",))


def test_an_unnamed_inventory_is_refused() -> None:
    with pytest.raises(ValueError, match="inventory's name"):
        Inventory("", ORDINARY)


def test_growing_the_inventory_moves_no_code_already_declared() -> None:
    inventory = built()
    grown = inventory.extended(("dog", "runs"))

    for symbol in (*RESERVED, *ORDINARY):
        assert grown.index(symbol) == inventory.index(symbol)

    assert grown.index("dog") == inventory.size
    assert grown.size == inventory.size + 2


def test_growing_the_inventory_keeps_the_reserved_half_reserved() -> None:
    grown = built().extended(("dog",))

    assert grown.reserved == RESERVED

    for symbol in RESERVED:
        assert grown.is_reserved(symbol)


def test_growing_by_what_is_already_declared_returns_an_equal_inventory() -> None:
    inventory = built()

    assert inventory.extended(("the", "not")) == inventory
    assert inventory.extended(()) == inventory


def test_growing_by_one_symbol_twice_is_refused() -> None:
    with pytest.raises(DuplicateSymbolError, match="dog"):
        built().extended(("dog", "dog"))


def test_two_inventories_declaring_the_same_symbols_are_one_value() -> None:
    assert built() == built()
    assert hash(built()) == hash(built())


def test_an_inventory_that_orders_its_symbols_differently_is_another_value() -> None:
    assert Inventory("surfaces", ("a", "b")) != Inventory("surfaces", ("b", "a"))
    assert Inventory("surfaces", ("a",), ("b",)) != Inventory(
        "surfaces", ("b",), ("a",)
    )
    assert built("surfaces") != built("categories")


def test_an_inventory_is_not_equal_to_something_else() -> None:
    assert built() != ORDINARY


def test_the_representation_names_the_constructor_and_both_halves() -> None:
    rendered = repr(built())

    assert rendered.startswith("Inventory('surfaces', ")
    assert repr(ORDINARY) in rendered
    assert repr(RESERVED) in rendered


def test_the_length_is_the_number_of_codes() -> None:
    assert len(built()) == RESERVED_COUNT + len(ORDINARY)
