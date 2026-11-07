from typing import NamedTuple

import pytest

from lofi.nn.codec import NeuralCodecView
from lofi.terms import Encoding, Operator, OperatorIndex, Signature

N_SYMBOLS = 5
CAPACITY = 7

# the alphabet has padding, two atomic symbols, and two binary symbols.
INDEX = OperatorIndex(
    Signature(
        "ports-fixture",
        ("T",),
        (
            Operator("a", "T"),
            Operator("b", "T"),
            Operator("f", "T", arg_sorts=("T", "T")),
            Operator("g", "T", arg_sorts=("T", "T")),
        ),
    )
)

PARENTS = (Encoding.NO_PARENT, 0, 1, 1, 0, 4, 4)
DEPTHS = (0, 1, 2, 2, 1, 2, 2)

ANY_SYMBOL = (True,) * N_SYMBOLS
NOT_PADDING = (False, True, True, True, True)
LEAF = (True, True, True, False, False)
LEGALITY = (NOT_PADDING, ANY_SYMBOL, LEAF, LEAF, ANY_SYMBOL, LEAF, LEAF)

SKELETON = Encoding((0,) * CAPACITY, PARENTS, DEPTHS, (False,) * CAPACITY, None)


class Codec(NamedTuple):
    n_symbols: int
    capacity: int
    skeleton: Encoding | None
    legal_symbols: tuple[tuple[bool, ...], ...]
    index: OperatorIndex


WELL_FORMED = Codec(N_SYMBOLS, CAPACITY, SKELETON, LEGALITY, INDEX)


def test_a_conforming_codec_is_the_views_codec_by_identity() -> None:
    view = NeuralCodecView(WELL_FORMED)

    assert view.codec is WELL_FORMED
    assert view.capacity == CAPACITY
    assert view.n_symbols == N_SYMBOLS
    assert view.path_addressed


def test_a_layout_the_term_settles_conforms_and_says_so() -> None:
    view = NeuralCodecView(WELL_FORMED._replace(skeleton=None))

    assert not view.path_addressed


def test_a_skeleton_of_another_width_than_the_capacity_is_refused() -> None:
    narrow = WELL_FORMED._replace(capacity=6, legal_symbols=LEGALITY[:-1])

    with pytest.raises(ValueError, match="skeleton is 7 slots wide beside a capacity"):
        NeuralCodecView(narrow)


def test_a_skeleton_with_two_roots_is_refused() -> None:
    forest = SKELETON._replace(
        parents=(Encoding.NO_PARENT, 0, 1, 1, Encoding.NO_PARENT, 4, 4),
        depths=(0, 1, 2, 2, 0, 1, 1),
    )

    with pytest.raises(ValueError, match="2 slots without a parent"):
        NeuralCodecView(WELL_FORMED._replace(skeleton=forest))


def test_a_parent_that_does_not_come_before_its_child_is_refused() -> None:
    tangled = SKELETON._replace(parents=(Encoding.NO_PARENT, 4, 1, 1, 0, 4, 4))

    with pytest.raises(ValueError, match="slot 1 has parent 4"):
        NeuralCodecView(WELL_FORMED._replace(skeleton=tangled))


def test_a_depth_its_parent_does_not_give_it_is_refused() -> None:
    sunk = SKELETON._replace(depths=(0, 1, 3, 2, 1, 2, 2))

    with pytest.raises(ValueError, match="slot 2 is at depth 3"):
        NeuralCodecView(WELL_FORMED._replace(skeleton=sunk))


def test_a_root_below_the_top_is_refused() -> None:
    sunk = SKELETON._replace(depths=(1, 2, 3, 3, 2, 3, 3))

    with pytest.raises(ValueError, match="the root is at depth 1"):
        NeuralCodecView(WELL_FORMED._replace(skeleton=sunk))


def test_a_legality_table_that_leaves_a_slot_out_is_refused() -> None:
    with pytest.raises(ValueError, match="covers 6 slots beside a capacity of 7"):
        NeuralCodecView(WELL_FORMED._replace(legal_symbols=LEGALITY[:-1]))


def test_a_legality_row_that_leaves_a_symbol_out_is_refused() -> None:
    short = (*LEGALITY[:2], LEGALITY[2][:-1], *LEGALITY[3:])

    with pytest.raises(ValueError, match=r"slot 2 .* covers 4 symbols"):
        NeuralCodecView(WELL_FORMED._replace(legal_symbols=short))


def test_an_index_assigning_another_number_of_codes_than_the_alphabet_is_refused() -> (
    None
):
    narrow = OperatorIndex(Signature("narrow", ("T",), (Operator("a", "T"),)))

    with pytest.raises(ValueError, match="assigns 2 codes beside an alphabet of 5"):
        NeuralCodecView(WELL_FORMED._replace(index=narrow))


def test_the_arities_come_from_the_index_the_codec_names() -> None:
    view = NeuralCodecView(WELL_FORMED)

    assert view.codec.index is INDEX
    assert view.arities == tuple(
        0 if INDEX.is_pad(code) else INDEX.arity(code) for code in range(N_SYMBOLS)
    )


def test_a_slot_no_symbol_may_occupy_is_refused() -> None:
    dead = (*LEGALITY[:3], (False,) * N_SYMBOLS, *LEGALITY[4:])

    with pytest.raises(ValueError, match="no symbol may occupy slot 3"):
        NeuralCodecView(WELL_FORMED._replace(legal_symbols=dead))


def test_a_codec_holding_no_slot_is_refused() -> None:
    empty = Codec(N_SYMBOLS, 0, None, (), INDEX)

    with pytest.raises(ValueError, match="holding no slot encodes no term"):
        NeuralCodecView(empty)


def test_a_slot_admitting_pad_alone_where_its_parent_fills_it_is_refused() -> None:
    padded = (*LEGALITY[:2], (True, False, False, False, False), *LEGALITY[3:])

    with pytest.raises(ValueError, match="slot 2 admits pad alone, and symbol 3 at"):
        NeuralCodecView(WELL_FORMED._replace(legal_symbols=padded))


def test_a_slot_refusing_pad_where_its_parent_leaves_it_empty_is_refused() -> None:
    unpadded = (*LEGALITY[:2], (False, True, True, False, False), *LEGALITY[3:])

    with pytest.raises(ValueError, match="slot 2 refuses pad, and symbol 0 at slot 1"):
        NeuralCodecView(WELL_FORMED._replace(legal_symbols=unpadded))


def test_a_layout_the_term_settles_is_not_read_beneath_its_parents() -> None:
    padded = (*LEGALITY[:2], (True, False, False, False, False), *LEGALITY[3:])
    view = NeuralCodecView(WELL_FORMED._replace(skeleton=None, legal_symbols=padded))

    assert not view.path_addressed
