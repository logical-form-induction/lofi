from typing import NamedTuple

import torch

import lofi.nn.codec
from lofi.nn.policy import DevicePolicy
from lofi.nn.ports import Consumption, StructuralRelation
from lofi.terms import BatchEncoding, Encoding, Operator, OperatorIndex, Signature

N_SYMBOLS = 5
CAPACITY = 7

# the alphabet has padding, two atomic symbols, and two binary symbols.
INDEX = OperatorIndex(
    Signature(
        "init-fixture",
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
LEAF = (True, True, True, False, False)
LEGALITY = (
    (False, True, True, True, True),
    ANY_SYMBOL,
    LEAF,
    LEAF,
    ANY_SYMBOL,
    LEAF,
    LEAF,
)


class Codec(NamedTuple):
    n_symbols: int
    capacity: int
    skeleton: Encoding | None
    legal_symbols: tuple[tuple[bool, ...], ...]
    index: OperatorIndex


def test_a_term_reaches_the_tensor_world_through_the_published_seam() -> None:
    codec = Codec(
        N_SYMBOLS,
        CAPACITY,
        Encoding((0,) * CAPACITY, PARENTS, DEPTHS, (False,) * CAPACITY, None),
        LEGALITY,
        INDEX,
    )
    view = lofi.nn.codec.NeuralCodecView(codec)
    symbols = (4, 1, 0, 0, 2, 0, 0)
    batch = BatchEncoding(
        (Encoding(symbols, PARENTS, DEPTHS, tuple(code != 0 for code in symbols)),)
    )
    policy = DevicePolicy(torch.device("cpu"), torch.float32)

    placed = view.place_batch(batch, policy=policy)
    rows = view.symbol_rows(placed, torch.eye(N_SYMBOLS))

    assert isinstance(placed, lofi.nn.codec.PlacedEncoding)
    assert view.codec is codec
    assert placed.lengths.tolist() == [3]
    assert rows.tensor[0, view.child_index(0, 1)].tolist() == [0, 0, 1, 0, 0]


def test_a_component_reads_structure_through_the_published_seam() -> None:
    codec = Codec(
        N_SYMBOLS,
        CAPACITY,
        Encoding((0,) * CAPACITY, PARENTS, DEPTHS, (False,) * CAPACITY, None),
        LEGALITY,
        INDEX,
    )
    view = lofi.nn.codec.NeuralCodecView(codec)
    policy = DevicePolicy(torch.device("cpu"), torch.float32)
    served = view.serve({StructuralRelation.DEPTH: Consumption.EMBEDDING})

    assert isinstance(served, lofi.nn.codec.DeclaredStructure)
    assert served.depth(policy=policy).tolist() == list(DEPTHS)


def test_the_closure_of_a_relation_is_read_through_the_published_seam() -> None:
    parent = torch.tensor(
        [[other == at for other in range(CAPACITY)] for at in PARENTS]
    )
    ancestor = lofi.nn.codec.reflexive_closure(parent)

    assert ancestor.tolist()[6] == [True, False, False, False, True, False, True]
