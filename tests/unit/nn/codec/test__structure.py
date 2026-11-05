import math
from typing import TYPE_CHECKING, Final, NamedTuple

import pytest
import torch

from lofi.nn._values import LogDistribution
from lofi.nn.codec import (
    DISTRIBUTION_AXES,
    NeuralCodecView,
    _structure,
    soft_symbol_rows,
)
from lofi.nn.policy import DevicePolicy
from lofi.terms import BatchEncoding, Encoding, Operator, OperatorIndex, Signature

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

N_SYMBOLS: Final = 5
CAPACITY: Final = 3
WIDTH: Final = 4

PARENTS: Final = (Encoding.NO_PARENT, 0, 0)
DEPTHS: Final = (0, 1, 1)
LIVE: Final = (True,) * CAPACITY
INDEX: Final = OperatorIndex(
    Signature(
        "structure-fixture",
        ("T",),
        (
            Operator("a", "T"),
            Operator("b", "T"),
            Operator("not", "T", arg_sorts=("T",)),
            Operator("pair", "T", arg_sorts=("T", "T")),
        ),
    )
)


class Codec(NamedTuple):
    n_symbols: int
    capacity: int
    skeleton: Encoding | None
    legal_symbols: tuple[tuple[bool, ...], ...]
    index: OperatorIndex


VIEW: Final = NeuralCodecView(
    Codec(N_SYMBOLS, CAPACITY, None, ((True,) * N_SYMBOLS,) * CAPACITY, INDEX)
)
POLICY: Final = DevicePolicy(torch.device("cpu"), torch.float32)
TERMS: Final = BatchEncoding(
    (
        Encoding((4, 3, 2), PARENTS, DEPTHS, LIVE),
        Encoding((3, 1, 0), PARENTS, DEPTHS, (True, True, False)),
    )
)


def table() -> Tensor:
    generator = torch.Generator().manual_seed(5)

    return torch.randn(N_SYMBOLS, WIDTH, generator=generator)


def one_hot(terms: BatchEncoding) -> LogDistribution:
    codes = torch.tensor(terms.symbols, dtype=torch.int64)
    mass = torch.nn.functional.one_hot(codes, N_SYMBOLS).double()

    return LogDistribution(mass.log(), DISTRIBUTION_AXES, normalized_over="symbol")


def test_a_one_hot_distribution_mixes_to_the_row_its_own_code_selects() -> None:
    rows, placed = soft_symbol_rows(VIEW, one_hot(TERMS), TERMS, table(), policy=POLICY)
    selected = VIEW.symbol_rows(VIEW.place_batch(TERMS, policy=POLICY), table()).tensor

    assert rows.axes == ("term", "slot", "width")
    assert torch.allclose(rows.tensor, selected, atol=1e-6)
    assert torch.equal(placed.parents, torch.tensor((PARENTS, PARENTS)))
    assert torch.equal(placed.depths, torch.tensor((DEPTHS, DEPTHS)))
    assert placed.mask.tolist() == [list(LIVE), [True, True, False]]


def test_the_structure_comes_back_with_its_symbol_channel_zeroed() -> None:
    _, placed = soft_symbol_rows(VIEW, one_hot(TERMS), TERMS, table(), policy=POLICY)

    assert torch.equal(placed.symbols, torch.zeros_like(placed.parents))


def test_a_structure_symbol_outside_the_alphabet_is_never_read() -> None:
    invalid = BatchEncoding(
        tuple(
            Encoding((N_SYMBOLS + 7,) * CAPACITY, row.parents, row.depths, row.occupied)
            for row in TERMS.rows
        )
    )
    held = table()
    plain, _ = soft_symbol_rows(VIEW, one_hot(TERMS), TERMS, held, policy=POLICY)
    odd, _ = soft_symbol_rows(VIEW, one_hot(TERMS), invalid, held, policy=POLICY)

    assert torch.equal(plain.tensor, odd.tensor)


def test_a_distribution_over_other_axes_is_refused() -> None:
    mass = one_hot(TERMS)
    renamed = LogDistribution(
        mass.tensor, ("term", "node", "symbol"), normalized_over="symbol"
    )

    with pytest.raises(ValueError, match="a type distribution runs over"):
        soft_symbol_rows(VIEW, renamed, TERMS, table(), policy=POLICY)


def test_a_distribution_masked_against_another_occupancy_is_refused() -> None:
    spread = torch.full((len(TERMS.rows), CAPACITY, N_SYMBOLS), -math.log(N_SYMBOLS))
    masked = LogDistribution(
        spread,
        DISTRIBUTION_AXES,
        torch.zeros((len(TERMS.rows), CAPACITY), dtype=torch.bool),
        normalized_over="symbol",
    )

    with pytest.raises(ValueError, match="differs from the structure"):
        soft_symbol_rows(VIEW, masked, TERMS, table(), policy=POLICY)


def test_a_distribution_of_another_term_count_is_refused() -> None:
    mass = one_hot(TERMS)
    single = BatchEncoding((TERMS.rows[0],))

    with pytest.raises(ValueError, match="term and slot shape"):
        soft_symbol_rows(VIEW, mass, single, table(), policy=POLICY)


def test_a_structural_channel_of_another_width_is_refused() -> None:
    narrow = BatchEncoding(
        (Encoding((4, 3), (Encoding.NO_PARENT, 0), (0, 1), (True, True)),)
    )

    with pytest.raises(ValueError, match="parent slots beside a capacity"):
        soft_symbol_rows(
            VIEW,
            one_hot(BatchEncoding((TERMS.rows[0],))),
            narrow,
            table(),
            policy=POLICY,
        )


def test_a_parent_naming_no_slot_is_refused(
    reload_with_checks: Callable[..., None],
) -> None:
    reload_with_checks(_structure, on=True)
    stray = BatchEncoding(
        (Encoding((4, 3, 2), (Encoding.NO_PARENT, 0, CAPACITY), DEPTHS, LIVE),)
    )
    single = BatchEncoding((TERMS.rows[0],))

    with pytest.raises(ValueError, match="names no slot"):
        _structure.soft_symbol_rows(
            VIEW, one_hot(single), stray, table(), policy=POLICY
        )


def test_a_negative_depth_is_refused(
    reload_with_checks: Callable[..., None],
) -> None:
    reload_with_checks(_structure, on=True)
    sunken = BatchEncoding((Encoding((4, 3, 2), PARENTS, (0, 1, -1), LIVE),))
    single = BatchEncoding((TERMS.rows[0],))

    with pytest.raises(ValueError, match="depth of term 0 slot 2"):
        _structure.soft_symbol_rows(
            VIEW, one_hot(single), sunken, table(), policy=POLICY
        )
