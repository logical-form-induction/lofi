import pytest
import torch

from lofi.nn.codec import PlacedEncoding
from lofi.terms import Encoding

SYMBOLS = torch.tensor([[4, 1, 0, 0, 2, 0, 0], [1, 0, 0, 0, 0, 0, 0]])
PARENTS = torch.tensor([Encoding.NO_PARENT, 0, 1, 1, 0, 4, 4]).expand(2, 7)
DEPTHS = torch.tensor([0, 1, 2, 2, 1, 2, 2]).expand(2, 7)
OCCUPIED = torch.tensor(
    [
        [True, True, False, False, True, False, False],
        [True, False, False, False, False, False, False],
    ]
)


def placed(
    symbols: torch.Tensor = SYMBOLS,
    parents: torch.Tensor = PARENTS,
    depths: torch.Tensor = DEPTHS,
    mask: torch.Tensor = OCCUPIED,
    payloads: torch.Tensor | None = None,
) -> PlacedEncoding:
    return PlacedEncoding(symbols, parents, depths, mask, payloads)


def test_the_occupied_count_is_summed_over_a_mask_with_a_gap_in_it() -> None:
    encoding = placed()

    assert encoding.lengths.tolist() == [3, 1]
    assert encoding.lengths.dtype is torch.int64


def test_the_channels_come_back_with_their_axes_named() -> None:
    encoding = placed()

    assert encoding.axes == ("term", "slot")
    assert encoding.term_count == 2
    assert encoding.width == 7
    assert encoding.device == SYMBOLS.device
    assert encoding.symbols.tolist()[0] == [4, 1, 0, 0, 2, 0, 0]
    assert encoding.parents.tolist()[0][:2] == [Encoding.NO_PARENT, 0]
    assert encoding.depths.tolist()[1] == [0, 1, 2, 2, 1, 2, 2]
    assert encoding.mask.tolist()[1][0]
    assert encoding.payloads is None


def test_a_payload_channel_rides_beside_the_symbols() -> None:
    payloads = torch.zeros(2, 7, dtype=torch.int64)
    payloads[0, 4] = 3

    assert placed(payloads=payloads).payloads is payloads


def test_a_float_channel_is_refused() -> None:
    with pytest.raises(TypeError, match="symbols holds int64 codes"):
        placed(symbols=SYMBOLS.to(torch.float32))


def test_a_channel_without_a_slot_axis_is_refused() -> None:
    with pytest.raises(ValueError, match="rank is 2, not 1"):
        placed(symbols=SYMBOLS[0])


def test_a_channel_of_another_shape_than_the_symbols_is_refused() -> None:
    with pytest.raises(ValueError, match=r"depths has shape \(1, 7\)"):
        placed(depths=DEPTHS[:1])


def test_an_integer_mask_is_refused() -> None:
    with pytest.raises(TypeError, match="mask holds booleans"):
        placed(mask=OCCUPIED.to(torch.int64))


def test_a_channel_on_another_device_is_refused() -> None:
    with pytest.raises(ValueError, match="payloads is on meta beside symbols on cpu"):
        placed(payloads=torch.zeros(2, 7, dtype=torch.int64, device="meta"))


def test_the_rendering_names_the_shape_and_the_payload_channel() -> None:
    assert repr(placed()) == (
        "PlacedEncoding(term_count=2, width=7, "
        f"device={SYMBOLS.device!r}, payloads=False)"
    )
