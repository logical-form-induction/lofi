"""Placement of the structural channels an encoder reads beside a soft symbol."""

from typing import TYPE_CHECKING, Final

import torch

from lofi._checks import CHECKS, ensure_non_negative
from lofi.nn._values import LogDistribution
from lofi.nn.codec._ports import CodecMetadata
from lofi.nn.codec._values import PlacedEncoding
from lofi.terms import Encoding

if TYPE_CHECKING:
    from torch import Tensor

    from lofi.nn._values import Encoded
    from lofi.nn.codec._view import NeuralCodecView
    from lofi.nn.policy import DevicePolicy
    from lofi.terms import BatchEncoding

__all__ = ["DISTRIBUTION_AXES", "soft_symbol_rows"]

DISTRIBUTION_AXES: Final[tuple[str, ...]] = ("term", "slot", "symbol")
"""The axes a distribution over the symbol at each slot runs over."""


def _place_structure[C: CodecMetadata](
    codec: NeuralCodecView[C], structure: BatchEncoding, *, policy: DevicePolicy
) -> PlacedEncoding:
    """Return a batch's parent, depth and occupancy channels, placed.

    The symbol channel comes back at zero rather than at what the batch holds, so a
    caller mixing its own symbols cannot read the hard codes by accident.

    Parameters
    ----------
    codec : NeuralCodecView[C]
        The grammar's tensor view of its term codec.
    structure : BatchEncoding
        The batch whose structural channels are wanted.
    policy : DevicePolicy
        Where this run's tensors live.

    Returns
    -------
    PlacedEncoding
        The channels, over the term axis and the slot axis, with the symbols zeroed.

    Raises
    ------
    ValueError
        If a channel is not as wide as the codec's capacity, if a parent names no slot,
        or if a depth is negative.
    """
    rows = structure.rows
    width = codec.capacity

    for term, row in enumerate(rows):
        for field, channel in (
            ("parent", row.parents),
            ("depth", row.depths),
            ("occupancy", row.occupied),
        ):
            if len(channel) != width:
                msg = (
                    f"term {term} has {len(channel)} {field} slots beside "
                    f"a capacity of {width}"
                )
                raise ValueError(msg)

        if CHECKS:
            for slot, parent in enumerate(row.parents):
                if parent != Encoding.NO_PARENT and not 0 <= parent < width:
                    msg = (
                        f"term {term} slot {slot} has parent {parent}, which "
                        "names no slot"
                    )
                    raise ValueError(msg)

            for slot, depth in enumerate(row.depths):
                ensure_non_negative(f"the depth of term {term} slot {slot}", depth)

    device = policy.device
    parents = torch.tensor(
        tuple(row.parents for row in rows), dtype=torch.int64, device=device
    ).reshape(len(rows), width)
    depths = torch.tensor(
        tuple(row.depths for row in rows), dtype=torch.int64, device=device
    ).reshape(len(rows), width)
    occupied = torch.tensor(
        tuple(row.occupied for row in rows), dtype=torch.bool, device=device
    ).reshape(len(rows), width)

    return PlacedEncoding(torch.zeros_like(parents), parents, depths, occupied)


def soft_symbol_rows[C: CodecMetadata](
    codec: NeuralCodecView[C],
    symbols: LogDistribution,
    structure: BatchEncoding,
    table: Tensor,
    *,
    policy: DevicePolicy,
) -> tuple[Encoded, PlacedEncoding]:
    """Return the table rows a symbol distribution mixes, beside its structure.

    Parameters
    ----------
    codec : NeuralCodecView[C]
        The grammar's tensor view of its term codec.
    symbols : LogDistribution
        The log probability of each symbol at each slot, normalized over the symbol
        axis.
    structure : BatchEncoding
        The parent, depth and occupancy channels aligned with the term and slot axes.
        Its symbol channel is ignored.
    table : Tensor
        The symbol embedding table, over the symbol axis and the width axis.
    policy : DevicePolicy
        Where this run's tensors live.

    Returns
    -------
    tuple[Encoded, PlacedEncoding]
        The mixed rows over the term, slot and width axes, and the placed structural
        channels they align with.

    Raises
    ------
    ValueError
        If the distribution does not use the declared axes, does not align with the
        structure in term and slot shape, or supplies a different occupancy mask.
    """
    if symbols.axes != DISTRIBUTION_AXES:
        msg = f"a type distribution runs over {DISTRIBUTION_AXES}, not {symbols.axes}"
        raise ValueError(msg)

    placed = _place_structure(codec, structure, policy=policy)
    tensor = symbols.tensor.to(device=policy.device, dtype=policy.dtype)
    mask = None if symbols.mask is None else symbols.mask.to(device=policy.device)

    if mask is not None and not torch.equal(mask, placed.mask):
        msg = "the symbol-distribution mask differs from the structure's occupancy"
        raise ValueError(msg)

    moved = LogDistribution(
        tensor, symbols.axes, mask, normalized_over=symbols.normalized_over
    )
    embedded = codec.soft_symbols(moved, table)

    if tuple(embedded.tensor.shape[:2]) != tuple(placed.parents.shape):
        msg = (
            f"the symbol distribution has term and slot shape "
            f"{tuple(embedded.tensor.shape[:2])}, not {tuple(placed.parents.shape)}"
        )
        raise ValueError(msg)

    return embedded, placed
