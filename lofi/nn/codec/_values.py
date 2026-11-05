"""Device-tensor storage for term encodings."""

from typing import TYPE_CHECKING, Final, override

import torch

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["PlacedEncoding"]

#: What the two axes of every channel mean, outermost first.
_AXES: Final[tuple[str, ...]] = ("term", "slot")


def _checked_codes(field: str, tensor: Tensor) -> Tensor:
    """Return an integer channel.

    Parameters
    ----------
    field : str
        The channel's name, for the message.
    tensor : Tensor
        The channel to check.

    Returns
    -------
    Tensor
        The channel, unchanged.

    Raises
    ------
    TypeError
        If the channel is not of ``int64`` dtype.
    ValueError
        If the channel does not have one axis for the term and one for the slot.
    """
    if tensor.dtype is not torch.int64:
        msg = f"{field} holds int64 codes, not {tensor.dtype}"
        raise TypeError(msg)

    if tensor.ndim != len(_AXES):
        msg = (
            f"{field} has axes {_AXES}, so its rank is {len(_AXES)}, not {tensor.ndim}"
        )
        raise ValueError(msg)

    return tensor


def _checked_beside(field: str, tensor: Tensor, symbols: Tensor) -> Tensor:
    """Return a channel.

    Parameters
    ----------
    field : str
        The channel's name, for the message.
    tensor : Tensor
        The channel to check.
    symbols : Tensor
        The symbol channel every other channel is laid out beside.

    Returns
    -------
    Tensor
        The channel, unchanged.

    Raises
    ------
    ValueError
        If the channel has another shape than the symbols, or is on another device.
    """
    if tensor.shape != symbols.shape:
        msg = (
            f"{field} has shape {tuple(tensor.shape)} beside symbols of shape "
            f"{tuple(symbols.shape)}"
        )
        raise ValueError(msg)

    if tensor.device != symbols.device:
        msg = f"{field} is on {tensor.device} beside symbols on {symbols.device}"
        raise ValueError(msg)

    return tensor


class PlacedEncoding:
    """Stores term-encoding channels on one device.

    Parameters
    ----------
    symbols : Tensor
        The operator code at each slot, of rank two over the term and the slot, in
        ``int64``.
    parents : Tensor
        The slot holding each slot's parent, of the same shape and dtype, with the
        encoding's no-parent sentinel at a root.
    depths : Tensor
        The distance from the root at each slot, of the same shape and dtype.
    mask : Tensor
        True where a slot holds a node, of the same shape, of boolean dtype.
    payloads : Tensor | None
        The payload code at each slot, of the same shape and dtype, or None where no
        operator of the signature takes a payload.

    Raises
    ------
    TypeError
        If a code channel is not ``int64`` or the mask is not boolean.
    ValueError
        If a channel does not have one axis for the term and one for the slot, or does
        not sit beside the symbols in shape and device.
    """

    __slots__ = ("_depths", "_lengths", "_mask", "_parents", "_payloads", "_symbols")

    def __init__(
        self,
        symbols: Tensor,
        parents: Tensor,
        depths: Tensor,
        mask: Tensor,
        payloads: Tensor | None = None,
    ) -> None:
        self._symbols = _checked_codes("symbols", symbols)
        self._parents = _checked_beside(
            "parents", _checked_codes("parents", parents), symbols
        )
        self._depths = _checked_beside(
            "depths", _checked_codes("depths", depths), symbols
        )

        if mask.dtype is not torch.bool:
            msg = f"mask holds booleans, not {mask.dtype}"
            raise TypeError(msg)

        self._mask = _checked_beside("mask", mask, symbols)
        self._payloads = (
            None
            if payloads is None
            else _checked_beside(
                "payloads", _checked_codes("payloads", payloads), symbols
            )
        )
        self._lengths = self._mask.sum(dim=-1)

    @property
    def symbols(self) -> Tensor:
        """Return the operator code at each slot, of rank two in ``int64``."""
        return self._symbols

    @property
    def parents(self) -> Tensor:
        """Return the parent slot of each slot, of rank two in ``int64``."""
        return self._parents

    @property
    def depths(self) -> Tensor:
        """Return the distance from the root at each slot, of rank two in ``int64``."""
        return self._depths

    @property
    def mask(self) -> Tensor:
        """Return the rank-2 boolean mask of occupied slots."""
        return self._mask

    @property
    def payloads(self) -> Tensor | None:
        """Return the payload code at each slot, of rank two in ``int64``, or None."""
        return self._payloads

    @property
    def lengths(self) -> Tensor:
        """Return the occupied slot count of each term, of rank one in ``int64``."""
        return self._lengths

    @property
    def axes(self) -> tuple[str, ...]:
        """Return what each axis of a channel means, outermost first."""
        return _AXES

    @property
    def term_count(self) -> int:
        """Return the number of encoded terms."""
        return int(self._symbols.shape[0])

    @property
    def width(self) -> int:
        """Return the slots allocated to each term."""
        return int(self._symbols.shape[1])

    @property
    def device(self) -> torch.device:
        """Return the device holding the encoding channels."""
        return self._symbols.device

    @override
    def __repr__(self) -> str:
        """Return the term count, the width, the device and the payload channel."""
        return (
            f"{type(self).__name__}(term_count={self.term_count}, "
            f"width={self.width}, device={self.device!r}, "
            f"payloads={self._payloads is not None})"
        )
