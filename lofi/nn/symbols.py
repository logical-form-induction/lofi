"""A learned vector per codec symbol, read through an explicit norm cap."""

from typing import TYPE_CHECKING, override

import torch
from torch import nn

from lofi._checks import ensure_finite
from lofi.nn._guards import positive

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["SymbolTable"]


def _checked_cap(cap: float | None) -> float | None:
    """Return a norm cap.

    Parameters
    ----------
    cap : float | None
        The largest norm a row is read at, or None to read rows as stored.

    Raises
    ------
    TypeError
        If the cap is not a number.
    ValueError
        If the cap is not finite or not above zero.
    """
    if cap is None:
        return None

    bound = ensure_finite("a symbol table's norm cap", cap)

    if bound <= 0.0:
        msg = f"a symbol table's norm cap lies above zero, not at {bound}"
        raise ValueError(msg)

    return bound


class SymbolTable(nn.Module):
    """One learned vector per codec symbol, read through a norm cap.

    The stored weight is what training updates. What an encoder reads is ``rows``,
    where every row whose norm exceeds the cap is scaled down onto it, so a hard
    lookup and a soft mixture of the same rows see one table. The projection is a
    function of the weight and changes nothing in place.

    Parameters
    ----------
    symbols : int
        How many rows to hold, one per codec symbol.
    width : int
        How many features each row has.
    cap : float | None
        The largest norm a row is read at, or None to read rows as stored.

    Raises
    ------
    TypeError
        If a count is not an integer or the cap is not a number.
    ValueError
        If a count is not positive, or the cap is not finite or not above zero.
    """

    def __init__(self, *, symbols: int, width: int, cap: float | None = None) -> None:
        super().__init__()
        rows = positive("a symbol table's size", symbols)
        features = positive("a symbol table's width", width)
        self.weight = nn.Parameter(torch.randn(rows, features))
        self._cap = _checked_cap(cap)

    @property
    def cap(self) -> float | None:
        """Return the largest norm a row is read at, or None for no cap."""
        return self._cap

    @property
    def symbols(self) -> int:
        """Return how many rows the table holds."""
        return int(self.weight.shape[0])

    @property
    def width(self) -> int:
        """Return how many features each row has."""
        return int(self.weight.shape[1])

    @property
    def rows(self) -> Tensor:
        """Return every row, over the symbol axis and the feature axis.

        Returns
        -------
        Tensor
            The weight, with each row whose norm exceeds the cap scaled down onto
            it and every other row as stored.
        """
        weight = self.weight

        if self._cap is None:
            return weight

        norms = weight.norm(dim=1, keepdim=True)

        return weight * (self._cap / norms.clamp_min(self._cap))

    @override
    def extra_repr(self) -> str:
        """Return the size, the width and the cap."""
        return f"symbols={self.symbols}, width={self.width}, cap={self._cap}"
