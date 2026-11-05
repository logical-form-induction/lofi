"""Tensor construction for integer-coded corpus rows."""

from __future__ import annotations

from typing import TYPE_CHECKING, final

import torch

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor

__all__ = ["TorchCodeTensorizer"]


@final
class TorchCodeTensorizer:
    """Construct padded integer rows and their presence mask as torch tensors."""

    __slots__ = ()

    def __call__(
        self,
        rows: Sequence[tuple[int, ...]],
        /,
        *,
        padding: int,
    ) -> tuple[Tensor, Tensor]:
        """Build padded integer rows and their boolean presence mask.

        Parameters
        ----------
        rows : Sequence[tuple[int, ...]]
            Integer rows before padding.
        padding : int
            Code used to pad shorter rows.

        Returns
        -------
        tuple[Tensor, Tensor]
            Padded integer rows and a boolean mask over positions holding codes.
        """
        held = tuple(tuple(row) for row in rows)
        width = max((len(row) for row in held), default=0)
        vectors = torch.full((len(held), width), padding, dtype=torch.long)
        present = torch.zeros((len(held), width), dtype=torch.bool)

        for position, row in enumerate(held):
            if row:
                vectors[position, : len(row)] = torch.tensor(row, dtype=torch.long)
                present[position, : len(row)] = True

        return vectors, present
