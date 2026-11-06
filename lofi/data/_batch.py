"""Padded batches, masks, and token alignments."""

from typing import TYPE_CHECKING, NamedTuple, override

from lofi._alignment import Alignment
from lofi._checks import CHECKS, ensure_non_negative

if TYPE_CHECKING:
    from torch import Tensor

    from lofi._types import ItemId

__all__ = ["Alignment", "Batch", "BatchShape", "Mask"]

_MASK_RANK = 2
"""The item axis of a batch padding mask."""


class BatchShape(NamedTuple):
    """The mean sentence length and mean batch size of a stack's training batches."""

    sentence_length: float
    batch_size: float


class Mask:
    """A mask over nonpadding batch positions.

    Parameters
    ----------
    values : Tensor
        Boolean tensor with item and position axes.

    Raises
    ------
    TypeError
        If the tensor does not hold booleans.
    ValueError
        If the tensor is not rank 2.
    """

    __slots__ = ("_values",)

    def __init__(self, values: Tensor) -> None:
        bool_dtype = values.new_empty(()).eq(values.new_empty(())).dtype

        if values.dtype is not bool_dtype:
            msg = f"a padding mask holds booleans, not {values.dtype}"
            raise TypeError(msg)

        if values.ndim != _MASK_RANK:
            msg = (
                f"a padding mask runs over an item axis and a position axis, "
                f"so it is rank {_MASK_RANK} and not rank {values.ndim}"
            )
            raise ValueError(msg)

        self._values = values

    @property
    def values(self) -> Tensor:
        """Return the mask itself: rank 2 over the item and position axes, boolean."""
        return self._values

    @property
    def items(self) -> int:
        """Return how many items the mask covers."""
        return int(self._values.shape[0])

    @property
    def width(self) -> int:
        """Return how many positions each item is padded out to."""
        return int(self._values.shape[1])

    @override
    def __repr__(self) -> str:
        """Return the shape the mask covers."""
        return f"Mask(items={self.items}, width={self.width})"


class Batch:
    """A padded batch of featurized examples.

    Parameters
    ----------
    vectors : Tensor
        Featurizer output with items on the leading axis.
    lengths : tuple[int, ...]
        Nonpadding length of each item.
    mask : Mask
        Nonpadding positions.
    alignments : tuple[Alignment, ...]
        Token-to-row alignment for each item.
    item_ids : tuple[ItemId, ...]
        Item identifiers in batch order.

    Raises
    ------
    TypeError
        If a length is not an integer or no mask is supplied.
    ValueError
        If a length is negative or the item counts disagree. With ``LOFI_CHECK``,
        also if tensor axes, mask counts, or alignments are inconsistent.
    """

    __slots__ = ("_alignments", "_item_ids", "_lengths", "_mask", "_vectors")

    def __init__(
        self,
        vectors: Tensor,
        lengths: tuple[int, ...],
        *,
        mask: Mask,
        alignments: tuple[Alignment, ...],
        item_ids: tuple[ItemId, ...],
    ) -> None:
        self._lengths = tuple(
            ensure_non_negative(f"the length of item {position}", length)
            for position, length in enumerate(lengths)
        )

        for name, count in (
            ("alignments", len(alignments)),
            ("ids", len(item_ids)),
        ):
            if count != len(self._lengths):
                msg = (
                    f"a batch of {len(self._lengths)} lengths came with {count} {name}"
                )
                raise ValueError(msg)

        if mask is None:
            msg = "a batch requires a padding mask"
            raise TypeError(msg)

        self._vectors = vectors
        self._mask = mask
        self._alignments = tuple(alignments)
        self._item_ids = tuple(item_ids)

        if CHECKS:
            self._check()

    def _check(self) -> None:
        """Validate batch dimensions and alignments.

        Raises
        ------
        ValueError
            If batch dimensions disagree or an alignment is incomplete.
        """
        if self._vectors.ndim < 2:
            msg = (
                "batch vectors require an item axis and a position axis, "
                f"but have rank {self._vectors.ndim}"
            )
            raise ValueError(msg)

        items = int(self._vectors.shape[0])

        if len(self._lengths) != items:
            msg = (
                f"a batch declares {len(self._lengths)} lengths over vectors "
                f"whose leading axis holds {items} items"
            )
            raise ValueError(msg)

        vector_shape = tuple(int(size) for size in self._vectors.shape[:2])
        mask_shape = (self._mask.items, self._mask.width)

        if mask_shape != vector_shape:
            msg = (
                f"a batch's padding mask has shape {mask_shape}, but the vector "
                f"item and position axes have shape {vector_shape}"
            )
            raise ValueError(msg)

        width = max(self._lengths, default=0)
        if self._mask.width != width:
            msg = (
                f"a batch's padding mask has width {self._mask.width}, but its "
                f"longest item has length {width}"
            )
            raise ValueError(msg)

        live = tuple(int(count) for count in self._mask.values.sum(dim=1).tolist())
        if live != self._lengths:
            msg = (
                f"a batch's padding mask has row counts {live}, but its lengths "
                f"are {self._lengths}"
            )
            raise ValueError(msg)

        for position, alignment in enumerate(self._alignments):
            if not alignment.is_total:
                msg = (
                    f"item {position} leaves tokens {alignment.uncovered} on no "
                    "row, so a chart index over its tokens reaches nothing"
                )
                raise ValueError(msg)

    @property
    def vectors(self) -> Tensor:
        """Return the featurizer's output, items along the leading axis."""
        return self._vectors

    @property
    def lengths(self) -> tuple[int, ...]:
        """Return how many positions of each item hold data, in item order."""
        return self._lengths

    @property
    def mask(self) -> Mask:
        """Return which positions hold data."""
        return self._mask

    @property
    def alignments(self) -> tuple[Alignment, ...]:
        """Return the token-to-row correspondence for each item, in item order."""
        return self._alignments

    @property
    def item_ids(self) -> tuple[ItemId, ...]:
        """Return the id of each item, in item order."""
        return self._item_ids

    def __len__(self) -> int:
        """Return how many items the batch holds."""
        return len(self._lengths)

    @override
    def __repr__(self) -> str:
        """Return how many items the batch holds and how wide it is padded."""
        return f"Batch(items={len(self._lengths)}, width={self._mask.width})"
