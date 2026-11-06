"""Pieces every Transformer component in LoFI shares."""

from types import MappingProxyType
from typing import TYPE_CHECKING, Final, final, override

import torch
from torch import nn

from lofi._frozen import FrozenValue
from lofi.nn.ports import StandardDeclaration
from lofi.nn.sites import Role
from lofi.registry import Registry, RegistryError

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

    from lofi.nn.ports import StructuralDeclaration
    from lofi.spec.layers import TransformerStructure

__all__ = [
    "ANCESTOR_AND_SIBLING_BIAS",
    "ANCESTOR_MASK",
    "BLOCK_INPUT",
    "BLOCK_OUTPUT",
    "BLOCK_POST_ATTENTION",
    "PARENT_RECURRENCE",
    "POSITIONS",
    "POSITIONS_WITH_DEPTH",
    "AttentionBlock",
    "BlockObserver",
    "TransformerStructurePlugin",
    "transformer_structure",
]

BLOCK_INPUT: Final = "input"
BLOCK_POST_ATTENTION: Final = "post-attention"
BLOCK_OUTPUT: Final = "output"
"""The residual stages an attention block reports within each layer."""

type BlockObserver = Callable[
    [Role, tuple[int | str, ...], Tensor, Tensor | None], None
]
"""Receive one forward value at a role and key, with the slots it is live at."""


class TransformerStructurePlugin(FrozenValue):
    """Supply the structural relations visible to Transformer attention."""

    __slots__ = ("_declaration",)
    _noun = "transformer-structure plugin"

    def __init__(self, declaration: StructuralDeclaration, /) -> None:
        object.__setattr__(self, "_declaration", MappingProxyType(dict(declaration)))

    @property
    def declaration(self) -> StructuralDeclaration:
        """Return the structural declaration exposed to the attention stack."""
        return self._declaration


def transformer_structure(name: TransformerStructure, /) -> StructuralDeclaration:
    """Return the structural declaration installed under ``name``.

    Raises
    ------
    RegistryError
        If no installed ``lofi.transformer_structure`` entry is named ``name``, or
        the entry names something other than a structure plugin.
    """
    member = Registry[object]("transformer_structure").get(name)

    if not isinstance(member, TransformerStructurePlugin):
        msg = (
            f"the transformer_structure entry {name!r} names {member!r}, "
            "not a Transformer-structure plugin"
        )
        raise RegistryError(msg, family="transformer_structure")

    return member.declaration


def _where_live(occupied: Tensor | None, value: Tensor, /) -> Tensor:
    """Return a slot-major value zeroed at unoccupied slots, or as given."""
    if occupied is None:
        return value

    return torch.where(occupied.unsqueeze(-1), value, torch.zeros_like(value))


@final
class AttentionBlock(nn.Module):
    """One pre-norm Transformer layer whose attention takes an additive term.

    Every Transformer term encoder and decoder in LoFI stacks this layer. The
    attention term carries the declared structure: a mask is a term of minus
    infinity off the relation, a bias a learned term on it.

    Parameters
    ----------
    width : int
        The model and stream width.
    heads : int
        The number of attention heads, which divides the width.
    inner_width : int
        The hidden width of the feedforward block.
    activation : nn.Module
        The elementwise map the feedforward block applies.
    """

    def __init__(
        self, width: int, heads: int, inner_width: int, activation: nn.Module
    ) -> None:
        super().__init__()
        self._heads = heads
        self._activation = activation
        self._attention_norm = nn.LayerNorm(width)
        self._queries_keys_values = nn.Linear(width, 3 * width)
        self._merge = nn.Linear(width, width)
        self._feedforward_norm = nn.LayerNorm(width)
        self._expand = nn.Linear(width, inner_width)
        self._contract = nn.Linear(inner_width, width)

    @override
    def forward(
        self,
        stream: Tensor,
        attention: Tensor,
        /,
        *,
        occupied: Tensor | None = None,
        layer: int = 0,
        observe: BlockObserver | None = None,
    ) -> Tensor:
        """Return the stream after one layer.

        Parameters
        ----------
        stream : Tensor
            Rank three over the term, the slot and the width.
        attention : Tensor
            The term added to the attention weights before they are normalized, of
            rank four over the term, the head, the reading slot and the read slot,
            of size one over an axis it does not vary along.
        occupied : Tensor | None
            True where a slot holds a node, over the term and slot axes, so that
            an unoccupied slot reads and carries nothing; None when every slot is
            live.
        layer : int
            This layer's zero-based index in its stack, which keys what it
            reports.
        observe : BlockObserver | None
            Receive the residual stream, the attention logits and weights of each
            head and the feedforward preactivation, or None to report nothing.

        Returns
        -------
        Tensor
            The stream, of the same shape.
        """
        if observe is not None:
            observe(Role.RESIDUAL_STREAM, (layer, BLOCK_INPUT), stream, occupied)

        terms, slots, width = stream.shape
        queries, keys, values = (
            self._queries_keys_values(self._attention_norm(stream))
            .reshape(terms, slots, 3, self._heads, width // self._heads)
            .permute(2, 0, 3, 1, 4)
        )
        logits = (
            torch.matmul(queries, keys.transpose(-2, -1))
            / (width // self._heads) ** 0.5
        )
        logits = logits + attention

        if occupied is not None:
            live_queries = occupied.unsqueeze(1).unsqueeze(-1)
            logits = torch.where(live_queries, logits, torch.zeros_like(logits))
            weights = torch.softmax(logits, dim=-1)
            weights = torch.where(live_queries, weights, torch.zeros_like(weights))
        else:
            weights = torch.softmax(logits, dim=-1)

        if observe is not None:
            for head in range(self._heads):
                observe(Role.ATTENTION_LOGITS, (layer, head), logits[:, head], occupied)
                observe(
                    Role.ATTENTION_WEIGHTS, (layer, head), weights[:, head], occupied
                )

        mixed = torch.matmul(weights, values)
        merged = self._merge(mixed.transpose(1, 2).reshape(terms, slots, width))
        post_attention = _where_live(occupied, stream + merged)

        if observe is not None:
            observe(
                Role.RESIDUAL_STREAM,
                (layer, BLOCK_POST_ATTENTION),
                post_attention,
                occupied,
            )

        preactivation = _where_live(
            occupied, self._expand(self._feedforward_norm(post_attention))
        )

        if observe is not None:
            observe(Role.FFN_PREACTIVATION, (layer,), preactivation, occupied)

        feedforward = self._contract(self._activation(preactivation))
        output = _where_live(occupied, post_attention + feedforward)

        if observe is not None:
            observe(Role.RESIDUAL_STREAM, (layer, BLOCK_OUTPUT), output, occupied)

        return output


POSITIONS: Final = TransformerStructurePlugin(StandardDeclaration.POSITIONS.declaration)
POSITIONS_WITH_DEPTH: Final = TransformerStructurePlugin(
    StandardDeclaration.POSITIONS_WITH_DEPTH.declaration
)
ANCESTOR_MASK: Final = TransformerStructurePlugin(
    StandardDeclaration.ANCESTOR_MASK.declaration
)
ANCESTOR_AND_SIBLING_BIAS: Final = TransformerStructurePlugin(
    StandardDeclaration.ANCESTOR_AND_SIBLING_BIAS.declaration
)
PARENT_RECURRENCE: Final = TransformerStructurePlugin(
    StandardDeclaration.PARENT_RECURRENCE.declaration
)
