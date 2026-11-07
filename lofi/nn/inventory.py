"""Shared-DAG execution over one batch of encoded terms."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Final, NamedTuple, Never, Self, final, override

import torch
from torch import Tensor, nn
from torch.nn import functional

from lofi.terms import Encoding

if TYPE_CHECKING:
    from lofi.terms import BatchEncoding, ChoiceIndex

__all__ = [
    "EncodeLevel",
    "EntangledInventory",
    "InventoryEmbeddings",
    "InventoryTopology",
    "LevelBatch",
]

_EMBEDDING_RANK: Final = 2


@final
class LevelBatch:
    """Inputs for one level of a bottom-up shared-DAG pass."""

    __slots__ = ("_child_mask", "_children", "_choices", "_slots")

    def __init__(
        self,
        choices: Tensor,
        slots: Tensor,
        children: Tensor | None,
        child_mask: Tensor,
    ) -> None:
        object.__setattr__(self, "_choices", choices)
        object.__setattr__(self, "_slots", slots)
        object.__setattr__(self, "_children", children)
        object.__setattr__(self, "_child_mask", child_mask)

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        del name, value
        msg = "an inventory level batch is immutable"
        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        del name
        msg = "an inventory level batch is immutable"
        raise AttributeError(msg)

    def __copy__(self) -> Self:
        """Return this immutable level batch."""
        return self

    def __deepcopy__(self, memo: dict[int, Self]) -> Self:
        """Return this immutable level batch."""
        del memo
        return self

    @property
    def choices(self) -> Tensor:
        """Return payload-aware choice codes for this level's nodes."""
        return self._choices

    @property
    def slots(self) -> Tensor:
        """Return each node's layout address."""
        return self._slots

    @property
    def children(self) -> Tensor | None:
        """Return child embeddings, or none at the leaf level."""
        return self._children

    @property
    def child_mask(self) -> Tensor:
        """Return which child-axis positions are occupied."""
        return self._child_mask

    @property
    def arities(self) -> Tensor:
        """Return the child count of every node."""
        return self._child_mask.sum(dim=1)


type EncodeLevel = Callable[[LevelBatch], Tensor]


class _BuiltTopology(NamedTuple):
    choices: tuple[int, ...]
    slots: tuple[int, ...]
    children: tuple[tuple[int, ...], ...]
    levels: tuple[range, ...]
    roots: tuple[int, ...]
    occurrences: tuple[tuple[int, ...], ...]


type _NodeKey = tuple[int, int, tuple[int, ...]]


def _child_slots(
    parents: Sequence[int], occupied: Sequence[bool], /
) -> tuple[tuple[int, ...], ...]:
    """Return occupied children in layout order at every slot."""
    held: list[list[int]] = [[] for _ in parents]

    for child, (parent, live) in enumerate(zip(parents, occupied, strict=True)):
        if live and parent != Encoding.NO_PARENT:
            held[parent].append(child)

    return tuple(tuple(children) for children in held)


def _build_topology(terms: BatchEncoding, index: ChoiceIndex) -> _BuiltTopology:
    """Hash-cons equal nodes at equal layout addresses across batch rows."""
    raw_choices: list[int] = []
    raw_slots: list[int] = []
    raw_children: list[tuple[int, ...]] = []
    raw_levels: list[int] = []
    by_key: dict[_NodeKey, int] = {}
    roots: list[int] = []
    occurrences: list[tuple[int, ...]] = []

    for row in terms.rows:
        child_slots = _child_slots(row.parents, row.occupied)
        choices = index.codes(row)
        by_slot: dict[int, int] = {}

        for slot in sorted(
            (at for at, live in enumerate(row.occupied) if live),
            key=lambda at: row.depths[at],
            reverse=True,
        ):
            children = tuple(by_slot[child] for child in child_slots[slot])
            key = (slot, choices[slot], children)
            held = by_key.get(key)

            if held is None:
                level = (
                    0
                    if not children
                    else max(raw_levels[child] for child in children) + 1
                )
                held = len(raw_choices)
                by_key[key] = held
                raw_choices.append(choices[slot])
                raw_slots.append(slot)
                raw_children.append(children)
                raw_levels.append(level)

            by_slot[slot] = held

        row_roots = tuple(
            by_slot[slot]
            for slot, live in enumerate(row.occupied)
            if live and row.parents[slot] == Encoding.NO_PARENT
        )

        if len(row_roots) > 1:
            msg = f"an encoded term has {len(row_roots)} occupied roots, not one"
            raise ValueError(msg)

        roots.append(Encoding.NO_PARENT if not row_roots else row_roots[0])
        occurrences.append(
            tuple(by_slot.get(slot, Encoding.NO_PARENT) for slot in range(row.width))
        )

    if not raw_choices:
        return _BuiltTopology((), (), (), (), tuple(roots), tuple(occurrences))

    order = sorted(range(len(raw_choices)), key=lambda node: (raw_levels[node], node))
    remap = {old: new for new, old in enumerate(order)}
    ordered_levels = tuple(raw_levels[node] for node in order)
    levels: list[range] = []
    start = 0

    for level in range(max(ordered_levels) + 1):
        stop = start + ordered_levels.count(level)
        levels.append(range(start, stop))
        start = stop

    return _BuiltTopology(
        tuple(raw_choices[node] for node in order),
        tuple(raw_slots[node] for node in order),
        tuple(tuple(remap[child] for child in raw_children[node]) for node in order),
        tuple(levels),
        tuple(
            Encoding.NO_PARENT if root == Encoding.NO_PARENT else remap[root]
            for root in roots
        ),
        tuple(
            tuple(
                Encoding.NO_PARENT if node == Encoding.NO_PARENT else remap[node]
                for node in row
            )
            for row in occurrences
        ),
    )


@final
class InventoryTopology:
    """Immutable shared-DAG topology for one batch of encoded terms."""

    __slots__ = (
        "_children",
        "_choices",
        "_levels",
        "_occurrences",
        "_roots",
        "_slots",
    )

    def __init__(self, terms: BatchEncoding, index: ChoiceIndex, /) -> None:
        built = _build_topology(terms, index)
        object.__setattr__(self, "_choices", built.choices)
        object.__setattr__(self, "_slots", built.slots)
        object.__setattr__(self, "_children", built.children)
        object.__setattr__(self, "_levels", built.levels)
        object.__setattr__(self, "_roots", built.roots)
        object.__setattr__(self, "_occurrences", built.occurrences)

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        del name, value
        msg = "an inventory topology is immutable"
        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        del name
        msg = "an inventory topology is immutable"
        raise AttributeError(msg)

    def __copy__(self) -> Self:
        """Return this immutable topology."""
        return self

    def __deepcopy__(self, memo: dict[int, Self]) -> Self:
        """Return this immutable topology."""
        del memo
        return self

    @property
    def choices(self) -> tuple[int, ...]:
        """Return the choice code of every shared node."""
        return self._choices

    @property
    def slots(self) -> tuple[int, ...]:
        """Return the layout address of every shared node."""
        return self._slots

    @property
    def children(self) -> tuple[tuple[int, ...], ...]:
        """Return each shared node's children in argument order."""
        return self._children

    @property
    def levels(self) -> tuple[range, ...]:
        """Return bottom-up ranges over the shared nodes."""
        return self._levels

    @property
    def roots(self) -> tuple[int, ...]:
        """Return each term's root node."""
        return self._roots

    @property
    def occurrences(self) -> tuple[tuple[int, ...], ...]:
        """Return the shared node at each term and layout address."""
        return self._occurrences

    @property
    def node_count(self) -> int:
        """Return how many distinct addressed nodes are shared."""
        return len(self._choices)

    @property
    def term_count(self) -> int:
        """Return how many term rows the topology represents."""
        return len(self._roots)

    @property
    def max_arity(self) -> int:
        """Return the greatest child count among shared nodes."""
        return max(map(len, self._children), default=0)


@final
class InventoryEmbeddings:
    """Ephemeral shared-node embeddings from one forward pass."""

    __slots__ = ("_nodes", "_occurrences", "_roots")

    def __init__(self, nodes: Tensor, roots: Tensor, occurrences: Tensor) -> None:
        object.__setattr__(self, "_nodes", nodes)
        object.__setattr__(self, "_roots", roots)
        object.__setattr__(self, "_occurrences", occurrences)

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        del name, value
        msg = "an inventory embedding pass is immutable"
        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        del name
        msg = "an inventory embedding pass is immutable"
        raise AttributeError(msg)

    @property
    def nodes(self) -> Tensor:
        """Return the distinct shared-node embeddings."""
        return self._nodes

    @property
    def terms(self) -> Tensor:
        """Return root embeddings, at zero for empty term rows."""
        if self._roots.numel() == 0:
            return self._nodes.new_zeros((0, self._nodes.shape[-1]))

        valid = self._roots >= 0
        indices = self._roots.clamp_min(0)
        gathered = functional.embedding(indices, self._nodes)

        return gathered.masked_fill(~valid.unsqueeze(-1), 0.0)

    def placed(self) -> Tensor:
        """Return node embeddings restored to every batch slot."""
        if self._occurrences.numel() == 0:
            return self._nodes.new_zeros(
                (*self._occurrences.shape, self._nodes.shape[-1])
            )

        valid = self._occurrences >= 0
        gathered = functional.embedding(self._occurrences.clamp_min(0), self._nodes)

        return gathered.masked_fill(~valid.unsqueeze(-1), 0.0)


class EntangledInventory(nn.Module):
    """Run one encoder call per topological level of a shared inventory DAG."""

    _child_mask: Tensor
    _children: Tensor
    _choices: Tensor
    _occurrences: Tensor
    _roots: Tensor
    _slots: Tensor

    def __init__(self, topology: InventoryTopology, /) -> None:
        super().__init__()
        self._topology = topology
        width = topology.max_arity
        children = tuple(
            row + (Encoding.NO_PARENT,) * (width - len(row))
            for row in topology.children
        )
        child_mask = tuple(
            (True,) * len(row) + (False,) * (width - len(row))
            for row in topology.children
        )
        self.register_buffer(
            "_choices",
            torch.tensor(topology.choices, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "_slots", torch.tensor(topology.slots, dtype=torch.int64), persistent=False
        )
        self.register_buffer(
            "_children",
            torch.tensor(children, dtype=torch.int64).reshape(
                topology.node_count, width
            ),
            persistent=False,
        )
        self.register_buffer(
            "_child_mask",
            torch.tensor(child_mask, dtype=torch.bool).reshape(
                topology.node_count, width
            ),
            persistent=False,
        )
        self.register_buffer(
            "_roots", torch.tensor(topology.roots, dtype=torch.int64), persistent=False
        )
        self.register_buffer(
            "_occurrences",
            torch.tensor(topology.occurrences, dtype=torch.int64).reshape(
                topology.term_count,
                len(topology.occurrences[0]) if topology.occurrences else 0,
            ),
            persistent=False,
        )

    @property
    def topology(self) -> InventoryTopology:
        """Return the immutable topology this module executes."""
        return self._topology

    @override
    def forward(self, encode_level: EncodeLevel, /) -> InventoryEmbeddings:
        """Encode every shared node exactly once, bottom up by level."""
        encoded: list[Tensor] = []
        prefix: Tensor | None = None

        for level in self._topology.levels:
            start, stop = level.start, level.stop
            mask = self._child_mask[start:stop]
            children = None

            if prefix is not None:
                children = functional.embedding(
                    self._children[start:stop].clamp_min(0), prefix
                ).masked_fill(~mask.unsqueeze(-1), 0.0)

            values = _checked_values(
                encode_level(
                    LevelBatch(
                        self._choices[start:stop],
                        self._slots[start:stop],
                        children,
                        mask,
                    )
                ),
                nodes=len(level),
                width=None if prefix is None else int(prefix.shape[1]),
            )
            encoded.append(values)
            prefix = values if prefix is None else torch.cat((prefix, values), dim=0)

        if prefix is None:
            prefix = torch.empty((0, 0), device=self._choices.device)

        return InventoryEmbeddings(prefix, self._roots, self._occurrences)


def _checked_values(values: Tensor, *, nodes: int, width: int | None) -> Tensor:
    if not values.is_floating_point():
        msg = f"node embeddings must be floating point, not {values.dtype}"
        raise TypeError(msg)
    if values.ndim != _EMBEDDING_RANK:
        msg = f"a level encoder returns node by width, not rank {values.ndim}"
        raise ValueError(msg)
    if values.shape[0] != nodes:
        msg = f"a level holds {nodes} nodes, but its encoder returned {values.shape[0]}"
        raise ValueError(msg)
    if width is not None and values.shape[1] != width:
        msg = f"earlier levels have width {width}, but this level has {values.shape[1]}"
        raise ValueError(msg)

    return values
