"""Structural relations of a term layout, served under one component's declaration."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, NamedTuple

import torch

from lofi.nn.codec._ports import CodecMetadata, TermCoding
from lofi.nn.ports import (
    Consumption,
    StructuralRelation,
    describe_structure,
)
from lofi.terms import Encoding

if TYPE_CHECKING:
    from torch import Tensor

    from lofi.nn.codec._view import NeuralCodecView
    from lofi.nn.policy import DevicePolicy
    from lofi.nn.ports import StructuralDeclaration

__all__ = ["DeclaredStructure", "reflexive_closure"]


def _checked_declaration(declaration: StructuralDeclaration) -> StructuralDeclaration:
    """Return a read-only copy of a declaration keyed by relations with consumptions.

    Parameters
    ----------
    declaration : StructuralDeclaration
        The relations a component reads, each with how it uses it.

    Returns
    -------
    StructuralDeclaration
        The same entries in the same order, in a mapping that neither the caller
        nor anyone reading it back can change.

    Raises
    ------
    TypeError
        If a key is not a relation or a value is not a consumption.
    """
    for relation, consumption in declaration.items():
        if not isinstance(relation, StructuralRelation):
            msg = f"a declaration is keyed by StructuralRelation, not {relation!r}"
            raise TypeError(msg)

        if not isinstance(consumption, Consumption):
            msg = (
                f"a declaration maps {relation.value} to a Consumption, "
                f"not {consumption!r}"
            )
            raise TypeError(msg)

    return MappingProxyType(dict(declaration))


class _Tree(NamedTuple):
    """The parent function of a layout whose slots hold fixed tree positions.

    Attributes
    ----------
    parents : tuple[int, ...]
        The slot each slot descends from, with the root sentinel at the root.
    depths : tuple[int, ...]
        The number of edges between each slot and the root.
    sides : tuple[int, ...]
        The argument position each slot occupies under the slot it descends from,
        which is its rank in slot order among the slots sharing its parent, and zero
        at the root.
    """

    parents: tuple[int, ...]
    depths: tuple[int, ...]
    sides: tuple[int, ...]


def _sides(parents: tuple[int, ...]) -> tuple[int, ...]:
    """Return the argument position of each slot under its parent.

    Parameters
    ----------
    parents : tuple[int, ...]
        The slot each slot descends from, with the root sentinel at the root.

    Returns
    -------
    tuple[int, ...]
        The rank of each slot in slot order among the slots sharing its parent, and
        zero at the root.
    """
    seen = [0] * len(parents)
    sides = []

    for parent in parents:
        if parent == Encoding.NO_PARENT:
            sides.append(0)
            continue

        sides.append(seen[parent])
        seen[parent] += 1

    return tuple(sides)


def reflexive_closure(step: Tensor) -> Tensor:
    """Return the reflexive transitive closure of a relation over slots.

    Parameters
    ----------
    step : Tensor
        The relation, of rank two over the reading slot and the read slot, in
        boolean dtype.

    Returns
    -------
    Tensor
        True where the read slot is the reading slot or is reached from it by a
        chain of steps, of the relation's own shape and dtype.
    """
    reached = step | torch.eye(step.shape[0], dtype=torch.bool, device=step.device)

    while True:
        extended = reached | (reached.to(torch.int64) @ step.to(torch.int64)).gt(0)

        if torch.equal(extended, reached):
            return reached

        reached = extended


class DeclaredStructure[C: CodecMetadata = TermCoding]:
    """The relations one component declared, read from one codec's layout.

    Each accessor returns the relation's tensor for the layout: one value per slot
    for a unary relation, and one value per ordered pair of slots for a pairwise
    relation, the first slot of the pair being the one that reads. A pairwise
    accessor returns the same tensor whichever consumption the component declared.

    Parameters
    ----------
    codec : NeuralCodecView[C]
        The codec whose layout the relations are read from.
    declaration : StructuralDeclaration
        The relations the component reads, each with how it uses it.

    Raises
    ------
    TypeError
        If a key of the declaration is not a relation or a value is not a
        consumption.
    ValueError
        If the declaration names a relation the codec does not serve.
    """

    __slots__ = ("_capacity", "_codec", "_declaration", "_served", "_tree")

    def __init__(
        self, codec: NeuralCodecView[C], declaration: StructuralDeclaration
    ) -> None:
        served = codec.served
        held = _checked_declaration(declaration)

        for relation in held:
            if relation not in served:
                msg = (
                    f"the declaration {describe_structure(declaration)} names "
                    f"{relation.value}, which this codec does not serve: it fixes "
                    f"the structure per term rather than per slot, so it serves "
                    f"{', '.join(sorted(one.value for one in served))} alone"
                )
                raise ValueError(msg)

        skeleton = codec.codec.skeleton
        self._capacity = codec.capacity
        self._codec = codec
        self._declaration = held
        self._served = served
        self._tree = (
            None
            if skeleton is None
            else _Tree(skeleton.parents, skeleton.depths, _sides(skeleton.parents))
        )

    @property
    def declaration(self) -> StructuralDeclaration:
        """Return the relations the component reads, each with how it uses it."""
        return self._declaration

    @property
    def served(self) -> frozenset[StructuralRelation]:
        """Return the relations the codec fixes for every term."""
        return self._served

    def _undeclared(self, relation: StructuralRelation) -> str:
        """Return the message for a relation the declaration does not name.

        Parameters
        ----------
        relation : StructuralRelation
            The relation that was asked for.
        """
        return (
            f"the declaration {describe_structure(self._declaration)} does not "
            f"name {relation.value}, so this component does not read it"
        )

    def _fixed(self, relation: StructuralRelation) -> _Tree:
        """Return the layout's parent function, for a declared relation.

        A layout that fixes no parent function serves the position of a slot
        alone, so a declaration it accepted names no relation read from here.

        Parameters
        ----------
        relation : StructuralRelation
            The relation being read.

        Raises
        ------
        ValueError
            If the declaration does not name the relation.
        """
        tree = self._tree

        if tree is None or relation not in self._declaration:
            raise ValueError(self._undeclared(relation))

        return tree

    def _parent_matrix(self, tree: _Tree, policy: DevicePolicy) -> Tensor:
        """Return the parent relation of a tree as a placed boolean matrix.

        Parameters
        ----------
        tree : _Tree
            The parent function to read.
        policy : DevicePolicy
            Where this run's tensors live.

        Returns
        -------
        Tensor
            True where the read slot is the reading slot's parent, of rank two over
            the reading slot and the read slot. The root's sentinel names no slot,
            so its row is false throughout.
        """
        parents = torch.tensor(tree.parents, dtype=torch.int64, device=policy.device)
        slots = torch.arange(self._capacity, device=policy.device)

        return parents.unsqueeze(1).eq(slots.unsqueeze(0))

    def position(self, *, policy: DevicePolicy) -> Tensor:
        """Return the index of each slot in the linearization.

        Parameters
        ----------
        policy : DevicePolicy
            Where this run's tensors live.

        Returns
        -------
        Tensor
            Rank 1 over the slot, in ``int64``.

        Raises
        ------
        ValueError
            If the declaration does not name the relation.
        """
        if StructuralRelation.POSITION not in self._declaration:
            raise ValueError(self._undeclared(StructuralRelation.POSITION))

        return torch.arange(self._capacity, dtype=torch.int64, device=policy.device)

    def depth(self, *, policy: DevicePolicy) -> Tensor:
        """Return the number of edges between each slot and the root.

        Parameters
        ----------
        policy : DevicePolicy
            Where this run's tensors live.

        Returns
        -------
        Tensor
            Rank 1 over the slot, in ``int64``.

        Raises
        ------
        ValueError
            If the declaration does not name the relation.
        """
        tree = self._fixed(StructuralRelation.DEPTH)

        return torch.tensor(tree.depths, dtype=torch.int64, device=policy.device)

    def arity(self, *, policy: DevicePolicy) -> Tensor:
        """Return the number of child slots hanging off each slot.

        Parameters
        ----------
        policy : DevicePolicy
            Where this run's tensors live.

        Returns
        -------
        Tensor
            Rank 1 over the slot, in ``int64``.

        Raises
        ------
        ValueError
            If the declaration does not name the relation.
        """
        tree = self._fixed(StructuralRelation.ARITY)

        return self._parent_matrix(tree, policy).sum(dim=0)

    def parent(self, *, policy: DevicePolicy) -> Tensor:
        """Return whether the read slot is the reading slot's parent.

        Parameters
        ----------
        policy : DevicePolicy
            Where this run's tensors live.

        Returns
        -------
        Tensor
            Rank 2 over the reading slot and the read slot, in boolean dtype, with
            no slot related to the root.

        Raises
        ------
        ValueError
            If the declaration does not name the relation.
        """
        tree = self._fixed(StructuralRelation.PARENT)

        return self._parent_matrix(tree, policy)

    def legal_beneath(self, *, policy: DevicePolicy) -> Tensor:
        """Return which symbols may occupy a slot given the symbol at its parent.

        A symbol at the parent slot fills as many child slots as it takes
        arguments, from the leftmost. A slot it fills admits every symbol the
        layout admits there but pad, and a slot it does not fill admits pad alone.
        The root descends from no slot and admits what the layout admits there,
        regardless of the conditioning symbol.

        Parameters
        ----------
        policy : DevicePolicy
            Where this run's tensors live.

        Returns
        -------
        Tensor
            Rank 3 over the slot, the symbol at its parent and the symbol, in
            boolean dtype.

        Raises
        ------
        ValueError
            If the declaration does not name the parent relation.
        """
        tree = self._fixed(StructuralRelation.PARENT)
        codec = self._codec
        device = policy.device
        legal = codec.legality(policy=policy)
        arities = torch.tensor(codec.arities, dtype=torch.int64, device=device)
        sides = torch.tensor(tree.sides, dtype=torch.int64, device=device)
        parents = torch.tensor(tree.parents, dtype=torch.int64, device=device)
        rooted = parents.eq(Encoding.NO_PARENT).reshape(-1, 1, 1)
        pad = torch.arange(codec.n_symbols, device=device).eq(codec.codec.index.PAD)
        filled = sides.unsqueeze(1).lt(arities.unsqueeze(0))
        fitting = filled.unsqueeze(2).eq(~pad.reshape(1, 1, -1))

        return legal.unsqueeze(1) & (fitting | rooted)

    def ancestor(self, *, policy: DevicePolicy) -> Tensor:
        """Return whether the read slot is the reading slot or one it descends from.

        Parameters
        ----------
        policy : DevicePolicy
            Where this run's tensors live.

        Returns
        -------
        Tensor
            Rank 2 over the reading slot and the read slot, in boolean dtype, true
            on the diagonal.

        Raises
        ------
        ValueError
            If the declaration does not name the relation.
        """
        tree = self._fixed(StructuralRelation.ANCESTOR)

        return reflexive_closure(self._parent_matrix(tree, policy))

    def sibling(self, *, policy: DevicePolicy) -> Tensor:
        """Return whether the two slots are distinct and share a parent.

        Parameters
        ----------
        policy : DevicePolicy
            Where this run's tensors live.

        Returns
        -------
        Tensor
            Rank 2 over the reading slot and the read slot, in boolean dtype, false
            on the diagonal and at the root.

        Raises
        ------
        ValueError
            If the declaration does not name the relation.
        """
        tree = self._fixed(StructuralRelation.SIBLING)
        parent = self._parent_matrix(tree, policy)
        shared = (parent.to(torch.int64) @ parent.to(torch.int64).T).gt(0)

        return shared & ~torch.eye(
            self._capacity, dtype=torch.bool, device=policy.device
        )

    def tree_distance(self, *, policy: DevicePolicy) -> Tensor:
        """Return the number of edges on the path between the two slots.

        Parameters
        ----------
        policy : DevicePolicy
            Where this run's tensors live.

        Returns
        -------
        Tensor
            Rank 2 over the reading slot and the read slot, in ``int64``, zero on
            the diagonal.

        Raises
        ------
        ValueError
            If the declaration does not name the relation.
        """
        tree = self._fixed(StructuralRelation.TREE_DISTANCE)
        above = reflexive_closure(self._parent_matrix(tree, policy)).to(torch.int64)
        depths = torch.tensor(tree.depths, dtype=torch.int64, device=policy.device)
        shared = above @ above.T

        return depths.unsqueeze(1) + depths.unsqueeze(0) + 2 - 2 * shared

    def same_constituent(self, *, policy: DevicePolicy) -> Tensor:
        """Return whether the two slots belong to one operator application.

        Parameters
        ----------
        policy : DevicePolicy
            Where this run's tensors live.

        Returns
        -------
        Tensor
            Rank 2 over the reading slot and the read slot, in boolean dtype, true
            on the diagonal, between a slot and its parent, and between two slots
            sharing a parent.

        Raises
        ------
        ValueError
            If the declaration does not name the relation.
        """
        tree = self._fixed(StructuralRelation.SAME_CONSTITUENT)
        parent = self._parent_matrix(tree, policy)
        shared = (parent.to(torch.int64) @ parent.to(torch.int64).T).gt(0)

        return (
            shared
            | parent
            | parent.T
            | torch.eye(self._capacity, dtype=torch.bool, device=policy.device)
        )

    def span_contains(self, *, policy: DevicePolicy) -> Tensor:
        """Return whether the read slot lies within the reading slot's subtree span.

        Parameters
        ----------
        policy : DevicePolicy
            Where this run's tensors live.

        Returns
        -------
        Tensor
            Rank 2 over the reading slot and the read slot, in boolean dtype, true
            where the read slot lies between the lowest and the highest slot of the
            subtree rooted at the reading slot.

        Raises
        ------
        ValueError
            If the declaration does not name the relation.
        """
        tree = self._fixed(StructuralRelation.SPAN_CONTAINS)
        below = reflexive_closure(self._parent_matrix(tree, policy)).T
        slots = torch.arange(self._capacity, dtype=torch.int64, device=policy.device)
        lowest = torch.where(below, slots.unsqueeze(0), self._capacity).amin(dim=1)
        highest = torch.where(below, slots.unsqueeze(0), -1).amax(dim=1)

        return slots.unsqueeze(0).ge(lowest.unsqueeze(1)) & slots.unsqueeze(0).le(
            highest.unsqueeze(1)
        )
