"""Structural views of the gold derivations consumed by neural scorers."""

from collections.abc import Hashable
from typing import Protocol

__all__ = ["DerivationTree"]


class _TokenSpan(Protocol):
    """Expose the token offsets a derivation scorer reads."""

    @property
    def start(self) -> int:
        """Return the first covered token position."""
        ...

    @property
    def end(self) -> int:
        """Return the exclusive token end position."""
        ...

    @property
    def length(self) -> int:
        """Return the number of covered token positions."""
        ...


class _DerivationNode[T: Hashable](Protocol):
    """Expose one labeled derivation node to a scorer."""

    @property
    def span(self) -> _TokenSpan:
        """Return the node's covered token span."""
        ...

    @property
    def label(self) -> T:
        """Return the node's symbolic category."""
        ...

    @property
    def rule(self) -> str | None:
        """Return the licensing rule name, when recorded."""
        ...

    @property
    def children(self) -> tuple[int, ...]:
        """Return child identifiers in source order."""
        ...

    @property
    def is_lexical(self) -> bool:
        """Return whether this node is lexical."""
        ...


class DerivationTree[T: Hashable](Protocol):
    """Expose only the rooted tree operations a neural scorer consumes."""

    @property
    def root(self) -> int:
        """Return the root node identifier."""
        ...

    @property
    def preorder(self) -> tuple[int, ...]:
        """Return node identifiers in root-first source order."""
        ...

    def node(self, node_id: int) -> _DerivationNode[T]:
        """Return one node by identifier."""
        ...

    def children_of(self, node_id: int) -> tuple[_DerivationNode[T], ...]:
        """Return one node's children in source order."""
        ...
