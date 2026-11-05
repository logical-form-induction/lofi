"""Chart items, edge products, and derivations."""

from itertools import product
from math import exp, expm1, inf
from typing import TYPE_CHECKING, NamedTuple, override

if TYPE_CHECKING:
    from collections.abc import Iterator

    from lofi.deduce._spans import Cell

__all__ = ["Derivation", "Edge", "EdgeProduct", "ItemKey", "PrunedMass"]


class PrunedMass(NamedTuple):
    """The goal probability mass a pruned parse kept, against an exhaustive one.

    Both masses are goal-relative log partitions: the log of the summed
    probability of every goal derivation, once without pruning and once with
    the pruner the parse declared.

    Attributes
    ----------
    exhaustive : float
        The log mass of the parse without pruning.
    retained : float
        The log mass of the pruned parse.
    """

    exhaustive: float
    retained: float

    @property
    def lost(self) -> float:
        """Return the probability mass the pruner removed."""
        return exp(self.exhaustive) - exp(self.retained)

    @property
    def lost_fraction(self) -> float:
        """Return the share of the exhaustive mass the pruner removed.

        A parse with no goal derivation at all has no mass to lose, so its
        share is zero.
        """
        if self.exhaustive == -inf:
            return 0.0

        return -expm1(self.retained - self.exhaustive)


class ItemKey[T]:
    """Identifies a chart item.

    An item is one exact conclusion over one span under one normal-form tag.
    Two conclusions that a rule set's matching identifies, such as two feature
    variants of one category, are two items: the matching decides which rules
    apply, never which conclusions the chart tells apart.

    Parameters
    ----------
    span : Cell
        The half-open token interval the item covers.
    reading : T
        The exact conclusion.
    nf_tag : str
        Normal-form tag, or an empty string when unused.
    """

    __slots__ = ("_nf_tag", "_reading", "_span")

    def __init__(self, span: Cell, reading: T, nf_tag: str = "") -> None:
        self._span = span
        self._reading: T = reading
        self._nf_tag = nf_tag

    @property
    def span(self) -> Cell:
        """Return the half-open token interval the item covers."""
        return self._span

    @property
    def reading(self) -> T:
        """Return the exact conclusion."""
        return self._reading

    @property
    def nf_tag(self) -> str:
        """Return the normal-form tag, or an empty string if no normal form ran."""
        return self._nf_tag

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether two keys name the same span, reading and tag.

        Parameters
        ----------
        other : object
            The value to compare against.
        """
        if type(other) is not ItemKey:
            return NotImplemented

        return (
            self._span == other._span
            and self._reading == other._reading
            and self._nf_tag == other._nf_tag
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the span, the reading and the tag."""
        return hash((ItemKey, self._span, self._reading, self._nf_tag))

    @override
    def __repr__(self) -> str:
        """Return a rendering naming the span, the reading and the tag."""
        return (
            f"ItemKey(span={self._span!r}, reading={self._reading!r}, "
            f"nf_tag={self._nf_tag!r})"
        )


class Edge[T, W]:
    """One packed-forest hyperedge.

    Parameters
    ----------
    rule_name : str
        Applied rule name or lexical tag.
    children : tuple[ItemKey[T], ...]
        Premise item keys, or an empty tuple for an axiom.
    output : T
        Unquotiented conclusion.
    weight : W
        Local semiring weight.
    split : int | None
        Binary split point, or ``None`` for another arity.
    order : int
        Spine order, or zero when no spine variable exists.
    """

    __slots__ = ("_children", "_order", "_output", "_rule_name", "_split", "_weight")

    def __init__(
        self,
        rule_name: str,
        children: tuple[ItemKey[T], ...],
        output: T,
        weight: W,
        split: int | None = None,
        order: int = 0,
    ) -> None:
        self._rule_name = rule_name
        self._children: tuple[ItemKey[T], ...] = children
        self._output: T = output
        self._weight: W = weight
        self._split = split
        self._order = order

    @property
    def rule_name(self) -> str:
        """Return the rule name or lexical-axiom tag."""
        return self._rule_name

    @property
    def children(self) -> tuple[ItemKey[T], ...]:
        """Return the items consumed, in premise order."""
        return self._children

    @property
    def output(self) -> T:
        """Return the term produced by the step."""
        return self._output

    @property
    def weight(self) -> W:
        """Return the step's semiring weight."""
        return self._weight

    @property
    def split(self) -> int | None:
        """Return the binary split point, or None for a nonbinary step."""
        return self._split

    @property
    def order(self) -> int:
        """Return the rule instance's spine order, or zero if it has no spine."""
        return self._order

    @property
    def arity(self) -> int:
        """Return how many items the step consumed."""
        return len(self._children)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether two edges agree on all six fields.

        Parameters
        ----------
        other : object
            The value to compare against.
        """
        if type(other) is not Edge:
            return NotImplemented

        return (
            self._rule_name == other._rule_name
            and self._children == other._children
            and self._output == other._output
            and self._weight == other._weight
            and self._split == other._split
            and self._order == other._order
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over every field but the weight."""
        return hash(
            (
                Edge,
                self._rule_name,
                self._children,
                self._output,
                self._split,
                self._order,
            )
        )

    @override
    def __repr__(self) -> str:
        """Return a rendering naming the rule, the children and the weight."""
        return (
            f"Edge(rule_name={self._rule_name!r}, children={self._children!r}, "
            f"output={self._output!r}, weight={self._weight!r}, "
            f"split={self._split!r}, order={self._order!r})"
        )


class EdgeProduct[T, W]:
    """One exact Cartesian product of packed-forest hyperedges.

    Every axis contains the alternative items for one premise. The other
    fields are shared by every edge in the Cartesian product.
    """

    __slots__ = ("_axes", "_order", "_output", "_rule_name", "_split", "_weight")

    def __init__(
        self,
        rule_name: str,
        axes: tuple[tuple[ItemKey[T], ...], ...],
        output: T,
        weight: W,
        split: int | None = None,
        order: int = 0,
    ) -> None:
        self._rule_name = rule_name
        self._axes: tuple[tuple[ItemKey[T], ...], ...] = axes
        self._output: T = output
        self._weight: W = weight
        self._split = split
        self._order = order

    @property
    def rule_name(self) -> str:
        """Return the rule name or lexical-axiom tag."""
        return self._rule_name

    @property
    def axes(self) -> tuple[tuple[ItemKey[T], ...], ...]:
        """Return premise alternatives in premise order."""
        return self._axes

    @property
    def output(self) -> T:
        """Return the term produced by every represented edge."""
        return self._output

    @property
    def weight(self) -> W:
        """Return the local semiring weight shared by the represented edges."""
        return self._weight

    @property
    def split(self) -> int | None:
        """Return the binary split point, or None for another arity."""
        return self._split

    @property
    def order(self) -> int:
        """Return the rule instance's spine order."""
        return self._order

    @property
    def arity(self) -> int:
        """Return how many premise axes the product contains."""
        return len(self._axes)

    def edges(self) -> Iterator[Edge[T, W]]:
        """Expand represented edges lazily in premise-axis order."""
        for children in product(*self._axes):
            yield Edge(
                self._rule_name,
                children,
                self._output,
                self._weight,
                self._split,
                self._order,
            )

    def contains(self, edge: Edge[T, W], /) -> bool:
        """Return whether the product represents an exact edge."""
        return (
            edge.rule_name == self._rule_name
            and edge.output == self._output
            and edge.weight == self._weight
            and edge.split == self._split
            and edge.order == self._order
            and len(edge.children) == len(self._axes)
            and all(
                child in axis
                for child, axis in zip(edge.children, self._axes, strict=True)
            )
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether two products agree on every field."""
        if type(other) is not EdgeProduct:
            return NotImplemented

        return (
            self._rule_name == other._rule_name
            and self._axes == other._axes
            and self._output == other._output
            and self._weight == other._weight
            and self._split == other._split
            and self._order == other._order
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over every field but the weight."""
        return hash(
            (
                EdgeProduct,
                self._rule_name,
                self._axes,
                self._output,
                self._split,
                self._order,
            )
        )

    @override
    def __repr__(self) -> str:
        """Return a rendering naming the rule, axes and weight."""
        return (
            f"EdgeProduct(rule_name={self._rule_name!r}, axes={self._axes!r}, "
            f"output={self._output!r}, weight={self._weight!r}, "
            f"split={self._split!r}, order={self._order!r})"
        )


class Derivation[T, W]:
    """One unpacked derivation.

    Parameters
    ----------
    key : ItemKey[T]
        The item this derivation concludes.
    edge : Edge[T, W]
        The step at the root.
    children : tuple[Derivation[T, W], ...]
        One sub-derivation per premise of that edge, in premise order.
    """

    __slots__ = ("_children", "_edge", "_key")

    def __init__(
        self,
        key: ItemKey[T],
        edge: Edge[T, W],
        children: tuple[Derivation[T, W], ...] = (),
    ) -> None:
        self._key: ItemKey[T] = key
        self._edge: Edge[T, W] = edge
        self._children: tuple[Derivation[T, W], ...] = children

    @property
    def key(self) -> ItemKey[T]:
        """Return the item this derivation concludes."""
        return self._key

    @property
    def edge(self) -> Edge[T, W]:
        """Return the step at the root."""
        return self._edge

    @property
    def children(self) -> tuple[Derivation[T, W], ...]:
        """Return one sub-derivation per premise."""
        return self._children

    @override
    def __repr__(self) -> str:
        """Return a rendering naming the item, the rule and the arity."""
        return (
            f"Derivation(key={self._key!r}, edge={self._edge.rule_name!r}, "
            f"children={len(self._children)})"
        )
