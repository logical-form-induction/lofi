"""Formalism-independent gold analyses."""

from types import MappingProxyType
from typing import TYPE_CHECKING, override

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable, Mapping

    from lofi.data._values import TokenSpan

__all__ = [
    "Dependency",
    "Derivation",
    "DerivationNode",
    "GraphEdge",
    "GraphNode",
    "NodeId",
    "RuleName",
]

type NodeId = int
"""A node identifier local to one derivation."""

type RuleName = str
"""The name of the rule that licensed a node."""

type _Features = tuple[tuple[str, str], ...]
"""Feature key-value pairs in key order."""


def _normalized(features: Mapping[str, str] | None, subject: str) -> _Features:
    """Return a feature mapping in key order.

    Parameters
    ----------
    features : Mapping[str, str] | None
        Feature mapping, if present.
    subject : str
        Subject used in validation messages.

    Returns
    -------
    _Features
        The key and value pairs, ordered by key.

    Raises
    ------
    ValueError
        If a feature key is empty.
    """
    if features is None:
        return ()

    for key in features:
        if not key:
            msg = f"{subject} has a feature with an empty key"
            raise ValueError(msg)

    return tuple(sorted(features.items()))


class DerivationNode[L: Hashable]:
    """A node in a gold derivation.

    Parameters
    ----------
    span : TokenSpan
        Covered token span.
    label : L
        Node category.
    rule : RuleName | None
        Rule name, if recorded.
    children : tuple[NodeId, ...]
        Child node identifiers in source order.
    features : Mapping[str, str] | None
        Additional node features.

    Raises
    ------
    ValueError
        If the span, child identifiers, rule name, or feature keys are invalid.
    """

    __slots__ = ("_children", "_features", "_label", "_rule", "_span")

    def __init__(
        self,
        span: TokenSpan,
        label: L,
        *,
        rule: RuleName | None = None,
        children: tuple[NodeId, ...] = (),
        features: Mapping[str, str] | None = None,
    ) -> None:
        if span.length == 0:
            msg = f"a derivation node covers no token: [{span.start}, {span.end})"
            raise ValueError(msg)

        if rule is not None and not rule:
            msg = "a derivation node names an empty rule"
            raise ValueError(msg)

        for child in children:
            if child < 0:
                msg = f"a derivation node names a negative child id: {child}"
                raise ValueError(msg)

        if len(set(children)) != len(children):
            msg = f"a derivation node names one child twice: {children}"
            raise ValueError(msg)

        self._span = span
        self._label = label
        self._rule = rule
        self._children = children
        self._features = _normalized(features, "a derivation node")

    @property
    def span(self) -> TokenSpan:
        """Return the token positions the node covers."""
        return self._span

    @property
    def label(self) -> L:
        """Return the node's category in its corpus algebra."""
        return self._label

    @property
    def rule(self) -> RuleName | None:
        """Return the name of the rule that licensed the node."""
        return self._rule

    @property
    def children(self) -> tuple[NodeId, ...]:
        """Return the nodes this one was built from, by id."""
        return self._children

    @property
    def features(self) -> Mapping[str, str]:
        """Return the node's corpus-specific features."""
        return MappingProxyType(dict(self._features))

    @property
    def is_lexical(self) -> bool:
        """Return whether the node is a leaf."""
        return not self._children

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value is a node with the same structure."""
        if type(other) is not DerivationNode:
            return NotImplemented

        return (
            self._span == other._span
            and self._label == other._label
            and self._rule == other._rule
            and self._children == other._children
            and self._features == other._features
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over span, label, rule, children and features."""
        return hash(
            (
                DerivationNode,
                self._span,
                self._label,
                self._rule,
                self._children,
                self._features,
            )
        )

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this node."""
        return (
            f"DerivationNode({self._span!r}, {self._label!r}, rule={self._rule!r}, "
            f"children={self._children!r}, features={dict(self._features)!r})"
        )


def _walked[L: Hashable](
    root: NodeId,
    nodes: Mapping[NodeId, DerivationNode[L]],
) -> tuple[NodeId, ...]:
    """Return the node ids reachable from a root, parents before children.

    Parameters
    ----------
    root : NodeId
        The id to start from.
    nodes : Mapping[NodeId, DerivationNode[L]]
        Nodes keyed by identifier.

    Returns
    -------
    tuple[NodeId, ...]
        Reachable identifiers in preorder.

    Raises
    ------
    ValueError
        If an identifier is missing or a node is reached more than once.
    """
    order: list[NodeId] = []
    seen: set[NodeId] = set()
    pending = [root]

    while pending:
        current = pending.pop()

        if current not in nodes:
            msg = f"a derivation names node {current}, which it does not hold"
            raise ValueError(msg)

        if current in seen:
            msg = f"a derivation reaches node {current} twice"
            raise ValueError(msg)

        seen.add(current)
        order.append(current)
        pending.extend(reversed(nodes[current].children))

    return tuple(order)


def _check_tiling[L: Hashable](
    node: DerivationNode[L],
    children: Iterable[DerivationNode[L]],
) -> None:
    """Raise unless a node's children partition its span.

    Parameters
    ----------
    node : DerivationNode[L]
        The parent whose span the children must cover.
    children : Iterable[DerivationNode[L]]
        The parent's children, in any order.

    Raises
    ------
    ValueError
        If the children's spans overlap, leave a gap, or do not reach exactly
        from the parent's start to its end.
    """
    spans = sorted((child.span for child in children), key=lambda span: span.start)
    reached = node.span.start

    for span in spans:
        if span.start != reached:
            msg = (
                f"a derivation node covering [{node.span.start}, {node.span.end}) "
                f"has a child covering [{span.start}, {span.end}), which does not "
                f"continue from {reached}"
            )
            raise ValueError(msg)

        reached = span.end

    if reached != node.span.end:
        msg = (
            f"a derivation node covering [{node.span.start}, {node.span.end}) "
            f"has children reaching only to {reached}"
        )
        raise ValueError(msg)


class Derivation[L: Hashable]:
    """A rooted derivation with node confidences.

    Parameters
    ----------
    root : NodeId
        Root node identifier.
    nodes : Mapping[NodeId, DerivationNode[L]]
        Nodes keyed by identifier.
    confidences : Mapping[NodeId, float] | None
        Optional node confidences keyed by identifier.

    Raises
    ------
    ValueError
        If identifiers, reachability, span partitions, or confidence keys are
        invalid.
    """

    __slots__ = ("_by_id", "_confidences", "_nodes", "_preorder", "_root")

    def __init__(
        self,
        root: NodeId,
        nodes: Mapping[NodeId, DerivationNode[L]],
        confidences: Mapping[NodeId, float] | None = None,
    ) -> None:
        order = _walked(root, nodes)

        if len(order) != len(nodes):
            unreached = sorted(set(nodes) - set(order))
            msg = f"a derivation holds nodes unreachable from its root: {unreached}"
            raise ValueError(msg)

        for node_id in order:
            node = nodes[node_id]

            if node.children:
                _check_tiling(node, (nodes[child] for child in node.children))

        scored = {} if confidences is None else dict(confidences)

        for node_id in scored:
            if node_id not in nodes:
                msg = f"a derivation scores node {node_id}, which it does not hold"
                raise ValueError(msg)

        self._root = root
        self._by_id = dict(nodes)
        self._nodes = tuple(sorted(nodes.items()))
        self._preorder = order
        self._confidences = scored

    @property
    def root(self) -> NodeId:
        """Return the id of the node covering the whole analysis."""
        return self._root

    @property
    def span(self) -> TokenSpan:
        """Return the token positions the whole analysis covers."""
        return self.node(self._root).span

    @property
    def nodes(self) -> Mapping[NodeId, DerivationNode[L]]:
        """Return every node of the analysis, by id."""
        return MappingProxyType(dict(self._by_id))

    @property
    def preorder(self) -> tuple[NodeId, ...]:
        """Return node identifiers in root-first, left-to-right order."""
        return self._preorder

    def node(self, node_id: NodeId) -> DerivationNode[L]:
        """Return a node by identifier.

        Parameters
        ----------
        node_id : NodeId
            The id to look up.

        Returns
        -------
        DerivationNode[L]
            The requested node.

        Raises
        ------
        KeyError
            If the identifier is absent.
        """
        return self._by_id[node_id]

    def children_of(self, node_id: NodeId) -> tuple[DerivationNode[L], ...]:
        """Return a node's children in source order.

        Parameters
        ----------
        node_id : NodeId
            The id of the parent.

        Returns
        -------
        tuple[DerivationNode[L], ...]
            Its children, in the order the corpus recorded.

        Raises
        ------
        KeyError
            If the identifier is absent.
        """
        return tuple(self.node(child) for child in self.node(node_id).children)

    def leaves(self) -> tuple[DerivationNode[L], ...]:
        """Return the childless nodes, left to right.

        Returns
        -------
        tuple[DerivationNode[L], ...]
            The leaves, ordered by the position they start at.
        """
        return tuple(
            sorted(
                (node for node in self._by_id.values() if node.is_lexical),
                key=lambda leaf: leaf.span.start,
            )
        )

    def confidence_of(self, node_id: NodeId) -> float | None:
        """Return a node's recorded confidence, if present.

        Parameters
        ----------
        node_id : NodeId
            The id of the node.

        Returns
        -------
        float | None
            The confidence, or ``None`` if absent.
        """
        return self._confidences.get(node_id)

    def __len__(self) -> int:
        """Return how many nodes the analysis has."""
        return len(self._nodes)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value is the same analysis."""
        if type(other) is not Derivation:
            return NotImplemented

        return self._root == other._root and self._nodes == other._nodes

    @override
    def __hash__(self) -> int:
        """Return a hash over the root and the nodes."""
        return hash((Derivation, self._root, self._nodes))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this derivation."""
        return f"Derivation({self._root!r}, {self._by_id!r}, {self._confidences!r})"


class Dependency:
    """A labeled dependency.

    Parameters
    ----------
    head : int
        The position of the governing token.
    dependent : int
        The position of the governed token.
    label : str
        Relation label.
    slot : int | None
        Argument slot, if distinguished.

    Raises
    ------
    ValueError
        If a position or a slot is negative, or if the label is empty.
    """

    __slots__ = ("_dependent", "_head", "_label", "_slot")

    def __init__(
        self,
        head: int,
        dependent: int,
        label: str,
        slot: int | None = None,
    ) -> None:
        if head < 0:
            msg = f"a dependency has a negative head position: {head}"
            raise ValueError(msg)

        if dependent < 0:
            msg = f"a dependency has a negative dependent position: {dependent}"
            raise ValueError(msg)

        if not label:
            msg = "a dependency has no label"
            raise ValueError(msg)

        if slot is not None and slot < 0:
            msg = f"a dependency fills a negative slot: {slot}"
            raise ValueError(msg)

        self._head = head
        self._dependent = dependent
        self._label = label
        self._slot = slot

    @property
    def head(self) -> int:
        """Return the position of the governing token."""
        return self._head

    @property
    def dependent(self) -> int:
        """Return the position of the governed token."""
        return self._dependent

    @property
    def label(self) -> str:
        """Return the relation's name, in the corpus's own inventory."""
        return self._label

    @property
    def slot(self) -> int | None:
        """Return the dependent's argument position, if recorded."""
        return self._slot

    def _fields(self) -> tuple[int, int, str, int | None]:
        """Return the four fields in declaration order."""
        return (self._head, self._dependent, self._label, self._slot)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value is a dependency with the same four fields."""
        if type(other) is not Dependency:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the four fields."""
        return hash((Dependency, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this dependency."""
        return f"Dependency{self._fields()!r}"


class GraphNode:
    """A node in a meaning graph.

    Parameters
    ----------
    identifier : str
        Identifier within the graph.
    node_type : str
        Corpus-defined node type.
    label : str | None
        Optional node label.
    anchor : TokenSpan | None
        Optional token anchor.
    properties : Mapping[str, str] | None
        Additional node properties.

    Raises
    ------
    ValueError
        If the identifier or the node type is empty, if the anchor is empty, or
        if a property key is empty.
    """

    __slots__ = ("_anchor", "_identifier", "_label", "_node_type", "_properties")

    def __init__(
        self,
        identifier: str,
        node_type: str,
        *,
        label: str | None = None,
        anchor: TokenSpan | None = None,
        properties: Mapping[str, str] | None = None,
    ) -> None:
        if not identifier:
            msg = "a graph node has no identifier"
            raise ValueError(msg)

        if not node_type:
            msg = f"graph node {identifier!r} has no node type"
            raise ValueError(msg)

        if anchor is not None and anchor.length == 0:
            msg = f"graph node {identifier!r} is anchored to no token"
            raise ValueError(msg)

        self._identifier = identifier
        self._node_type = node_type
        self._label = label
        self._anchor = anchor
        self._properties = _normalized(properties, f"graph node {identifier!r}")

    @property
    def identifier(self) -> str:
        """Return the node's handle within its own graph."""
        return self._identifier

    @property
    def node_type(self) -> str:
        """Return what kind of thing the node is."""
        return self._node_type

    @property
    def label(self) -> str | None:
        """Return what the node says, if it says anything."""
        return self._label

    @property
    def anchor(self) -> TokenSpan | None:
        """Return the token positions the node is anchored to, if any."""
        return self._anchor

    @property
    def properties(self) -> Mapping[str, str]:
        """Return the node's corpus-specific features."""
        return MappingProxyType(dict(self._properties))

    def _identity(self) -> tuple[str, str, str | None, TokenSpan | None, _Features]:
        """Return the fields a graph node is equal on."""
        return (
            self._identifier,
            self._node_type,
            self._label,
            self._anchor,
            self._properties,
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value is a graph node with the same fields."""
        if type(other) is not GraphNode:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over identifier, type, label, anchor and properties."""
        return hash((GraphNode, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this node."""
        return (
            f"GraphNode({self._identifier!r}, {self._node_type!r}, "
            f"label={self._label!r}, anchor={self._anchor!r}, "
            f"properties={dict(self._properties)!r})"
        )


class GraphEdge:
    """A directed, typed graph edge.

    Parameters
    ----------
    source : str
        The identifier of the node the edge leaves.
    target : str
        The identifier of the node the edge enters.
    edge_type : str
        The relation's name, in the corpus's own vocabulary.
    label : str | None
        Optional edge label.
    properties : Mapping[str, str] | None
        Additional edge properties.

    Raises
    ------
    ValueError
        If an endpoint or the edge type is empty, or if a property key is empty.
    """

    __slots__ = ("_edge_type", "_label", "_properties", "_source", "_target")

    def __init__(
        self,
        source: str,
        target: str,
        edge_type: str,
        *,
        label: str | None = None,
        properties: Mapping[str, str] | None = None,
    ) -> None:
        if not source:
            msg = "a graph edge has no source"
            raise ValueError(msg)

        if not target:
            msg = "a graph edge has no target"
            raise ValueError(msg)

        if not edge_type:
            msg = f"the graph edge from {source!r} to {target!r} has no edge type"
            raise ValueError(msg)

        self._source = source
        self._target = target
        self._edge_type = edge_type
        self._label = label
        self._properties = _normalized(properties, "a graph edge")

    @property
    def source(self) -> str:
        """Return the identifier of the node the edge leaves."""
        return self._source

    @property
    def target(self) -> str:
        """Return the identifier of the node the edge enters."""
        return self._target

    @property
    def edge_type(self) -> str:
        """Return the relation's name, in the corpus's own vocabulary."""
        return self._edge_type

    @property
    def label(self) -> str | None:
        """Return what the edge says, if it says anything beyond its type."""
        return self._label

    @property
    def properties(self) -> Mapping[str, str]:
        """Return the edge's corpus-specific properties."""
        return MappingProxyType(dict(self._properties))

    def _identity(self) -> tuple[str, str, str, str | None, _Features]:
        """Return the fields a graph edge is equal on."""
        return (
            self._source,
            self._target,
            self._edge_type,
            self._label,
            self._properties,
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value is a graph edge with the same fields."""
        if type(other) is not GraphEdge:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over source, target, type, label and properties."""
        return hash((GraphEdge, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this edge."""
        return (
            f"GraphEdge({self._source!r}, {self._target!r}, {self._edge_type!r}, "
            f"label={self._label!r}, properties={dict(self._properties)!r})"
        )
