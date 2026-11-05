"""Model computations represented as nodes, edges, and sites.

Nodes distinguish packed-forest items, including their readings and normal-form
tags. Edges follow the direction of the forest's inside recursion.
"""

from typing import TYPE_CHECKING, override

from lofi._checks import ensure_non_negative
from lofi.deduce import ItemKey
from lofi.nn.sites import ComponentId, Role, Site

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from lofi.deduce import Cell, Chart

__all__ = ["ComputationGraph", "GraphEdge", "GraphNode"]


class GraphNode[T]:
    """One computation step: a component concluding one forest item.

    Parameters
    ----------
    component : ComponentId
        The component that took the step.
    key : ItemKey[T]
        The item concluded, which names its span, its reading and its normal-form
        tag.

    Raises
    ------
    TypeError
        If the key is not an item key or its span is not a pair of integers.
    ValueError
        If the component is empty or the span is empty, reversed, or negative.
    """

    __slots__ = ("_component", "_key")

    def __init__(self, component: ComponentId, key: ItemKey[T]) -> None:
        if not component:
            msg = "a graph node names the component that took the step"
            raise ValueError(msg)

        if type(key) is not ItemKey:
            msg = f"a graph node concludes a forest item, not {key!r}"
            raise TypeError(msg)

        try:
            given_start, given_end = key.span
        except (TypeError, ValueError) as unpacking:
            msg = f"a span is a start and an end, not {key.span!r}"
            raise TypeError(msg) from unpacking

        start = ensure_non_negative("a span's start", given_start)
        end = ensure_non_negative("a span's end", given_end)

        if end <= start:
            msg = f"a span covers nothing: [{start}, {end})"
            raise ValueError(msg)

        self._component = component
        self._key: ItemKey[T] = key

    @property
    def component(self) -> ComponentId:
        """Return the component that took the step."""
        return self._component

    @property
    def key(self) -> ItemKey[T]:
        """Return the item the step concluded."""
        return self._key

    @property
    def span(self) -> Cell:
        """Return the half-open span the step covered."""
        return self._key.span

    def _identity(self) -> tuple[str, ItemKey[T]]:
        """Return the two fields a node is equal on."""
        return (self._component, self._key)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value names the same step."""
        if type(other) is not GraphNode:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the component and the item."""
        return hash((GraphNode, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this node."""
        return f"GraphNode({self._component!r}, {self._key!r})"


class GraphEdge[T]:
    """A directed connection between computation steps.

    Parameters
    ----------
    source : GraphNode[T]
        The step whose value was used.
    target : GraphNode[T]
        The step that used it.
    role : Role
        What the value played on the way.

    Raises
    ------
    TypeError
        If an endpoint is not a node, or the role is not a role.
    """

    __slots__ = ("_role", "_source", "_target")

    def __init__(self, source: GraphNode[T], target: GraphNode[T], role: Role) -> None:
        for endpoint in (source, target):
            if type(endpoint) is not GraphNode:
                msg = f"an edge runs between graph nodes, not {endpoint!r}"
                raise TypeError(msg)

        if not isinstance(role, Role):
            msg = f"an edge's role is a Role, not {role!r}"
            raise TypeError(msg)

        self._source: GraphNode[T] = source
        self._target: GraphNode[T] = target
        self._role = role

    @property
    def source(self) -> GraphNode[T]:
        """Return the step whose value was used."""
        return self._source

    @property
    def target(self) -> GraphNode[T]:
        """Return the step that used the value."""
        return self._target

    @property
    def role(self) -> Role:
        """Return the role the value played."""
        return self._role

    def _identity(self) -> tuple[GraphNode[T], GraphNode[T], Role]:
        """Return the three fields an edge is equal on."""
        return (self._source, self._target, self._role)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value records the same three fields."""
        if type(other) is not GraphEdge:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the endpoints and the role."""
        return hash((GraphEdge, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this edge."""
        return f"GraphEdge({self._source!r}, {self._target!r}, {self._role})"


def _a_node_on_a_cycle[T](
    nodes: frozenset[GraphNode[T]], edges: frozenset[GraphEdge[T]]
) -> GraphNode[T] | None:
    """Return one node that no topological order can place, or None if all can.

    Parameters
    ----------
    nodes : frozenset[GraphNode[T]]
        Every node in the graph.
    edges : frozenset[GraphEdge[T]]
        Every edge between them.

    Returns
    -------
    GraphNode[T] | None
        A node inside or downstream of a cycle, or None where the edges are acyclic.
    """
    incoming = dict.fromkeys(nodes, 0)
    outgoing: dict[GraphNode[T], list[GraphNode[T]]] = {node: [] for node in nodes}

    for edge in edges:
        incoming[edge.target] += 1
        outgoing[edge.source].append(edge.target)

    ready = [node for node, count in incoming.items() if count == 0]
    placed = 0

    while ready:
        node = ready.pop()
        placed += 1

        for target in outgoing[node]:
            incoming[target] -= 1

            if incoming[target] == 0:
                ready.append(target)

    if placed == len(nodes):
        return None

    return next(node for node, count in incoming.items() if count > 0)


def _checked_nodes[T](nodes: Iterable[GraphNode[T]]) -> frozenset[GraphNode[T]]:
    """Return the steps of a graph.

    Parameters
    ----------
    nodes : Iterable[GraphNode[T]]
        The steps to check, in any order and with repeats.

    Returns
    -------
    frozenset[GraphNode[T]]
        Them, deduplicated.

    Raises
    ------
    TypeError
        If a member is not a graph node.
    """
    held = frozenset(nodes)

    for node in held:
        if type(node) is not GraphNode:
            msg = f"a computation graph holds graph nodes, not {node!r}"
            raise TypeError(msg)

    return held


def _checked_edges[T](
    edges: Iterable[GraphEdge[T]], nodes: frozenset[GraphNode[T]]
) -> frozenset[GraphEdge[T]]:
    """Return the edges of a graph.

    Parameters
    ----------
    edges : Iterable[GraphEdge[T]]
        The edges to check, in any order and with repeats.
    nodes : frozenset[GraphNode[T]]
        The steps the graph holds.

    Returns
    -------
    frozenset[GraphEdge[T]]
        Them, deduplicated.

    Raises
    ------
    TypeError
        If a member is not a graph edge.
    ValueError
        If an endpoint is not one of the graph's steps.
    """
    drawn = frozenset(edges)

    for edge in drawn:
        if type(edge) is not GraphEdge:
            msg = f"a computation graph holds graph edges, not {edge!r}"
            raise TypeError(msg)

        for endpoint in (edge.source, edge.target):
            if endpoint not in nodes:
                msg = f"an edge reaches {endpoint!r}, which is not a node here"
                raise ValueError(msg)

    return drawn


def _checked_sites[T](
    sites: Mapping[GraphNode[T], Site] | None, nodes: frozenset[GraphNode[T]]
) -> dict[GraphNode[T], Site]:
    """Return where each step's value is published.

    Parameters
    ----------
    sites : Mapping[GraphNode[T], Site] | None
        The published point of each step, or None where none is published.
    nodes : frozenset[GraphNode[T]]
        The steps the graph holds.

    Returns
    -------
    dict[GraphNode[T], Site]
        A copy.

    Raises
    ------
    TypeError
        If a value is not a site.
    ValueError
        If a key is not one of the graph's steps.
    """
    published: dict[GraphNode[T], Site] = {}

    for node, site in (sites or {}).items():
        if not isinstance(site, Site):
            msg = f"a graph node is published at a Site, not at {site!r}"
            raise TypeError(msg)

        if node not in nodes:
            msg = f"a site is published for {node!r}, which is not a node here"
            raise ValueError(msg)

        published[node] = site

    return published


def _forest_items[T, W](chart: Chart[T, W]) -> tuple[ItemKey[T], ...]:
    """Return every item a forest holds, narrowest spans first.

    Parameters
    ----------
    chart : Chart[T, W]
        The forest to read.
    """
    return tuple(
        key for span in chart.index.spans_by_length() for key in chart.items_at(span)
    )


def _ensure_covered[T](
    items: tuple[ItemKey[T], ...], sites: Mapping[ItemKey[T], Site]
) -> None:
    """Check that the published sites cover the forest's items exactly.

    Parameters
    ----------
    items : tuple[ItemKey[T], ...]
        Every item the forest holds.
    sites : Mapping[ItemKey[T], Site]
        The published point of each item.

    Raises
    ------
    TypeError
        If a published point is not a site.
    ValueError
        If an item has no published point, or a point is published for an item
        the forest does not hold.
    """
    held = frozenset(items)

    for key, site in sites.items():
        if not isinstance(site, Site):
            msg = f"a forest item is published at a Site, not at {site!r}"
            raise TypeError(msg)

        if key not in held:
            msg = f"a site is published for {key!r}, which the forest does not hold"
            raise ValueError(msg)

    for key in items:
        if key not in sites:
            msg = f"the forest holds {key!r}, which is published at no site"
            raise ValueError(msg)


class ComputationGraph[T]:
    """Model steps, edges, and published sites.

    Parameters
    ----------
    nodes : Iterable[GraphNode[T]]
        Every step, in any order.
    edges : Iterable[GraphEdge[T]]
        Every claim that one step fed another.
    sites : Mapping[GraphNode[T], Site] | None
        The published point each step's value can be read at.

    Raises
    ------
    TypeError
        If a node, an edge or a site is not of its declared type.
    ValueError
        If an edge has an endpoint that is not a node, if the mapping names a step that
        is not a node, or if the edges form a cycle.
    """

    __slots__ = ("_edges", "_nodes", "_sites")

    def __init__(
        self,
        nodes: Iterable[GraphNode[T]],
        edges: Iterable[GraphEdge[T]],
        sites: Mapping[GraphNode[T], Site] | None = None,
    ) -> None:
        held = _checked_nodes(nodes)
        drawn = _checked_edges(edges, held)
        published = _checked_sites(sites, held)
        offender = _a_node_on_a_cycle(held, drawn)

        if offender is not None:
            msg = f"the edges cycle, through {offender!r} among others"
            raise ValueError(msg)

        self._nodes: frozenset[GraphNode[T]] = held
        self._edges: frozenset[GraphEdge[T]] = drawn
        self._sites: dict[GraphNode[T], Site] = published

    @classmethod
    def from_forest[W](
        cls, chart: Chart[T, W], sites: Mapping[ItemKey[T], Site]
    ) -> ComputationGraph[T]:
        """Return the graph of a packed forest's own dataflow.

        Parameters
        ----------
        chart : Chart[T, W]
            The forest whose items are the steps and whose alternatives feed the
            edges.
        sites : Mapping[ItemKey[T], Site]
            The published point each item's value can be read at, covering the
            forest's items exactly. A node's component is its site's, and an edge's
            role is the role of the point its source is published at.

        Returns
        -------
        ComputationGraph[T]
            One node per item, and one edge from each premise of each alternative
            to the item the alternative concluded. Two alternatives with the same
            premise and conclusion are one edge.

        Raises
        ------
        TypeError
            If a published point is not a site.
        ValueError
            If an item has no published point, if a point is published for an item
            the forest does not hold, if an alternative consumes an item the forest
            does not hold, or if the forest's alternatives cycle.
        """
        items = _forest_items(chart)
        _ensure_covered(items, sites)
        nodes = {key: GraphNode(sites[key].component, key) for key in items}
        edges: set[GraphEdge[T]] = set()

        for key in items:
            for alternative in chart.edges(key):
                for premise in alternative.children:
                    if premise not in nodes:
                        msg = (
                            f"an alternative of {key!r} consumes {premise!r}, "
                            "which the forest does not hold"
                        )
                        raise ValueError(msg)

                    edges.add(
                        GraphEdge(nodes[premise], nodes[key], sites[premise].role)
                    )

        return cls(
            nodes.values(),
            edges,
            {node: sites[key] for key, node in nodes.items()},
        )

    @property
    def nodes(self) -> frozenset[GraphNode[T]]:
        """Return the model's computation steps."""
        return self._nodes

    @property
    def edges(self) -> frozenset[GraphEdge[T]]:
        """Return the directed connections between computation steps."""
        return self._edges

    def site_of(self, node: GraphNode[T]) -> Site | None:
        """Return the published point a step's value can be read at.

        Parameters
        ----------
        node : GraphNode[T]
            The step to look up.

        Returns
        -------
        Site | None
            The point, or None where the model publishes none for that step.
        """
        return self._sites.get(node)

    def _identity(
        self,
    ) -> tuple[
        frozenset[GraphNode[T]],
        frozenset[GraphEdge[T]],
        frozenset[tuple[GraphNode[T], Site]],
    ]:
        """Return the three fields a graph is equal on, none of them ordered."""
        return (self._nodes, self._edges, frozenset(self._sites.items()))

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value records the same steps, edges and sites."""
        if type(other) is not ComputationGraph:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the nodes, the edges and the published points."""
        return hash((ComputationGraph, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return how many steps, edges and published points the graph holds."""
        return (
            f"ComputationGraph(nodes={len(self._nodes)}, "
            f"edges={len(self._edges)}, sites={len(self._sites)})"
        )
