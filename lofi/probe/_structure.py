"""Term structure read beside the node encodings a term encoder publishes.

A term encoder publishes one point holding a vector per term and one holding a
vector per node of a term. A node reading on its own says nothing about the tree
its nodes came from, so the diagnostics that measure structure hold the
linearized terms the scope encoded and read each slot's depth, parent and
constructor off them, position by position against the reading.
"""

from typing import TYPE_CHECKING, Final, NamedTuple, override

import torch

from lofi.nn.sites import Role
from lofi.probe._roles import ROLE_AXES
from lofi.terms import Encoding

if TYPE_CHECKING:
    from torch import Tensor

    from lofi.nn.sites import Site
    from lofi.probe._diagnose import DiagnosticContext
    from lofi.probe._values import Observation, Trace
    from lofi.terms import BatchEncoding

__all__ = [
    "NODE_AXES",
    "NodeAxes",
    "NodeStructure",
    "correlation",
    "node_readings",
    "pairwise_similarity",
    "similarity",
    "strictly_above",
]


class NodeAxes(NamedTuple):
    """The axes a node encoding runs over, outermost first.

    Attributes
    ----------
    terms : str
        The axis the batch's terms run along.
    nodes : str
        The axis a term's slots run along.
    features : str
        The axis a node's coordinates run along.
    """

    terms: str
    nodes: str
    features: str


def _declared() -> NodeAxes:
    """Return the axes the node-encoding role declares, outermost first."""
    terms, nodes, features = ROLE_AXES[Role.NODE_ENCODING]

    return NodeAxes(terms, nodes, features)


NODE_AXES: Final = _declared()
"""The axes a node encoding runs over under its role's declaration."""


def similarity(left: Tensor, right: Tensor) -> Tensor:
    """Return the cosine of the angle between two stacks of vectors.

    Parameters
    ----------
    left : Tensor
        Vectors over the innermost axis.
    right : Tensor
        Vectors over the innermost axis, broadcasting against the first.

    Returns
    -------
    Tensor
        One number per broadcast position, between minus one and one, and zero
        where either vector sits at the origin and no angle is defined.
    """
    first = left.to(torch.float64)
    second = right.to(torch.float64)
    spread = first.norm(dim=-1) * second.norm(dim=-1)
    dotted = (first * second).sum(dim=-1)
    defined = spread > 0

    return torch.where(
        defined,
        dotted / spread.masked_fill(~defined, 1.0),
        torch.zeros_like(dotted),
    ).clamp(-1.0, 1.0)


def pairwise_similarity(vectors: Tensor) -> Tensor:
    """Return the cosine between every ordered pair of a stack of vectors.

    Parameters
    ----------
    vectors : Tensor
        Vectors over the innermost axis, with the axis they are stacked along
        just outside it.

    Returns
    -------
    Tensor
        The stacking axis twice in place of the feature axis, holding at each
        position the cosine between the two vectors it indexes.
    """
    return similarity(vectors.unsqueeze(-2), vectors.unsqueeze(-3))


def strictly_above(count: int) -> Tensor:
    """Return which ordered pairs of a stack take the second member later.

    Parameters
    ----------
    count : int
        How many members the stack has.

    Returns
    -------
    Tensor
        Truth values of the pairing's own shape, true above the diagonal, so a
        pair and its reverse are counted once and no member is paired with
        itself.
    """
    return torch.triu(torch.ones(count, count, dtype=torch.bool), diagonal=1)


def correlation(left: Tensor, right: Tensor) -> float:
    """Return the linear correlation of two series.

    Parameters
    ----------
    left : Tensor
        One number per observation.
    right : Tensor
        One number per observation, as many as the first holds.

    Returns
    -------
    float
        Pearson's coefficient, between minus one and one, and zero where either
        series takes one value throughout or holds nothing, since no line through
        those points is better than another.
    """
    first = left.to(torch.float64).flatten()
    second = right.to(torch.float64).flatten()
    centered = first - first.mean()
    against = second - second.mean()
    spread = centered.norm() * against.norm()

    if not bool(spread > 0):
        return 0.0

    return float(((centered * against).sum() / spread).clamp(-1.0, 1.0))


def node_readings(
    context: DiagnosticContext, trace: Trace, /
) -> tuple[tuple[Site, Observation], ...]:
    """Return the node encodings a scope read, in publication order.

    Parameters
    ----------
    context : DiagnosticContext
        What the components under study published.
    trace : Trace
        The readings the scope took.

    Returns
    -------
    tuple[tuple[Site, Observation], ...]
        Each node point the scope read, beside what was read there.

    Raises
    ------
    KeyError
        If the trace holds a reading at a point the context does not publish.
    ValueError
        If node encodings are published and the scope read none of them.
    """
    taken = dict(context.readings(trace))
    published = context.sites_for(Role.NODE_ENCODING)
    read = tuple((site, taken[site]) for site in published if site in taken)

    if published and not read:
        msg = (
            f"{len(published)} points encode nodes and the scope read none of "
            "them, so there is no encoding to measure"
        )
        raise ValueError(msg)

    return read


class NodeStructure:
    """The slots of a batch of linearized terms, aligned with one node reading.

    Parameters
    ----------
    terms : BatchEncoding | None
        The linearized terms the scope encoded, in batch order, or None where the
        diagnostic holding it was built without them.
    reading : Observation
        The node encodings read at one point.
    axes : NodeAxes
        The names the reading runs its terms, nodes and features over.

    Raises
    ------
    KeyError
        If the reading declares no axis under one of the three names.
    ValueError
        If there are no terms, or the terms and the reading disagree about how
        many terms or how many slots they hold.
    """

    __slots__ = ("_depths", "_internal", "_live", "_roots", "_symbols", "_vectors")

    def __init__(
        self,
        terms: BatchEncoding | None,
        reading: Observation,
        /,
        *,
        axes: NodeAxes = NODE_AXES,
    ) -> None:
        if terms is None:
            msg = (
                f"the encodings read at {reading.site!r} were measured by a "
                "diagnostic built without the terms they encode"
            )
            raise ValueError(msg)

        if not terms.rows:
            msg = (
                f"the encodings read at {reading.site!r} were measured against a "
                "batch holding no term"
            )
            raise ValueError(msg)

        held = (len(terms.rows), terms.rows[0].width)
        shape = (reading.extent(axes.terms), reading.extent(axes.nodes))

        if shape != held:
            msg = (
                f"the encodings read at {reading.site!r} run over {shape[0]} terms "
                f"of {shape[1]} slots, and the batch holds {held[0]} terms of "
                f"{held[1]} slots"
            )
            raise ValueError(msg)

        order = (
            reading.position(axes.terms),
            reading.position(axes.nodes),
            reading.position(axes.features),
        )
        parents = torch.tensor(terms.parents, dtype=torch.int64)
        occupied = torch.tensor(terms.mask, dtype=torch.bool)

        self._vectors = reading.tensor.permute(order).to(torch.float64)
        self._live = occupied & reading.covering().permute(order).all(dim=-1)
        self._depths = torch.tensor(terms.depths, dtype=torch.int64)
        self._symbols = torch.tensor(terms.symbols, dtype=torch.int64)
        self._roots = self._live & (parents == Encoding.NO_PARENT)

        children = torch.zeros_like(parents)
        children.scatter_add_(
            1, parents.clamp(min=0), (self._live & (parents >= 0)).to(torch.int64)
        )
        self._internal = children > 0

    @property
    def vectors(self) -> Tensor:
        """Return the encoding of each slot, over terms, slots and coordinates."""
        return self._vectors

    @property
    def live(self) -> Tensor:
        """Return which slots hold a node the reading also holds data for."""
        return self._live

    @property
    def depths(self) -> Tensor:
        """Return how many edges lie between each slot and the root above it."""
        return self._depths

    @property
    def symbols(self) -> Tensor:
        """Return the constructor code at each slot."""
        return self._symbols

    @property
    def internal(self) -> Tensor:
        """Return which slots have a live slot hanging off them."""
        return self._internal

    def counts(self) -> Tensor:
        """Return how many live slots each term holds.

        Returns
        -------
        Tensor
            One count per term.
        """
        return self._live.sum(dim=1)

    def rooted(self) -> tuple[Tensor, Tensor]:
        """Return which terms hold a root and the encoding at each one's root.

        Returns
        -------
        tuple[Tensor, Tensor]
            Truth values over the terms, true where a live slot has no parent
            above it, and one vector per term, which is the first slot's where no
            live slot is a root.
        """
        present = self._roots.any(dim=1)
        at = self._roots.to(torch.int64).argmax(dim=1)
        width = self._vectors.shape[-1]
        taken = self._vectors.gather(1, at.reshape(-1, 1, 1).expand(-1, 1, width))

        return present, taken.squeeze(1)

    @override
    def __repr__(self) -> str:
        """Return how many terms and slots the structure covers and how many live."""
        terms, slots = self._live.shape

        return (
            f"NodeStructure({terms} terms, {slots} slots, {int(self._live.sum())} live)"
        )
