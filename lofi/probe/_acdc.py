"""Budgeted receiver-local ACDC edge pruning."""

from collections.abc import Mapping
from math import isfinite
from time import perf_counter
from types import MappingProxyType
from typing import TYPE_CHECKING, override

import torch

from lofi._checks import ensure_non_negative
from lofi._frozen import FrozenValue
from lofi.deduce import ItemKey
from lofi.nn.sites import Access, Role, Site, SiteMap
from lofi.probe._attribution import EdgeAttribution
from lofi.probe._circuits import Unit
from lofi.probe._graph import ComputationGraph, GraphEdge, GraphNode
from lofi.probe._instrument import instrument
from lofi.probe._interventions import PatchFrom, Within
from lofi.probe._provenance import Provenance

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from torch import Tensor

    from lofi._types import JsonValue
    from lofi.deduce import SpanIndex
    from lofi.nn.ports import Reading, ReceiverEdge, ReceiverLocalChartScorer
    from lofi.probe._ports import Probed
    from lofi.probe._values import Observation, Trace

__all__ = [
    "ACDCResult",
    "AbsoluteThreshold",
    "PruningBudget",
    "PruningThreshold",
    "ReceiverCircuit",
    "RelativeThreshold",
    "acdc",
    "matched_cardinality_jaccard",
    "receiver_circuit",
]


def _finite_non_negative(name: str, value: float) -> float:
    """Return a finite, nonnegative real number."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"{name} is not a number: {value!r}"
        raise TypeError(msg)

    held = float(value)

    if not isfinite(held) or held < 0.0:
        msg = f"{name} is not finite and nonnegative: {value!r}"
        raise ValueError(msg)

    return held


class PruningBudget(FrozenValue):
    """A hard upper bound on the candidate edges one ACDC run tests.

    Parameters
    ----------
    maximum_trials : int
        The nonnegative maximum number of candidate forwards.
    """

    __slots__ = ("_maximum_trials",)

    _noun = "pruning budget"

    def __init__(self, maximum_trials: int) -> None:
        object.__setattr__(
            self,
            "_maximum_trials",
            ensure_non_negative("a pruning budget's maximum trials", maximum_trials),
        )

    @property
    def maximum_trials(self) -> int:
        """Return the most candidate forwards the run may make."""
        return self._maximum_trials

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another budget permits the same number of trials."""
        if type(other) is not PruningBudget:
            return NotImplemented

        return self._maximum_trials == other._maximum_trials

    @override
    def __hash__(self) -> int:
        """Return a hash over the trial bound."""
        return hash((PruningBudget, self._maximum_trials))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this budget."""
        return f"PruningBudget({self._maximum_trials!r})"


class AbsoluteThreshold(FrozenValue):
    """Prune a candidate whose divergence increment is below a fixed bound.

    Parameters
    ----------
    bound : float
        The finite nonnegative strict upper bound on an accepted increment, in
        the units of the declared divergence.
    """

    __slots__ = ("_bound",)

    _noun = "pruning threshold"

    def __init__(self, bound: float) -> None:
        object.__setattr__(
            self, "_bound", _finite_non_negative("an absolute threshold's bound", bound)
        )

    @property
    def bound(self) -> float:
        """Return the strict upper bound on an accepted increment."""
        return self._bound

    def cutoff(self, full_corruption: float, /) -> float:
        """Return the bound, whatever the full-corruption divergence.

        Parameters
        ----------
        full_corruption : float
            The divergence of the counterfactual output from the clean output.
        """
        del full_corruption

        return self._bound

    def to_spec(self) -> dict[str, JsonValue]:
        """Return the declaration that rebuilds this threshold."""
        return {"kind": "absolute", "bound": self._bound}

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another threshold is absolute at the same bound."""
        if type(other) is not AbsoluteThreshold:
            return NotImplemented

        return self._bound == other._bound

    @override
    def __hash__(self) -> int:
        """Return a hash over the bound."""
        return hash((AbsoluteThreshold, self._bound))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this threshold."""
        return f"AbsoluteThreshold({self._bound!r})"


class RelativeThreshold(FrozenValue):
    """Prune a candidate whose increment is below a share of full corruption.

    The cutoff is the fraction times the divergence of the counterfactual
    output from the clean output, so one fraction prunes alike in cells whose
    divergences differ in scale. Where that divergence is zero the cutoff is
    zero, and only a candidate that decreases the divergence is pruned.

    Parameters
    ----------
    fraction : float
        The finite nonnegative fraction of the full-corruption divergence that
        strictly bounds an accepted increment.
    """

    __slots__ = ("_fraction",)

    _noun = "pruning threshold"

    def __init__(self, fraction: float) -> None:
        object.__setattr__(
            self,
            "_fraction",
            _finite_non_negative("a relative threshold's fraction", fraction),
        )

    @property
    def fraction(self) -> float:
        """Return the fraction of the full-corruption divergence."""
        return self._fraction

    def cutoff(self, full_corruption: float, /) -> float:
        """Return the fraction of the full-corruption divergence.

        Parameters
        ----------
        full_corruption : float
            The divergence of the counterfactual output from the clean output.
        """
        return self._fraction * full_corruption

    def to_spec(self) -> dict[str, JsonValue]:
        """Return the declaration that rebuilds this threshold."""
        return {"kind": "relative", "fraction": self._fraction}

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another threshold is relative at the same fraction."""
        if type(other) is not RelativeThreshold:
            return NotImplemented

        return self._fraction == other._fraction

    @override
    def __hash__(self) -> int:
        """Return a hash over the fraction."""
        return hash((RelativeThreshold, self._fraction))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this threshold."""
        return f"RelativeThreshold({self._fraction!r})"


type PruningThreshold = AbsoluteThreshold | RelativeThreshold
"""The rule giving the strict upper bound on an accepted divergence increment."""

_THRESHOLDS: frozenset[type[PruningThreshold]] = frozenset(
    {AbsoluteThreshold, RelativeThreshold}
)
"""Every pruning-threshold kind."""


class ACDCResult[T](FrozenValue):
    """The immutable edge partition produced by one budgeted ACDC run.

    Parameters
    ----------
    graph : ComputationGraph[T]
        The graph that was pruned.
    tested : Sequence[GraphEdge[T]]
        Candidate edges tested, in the caller's declared order.
    untested : Sequence[GraphEdge[T]]
        Edges the budget did not reach, in the remaining declared order.
    retained : Sequence[GraphEdge[T]]
        Every edge not pruned, including every untested edge, in declared order.
    pruned : Sequence[GraphEdge[T]]
        Accepted removals, in declared order.
    divergence : float
        The divergence after every accepted removal.
    threshold : PruningThreshold
        The declared rule bounding a candidate's divergence increment.
    cutoff : float
        The strict upper bound that rule gave this run's increments.
    wall_seconds : float
        Wall seconds spent only inside model forwards.
    budget : PruningBudget
        The required trial budget.
    data_provenance : Provenance
        The data identity for the endpoint inputs.
    policy_provenance : Provenance
        The policy identity for the pruning decision.

    Raises
    ------
    TypeError
        If a value has the wrong declared type.
    ValueError
        If the edge tuples do not form the required partitions, or a number is
        invalid.
    """

    __slots__ = (
        "_budget",
        "_cutoff",
        "_data_provenance",
        "_divergence",
        "_graph",
        "_policy_provenance",
        "_pruned",
        "_retained",
        "_tested",
        "_threshold",
        "_untested",
        "_wall_seconds",
    )

    _noun = "ACDC result"

    def __init__(
        self,
        graph: ComputationGraph[T],
        *,
        tested: Sequence[GraphEdge[T]],
        untested: Sequence[GraphEdge[T]],
        retained: Sequence[GraphEdge[T]],
        pruned: Sequence[GraphEdge[T]],
        divergence: float,
        threshold: PruningThreshold,
        cutoff: float,
        wall_seconds: float,
        budget: PruningBudget,
        data_provenance: Provenance,
        policy_provenance: Provenance,
    ) -> None:
        if type(graph) is not ComputationGraph:
            msg = f"an ACDC result is over a computation graph, not {graph!r}"
            raise TypeError(msg)

        if type(threshold) not in _THRESHOLDS:
            msg = f"an ACDC result carries a pruning threshold, not {threshold!r}"
            raise TypeError(msg)

        if type(budget) is not PruningBudget:
            msg = f"an ACDC result carries a PruningBudget, not {budget!r}"
            raise TypeError(msg)

        for name, provenance in (
            ("data", data_provenance),
            ("policy", policy_provenance),
        ):
            if type(provenance) is not Provenance:
                msg = f"an ACDC result's {name} provenance is not one: {provenance!r}"
                raise TypeError(msg)

        held_tested = _edge_tuple("tested", tested)
        held_untested = _edge_tuple("untested", untested)
        held_retained = _edge_tuple("retained", retained)
        held_pruned = _edge_tuple("pruned", pruned)
        scheduled = held_tested + held_untested
        decided = held_retained + held_pruned

        if len(scheduled) != len(set(scheduled)) or set(scheduled) != graph.edges:
            msg = (
                "an ACDC result's tested and untested edges do not partition its graph"
            )
            raise ValueError(msg)

        if len(decided) != len(set(decided)) or set(decided) != graph.edges:
            msg = (
                "an ACDC result's retained and pruned edges do not partition its graph"
            )
            raise ValueError(msg)

        if not set(held_untested).issubset(held_retained):
            msg = "an ACDC result must retain every edge its budget left untested"
            raise ValueError(msg)

        if not set(held_pruned).issubset(held_tested):
            msg = "an ACDC result cannot prune an edge it did not test"
            raise ValueError(msg)

        if len(held_tested) > budget.maximum_trials:
            msg = (
                f"an ACDC result tested {len(held_tested)} edges under a budget of "
                f"{budget.maximum_trials}"
            )
            raise ValueError(msg)

        order = {edge: position for position, edge in enumerate(scheduled)}

        for name, edges in (("retained", held_retained), ("pruned", held_pruned)):
            positions = tuple(order[edge] for edge in edges)

            if positions != tuple(sorted(positions)):
                msg = f"an ACDC result's {name} edges depart from its declared order"
                raise ValueError(msg)

        object.__setattr__(self, "_graph", graph)
        object.__setattr__(self, "_tested", held_tested)
        object.__setattr__(self, "_untested", held_untested)
        object.__setattr__(self, "_retained", held_retained)
        object.__setattr__(self, "_pruned", held_pruned)
        object.__setattr__(
            self,
            "_divergence",
            _finite_non_negative("an ACDC result's divergence", divergence),
        )
        object.__setattr__(self, "_threshold", threshold)
        object.__setattr__(
            self, "_cutoff", _finite_non_negative("an ACDC result's cutoff", cutoff)
        )
        object.__setattr__(
            self,
            "_wall_seconds",
            _finite_non_negative("an ACDC result's model wall time", wall_seconds),
        )
        object.__setattr__(self, "_budget", budget)
        object.__setattr__(self, "_data_provenance", data_provenance)
        object.__setattr__(self, "_policy_provenance", policy_provenance)

    @property
    def graph(self) -> ComputationGraph[T]:
        """Return the graph that was pruned."""
        return self._graph

    @property
    def tested(self) -> tuple[GraphEdge[T], ...]:
        """Return the tested candidates in declared order."""
        return self._tested

    @property
    def untested(self) -> tuple[GraphEdge[T], ...]:
        """Return the candidates the budget did not reach, in declared order."""
        return self._untested

    @property
    def retained(self) -> tuple[GraphEdge[T], ...]:
        """Return every edge not pruned, in declared order."""
        return self._retained

    @property
    def pruned(self) -> tuple[GraphEdge[T], ...]:
        """Return every accepted removal, in declared order."""
        return self._pruned

    @property
    def divergence(self) -> float:
        """Return the divergence after every accepted removal."""
        return self._divergence

    @property
    def threshold(self) -> PruningThreshold:
        """Return the declared rule bounding an accepted divergence increment."""
        return self._threshold

    @property
    def cutoff(self) -> float:
        """Return the strict upper bound on an accepted divergence increment."""
        return self._cutoff

    @property
    def scheduling_residual(self) -> float:
        """Return the share of graph edges the budget left untested."""
        if not self._graph.edges:
            return 0.0

        return len(self._untested) / len(self._graph.edges)

    @property
    def wall_seconds(self) -> float:
        """Return wall seconds spent only inside model forwards."""
        return self._wall_seconds

    @property
    def budget(self) -> PruningBudget:
        """Return the trial budget."""
        return self._budget

    @property
    def data_provenance(self) -> Provenance:
        """Return the identity of the endpoint data."""
        return self._data_provenance

    @property
    def policy_provenance(self) -> Provenance:
        """Return the identity of the pruning policy."""
        return self._policy_provenance

    @override
    def __repr__(self) -> str:
        """Return the graph size, partition sizes, and scheduling residual."""
        return (
            f"ACDCResult(edges={len(self._graph.edges)}, tested={len(self._tested)}, "
            f"pruned={len(self._pruned)}, residual={self.scheduling_residual!r})"
        )


class ReceiverCircuit[T](FrozenValue):
    """The computation graph of one chart's receiver edges and their targets.

    Each receiver edge of the chart is one graph edge, from the node of its child
    span to the node of the span that receives it, under the receiver-edge role.
    Its target is the unit at the scorer's receiver site that indexes that edge's
    coordinate on the edge axis alone. The order runs the edges into the widest
    receivers first and, among receivers of one width, by ascending edge-axis
    coordinate, which is a total reverse-topological order.

    Parameters
    ----------
    graph : ComputationGraph[T]
        One node per chart span and one edge per receiver edge.
    targets : Mapping[GraphEdge[T], Unit]
        The receiver-site unit of every graph edge.
    order : tuple[GraphEdge[T], ...]
        Every graph edge once, in the declared pruning order.
    """

    __slots__ = ("_graph", "_order", "_targets")

    _noun = "receiver circuit"

    def __init__(
        self,
        graph: ComputationGraph[T],
        targets: Mapping[GraphEdge[T], Unit],
        order: tuple[GraphEdge[T], ...],
    ) -> None:
        object.__setattr__(self, "_graph", graph)
        object.__setattr__(self, "_targets", MappingProxyType(dict(targets)))
        object.__setattr__(self, "_order", order)

    @property
    def graph(self) -> ComputationGraph[T]:
        """Return the graph of child-to-receiver edges."""
        return self._graph

    @property
    def targets(self) -> Mapping[GraphEdge[T], Unit]:
        """Return the receiver-site unit of every edge, as a read-only view."""
        return self._targets

    @property
    def order(self) -> tuple[GraphEdge[T], ...]:
        """Return every edge, widest receivers first, then by edge coordinate."""
        return self._order

    @override
    def __repr__(self) -> str:
        """Return how many nodes and edges the circuit holds."""
        return (
            f"ReceiverCircuit(nodes={len(self._graph.nodes)}, "
            f"edges={len(self._graph.edges)})"
        )


def _receiver_width(index: SpanIndex, edge: ReceiverEdge) -> int:
    """Return the width of the span an edge enters."""
    start, end = index.span_of(edge.target)

    return end - start


def receiver_circuit[T](
    scorer: ReceiverLocalChartScorer, index: SpanIndex, /, *, reading: T
) -> ReceiverCircuit[T]:
    """Return the receiver-edge graph, targets and pruning order of one chart.

    Parameters
    ----------
    scorer : ReceiverLocalChartScorer
        The scorer whose published receiver edges the circuit is drawn from.
    index : SpanIndex
        The chart the edges are enumerated over.
    reading : T
        The reading every span's node concludes.

    Returns
    -------
    ReceiverCircuit[T]
        One graph edge and one receiver-site unit per receiver edge, with the
        edges ordered widest receivers first and, within one receiver width, by
        ascending edge-axis coordinate.

    Raises
    ------
    ValueError
        If the scorer's edges are not in edge-axis order, or two of them join the
        same child and receiver, so the correspondence is not one-to-one.
    """
    site = scorer.receiver_site
    edges = scorer.receiver_edges(index)
    nodes = {
        span: GraphNode(site.component, ItemKey(index.span_of(span), reading))
        for span in range(index.num_spans)
    }
    targets: dict[GraphEdge[T], Unit] = {}
    receivers: dict[GraphEdge[T], ReceiverEdge] = {}

    for position, edge in enumerate(edges):
        if edge.position != position:
            msg = (
                f"receiver edge {edge!r} is at coordinate {edge.position}, not at "
                f"its axis order {position}"
            )
            raise ValueError(msg)

        drawn = GraphEdge(nodes[edge.source], nodes[edge.target], Role.RECEIVER_EDGE)

        if drawn in targets:
            msg = (
                f"receiver edges {receivers[drawn]!r} and {edge!r} join one child "
                "and receiver, so receiver edges and graph edges are not one-to-one"
            )
            raise ValueError(msg)

        targets[drawn] = Unit(site, edge.coordinates)
        receivers[drawn] = edge

    order = tuple(
        sorted(
            receivers,
            key=lambda drawn: (
                -_receiver_width(index, receivers[drawn]),
                receivers[drawn].position,
            ),
        )
    )

    return ReceiverCircuit(ComputationGraph(nodes.values(), targets), targets, order)


def _edge_tuple[T](
    name: str, edges: Sequence[GraphEdge[T]]
) -> tuple[GraphEdge[T], ...]:
    """Return an edge sequence as a validated tuple."""
    held = tuple(edges)

    for edge in held:
        if type(edge) is not GraphEdge:
            msg = f"an ACDC result's {name} member is not a GraphEdge: {edge!r}"
            raise TypeError(msg)

    return held


def _inputs(clean: Tensor, counterfactual: Tensor) -> tuple[Tensor, Tensor]:
    """Return two corresponding finite endpoint inputs."""
    for name, value in (("clean", clean), ("counterfactual", counterfactual)):
        if not isinstance(value, torch.Tensor):
            msg = f"ACDC's {name} input is not a tensor: {value!r}"
            raise TypeError(msg)

        if not bool(torch.isfinite(value).all()):
            msg = f"ACDC's {name} input is not finite"
            raise ValueError(msg)

    if clean.shape != counterfactual.shape:
        msg = (
            f"ACDC's counterfactual input has shape {tuple(counterfactual.shape)}, "
            f"not the clean input's shape {tuple(clean.shape)}"
        )
        raise ValueError(msg)

    if clean.dtype != counterfactual.dtype:
        msg = (
            f"ACDC's counterfactual input has dtype {counterfactual.dtype}, not the "
            f"clean input's dtype {clean.dtype}"
        )
        raise ValueError(msg)

    if clean.device != counterfactual.device:
        msg = (
            f"ACDC's counterfactual input is on {counterfactual.device}, not the "
            f"clean input's device {clean.device}"
        )
        raise ValueError(msg)

    return clean, counterfactual


def _ordered[T](
    graph: ComputationGraph[T], order: Sequence[GraphEdge[T]]
) -> tuple[GraphEdge[T], ...]:
    """Return the graph's total reverse-topological edge order."""
    held = _edge_tuple("order", order)

    if len(held) != len(set(held)):
        msg = "ACDC's edge order repeats an edge"
        raise ValueError(msg)

    if set(held) != graph.edges:
        missing = graph.edges.difference(held)
        foreign = set(held).difference(graph.edges)
        msg = (
            "ACDC's edge order is not total over the graph; "
            f"missing={sorted(missing, key=repr)!r}, "
            f"foreign={sorted(foreign, key=repr)!r}"
        )
        raise ValueError(msg)

    positions = {edge: position for position, edge in enumerate(held)}

    for upstream in held:
        for downstream in held:
            if (
                upstream.target == downstream.source
                and positions[downstream] > positions[upstream]
            ):
                msg = (
                    "ACDC's edge order is not reverse-topological: downstream "
                    f"{downstream!r} follows upstream {upstream!r}"
                )
                raise ValueError(msg)

    return held


def _overlap(left: Unit, right: Unit) -> bool:
    """Return whether two coordinate selections can cover a common entry."""
    if left.site != right.site:
        return False

    return all(
        left.index[axis] == right.index[axis]
        for axis in left.index.keys() & right.index.keys()
    )


def _targets[T](
    graph: ComputationGraph[T], targets: Mapping[GraphEdge[T], Unit]
) -> dict[GraphEdge[T], Unit]:
    """Return a complete receiver-local, nonoverlapping edge target mapping."""
    if not isinstance(targets, Mapping):
        msg = f"ACDC's edge targets are not a mapping: {targets!r}"
        raise TypeError(msg)

    held: dict[GraphEdge[T], Unit] = {}

    for edge, unit in targets.items():
        if type(edge) is not GraphEdge:
            msg = f"an ACDC target key is not a GraphEdge: {edge!r}"
            raise TypeError(msg)

        if edge not in graph.edges:
            msg = f"ACDC has a target for {edge!r}, which is not an edge in its graph"
            raise ValueError(msg)

        if type(unit) is not Unit:
            msg = f"ACDC's target for {edge!r} is not a Unit: {unit!r}"
            raise TypeError(msg)

        if unit.site.role is not Role.RECEIVER_EDGE:
            msg = (
                f"ACDC's target for {edge!r} has role {unit.site.role.name}, not "
                "the receiver-local RECEIVER_EDGE role"
            )
            raise ValueError(msg)

        if unit.site.component != edge.target.component:
            msg = (
                f"ACDC's target for {edge!r} belongs to {unit.site.component!r}, "
                f"not its receiver {edge.target.component!r}"
            )
            raise ValueError(msg)

        held[edge] = unit

    missing = graph.edges.difference(held)

    if missing:
        msg = f"ACDC has no receiver-local target for {min(missing, key=repr)!r}"
        raise ValueError(msg)

    return held


def _nonoverlapping[T](targets: Mapping[GraphEdge[T], Unit]) -> None:
    """Require distinct edge targets to cover no common entry."""
    items = tuple(targets.items())

    for position, (left_edge, left) in enumerate(items):
        for right_edge, right in items[position + 1 :]:
            if _overlap(left, right):
                msg = (
                    f"ACDC targets for {left_edge!r} and {right_edge!r} overlap at "
                    f"{left.site!r}, so neither is edge-local"
                )
                raise ValueError(msg)


def _published[T, ReadingT: Reading](
    model: Probed[ReadingT], targets: Mapping[GraphEdge[T], Unit]
) -> SiteMap:
    """Return a site map that permits observing and patching every target."""
    published = model.site_map()

    if not isinstance(published, SiteMap):
        msg = f"ACDC's model publishes a SiteMap, not {published!r}"
        raise TypeError(msg)

    for unit in targets.values():
        if unit.site not in published:
            msg = f"ACDC target {unit!r} is not published by the model"
            raise ValueError(msg)

        access = published[unit.site].access

        if Access.OBSERVE not in access or Access.PATCH not in access:
            msg = (
                f"ACDC target {unit!r} must permit observation and patching, "
                f"but declares {access!r}"
            )
            raise ValueError(msg)

        for axis in unit.index:
            if axis not in published[unit.site].axes:
                msg = f"ACDC target {unit!r} names undeclared axis {axis!r}"
                raise KeyError(msg)

    return published


def _output(value: object, *, point: str, reference: Tensor | None = None) -> Tensor:
    """Return one finite floating output corresponding to a reference output."""
    if not isinstance(value, torch.Tensor):
        msg = f"ACDC's {point} forward returned no tensor: {value!r}"
        raise TypeError(msg)

    if not value.is_floating_point():
        msg = f"ACDC's {point} output is not floating point: {value.dtype}"
        raise TypeError(msg)

    if not bool(torch.isfinite(value).all()):
        msg = f"ACDC's {point} output is not finite"
        raise ValueError(msg)

    if reference is not None:
        if value.shape != reference.shape:
            msg = (
                f"ACDC's {point} output has shape {tuple(value.shape)}, not the "
                f"clean output's shape {tuple(reference.shape)}"
            )
            raise ValueError(msg)

        if value.dtype != reference.dtype:
            msg = (
                f"ACDC's {point} output has dtype {value.dtype}, not the clean "
                f"output's dtype {reference.dtype}"
            )
            raise ValueError(msg)

        if value.device != reference.device:
            msg = (
                f"ACDC's {point} output is on {value.device}, not the clean "
                f"output's device {reference.device}"
            )
            raise ValueError(msg)

    return value


def _divergence(
    measure: Callable[[Tensor, Tensor], float | Tensor],
    clean: Tensor,
    candidate: Tensor,
    *,
    point: str,
) -> float:
    """Return one finite, nonnegative scalar divergence."""
    measured = measure(clean, candidate)

    if isinstance(measured, torch.Tensor):
        if measured.ndim != 0:
            msg = (
                f"ACDC's {point} divergence is not scalar: shape "
                f"{tuple(measured.shape)}"
            )
            raise ValueError(msg)

        if not measured.is_floating_point():
            msg = f"ACDC's {point} divergence is not floating point: {measured.dtype}"
            raise TypeError(msg)

        value = float(measured.detach())
    elif isinstance(measured, bool) or not isinstance(measured, int | float):
        msg = f"ACDC's {point} divergence is not a scalar number: {measured!r}"
        raise TypeError(msg)
    else:
        value = float(measured)

    return _finite_non_negative(f"ACDC's {point} divergence", value)


def _corresponds(
    reference: Observation,
    candidate: Observation,
    unit: Unit,
    *,
    point: str,
) -> None:
    """Require one target reading to correspond to its clean endpoint reading."""
    if candidate.axes != reference.axes:
        msg = (
            f"ACDC's {point} reading at {reference.site!r} has axes "
            f"{candidate.axes!r}, not the clean reading's {reference.axes!r}"
        )
        raise ValueError(msg)

    if candidate.tensor.shape != reference.tensor.shape:
        msg = (
            f"ACDC's {point} reading at {reference.site!r} has shape "
            f"{tuple(candidate.tensor.shape)}, not the clean reading's shape "
            f"{tuple(reference.tensor.shape)}"
        )
        raise ValueError(msg)

    if candidate.tensor.dtype != reference.tensor.dtype:
        msg = (
            f"ACDC's {point} reading at {reference.site!r} has dtype "
            f"{candidate.tensor.dtype}, not the clean reading's dtype "
            f"{reference.tensor.dtype}"
        )
        raise ValueError(msg)

    if not candidate.tensor.is_floating_point():
        msg = f"ACDC target {unit!r} is not floating point: {candidate.tensor.dtype}"
        raise TypeError(msg)

    if not bool(torch.isfinite(candidate.tensor).all()):
        msg = f"ACDC's {point} reading at {reference.site!r} is not finite"
        raise ValueError(msg)

    if not torch.equal(reference.covering(), candidate.covering()):
        msg = (
            f"ACDC's clean and {point} readings at {reference.site!r} hold data "
            "at different positions"
        )
        raise ValueError(msg)

    for axis, coordinate in unit.index.items():
        if coordinate >= candidate.extent(axis):
            msg = (
                f"ACDC target {unit!r} indexes {coordinate} on axis {axis!r}, "
                f"which has {candidate.extent(axis)} positions"
            )
            raise ValueError(msg)


def _endpoint[ReadingT: Reading](
    model: Probed[ReadingT],
    sites: frozenset[Site],
    given: Tensor,
    forward: Callable[[Tensor], Tensor],
) -> tuple[Tensor, Trace, float]:
    """Run and observe one endpoint, timing only its model forward."""
    with torch.no_grad(), instrument(model, observe=sites) as run:
        output, elapsed = run(lambda: _timed(forward, given))

    return _output(output, point="endpoint"), run.trace, elapsed


def _timed(forward: Callable[[Tensor], Tensor], given: Tensor) -> tuple[Tensor, float]:
    """Run one model forward and return its unsynchronized host wall time."""
    start = perf_counter()
    output = forward(given)

    return output, perf_counter() - start


def _patches[T](
    edges: Sequence[GraphEdge[T]],
    targets: Mapping[GraphEdge[T], Unit],
    counterfactual: Trace,
) -> dict[Site, Within]:
    """Return cumulative fixed-trace patches grouped by target site."""
    grouped: dict[Site, list[Unit]] = {}

    for edge in edges:
        unit = targets[edge]
        grouped.setdefault(unit.site, []).append(unit)

    return {
        site: Within(units, PatchFrom(counterfactual, site))
        for site, units in grouped.items()
    }


def acdc[T, ReadingT: Reading](
    model: Probed[ReadingT],
    graph: ComputationGraph[T],
    /,
    *,
    targets: Mapping[GraphEdge[T], Unit],
    order: Sequence[GraphEdge[T]],
    clean: Tensor,
    counterfactual: Tensor,
    forward: Callable[[Tensor], Tensor],
    divergence: Callable[[Tensor, Tensor], float | Tensor],
    threshold: PruningThreshold,
    budget: PruningBudget,
    data_provenance: Provenance,
    policy_provenance: Provenance,
) -> ACDCResult[T]:
    """Prune receiver-local edges with a budgeted greedy ACDC oracle.

    The clean and counterfactual endpoints are each observed once. Each tested
    candidate then receives exactly one clean-input forward with all accepted
    removals and that candidate patched from the fixed counterfactual trace. The
    candidate is pruned exactly when its divergence increment is strictly below
    the cutoff ``threshold`` gives the divergence of the counterfactual output
    from the clean output.

    Parameters
    ----------
    model : Probed[ReadingT]
        The component publishing every receiver-local target.
    graph : ComputationGraph[T]
        The directed acyclic computation graph to prune.
    targets : Mapping[GraphEdge[T], Unit]
        Exactly one nonoverlapping receiver-local unit for every graph edge.
    order : Sequence[GraphEdge[T]]
        Every edge once in total reverse-topological order.
    clean : Tensor
        The clean endpoint input used for the baseline and candidate trials.
    counterfactual : Tensor
        The endpoint whose fixed receiver-local readings replace clean readings.
    forward : Callable[[Tensor], Tensor]
        The model pass from an endpoint input to a floating output.
    divergence : Callable[[Tensor, Tensor], float | Tensor]
        A nonnegative finite scalar divergence from clean to candidate output.
    threshold : PruningThreshold
        The required rule giving the strict upper bound on an increment.
    budget : PruningBudget
        The required maximum number of candidate trials.
    data_provenance : Provenance
        The identity of the endpoint data.
    policy_provenance : Provenance
        The identity of the ordering, threshold and budget policy.

    Returns
    -------
    ACDCResult[T]
        The tested, untested, retained and pruned partitions, final divergence,
        scheduling residual, model-only wall time, budget and provenance.

    Raises
    ------
    TypeError
        If a graph, target, callable, input, output, divergence, budget or
        provenance has the wrong type.
    ValueError
        If the graph mapping, edge order, site access, endpoints, readings,
        divergence or policy number violates the pruning contract.
    """
    if type(graph) is not ComputationGraph:
        msg = f"ACDC requires a ComputationGraph, not {graph!r}"
        raise TypeError(msg)

    if not callable(forward):
        msg = f"ACDC's forward is not callable: {forward!r}"
        raise TypeError(msg)

    if not callable(divergence):
        msg = f"ACDC's divergence is not callable: {divergence!r}"
        raise TypeError(msg)

    if type(budget) is not PruningBudget:
        msg = f"ACDC requires a PruningBudget, not {budget!r}"
        raise TypeError(msg)

    for name, provenance in (
        ("data", data_provenance),
        ("policy", policy_provenance),
    ):
        if type(provenance) is not Provenance:
            msg = f"ACDC's {name} provenance is not a Provenance: {provenance!r}"
            raise TypeError(msg)

    if type(threshold) not in _THRESHOLDS:
        msg = f"ACDC requires a pruning threshold, not {threshold!r}"
        raise TypeError(msg)

    held_order = _ordered(graph, order)
    held_targets = _targets(graph, targets)
    clean, counterfactual = _inputs(clean, counterfactual)

    if not graph.edges:
        return ACDCResult(
            graph,
            tested=(),
            untested=(),
            retained=(),
            pruned=(),
            divergence=0.0,
            threshold=threshold,
            cutoff=threshold.cutoff(0.0),
            wall_seconds=0.0,
            budget=budget,
            data_provenance=data_provenance,
            policy_provenance=policy_provenance,
        )

    _published(model, held_targets)
    _nonoverlapping(held_targets)
    sites = frozenset(unit.site for unit in held_targets.values())
    clean_output, clean_trace, clean_seconds = _endpoint(model, sites, clean, forward)
    counter_output, counter_trace, counter_seconds = _endpoint(
        model, sites, counterfactual, forward
    )
    _output(counter_output, point="counterfactual", reference=clean_output)

    for unit in held_targets.values():
        clean_reading = clean_trace[unit.site]
        _corresponds(clean_reading, clean_reading, unit, point="clean")
        _corresponds(
            clean_reading,
            counter_trace[unit.site],
            unit,
            point="counterfactual",
        )

    tested = held_order[: budget.maximum_trials]
    untested = held_order[len(tested) :]
    pruned: list[GraphEdge[T]] = []
    current = _divergence(
        divergence, clean_output, clean_output, point="clean baseline"
    )
    cutoff = threshold.cutoff(
        _divergence(divergence, clean_output, counter_output, point="full corruption")
    )
    wall_seconds = clean_seconds + counter_seconds

    for position, edge in enumerate(tested, start=1):
        patch = _patches((*pruned, edge), held_targets, counter_trace)

        with torch.no_grad(), instrument(model, patch=patch) as run:
            trial_value, trial_seconds = run(lambda: _timed(forward, clean))

        wall_seconds += trial_seconds

        trial = _output(
            trial_value,
            point=f"candidate {position}",
            reference=clean_output,
        )
        measured = _divergence(
            divergence,
            clean_output,
            trial,
            point=f"candidate {position}",
        )
        increment = measured - current

        if not isfinite(increment):
            msg = f"ACDC's candidate {position} divergence increment is not finite"
            raise ValueError(msg)

        if increment < cutoff:
            pruned.append(edge)
            current = measured

    removed = frozenset(pruned)
    retained = tuple(edge for edge in held_order if edge not in removed)

    return ACDCResult(
        graph,
        tested=tested,
        untested=untested,
        retained=retained,
        pruned=tuple(pruned),
        divergence=current,
        threshold=threshold,
        cutoff=cutoff,
        wall_seconds=wall_seconds,
        budget=budget,
        data_provenance=data_provenance,
        policy_provenance=policy_provenance,
    )


def matched_cardinality_jaccard[T](
    result: ACDCResult[T], attribution: EdgeAttribution[T]
) -> float:
    """Compare retained ACDC edges with equally many top EAP-IG edges.

    Parameters
    ----------
    result : ACDCResult[T]
        The ACDC circuit, represented by its retained edges.
    attribution : EdgeAttribution[T]
        EAP-IG effects over the same graph.

    Returns
    -------
    float
        Jaccard agreement between the retained edges and the same-cardinality
        prefix of the attribution ranking. Two empty circuits agree at one.

    Raises
    ------
    TypeError
        If either argument has the wrong value type.
    ValueError
        If the two values are over different graphs.
    """
    if type(result) is not ACDCResult:
        msg = f"matched-cardinality overlap requires an ACDCResult, not {result!r}"
        raise TypeError(msg)

    if type(attribution) is not EdgeAttribution:
        msg = (
            "matched-cardinality overlap requires an EdgeAttribution, not "
            f"{attribution!r}"
        )
        raise TypeError(msg)

    if result.graph != attribution.graph:
        msg = "matched-cardinality overlap requires results over the same graph"
        raise ValueError(msg)

    retained = frozenset(result.retained)
    ranked = frozenset(attribution.ranked()[: len(retained)])

    if not retained and not ranked:
        return 1.0

    return len(retained & ranked) / len(retained | ranked)
