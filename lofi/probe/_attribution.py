"""Integrated-gradient and first-order edge attribution."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, override

import torch
from torch import Tensor

from lofi._checks import ensure_name, ensure_positive
from lofi.probe._circuits import AttributionEvidence, Unit
from lofi.probe._graph import ComputationGraph, GraphEdge
from lofi.probe._instrument import GradientSites, differentiate
from lofi.probe._provenance import Provenance

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from lofi.nn.ports import Reading
    from lofi.nn.sites import Site, SiteMap
    from lofi.probe._ports import Probed
    from lofi.probe._values import Observation, Trace

__all__ = ["EdgeAttribution", "attribute_edges", "first_order_attribute_edges"]


class EdgeAttribution[T]:
    """The estimated effect of patching each edge of a computation graph.

    Parameters
    ----------
    graph : ComputationGraph[T]
        The graph the edges are drawn from.
    effects : Mapping[GraphEdge[T], AttributionEvidence]
        The evidence for every edge of the graph, keyed by the edge.

    Raises
    ------
    TypeError
        If the graph is not a computation graph, or a value is not attribution
        evidence.
    ValueError
        If an edge of the graph has no evidence, or evidence is given for an edge
        the graph does not hold.
    """

    __slots__ = ("_effects", "_graph")

    def __init__(
        self,
        graph: ComputationGraph[T],
        effects: Mapping[GraphEdge[T], AttributionEvidence],
    ) -> None:
        if type(graph) is not ComputationGraph:
            msg = f"an attribution is over a computation graph, not {graph!r}"
            raise TypeError(msg)

        held: dict[GraphEdge[T], AttributionEvidence] = {}

        for edge, evidence in effects.items():
            if type(evidence) is not AttributionEvidence:
                msg = f"an edge's effect is attribution evidence, not {evidence!r}"
                raise TypeError(msg)

            if edge not in graph.edges:
                msg = f"an effect is given for {edge!r}, which is not an edge here"
                raise ValueError(msg)

            held[edge] = evidence

        for edge in graph.edges:
            if edge not in held:
                msg = f"the graph holds {edge!r}, which has no effect"
                raise ValueError(msg)

        self._graph = graph
        self._effects = MappingProxyType(held)

    @property
    def graph(self) -> ComputationGraph[T]:
        """Return the graph the edges are drawn from."""
        return self._graph

    @property
    def effects(self) -> Mapping[GraphEdge[T], AttributionEvidence]:
        """Return the evidence for every edge, as a read-only view."""
        return self._effects

    def __getitem__(self, edge: GraphEdge[T]) -> AttributionEvidence:
        """Return the evidence for one edge.

        Parameters
        ----------
        edge : GraphEdge[T]
            The edge to look up.

        Raises
        ------
        KeyError
            If the graph does not hold the edge.
        """
        return self._effects[edge]

    def __len__(self) -> int:
        """Return how many edges have an effect."""
        return len(self._effects)

    def __iter__(self) -> Iterator[GraphEdge[T]]:
        """Return the edges, in no particular order."""
        return iter(self._effects)

    def ranked(self) -> tuple[GraphEdge[T], ...]:
        """Return the edges from the largest effect in magnitude to the smallest.

        Returns
        -------
        tuple[GraphEdge[T], ...]
            Every edge once. Two edges of one magnitude fall in the order of their
            representations.
        """
        return tuple(
            sorted(
                self._effects,
                key=lambda edge: (-abs(self._effects[edge].effect), repr(edge)),
            )
        )

    def _identity(
        self,
    ) -> tuple[
        ComputationGraph[T], frozenset[tuple[GraphEdge[T], AttributionEvidence]]
    ]:
        """Return the two fields an attribution is equal on."""
        return (self._graph, frozenset(self._effects.items()))

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value records the same effects over the same graph."""
        if type(other) is not EdgeAttribution:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the graph and the effects."""
        return hash((EdgeAttribution, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the graph and how many edges have an effect."""
        return f"EdgeAttribution({self._graph!r}, edges={len(self._effects)})"


def _targets[T](
    graph: ComputationGraph[T],
    targets: Mapping[GraphEdge[T], Unit],
) -> dict[GraphEdge[T], Unit]:
    """Return a complete edge target mapping over a graph.

    Parameters
    ----------
    graph : ComputationGraph[T]
        The graph whose edges are to be attributed.
    targets : Mapping[GraphEdge[T], Unit]
        The coordinate each edge's effect is integrated over.

    Returns
    -------
    dict[GraphEdge[T], Unit]
        One unit per graph edge.

    Raises
    ------
    TypeError
        If the graph, the mapping, a key or a unit has the wrong type.
    ValueError
        If an edge has no target, or a target names an edge outside the graph.
    """
    if type(graph) is not ComputationGraph:
        msg = f"an attribution is over a computation graph, not {graph!r}"
        raise TypeError(msg)

    if not isinstance(targets, Mapping):
        msg = f"an attribution's edge targets are not a mapping: {targets!r}"
        raise TypeError(msg)

    held: dict[GraphEdge[T], Unit] = {}

    for edge, unit in targets.items():
        if type(edge) is not GraphEdge:
            msg = f"an attribution target key is not a GraphEdge: {edge!r}"
            raise TypeError(msg)

        if edge not in graph.edges:
            msg = f"an attribution has a target for {edge!r}, which is not an edge here"
            raise ValueError(msg)

        if type(unit) is not Unit:
            msg = f"the attribution target for {edge!r} is not a Unit: {unit!r}"
            raise TypeError(msg)

        held[edge] = unit

    missing = graph.edges.difference(held)

    if missing:
        msg = f"an attribution has no target for {min(missing, key=repr)!r}"
        raise ValueError(msg)

    return held


def _declared_axes(published: SiteMap, units: Iterable[Unit]) -> None:
    """Require every unit to index only axes its point declares.

    Parameters
    ----------
    published : SiteMap
        The validated map every unit's point is published in.
    units : Iterable[Unit]
        The targets whose coordinates are checked.

    Raises
    ------
    KeyError
        If a unit names an axis its point does not declare.
    """
    for unit in units:
        declared = published[unit.site].axes

        for axis in unit.index:
            if axis not in declared:
                msg = f"the attribution target {unit!r} names undeclared axis {axis!r}"
                raise KeyError(msg)


def _weighed(
    clean: Observation, counterfactual: Observation, gradient: Tensor
) -> Tensor:
    """Return the displacement times the gradient, zero where no data is held.

    Parameters
    ----------
    clean : Observation
        The value at the point under the clean pass.
    counterfactual : Observation
        The value at the same point under the counterfactual pass.
    gradient : Tensor
        The mean gradient of the metric at the point over the path.

    Returns
    -------
    Tensor
        The float64 elementwise product of the counterfactual minus the clean
        value with the gradient, over the positions the clean reading holds data
        at.

    Raises
    ------
    ValueError
        If the two readings differ in shape or in which positions hold data, since
        one then cannot stand in for the other.
    """
    if tuple(counterfactual.tensor.shape) != tuple(clean.tensor.shape):
        msg = (
            f"the counterfactual reading at {clean.site!r} is of shape "
            f"{tuple(counterfactual.tensor.shape)} and cannot stand in for one "
            f"of shape {tuple(clean.tensor.shape)}"
        )
        raise ValueError(msg)

    if tuple(gradient.shape) != tuple(clean.tensor.shape):
        msg = (
            f"the gradient at {clean.site!r} is of shape {tuple(gradient.shape)} "
            f"rather than the reading's shape {tuple(clean.tensor.shape)}"
        )
        raise ValueError(msg)

    difference = counterfactual.tensor.to(torch.float64) - clean.tensor.to(
        torch.float64
    )
    weighed = difference * gradient.to(torch.float64)

    if clean.mask is None and counterfactual.mask is None:
        return weighed

    covering = clean.covering()

    if not torch.equal(covering, counterfactual.covering()):
        msg = (
            f"the clean and counterfactual readings at {clean.site!r} hold data "
            "at different positions, so one cannot stand in for the other"
        )
        raise ValueError(msg)

    return torch.where(covering, weighed, 0.0)


def _effect(weighed: Tensor, reading: Observation, unit: Unit) -> float:
    """Return the weighed displacement summed over the entries one unit indexes.

    Parameters
    ----------
    weighed : Tensor
        The displacement times the gradient at the unit's point.
    reading : Observation
        The clean reading there, which names the axes and their extents.
    unit : Unit
        The coordinate the effect is summed over. A unit with no coordinate covers
        the whole point.

    Returns
    -------
    float
        The effect.

    Raises
    ------
    ValueError
        If a coordinate lies beyond its axis, or the effect is not finite.
    """
    index: list[slice | int] = [slice(None)] * weighed.ndim

    for axis, coordinate in unit.index.items():
        if coordinate >= reading.extent(axis):
            msg = (
                f"the attribution target {unit!r} indexes {coordinate} on axis "
                f"{axis!r}, which has {reading.extent(axis)} positions"
            )
            raise ValueError(msg)

        index[reading.position(axis)] = coordinate

    effect = weighed[tuple(index)].sum()

    if not bool(torch.isfinite(effect)):
        msg = f"the attributed effect at {unit.site!r} for {unit!r} is not finite"
        raise ValueError(msg)

    return float(effect)


def _inputs(clean: Tensor, counterfactual: Tensor) -> tuple[Tensor, Tensor]:
    """Validate and return the two endpoints of an attribution path."""
    for name, value in (("clean", clean), ("counterfactual", counterfactual)):
        if not isinstance(value, torch.Tensor):
            msg = f"an attribution's {name} input is not a tensor: {value!r}"
            raise TypeError(msg)

        if not value.is_floating_point():
            msg = f"an attribution's {name} input is not floating point: {value.dtype}"
            raise TypeError(msg)

        if not bool(torch.isfinite(value).all()):
            msg = f"an attribution's {name} input is not finite"
            raise ValueError(msg)

    if clean.shape != counterfactual.shape:
        msg = (
            f"the counterfactual input is of shape {tuple(counterfactual.shape)} "
            f"and cannot stand in for the clean input of shape {tuple(clean.shape)}"
        )
        raise ValueError(msg)

    if clean.dtype != counterfactual.dtype:
        msg = (
            f"the counterfactual input has dtype {counterfactual.dtype} and cannot "
            f"stand in for the clean input with dtype {clean.dtype}"
        )
        raise ValueError(msg)

    if clean.device != counterfactual.device:
        msg = (
            f"the counterfactual input is on {counterfactual.device} and cannot "
            f"stand in for the clean input on {clean.device}"
        )
        raise ValueError(msg)

    return clean, counterfactual


def _corresponds(reference: Observation, candidate: Observation, *, point: str) -> None:
    """Require a path reading to correspond to its clean endpoint reading."""
    if tuple(candidate.tensor.shape) != tuple(reference.tensor.shape):
        msg = (
            f"the {point} reading at {reference.site!r} is of shape "
            f"{tuple(candidate.tensor.shape)} and cannot stand in for one of shape "
            f"{tuple(reference.tensor.shape)}"
        )
        raise ValueError(msg)

    if candidate.axes != reference.axes:
        msg = (
            f"the {point} reading at {reference.site!r} has axes {candidate.axes!r} "
            f"and cannot stand in for one with axes {reference.axes!r}"
        )
        raise ValueError(msg)

    if not torch.equal(reference.covering(), candidate.covering()):
        msg = (
            f"the clean and {point} readings at {reference.site!r} hold data at "
            "different positions, so one cannot stand in for the other"
        )
        raise ValueError(msg)


def _gradient_pass[ReadingT: Reading](
    model: Probed[ReadingT],
    sites: GradientSites,
    forward: Callable[[], Tensor],
) -> tuple[Trace, dict[Site, Tensor]]:
    """Run one pass with every point tapped and return its trace and gradients.

    Parameters
    ----------
    model : Probed[ReadingT]
        The component publishing the points.
    sites : GradientSites
        The checked points to observe and differentiate at.
    forward : Callable[[], Tensor]
        The pass to the scalar metric.

    Returns
    -------
    tuple[Trace, dict[Site, Tensor]]
        The closed trace of the pass and the float64 gradient at each point.

    Raises
    ------
    TypeError
        If the pass returns no tensor.
    RuntimeError
        If the pass reaches a point twice at one position.
    ValueError
        If the metric is not a scalar or no gradient reaches it, the pass never
        reached a point, or, with the checks on, a reading departs from what its
        point declared.
    """
    with differentiate(model, sites=sites) as region:
        metric = region(forward)

        if not isinstance(metric, torch.Tensor):
            msg = f"an attribution's forward returned no tensor: {metric!r}"
            raise TypeError(msg)

        gradients = {
            site: gradient.to(torch.float64)
            for site, gradient in region.gradients(metric).items()
        }

    return region.trace, gradients


def _attribute[T, ReadingT: Reading](
    model: Probed[ReadingT],
    graph: ComputationGraph[T],
    /,
    *,
    targets: Mapping[GraphEdge[T], Unit],
    counterfactual: Tensor,
    forward: Callable[[Tensor], Tensor],
    path: Iterator[Tensor],
    steps: int,
    dataset: str,
    method: str,
) -> EdgeAttribution[T]:
    """Stream target gradients over a path and integrate each edge's effect.

    The site map is validated once, before the first pass, and every pass on the
    path and the counterfactual pass are opened over the points it checked.
    """
    named = ensure_name("an attribution's dataset", dataset)
    held = _targets(graph, targets)
    provenance = Provenance(method, named, None)

    if not held:
        return EdgeAttribution(graph, {})

    points = GradientSites(
        model.site_map(), frozenset(unit.site for unit in held.values())
    )
    _declared_axes(points.published, held.values())
    sites = points.sites

    totals: dict[Site, Tensor] = {}
    reference: dict[Site, Observation] = {}
    clean_observations: dict[Site, Observation] = {}

    for index, point in enumerate(path, start=1):
        trace, gradients = _gradient_pass(
            model, points, lambda point=point: forward(point)
        )

        for site in sites:
            observation = trace[site]

            if site in reference:
                _corresponds(reference[site], observation, point=f"path step {index}")
            else:
                reference[site] = observation

            gradient = gradients[site]

            if not bool(torch.isfinite(gradient).all()):
                msg = f"the gradient at {site!r} on path step {index} is not finite"
                raise ValueError(msg)

            if site in totals:
                totals[site].add_(gradient)
            else:
                totals[site] = gradient

            if index == steps:
                clean_observations[site] = observation

    if len(clean_observations) != len(sites):
        msg = f"the attribution path yielded fewer than its declared {steps} steps"
        raise ValueError(msg)

    with torch.no_grad(), differentiate(model, sites=points) as corrupted:
        corrupted(lambda: forward(counterfactual))

    weighed: dict[Site, Tensor] = {}

    for site in sites:
        counterfactual_observation = corrupted.trace[site]
        _corresponds(
            clean_observations[site], counterfactual_observation, point="counterfactual"
        )
        weighed[site] = _weighed(
            clean_observations[site], counterfactual_observation, totals[site] / steps
        )

    return EdgeAttribution(
        graph,
        {
            edge: AttributionEvidence(
                _effect(weighed[unit.site], clean_observations[unit.site], unit),
                provenance,
            )
            for edge, unit in sorted(held.items(), key=lambda item: repr(item[0]))
        },
    )


def attribute_edges[T, ReadingT: Reading](
    model: Probed[ReadingT],
    graph: ComputationGraph[T],
    /,
    *,
    targets: Mapping[GraphEdge[T], Unit],
    clean: Tensor,
    counterfactual: Tensor,
    forward: Callable[[Tensor], Tensor],
    steps: int,
    dataset: str,
) -> EdgeAttribution[T]:
    """Attribute edges with integrated gradients from counterfactual to clean.

    Parameters
    ----------
    model : Probed[ReadingT]
        The component the graph's nodes are published by.
    graph : ComputationGraph[T]
        The graph whose edges are to be attributed.
    targets : Mapping[GraphEdge[T], Unit]
        The coordinate each edge's effect is integrated over, one unit per edge,
        each at a published point that permits observation and patching and is
        differentiable. A unit with no coordinate covers its whole point, and
        units of several edges may overlap.
    clean : Tensor
        The endpoint whose scalar metric is attributed.
    counterfactual : Tensor
        The endpoint whose target values stand in for the clean values.
    forward : Callable[[Tensor], Tensor]
        The pass from one path input to the scalar metric.
    steps : int
        The positive number of right-endpoint integration steps.
    dataset : str
        The handle of the data the two passes ran over, recorded in the provenance
        of every effect.

    Returns
    -------
    EdgeAttribution[T]
        For each edge, the counterfactual-minus-clean displacement at its target
        dotted with the float64 mean gradient there over the right endpoints,
        summed over the entries the edge's unit indexes. The estimator costs
        ``steps + 1`` forwards and ``steps`` backwards.

    Raises
    ------
    ValueError
        If an input, step count, site, metric, reading, gradient or result is invalid.
    """
    many = ensure_positive("an attribution's integration step count", steps)
    clean, counterfactual = _inputs(clean, counterfactual)
    difference = clean - counterfactual

    def right_endpoints() -> Iterator[Tensor]:
        for index in range(1, many + 1):
            if index == many:
                yield clean
            else:
                yield counterfactual + difference * (index / many)

    return _attribute(
        model,
        graph,
        targets=targets,
        counterfactual=counterfactual,
        forward=forward,
        path=right_endpoints(),
        steps=many,
        dataset=dataset,
        method="eap-ig",
    )


def first_order_attribute_edges[T, ReadingT: Reading](
    model: Probed[ReadingT],
    graph: ComputationGraph[T],
    /,
    *,
    targets: Mapping[GraphEdge[T], Unit],
    clean: Tensor,
    counterfactual: Tensor,
    forward: Callable[[Tensor], Tensor],
    dataset: str,
) -> EdgeAttribution[T]:
    """Attribute edges with the gradient at the clean endpoint alone.

    Parameters
    ----------
    model : Probed[ReadingT]
        The component the graph's nodes are published by.
    graph : ComputationGraph[T]
        The graph whose edges are to be attributed.
    targets : Mapping[GraphEdge[T], Unit]
        The coordinate each edge's effect is read over, as for `attribute_edges`.
    clean : Tensor
        The endpoint whose scalar metric is attributed.
    counterfactual : Tensor
        The endpoint whose target values stand in for the clean values.
    forward : Callable[[Tensor], Tensor]
        The pass from one endpoint to the scalar metric.
    dataset : str
        The handle of the data the two passes ran over.

    Returns
    -------
    EdgeAttribution[T]
        The counterfactual-minus-clean displacement dotted with the clean endpoint
        gradient over the entries each edge's unit indexes.

    Raises
    ------
    ValueError
        If an input, site, metric, reading, gradient or result is invalid.
    """
    clean, counterfactual = _inputs(clean, counterfactual)

    return _attribute(
        model,
        graph,
        targets=targets,
        counterfactual=counterfactual,
        forward=forward,
        path=iter((clean,)),
        steps=1,
        dataset=dataset,
        method="eap",
    )
