"""Constituent marginals over computation graph steps, and aggregates under them.

A step's marginal is the probability, under the chart's own distribution over
derivations, that a derivation passes through the item the step concluded: the
product of the item's inside and outside weights, divided by the partition. An
aggregate over spans is reported under two weightings at once, the marginals and
the uniform one, so that the two can be compared and neither stands alone.
"""

from math import fsum
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, override

import torch

from lofi._checks import ensure_finite, ensure_name, ensure_non_negative
from lofi.probe._graph import ComputationGraph, GraphNode

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence

    from torch import Tensor

    from lofi.deduce import Cell, Chart, Outside, SpanIndex

__all__ = ["DerivationWeights", "SpanAggregate", "SpanWeighting"]

_ROUNDING: Final[float] = 1e-9
"""How far a marginal may fall outside the unit interval before it is not one."""


class SpanAggregate:
    """One quantity aggregated over spans, under the marginals and uniformly.

    Parameters
    ----------
    weighted : float
        The mean of the quantity over the spans, each span weighted by its
        constituent marginal.
    uniform : float
        The mean of the quantity over the same spans, each weighted alike.
    count : int
        How many spans were aggregated over.

    Raises
    ------
    TypeError
        If a mean is not a number, or the count is not an integer.
    ValueError
        If a mean is infinite or a NaN, or the count is not positive.
    """

    __slots__ = ("_count", "_uniform", "_weighted")

    def __init__(self, weighted: float, uniform: float, count: int) -> None:
        self._weighted = ensure_finite("a weighted aggregate", weighted)
        self._uniform = ensure_finite("a uniform aggregate", uniform)
        self._count = ensure_non_negative("an aggregate's span count", count)

        if self._count == 0:
            msg = "an aggregate over no spans is not a number"
            raise ValueError(msg)

    @classmethod
    def over(
        cls, values: Tensor, weights: Tensor, live: Tensor | None = None
    ) -> SpanAggregate:
        """Return the aggregate of one number per span under one weight per span.

        Parameters
        ----------
        values : Tensor
            The quantity at each span, of any shape.
        weights : Tensor
            The constituent marginal of each span, of the same shape, each
            nonnegative.
        live : Tensor | None
            Which positions hold data, of the same shape, or None where all do.

        Returns
        -------
        SpanAggregate
            Over the live positions, the marginal-weighted mean and the uniform mean of
            the quantity.

        Raises
        ------
        ValueError
            If the shapes disagree, if no position is live, if a live weight is
            negative or not finite, or if the live weights sum to zero, since the
            model then puts no mass on any span aggregated over.
        """
        if tuple(weights.shape) != tuple(values.shape):
            msg = (
                f"weights of shape {tuple(weights.shape)} cannot weigh values of "
                f"shape {tuple(values.shape)}"
            )
            raise ValueError(msg)

        covering = torch.ones_like(values, dtype=torch.bool) if live is None else live

        if tuple(covering.shape) != tuple(values.shape):
            msg = (
                f"a mask of shape {tuple(covering.shape)} does not cover values of "
                f"shape {tuple(values.shape)}"
            )
            raise ValueError(msg)

        count = int(covering.sum())

        if count == 0:
            msg = "every span is masked off, so there is nothing to aggregate"
            raise ValueError(msg)

        kept_values = values[covering].to(torch.float64)
        kept_weights = weights[covering].to(torch.float64)

        if not bool(torch.isfinite(kept_weights).all()) or bool(
            (kept_weights < 0).any()
        ):
            msg = "a span's weight is negative or not finite"
            raise ValueError(msg)

        mass = float(kept_weights.sum())

        if mass == 0.0:
            msg = "the weights put no mass on any live span, so no weighted mean exists"
            raise ValueError(msg)

        weighted = float((kept_values * kept_weights).sum()) / mass
        uniform = float(kept_values.sum()) / count

        return cls(weighted, uniform, count)

    @property
    def weighted(self) -> float:
        """Return the mean under the constituent marginals."""
        return self._weighted

    @property
    def uniform(self) -> float:
        """Return the mean under equal weights."""
        return self._uniform

    @property
    def count(self) -> int:
        """Return how many spans were aggregated over."""
        return self._count

    def _identity(self) -> tuple[float, float, int]:
        """Return the three fields an aggregate is equal on."""
        return (self._weighted, self._uniform, self._count)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value holds the same two means over as many spans."""
        if type(other) is not SpanAggregate:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the two means and the count."""
        return hash((SpanAggregate, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this aggregate."""
        return (
            f"SpanAggregate(weighted={self._weighted!r}, "
            f"uniform={self._uniform!r}, count={self._count!r})"
        )


def _checked_marginals[T](
    graph: ComputationGraph[T], marginals: Mapping[GraphNode[T], float]
) -> dict[GraphNode[T], float]:
    """Return one marginal per step of a graph.

    Parameters
    ----------
    graph : ComputationGraph[T]
        The graph whose steps are weighted.
    marginals : Mapping[GraphNode[T], float]
        The marginal of each step.

    Returns
    -------
    dict[GraphNode[T], float]
        A copy.

    Raises
    ------
    TypeError
        If a marginal is not a number.
    ValueError
        If a step of the graph has no marginal, if a marginal is given for a step the
        graph does not hold, or if a marginal is outside the closed unit interval by
        more than rounding.
    """
    held: dict[GraphNode[T], float] = {}

    for node, marginal in marginals.items():
        if node not in graph.nodes:
            msg = f"a marginal is given for {node!r}, which is not a node here"
            raise ValueError(msg)

        probability = ensure_finite(f"the marginal of {node!r}", marginal)

        if not -_ROUNDING <= probability <= 1.0 + _ROUNDING:
            msg = f"the marginal of {node!r} is {probability!r}, not a probability"
            raise ValueError(msg)

        held[node] = min(max(probability, 0.0), 1.0)

    for node in graph.nodes:
        if node not in held:
            msg = f"the graph holds {node!r}, which has no marginal"
            raise ValueError(msg)

    return held


class DerivationWeights[T]:
    """The constituent marginal of every step of a computation graph.

    Parameters
    ----------
    graph : ComputationGraph[T]
        The graph whose steps are weighted.
    marginals : Mapping[GraphNode[T], float]
        The probability, under the chart's distribution over derivations, that a
        derivation passes through each step's item, keyed by the step.

    Raises
    ------
    TypeError
        If the graph is not a computation graph, or a marginal is not a number.
    ValueError
        If a step of the graph has no marginal, if a marginal is given for a step the
        graph does not hold, or if a marginal is outside the closed unit interval.
    """

    __slots__ = ("_graph", "_marginals")

    def __init__(
        self, graph: ComputationGraph[T], marginals: Mapping[GraphNode[T], float]
    ) -> None:
        if type(graph) is not ComputationGraph:
            msg = f"derivation weights are over a computation graph, not {graph!r}"
            raise TypeError(msg)

        self._graph = graph
        self._marginals = MappingProxyType(_checked_marginals(graph, marginals))

    @classmethod
    def from_outside(
        cls, graph: ComputationGraph[T], chart: Chart[T, float], outside: Outside[T]
    ) -> DerivationWeights[T]:
        """Return the marginal of every step, read off an inside-outside pass.

        Parameters
        ----------
        graph : ComputationGraph[T]
            The graph of the forest's own dataflow, one step per item.
        chart : Chart[T, float]
            The forest the graph was built from.
        outside : Outside[T]
            The inside-outside pass over that forest, which holds each item's inside
            weight, its outside weight and the partition.

        Returns
        -------
        DerivationWeights[T]
            For each step, the item's inside weight times its outside weight, divided
            by the partition.

        Raises
        ------
        ValueError
            If the pass was computed from another chart, or if the graph's steps and
            the forest's items do not correspond exactly.
        """
        if not outside.is_for(chart):
            msg = "the outside result was computed from another chart"
            raise ValueError(msg)

        items = frozenset(
            key
            for span in chart.index.spans_by_length()
            for key in chart.items_at(span)
        )
        concluded = frozenset(node.key for node in graph.nodes)

        unconcluded = items - concluded

        if unconcluded:
            msg = (
                f"the forest holds {next(iter(unconcluded))!r}, which no step of "
                "the graph concludes"
            )
            raise ValueError(msg)

        unheld = concluded - items

        if unheld:
            msg = (
                f"a step concludes {next(iter(unheld))!r}, which the forest does "
                "not hold"
            )
            raise ValueError(msg)

        return cls(
            graph,
            {node: outside.constituent_marginal(node.key) for node in graph.nodes},
        )

    @property
    def graph(self) -> ComputationGraph[T]:
        """Return the graph whose steps are weighted."""
        return self._graph

    @property
    def nodes(self) -> frozenset[GraphNode[T]]:
        """Return the weighted steps."""
        return self._graph.nodes

    def of(self, node: GraphNode[T]) -> float:
        """Return the constituent marginal of one step.

        Parameters
        ----------
        node : GraphNode[T]
            The step to look up.

        Raises
        ------
        KeyError
            If the graph does not hold the step.
        """
        try:
            return self._marginals[node]

        except KeyError as error:
            msg = f"{node!r} is not a step of the weighted graph"
            raise KeyError(msg) from error

    def at(self, span: Cell) -> float:
        """Return the expected number of items concluded over one span.

        Parameters
        ----------
        span : Cell
            The half-open span.

        Returns
        -------
        float
            The sum of the marginals of the steps covering exactly that span, which is
            the number of items over it that a derivation is expected to pass through.
        """
        return fsum(
            marginal for node, marginal in self._marginals.items() if node.span == span
        )

    def items(self) -> tuple[tuple[GraphNode[T], float], ...]:
        """Return every step beside its marginal, in no particular order."""
        return tuple(self._marginals.items())

    def aggregate(self, values: Mapping[GraphNode[T], float]) -> SpanAggregate:
        """Return the aggregate of one number per step under the marginals.

        Parameters
        ----------
        values : Mapping[GraphNode[T], float]
            The quantity at each step to aggregate over, keyed by the step.

        Returns
        -------
        SpanAggregate
            Over exactly the steps given, the marginal-weighted mean and the uniform
            mean of the quantity.

        Raises
        ------
        TypeError
            If a value is not a number.
        ValueError
            If no step is given, if a value is infinite or a NaN, or if the marginals
            of the given steps sum to zero.
        KeyError
            If a value is given for a step the graph does not hold.
        """
        if not values:
            msg = "an aggregate over no steps is not a number"
            raise ValueError(msg)

        weights = [self.of(node) for node in values]
        numbers = [
            ensure_finite(f"the value at {node!r}", value)
            for node, value in values.items()
        ]
        mass = fsum(weights)

        if mass == 0.0:
            msg = (
                "the marginals put no mass on any step given, so no weighted mean "
                "exists"
            )
            raise ValueError(msg)

        weighted = fsum(
            weight * number for weight, number in zip(weights, numbers, strict=True)
        )

        return SpanAggregate(
            weighted / mass, fsum(numbers) / len(numbers), len(numbers)
        )

    def __len__(self) -> int:
        """Return how many steps are weighted."""
        return len(self._marginals)

    def __iter__(self) -> Iterator[GraphNode[T]]:
        """Return the weighted steps, in no particular order."""
        return iter(self._marginals)

    def _identity(
        self,
    ) -> tuple[ComputationGraph[T], frozenset[tuple[GraphNode[T], float]]]:
        """Return the two fields the weights are equal on."""
        return (self._graph, frozenset(self._marginals.items()))

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value weighs the same graph the same way."""
        if type(other) is not DerivationWeights:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the graph and the marginals."""
        return hash((DerivationWeights, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the graph and how many steps are weighted."""
        return f"DerivationWeights({self._graph!r}, steps={len(self._marginals)})"


def _weighting_axes(axes: Iterable[str], rank: int) -> tuple[str, ...]:
    """Return one axis name per axis of a weighting.

    Parameters
    ----------
    axes : Iterable[str]
        The names, outermost first.
    rank : int
        How many axes the weighting has.

    Returns
    -------
    tuple[str, ...]
        The names, unchanged.

    Raises
    ------
    TypeError
        If a name is not a string.
    ValueError
        If a name is empty, if two names agree, or if the count does not match the rank.
    """
    named = tuple(
        ensure_name(f"axis {position} of a weighting", name)
        for position, name in enumerate(axes)
    )

    if len(named) != rank:
        msg = f"a weighting of rank {rank} needs {rank} axis names, but got {named}"
        raise ValueError(msg)

    if len(set(named)) != len(named):
        msg = f"a weighting declares an axis twice: {named}"
        raise ValueError(msg)

    return named


class SpanWeighting:
    """Constituent marginals laid out over the leading axes of a batch of readings.

    Parameters
    ----------
    tensor : Tensor
        The expected number of constituents at each position, nonnegative and
        finite throughout.
    axes : Iterable[str]
        The name of each axis, outermost first, as the readings it weighs declare
        them.

    Raises
    ------
    TypeError
        If the weights are not a tensor, or an axis name is not a string.
    ValueError
        If a weight is negative or not finite, or the axis names do not cover the
        tensor's axes exactly.
    """

    __slots__ = ("_axes", "_tensor")

    def __init__(self, tensor: Tensor, *, axes: Iterable[str]) -> None:
        if not isinstance(tensor, torch.Tensor):
            msg = f"a weighting is a tensor, not {tensor!r}"
            raise TypeError(msg)

        held = tensor.detach()

        if not bool(torch.isfinite(held).all()) or bool((held < 0).any()):
            msg = "a weighting holds a weight that is negative or not finite"
            raise ValueError(msg)

        self._tensor = held
        self._axes = _weighting_axes(axes, held.ndim)

    @classmethod
    def from_weights[T](
        cls,
        weights: Sequence[DerivationWeights[T]],
        index: SpanIndex,
        *,
        axes: tuple[str, str],
    ) -> SpanWeighting:
        """Return the marginals of a batch of graphs, laid out by item and by span.

        Parameters
        ----------
        weights : Sequence[DerivationWeights[T]]
            One weighted graph per item of the batch, in batch order.
        index : SpanIndex
            The span addressing the batch's readings are laid out under, which covers
            every item's spans.
        axes : tuple[str, str]
            The names of the item axis and the span axis, as the readings declare
            them.

        Returns
        -------
        SpanWeighting
            Of rank two, holding at each item and span the expected number of items
            concluded over that span, and zero at spans the graph has no step over.

        Raises
        ------
        ValueError
            If a step covers a span the index does not address.
        """
        laid = torch.zeros(len(weights), index.num_spans, dtype=torch.float64)

        for item, weighted in enumerate(weights):
            for node, marginal in weighted.items():
                start, end = node.span

                if end > index.n:
                    msg = (
                        f"{node!r} covers [{start}, {end}), which an index over "
                        f"{index.n} tokens does not address"
                    )
                    raise ValueError(msg)

                laid[item, index.span_id(start, end)] += marginal

        return cls(laid, axes=axes)

    @property
    def tensor(self) -> Tensor:
        """Return the weight at each position."""
        return self._tensor

    @property
    def axes(self) -> tuple[str, ...]:
        """Return the name of each axis, outermost first."""
        return self._axes

    def aligned(self, axes: Sequence[str], shape: Sequence[int]) -> Tensor:
        """Return the weights broadcast over a reading's axes.

        Parameters
        ----------
        axes : Sequence[str]
            The reading's axis names, outermost first, which must begin with this
            weighting's own.
        shape : Sequence[int]
            The reading's extent along each of those axes.

        Returns
        -------
        Tensor
            Of the reading's shape, holding at each position the weight of the
            leading coordinates it sits under.

        Raises
        ------
        ValueError
            If the reading's leading axes are not this weighting's, by name or by
            extent.
        """
        leading = tuple(axes[: len(self._axes)])
        extents = tuple(shape[: len(self._axes)])

        if leading != self._axes:
            msg = (
                f"a weighting over {self._axes} cannot weigh a reading over "
                f"{tuple(axes)}"
            )
            raise ValueError(msg)

        if extents != tuple(self._tensor.shape):
            msg = (
                f"a weighting of shape {tuple(self._tensor.shape)} cannot weigh a "
                f"reading of shape {tuple(shape)}"
            )
            raise ValueError(msg)

        trailing = (1,) * (len(shape) - len(self._axes))

        return self._tensor.reshape(tuple(self._tensor.shape) + trailing).expand(
            tuple(shape)
        )

    @override
    def __repr__(self) -> str:
        """Return the axes and the extents, without the numbers."""
        return (
            f"SpanWeighting(axes={self._axes!r}, shape={tuple(self._tensor.shape)!r})"
        )
