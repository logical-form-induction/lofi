"""Diagnostics over exact chart split distributions and constituent marginals."""

from math import inf
from typing import TYPE_CHECKING, ClassVar, Final, override

import torch

from lofi.nn.sites import DTypeClass, Role
from lofi.probe._diagnose import Aggregation, DiagnosticResult, Measurement
from lofi.probe._dtypes import ensure_declared
from lofi.probe._marginals import SpanAggregate
from lofi.probe._scores import normalized_entropy
from lofi.probe._values import Observation

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import Tensor

    from lofi.deduce import SpanIndex
    from lofi.nn.sites import Site
    from lofi.probe._diagnose import DiagnosticContext
    from lofi.probe._values import Trace

__all__ = ["CutpointEntropy", "SpanLengthBias"]

_SPAN_AXES: Final = ("item", "span")
"""The axes of one scalar per chart span."""

_SPLIT_AXES: Final = ("item", "span", "split")
"""The axes of one value per split of every chart span."""

_CUTPOINT_ENTROPY: Final = "normalized-cutpoint-entropy"
"""The score name for split entropy divided by its rowwise maximum."""

_SPAN_LENGTH: Final = "normalized-span-length"
"""The score name for span width divided by item length."""


def _index(context: DiagnosticContext) -> SpanIndex:
    """Return the context's chart addressing.

    Parameters
    ----------
    context : DiagnosticContext
        What the observed components publish.

    Returns
    -------
    SpanIndex
        The addressing shared by their chart readings.

    Raises
    ------
    ValueError
        If no span index was supplied.
    """
    index = context.span_index

    if index is None:
        msg = "the diagnostic context has no SpanIndex for its chart observations"
        raise ValueError(msg)

    return index


def _observed(
    context: DiagnosticContext,
    taken: Mapping[Site, Observation],
    role: Role,
) -> tuple[tuple[Site, Observation], ...]:
    """Return all observations for one role, refusing omissions.

    Parameters
    ----------
    context : DiagnosticContext
        What the observed components publish.
    taken : Mapping[Site, Observation]
        What one scope observed.
    role : Role
        The role to collect.

    Returns
    -------
    tuple[tuple[Site, Observation], ...]
        Each site and its observation in publication order.

    Raises
    ------
    ValueError
        If a published site was not observed.
    """
    published = context.sites_for(role)

    for site in published:
        if site not in taken:
            msg = f"missing observation at published {role.name} site {site!r}"
            raise ValueError(msg)

    return tuple((site, taken[site]) for site in published)


def _validate_reading(
    context: DiagnosticContext,
    site: Site,
    reading: Observation,
    *,
    axes: tuple[str, ...],
    dtype_class: DTypeClass,
) -> None:
    """Validate one chart site's declaration and observed numeric domain.

    Parameters
    ----------
    context : DiagnosticContext
        What the site publishes.
    site : Site
        The site being checked.
    reading : Observation
        What was observed there.
    axes : tuple[str, ...]
        The exact chart axes the role requires.
    dtype_class : DTypeClass
        The exact numeric domain the role requires.

    Raises
    ------
    ValueError
        If the published or observed axes or numeric domain differ from the role's
        contract, or the values contradict that domain.
    """
    spec = context.spec_at(site)

    if spec.axes != axes or reading.axes != axes:
        msg = (
            f"{site!r} must publish and observe axes {axes!r}, not "
            f"{spec.axes!r} and {reading.axes!r}"
        )
        raise ValueError(msg)

    if spec.dtype_class is not dtype_class:
        msg = (
            f"{site!r} must publish {dtype_class.value} numbers, not "
            f"{spec.dtype_class.value}"
        )
        raise ValueError(msg)

    ensure_declared(reading, dtype_class)


def _shape(reading: Observation, index: SpanIndex, *, split: bool) -> None:
    """Validate one chart observation's extents against its span index.

    Parameters
    ----------
    reading : Observation
        The chart observation.
    index : SpanIndex
        The addressing it must follow.
    split : bool
        Whether the observation has a split axis.

    Raises
    ------
    ValueError
        If a span or split extent differs from the index.
    """
    wanted = (reading.extent("item"), index.num_spans)

    if split:
        wanted += (index.max_splits,)

    if tuple(reading.tensor.shape) != wanted:
        msg = (
            f"{reading.site!r} was observed with shape "
            f"{tuple(reading.tensor.shape)}, but {index!r} addresses {wanted}"
        )
        raise ValueError(msg)


def _span_geometry(index: SpanIndex, like: Tensor) -> tuple[Tensor, Tensor]:
    """Return every span's end and width beside a chart tensor.

    Parameters
    ----------
    index : SpanIndex
        The addressing to enumerate.
    like : Tensor
        The tensor that fixes device placement.

    Returns
    -------
    tuple[Tensor, Tensor]
        Rank-one integer tensors of span ends and widths in dense-id order.
    """
    spans = tuple(index.spans_by_length())
    ends = like.new_tensor([end for _, end in spans], dtype=torch.long)
    widths = like.new_tensor([end - start for start, end in spans], dtype=torch.long)

    return ends, widths


def _item_lengths(live: Tensor, ends: Tensor, site: Site) -> Tensor:
    """Return each item's length from an exact live-span mask.

    Parameters
    ----------
    live : Tensor
        Rank-two truth values over items and spans.
    ends : Tensor
        The end coordinate of each span.
    site : Site
        The source site, for refusal messages.

    Returns
    -------
    Tensor
        One positive length per item.

    Raises
    ------
    ValueError
        If an item has no live span.
    """
    lengths = torch.where(live, ends.unsqueeze(0), 0).amax(dim=1)

    if bool((lengths == 0).any()):
        missing = (lengths == 0).nonzero().flatten().tolist()
        msg = f"the observation at {site!r} has no live span for items {missing}"
        raise ValueError(msg)

    return lengths


def _roots(index: SpanIndex, lengths: Tensor, like: Tensor) -> Tensor:
    """Return a mask selecting each item's full-span root.

    Parameters
    ----------
    index : SpanIndex
        The chart addressing.
    lengths : Tensor
        One positive token count per item.
    like : Tensor
        A rank-two tensor over items and spans.

    Returns
    -------
    Tensor
        Truth values of the same shape, one true root per item.
    """
    roots = like.new_zeros(like.shape, dtype=torch.bool)

    for item, length in enumerate(lengths.tolist()):
        roots[item, index.span_id(0, length)] = True

    return roots


def _split_support(
    reading: Observation, index: SpanIndex
) -> tuple[Tensor, Tensor, Tensor]:
    """Return exact split support, item lengths and span widths.

    Parameters
    ----------
    reading : Observation
        One split-distribution observation.
    index : SpanIndex
        The addressing the observation follows.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        The rank-three live-split mask, one item length per row, and one width per
        span.

    Raises
    ------
    ValueError
        If a mask marks a structurally absent split, omits a live span's split, or
        does not describe a contiguous item prefix.
    """
    _shape(reading, index, split=True)
    items = reading.extent("item")
    structural = reading.tensor.new_tensor(index.split_mask, dtype=torch.bool)
    full = structural.unsqueeze(0).expand(items, -1, -1)

    if reading.mask is None:
        support = full
        lengths = reading.tensor.new_full((items,), index.n, dtype=torch.long)
    else:
        support = reading.covering()

        if bool((support & ~full).any()):
            msg = (
                f"the split mask at {reading.site!r} marks a structurally absent split"
            )
            raise ValueError(msg)

        ends, _ = _span_geometry(index, reading.tensor)
        rows = support.any(dim=2)
        inferred = torch.where(rows, ends.unsqueeze(0), 0).amax(dim=1)
        lengths = inferred.clamp(min=1)
        expected_rows = ends.unsqueeze(0) <= lengths.unsqueeze(1)
        expected = full & expected_rows.unsqueeze(2)

        if not torch.equal(support, expected):
            msg = (
                f"the split mask at {reading.site!r} does not match {index!r} "
                "over each item's live prefix"
            )
            raise ValueError(msg)

    _, widths = _span_geometry(index, reading.tensor)

    return support, lengths, widths


def _span_support(
    reading: Observation, index: SpanIndex
) -> tuple[Tensor, Tensor, Tensor]:
    """Return exact live spans, item lengths and span widths.

    Parameters
    ----------
    reading : Observation
        One constituent-marginal observation.
    index : SpanIndex
        The addressing the observation follows.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        The rank-two live-span mask, one item length per row, and one width per span.

    Raises
    ------
    ValueError
        If the mask does not describe a positive-length contiguous item prefix.
    """
    _shape(reading, index, split=False)
    ends, widths = _span_geometry(index, reading.tensor)

    if reading.mask is None:
        support = reading.tensor.new_ones(reading.tensor.shape, dtype=torch.bool)
        lengths = reading.tensor.new_full(
            (reading.extent("item"),), index.n, dtype=torch.long
        )
    else:
        support = reading.covering()
        lengths = _item_lengths(support, ends, reading.site)
        expected = ends.unsqueeze(0) <= lengths.unsqueeze(1)

        if not torch.equal(support, expected):
            msg = (
                f"the span mask at {reading.site!r} does not match {index!r} "
                "over each item's live prefix"
            )
            raise ValueError(msg)

    return support, lengths, widths


class CutpointEntropy:
    """Measures normalized entropy over non-forced, nonroot split choices."""

    __slots__ = ()

    tag: ClassVar[str] = "cutpoint-entropy"
    requires: ClassVar[frozenset[Role]] = frozenset({Role.SPLIT_DISTRIBUTION})
    scores: ClassVar[frozenset[str]] = frozenset({_CUTPOINT_ENTROPY})

    def run(self, trace: Trace, /, *, context: DiagnosticContext) -> DiagnosticResult:
        """Return cutpoint entropy under constituent and uniform span weights.

        Parameters
        ----------
        trace : Trace
            The exact split distributions observed during one scorer pass.
        context : DiagnosticContext
            The scorer's declarations, chart addressing and span weighting.

        Returns
        -------
        DiagnosticResult
            One aggregate and eligible-span count per split-distribution site.

        Raises
        ------
        ValueError
            If the context has no span index or weighting, a published site was not
            read, an observation disagrees with its role or index, no eligible span
            remains, or the eligible spans have no constituent mass.
        """
        taken = dict(context.readings(trace))
        observed = _observed(context, taken, Role.SPLIT_DISTRIBUTION)

        if not observed:
            return DiagnosticResult(self.tag, [])

        index = _index(context)
        measured: list[Measurement] = []
        aggregated: list[Aggregation] = []

        for site, reading in observed:
            _validate_reading(
                context,
                site,
                reading,
                axes=_SPLIT_AXES,
                dtype_class=DTypeClass.LOG_PROBABILITY,
            )
            support, lengths, _ = _split_support(reading, index)
            values = reading.tensor.masked_fill(~support, -inf)
            split_reading = Observation(site, values, axes=_SPLIT_AXES, mask=support)
            entropy = normalized_entropy(
                split_reading, DTypeClass.LOG_PROBABILITY, over="split"
            )
            choices = support.sum(dim=2)
            eligible = (choices >= 2) & ~_roots(index, lengths, choices)
            weights = context.weights_over(entropy.axes, entropy.tensor.shape)
            summary = SpanAggregate.over(entropy.tensor, weights, eligible)
            aggregated.append(Aggregation(site, _CUTPOINT_ENTROPY, summary))
            measured.append(Measurement(site, "spans", float(summary.count)))

        return DiagnosticResult(self.tag, measured, aggregated)

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this diagnostic."""
        return "CutpointEntropy()"


class SpanLengthBias:
    """Measures normalized proper-internal span length under observed marginals."""

    __slots__ = ()

    tag: ClassVar[str] = "span-length-bias"
    requires: ClassVar[frozenset[Role]] = frozenset({Role.CONSTITUENT_MARGINAL})
    scores: ClassVar[frozenset[str]] = frozenset({_SPAN_LENGTH})

    def run(self, trace: Trace, /, *, context: DiagnosticContext) -> DiagnosticResult:
        """Return proper-internal span length under marginal and uniform weights.

        Parameters
        ----------
        trace : Trace
            The exact constituent marginals observed during one scorer pass.
        context : DiagnosticContext
            The scorer's declarations and chart addressing.

        Returns
        -------
        DiagnosticResult
            One aggregate and eligible-span count per marginal site.

        Raises
        ------
        ValueError
            If the context has no span index, a published site was not read, an
            observation disagrees with its role or index, no proper internal span
            remains, or those spans have no marginal mass.
        """
        taken = dict(context.readings(trace))
        observed = _observed(context, taken, Role.CONSTITUENT_MARGINAL)

        if not observed:
            return DiagnosticResult(self.tag, [])

        index = _index(context)
        measured: list[Measurement] = []
        aggregated: list[Aggregation] = []

        for site, reading in observed:
            _validate_reading(
                context,
                site,
                reading,
                axes=_SPAN_AXES,
                dtype_class=DTypeClass.PROBABILITY,
            )
            support, lengths, widths = _span_support(reading, index)
            roots = _roots(index, lengths, support)
            eligible = support & (widths.unsqueeze(0) > 1) & ~roots
            normalized = widths.to(reading.tensor.dtype).unsqueeze(0) / lengths.to(
                reading.tensor.dtype
            ).unsqueeze(1)
            values = normalized.expand_as(reading.tensor)
            summary = SpanAggregate.over(values, reading.tensor, eligible)
            aggregated.append(Aggregation(site, _SPAN_LENGTH, summary))
            measured.append(Measurement(site, "spans", float(summary.count)))

        return DiagnosticResult(self.tag, measured, aggregated)

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this diagnostic."""
        return "SpanLengthBias()"
