"""Span reconstruction against the representation the chart holds."""

from enum import Enum, auto
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, NamedTuple, override

import torch

from lofi._checks import CHECKS, ensure_name
from lofi.nn._values import Encoded
from lofi.nn.loss._errors import MissingOutputError
from lofi.nn.loss._parser import SPAN_AXES
from lofi.nn.loss._ports import LossTerm
from lofi.registry import ComponentPlugin
from lofi.spec.losses.span_reconstruction import (
    SpanNormalizationName,
    SpanReconstructionLossSettings,
    SpanReconstructionLossSpec,
    SpanWeightingName,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import Tensor

    from lofi.deduce import SpanIndex
    from lofi.nn._values import LogDistribution
    from lofi.nn.loss._ports import (
        LossComponentBindings,
        PublishedOutputs,
        SpanTokenDecoder,
        SurfaceItems,
    )
    from lofi.nn.tokens.embedder import TokenVocabulary

__all__ = [
    "SpanNormalization",
    "SpanReconstructionObjective",
    "SpanWeighting",
]

_CELL_RANK: Final[int] = 3
"""The rank of a chart cell representation."""

_MASK_RANK: Final[int] = 2
"""The widest mask a chart cell representation may have."""

_SCORE_AXES: Final[tuple[str, ...]] = ("span", "position", "token")
"""The axes the decoder scores a batch of spans over."""


class SpanWeighting(Enum):
    """Specifies how much of the term each span takes."""

    #: Every span scored takes the same share.
    UNIFORM = auto()

    #: A span takes the probability that it is a constituent.
    MARGINAL = auto()


class SpanNormalization(Enum):
    """Specifies the denominator of the reconstruction objective."""

    SPAN = auto()
    POSITION = auto()


class _Layer(NamedTuple):
    """One span length's contribution to the term.

    Parameters
    ----------
    weighted : Tensor
        The weighted sum of the mean negative log probability of each span's row.
    weight : Tensor
        The weight those spans took together.
    tokens : Tensor
        The unweighted sum over every position scored, detached.
    positions : int
        How many positions were scored.
    spans : int
        How many spans were scored.
    """

    weighted: Tensor
    weight: Tensor
    tokens: Tensor
    positions: int
    spans: int


def _checked_cells(cells: Encoded, spans: SpanIndex, *, items: int) -> None:
    """Raise unless the cells are addressed the way the bundle's spans are.

    Parameters
    ----------
    cells : Encoded
        The representation the chart holds for each span.
    spans : SpanIndex
        The addressing the bundle's span axis is over.
    items : int
        How many items the batch holds.

    Raises
    ------
    ValueError
        If the cells are not of rank three over an item and a span axis, if either
        extent disagrees with the batch or the index, or if a mask reaches past the
        span axis.
    """
    if cells.tensor.ndim != _CELL_RANK:
        msg = (
            f"the chart cells are over the axes {cells.axes}, and one vector "
            "per span of each item is of rank three"
        )
        raise ValueError(msg)

    if cells.axes[:_MASK_RANK] != SPAN_AXES:
        msg = (
            f"the chart cells are over the axes {cells.axes}, and this "
            f"objective reads them over {SPAN_AXES} and a feature axis"
        )
        raise ValueError(msg)

    for axis, extent in zip(SPAN_AXES, (items, spans.num_spans), strict=True):
        if cells.size(axis) != extent:
            msg = (
                f"the chart cells hold {cells.size(axis)} positions on the "
                f"axis {axis!r}, and the batch and its index name {extent}"
            )
            raise ValueError(msg)

    mask = cells.mask

    if mask is not None and mask.ndim > _MASK_RANK:
        msg = (
            f"the chart cells are masked over {cells.mask_axes}, so the mask "
            "divides a cell's features rather than saying which cells are real"
        )
        raise ValueError(msg)


def _live_spans(cells: Encoded, *, items: int, num_spans: int) -> Tensor:
    """Return which cell of which item holds a representation.

    Parameters
    ----------
    cells : Encoded
        The representation the chart holds for each span.
    items : int
        How many items the batch holds.
    num_spans : int
        How many spans the index addresses.

    Returns
    -------
    Tensor
        A boolean tensor of rank 2 over the item and span axes.
    """
    mask = cells.mask

    if mask is None:
        return torch.ones(
            (items, num_spans), dtype=torch.bool, device=cells.tensor.device
        )

    if mask.ndim < _MASK_RANK:
        return mask.unsqueeze(1).expand(items, num_spans)

    return mask


def _item_lengths(words: tuple[tuple[str, ...], ...], device: torch.device) -> Tensor:
    """Return how many words each item holds.

    Parameters
    ----------
    words : tuple[tuple[str, ...], ...]
        The words of each item, in item order and then in word order.
    device : torch.device
        Where the batch's tensors sit.

    Returns
    -------
    Tensor
        Rank 1 over the item axis, at integer precision.
    """
    return torch.tensor([len(item) for item in words], dtype=torch.long, device=device)


def _token_ids(
    words: tuple[tuple[str, ...], ...],
    vocabulary: TokenVocabulary,
    *,
    width: int,
    device: torch.device,
) -> Tensor:
    """Return the vocabulary row of every word, padded to the chart's width.

    Parameters
    ----------
    words : tuple[tuple[str, ...], ...]
        The words of each item, in item order and then in word order.
    vocabulary : TokenVocabulary
        The rows the decoder scores over.
    width : int
        How many words the longest item the index addresses may hold.
    device : torch.device
        Where the batch's tensors sit.

    Returns
    -------
    Tensor
        Rank 2 over the item and word axes, at integer precision, with the positions
        past an item's own words holding the unknown row, which no span reaches.

    Raises
    ------
    ValueError
        If an item holds more words than the index addresses.
    """
    rows: list[list[int]] = []

    for position, item in enumerate(words):
        if len(item) > width:
            msg = (
                f"item {position} holds {len(item)} words, and its chart is "
                f"indexed over {width}"
            )
            raise ValueError(msg)

        ids = [vocabulary.index_of(word) for word in item]

        ids.extend([vocabulary.unknown_index] * (width - len(item)))
        rows.append(ids)

    return torch.tensor(rows, dtype=torch.long, device=device)


def _checked_marginal(constituent: Encoded, spans: SpanIndex, *, items: int) -> Tensor:
    """Return which spans a derivation may build, refusing a misaddressed marginal.

    Parameters
    ----------
    constituent : Encoded
        The probability that each span is a constituent.
    spans : SpanIndex
        The addressing the bundle's span axis is over.
    items : int
        How many items the batch holds.

    Returns
    -------
    Tensor
        A boolean tensor of rank 2 over the item and span axes.

    Raises
    ------
    ValueError
        If the marginal is not over the item and span axes, if either extent disagrees
        with the batch or the index, or if it names no buildable spans.
    """
    if constituent.axes != SPAN_AXES:
        msg = (
            f"the constituent marginal is over the axes {constituent.axes}, "
            f"and this objective reads it over {SPAN_AXES}"
        )
        raise ValueError(msg)

    for axis, extent in zip(SPAN_AXES, (items, spans.num_spans), strict=True):
        if constituent.size(axis) != extent:
            msg = (
                f"the constituent marginal holds {constituent.size(axis)} "
                f"positions on the axis {axis!r}, and the batch and its index "
                f"name {extent}"
            )
            raise ValueError(msg)

    weighed = constituent.mask

    if weighed is None or weighed.ndim != _MASK_RANK:
        msg = "a constituent marginal names which spans a derivation may build"
        raise ValueError(msg)

    return weighed


def _checked_scores(scored: LogDistribution, targets: Tensor) -> None:
    """Raise unless the decoder scored the rows it was handed.

    Parameters
    ----------
    scored : LogDistribution
        What the decoder returned.
    targets : Tensor
        The rows it was asked to score.

    Raises
    ------
    ValueError
        If the axes or the span and position extents do not match the request, or if
        the distribution is not normalized over the token axis.
    """
    if scored.axes != _SCORE_AXES:
        msg = (
            f"the decoder scored the axes {scored.axes}, and a span's words "
            f"are scored over {_SCORE_AXES}"
        )
        raise ValueError(msg)

    wanted = tuple(targets.shape)
    given = tuple(scored.tensor.shape[:_MASK_RANK])

    if given != wanted:
        msg = (
            f"the decoder was handed {wanted[0]} spans of {wanted[1]} "
            f"positions and scored {given[0]} of {given[1]}"
        )
        raise ValueError(msg)

    if scored.normalized_over != _SCORE_AXES[-1]:
        msg = (
            f"the decoder's scores are normalized over {scored.normalized_over!r}, "
            f"and a token's log probability is read from a distribution over "
            f"{_SCORE_AXES[-1]!r}"
        )
        raise ValueError(msg)


class SpanReconstructionObjective[BatchT: SurfaceItems]:
    """Scores the words a span covers against the representation published for it.

    Each computed span is decoded against its words followed by the decoder's end
    token. Uniform
    weighting treats scored spans equally; marginal weighting uses detached
    constituent probabilities.

    Parameters
    ----------
    decoder : SpanTokenDecoder
        What scores a span's words given the representation of that span.
    weighting : SpanWeighting
        How much of the term each span takes.
    max_span_length : int | None
        Largest span reconstructed, or None for every represented length.
    normalization : SpanNormalization
        Whether spans or token positions form the denominator.
    name : str
        The name the term is logged and namespaced under.

    Raises
    ------
    KeyError
        If the decoder's vocabulary sets no row aside for its end token.
    TypeError
        If a name is not a string.
    ValueError
        If a name is empty.
    """

    __slots__ = (
        "_decoder",
        "_end",
        "_end_id",
        "_max_span_length",
        "_name",
        "_normalization",
        "_requires",
        "_weighting",
    )

    def __init__(
        self,
        decoder: SpanTokenDecoder,
        *,
        weighting: SpanWeighting,
        max_span_length: int | None,
        normalization: SpanNormalization,
        name: str,
    ) -> None:
        self._decoder = decoder
        self._end = ensure_name("the end token", decoder.end)
        self._end_id = decoder.vocabulary.reserved.index_of(self._end)
        self._weighting = weighting
        if max_span_length is not None and max_span_length < 1:
            msg = f"a maximum span length is positive, not {max_span_length!r}"
            raise ValueError(msg)

        self._max_span_length = max_span_length
        self._normalization = normalization
        self._name = ensure_name("the objective's name", name)
        reads = {"spans", "cells", "batch"}

        if weighting is SpanWeighting.MARGINAL:
            reads.add("marginals")

        self._requires = frozenset(reads)

    @property
    def name(self) -> str:
        """Return the name this objective is logged and namespaced under."""
        return self._name

    @property
    def requires(self) -> frozenset[str]:
        """Return the published outputs this objective reads, by name."""
        return self._requires

    @property
    def weighting(self) -> SpanWeighting:
        """Return how much of the term each span takes."""
        return self._weighting

    @property
    def max_span_length(self) -> int | None:
        """Return the largest reconstructed span, if bounded."""
        return self._max_span_length

    @property
    def normalization(self) -> SpanNormalization:
        """Return whether spans or token positions form the denominator."""
        return self._normalization

    @property
    def end(self) -> str:
        """Return the reserved token each span's words are followed by."""
        return self._end

    @property
    def vocabulary(self) -> TokenVocabulary:
        """Return the vocabulary the target rows are drawn from."""
        return self._decoder.vocabulary

    def _weights(
        self,
        outputs: PublishedOutputs[BatchT],
        *,
        spans: SpanIndex,
        items: int,
    ) -> tuple[Tensor, Tensor] | None:
        """Return the per-span weight and which spans take one.

        Parameters
        ----------
        outputs : PublishedOutputs[BatchT]
            What the graph published for the step.
        spans : SpanIndex
            The addressing the bundle's span axis is over.
        items : int
            How many items the batch holds.

        Returns
        -------
        tuple[Tensor, Tensor] | None
            The detached weight of each span and which spans are weighed, both of rank
            2 over the item and span axes, or None under the uniform weighting.

        Raises
        ------
        MissingOutputError
            If the step published no marginals and the weighting needs them.
        ValueError
            If the constituent marginal is not addressed the way the cells are.
        """
        if self._weighting is not SpanWeighting.MARGINAL:
            return None

        marginals = outputs.marginals

        if marginals is None:
            raise MissingOutputError(
                objective=self._name,
                missing=frozenset({"marginals"}),
                published=frozenset(),
            )

        constituent = marginals.constituent
        weighed = _checked_marginal(constituent, spans, items=items)

        return constituent.tensor.detach(), weighed

    def _layer(
        self,
        *,
        selected: Tensor,
        cells: Tensor,
        targets: Tensor,
        weights: Tensor,
    ) -> _Layer:
        """Return one span length's contribution to the term.

        Parameters
        ----------
        selected : Tensor
            Which span of which item is scored, of rank 2 over the item axis and the
            starts of this length.
        cells : Tensor
            The representation of every span of this length, of rank 3 over the item
            axis, the starts and the feature axis.
        targets : Tensor
            The rows to score, of rank 3 over the item axis, the starts and the
            positions.
        weights : Tensor
            The share each span takes, of rank 2 over the item axis and the starts.

        Raises
        ------
        ValueError
            If the decoder did not score the rows it was handed.
        """
        rows = targets[selected]
        prefix = cells[selected]
        scored = self._decoder.score_tokens(
            Encoded(prefix, ("span", "width")), targets=rows
        )

        if CHECKS:
            _checked_scores(scored, rows)

        taken = scored.tensor.gather(2, rows.unsqueeze(2)).squeeze(2)
        nll = -taken
        share = weights[selected]

        if self._normalization is SpanNormalization.SPAN:
            weighted = (nll.mean(dim=1) * share).sum()
            denominator = share.sum()
        else:
            weighted = (nll.sum(dim=1) * share).sum()
            denominator = share.sum() * nll.shape[1]

        return _Layer(
            weighted,
            denominator,
            nll.detach().sum(),
            nll.numel(),
            int(selected.sum()),
        )

    def evaluate(self, outputs: PublishedOutputs[BatchT], /) -> LossTerm:
        """Return how poorly the step's cells reconstruct the words they cover.

        Parameters
        ----------
        outputs : PublishedOutputs[BatchT]
            What the graph published for the step.

        Returns
        -------
        LossTerm
            The weighted mean negative log probability of a span's words, counted in
            spans, reporting the unweighted mean over scored positions as
            ``per_token``.

        Raises
        ------
        MissingOutputError
            If the step published no cells, or no marginals under a weighting that
            needs them.
        ValueError
            If the cells, the marginals or the words disagree with the span index the
            step published, or if the decoder did not score the rows it was handed.
        """
        cells = outputs.cells

        if cells is None:
            raise MissingOutputError(
                objective=self._name,
                missing=frozenset({"cells"}),
                published=frozenset(),
            )

        spans = outputs.spans
        words = outputs.batch.words

        _checked_cells(cells, spans, items=len(words))

        if not words or spans.num_spans == 0:
            return LossTerm.inapplicable(cells.tensor)

        device = cells.tensor.device
        live = _live_spans(cells, items=len(words), num_spans=spans.num_spans)
        weighed = self._weights(outputs, spans=spans, items=len(words))
        lengths = _item_lengths(words, device)
        ids = _token_ids(words, self._decoder.vocabulary, width=spans.n, device=device)
        ends = ids.new_full((len(words), 1, 1), self._end_id)
        total = cells.tensor.new_zeros(())
        weight = cells.tensor.new_zeros(())
        tokens = cells.tensor.new_zeros(())
        positions = 0
        scored = 0

        maximum = (
            spans.n
            if self._max_span_length is None
            else min(spans.n, self._max_span_length)
        )

        for length in range(1, maximum + 1):
            base = spans.span_id(0, length)
            starts = spans.n - length + 1
            here = live[:, base : base + starts]
            fits = torch.arange(starts, device=device) + length <= lengths.unsqueeze(1)

            selected = here & fits
            share = total.new_ones((len(words), starts))

            if weighed is not None:
                weights, allowed = weighed
                selected = selected & allowed[:, base : base + starts]
                share = weights[:, base : base + starts]

            if not bool(selected.any()):
                continue

            windows = ids.unfold(1, length, 1)
            layer = self._layer(
                selected=selected,
                cells=cells.tensor[:, base : base + starts, :],
                targets=torch.cat((windows, ends.expand(-1, starts, -1)), dim=2),
                weights=share,
            )
            total = total + layer.weighted
            weight = weight + layer.weight
            tokens = tokens + layer.tokens
            positions += layer.positions
            scored += layer.spans

        if scored == 0 or float(weight) == 0.0:
            return LossTerm.inapplicable(cells.tensor)

        return LossTerm(
            total / weight,
            scored,
            diagnostics={"per_token": tokens / positions},
        )

    @override
    def __repr__(self) -> str:
        """Return the name, the end token and the weighting."""
        return (
            f"SpanReconstructionObjective(name={self._name!r}, "
            f"end={self._end!r}, weighting={self._weighting.name}, "
            f"max_span_length={self._max_span_length!r}, "
            f"normalization={self._normalization.name})"
        )


_WEIGHTINGS: Final[Mapping[SpanWeightingName, SpanWeighting]] = MappingProxyType(
    {
        SpanWeightingName.UNIFORM: SpanWeighting.UNIFORM,
        SpanWeightingName.MARGINAL: SpanWeighting.MARGINAL,
    }
)
"""Each declared weighting and the share it gives a span."""

_NORMALIZATIONS: Final[Mapping[SpanNormalizationName, SpanNormalization]] = (
    MappingProxyType(
        {
            SpanNormalizationName.SPAN: SpanNormalization.SPAN,
            SpanNormalizationName.POSITION: SpanNormalization.POSITION,
        }
    )
)
"""Each declared denominator and its runtime normalization."""


def _build[BatchT: SurfaceItems](
    settings: SpanReconstructionLossSettings, context: LossComponentBindings, /
) -> SpanReconstructionObjective[BatchT]:
    """Build span reconstruction through the declared span decoder.

    Parameters
    ----------
    settings : SpanReconstructionLossSettings
        The span-decoder role and the share each span takes.
    context : LossComponentBindings
        The declared name and the built components the role names.

    Raises
    ------
    ValueError
        If the role is not built.
    KeyError
        If the decoder's vocabulary sets no row aside for the end token.
    """
    return SpanReconstructionObjective(
        context.span_decoder(settings.decoder),
        weighting=_WEIGHTINGS[settings.weighting],
        max_span_length=settings.max_span_length,
        normalization=_NORMALIZATIONS[settings.normalization],
        name=context.name,
    )


PLUGIN: Final = ComponentPlugin(spec=SpanReconstructionLossSpec, build=_build)
"""The span-reconstruction objective's plugin."""
