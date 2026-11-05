"""A globally normalized scalar-potential chart over binary span trees."""

from math import inf
from typing import TYPE_CHECKING, Final, cast, override

import torch
from torch import nn

from lofi.deduce import SpanIndex
from lofi.nn._values import ChartReadouts, Encoded, LogDistribution
from lofi.nn.parsers._decode import mbr_trees
from lofi.nn.parsers._inside import InsideKernel
from lofi.nn.parsers._marginals import (
    derivation_entropy,
    log_constituent_marginals,
    log_split_posterior_by_gradient,
)
from lofi.nn.parsers._outside import OutsideKernel
from lofi.nn.parsers._reduction import LogSumExpReduction
from lofi.nn.policy import place
from lofi.nn.sites import ComponentKind, SiteMap
from lofi.registry import ComponentPlugin
from lofi.spec.parsers.span_crf import SpanCrfParserSettings, SpanCrfParserSpec

if TYPE_CHECKING:
    from torch import Tensor

    from lofi.nn.build import BuildContext
    from lofi.nn.parsers._values import BracketedTree
    from lofi.nn.ports import SubwordPooler
    from lofi.nn.sites import ComponentId

__all__ = ["SpanCrfParser"]

_LEAF_AXES: Final = ("item", "word", "width")
_SPAN_AXES: Final = ("item", "span")
_SPLIT_AXES: Final = ("item", "span", "split")
_ITEM_AXES: Final = ("item",)


def _lengths(leaves: Encoded, /) -> Tensor:
    """Return prefix lengths after validating the pooled-word boundary."""
    if leaves.axes != _LEAF_AXES:
        msg = f"pooled leaves run over {_LEAF_AXES}, not {leaves.axes}"
        raise ValueError(msg)

    values = leaves.tensor

    if not values.is_floating_point():
        msg = f"pooled leaves are floating point, not {values.dtype}"
        raise TypeError(msg)

    mask = leaves.mask

    if mask is None:
        return values.new_full(
            (int(values.shape[0]),), int(values.shape[1]), dtype=torch.long
        )

    lengths = mask.sum(dim=1).to(torch.long)
    positions = torch.arange(int(mask.shape[1]), device=mask.device).unsqueeze(0)
    expected = positions < lengths.unsqueeze(1)

    if not bool(torch.equal(mask, expected)):
        msg = "pooled leaves must mask one suffix after each item's live prefix"
        raise ValueError(msg)

    if bool((lengths < 1).any()):
        msg = "a span CRF needs at least one live word in every item"
        raise ValueError(msg)

    return lengths


def _span_tables(index: SpanIndex, like: Tensor, /) -> tuple[Tensor, Tensor, Tensor]:
    """Return span starts, ends, and token counts in dense span order."""
    spans = tuple(index.spans_by_length())
    starts = like.new_tensor([start for start, _ in spans], dtype=torch.long)
    ends = like.new_tensor([end for _, end in spans], dtype=torch.long)
    return starts, ends, ends - starts


def _span_means(words: Tensor, index: SpanIndex, /) -> Tensor:
    """Return each contiguous span's mean pooled-word vector."""
    starts, ends, widths = _span_tables(index, words)
    prefix = torch.cat(
        (
            words.new_zeros((int(words.shape[0]), 1, int(words.shape[2]))),
            words.cumsum(dim=1),
        ),
        dim=1,
    )
    totals = prefix.index_select(1, ends) - prefix.index_select(1, starts)
    return totals / widths.to(words.dtype).reshape(1, -1, 1)


def _live_spans(index: SpanIndex, lengths: Tensor, /) -> Tensor:
    """Return the spans contained in each item's live prefix."""
    _, ends, _ = _span_tables(index, lengths)
    return ends.unsqueeze(0) <= lengths.unsqueeze(1)


def _mbr_indicators(
    trees: tuple[BracketedTree | None, ...],
    index: SpanIndex,
    live: Tensor,
    like: Tensor,
    /,
) -> Tensor:
    """Encode each decoded tree as one binary row over dense spans."""
    values = like.new_zeros((len(trees), index.num_spans))

    for item, tree in enumerate(trees):
        if tree is None:
            continue

        for start, end in tree.constituents:
            values[item, index.span_id(start, end)] = 1.0

    return values.masked_fill(~live, 0.0)


class SpanCrfParser(nn.Module):
    """Score a CRF over binary trees without constructing vector chart cells."""

    def __init__(
        self,
        *,
        input_width: int,
        hidden: int,
        component: ComponentId,
    ) -> None:
        super().__init__()

        if input_width < 1:
            msg = f"a pooled leaf width is positive, not {input_width}"
            raise ValueError(msg)
        if hidden < 1:
            msg = f"a span-CRF hidden width is positive, not {hidden}"
            raise ValueError(msg)

        self._leaf_score = nn.Linear(input_width, 1)
        self._left = nn.Linear(input_width, hidden, bias=False)
        self._right = nn.Linear(input_width, hidden, bias=False)
        self._split_score = nn.Linear(hidden, 1)
        self._input_width = input_width
        self._hidden = hidden
        self._component = component
        self._kernel = InsideKernel(LogSumExpReduction())
        self._outside = OutsideKernel(LogSumExpReduction())

    @property
    def publishes(self) -> frozenset[str]:
        """Return the globally normalized chart quantities this parser fills."""
        return frozenset(
            {"inside", "outside", "alpha", "mu", "log_z", "entropy", "mbr"}
        )

    @property
    def input_width(self) -> int:
        """Return the pooled-word width this parser consumes."""
        return self._input_width

    @property
    def hidden(self) -> int:
        """Return the hidden width of the split-potential scorer."""
        return self._hidden

    @property
    def kernel(self) -> InsideKernel:
        """Return the log-sum-exp inside kernel defining the CRF partition."""
        return self._kernel

    def potentials(
        self, leaves: Encoded, /
    ) -> tuple[Tensor, Tensor, SpanIndex, Tensor, Tensor]:
        """Return lexical and binary potentials with their addressing and masks."""
        lengths = _lengths(leaves)
        words = leaves.tensor

        if int(words.shape[-1]) != self._input_width:
            msg = (
                f"the parser consumes leaf width {self._input_width}, "
                f"not {int(words.shape[-1])}"
            )
            raise ValueError(msg)

        index = SpanIndex(int(words.shape[1]))
        live = _live_spans(index, lengths)
        leaf_live = (
            words.new_ones(words.shape[:2], dtype=torch.bool)
            if leaves.mask is None
            else leaves.mask
        )
        leaf_scores = self._leaf_score(words).squeeze(-1).masked_fill(~leaf_live, -inf)
        spans = _span_means(words, index)

        if index.max_splits == 0:
            split_scores = words.new_empty((int(words.shape[0]), index.num_spans, 0))
        else:
            left_ids = words.new_tensor(index.left_child_ids, dtype=torch.long)
            right_ids = words.new_tensor(index.right_child_ids, dtype=torch.long)
            left = spans.index_select(1, left_ids.reshape(-1)).reshape(
                int(words.shape[0]), index.num_spans, index.max_splits, -1
            )
            right = spans.index_select(1, right_ids.reshape(-1)).reshape_as(left)
            split_scores = self._split_score(
                torch.tanh(self._left(left) + self._right(right))
            ).squeeze(-1)
            valid = words.new_tensor(index.split_mask, dtype=torch.bool).unsqueeze(0)
            valid = valid & live.unsqueeze(-1)
            split_scores = split_scores.masked_fill(~valid, -inf)

        return leaf_scores, split_scores, index, lengths, live

    def readouts(self, leaves: Encoded, /) -> ChartReadouts:
        """Return inside/outside values, posterior marginals, entropy, and MBR trees."""
        leaf_scores, split_scores, index, lengths, live = self.potentials(leaves)
        inside = self._kernel.inside(leaf_scores, split_scores, index)
        outside = self._outside.outside(inside, split_scores, index, lengths=lengths)
        alpha = log_split_posterior_by_gradient(
            leaf_scores, split_scores, index, lengths=lengths
        )
        log_mu = log_constituent_marginals(inside, outside, index, lengths=lengths)
        log_z = self._kernel.log_partition(
            leaf_scores, split_scores, index, lengths=lengths
        )
        entropy = derivation_entropy(
            inside, outside, split_scores, index, lengths=lengths
        )
        trees = mbr_trees(inside, outside, split_scores, index, lengths=lengths)
        valid_splits = split_scores.new_tensor(
            index.split_mask, dtype=torch.bool
        ).unsqueeze(0) & live.unsqueeze(-1)

        return ChartReadouts(
            inside=LogDistribution(inside.masked_fill(~live, -inf), _SPAN_AXES, live),
            outside=LogDistribution(outside.masked_fill(~live, -inf), _SPAN_AXES, live),
            alpha=LogDistribution(
                alpha.masked_fill(~valid_splits, -inf),
                _SPLIT_AXES,
                valid_splits,
                normalized_over="split",
            ),
            mu=Encoded(log_mu.exp().masked_fill(~live, 0.0), _SPAN_AXES, live),
            log_z=LogDistribution(log_z, _ITEM_AXES),
            entropy=Encoded(entropy, _ITEM_AXES),
            mbr=Encoded(_mbr_indicators(trees, index, live, inside), _SPAN_AXES, live),
        )

    def site_map(self) -> SiteMap:
        """Return no intervention points for the derived scalar analysis."""
        return SiteMap(self._component, ComponentKind.PARSER, {})

    @override
    def extra_repr(self) -> str:
        """Return the derived input width, hidden width, and component handle."""
        return (
            f"input_width={self._input_width}, hidden={self._hidden}, "
            f"component={self._component!r}"
        )


def _build(settings: SpanCrfParserSettings, context: BuildContext, /) -> SpanCrfParser:
    """Build and place the scalar-potential chart declared by a run."""
    pooler = cast("SubwordPooler", context.producer("leaves"))
    return place(
        SpanCrfParser(
            input_width=pooler.output_width,
            hidden=settings.hidden,
            component=context.component,
        ),
        context.policy,
    )


PLUGIN: Final = ComponentPlugin(spec=SpanCrfParserSpec, build=_build)
"""The span-CRF chart plugin."""
