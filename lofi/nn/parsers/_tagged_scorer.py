"""Configured normal-form-refined neural chart scorer."""

from math import inf
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, cast, override

import torch
from torch import nn

from lofi.deduce import SpanIndex
from lofi.judge import NoNormalForm
from lofi.nn._values import ChartReadouts, Encoded, LogDistribution
from lofi.nn.parsers._derivation_scoring import TaggedDerivationScoreChart
from lofi.nn.parsers._ports import LogWeightReduction
from lofi.nn.parsers._reduction import LogSumExpReduction, MaximumReduction
from lofi.nn.parsers._scorer import DeepInsideOutside
from lofi.nn.parsers._tagged import ScoreChannels, TaggedInsideKernel, span_marginals
from lofi.nn.parsers._tagged_decode import TaggedDecoder
from lofi.nn.parsers._weigher import SplitWeigher, build_split_weigher
from lofi.nn.parsers.composition import (
    ChartCompositionModule,
    action_arities,
    build_chart_composition,
)
from lofi.nn.parsers.functions import SpanFunctionModule, build_span_function
from lofi.nn.policy import place
from lofi.nn.psi import RuleScorer
from lofi.nn.sites import (
    Access,
    ComponentKind,
    DTypeClass,
    Role,
    Site,
    SiteMap,
    SiteSpec,
)
from lofi.registry import ComponentPlugin
from lofi.spec.parsers.reduction import ReductionResolver
from lofi.spec.parsers.tagged import TaggedParserSettings, TaggedParserSpec

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping

    from torch import Tensor

    from lofi._types import SharedSubmodule
    from lofi.judge import ActionSet, RuleSet
    from lofi.nn._values import DerivationScoreChart
    from lofi.nn.build import BuildContext
    from lofi.nn.parsers._values import TaggedTree
    from lofi.nn.ports import Controller, SubwordPooler
    from lofi.nn.sites import ComponentId
    from lofi.spec.parsers.chart import OutsideWeights, RootOutside
    from lofi.terms import Term

__all__ = ["TaggedChartScorer"]

_LEAF_AXES: Final = ("item", "word", "width")
_SPAN_AXES: Final = ("item", "span")
_SPLIT_AXES: Final = ("item", "span", "split")
_CELL_AXES: Final = ("item", "span", "width")
_SPAN_VECTOR_AXES: Final = ("item", "span", "span_width")
_ITEM_AXES: Final = ("item",)

_SITE_FIELDS: Final = MappingProxyType(
    {
        "inside": (Role.INSIDE_SCORE, _SPAN_AXES, DTypeClass.REAL),
        "alpha": (Role.SPLIT_DISTRIBUTION, _SPLIT_AXES, DTypeClass.LOG_PROBABILITY),
        "mu": (Role.CONSTITUENT_MARGINAL, _SPAN_AXES, DTypeClass.PROBABILITY),
        "log_z": (Role.LOG_PARTITION, _ITEM_AXES, DTypeClass.REAL),
        "identity": (Role.IDENTITY_SCORE, _SPLIT_AXES, DTypeClass.REAL),
        "cells": (Role.CHART_CELL, _CELL_AXES, DTypeClass.REAL),
        "outside_cells": (Role.OUTSIDE_CELL, _CELL_AXES, DTypeClass.REAL),
        "spans": (Role.SPAN_REPRESENTATION, _SPAN_VECTOR_AXES, DTypeClass.REAL),
    }
)


def _lengths(leaves: Encoded, /) -> Tensor | None:
    """Return each item's word count, or None when the batch is rectangular."""
    mask = leaves.mask

    return None if mask is None else mask.reshape(mask.shape[0], -1).sum(dim=-1)


def _live_spans(lengths: Tensor | None, like: Tensor, index: SpanIndex, /) -> Tensor:
    """Return which indexed spans fall within each item's word count."""
    if lengths is None:
        return like.new_ones(like.shape, dtype=torch.bool)

    ends = like.new_tensor(
        [end for _, end in index.spans_by_length()], dtype=torch.long
    )

    return ends.unsqueeze(0) <= lengths.to(torch.long).unsqueeze(-1)


def _site_map(component: ComponentId, published: frozenset[str], /) -> SiteMap:
    """Return the observable readouts the tagged scorer actually fills."""
    sites = {
        Site(component, role, (name,)): SiteSpec(
            axes,
            dtype_class,
            differentiable=True,
            access=Access.OBSERVE,
        )
        for name, (role, axes, dtype_class) in _SITE_FIELDS.items()
        if name in published
    }

    return SiteMap(component, ComponentKind.PARSER, sites)


def _action_columns(
    rule_set: RuleSet[Term], actions: ActionSet, /
) -> tuple[tuple[int, ...], ...]:
    """Return all controller columns of each combining rule in kernel order."""
    inventory = actions.actions
    columns = []

    for rule in rule_set:
        if rule.arity < 2:
            continue

        found = tuple(
            column
            for column, action in enumerate(inventory)
            if action.rule == rule.name
        )

        if not found:
            msg = f"combining rule {rule.name!r} has no controller action"
            raise ValueError(msg)

        columns.append(found)

    return tuple(columns)


class TaggedChartScorer(nn.Module):
    """Score vector cells and a category- and normal-form-refined chart together."""

    def __init__(
        self,
        rules: RuleScorer,
        rule_set: RuleSet[Term],
        categories: tuple[Term, ...],
        goals: tuple[Term, ...],
        *,
        composition: ChartCompositionModule,
        split_left: SpanFunctionModule,
        split_right: SpanFunctionModule,
        outside_weights: OutsideWeights,
        root_outside: RootOutside,
        root_init_gain: float,
        component: ComponentId,
        reduction: LogWeightReduction,
        weigher: SplitWeigher,
        unary_depth: int,
        k_best: int,
        checkpoint_from_width: int | None,
    ) -> None:
        super().__init__()
        self._base = DeepInsideOutside(
            rules,
            composition=composition,
            split_left=split_left,
            split_right=split_right,
            outside_weights=outside_weights,
            root_outside=root_outside,
            root_init_gain=root_init_gain,
            component=component,
            checkpoint_from_width=checkpoint_from_width,
            reduction=reduction,
            weigher=weigher,
        )
        self._kernel = TaggedInsideKernel(
            reduction, rule_set, categories, unary_depth=unary_depth
        )
        self._unrestricted_kernel = (
            TaggedInsideKernel(
                reduction,
                rule_set.with_normal_form(NoNormalForm()),
                categories,
                unary_depth=unary_depth,
            )
            if reduction.probabilistic
            else None
        )
        self._decoder = TaggedDecoder(self._kernel, k=k_best)
        self._lexical = nn.Linear(composition.input_width, len(categories))
        unary_rules = self._kernel.transitions.unary_rules
        self._unary = (
            None if not unary_rules else nn.Linear(rules.input_width, len(unary_rules))
        )
        self._rule_columns = _action_columns(rule_set, rules.action_set)
        self._goals = tuple(goals)
        self._component = component
        self._sites = _site_map(component, self.publishes)

    @property
    def publishes(self) -> frozenset[str]:
        """Return exactly the tagged readouts this scorer fills."""
        always = frozenset({"inside", "log_z", "identity", "cells", "actions"})

        if self.reduction.probabilistic:
            outside = frozenset({"alpha", "mu", "spans", "derivations"})

            if self._base.contextual_outside:
                outside |= frozenset({"outside_cells"})

            return always | outside

        return always

    @property
    def input_width(self) -> int:
        """Return the width of each pooled leaf vector."""
        return self._base.input_width

    @property
    def output_width(self) -> int:
        """Return the width of each chart cell."""
        return self._base.output_width

    @property
    def span_width(self) -> int:
        """Return the width of the concatenated inside-outside representation."""
        return self._base.span_width

    @property
    def rules(self) -> RuleScorer:
        """Return what weighs the controller actions."""
        return self._base.rules

    @property
    def reduction(self) -> LogWeightReduction:
        """Return how alternative tagged derivations are combined."""
        return self._kernel.reduction

    @property
    def kernel(self) -> TaggedInsideKernel[Term]:
        """Return the configured tagged inside kernel."""
        return self._kernel

    @property
    def decoder(self) -> TaggedDecoder[Term]:
        """Return the configured k-best decoder."""
        return self._decoder

    @property
    def shared(self) -> Mapping[str, SharedSubmodule[nn.Module]]:
        """Return the base chart's cell combination publication."""
        return self._base.shared

    def _channels(
        self, leaves: Encoded, /
    ) -> tuple[ChartReadouts, ScoreChannels, SpanIndex, Tensor | None, Tensor]:
        """Return base readouts and the tagged channels derived from them."""
        if leaves.axes != _LEAF_AXES:
            msg = f"pooled leaves run over {_LEAF_AXES}, not over {leaves.axes}"
            raise ValueError(msg)

        base = self._base.readouts(leaves)
        actions = base.actions
        cells = base.cells

        if actions is None or cells is None:
            msg = "a tagged chart requires its vector chart and controller actions"
            raise RuntimeError(msg)

        index = SpanIndex(leaves.size("word"))
        lengths = _lengths(leaves)
        live = _live_spans(
            lengths,
            cells.tensor.new_empty(cells.tensor.shape[:2]),
            index,
        )
        lexical = self._lexical(leaves.tensor)
        leaf_mask = leaves.mask

        if leaf_mask is not None:
            lexical = lexical.masked_fill(~leaf_mask.unsqueeze(-1), -inf)

        action_scores = actions.scores.tensor
        combining = torch.stack(
            tuple(
                self.reduction.reduce(
                    action_scores.index_select(
                        -1,
                        action_scores.new_tensor(columns, dtype=torch.long),
                    ),
                    action_scores.new_ones(
                        (*action_scores.shape[:-1], len(columns)), dtype=torch.bool
                    ),
                    dim=-1,
                )
                for columns in self._rule_columns
            ),
            dim=-1,
        )
        unary = None if self._unary is None else self._unary(cells.tensor)

        if unary is not None:
            unary = unary.masked_fill(~live.unsqueeze(-1), -inf)

        return base, ScoreChannels(lexical, combining, unary), index, lengths, live

    def readouts(self, leaves: Encoded, /) -> ChartReadouts:
        """Return vector, action, and tagged chart readouts for one batch."""
        base, channels, index, lengths, live = self._channels(leaves)
        tagged = self._kernel.inside(channels, index).inside
        state_mask = torch.ones_like(tagged, dtype=torch.bool)
        by_category = self.reduction.reduce(tagged, state_mask, dim=-1)
        inside = self.reduction.reduce(
            by_category, torch.ones_like(by_category, dtype=torch.bool), dim=-1
        )
        log_z = self._kernel.log_partition(
            channels, index, goals=self._goals, lengths=lengths
        )
        alpha: LogDistribution | None = None
        mu: Encoded | None = None
        derivations = None

        if self.reduction.probabilistic:
            counts = self._kernel.expected_counts(
                channels, index, goals=self._goals, lengths=lengths
            )
            constituent = span_marginals(counts, index)
            split = counts.rule.sum(dim=-1)
            denominator = split.sum(dim=-1, keepdim=True)
            positive = split > 0.0
            tiny = torch.finfo(split.dtype).tiny
            conditional = (
                split.clamp_min(tiny).log() - denominator.clamp_min(tiny).log()
            ).masked_fill(~positive, -inf)
            real = inside.new_tensor(index.split_mask, dtype=torch.bool).unsqueeze(0)
            live_splits = live.unsqueeze(-1) & real & (denominator > 0.0)
            alpha = LogDistribution(
                conditional.masked_fill(~live_splits, -inf),
                _SPLIT_AXES,
                live_splits,
                normalized_over="split",
            )
            mu = Encoded(constituent.masked_fill(~live, 0.0), _SPAN_AXES, live)
            unrestricted = self._unrestricted_kernel

            if unrestricted is None:
                msg = "a probabilistic tagged chart has no unrestricted kernel"
                raise RuntimeError(msg)

            derivations = TaggedDerivationScoreChart(
                channels,
                index,
                self._kernel.transitions.rule_set,
                self._kernel.transitions.categories,
                self._goals,
                self._kernel,
                unrestricted,
                log_z,
                lengths=lengths,
            )

        return ChartReadouts(
            inside=LogDistribution(inside.masked_fill(~live, -inf), _SPAN_AXES, live),
            alpha=alpha,
            mu=mu,
            log_z=LogDistribution(log_z, _ITEM_AXES),
            actions=base.actions,
            cells=base.cells,
            outside_cells=base.outside_cells,
            spans=base.spans,
            derivations=cast("DerivationScoreChart[Hashable] | None", derivations),
        )

    def k_best(self, leaves: Encoded, /) -> tuple[tuple[TaggedTree, ...], ...]:
        """Return each item's configured number of best tagged derivations."""
        _, channels, index, lengths, _ = self._channels(leaves)

        return self._decoder.k_best(channels, index, goals=self._goals, lengths=lengths)

    def site_map(self) -> SiteMap:
        """Return the observable readouts of this tagged scorer."""
        return self._sites

    @override
    def extra_repr(self) -> str:
        """Return the category, unary-depth, and k-best dimensions."""
        transitions = self._kernel.transitions

        return (
            f"categories={len(transitions.categories)}, "
            f"unary_depth={transitions.closure.depth}, k_best={self._decoder.k}"
        )


class _ReductionBuilder(ReductionResolver[LogWeightReduction]):
    """Build a declared tensor reduction for the tagged kernel."""

    @override
    def log_sum_exp(self) -> LogWeightReduction:
        """Return probability-summing reduction."""
        return LogSumExpReduction()

    @override
    def maximum(self) -> LogWeightReduction:
        """Return best-alternative reduction."""
        return MaximumReduction()


_REDUCTIONS: Final = _ReductionBuilder()


def _build(
    settings: TaggedParserSettings, context: BuildContext, /
) -> TaggedChartScorer:
    """Build and place a normal-form-refined chart scorer."""
    controller = cast("Controller", context.producer("rules"))
    pooler = cast("SubwordPooler", context.producer("leaves"))

    if controller.action_set is not context.actions:
        msg = "the tagged parser needs the run's exact controller action inventory"
        raise ValueError(msg)

    goals = context.targets["grammar-goal"].categories
    arities = action_arities(context.actions, context.rules)
    unsupported = tuple(arity for arity in arities if arity < 1)

    if unsupported:
        arity = unsupported[0]
        msg = (
            f"chart composition {settings.composition.kind!r} cannot serve rule "
            f"arity {arity}; the tagged chart requires positive-arity rules"
        )
        raise ValueError(msg)

    width = settings.width

    if controller.input_width != width:
        msg = (
            f"the controller reads width {controller.input_width}, "
            f"not the tagged chart's declared node width {width}"
        )
        raise ValueError(msg)

    split_left = build_span_function(settings.split_left, (width, width), width)
    split_right = (
        split_left
        if settings.share_splits
        else build_span_function(settings.split_right, (width, width), width)
    )

    return place(
        TaggedChartScorer(
            RuleScorer(controller, width=width),
            context.rules,
            context.categories,
            goals,
            composition=build_chart_composition(
                settings.composition, pooler.output_width, width
            ),
            split_left=split_left,
            split_right=split_right,
            outside_weights=settings.outside_weights,
            root_outside=settings.root_outside,
            root_init_gain=settings.root_init_gain,
            component=context.component,
            reduction=settings.reduction.resolve(_REDUCTIONS),
            weigher=build_split_weigher(settings.attend, width),
            unary_depth=settings.unary_depth,
            k_best=settings.k_best,
            checkpoint_from_width=context.policy.checkpoint_from_width,
        ),
        context.policy,
    )


PLUGIN: Final = ComponentPlugin(spec=TaggedParserSpec, build=_build)
"""The normal-form-refined neural chart plugin."""
