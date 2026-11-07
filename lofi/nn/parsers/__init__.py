"""Differentiable chart parsing utilities."""

from lofi.nn.parsers._channel import (
    Composition,
    SplitChannel,
    SplitScoring,
    log_split_posterior,
)
from lofi.nn.parsers._chart import TensorChart
from lofi.nn.parsers._closure import UnaryClosure, UnaryTransition
from lofi.nn.parsers._decode import k_best, mbr_trees
from lofi.nn.parsers._inside import ChildGathers, InsideKernel
from lofi.nn.parsers._marginals import (
    derivation_entropy,
    expected_rule_loss,
    log_constituent_marginals,
    log_rule_marginals,
    log_split_posterior_by_gradient,
    rule_marginals_by_gradient,
)
from lofi.nn.parsers._outside import (
    OutsideKernel,
    ParentContexts,
    VectorOutsideKernel,
    log_parent_posterior,
    parent_contexts,
)
from lofi.nn.parsers._ports import LogWeightReduction
from lofi.nn.parsers._reduction import LogSumExpReduction, MaximumReduction
from lofi.nn.parsers._released_diora import ReleasedDioraMLP
from lofi.nn.parsers._scorer import DeepInsideOutside
from lofi.nn.parsers._span_crf import SpanCrfParser
from lofi.nn.parsers._stacked import SpanTransition, StackedInsideOutside
from lofi.nn.parsers._tagged import (
    NormalFormTransitions,
    ScoreChannels,
    TaggedInsideChart,
    TaggedInsideKernel,
    TaggedRuleScores,
    TagTransition,
    span_marginals,
)
from lofi.nn.parsers._tagged_decode import TaggedDecoder
from lofi.nn.parsers._tagged_scorer import TaggedChartScorer
from lofi.nn.parsers._values import (
    BracketedTree,
    ChanneledChart,
    ScoredChart,
    TaggedTree,
)
from lofi.nn.parsers._weigher import (
    AttendSplitWeigher,
    MarginalSplitWeigher,
    SplitWeigher,
)
from lofi.nn.parsers.composition import (
    ChartComposition,
    ChartCompositionModule,
    GatedResidualComposition,
    MeanFieldComposition,
    action_arities,
    build_chart_composition,
)
from lofi.nn.parsers.flat import FlatSpanEncoder
from lofi.nn.parsers.functions import (
    BilinearSpanFunction,
    CpBilinearSpanFunction,
    GatedMlpSpanFunction,
    HrrSpanFunction,
    MlpSpanFunction,
    SpanFunction,
    SpanFunctionModule,
    TransformerSpanFunction,
    build_span_function,
)

__all__ = [
    "AttendSplitWeigher",
    "BilinearSpanFunction",
    "BracketedTree",
    "ChanneledChart",
    "ChartComposition",
    "ChartCompositionModule",
    "ChildGathers",
    "Composition",
    "CpBilinearSpanFunction",
    "DeepInsideOutside",
    "FlatSpanEncoder",
    "GatedMlpSpanFunction",
    "GatedResidualComposition",
    "HrrSpanFunction",
    "InsideKernel",
    "LogSumExpReduction",
    "LogWeightReduction",
    "MarginalSplitWeigher",
    "MaximumReduction",
    "MeanFieldComposition",
    "MlpSpanFunction",
    "NormalFormTransitions",
    "OutsideKernel",
    "ParentContexts",
    "ReleasedDioraMLP",
    "ScoreChannels",
    "ScoredChart",
    "SpanCrfParser",
    "SpanFunction",
    "SpanFunctionModule",
    "SpanTransition",
    "SplitChannel",
    "SplitScoring",
    "SplitWeigher",
    "StackedInsideOutside",
    "TagTransition",
    "TaggedChartScorer",
    "TaggedDecoder",
    "TaggedInsideChart",
    "TaggedInsideKernel",
    "TaggedRuleScores",
    "TaggedTree",
    "TensorChart",
    "TransformerSpanFunction",
    "UnaryClosure",
    "UnaryTransition",
    "VectorOutsideKernel",
    "action_arities",
    "build_chart_composition",
    "build_span_function",
    "derivation_entropy",
    "expected_rule_loss",
    "k_best",
    "log_constituent_marginals",
    "log_parent_posterior",
    "log_rule_marginals",
    "log_split_posterior",
    "log_split_posterior_by_gradient",
    "mbr_trees",
    "parent_contexts",
    "rule_marginals_by_gradient",
    "span_marginals",
]
