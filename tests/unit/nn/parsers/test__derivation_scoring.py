"""Tests for scoring gold derivations against tagged chart channels."""

from math import log

import pytest
import torch
from tests.unit.nn.parsers.test__tagged import (
    CATEGORIES,
    FORWARD,
    MATCHING,
    RULES,
    NaryRule,
    ruleset,
)

from lofi.data import Derivation, DerivationNode, TokenSpan
from lofi.deduce import SpanIndex
from lofi.judge import EisnerNormalForm, NoNormalForm
from lofi.nn.parsers import LogSumExpReduction, ScoreChannels, TaggedInsideKernel
from lofi.nn.parsers._derivation_scoring import TaggedDerivationScoreChart
from lofi.terms import Term


def _binary_chart() -> TaggedDerivationScoreChart[Term]:
    """Return a zero-weight chart with one ambiguous composition rule."""
    index = SpanIndex(3)
    rule_set = ruleset(EisnerNormalForm[Term](), (RULES[1],))
    constrained = TaggedInsideKernel(
        LogSumExpReduction(), rule_set, CATEGORIES, unary_depth=0
    )
    unconstrained = TaggedInsideKernel(
        LogSumExpReduction(),
        rule_set.with_normal_form(NoNormalForm[Term]()),
        CATEGORIES,
        unary_depth=0,
    )
    channels = ScoreChannels(
        torch.zeros(1, 3, 1, dtype=torch.float64),
        torch.zeros(1, index.num_spans, index.max_splits, 1, dtype=torch.float64),
    )

    return TaggedDerivationScoreChart(
        channels,
        index,
        rule_set,
        CATEGORIES,
        CATEGORIES,
        constrained,
        unconstrained,
        constrained.log_partition(channels, index, goals=CATEGORIES),
        lengths=None,
    )


def _right_branching(*, reversed_root: bool = False) -> Derivation[Term]:
    """Return the Eisner-admitted derivation over three equal categories."""
    category = CATEGORIES[0]
    root_children = (3, 0) if reversed_root else (0, 3)

    return Derivation(
        4,
        {
            0: DerivationNode(TokenSpan(0, 1), category),
            1: DerivationNode(TokenSpan(1, 2), category),
            2: DerivationNode(TokenSpan(2, 3), category),
            3: DerivationNode(
                TokenSpan(1, 3),
                category,
                rule=RULES[1].name,
                children=(1, 2),
            ),
            4: DerivationNode(
                TokenSpan(0, 3),
                category,
                rule=RULES[1].name,
                children=root_children,
            ),
        },
    )


def test_spurious_ambiguity_changes_only_the_normalizing_derivation_set() -> None:
    scored = _binary_chart().score(0, _right_branching())

    assert scored.constrained.normal_form == "eisner"
    assert scored.constrained.score == pytest.approx(0.0)
    assert scored.constrained.log_partition == pytest.approx(0.0)
    assert scored.unconstrained.normal_form == "none"
    assert scored.unconstrained.score == pytest.approx(0.0)
    assert scored.unconstrained.log_partition == pytest.approx(log(2.0))


def test_children_must_tile_the_parent_in_recorded_order() -> None:
    with pytest.raises(ValueError, match="not an ordered tiling"):
        _binary_chart().score(0, _right_branching(reversed_root=True))


def test_an_nary_rule_takes_its_score_from_the_final_prefix_edge() -> None:
    index = SpanIndex(3)
    rule = NaryRule("combine-three", MATCHING, arity=3, slash=FORWARD, order=0)
    rule_set = ruleset(EisnerNormalForm[Term](), (rule,))
    constrained = TaggedInsideKernel(
        LogSumExpReduction(), rule_set, CATEGORIES, unary_depth=0
    )
    unconstrained = TaggedInsideKernel(
        LogSumExpReduction(),
        rule_set.with_normal_form(NoNormalForm[Term]()),
        CATEGORIES,
        unary_depth=0,
    )
    steps = torch.zeros(1, index.num_spans, index.max_splits, 1, dtype=torch.float64)
    _, final_offset = index.split_id(0, 3, 2)
    steps[0, index.span_id(0, 3), final_offset, 0] = 7.0
    channels = ScoreChannels(
        torch.tensor([[[1.0], [2.0], [3.0]]], dtype=torch.float64), steps
    )
    chart = TaggedDerivationScoreChart(
        channels,
        index,
        rule_set,
        CATEGORIES,
        CATEGORIES,
        constrained,
        unconstrained,
        constrained.log_partition(channels, index, goals=CATEGORIES),
        lengths=None,
    )
    category = CATEGORIES[0]
    gold = Derivation(
        3,
        {
            0: DerivationNode(TokenSpan(0, 1), category),
            1: DerivationNode(TokenSpan(1, 2), category),
            2: DerivationNode(TokenSpan(2, 3), category),
            3: DerivationNode(
                TokenSpan(0, 3), category, rule=rule.name, children=(0, 1, 2)
            ),
        },
    )

    assert chart.score(0, gold).constrained.score == pytest.approx(13.0)
