"""Tests for registered neural chart-composition families."""

from typing import Final

import pytest
import torch
from tests._controllers import mlp_controller
from tests.conformance.component_plugin import plugin_context
from tests.unit.nn.parsers.test__tagged import MATCHING, NaryRule, ruleset

from lofi.judge import ActionSet, NoNormalForm
from lofi.nn.parsers import _scorer as scorer_module
from lofi.nn.parsers.composition import (
    GatedResidualComposition,
    MeanFieldComposition,
    action_arities,
    build_chart_composition,
)
from lofi.nn.parsers.functions import MlpSpanFunction
from lofi.nn.sites import ComponentId
from lofi.nn.tokens.pool import FirstSubwordPooler
from lofi.spec.chart_compositions import (
    GatedResidualCompositionSpec,
    MeanFieldCompositionSpec,
)
from lofi.spec.parsers.chart import InsideOutsideParserSpec
from lofi.terms import Term

INPUT: Final = 3
WIDTH: Final = 5


def function(inputs: tuple[int, ...]) -> MlpSpanFunction:
    """Return one test MLP with the chart's node width."""
    return MlpSpanFunction(inputs, WIDTH, layers=1, nonlinearity="tanh")


def test_mean_field_projects_leaves_and_composes_pairs() -> None:
    composition = MeanFieldComposition(
        function((INPUT,)), function((WIDTH, WIDTH)), width=WIDTH
    )
    leaves = torch.randn(2, 4, INPUT)
    left = torch.randn(8, WIDTH)
    right = torch.randn(8, WIDTH)

    assert composition.leaf(leaves).shape == (2, 4, WIDTH)
    assert composition.compose(left, right).shape == (8, WIDTH)
    assert composition.input_width == INPUT
    assert composition.output_width == WIDTH


def test_gated_residual_uses_the_declared_gate_to_mix_the_child_mean() -> None:
    leaf = function((INPUT,))
    candidate = function((WIDTH, WIDTH))
    gate = function((WIDTH, WIDTH))
    composition = GatedResidualComposition(leaf, candidate, gate, width=WIDTH).to(
        torch.float64
    )
    left = torch.randn(6, WIDTH, dtype=torch.float64)
    right = torch.randn(6, WIDTH, dtype=torch.float64)
    mixed = composition.compose(left, right)
    probability = torch.sigmoid(gate(left, right))
    expected = (
        probability * candidate(left, right)
        + (1.0 - probability) * (left + right) * 0.5
    )

    assert torch.allclose(mixed, expected)

    (
        mixed.sum() + composition.leaf(torch.randn(6, INPUT, dtype=torch.float64)).sum()
    ).backward()
    assert all(parameter.grad is not None for parameter in composition.parameters())


def test_both_registered_compositions_build_at_the_declared_width() -> None:
    for declared, expected in (
        (MeanFieldCompositionSpec(), MeanFieldComposition),
        (GatedResidualCompositionSpec(), GatedResidualComposition),
    ):
        built = build_chart_composition(declared.declaration(), INPUT, WIDTH)

        assert type(built) is expected
        assert built.input_width == INPUT
        assert built.output_width == WIDTH


def test_rule_arity_is_derived_from_the_rules_named_by_the_action_set() -> None:
    rule = NaryRule("combine-three", MATCHING, arity=3, slash="/", order=0)
    actions = ActionSet.derive((rule.name,), identity="identity", null=None)

    assert action_arities(actions, ruleset(NoNormalForm[Term](), (rule,))) == (3,)


def test_the_pair_chart_refuses_a_three_premise_action_by_arity() -> None:
    rule = NaryRule("combine-three", MATCHING, arity=3, slash="/", order=0)
    actions = ActionSet.derive((rule.name,), identity="identity", null=None)
    controller = mlp_controller(
        actions, width=WIDTH, component=ComponentId("controller"), hidden=7
    )
    context = plugin_context(
        actions=actions,
        rules=ruleset(NoNormalForm[Term](), (rule,)),
        producers={
            "rules": controller,
            "leaves": FirstSubwordPooler(INPUT, component=ComponentId("pooler")),
        },
    )

    with pytest.raises(ValueError, match="rule arity 3"):
        scorer_module.PLUGIN.construct(InsideOutsideParserSpec(width=WIDTH), context)
