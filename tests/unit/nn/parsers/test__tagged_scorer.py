"""Tests for the configured normal-form-refined neural chart scorer."""

from typing import Final, Literal

import torch
from tests._controllers import mlp_controller
from tests.conformance.component_plugin import plugin_context
from tests.unit.nn.parsers.test__tagged import (
    CATEGORIES,
    MATCHING,
    RULES,
    NaryRule,
    ruleset,
)

from lofi.judge import ActionSet, NoNormalForm
from lofi.nn._values import Encoded
from lofi.nn.parsers import (
    LogSumExpReduction,
    LogWeightReduction,
    MarginalSplitWeigher,
    MaximumReduction,
    MlpSpanFunction,
    TaggedChartScorer,
)
from lofi.nn.parsers import _tagged_scorer as tagged_scorer_module
from lofi.nn.parsers.composition import MeanFieldComposition
from lofi.nn.psi import RuleScorer
from lofi.nn.sites import ComponentId
from lofi.nn.tokens.pool import FirstSubwordPooler
from lofi.spec.parsers.tagged import TaggedParserSpec
from lofi.terms import Identity, TargetSet, Term

INPUT_WIDTH: Final = 3
CELL_WIDTH: Final = 4
HIDDEN: Final = 5
CHART: Final = ComponentId("chart")
CONTROLLER: Final = ComponentId("controller")


def build(
    reduction: LogWeightReduction | None = None,
    *,
    outside_weights: Literal["reuse", "contextual"] = "contextual",
) -> TaggedChartScorer:
    """Return a tagged scorer over the shared toy grammar."""
    actions = ActionSet.derive(
        (rule.name for rule in RULES), identity="identity", null=None
    )
    controller = mlp_controller(
        actions, width=CELL_WIDTH, component=CONTROLLER, hidden=HIDDEN
    )

    return TaggedChartScorer(
        RuleScorer(controller, width=CELL_WIDTH),
        ruleset(NoNormalForm[Term]()),
        CATEGORIES,
        CATEGORIES,
        composition=MeanFieldComposition(
            MlpSpanFunction((INPUT_WIDTH,), CELL_WIDTH, layers=1, nonlinearity="tanh"),
            MlpSpanFunction(
                (CELL_WIDTH, CELL_WIDTH),
                CELL_WIDTH,
                layers=1,
                nonlinearity="tanh",
            ),
            width=CELL_WIDTH,
        ),
        split_left=MlpSpanFunction(
            (CELL_WIDTH, CELL_WIDTH),
            CELL_WIDTH,
            layers=1,
            nonlinearity="tanh",
        ),
        split_right=MlpSpanFunction(
            (CELL_WIDTH, CELL_WIDTH),
            CELL_WIDTH,
            layers=1,
            nonlinearity="tanh",
        ),
        outside_weights=outside_weights,
        root_outside="learned",
        root_init_gain=1.0,
        component=CHART,
        reduction=LogSumExpReduction() if reduction is None else reduction,
        weigher=MarginalSplitWeigher(),
        unary_depth=1,
        k_best=2,
        checkpoint_from_width=None,
    )


def test_reused_outside_vectors_are_not_published_independently() -> None:
    scorer = build(outside_weights="reuse")
    readouts = scorer.readouts(leaves())

    assert "outside_cells" not in scorer.publishes
    assert readouts.outside_cells is None
    assert readouts.spans is not None


def leaves(*, dtype: torch.dtype = torch.float32) -> Encoded:
    """Return a small rectangular input batch."""
    generator = torch.Generator().manual_seed(71)

    return Encoded(
        torch.randn(2, 3, INPUT_WIDTH, generator=generator, dtype=dtype),
        ("item", "word", "width"),
    )


def test_the_scorer_builds_the_declared_tagged_kernel_and_decoder() -> None:
    scorer = build()

    assert scorer.kernel.transitions.closure.depth == 1
    assert scorer.decoder.k == 2
    assert scorer.output_width == CELL_WIDTH
    assert scorer.span_width == 2 * CELL_WIDTH
    assert scorer.publishes == {
        "actions",
        "alpha",
        "cells",
        "derivations",
        "identity",
        "inside",
        "log_z",
        "mu",
        "outside_cells",
        "spans",
    }


def test_the_tagged_chart_serves_a_three_premise_action_by_binarization() -> None:
    rule = NaryRule("combine-three", MATCHING, arity=3, slash="/", order=0)
    actions = ActionSet.derive((rule.name,), identity="identity", null=None)
    controller = mlp_controller(
        actions, width=CELL_WIDTH, component=CONTROLLER, hidden=HIDDEN
    )
    target = TargetSet(CATEGORIES, Identity[Term]())
    context = plugin_context(
        actions=actions,
        rules=ruleset(NoNormalForm[Term](), (rule,)),
        categories=CATEGORIES,
        targets={"grammar-goal": target},
        producers={
            "rules": controller,
            "leaves": FirstSubwordPooler(INPUT_WIDTH, component=ComponentId("pooler")),
        },
    )
    scorer = tagged_scorer_module.PLUGIN.construct(
        TaggedParserSpec(width=CELL_WIDTH, unary_depth=0), context
    )
    readouts = scorer.readouts(
        Encoded(torch.randn(1, 3, INPUT_WIDTH), ("item", "word", "width"))
    )

    assert readouts.log_z is not None
    assert bool(torch.isfinite(readouts.log_z.tensor).all())


def test_log_sum_exp_publishes_probabilistic_tagged_readouts() -> None:
    scorer = build().to(torch.float64)
    words = leaves(dtype=torch.float64)
    readouts = scorer.readouts(words)

    assert readouts.inside is not None
    assert readouts.log_z is not None
    assert readouts.alpha is not None
    assert readouts.mu is not None
    assert readouts.outside is None
    assert readouts.actions is not None
    assert readouts.cells is not None
    assert readouts.derivations is not None
    assert readouts.derivations.normal_form == "none"
    assert bool(torch.isfinite(readouts.log_z.tensor).all())

    readouts.log_z.tensor.sum().backward()

    gradients = tuple(
        parameter.grad
        for parameter in scorer.parameters()
        if parameter.grad is not None
    )
    assert gradients
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)


def test_the_configured_decoder_returns_at_most_the_declared_k_best() -> None:
    decoded = build().k_best(leaves())

    assert all(0 < len(item) <= 2 for item in decoded)


def test_maximum_does_not_publish_probabilistic_marginals() -> None:
    scorer = build(MaximumReduction())
    readouts = scorer.readouts(leaves())

    assert scorer.publishes == {
        "actions",
        "cells",
        "identity",
        "inside",
        "log_z",
    }
    assert readouts.alpha is None
    assert readouts.mu is None
    assert readouts.outside is None
    assert readouts.log_z is not None
    assert readouts.derivations is None
    assert bool(torch.isfinite(readouts.log_z.tensor).all())


def test_the_site_map_contains_only_readouts_the_reduction_publishes() -> None:
    sites = tuple(build(MaximumReduction()).site_map())
    fields = {site.key[0] for site in sites}

    assert fields == {"cells", "identity", "inside", "log_z"}
