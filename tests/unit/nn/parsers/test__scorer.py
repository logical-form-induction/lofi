from math import inf
from typing import TYPE_CHECKING, Final, Literal, Never, cast

import pytest
import torch
from tests._controllers import mlp_controller
from tests.conformance._sites import SiteObservation
from tests.conformance.chart_scorer import (
    ChartInstrumentationConformance,
    ChartInstrumentationSuite,
    ChartScorerConformance,
    ChartScorerSuite,
    published,
)
from torch import nn

from lofi.deduce import SpanIndex
from lofi.judge import Action, ActionSet
from lofi.nn._values import Encoded, LogDistribution
from lofi.nn.parsers import (
    AttendSplitWeigher,
    DeepInsideOutside,
    InsideKernel,
    LogSumExpReduction,
    MarginalSplitWeigher,
    MlpSpanFunction,
    OutsideKernel,
    SplitChannel,
    SplitScoring,
)
from lofi.nn.parsers import _scorer as scorer_module
from lofi.nn.parsers._scorer import _live_spans
from lofi.nn.parsers.composition import MeanFieldComposition
from lofi.nn.ports import ChartScorer, ReceiverLocalChartScorer
from lofi.nn.psi import RuleScorer, RuleScores
from lofi.nn.sites import (
    Access,
    ComponentId,
    ComponentKind,
    DTypeClass,
    ReceiverSide,
    Role,
    Site,
    SiteMap,
)
from lofi.probe import (
    PatchFrom,
    SiteDecl,
    SiteMapIso,
    SiteMapSpec,
    Unit,
    Within,
    Zero,
    ensure_publishes,
    instrument,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import Tensor

    from lofi.nn._values import ChartReadouts
    from lofi.nn.ports import Reading

INPUT_WIDTH: Final = 5
WIDTH: Final = 6
HIDDEN: Final = 7
CHART: Final = ComponentId("chart")
CONTROLLER: Final = ComponentId("controller")

APPLY: Final = Action("apply", "application")
COMPOSE: Final = Action("compose", "composition")
KEEP: Final = Action("keep")
ACTIONS: Final = ActionSet((APPLY, COMPOSE, KEEP), KEEP)

LEAF_AXES: Final = ("item", "word", "width")

AXES: Final[Mapping[str, tuple[str, ...]]] = {
    "inside": ("item", "span"),
    "outside": ("item", "span"),
    "alpha": ("item", "span", "split"),
    "mu": ("item", "span"),
    "log_z": ("item",),
    "identity": ("item", "span", "split"),
    "cells": ("item", "span", "width"),
    "outside_cells": ("item", "span", "width"),
    "spans": ("item", "span", "span_width"),
    "actions": ("item", "span", "split", "action"),
    "rules": ("item", "span", "split", "rule"),
}

ROLES: Final[Mapping[str, Role]] = {
    "inside": Role.INSIDE_SCORE,
    "outside": Role.OUTSIDE_SCORE,
    "alpha": Role.SPLIT_DISTRIBUTION,
    "mu": Role.CONSTITUENT_MARGINAL,
    "log_z": Role.LOG_PARTITION,
    "identity": Role.IDENTITY_SCORE,
    "cells": Role.CHART_CELL,
    "outside_cells": Role.OUTSIDE_CELL,
    "spans": Role.SPAN_REPRESENTATION,
}

DTYPES: Final[Mapping[str, DTypeClass]] = {
    "inside": DTypeClass.REAL,
    "outside": DTypeClass.REAL,
    "alpha": DTypeClass.LOG_PROBABILITY,
    "mu": DTypeClass.PROBABILITY,
    "log_z": DTypeClass.REAL,
    "identity": DTypeClass.REAL,
    "cells": DTypeClass.REAL,
    "outside_cells": DTypeClass.REAL,
    "spans": DTypeClass.REAL,
}

ITEMS: Final = 2
WORDS: Final = 5

#: How far two ways of computing one probability may differ at double precision.
TOLERANCE: Final = 1e-6


def span_function(*input_widths: int) -> MlpSpanFunction:
    """Return the one-layer function used by the scorer fixture."""
    return MlpSpanFunction(input_widths, WIDTH, layers=1, nonlinearity="tanh")


def composition() -> MeanFieldComposition:
    """Return the direct composition used by the scorer fixture."""
    return MeanFieldComposition(
        span_function(INPUT_WIDTH), span_function(WIDTH, WIDTH), width=WIDTH
    )


def build(
    *,
    seed: int = 11,
    dtype: torch.dtype = torch.float32,
    attend: bool = False,
    controller_input: Literal["inside", "span"] = "inside",
    outside_weights: Literal["reuse", "contextual"] = "contextual",
) -> DeepInsideOutside:
    torch.manual_seed(seed)
    controller_width = 2 * WIDTH if controller_input == "span" else WIDTH
    controller = mlp_controller(
        ACTIONS,
        width=controller_width,
        component=CONTROLLER,
        hidden=HIDDEN,
        input_source=controller_input,
    )
    rules = RuleScorer(controller, width=controller_width)

    weigher = (
        AttendSplitWeigher(
            MlpSpanFunction((WIDTH, WIDTH), 1, layers=1, nonlinearity="tanh")
        )
        if attend
        else MarginalSplitWeigher()
    )

    return DeepInsideOutside(
        rules,
        composition=composition(),
        split_left=span_function(WIDTH, WIDTH),
        split_right=span_function(WIDTH, WIDTH),
        outside_weights=outside_weights,
        root_outside="learned",
        root_init_gain=1.0,
        component=CHART,
        reduction=LogSumExpReduction(),
        weigher=weigher,
        checkpoint_from_width=None,
    ).to(dtype)


def test_only_contextual_outside_vectors_are_published_independently() -> None:
    contextual = build(outside_weights="contextual")
    reused = build(outside_weights="reuse")

    assert contextual.contextual_outside
    assert "outside_cells" in contextual.publishes
    assert contextual.readouts(leaves(ITEMS, WORDS)).outside_cells is not None
    assert not reused.contextual_outside
    assert "outside_cells" not in reused.publishes
    assert reused.readouts(leaves(ITEMS, WORDS)).outside_cells is None
    assert reused.readouts(leaves(ITEMS, WORDS)).spans is not None


def leaves(
    items: int, words: int, *, seed: int = 3, dtype: torch.dtype = torch.float32
) -> Encoded:
    generator = torch.Generator().manual_seed(seed)
    lengths = [max(1, words - item) for item in range(items)]
    values = torch.randn(items, words, INPUT_WIDTH, generator=generator, dtype=dtype)
    mask = torch.zeros(items, words, dtype=torch.bool)

    for item, length in enumerate(lengths):
        mask[item, :length] = True

    return Encoded(values, LEAF_AXES, mask)


def wire(scorer: DeepInsideOutside, readout: str) -> None:
    if readout not in scorer.publishes:
        msg = f"no pass fills {readout!r}"
        raise ValueError(msg)


def observe(scorer: DeepInsideOutside) -> Mapping[Site, Reading]:
    watched = frozenset(scorer.site_map())

    with scorer.instrument(watched, patches={}) as scope:
        scorer.readouts(leaves(ITEMS, WORDS))

    return scope.readings


def read(scorer: DeepInsideOutside) -> tuple[Tensor, ...]:
    return tuple(
        value.tensor
        for value in published(scorer.readouts(leaves(ITEMS, WORDS))).values()
    )


class TestTheDeepInsideOutsideScorer(ChartScorerSuite):
    @pytest.fixture
    def conformance(self) -> ChartScorerConformance[DeepInsideOutside]:
        return ChartScorerConformance(
            build=build,
            leaves=leaves,
            axes=AXES,
            split_axis="split",
            ragged_axis="span",
            wire=wire,
            sites=SiteObservation(build=build, observe=observe, read=read),
        )


def chart_instrumentation() -> ChartInstrumentationConformance[
    DeepInsideOutside, Reading
]:
    return ChartInstrumentationConformance(
        build=build,
        leaves=leaves,
        scope=lambda subject, sites: subject.instrument(sites, patches={}),
        items=ITEMS,
        words=WORDS,
    )


class TestDeepInsideOutsideInstrumentation(ChartInstrumentationSuite):
    @pytest.fixture
    def chart_instrumentation(
        self,
    ) -> ChartInstrumentationConformance[DeepInsideOutside, Reading]:
        return chart_instrumentation()


def packed(readouts: ChartReadouts) -> Encoded:
    cells = readouts.cells

    assert cells is not None

    return Encoded(cells.tensor.transpose(0, 1), ("span", "item", "width"))


def rescored(
    scorer: DeepInsideOutside, readouts: ChartReadouts, index: SpanIndex
) -> tuple[Tensor, Tensor]:
    scores = scorer.rules.score(packed(readouts), index=index)

    return (scores.split_scores.tensor, scores.log_scores.tensor)


def test_the_scorer_is_a_chart_scorer() -> None:
    assert isinstance(build(), ChartScorer)
    assert isinstance(build(), ReceiverLocalChartScorer)


def test_the_root_and_every_leaf_are_constituents_with_certainty() -> None:
    scorer = build(dtype=torch.float64)
    words = leaves(ITEMS, WORDS, dtype=torch.float64)
    readouts = scorer.readouts(words)
    mu = readouts.mu
    index = SpanIndex(WORDS)

    assert mu is not None
    assert mu.mask is not None

    for item in range(ITEMS):
        length = WORDS - item
        root = index.span_id(0, length)

        assert mu.tensor[item, root].item() == pytest.approx(1.0, abs=TOLERANCE)
        assert torch.allclose(
            mu.tensor[item, :length], torch.ones(length, dtype=torch.float64)
        )
        # a binary tree over the item brackets one span per internal node
        assert mu.tensor[item, WORDS:].sum().item() == pytest.approx(
            length - 1, abs=TOLERANCE
        )
        assert bool(mu.tensor[item][~mu.mask[item]].eq(0.0).all())


def test_the_marginals_are_the_gradient_of_the_partition_at_the_rule_score() -> None:
    scorer = build(dtype=torch.float64)
    words = leaves(ITEMS, WORDS, dtype=torch.float64)
    readouts = scorer.readouts(words)
    index = SpanIndex(WORDS)
    lengths = torch.tensor([WORDS - item for item in range(ITEMS)])
    split_scores, _ = rescored(scorer, readouts, index)
    psi = split_scores.detach().requires_grad_()
    log_z = InsideKernel(LogSumExpReduction()).log_partition(
        torch.zeros(ITEMS, WORDS, dtype=torch.float64), psi, index, lengths=lengths
    )
    (counts,) = torch.autograd.grad(log_z.sum(), psi)
    live = _live_spans(lengths, log_z.new_zeros(ITEMS, index.num_spans), index)
    mu = readouts.mu
    alpha = readouts.alpha
    published_z = readouts.log_z

    assert mu is not None
    assert alpha is not None
    assert published_z is not None
    assert torch.allclose(published_z.tensor, log_z, atol=TOLERANCE)
    assert torch.allclose(
        counts.sum(dim=-1)[:, WORDS:], mu.tensor[:, WORDS:], atol=TOLERANCE
    )

    bracketed = live & (mu.tensor > TOLERANCE)
    bracketed[:, :WORDS] = False
    expected = counts / mu.tensor.unsqueeze(-1).clamp_min(TOLERANCE)

    assert torch.allclose(
        alpha.tensor.exp()[bracketed], expected[bracketed], atol=TOLERANCE
    )


def test_the_outside_pass_reads_the_split_scores_the_inside_pass_built() -> None:
    scorer = build(dtype=torch.float64)
    words = leaves(ITEMS, WORDS, dtype=torch.float64)
    readouts = scorer.readouts(words)
    index = SpanIndex(WORDS)
    lengths = torch.tensor([WORDS - item for item in range(ITEMS)])
    split_scores, _ = rescored(scorer, readouts, index)
    inside = readouts.inside
    outside = readouts.outside

    assert inside is not None
    assert outside is not None
    assert inside.mask is not None

    swept = InsideKernel(LogSumExpReduction()).inside(
        torch.zeros(ITEMS, WORDS, dtype=torch.float64), split_scores, index
    )
    surrounding = OutsideKernel(LogSumExpReduction()).outside(
        swept, split_scores, index, lengths=lengths
    )
    live = inside.mask

    assert torch.allclose(inside.tensor[live], swept[live], atol=TOLERANCE)
    assert torch.allclose(outside.tensor[live], surrounding[live], atol=TOLERANCE)
    assert bool(inside.tensor[~live].eq(-inf).all())
    assert bool(outside.tensor[~live].eq(-inf).all())


NONE_APPLIES: Final = Action("none-applies")
NULL_ACTIONS: Final = ActionSet(
    (APPLY, COMPOSE, KEEP, NONE_APPLIES), KEEP, null=NONE_APPLIES
)


def test_a_split_is_weighed_by_every_action_but_the_null_action() -> None:
    torch.manual_seed(11)
    controller = mlp_controller(
        NULL_ACTIONS, width=WIDTH, component=CONTROLLER, hidden=HIDDEN
    )
    scorer = DeepInsideOutside(
        RuleScorer(controller, width=WIDTH),
        composition=composition(),
        split_left=span_function(WIDTH, WIDTH),
        split_right=span_function(WIDTH, WIDTH),
        outside_weights="contextual",
        root_outside="learned",
        root_init_gain=1.0,
        component=CHART,
        reduction=LogSumExpReduction(),
        weigher=MarginalSplitWeigher(),
        checkpoint_from_width=None,
    ).to(torch.float64)
    words = leaves(ITEMS, WORDS, dtype=torch.float64)
    readouts = scorer.readouts(words)
    index = SpanIndex(WORDS)
    _, log_scores = rescored(scorer, readouts, index)
    null = NULL_ACTIONS.index(NONE_APPLIES.name)
    kept = torch.cat((log_scores[..., :null], log_scores[..., null + 1 :]), dim=-1)
    inside = readouts.inside

    assert inside is not None
    assert inside.mask is not None

    live = inside.mask
    leaf_scores = torch.zeros(ITEMS, WORDS, dtype=torch.float64)
    applying = InsideKernel(LogSumExpReduction()).inside(
        leaf_scores, kept.logsumexp(dim=-1), index
    )
    every = InsideKernel(LogSumExpReduction()).inside(
        leaf_scores, log_scores.logsumexp(dim=-1), index
    )

    assert torch.allclose(inside.tensor[live], applying[live], atol=TOLERANCE)
    assert not torch.allclose(inside.tensor[live], every[live], atol=TOLERANCE)


def test_the_identity_weight_is_the_identity_column_of_the_rule_score() -> None:
    scorer = build(dtype=torch.float64)
    words = leaves(ITEMS, WORDS, dtype=torch.float64)
    readouts = scorer.readouts(words)
    index = SpanIndex(WORDS)
    _, log_scores = rescored(scorer, readouts, index)
    actions = readouts.actions
    identity = readouts.identity

    assert actions is not None
    assert identity is not None
    assert identity.mask is not None
    assert identity is actions.identity
    assert identity is actions.by_action[KEEP]

    column = ACTIONS.index(KEEP.name)
    live = identity.mask

    assert torch.allclose(
        identity.tensor[live], log_scores[..., column][live], atol=TOLERANCE
    )
    assert bool(identity.tensor[~live].eq(-inf).all())


def test_every_action_score_is_the_masked_rule_score_column_and_keeps_gradients() -> (
    None
):
    scorer = build(dtype=torch.float64)
    words = leaves(ITEMS, WORDS, dtype=torch.float64)
    readouts = scorer.readouts(words)
    index = SpanIndex(WORDS)
    _, rescored_actions = rescored(scorer, readouts, index)
    actions = readouts.actions

    assert actions is not None
    assert actions.action_set is ACTIONS
    assert actions.scores.mask is not None
    assert tuple(actions.by_action) == ACTIONS.actions

    lengths = torch.tensor([WORDS - item for item in range(ITEMS)])
    ends = torch.tensor([end for _, end in index.spans_by_length()])
    live_spans = ends.unsqueeze(0) <= lengths.unsqueeze(-1)
    real_splits = torch.tensor(index.split_mask, dtype=torch.bool)
    expected_mask = live_spans.unsqueeze(-1) & real_splits.unsqueeze(0)
    mask = actions.scores.mask
    expected = rescored_actions.masked_fill(~expected_mask.unsqueeze(-1), -inf)

    assert torch.equal(mask, expected_mask)
    assert torch.allclose(actions.scores.tensor, expected, atol=TOLERANCE)

    compatibility_bias = dict(scorer.rules.named_parameters())["_compatibility_bias"]
    expected_upstream_gradient = torch.full_like(
        compatibility_bias, int(expected_mask.sum())
    )

    for column, action in enumerate(ACTIONS.actions):
        projected = actions.by_action[action]

        assert projected.mask is mask
        assert torch.equal(projected.tensor, actions.scores.tensor[..., column])
        assert bool(projected.tensor[~mask].eq(-inf).all())
        (gradient,) = torch.autograd.grad(
            projected.tensor[mask].sum(),
            actions.scores.tensor,
            retain_graph=True,
        )
        expected_gradient = torch.zeros_like(actions.scores.tensor)
        expected_gradient[..., column][mask] = 1.0

        assert torch.equal(gradient, expected_gradient)
        (upstream_gradient,) = torch.autograd.grad(
            projected.tensor[expected_mask].sum(),
            compatibility_bias,
            retain_graph=True,
        )

        assert torch.equal(upstream_gradient, expected_upstream_gradient)


def test_a_cell_is_the_composition_of_its_children_mixed_under_the_posterior() -> None:
    scorer = build(dtype=torch.float64)
    words = leaves(1, 4, dtype=torch.float64)
    readouts = scorer.readouts(words)
    index = SpanIndex(4)
    cells = readouts.cells
    alpha = readouts.alpha

    assert cells is not None
    assert alpha is not None

    for i, j in index.spans_by_length(min_length=2):
        mixed = torch.zeros(WIDTH, dtype=torch.float64)

        for column, k in enumerate(index.splits(i, j)):
            left = cells.tensor[0, index.span_id(i, k)]
            right = cells.tensor[0, index.span_id(k, j)]
            pair = scorer.compose(left.unsqueeze(0), right.unsqueeze(0)).squeeze(0)
            mixed = mixed + alpha.tensor[0, index.span_id(i, j), column].exp() * pair

        assert torch.allclose(cells.tensor[0, index.span_id(i, j)], mixed, atol=1e-9)


def test_an_item_is_scored_as_though_it_had_been_scored_alone() -> None:
    scorer = build(dtype=torch.float64)
    batch = leaves(ITEMS, WORDS, dtype=torch.float64)
    together = scorer.readouts(batch)
    shorter = ITEMS - 1
    length = WORDS - shorter
    alone = scorer.readouts(
        Encoded(
            batch.tensor[shorter : shorter + 1, :length].clone(),
            LEAF_AXES,
            torch.ones(1, length, dtype=torch.bool),
        )
    )
    whole = SpanIndex(WORDS)
    part = SpanIndex(length)

    for name in ("inside", "outside", "mu", "log_z"):
        batched = cast("LogDistribution | Encoded", getattr(together, name)).tensor
        single = cast("LogDistribution | Encoded", getattr(alone, name)).tensor

        if name == "log_z":
            assert batched[shorter].item() == pytest.approx(
                single[0].item(), abs=TOLERANCE
            )
            continue

        for i, j in part.spans_by_length():
            assert batched[shorter, whole.span_id(i, j)].item() == pytest.approx(
                single[0, part.span_id(i, j)].item(), abs=TOLERANCE
            )

    cells = together.cells
    own = alone.cells

    assert cells is not None
    assert own is not None

    for i, j in part.spans_by_length():
        assert torch.allclose(
            cells.tensor[shorter, whole.span_id(i, j)],
            own.tensor[0, part.span_id(i, j)],
            atol=TOLERANCE,
        )


def test_a_gradient_reaches_every_parameter_of_the_scorer() -> None:
    scorer = build()
    readouts = scorer.readouts(leaves(ITEMS, WORDS))
    log_z = readouts.log_z
    mu = readouts.mu
    cells = readouts.cells
    spans = readouts.spans

    assert log_z is not None
    assert mu is not None
    assert cells is not None
    assert spans is not None

    (
        log_z.tensor.sum() + mu.tensor.sum() + cells.tensor.sum() + spans.tensor.sum()
    ).backward()
    named = dict(scorer.named_parameters())

    assert named

    for name, parameter in named.items():
        assert parameter.grad is not None, name
        assert bool(parameter.grad.ne(0.0).any()), name


def test_attend_split_parameters_train_through_the_complete_chart() -> None:
    scorer = build(attend=True)
    readouts = scorer.readouts(leaves(ITEMS, WORDS))
    log_z = readouts.log_z
    spans = readouts.spans

    assert log_z is not None
    assert spans is not None

    (log_z.tensor.sum() + spans.tensor.sum()).backward()
    learned = {
        name: parameter
        for name, parameter in scorer.named_parameters()
        if name.startswith("_learned_weigher.")
    }

    assert learned
    for name, parameter in learned.items():
        assert parameter.grad is not None, name
        assert bool(parameter.grad.ne(0.0).any()), name


def test_a_span_controller_trains_from_post_outside_child_representations() -> None:
    scorer = build(attend=True, controller_input="span")
    readouts = scorer.readouts(leaves(ITEMS, WORDS))
    actions = readouts.actions

    assert actions is not None
    assert actions.scores.mask is not None
    assert scorer.rules.controller.input == "span"
    assert scorer.rules.controller.input_width == 2 * WIDTH

    first = actions.scores.tensor[..., 0]
    (-first[actions.scores.mask].mean()).backward()

    controller = cast("nn.Module", scorer.rules.controller)
    parameters = tuple(controller.parameters())
    assert parameters
    assert all(parameter.grad is not None for parameter in parameters)
    assert any(
        bool(parameter.grad is not None and parameter.grad.ne(0.0).any())
        for parameter in parameters
    )


def test_a_span_controller_is_refused_a_weigher_that_reads_no_children() -> None:
    assert not MarginalSplitWeigher().reads_children
    assert build(attend=True, controller_input="span").rules.controller.input == "span"

    with pytest.raises(ValueError, match="reads the child vectors"):
        build(controller_input="span")


def test_a_full_batch_needs_no_mask_and_marks_every_span_live() -> None:
    scorer = build()
    values = torch.randn(ITEMS, 3, INPUT_WIDTH)
    readouts = scorer.readouts(Encoded(values, LEAF_AXES))
    inside = readouts.inside
    alpha = readouts.alpha

    assert inside is not None
    assert inside.mask is not None
    assert bool(inside.mask.all())
    assert bool(inside.tensor.isfinite().all())
    assert alpha is not None
    assert alpha.mask is not None
    assert torch.equal(
        alpha.mask[0], torch.tensor(SpanIndex(3).split_mask, dtype=torch.bool)
    )


def test_a_single_word_is_a_chart_with_one_cell_and_no_split() -> None:
    scorer = build()
    readouts = scorer.readouts(Encoded(torch.randn(1, 1, INPUT_WIDTH), LEAF_AXES))
    log_z = readouts.log_z
    alpha = readouts.alpha
    mu = readouts.mu

    assert log_z is not None
    assert alpha is not None
    assert mu is not None
    assert log_z.tensor.item() == pytest.approx(0.0)
    assert tuple(alpha.tensor.shape) == (1, 1, 0)
    assert mu.tensor.item() == pytest.approx(1.0)


def test_alpha_and_mu_observations_are_the_exact_returned_values() -> None:
    scorer = build(dtype=torch.float64)
    alpha_site = Site(CHART, Role.SPLIT_DISTRIBUTION, ("alpha",))
    mu_site = Site(CHART, Role.CONSTITUENT_MARGINAL, ("mu",))

    with scorer.instrument(frozenset({alpha_site, mu_site}), patches={}) as scope:
        readouts = scorer.readouts(leaves(ITEMS, WORDS, dtype=torch.float64))

    alpha = readouts.alpha
    mu = readouts.mu

    assert alpha is not None
    assert mu is not None
    assert scope.readings[alpha_site].tensor is alpha.tensor
    assert scope.readings[alpha_site].axes is alpha.axes
    assert scope.readings[alpha_site].mask is alpha.mask
    assert scope.readings[mu_site].tensor is mu.tensor
    assert scope.readings[mu_site].axes is mu.axes
    assert scope.readings[mu_site].mask is mu.mask


def test_publishing_does_not_interpret_a_site_key_as_a_readout_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scorer = build(dtype=torch.float64)
    site = Site(CHART, Role.INSIDE_SCORE, ("inside",))

    with (
        scorer.instrument(frozenset({site}), patches={}) as scope,
        monkeypatch.context() as patch,
    ):
        patch.setattr(Site, "key", property(lambda _: ("not-a-readout",)))
        readouts = scorer.readouts(leaves(ITEMS, WORDS, dtype=torch.float64))

    inside = readouts.inside

    assert inside is not None
    assert scope.readings[site].tensor is inside.tensor
    assert scope.readings[site].axes is inside.axes
    assert scope.readings[site].mask is inside.mask


def test_selected_sites_are_recorded_in_the_site_map_s_order() -> None:
    scorer = build()
    selected = frozenset(
        {
            Site(CHART, Role.CONSTITUENT_MARGINAL, ("mu",)),
            Site(CHART, Role.INSIDE_SCORE, ("inside",)),
            Site(CHART, Role.SPLIT_DISTRIBUTION, ("alpha",)),
        }
    )

    with scorer.instrument(selected, patches={}) as scope:
        scorer.readouts(leaves(ITEMS, WORDS))

    assert tuple(scope.readings) == tuple(
        site for site in scorer.site_map() if site in selected
    )


def test_nested_scopes_receive_only_the_chart_readouts_they_select() -> None:
    scorer = build()
    alpha_site = Site(CHART, Role.SPLIT_DISTRIBUTION, ("alpha",))
    mu_site = Site(CHART, Role.CONSTITUENT_MARGINAL, ("mu",))

    with (
        scorer.instrument(frozenset({alpha_site}), patches={}) as outer,
        scorer.instrument(frozenset({mu_site}), patches={}) as inner,
    ):
        readouts = scorer.readouts(leaves(ITEMS, WORDS))

    assert set(outer.readings) == {alpha_site}
    assert set(inner.readings) == {mu_site}
    assert readouts.alpha is not None
    assert readouts.mu is not None
    assert outer.readings[alpha_site].tensor is readouts.alpha.tensor
    assert inner.readings[mu_site].tensor is readouts.mu.tensor


def test_a_scope_detaches_and_the_scorer_is_reusable_after_forward_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scorer = build()
    site = Site(CHART, Role.SPLIT_DISTRIBUTION, ("alpha",))

    def planted_failure(
        channel: SplitChannel,
        leaf_vectors: torch.Tensor,
        leaf_scores: torch.Tensor,
        scoring: SplitScoring,
        index: SpanIndex,
        /,
        *,
        recompute: bool,
    ) -> Never:
        del channel, leaf_vectors, leaf_scores, scoring, index, recompute
        msg = "planted chart failure"
        raise RuntimeError(msg)

    failed = scorer.instrument(frozenset({site}), patches={})

    with monkeypatch.context() as patch:
        patch.setattr(scorer_module.SplitChannel, "sweep_scoring", planted_failure)

        with pytest.raises(RuntimeError, match="planted chart failure"), failed:
            scorer.readouts(leaves(ITEMS, WORDS))

    assert not failed.readings
    scorer.readouts(leaves(1, 3))

    with scorer.instrument(frozenset({site}), patches={}) as recovered:
        readouts = scorer.readouts(leaves(1, 3))

    assert readouts.alpha is not None
    assert recovered.readings[site].tensor is readouts.alpha.tensor


class _IdentityPatch:
    def stands_in_for(self, reading: Reading, /) -> torch.Tensor:
        return reading.tensor


def test_an_unknown_site_is_refused_by_name() -> None:
    scorer = build()
    unknown = Site(CHART, Role.SPLIT_DISTRIBUTION, ("missing",))

    with pytest.raises(ValueError, match="SPLIT_DISTRIBUTION") as refused:
        scorer.instrument(frozenset({unknown}), patches={})

    assert "missing" in str(refused.value)

    with pytest.raises(ValueError, match="SPLIT_DISTRIBUTION") as patched:
        scorer.instrument(frozenset(), patches={unknown: _IdentityPatch()})

    assert "missing" in str(patched.value)


@pytest.mark.parametrize(
    "site", tuple(site for site in build().site_map() if site != build().receiver_site)
)
def test_every_chart_site_is_observation_only(site: Site) -> None:
    with pytest.raises(ValueError, match=site.role.name) as refused:
        build().instrument(frozenset(), patches={site: _IdentityPatch()})

    assert "observation-only" in str(refused.value)


def test_an_uninstrumented_call_allocates_no_site_or_reading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scorer = build()

    def forbidden_site(
        component: ComponentId,
        role: Role,
        key: tuple[int | str, ...] = (),
    ) -> Never:
        del component, role, key
        pytest.fail("an uninstrumented scorer call allocated a site")

    def forbidden_reading(
        site: Site,
        tensor: torch.Tensor,
        axes: tuple[str, ...],
        mask: torch.Tensor | None = None,
    ) -> Never:
        del site, tensor, axes, mask
        pytest.fail("an uninstrumented scorer call allocated a reading")

    def forbidden_publish(readouts: ChartReadouts, /) -> Never:
        del readouts
        pytest.fail("an uninstrumented scorer call entered observation")

    monkeypatch.setattr(scorer_module, "Site", forbidden_site)
    monkeypatch.setattr(scorer_module, "_Reading", forbidden_reading)
    monkeypatch.setattr(scorer, "_publish", forbidden_publish)

    scorer.readouts(leaves(ITEMS, WORDS))


def test_the_site_map_names_every_readout_and_survives_the_probe_codec() -> None:
    scorer = build()
    points = scorer.site_map()
    filled = published(scorer.readouts(leaves(ITEMS, WORDS)))
    instrumented = {name: value for name, value in filled.items() if name in ROLES}

    assert {str(site.key[0]) for site in points} == {*instrumented, "receiver"}
    assert scorer.publishes == set(filled)
    assert scorer.publishes - set(instrumented) == {"actions"}

    for site, spec in points.items():
        name = str(site.key[0])

        if site == scorer.receiver_site:
            assert site.role is Role.RECEIVER_EDGE
            assert spec.axes == ("item", "edge", "width")
            assert spec.dtype_class is DTypeClass.REAL
            assert spec.differentiable
            assert spec.access == Access.OBSERVE | Access.PATCH
            continue

        assert site.component == CHART
        assert site.role is ROLES[name]
        assert spec.axes == instrumented[name].axes
        assert spec.dtype_class is DTYPES[name]
        assert spec.differentiable
        assert spec.access == Access.OBSERVE

    codec = SiteMapIso()

    assert codec.forward(codec.backward(points)) == points


def test_receiver_edges_are_a_bijection_in_public_axis_order() -> None:
    scorer = build()
    index = SpanIndex(3)
    edges = scorer.receiver_edges(index)

    assert tuple(edge.position for edge in edges) == tuple(range(8))
    identities = tuple(
        (edge.source, edge.target, edge.split, edge.side) for edge in edges
    )

    assert identities == (
        (0, 3, 0, ReceiverSide.LEFT),
        (1, 3, 0, ReceiverSide.RIGHT),
        (1, 4, 0, ReceiverSide.LEFT),
        (2, 4, 0, ReceiverSide.RIGHT),
        (0, 5, 0, ReceiverSide.LEFT),
        (4, 5, 0, ReceiverSide.RIGHT),
        (3, 5, 1, ReceiverSide.LEFT),
        (2, 5, 1, ReceiverSide.RIGHT),
    )
    assert tuple(edge.coordinates for edge in edges) == tuple(
        {"edge": position} for position in range(8)
    )
    assert len({(edge.source, edge.target) for edge in edges}) == len(edges)


def test_receiver_observation_is_bit_exact_with_the_rule_scorer_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scorer = build(dtype=torch.float64)
    site = scorer.receiver_site
    items = 2
    captured: list[Tensor] = []
    original = scorer.rules.score

    def capture(cells: Encoded, /, *, index: SpanIndex) -> RuleScores:
        paired = torch.stack((cells.tensor[0], cells.tensor[1]), dim=1)
        captured.append(paired.reshape(items, -1, WIDTH))

        return original(cells, index=index)

    monkeypatch.setattr(scorer.rules, "score", capture)

    with scorer.instrument(frozenset({site}), patches={}) as scope:
        scorer.readouts(leaves(items, 4, dtype=torch.float64))

    reading = scope.readings[site]

    assert reading.axes == ("item", "edge", "width")
    assert torch.equal(reading.tensor, torch.cat(captured, dim=1))
    assert reading.mask is not None


def test_a_receiver_mask_names_exactly_the_live_edges_of_each_item() -> None:
    scorer = build()
    words = 5
    index = SpanIndex(words)
    site = scorer.receiver_site

    with scorer.instrument(frozenset({site}), patches={}) as scope:
        scorer.readouts(leaves(ITEMS, words))

    mask = scope.readings[site].mask

    assert mask is not None
    assert tuple(mask.shape) == (ITEMS, len(scorer.receiver_edges(index)))

    for item, length in enumerate((words, words - 1)):
        expected = torch.tensor(
            [
                index.span_of(edge.target)[1] <= length
                for edge in scorer.receiver_edges(index)
            ]
        )

        assert torch.equal(mask[item], expected)


def test_one_receiver_patch_leaves_its_sibling_live_and_two_scopes_compose() -> None:
    scorer = build(dtype=torch.float64)
    values = leaves(1, 3, dtype=torch.float64)
    site = scorer.receiver_site
    first, sibling = scorer.receiver_edges(SpanIndex(3))[:2]

    with scorer.instrument(frozenset({site}), patches={}) as baseline_scope:
        baseline = scorer.readouts(values)

    first_patch = Within((Unit(site, first.coordinates),), Zero())
    sibling_patch = Within((Unit(site, sibling.coordinates),), Zero())

    with (
        instrument(scorer, patch={site: first_patch}) as outer,
        instrument(scorer, observe=frozenset({site})) as inner,
    ):
        one_patched = inner(lambda: outer(lambda: scorer.readouts(values)))

    observed = inner.trace[site].tensor
    original = baseline_scope.readings[site].tensor

    assert torch.equal(
        observed[:, first.position], torch.zeros_like(observed[:, first.position])
    )
    assert torch.equal(observed[:, sibling.position], original[:, sibling.position])

    with (
        instrument(scorer, patch={site: first_patch}) as outer,
        instrument(scorer, patch={site: sibling_patch}) as inner,
    ):
        nested = inner(lambda: outer(lambda: scorer.readouts(values)))

    both = Within(
        (Unit(site, first.coordinates), Unit(site, sibling.coordinates)), Zero()
    )

    with instrument(scorer, patch={site: both}) as patched:
        together = patched(lambda: scorer.readouts(values))

    assert baseline.cells is not None
    assert one_patched.cells is not None
    assert not torch.equal(baseline.cells.tensor, one_patched.cells.tensor)

    for name, nested_value in published(nested).items():
        assert torch.equal(nested_value.tensor, published(together)[name].tensor)


def test_patching_both_receiver_edges_matches_the_two_word_counterfactual() -> None:
    scorer = build(dtype=torch.float64)
    site = scorer.receiver_site
    source = leaves(1, 2, seed=5, dtype=torch.float64)
    counterfactual_source = leaves(1, 2, seed=17, dtype=torch.float64)

    with instrument(scorer, observe=frozenset({site})) as observed:
        counterfactual = observed(lambda: scorer.readouts(counterfactual_source))

    replacement = PatchFrom(observed.trace, site)

    with instrument(scorer, patch={site: replacement}) as patched:
        result = patched(lambda: scorer.readouts(source))

    assert len(scorer.receiver_edges(SpanIndex(2))) == 2
    root = SpanIndex(2).span_id(0, 2)

    for name, value in published(counterfactual).items():
        actual = published(result)[name].tensor

        if name in {"cells", "outside_cells", "spans"}:
            assert torch.equal(value.tensor[:, root], actual[:, root])
        else:
            assert torch.equal(value.tensor, actual)


class _WrongShapePatch:
    def stands_in_for(self, reading: Reading, /) -> torch.Tensor:
        return reading.tensor[:, :-1]


def test_receiver_patches_refuse_unknown_axes_coordinates_and_shapes_by_name() -> None:
    scorer = build()
    values = leaves(1, 3)
    site = scorer.receiver_site
    past = len(scorer.receiver_edges(SpanIndex(3)))

    for unit, message in (
        (Unit(site, {"span": 0}), "span"),
        (Unit(site, {"edge": past}), "edge"),
    ):
        patch = Within((unit,), Zero())

        with (
            instrument(scorer, patch={site: patch}) as patched,
            pytest.raises((KeyError, ValueError), match=message),
        ):
            patched(lambda: scorer.readouts(values))

    with (
        scorer.instrument(frozenset(), patches={site: _WrongShapePatch()}),
        pytest.raises(ValueError, match="RECEIVER_EDGE") as refused,
    ):
        scorer.readouts(values)

    assert "shape" in str(refused.value)


def test_receiver_instrumentation_preserves_parameters_and_restores_after_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scorer = build()
    values = leaves(1, 3)
    site = scorer.receiver_site
    edge = scorer.receiver_edges(SpanIndex(3))[0]
    intervention = Within((Unit(site, edge.coordinates),), Zero())
    patching = scorer.instrument(frozenset({site}), patches={})
    expected = scorer.readouts(values)
    state = (*scorer.named_parameters(), *scorer.named_buffers())
    before = {name: value.detach().clone() for name, value in state}

    def planted_failure(
        channel: SplitChannel,
        leaf_vectors: torch.Tensor,
        leaf_scores: torch.Tensor,
        scoring: SplitScoring,
        index: SpanIndex,
        /,
        *,
        recompute: bool,
    ) -> Never:
        del channel, leaf_vectors, leaf_scores, scoring, index, recompute
        msg = "receiver pass failed"
        raise RuntimeError(msg)

    with monkeypatch.context() as planted:
        planted.setattr(scorer_module.SplitChannel, "sweep_scoring", planted_failure)

        with pytest.raises(RuntimeError, match="receiver pass failed"), patching:
            scorer.readouts(values)

    assert not patching.readings
    assert all(
        torch.equal(before[name], value)
        for name, value in (*scorer.named_parameters(), *scorer.named_buffers())
    )

    recovered = scorer.readouts(values)

    for name, value in published(recovered).items():
        assert torch.equal(value.tensor, published(expected)[name].tensor)

    with instrument(scorer, patch={site: intervention}) as patched:
        patched(lambda: scorer.readouts(values))

    assert all(
        torch.equal(before[name], value)
        for name, value in (*scorer.named_parameters(), *scorer.named_buffers())
    )


class _ReachRecorder:
    def __init__(self) -> None:
        self.masks: list[Tensor] = []

    def stands_in_for(self, reading: Reading, /) -> torch.Tensor:
        if reading.mask is None:
            msg = "a receiver reach carries a mask"
            raise ValueError(msg)

        self.masks.append(reading.mask.clone())

        return reading.tensor


def test_each_receiver_reach_masks_exactly_the_edges_it_consumes() -> None:
    scorer = build()
    words = 5
    site = scorer.receiver_site
    recorder = _ReachRecorder()

    with scorer.instrument(frozenset({site}), patches={site: recorder}) as scope:
        scorer.readouts(leaves(ITEMS, words))

    final = scope.readings[site].mask
    reached = torch.stack(recorder.masks)

    assert final is not None
    assert len(recorder.masks) == words - 1
    assert bool(reached.sum(dim=0).le(1).all())
    assert torch.equal(reached.any(dim=0), final)


def test_a_one_word_chart_publishes_an_empty_receiver_axis() -> None:
    scorer = build()
    site = scorer.receiver_site

    with scorer.instrument(frozenset({site}), patches={}) as scope:
        scorer.readouts(leaves(1, 1))

    reading = scope.readings[site]

    assert tuple(reading.tensor.shape) == (1, 0, WIDTH)
    assert reading.mask is not None
    assert tuple(reading.mask.shape) == (1, 0)


def test_the_deep_parser_site_map_is_tagged_and_accepted() -> None:
    points = build().site_map()

    assert points.component == CHART
    assert points.kind is ComponentKind.PARSER
    assert ensure_publishes(points) is points


def test_a_deep_parser_publishing_only_part_of_its_deep_structure_is_refused() -> None:
    points = build().site_map()
    partial = SiteMap(
        CHART,
        ComponentKind.PARSER,
        {
            site: spec
            for site, spec in points.items()
            if site.role is not Role.OUTSIDE_SCORE
        },
    )

    with pytest.raises(ValueError, match="deep is half exposed") as refusal:
        ensure_publishes(partial)

    assert "OUTSIDE_SCORE" in str(refusal.value)


def redeclared(
    name: str,
    *,
    axes: tuple[str, ...] | None = None,
    dtype_class: DTypeClass | None = None,
) -> SiteMapSpec:
    """Return the real scorer's declaration with one site specification changed.

    Parameters
    ----------
    name : str
        The readout to change.
    axes : tuple[str, ...] | None
        Replacement axes, or the published axes when absent.
    dtype_class : DTypeClass | None
        Replacement numeric class, or the published class when absent.

    Returns
    -------
    SiteMapSpec
        The altered declaration.
    """
    declared = SiteMapIso().backward(build().site_map())
    sites = tuple(
        SiteDecl(
            role=site.role,
            key=site.key,
            axes=axes if site.key == (name,) and axes is not None else site.axes,
            dtype_class=(
                dtype_class.value
                if site.key == (name,) and dtype_class is not None
                else site.dtype_class
            ),
            differentiable=site.differentiable,
            access=site.access,
        )
        for site in declared.sites
    )

    return SiteMapSpec(
        component=declared.component,
        kind=declared.kind,
        sites=sites,
    )


def test_the_boundary_refuses_a_real_deep_parser_with_wrong_axes() -> None:
    with pytest.raises(ValueError, match="OUTSIDE_SCORE runs over") as refusal:
        SiteMapIso().forward(redeclared("outside", axes=("span", "item")))

    message = str(refusal.value)

    assert repr(Site(CHART, Role.OUTSIDE_SCORE, ("outside",))) in message
    assert "('span', 'item')" in message
    assert "('item', 'span')" in message


def test_the_boundary_refuses_a_real_deep_parser_with_the_wrong_numeric_class() -> None:
    with pytest.raises(
        ValueError, match="CONSTITUENT_MARGINAL holds probability"
    ) as refusal:
        SiteMapIso().forward(redeclared("mu", dtype_class=DTypeClass.REAL))

    message = str(refusal.value)

    assert repr(Site(CHART, Role.CONSTITUENT_MARGINAL, ("mu",))) in message
    assert "as real" in message
    assert "holds probability" in message


def test_the_scorer_reports_its_widths_and_its_rules() -> None:
    scorer = build()

    assert scorer.input_width == INPUT_WIDTH
    assert scorer.output_width == WIDTH
    assert scorer.span_width == 2 * WIDTH
    assert scorer.rules.input_width == WIDTH
    assert scorer.rules.action_set == ACTIONS
    assert f"input_width={INPUT_WIDTH}" in repr(scorer)


def test_leaves_over_other_axes_or_no_words_are_refused() -> None:
    scorer = build()

    with pytest.raises(ValueError, match="pooled leaves run over"):
        scorer.readouts(
            Encoded(torch.randn(ITEMS, WORDS, INPUT_WIDTH), ("item", "row", "width"))
        )

    with pytest.raises(ValueError, match="at least one word"):
        scorer.readouts(Encoded(torch.randn(ITEMS, 0, INPUT_WIDTH), LEAF_AXES))


def checkpointed_build(threshold: int | None) -> DeepInsideOutside:
    torch.manual_seed(11)
    controller = mlp_controller(
        ACTIONS, width=WIDTH, component=CONTROLLER, hidden=HIDDEN
    )

    return DeepInsideOutside(
        RuleScorer(controller, width=WIDTH),
        composition=composition(),
        split_left=span_function(WIDTH, WIDTH),
        split_right=span_function(WIDTH, WIDTH),
        outside_weights="contextual",
        root_outside="learned",
        root_init_gain=1.0,
        component=CHART,
        reduction=LogSumExpReduction(),
        weigher=MarginalSplitWeigher(),
        checkpoint_from_width=threshold,
    )


def test_checkpointed_chart_widths_publish_the_same_readouts_and_gradients() -> None:
    published: list[tuple[torch.Tensor, ...]] = []
    gradients: list[dict[str, torch.Tensor]] = []

    for threshold in (None, 2):
        scorer = checkpointed_build(threshold)
        readouts = scorer.readouts(leaves(ITEMS, 7))
        actions = readouts.actions
        cells = readouts.cells
        log_z = readouts.log_z

        assert actions is not None
        assert cells is not None
        assert log_z is not None

        total = (
            log_z.tensor.sum()
            + actions.scores.tensor.clamp(min=-30.0).sum()
            + cells.tensor.sum()
        )
        total.backward()
        published.append((log_z.tensor, actions.scores.tensor, cells.tensor))
        gradients.append(
            {
                name: parameter.grad.clone()
                for name, parameter in scorer.named_parameters()
                if parameter.grad is not None
            }
        )

    assert all(
        torch.equal(found, expected)
        for found, expected in zip(published[1], published[0], strict=True)
    )
    assert set(gradients[1]) == set(gradients[0])
    assert gradients[0]
    assert all(
        torch.allclose(gradients[1][name], gradients[0][name], rtol=0.0, atol=1e-6)
        for name in gradients[0]
    )


def test_an_instrumented_pass_reads_the_same_receiver_edges_under_checkpointing() -> (
    None
):
    readings = [observe(checkpointed_build(threshold)) for threshold in (None, 2)]

    assert set(readings[0]) == set(readings[1])
    assert all(
        torch.equal(readings[0][site].tensor, readings[1][site].tensor)
        for site in readings[0]
    )
