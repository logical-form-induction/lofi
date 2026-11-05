from math import inf
from typing import TYPE_CHECKING, Final

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from tests._controllers import mlp_controller

from lofi.deduce import SpanIndex
from lofi.judge import Action, ActionSet
from lofi.nn._values import Encoded, LogDistribution
from lofi.nn.parsers import InsideKernel, LogSumExpReduction
from lofi.nn.psi import RuleScorer, RuleScores
from lofi.nn.sites import ComponentId, SiteMap

if TYPE_CHECKING:
    from torch import Tensor

    from lofi.nn.controllers import MlpController


WIDTH: Final = 4
HIDDEN: Final = 5
WIRED: Final = ComponentId("chart-controller")

JOIN: Final = Action("join", "concatenation")
KEEP: Final = Action("keep")
APPLY: Final = Action("apply", "application")

ACTIONS: Final = ActionSet((JOIN, KEEP, APPLY), KEEP)

SCORE_AXES: Final = ("item", "span", "split", "action")
SPLIT_AXES: Final = ("item", "span", "split")


def controller(actions: ActionSet = ACTIONS) -> MlpController:
    torch.manual_seed(17)

    return mlp_controller(actions, width=WIDTH, component=WIRED, hidden=HIDDEN)


def scorer(actions: ActionSet = ACTIONS) -> RuleScorer:
    built = RuleScorer(controller(actions), width=WIDTH)

    torch.manual_seed(23)

    with torch.no_grad():
        for parameter in built.parameters():
            parameter.normal_()

    return built


def test_a_rule_scorer_may_borrow_the_controller_a_stack_owns_once() -> None:
    shared = controller()
    owned = RuleScorer(shared, width=WIDTH)
    borrowed = RuleScorer(shared, width=WIDTH, register_controller=False)

    assert owned.controller is shared
    assert borrowed.controller is shared
    assert shared in tuple(owned.modules())
    assert shared not in tuple(borrowed.modules())
    assert not any(name.startswith("_controller.") for name in borrowed.state_dict())


def cells(index: SpanIndex, items: int = 2) -> Encoded:
    numbers = torch.linspace(
        -1.0, 1.0, index.num_spans * items * WIDTH, dtype=torch.float32
    )

    return Encoded(
        numbers.reshape(index.num_spans, items, WIDTH), ("span", "item", "width")
    )


class Recording:
    def __init__(self, wrapped: MlpController) -> None:
        self._wrapped = wrapped
        self.pairs: list[int] = []
        self.handed: list[tuple[Encoded, Encoded]] = []

    @property
    def action_set(self) -> ActionSet:
        return self._wrapped.action_set

    @property
    def input_width(self) -> int:
        return self._wrapped.input_width

    @property
    def input(self) -> str:
        return self._wrapped.input

    def weigh(self, left: Encoded, /, *, right: Encoded) -> LogDistribution:
        self.pairs.append(left.size("cell"))
        self.handed.append((left, right))

        return self._wrapped.weigh(left, right=right)

    def site_map(self) -> SiteMap:
        return self._wrapped.site_map()


class Sideways:
    def __init__(self, wrapped: MlpController) -> None:
        self._wrapped = wrapped

    @property
    def action_set(self) -> ActionSet:
        return self._wrapped.action_set

    @property
    def input_width(self) -> int:
        return self._wrapped.input_width

    @property
    def input(self) -> str:
        return self._wrapped.input

    def weigh(self, left: Encoded, /, *, right: Encoded) -> LogDistribution:
        weights = self._wrapped.weigh(left, right=right)

        return LogDistribution(weights.tensor, ("cell", "combinator"))

    def site_map(self) -> SiteMap:
        return self._wrapped.site_map()


def test_the_scores_carry_the_pinned_axis_order_and_the_indexs_shape() -> None:
    index = SpanIndex(4)
    scores = scorer().score(cells(index), index=index)

    assert scores.log_scores.axes == SCORE_AXES
    assert scores.split_scores.axes == SPLIT_AXES
    assert tuple(scores.log_scores.tensor.shape) == (
        2,
        index.num_spans,
        index.max_splits,
        len(ACTIONS.actions),
    )
    assert tuple(scores.split_scores.tensor.shape) == (
        2,
        index.num_spans,
        index.max_splits,
    )
    assert scores.index is index


def test_the_mask_marks_exactly_the_splits_the_index_says_a_span_has() -> None:
    index = SpanIndex(4)
    scores = scorer().score(cells(index), index=index)
    declared = torch.tensor(index.split_mask).unsqueeze(0).expand(2, -1, -1)

    assert torch.equal(scores.mask, declared)


def test_a_split_a_span_does_not_have_weighs_the_reductions_zero() -> None:
    index = SpanIndex(4)
    scores = scorer().score(cells(index), index=index)
    padded = ~scores.mask

    assert bool(scores.split_scores.tensor[padded].eq(-inf).all())
    assert bool(scores.log_scores.tensor[padded].eq(-inf).all())
    assert bool(scores.split_scores.tensor[scores.mask].isfinite().all())


def test_summing_a_splits_actions_leaves_the_pairs_compatibility() -> None:
    index = SpanIndex(4)
    scores = scorer().score(cells(index), index=index)
    live = scores.mask
    summed = scores.log_scores.tensor.logsumexp(dim=-1)

    assert torch.allclose(summed[live], scores.split_scores.tensor[live], atol=1e-5)


def test_the_action_weights_inside_a_split_are_a_distribution() -> None:
    index = SpanIndex(4)
    scores = scorer().score(cells(index), index=index)
    live = scores.mask
    shares = scores.log_scores.tensor - scores.split_scores.tensor.unsqueeze(-1)

    assert torch.allclose(
        shares.logsumexp(dim=-1)[live],
        torch.zeros_like(shares[..., 0][live]),
        atol=1e-5,
    )


def test_every_reader_of_one_pass_is_handed_the_same_tensor() -> None:
    index = SpanIndex(4)
    scores = scorer().score(cells(index), index=index)

    assert scores.split_scores is scores.split_scores
    assert scores.split_scores.tensor is scores.split_scores.tensor
    assert scores.log_scores is scores.log_scores
    assert scores.log_scores.tensor is scores.log_scores.tensor


def test_the_inside_sweep_and_the_outside_counts_read_one_tensor() -> None:
    index = SpanIndex(4)
    scores = scorer().score(cells(index), index=index)
    weights = scores.split_scores.tensor
    partition = InsideKernel(LogSumExpReduction()).log_partition(
        torch.zeros(2, index.n), weights, index
    )

    (counts,) = torch.autograd.grad(partition.sum(), scores.split_scores.tensor)

    assert bool(counts.isfinite().all())
    assert torch.allclose(
        counts[:, index.span_id(0, index.n), :].sum(dim=-1), torch.ones(2), atol=1e-5
    )
    assert bool(counts[~scores.mask].eq(0.0).all())


def test_the_expected_counts_of_a_span_never_exceed_one() -> None:
    index = SpanIndex(5)
    scores = scorer().score(cells(index), index=index)
    weights = scores.split_scores.tensor
    partition = InsideKernel(LogSumExpReduction()).log_partition(
        torch.zeros(2, index.n), weights, index
    )
    (counts,) = torch.autograd.grad(partition.sum(), weights)

    assert bool((counts.sum(dim=-1) <= 1.0 + 1e-5).all())
    assert bool((counts >= -1e-6).all())


def test_the_controller_is_called_once_for_a_whole_pass() -> None:
    index = SpanIndex(4)
    recording = Recording(controller())
    subject = RuleScorer(recording, width=WIDTH)

    subject.score(cells(index), index=index)

    assert recording.pairs == [2 * index.num_spans * index.max_splits]

    subject.score(cells(index, items=3), index=index)

    assert len(recording.pairs) == 2
    assert recording.pairs[1] == 3 * index.num_spans * index.max_splits


def test_the_pair_at_one_row_is_the_two_children_the_index_names() -> None:
    index = SpanIndex(3)
    recording = Recording(controller())
    chart = cells(index)

    RuleScorer(recording, width=WIDTH).score(chart, index=index)

    left, right = recording.handed[0]
    span = index.span_id(0, index.n)
    items = chart.size("item")

    for offset, cut in enumerate(index.splits(0, index.n)):
        for item in range(items):
            row = (item * index.num_spans + span) * index.max_splits + offset

            assert torch.equal(
                left.tensor[row], chart.tensor[index.span_id(0, cut), item]
            )
            assert torch.equal(
                right.tensor[row], chart.tensor[index.span_id(cut, index.n), item]
            )


def test_how_well_a_pair_composes_reaches_the_split_weight() -> None:
    index = SpanIndex(4)
    scores = scorer().score(cells(index), index=index)
    live = scores.split_scores.tensor[scores.mask]

    assert float(live.max().detach() - live.min().detach()) > 1e-3


def test_the_split_weight_changes_with_the_cells_it_weighs() -> None:
    index = SpanIndex(3)
    subject = scorer()
    chart = cells(index)
    other = Encoded(chart.tensor.flip(0), ("span", "item", "width"))
    first = subject.score(chart, index=index)
    second = subject.score(other, index=index)

    assert not torch.allclose(
        first.split_scores.tensor[first.mask], second.split_scores.tensor[second.mask]
    )


def test_a_gradient_reaches_the_controller_and_the_compatibility() -> None:
    index = SpanIndex(3)
    subject = scorer()
    scores = subject.score(cells(index), index=index)

    scores.split_scores.tensor[scores.mask].sum().backward()

    named = dict(subject.named_parameters())
    reached = {
        name: parameter.grad
        for name, parameter in named.items()
        if parameter.grad is not None and bool(parameter.grad.abs().sum() > 0.0)
    }

    assert "_compatibility.weight" in reached
    assert any(name.startswith("_controller.") for name in named)
    assert any(name.startswith("_controller.") for name in reached)


def test_the_cells_reach_the_score_through_the_children_the_index_names() -> None:
    index = SpanIndex(3)
    subject = scorer()
    numbers = cells(index).tensor.clone().requires_grad_()

    scores = subject.score(Encoded(numbers, ("span", "item", "width")), index=index)
    span = index.span_id(0, 3)
    scores.split_scores.tensor[0, span, :].sum().backward()

    assert numbers.grad is not None
    touched = {
        cell for cell in range(index.num_spans) if bool(numbers.grad[cell, 0].any())
    }

    assert touched == {
        index.span_id(0, 1),
        index.span_id(1, 3),
        index.span_id(0, 2),
        index.span_id(2, 3),
    }
    assert not bool(numbers.grad[:, 1].any())


def test_the_action_axis_is_the_action_sets_own_columns() -> None:
    index = SpanIndex(3)
    wider = ActionSet((JOIN, KEEP, APPLY, Action("raise", "type-raising")), KEEP)
    narrow = scorer().score(cells(index), index=index)
    broad = scorer(wider).score(cells(index), index=index)

    assert narrow.log_scores.size("action") == 3
    assert broad.log_scores.size("action") == 4
    assert scorer(wider).action_set.actions == wider.actions


def test_a_one_token_input_has_no_split_to_weigh() -> None:
    index = SpanIndex(1)
    scores = scorer().score(cells(index), index=index)

    assert tuple(scores.log_scores.tensor.shape) == (2, 1, 0, 3)
    assert scores.mask.numel() == 0


@settings(max_examples=25)
@given(
    n=st.integers(min_value=1, max_value=6), items=st.integers(min_value=1, max_value=3)
)
def test_any_length_scores_every_split_the_index_addresses(n: int, items: int) -> None:
    index = SpanIndex(n)
    scores = scorer().score(cells(index, items=items), index=index)
    live = scores.mask

    assert tuple(scores.split_scores.tensor.shape) == (
        items,
        index.num_spans,
        index.max_splits,
    )
    assert int(live.sum()) == items * sum(map(sum, index.split_mask))
    assert bool(scores.split_scores.tensor[live].isfinite().all())


def test_cells_over_the_wrong_axes_are_refused() -> None:
    index = SpanIndex(3)
    sideways = Encoded(
        torch.zeros(2, index.num_spans, WIDTH), ("item", "span", "width")
    )

    with pytest.raises(ValueError, match="cells run over"):
        scorer().score(sideways, index=index)


def test_cells_for_another_index_are_refused() -> None:
    index = SpanIndex(3)

    with pytest.raises(ValueError, match="spans was handed cells"):
        scorer().score(cells(SpanIndex(4)), index=index)


def test_cells_of_another_width_are_refused() -> None:
    index = SpanIndex(3)
    narrow = Encoded(
        torch.zeros(index.num_spans, 2, WIDTH - 1), ("span", "item", "width")
    )

    with pytest.raises(ValueError, match="wide"):
        scorer().score(narrow, index=index)


def test_a_controller_weighing_over_other_axes_is_refused() -> None:
    index = SpanIndex(3)
    subject = RuleScorer(Sideways(controller()), width=WIDTH)

    with pytest.raises(ValueError, match="controller weighs over"):
        subject.score(cells(index), index=index)


def test_scores_of_the_wrong_rank_or_dtype_are_refused() -> None:
    index = SpanIndex(3)
    mask = torch.ones(2, index.num_spans, index.max_splits, dtype=torch.bool)

    with pytest.raises(ValueError, match="rank"):
        RuleScores(
            torch.zeros(2, index.num_spans, index.max_splits), mask, index, null=None
        )

    with pytest.raises(TypeError, match="log weight"):
        RuleScores(
            torch.zeros(2, index.num_spans, index.max_splits, 3, dtype=torch.long),
            mask,
            index,
            null=None,
        )


def test_a_mask_that_does_not_cover_the_scores_is_refused() -> None:
    index = SpanIndex(3)
    scores = torch.zeros(2, index.num_spans, index.max_splits, 3)

    with pytest.raises(ValueError, match="does not cover"):
        RuleScores(
            scores, torch.ones(2, index.num_spans, dtype=torch.bool), index, null=None
        )

    with pytest.raises(TypeError, match="booleans"):
        RuleScores(
            scores,
            torch.ones(2, index.num_spans, index.max_splits, dtype=torch.long),
            index,
            null=None,
        )


def test_scores_laid_out_under_another_index_are_refused() -> None:
    index = SpanIndex(3)
    scores = torch.zeros(2, index.num_spans, index.max_splits, 3)
    mask = torch.ones(2, index.num_spans, index.max_splits, dtype=torch.bool)

    with pytest.raises(ValueError, match="spans of at most"):
        RuleScores(scores, mask, SpanIndex(4), null=None)


def test_a_null_column_outside_the_action_columns_is_refused() -> None:
    index = SpanIndex(3)
    scores = torch.zeros(2, index.num_spans, index.max_splits, 3)
    mask = torch.ones(2, index.num_spans, index.max_splits, dtype=torch.bool)

    with pytest.raises(ValueError, match="null column 3 is not one of the 3"):
        RuleScores(scores, mask, index, null=3)


NONE_APPLIES: Final = Action("none-applies")
NULL_ACTIONS: Final = ActionSet(
    (JOIN, KEEP, APPLY, NONE_APPLIES), KEEP, null=NONE_APPLIES
)


def test_a_split_score_leaves_out_the_null_action() -> None:
    index = SpanIndex(4)
    scores = scorer(NULL_ACTIONS).score(cells(index), index=index)
    live = scores.mask
    kept = scores.log_scores.tensor[..., :3].logsumexp(dim=-1)
    every = scores.log_scores.tensor.logsumexp(dim=-1)

    assert torch.allclose(scores.split_scores.tensor[live], kept[live])
    assert not torch.allclose(scores.split_scores.tensor[live], every[live])


def test_the_scores_render_the_shape_they_were_computed_over() -> None:
    index = SpanIndex(3)
    scores = scorer().score(cells(index), index=index)

    assert repr(scores) == "RuleScores(shape=(2, 6, 2, 3), n=3)"


def _live_positions(scores: RuleScores) -> Tensor:
    return scores.split_scores.tensor[scores.mask]


def test_two_scorers_over_one_pair_of_cells_agree() -> None:
    index = SpanIndex(4)
    first = _live_positions(scorer().score(cells(index), index=index))
    second = _live_positions(scorer().score(cells(index), index=index))

    assert torch.equal(first, second)
