import math
from typing import TYPE_CHECKING

import pytest
import torch

import lofi.nn.loss._parser as parser_module
from lofi.deduce import SpanIndex
from lofi.nn._values import Encoded, LogDistribution
from lofi.nn.loss._errors import InvalidCutpointError, UncomputedCellError
from lofi.nn.loss._parser import (
    SPAN_AXES,
    SPLIT_AXES,
    SpanMarginals,
    SplitPosterior,
)

if TYPE_CHECKING:
    from collections.abc import Callable

ITEMS = 2
LENGTH = 4


def spans(length: int = LENGTH) -> SpanIndex:
    return SpanIndex(length)


def cutpoints(index: SpanIndex, items: int = ITEMS) -> torch.Tensor:
    valid = torch.zeros(items, index.num_spans, index.max_splits, dtype=torch.bool)

    for i, j in index.spans_by_length(min_length=2):
        for k in index.splits(i, j):
            span, column = index.split_id(i, j, k)
            valid[:, span, column] = True

    return valid


def uniform_scores(valid: torch.Tensor) -> torch.Tensor:
    counts = valid.sum(dim=2, keepdim=True).clamp(min=1)

    return torch.where(valid, -counts.to(torch.float).log(), -torch.inf)


def uniform_splits(index: SpanIndex, items: int = ITEMS) -> LogDistribution:
    valid = cutpoints(index, items)

    return LogDistribution(
        uniform_scores(valid),
        SPLIT_AXES,
        valid,
        normalized_over="split",
    )


def posterior(index: SpanIndex, items: int = ITEMS) -> SplitPosterior:
    return SplitPosterior(index, uniform_splits(index, items))


def marginals(index: SpanIndex, items: int = ITEMS) -> SpanMarginals:
    split = posterior(index, items)
    valid = split.valid
    weights = torch.where(valid, split.log_probabilities.tensor.exp(), 0.0)
    constituent = torch.zeros(items, index.num_spans)
    rule = torch.zeros(items, index.num_spans, index.max_splits)
    constituent[:, index.span_id(0, index.n)] = 1.0

    for i, j in index.spans_by_length_desc(min_length=2):
        span = index.span_id(i, j)

        for k in index.splits(i, j):
            _, column = index.split_id(i, j, k)
            share = constituent[:, span] * weights[:, span, column]
            rule[:, span, column] = share

            for start, end in ((i, k), (k, j)):
                if end - start > 1:
                    constituent[:, index.span_id(start, end)] += share

    return SpanMarginals(
        index,
        Encoded(constituent, SPAN_AXES, valid.any(dim=2)),
        Encoded(rule, SPLIT_AXES, valid),
        torch.full((items,), index.n),
    )


def test_a_posterior_reads_back_what_the_parser_published() -> None:
    index = spans()
    published = uniform_splits(index)

    split = SplitPosterior(index, published)

    assert split.spans is index
    assert split.log_probabilities is published
    assert split.valid is published.mask
    assert split.items == ITEMS
    assert not hasattr(split, "__dict__")


def test_a_posterior_over_the_wrong_axes_is_refused() -> None:
    index = spans()
    valid = cutpoints(index)
    renamed = LogDistribution(
        uniform_scores(valid),
        ("item", "cell", "cut"),
        valid,
        normalized_over="cut",
    )

    with pytest.raises(ValueError, match="this contract is over"):
        SplitPosterior(index, renamed)


def test_a_posterior_normalizing_over_the_wrong_axis_is_refused() -> None:
    index = spans()
    valid = cutpoints(index)
    unnormalized = LogDistribution(uniform_scores(valid), SPLIT_AXES, valid)

    with pytest.raises(ValueError, match="mass sums over 'split'"):
        SplitPosterior(index, unnormalized)


def test_a_posterior_with_no_validity_mask_is_refused() -> None:
    index = spans()
    scores = torch.zeros(ITEMS, index.num_spans, index.max_splits).log_softmax(dim=2)
    unmasked = LogDistribution(scores, SPLIT_AXES, normalized_over="split")

    with pytest.raises(ValueError, match="validity mask over its cutpoints"):
        SplitPosterior(index, unmasked)


def test_a_mask_that_stops_short_of_the_cutpoints_is_refused() -> None:
    index = spans()
    valid = cutpoints(index)
    coarse = LogDistribution(
        uniform_scores(valid),
        SPLIT_AXES,
        valid.any(dim=2),
        normalized_over="split",
    )

    with pytest.raises(ValueError, match="covers its cutpoint axis"):
        SplitPosterior(index, coarse)


def test_a_posterior_the_index_does_not_address_is_refused() -> None:
    index = spans()
    wider = SpanIndex(LENGTH + 1)

    with pytest.raises(ValueError, match="spans, and the index addresses"):
        SplitPosterior(wider, uniform_splits(index))


def test_a_cutpoint_the_span_has_reads_back_its_log_probability() -> None:
    index = spans()
    split = posterior(index)

    read = split.at(0, 0, LENGTH, 1)

    assert read.ndim == 0
    assert float(read.exp()) == pytest.approx(1.0 / (LENGTH - 1))


def test_a_cutpoint_outside_the_span_is_a_named_error() -> None:
    split = posterior(spans())

    with pytest.raises(InvalidCutpointError) as raised:
        split.at(1, 0, 2, 5)

    assert raised.value.item == 1
    assert raised.value.span == (0, 2)
    assert raised.value.cut == 5


def ruling_out(index: SpanIndex, item: int, i: int, j: int, k: int) -> SplitPosterior:
    valid = cutpoints(index)
    span, column = index.split_id(i, j, k)
    valid[item, span, column] = False

    return SplitPosterior(
        index,
        LogDistribution(
            uniform_scores(valid),
            SPLIT_AXES,
            valid,
            normalized_over="split",
        ),
    )


def test_a_cutpoint_the_mask_rules_out_is_a_named_error() -> None:
    index = spans()
    split = ruling_out(index, 0, 0, LENGTH, 1)

    with pytest.raises(InvalidCutpointError):
        split.at(0, 0, LENGTH, 1)

    assert float(split.at(1, 0, LENGTH, 1).exp()) == pytest.approx(
        1.0 / (LENGTH - 1),
    )


def test_a_span_outside_the_input_is_refused() -> None:
    split = posterior(spans())

    with pytest.raises(ValueError, match="not a span of an input"):
        split.at(0, 0, LENGTH + 2, 1)


def test_the_mass_of_a_span_is_one_where_it_divides_and_zero_where_it_does_not() -> (
    None
):
    index = spans()
    split = posterior(index)

    mass = split.mass()

    for i, j in index.spans_by_length():
        wanted = 1.0 if j - i > 1 else 0.0
        assert float(mass[0, index.span_id(i, j)]) == pytest.approx(wanted)


def test_the_mass_of_a_span_stays_one_when_a_cutpoint_is_ruled_out() -> None:
    index = spans()
    split = ruling_out(index, 0, 0, LENGTH, 1)

    mass = split.mass()
    root = index.span_id(0, LENGTH)

    assert float(mass[0, root]) == pytest.approx(1.0)
    assert float(split.at(0, 0, LENGTH, 2).exp()) == pytest.approx(1.0 / (LENGTH - 2))


def test_a_posterior_says_what_it_is_over() -> None:
    assert "SplitPosterior(items=2, spans=4" in repr(posterior(spans()))


def test_marginals_read_back_what_the_parser_published() -> None:
    index = spans()

    published = marginals(index)

    assert published.spans is index
    assert published.items == ITEMS
    assert published.constituent.axes == SPAN_AXES
    assert published.rule.axes == SPLIT_AXES
    assert published.built is published.constituent.mask
    assert published.lengths.tolist() == [LENGTH] * ITEMS
    assert not hasattr(published, "__dict__")


def test_the_constituent_mass_of_an_item_is_one_less_than_its_length() -> None:
    for length in (1, 2, 3, 5):
        published = marginals(spans(length))

        assert published.constituent_mass().tolist() == pytest.approx(
            [length - 1.0] * ITEMS,
        )


def test_a_constituent_marginal_over_the_wrong_axes_is_refused() -> None:
    index = spans()
    published = marginals(index)

    with pytest.raises(ValueError, match="this contract is over"):
        SpanMarginals(
            index,
            Encoded(
                published.constituent.tensor,
                ("item", "cell"),
                published.constituent.mask,
            ),
            published.rule,
            published.lengths,
        )


def test_marginals_with_no_mask_are_refused() -> None:
    index = spans()
    published = marginals(index)

    with pytest.raises(ValueError, match="which spans and cutpoints are real"):
        SpanMarginals(
            index,
            Encoded(published.constituent.tensor, SPAN_AXES),
            published.rule,
            published.lengths,
        )


def test_lengths_that_are_not_counts_are_refused() -> None:
    index = spans()
    published = marginals(index)

    with pytest.raises(TypeError, match="a token count is an integer"):
        SpanMarginals(
            index,
            published.constituent,
            published.rule,
            published.lengths.to(torch.float),
        )


def test_lengths_that_do_not_cover_the_items_are_refused() -> None:
    index = spans()
    published = marginals(index)

    with pytest.raises(ValueError, match="items to give a length to"):
        SpanMarginals(
            index,
            published.constituent,
            published.rule,
            torch.full((ITEMS + 1,), LENGTH),
        )


def test_a_length_the_index_cannot_address_is_refused() -> None:
    index = spans()
    published = marginals(index)

    with pytest.raises(ValueError, match="outside the range"):
        SpanMarginals(
            index,
            published.constituent,
            published.rule,
            torch.full((ITEMS,), LENGTH + 3),
        )


def test_the_debug_checks_reject_a_leaf_counted_as_a_constituent(
    reload_with_checks: Callable[..., None],
) -> None:
    reload_with_checks(parser_module, on=True)
    index = spans()
    published = marginals(index)
    mask = published.constituent.mask

    assert mask is not None
    with_leaves = mask.clone()
    with_leaves[:, index.span_id(0, 1)] = True

    with pytest.raises(ValueError, match="span of width one"):
        parser_module.SpanMarginals(
            index,
            Encoded(published.constituent.tensor, SPAN_AXES, with_leaves),
            published.rule,
            published.lengths,
        )


def test_a_span_reads_back_its_constituent_marginal() -> None:
    index = spans()
    published = marginals(index)

    assert float(published.at(0, 0, LENGTH)) == pytest.approx(1.0)
    assert float(published.at(0, 0, 2)) < 1.0


def shortened_marginals() -> SpanMarginals:
    """Return differentiable marginals over lengths four and two."""
    index = spans()
    published = marginals(index)
    lengths = torch.tensor((4, 2))
    built = published.built.clone()
    cutpoint_mask = published.rule.mask

    assert cutpoint_mask is not None
    valid = cutpoint_mask.clone()

    for i, j in index.spans_by_length():
        if j > 2:
            built[1, index.span_id(i, j)] = False

            if j - i > 1:
                valid[1, index.span_id(i, j)] = False

    return SpanMarginals(
        index,
        Encoded(
            published.constituent.tensor.detach().requires_grad_(),
            SPAN_AXES,
            built,
        ),
        Encoded(published.rule.tensor, SPLIT_AXES, valid),
        lengths,
    )


def test_the_complete_constituent_factor_owns_leaf_root_and_internal_mass() -> None:
    published = shortened_marginals()
    index = published.spans
    factor = published.constituent_factor()

    assert factor.mask is not None
    assert float(published.at(0, 1, 2).detach()) == 1.0
    assert float(published.at(1, 0, 2).detach()) == 1.0
    assert float(published.at(0, 0, 2).detach()) == pytest.approx(
        float(published.constituent.tensor[0, index.span_id(0, 2)].detach())
    )
    assert not bool(factor.mask[1, index.span_id(2, 3)])


def test_the_complete_constituent_factor_preserves_only_internal_gradients() -> None:
    published = shortened_marginals()
    index = published.spans
    factor = published.constituent_factor()
    loss = (
        factor.tensor[0, index.span_id(0, 2)]
        + factor.tensor[0, index.span_id(0, 1)]
        + factor.tensor[1, index.span_id(0, 2)]
    )

    loss.backward()
    gradient = published.constituent.tensor.grad

    assert gradient is not None
    assert float(gradient[0, index.span_id(0, 2)]) == 1.0
    assert float(gradient[0, index.span_id(0, 1)]) == 0.0
    assert float(gradient[1, index.span_id(0, 2)]) == 0.0


def test_gathering_a_padded_constituent_is_a_named_refusal() -> None:
    published = shortened_marginals()

    with pytest.raises(UncomputedCellError) as raised:
        published.gather_constituent(((1, 2, 3),))

    assert raised.value.item == 1
    assert raised.value.span == (2, 3)


def test_zero_constituent_mass_has_exact_log_complement_and_finite_gradient() -> None:
    published = shortened_marginals()
    index = published.spans
    values = published.constituent.tensor.detach().clone()
    internal = index.span_id(0, 2)
    values[0, internal] = 0.0
    values.requires_grad_()
    endpoint = SpanMarginals(
        index,
        Encoded(values, SPAN_AXES, published.built),
        published.rule,
        published.lengths,
    )

    read = endpoint.log_nonconstituent(((0, 0, 2),))[0]
    read.backward()

    assert read.item() == 0.0
    assert values.grad is not None
    assert values.grad[0, internal].item() == -1.0
    assert bool(values.grad.isfinite().all())


def test_unit_internal_mass_has_exact_log_complement_and_zero_gradient() -> None:
    published = shortened_marginals()
    index = published.spans
    values = published.constituent.tensor.detach().clone()
    internal = index.span_id(0, 2)
    values[0, internal] = 1.0
    values.requires_grad_()
    endpoint = SpanMarginals(
        index,
        Encoded(values, SPAN_AXES, published.built),
        published.rule,
        published.lengths,
    )

    read = endpoint.log_nonconstituent(((0, 0, 2),))[0]
    read.backward()

    assert read.item() == -math.inf
    assert values.grad is not None
    assert not bool(values.grad.any())
    assert bool(values.grad.isfinite().all())


@pytest.mark.parametrize(("item", "span"), [(0, (1, 2)), (1, (0, 2))])
def test_fixed_leaf_and_root_have_exact_log_complement_and_zero_gradient(
    item: int,
    span: tuple[int, int],
) -> None:
    published = shortened_marginals()
    values = published.constituent.tensor

    read = published.log_nonconstituent(((item, *span),))[0]
    read.backward()

    assert read.item() == -math.inf
    assert values.grad is not None
    assert not bool(values.grad.any())
    assert bool(values.grad.isfinite().all())


def test_log_nonconstituent_refuses_padding_before_autograd() -> None:
    published = shortened_marginals()

    with pytest.raises(UncomputedCellError) as raised:
        published.log_nonconstituent(((1, 2, 3),))

    assert raised.value.item == 1
    assert raised.value.span == (2, 3)
    assert published.constituent.tensor.grad is None


@pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan")])
def test_gathering_an_improper_constituent_probability_is_refused(
    probability: float,
) -> None:
    published = shortened_marginals()
    index = published.spans
    values = published.constituent.tensor.detach().clone()
    values[0, index.span_id(0, 2)] = probability
    malformed = SpanMarginals(
        index,
        Encoded(values, SPAN_AXES, published.built),
        published.rule,
        published.lengths,
    )

    with pytest.raises(ValueError, match="outside the unit interval"):
        malformed.gather_constituent(((0, 0, 2),))


def test_a_cutpoint_reads_back_its_rule_marginal() -> None:
    index = spans()
    published = marginals(index)

    total = sum(
        float(published.at_cutpoint(0, 0, LENGTH, k)) for k in index.splits(0, LENGTH)
    )

    assert total == pytest.approx(1.0)


def test_a_rule_marginal_at_a_cutpoint_the_span_lacks_is_a_named_error() -> None:
    published = marginals(spans())

    with pytest.raises(InvalidCutpointError):
        published.at_cutpoint(0, 0, 2, 3)


def test_marginals_say_what_they_are_over() -> None:
    assert "SpanMarginals(items=2, spans=4" in repr(marginals(spans()))
