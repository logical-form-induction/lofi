import math
from typing import TYPE_CHECKING

import pytest
import torch

from lofi.nn._values import LogDistribution
from lofi.nn.loss._divergence import (
    KL,
    CrossEntropy,
    Divergence,
    FenchelYoung,
)
from lofi.nn.loss._kernels import (
    cross_entropy,
    fenchel_young,
    kullback_leibler,
    node_divergence,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

AXES = ("item", "node", "symbol")
ITEMS = 2
NODES = 3
SYMBOLS = 4
ALPHA = 1.5


def scores(seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)

    return torch.rand(
        ITEMS,
        NODES,
        SYMBOLS,
        generator=generator,
    ).log_softmax(dim=2)


def distribution(
    tensor: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> LogDistribution:
    return LogDistribution(tensor, AXES, mask, normalized_over="symbol")


def rows(tensor: torch.Tensor) -> list[list[float]]:
    return [
        [float(tensor[item, node, symbol].exp()) for symbol in range(SYMBOLS)]
        for item in range(ITEMS)
        for node in range(NODES)
    ]


def by_hand(
    left: torch.Tensor,
    right: torch.Tensor,
    node: Callable[[list[float], list[float]], float],
) -> float:
    return sum(node(p, q) for p, q in zip(rows(left), rows(right), strict=True))


def kl_node(p: list[float], q: list[float]) -> float:
    return sum(
        0.0 if a == 0.0 else a * (math.log(a) - math.log(b))
        for a, b in zip(p, q, strict=True)
    )


def cross_node(p: list[float], q: list[float]) -> float:
    return -sum(0.0 if a == 0.0 else a * math.log(b) for a, b in zip(p, q, strict=True))


def tsallis(values: list[float], alpha: float) -> float:
    return (sum(value**alpha for value in values) - 1.0) / (alpha * (alpha - 1.0))


def bregman_node(p: list[float], q: list[float]) -> float:
    slope = [value ** (ALPHA - 1.0) / (ALPHA - 1.0) for value in q]

    return (
        tsallis(p, ALPHA)
        - tsallis(q, ALPHA)
        - sum(gradient * (a - b) for gradient, a, b in zip(slope, p, q, strict=True))
    )


def test_the_forward_divergence_matches_a_term_by_term_sum() -> None:
    reference, model = scores(0), scores(1)

    computed = float(kullback_leibler(distribution(reference), distribution(model)))

    assert computed == pytest.approx(by_hand(reference, model, kl_node), rel=1e-6)


def test_the_cross_entropy_matches_a_term_by_term_sum() -> None:
    reference, model = scores(0), scores(1)

    computed = float(cross_entropy(distribution(reference), distribution(model)))

    assert computed == pytest.approx(by_hand(reference, model, cross_node), rel=1e-6)


def test_the_fenchel_young_loss_is_the_bregman_divergence_of_its_entropy() -> None:
    reference, model = scores(0), scores(1)

    computed = float(
        fenchel_young(distribution(reference), distribution(model), alpha=ALPHA),
    )

    assert computed == pytest.approx(by_hand(reference, model, bregman_node), rel=1e-5)


def test_the_cross_entropy_exceeds_the_divergence_by_the_reference_entropy() -> None:
    reference, model = distribution(scores(0)), distribution(scores(1))

    entropy = float(cross_entropy(reference, reference))
    gap = float(cross_entropy(reference, model)) - float(
        kullback_leibler(reference, model),
    )

    assert gap == pytest.approx(entropy, rel=1e-6)


@pytest.mark.parametrize(
    ("kernel", "wanted"),
    [(kullback_leibler, 0.0), (fenchel_young, 0.0)],
    ids=("kullback-leibler", "fenchel-young"),
)
def test_a_distribution_is_no_distance_from_itself(
    kernel: Callable[[LogDistribution, LogDistribution], Tensor],
    wanted: float,
) -> None:
    reference = distribution(scores(0))

    assert abs(float(kernel(reference, reference)) - wanted) < 1e-7


def sparse_model(reference: torch.Tensor) -> LogDistribution:
    model = scores(1)
    model[0, 0, 0] = -math.inf
    model = model.log_softmax(dim=2)
    model[0, 0, 0] = -math.inf

    assert float(reference[0, 0, 0].exp()) > 0.0

    return distribution(model)


def test_the_divergence_is_infinite_where_the_model_rules_out_held_mass() -> None:
    reference = scores(0)
    model = sparse_model(reference)

    assert math.isinf(float(kullback_leibler(distribution(reference), model)))
    assert math.isinf(float(cross_entropy(distribution(reference), model)))


def test_the_fenchel_young_loss_stays_finite_on_a_sparse_model() -> None:
    reference = scores(0)
    model = sparse_model(reference)

    loss = float(fenchel_young(distribution(reference), model, alpha=ALPHA))

    assert math.isfinite(loss)
    assert loss > 0.0


def test_a_symbol_the_reference_rules_out_contributes_nothing_and_no_gradient() -> None:
    reference = scores(0)
    reference[1, 2, 3] = -math.inf
    reference = reference.log_softmax(dim=2)
    reference[1, 2, 3] = -math.inf
    model = scores(1).requires_grad_()

    value = kullback_leibler(distribution(reference), distribution(model))
    value.backward()

    assert math.isfinite(float(value.detach()))
    assert model.grad is not None
    assert float(model.grad[1, 2, 3]) == 0.0
    assert bool(torch.isfinite(model.grad).all())


@pytest.mark.parametrize(
    "kernel",
    [kullback_leibler, cross_entropy, fenchel_young],
    ids=("kullback-leibler", "cross-entropy", "fenchel-young"),
)
def test_masking_a_node_subtracts_exactly_that_node_and_leaves_its_gradient_at_zero(
    kernel: Callable[..., Tensor],
) -> None:
    reference = distribution(scores(0))
    keep = torch.ones(ITEMS, NODES, dtype=torch.bool)
    keep[0, 1] = False
    alone = ~keep

    full = float(kernel(reference, distribution(scores(1))))
    kept = float(kernel(reference, distribution(scores(1)), nodes=keep))
    dropped = float(kernel(reference, distribution(scores(1)), nodes=alone))

    assert kept + dropped == pytest.approx(full, rel=1e-6)

    model = scores(1).requires_grad_()
    kernel(reference, distribution(model), nodes=keep).backward()

    assert model.grad is not None
    assert float(model.grad[0, 1].abs().max()) == 0.0
    assert float(model.grad[0, 0].abs().max()) > 0.0


def test_a_masked_node_cannot_poison_the_total_with_its_own_infinity() -> None:
    reference = distribution(scores(0))
    model = scores(1)
    model[0, 1, 0] = -math.inf
    model = model.log_softmax(dim=2)
    model[0, 1, 0] = -math.inf
    model = model.detach().requires_grad_()
    keep = torch.ones(ITEMS, NODES, dtype=torch.bool)
    keep[0, 1] = False

    value = kullback_leibler(reference, distribution(model), nodes=keep)
    value.backward()

    assert math.isfinite(float(value.detach()))
    assert model.grad is not None
    assert float(model.grad[0, 1].abs().max()) == 0.0
    assert bool(torch.isfinite(model.grad).all())


def test_the_value_mask_and_the_node_mask_are_both_honored() -> None:
    padded = torch.ones(ITEMS, NODES, dtype=torch.bool)
    padded[1, 2] = False
    reference = distribution(scores(0), padded)
    model = distribution(scores(1), padded)
    asked = torch.ones(ITEMS, NODES, dtype=torch.bool)
    asked[0, 0] = False

    both = float(kullback_leibler(reference, model, nodes=asked))
    padding_only = float(kullback_leibler(reference, model))
    kept = padded & asked

    assert both == pytest.approx(
        float(
            kullback_leibler(
                distribution(scores(0)),
                distribution(scores(1)),
                nodes=kept,
            ),
        ),
        rel=1e-6,
    )
    assert both < padding_only


def test_the_node_sum_is_the_node_count_times_the_per_node_average() -> None:
    counted = 3
    keep = torch.zeros(ITEMS, NODES, dtype=torch.bool)
    keep.view(-1)[:counted] = True
    reference = distribution(scores(0))
    model = distribution(scores(1))

    summed = float(kullback_leibler(reference, model, nodes=keep))
    each = [
        float(
            kullback_leibler(
                reference,
                model,
                nodes=keep & torch.arange(ITEMS * NODES).reshape(ITEMS, NODES).eq(one),
            ),
        )
        for one in range(counted)
    ]
    average = sum(each) / counted

    assert summed == pytest.approx(sum(each), rel=1e-6)
    assert summed == pytest.approx(counted * average, rel=1e-6)


def test_the_fenchel_young_loss_at_alpha_one_is_the_forward_divergence() -> None:
    reference, model = distribution(scores(0)), distribution(scores(1))

    assert float(fenchel_young(reference, model, alpha=1.0)) == pytest.approx(
        float(kullback_leibler(reference, model)),
    )


@pytest.mark.parametrize(
    ("divergence", "kernel"),
    [
        (KL, kullback_leibler),
        (CrossEntropy(), cross_entropy),
        (FenchelYoung(ALPHA), fenchel_young),
    ],
    ids=("kullback-leibler", "cross-entropy", "fenchel-young"),
)
def test_a_declared_divergence_dispatches_to_its_kernel(
    divergence: Divergence,
    kernel: Callable[..., Tensor],
) -> None:
    reference, model = distribution(scores(0)), distribution(scores(1))

    assert float(node_divergence(divergence, reference, model)) == pytest.approx(
        float(kernel(reference, model)),
    )


def test_two_distributions_over_different_axes_are_refused() -> None:
    reference = distribution(scores(0))
    model = LogDistribution(scores(1), ("item", "cell", "tag"), normalized_over="tag")

    with pytest.raises(ValueError, match="over the axes"):
        kullback_leibler(reference, model)


def test_two_distributions_of_different_shapes_are_refused() -> None:
    reference = distribution(scores(0))
    model = LogDistribution(
        scores(1)[:, :, :2].log_softmax(dim=2),
        AXES,
        normalized_over="symbol",
    )

    with pytest.raises(ValueError, match="has shape"):
        kullback_leibler(reference, model)


def test_a_value_that_normalizes_over_nothing_is_refused() -> None:
    reference = distribution(scores(0))
    model = LogDistribution(scores(1), AXES)

    with pytest.raises(ValueError, match="normalizes over no axis"):
        kullback_leibler(reference, model)


def test_a_symbol_axis_that_is_not_the_innermost_is_refused() -> None:
    moved = scores(0).permute(0, 2, 1).contiguous()
    outer = LogDistribution(
        moved,
        ("item", "symbol", "node"),
        normalized_over="symbol",
    )

    with pytest.raises(ValueError, match="is not the innermost"):
        kullback_leibler(outer, outer)


def test_a_node_mask_that_is_not_boolean_is_refused() -> None:
    reference = distribution(scores(0))

    with pytest.raises(TypeError, match="node mask holds booleans"):
        kullback_leibler(reference, reference, nodes=torch.ones(ITEMS, NODES))


def test_a_node_mask_that_does_not_cover_the_node_axes_is_refused() -> None:
    reference = distribution(scores(0))
    wrong = torch.ones(ITEMS, NODES + 1, dtype=torch.bool)

    with pytest.raises(ValueError, match="does not cover the"):
        kullback_leibler(reference, reference, nodes=wrong)


def test_a_node_mask_deeper_than_the_node_axes_is_refused() -> None:
    reference = distribution(scores(0))
    deep = torch.ones(ITEMS, NODES, SYMBOLS, dtype=torch.bool)

    with pytest.raises(ValueError, match="rank is between"):
        kullback_leibler(reference, reference, nodes=deep)


def weights_by_hand(
    left: torch.Tensor,
    right: torch.Tensor,
    weights: torch.Tensor,
    node: Callable[[list[float], list[float]], float],
) -> float:
    flat = [
        float(weights[item, position])
        for item in range(ITEMS)
        for position in range(NODES)
    ]

    return sum(
        weight * node(p, q)
        for weight, p, q in zip(flat, rows(left), rows(right), strict=True)
    )


@pytest.mark.parametrize(
    ("kernel", "node"),
    [
        (kullback_leibler, kl_node),
        (cross_entropy, cross_node),
        (fenchel_young, bregman_node),
    ],
    ids=("kullback-leibler", "cross-entropy", "fenchel-young"),
)
def test_a_weighted_sum_counts_each_node_at_its_weight(
    kernel: Callable[..., Tensor],
    node: Callable[[list[float], list[float]], float],
) -> None:
    generator = torch.Generator().manual_seed(7)
    weights = torch.rand(ITEMS, NODES, generator=generator)

    value = kernel(distribution(scores(0)), distribution(scores(1)), weights=weights)

    assert float(value) == pytest.approx(
        weights_by_hand(scores(0), scores(1), weights, node), rel=1e-5
    )


def test_a_weight_over_the_outermost_axis_alone_scales_every_node_of_the_item() -> None:
    reference, model = distribution(scores(0)), distribution(scores(1))
    per_item = torch.tensor([2.0, 0.5])
    spread = per_item.unsqueeze(1).expand(ITEMS, NODES)

    assert float(kullback_leibler(reference, model, weights=per_item)) == pytest.approx(
        float(kullback_leibler(reference, model, weights=spread))
    )


def test_unit_weights_leave_the_node_sum_alone() -> None:
    reference, model = distribution(scores(0)), distribution(scores(1))

    assert float(
        kullback_leibler(reference, model, weights=torch.ones(ITEMS, NODES))
    ) == pytest.approx(float(kullback_leibler(reference, model)))


def test_the_gradient_reaches_the_weights() -> None:
    reference, model = distribution(scores(0)), distribution(scores(1))
    weights = torch.ones(ITEMS, NODES).requires_grad_()

    kullback_leibler(reference, model, weights=weights).backward()

    assert weights.grad is not None
    assert float(weights.grad[0, 0]) == pytest.approx(
        kl_node(rows(scores(0))[0], rows(scores(1))[0]), rel=1e-5
    )


def test_a_node_weighted_zero_cannot_poison_the_total_with_its_own_infinity() -> None:
    reference = distribution(scores(0))
    model = scores(1)
    model[0, 1, 0] = -math.inf
    model = model.log_softmax(dim=2)
    model[0, 1, 0] = -math.inf
    model = model.detach().requires_grad_()
    weights = torch.ones(ITEMS, NODES)
    weights[0, 1] = 0.0
    weights = weights.requires_grad_()
    keep = torch.ones(ITEMS, NODES, dtype=torch.bool)
    keep[0, 1] = False

    value = kullback_leibler(reference, distribution(model), weights=weights)
    value.backward()

    assert float(value.detach()) == pytest.approx(
        float(kullback_leibler(reference, distribution(model.detach()), nodes=keep))
    )
    assert model.grad is not None
    assert weights.grad is not None
    assert float(model.grad[0, 1].abs().max()) == 0.0
    assert bool(torch.isfinite(model.grad).all())
    assert bool(torch.isfinite(weights.grad).all())


def test_a_weighted_masked_node_contributes_nothing() -> None:
    reference, model = distribution(scores(0)), distribution(scores(1))
    keep = torch.ones(ITEMS, NODES, dtype=torch.bool)
    keep[0, 1] = False
    weights = torch.full((ITEMS, NODES), 3.0)

    kept = float(kullback_leibler(reference, model, nodes=keep, weights=weights))

    assert kept == pytest.approx(
        3.0 * float(kullback_leibler(reference, model, nodes=keep))
    )


def test_a_declared_divergence_passes_the_weights_through() -> None:
    reference, model = distribution(scores(0)), distribution(scores(1))
    weights = torch.full((ITEMS, NODES), 2.0)

    for divergence in (KL, CrossEntropy(), FenchelYoung(ALPHA)):
        assert float(
            node_divergence(divergence, reference, model, weights=weights)
        ) == pytest.approx(2.0 * float(node_divergence(divergence, reference, model)))


def test_weights_that_are_not_floating_point_are_refused() -> None:
    reference = distribution(scores(0))

    with pytest.raises(TypeError, match="floating-point"):
        kullback_leibler(
            reference, reference, weights=torch.ones(ITEMS, NODES, dtype=torch.int64)
        )


def test_weights_that_do_not_cover_the_node_axes_are_refused() -> None:
    reference = distribution(scores(0))

    with pytest.raises(ValueError, match="does not cover the"):
        kullback_leibler(reference, reference, weights=torch.ones(ITEMS, NODES + 1))


def test_weights_deeper_than_the_node_axes_are_refused() -> None:
    reference = distribution(scores(0))

    with pytest.raises(ValueError, match="rank is between"):
        kullback_leibler(
            reference, reference, weights=torch.ones(ITEMS, NODES, SYMBOLS)
        )
