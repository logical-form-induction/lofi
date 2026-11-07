import math
from typing import TYPE_CHECKING, cast, override

import pytest
import torch

from lofi.deduce import LogInside as SymbolicLogInside
from lofi.deduce import SpanIndex
from lofi.nn.parsers._channel import SplitChannel, log_split_posterior
from lofi.nn.parsers._inside import InsideKernel
from lofi.nn.parsers._ports import LogWeightReduction
from lofi.nn.parsers._reduction import LogSumExpReduction
from lofi.nn.parsers._weigher import AttendSplitWeigher, MarginalSplitWeigher
from lofi.nn.parsers.functions import SpanFunctionModule

if TYPE_CHECKING:
    from collections.abc import Callable

    from lofi.nn.parsers._channel import Composition
    from lofi.nn.parsers._values import ScoredChart

NEG_INF = float("-inf")

#: Absolute tolerance for probabilities.
TOLERANCE = 1e-6

#: Minimum measurable distribution difference.
SEPARATION = 0.05


class DotScore(SpanFunctionModule):
    """Score a child pair by its inner product."""

    def __init__(self, width: int) -> None:
        super().__init__((width, width), 1)

    @override
    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        """Return one inner-product logit per child pair."""
        self.joined(inputs)
        return (inputs[0] * inputs[1]).sum(dim=-1, keepdim=True)


def marginal_channel(
    compose: Composition,
    *,
    reduction: LogWeightReduction | None = None,
    checkpoint_from_width: int | None = None,
) -> SplitChannel:
    """Return a channel using the declared default split weigher."""
    return SplitChannel(
        compose,
        weigher=MarginalSplitWeigher(),
        reduction=LogSumExpReduction() if reduction is None else reduction,
        checkpoint_from_width=checkpoint_from_width,
    )


def scored(
    index: SpanIndex, seed: int, batch: int = 1, width: int = 3
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    vectors = torch.randn(
        batch, index.n, width, generator=generator, dtype=torch.float64
    )
    leaves = torch.randn(batch, index.n, generator=generator, dtype=torch.float64)
    splits = torch.randn(
        batch,
        index.num_spans,
        index.max_splits,
        generator=generator,
        dtype=torch.float64,
    )

    return (vectors, leaves, splits)


def composer(seed: int, width: int) -> Composition:
    generator = torch.Generator().manual_seed(seed)
    left_map = torch.randn(width, width, generator=generator, dtype=torch.float64)
    right_map = torch.randn(width, width, generator=generator, dtype=torch.float64)

    def compose(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return torch.tanh(left @ left_map + 2.0 * (right @ right_map))

    return compose


def trees(
    index: SpanIndex, leaves: torch.Tensor, splits: torch.Tensor, i: int, j: int
) -> list[tuple[float, int]]:
    if j - i == 1:
        return [(leaves[i].item(), i)]

    found = []

    for k in range(i + 1, j):
        edge = splits[index.span_id(i, j), k - i - 1].item()
        found += [
            (left + right + edge, k)
            for left, _ in trees(index, leaves, splits, i, k)
            for right, _ in trees(index, leaves, splits, k, j)
        ]

    return found


def symbolic_total(scores: list[float]) -> float:
    reduction = SymbolicLogInside()
    total = reduction.zero

    for score in scores:
        total = reduction.plus(total, score)

    return total


def reference_alpha(
    index: SpanIndex, leaves: torch.Tensor, splits: torch.Tensor, i: int, j: int
) -> list[float]:
    found = trees(index, leaves, splits, i, j)
    total = symbolic_total([score for score, _ in found])

    return [
        math.exp(symbolic_total([score for score, top in found if top == k]) - total)
        for k in range(i + 1, j)
    ]


def reference_cells(
    index: SpanIndex,
    vectors: torch.Tensor,
    leaves: torch.Tensor,
    splits: torch.Tensor,
    compose: Composition,
) -> dict[tuple[int, int], torch.Tensor]:
    cells = {(i, i + 1): vectors[i] for i in range(index.n)}

    for i, j in index.spans_by_length(min_length=2):
        weights = reference_alpha(index, leaves, splits, i, j)
        total = torch.zeros_like(vectors[0])

        for offset, k in enumerate(index.splits(i, j)):
            pair = compose(cells[i, k].unsqueeze(0), cells[k, j].unsqueeze(0)).squeeze(
                0
            )
            total = total + weights[offset] * pair

        cells[i, j] = total

    return cells


def attention_sweep(
    index: SpanIndex, vectors: torch.Tensor, compose: Composition
) -> tuple[torch.Tensor, torch.Tensor]:
    cells = {(i, i + 1): vectors[i] for i in range(index.n)}
    weights = torch.full((index.num_spans, index.max_splits), NEG_INF).to(vectors.dtype)

    for i, j in index.spans_by_length(min_length=2):
        pairs = [
            (cells[i, k], cells[k, j], compose(cells[i, k][None], cells[k, j][None])[0])
            for k in index.splits(i, j)
        ]
        share = torch.stack([(left * right).sum() for left, right, _ in pairs]).softmax(
            dim=0
        )
        weights[index.span_id(i, j), : j - i - 1] = share
        cells[i, j] = sum(
            (share[offset] * pair for offset, (_, _, pair) in enumerate(pairs)),
            start=torch.zeros_like(vectors[i]),
        )

    return (
        weights,
        torch.stack([cells[index.span_of(sid)] for sid in range(index.num_spans)]),
    )


def total_variation(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).abs().sum().item() / 2.0)


class CountingLogSumExp:
    def __init__(self) -> None:
        self.rows = 0
        self._inner = LogSumExpReduction()

    def reduce(
        self, weights: torch.Tensor, mask: torch.Tensor, /, *, dim: int
    ) -> torch.Tensor:
        self.rows += weights.numel() // int(weights.shape[dim])

        return self._inner.reduce(weights, mask, dim=dim)

    @property
    def probabilistic(self) -> bool:
        return self._inner.probabilistic

    def mixing_weights(
        self, weights: torch.Tensor, mask: torch.Tensor, /, *, dim: int
    ) -> torch.Tensor:
        return self._inner.mixing_weights(weights, mask, dim=dim)

    def validate_expected_counts(self) -> None:
        self._inner.validate_expected_counts()


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_the_posterior_matches_enumeration_over_every_binary_tree(n: int) -> None:
    index = SpanIndex(n)
    _, leaves, splits = scored(index, seed=n)

    inside = InsideKernel(LogSumExpReduction()).inside(leaves, splits, index)
    alpha = log_split_posterior(inside, splits, index).exp()

    for i, j in index.spans_by_length(min_length=2):
        found = alpha[0, index.span_id(i, j), : j - i - 1].tolist()
        assert found == pytest.approx(
            reference_alpha(index, leaves[0], splits[0], i, j), abs=TOLERANCE
        )


def test_the_posterior_matches_the_gradient_of_the_log_partition() -> None:
    index = SpanIndex(6)
    draws = 20

    for seed in range(draws):
        _, leaves, splits = scored(index, seed=seed)
        splits.requires_grad_()

        kernel = InsideKernel(LogSumExpReduction())
        kernel.log_partition(leaves, splits, index).sum().backward()
        inside = kernel.inside(leaves, splits.detach(), index)
        alpha = log_split_posterior(inside, splits.detach(), index).exp()

        assert splits.grad is not None
        counts = splits.grad[0]
        mu = counts.sum(dim=-1, keepdim=True)
        expected = counts[mu.squeeze(-1) > 0.0] / mu[mu > 0.0].unsqueeze(-1)
        found = alpha[0][mu.squeeze(-1) > 0.0]

        assert found.reshape(-1).tolist() == pytest.approx(
            expected.reshape(-1).tolist(), abs=1e-9
        )


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_the_posterior_sums_to_one_over_the_real_splits_of_every_span(n: int) -> None:
    index = SpanIndex(n)
    _, leaves, splits = scored(index, seed=n + 100)

    inside = InsideKernel(LogSumExpReduction()).inside(leaves, splits, index)
    alpha = log_split_posterior(inside, splits, index).exp()

    for i, j in index.spans_by_length():
        row = alpha[0, index.span_id(i, j)]
        mass = row[: j - i - 1].sum().item()

        assert mass == pytest.approx(1.0 if j - i > 1 else 0.0, abs=TOLERANCE)
        assert row[j - i - 1 :].tolist() == [0.0] * (index.max_splits - (j - i - 1))
        assert row[: j - i].sum().item() == pytest.approx(mass, abs=TOLERANCE)


def test_a_padded_split_weighs_negative_infinity_rather_than_a_small_number() -> None:
    index = SpanIndex(4)
    _, leaves, splits = scored(index, seed=21)

    inside = InsideKernel(LogSumExpReduction()).inside(leaves, splits, index)
    alpha = log_split_posterior(inside, splits, index)

    assert alpha[0, index.span_id(0, 1)].tolist() == [NEG_INF] * index.max_splits
    assert alpha[0, index.span_id(0, 2), 1:].tolist() == [NEG_INF] * (
        index.max_splits - 1
    )


def test_a_span_nothing_derives_has_no_posterior_and_no_nan_gradient() -> None:
    index = SpanIndex(3)
    leaves = torch.zeros(1, index.n, requires_grad=True)
    splits = torch.full(
        (1, index.num_spans, index.max_splits), NEG_INF, requires_grad=True
    )

    inside = InsideKernel(LogSumExpReduction()).inside(leaves, splits, index)
    alpha = log_split_posterior(inside, splits, index)
    alpha.exp().sum().backward()

    assert alpha[0, index.span_id(0, 3)].tolist() == [NEG_INF, NEG_INF]
    assert leaves.grad is not None
    assert splits.grad is not None
    assert bool(torch.isfinite(leaves.grad).all())
    assert bool(torch.isfinite(splits.grad).all())


def test_a_one_token_input_has_an_empty_posterior() -> None:
    index = SpanIndex(1)
    _, leaves, splits = scored(index, seed=1)

    alpha = log_split_posterior(
        InsideKernel(LogSumExpReduction()).inside(leaves, splits, index), splits, index
    )

    assert alpha.shape == (1, 1, 0)


def test_the_posterior_refuses_a_chart_or_scores_it_cannot_read() -> None:
    index = SpanIndex(3)
    _, leaves, splits = scored(index, seed=2)
    inside = InsideKernel(LogSumExpReduction()).inside(leaves, splits, index)

    with pytest.raises(TypeError, match="an inside chart holds log weights"):
        log_split_posterior(inside.long(), splits, index)

    with pytest.raises(TypeError, match="split scores are log weights"):
        log_split_posterior(inside, splits.long(), index)

    with pytest.raises(ValueError, match="rank 2"):
        log_split_posterior(inside[0], splits, index)

    with pytest.raises(ValueError, match="rank 3"):
        log_split_posterior(inside, splits[0], index)

    with pytest.raises(ValueError, match="an inside chart of shape"):
        log_split_posterior(inside[:, :-1], splits, index)

    with pytest.raises(ValueError, match="split scores of shape"):
        log_split_posterior(inside, splits[:, :-1], index)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
def test_the_scalar_channel_is_the_kernel_sweep_under_log_sum_exp(
    n: int,
) -> None:
    index = SpanIndex(n)
    vectors, leaves, splits = scored(index, seed=n + 200)

    chart = marginal_channel(composer(n, 3)).sweep(vectors, leaves, splits, index)

    assert torch.equal(
        chart.inside, InsideKernel(LogSumExpReduction()).inside(leaves, splits, index)
    )


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_the_root_score_matches_the_symbolic_log_inside_partition(n: int) -> None:
    index = SpanIndex(n)
    vectors, leaves, splits = scored(index, seed=n + 300)

    chart = marginal_channel(composer(n, 3)).sweep(vectors, leaves, splits, index)
    goal = chart.inside[0, index.span_id(0, n)].item()

    assert goal == pytest.approx(
        symbolic_total(
            [score for score, _ in trees(index, leaves[0], splits[0], 0, n)]
        ),
        abs=1e-5,
    )


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_a_cell_is_the_expectation_of_its_composed_children_under_the_posterior(
    n: int,
) -> None:
    index = SpanIndex(n)
    vectors, leaves, splits = scored(index, seed=n + 400)
    compose = composer(n, 3)

    chart = marginal_channel(compose).sweep(vectors, leaves, splits, index)
    reference = reference_cells(index, vectors[0], leaves[0], splits[0], compose)

    for i, j in index.spans_by_length():
        found = chart.cells[index.span_id(i, j), 0]
        assert found.tolist() == pytest.approx(reference[i, j].tolist(), abs=TOLERANCE)


def test_a_constant_composition_leaves_every_derived_cell_at_that_constant() -> None:
    index = SpanIndex(5)
    vectors, leaves, splits = scored(index, seed=41)
    constant = torch.tensor([2.0, -1.0, 0.5], dtype=torch.float64)

    chart = marginal_channel(
        lambda left, _: constant.expand_as(left),
    ).sweep(vectors, leaves, splits, index)

    for i, j in index.spans_by_length(min_length=2):
        assert chart.cells[index.span_id(i, j), 0].tolist() == pytest.approx(
            constant.tolist(), abs=TOLERANCE
        )


def test_the_leaves_enter_the_chart_as_the_first_cells() -> None:
    index = SpanIndex(4)
    vectors, leaves, splits = scored(index, seed=43, batch=2)

    chart = marginal_channel(composer(1, 3)).sweep(vectors, leaves, splits, index)

    for i in range(index.n):
        assert torch.equal(chart.cells[index.span_id(i, i + 1)], vectors[:, i])


def test_the_posterior_differs_from_a_weight_read_off_the_child_vectors() -> None:
    index = SpanIndex(5)
    vectors, leaves, splits = scored(index, seed=47)
    compose = composer(5, 3)

    chart = marginal_channel(compose).sweep(vectors, leaves, splits, index)
    attention, _ = attention_sweep(index, vectors[0], compose)
    apart = {
        (i, j): total_variation(
            chart.alpha[0, index.span_id(i, j), : j - i - 1].exp(),
            attention[index.span_id(i, j), : j - i - 1],
        )
        for i, j in index.spans_by_length(min_length=2)
    }
    ambiguous = [gap for (i, j), gap in apart.items() if j - i > 2]

    # a width-two span has only one possible split
    assert all(
        apart[i, j] == 0.0 for i, j in index.spans_by_length(min_length=2) if j - i == 2
    )
    assert sum(ambiguous) / len(ambiguous) > SEPARATION


def test_only_the_posterior_moves_when_a_grandchild_becomes_less_likely() -> None:
    index = SpanIndex(4)
    vectors, leaves, splits = scored(index, seed=53)
    compose = composer(4, 3)
    moved = splits.clone()
    moved[0, index.span_id(1, 3), 0] -= 3.0

    channel = marginal_channel(compose)
    before = channel.sweep(vectors, leaves, splits, index)
    after = channel.sweep(vectors, leaves, moved, index)
    root = index.span_id(0, 4)
    attention, _ = attention_sweep(index, vectors[0], compose)

    assert not torch.allclose(before.alpha[0, root], after.alpha[0, root])
    assert torch.equal(attention, attention_sweep(index, vectors[0], compose)[0])


def test_a_gradient_reaches_the_vectors_and_both_score_tensors() -> None:
    index = SpanIndex(4)
    vectors, leaves, splits = scored(index, seed=59)
    vectors.requires_grad_()
    leaves.requires_grad_()
    splits.requires_grad_()

    chart = marginal_channel(composer(4, 3)).sweep(vectors, leaves, splits, index)
    chart.cells[index.span_id(0, 4)].sum().backward()

    assert vectors.grad is not None
    assert leaves.grad is not None
    assert splits.grad is not None
    assert bool(torch.isfinite(vectors.grad).all())
    assert bool(torch.isfinite(splits.grad).all())
    assert splits.grad.abs().sum().item() > 0.0
    assert leaves.grad.abs().sum().item() > 0.0


def test_an_item_is_swept_as_though_it_had_been_swept_alone() -> None:
    index = SpanIndex(4)
    vectors, leaves, splits = scored(index, seed=61, batch=3)
    channel = marginal_channel(composer(4, 3))

    together = channel.sweep(vectors, leaves, splits, index)

    for item in range(3):
        alone = channel.sweep(
            vectors[item : item + 1],
            leaves[item : item + 1],
            splits[item : item + 1],
            index,
        )

        assert together.inside[item].tolist() == pytest.approx(alone.inside[0].tolist())
        assert together.cells[:, item].reshape(-1).tolist() == pytest.approx(
            alone.cells[:, 0].reshape(-1).tolist()
        )
        assert together.alpha[item].reshape(-1).exp().tolist() == pytest.approx(
            alone.alpha[0].reshape(-1).exp().tolist()
        )


def test_a_one_token_input_keeps_its_leaf_as_the_whole_chart() -> None:
    index = SpanIndex(1)
    vectors, leaves, splits = scored(index, seed=67)

    chart = marginal_channel(composer(1, 3)).sweep(vectors, leaves, splits, index)

    assert chart.cells.shape == (1, 1, 3)
    assert chart.cells[0, 0].tolist() == vectors[0, 0].tolist()
    assert chart.inside[0, 0].item() == pytest.approx(leaves[0, 0].item())


def test_the_cells_keep_the_precision_the_leaf_vectors_arrived_in() -> None:
    index = SpanIndex(3)
    vectors, leaves, splits = scored(index, seed=71)

    chart = marginal_channel(lambda left, right: left + right).sweep(
        vectors.to(torch.float32), leaves, splits, index
    )

    assert chart.cells.dtype is torch.float32
    assert chart.inside.dtype is torch.float64
    assert chart.alpha.dtype is torch.float64


def test_the_scalar_channel_costs_one_float_per_cell() -> None:
    index = SpanIndex(8)
    width = 64
    batch = 4
    vectors, leaves, splits = scored(index, seed=73, batch=batch, width=width)

    chart = marginal_channel(composer(8, width)).sweep(vectors, leaves, splits, index)
    scalar = chart.inside.numel() * chart.inside.element_size()
    vector = chart.cells.numel() * chart.cells.element_size()

    assert chart.inside.numel() == batch * index.num_spans
    assert scalar * width == vector


def test_the_scalar_channel_reduces_one_row_per_cell_that_divides() -> None:
    index = SpanIndex(8)
    batch = 4
    vectors, leaves, splits = scored(index, seed=79, batch=batch)
    counting = CountingLogSumExp()

    assert isinstance(counting, LogWeightReduction)

    swept = InsideKernel(counting).inside(leaves, splits, index)
    chart = marginal_channel(composer(8, 3)).sweep(vectors, leaves, splits, index)

    assert torch.equal(swept, chart.inside)
    assert counting.rows == batch * (index.num_spans - index.n)


def test_the_composition_is_called_once_per_width_over_a_run_of_pairs() -> None:
    index = SpanIndex(5)
    batch = 2
    vectors, leaves, splits = scored(index, seed=83, batch=batch)
    inner = composer(5, 3)
    seen: list[tuple[int, ...]] = []

    def counted(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        seen.append(tuple(left.shape))

        return inner(left, right)

    marginal_channel(counted).sweep(vectors, leaves, splits, index)

    assert len(seen) == index.n - 1
    assert seen == [
        (batch * (width - 1) * (index.n - width + 1), 3)
        for width in range(2, index.n + 1)
    ]


def test_the_channel_publishes_a_sweep_and_nothing_grammatical() -> None:
    published = {name for name in dir(SplitChannel) if not name.startswith("_")}

    assert published == {
        "checkpoint_from_width",
        "compose",
        "reduction",
        "sweep",
        "sweep_scoring",
        "weigher",
    }


def test_the_channel_reports_what_composes_a_pair_and_holds_nothing_else() -> None:
    compose = composer(1, 3)
    channel = marginal_channel(compose)

    assert channel.compose is compose
    assert not hasattr(channel, "__dict__")


def test_something_that_cannot_compose_a_pair_of_cells_is_refused() -> None:
    with pytest.raises(TypeError, match="composed by a callable"):
        marginal_channel(cast("Composition", torch.zeros(3)))


def test_leaf_vectors_the_index_does_not_fit_are_refused() -> None:
    index = SpanIndex(3)
    vectors, leaves, splits = scored(index, seed=89)
    channel = marginal_channel(composer(3, 3))

    with pytest.raises(TypeError, match="a cell holds a vector"):
        channel.sweep(vectors.long(), leaves, splits, index)

    with pytest.raises(ValueError, match="rank 3"):
        channel.sweep(vectors[0], leaves, splits, index)

    with pytest.raises(ValueError, match="that many leaf vectors"):
        channel.sweep(vectors[:, :-1], leaves, splits, index)

    with pytest.raises(ValueError, match="items came with scores for"):
        channel.sweep(vectors.expand(2, index.n, 3), leaves, splits, index)


def test_a_composition_returning_the_wrong_shape_is_refused() -> None:
    index = SpanIndex(3)
    vectors, leaves, splits = scored(index, seed=97)

    with pytest.raises(ValueError, match="one vector each"):
        marginal_channel(lambda left, _: left[:, :-1]).sweep(
            vectors, leaves, splits, index
        )


def lookup(index: SpanIndex, splits: torch.Tensor) -> tuple[Composition, list[int]]:
    widths: list[int] = []

    def score(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        del right
        width = int(left.shape[2]) + 1
        widths.append(width)
        start = index.span_id(0, width)

        return splits[:, start : start + index.n - width + 1, : width - 1]

    return (score, widths)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 6])
def test_scoring_the_sweep_by_lookup_is_the_sweep_over_those_scores(n: int) -> None:
    index = SpanIndex(n)
    batch = 2
    vectors, leaves, splits = scored(index, seed=n + 500, batch=batch)
    compose = composer(n, 3)
    score, widths = lookup(index, splits)

    given = marginal_channel(compose).sweep(vectors, leaves, splits, index)
    chart, read = marginal_channel(compose).sweep_scoring(
        vectors, leaves, score, index, recompute=True
    )

    assert widths == list(range(2, n + 1))
    assert torch.allclose(chart.inside, given.inside)
    assert torch.allclose(chart.alpha, given.alpha)
    assert torch.allclose(chart.cells, given.cells)
    assert chart.index is index
    real = torch.tensor(index.split_mask, dtype=torch.bool).unsqueeze(0)

    assert torch.equal(read, splits.masked_fill(~real, NEG_INF))


def test_attend_scores_child_pairs_at_each_intermediate_width() -> None:
    index = SpanIndex(5)
    vectors, leaves, splits = scored(index, seed=550)
    score, _ = lookup(index, splits)
    compose = composer(5, 3)
    channel = SplitChannel(compose, weigher=AttendSplitWeigher(DotScore(3)))

    chart, read = channel.sweep_scoring(vectors, leaves, score, index, recompute=True)
    expected_weights, expected_cells = attention_sweep(index, vectors[0], compose)
    real = torch.tensor(index.split_mask, dtype=torch.bool).unsqueeze(0)

    assert torch.equal(read, splits.masked_fill(~real, NEG_INF))
    torch.testing.assert_close(
        chart.alpha[0].exp(),
        expected_weights.masked_fill(expected_weights.isneginf(), 0.0),
    )
    torch.testing.assert_close(chart.cells[:, 0], expected_cells)


def test_the_scoring_sees_the_cells_the_sweep_built_for_the_children() -> None:
    index = SpanIndex(5)
    vectors, leaves, splits = scored(index, seed=600, batch=2)
    compose = composer(5, 3)
    seen: list[tuple[torch.Tensor, torch.Tensor]] = []
    inner, _ = lookup(index, splits)

    def score(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        seen.append((left, right))

        return inner(left, right)

    chart, _ = marginal_channel(compose).sweep_scoring(
        vectors, leaves, score, index, recompute=True
    )

    for width, (left, right) in enumerate(seen, start=2):
        assert tuple(left.shape) == (2, index.n - width + 1, width - 1, 3)
        assert tuple(right.shape) == tuple(left.shape)

        for row, (i, j) in enumerate(index.spans_by_length(min_length=width)):
            if j - i != width:
                break

            for column, k in enumerate(index.splits(i, j)):
                assert torch.equal(
                    left[:, row, column], chart.cells[index.span_id(i, k)]
                )
                assert torch.equal(
                    right[:, row, column], chart.cells[index.span_id(k, j)]
                )


def test_a_gradient_reaches_the_scores_the_scoring_returned() -> None:
    index = SpanIndex(4)
    vectors, leaves, splits = scored(index, seed=700)
    splits.requires_grad_()
    score, _ = lookup(index, splits)

    chart, _ = marginal_channel(composer(4, 3)).sweep_scoring(
        vectors, leaves, score, index, recompute=True
    )
    chart.inside[0, index.span_id(0, 4)].backward()

    assert splits.grad is not None
    assert splits.grad[0].masked_select(
        torch.tensor(index.split_mask)
    ).sum().item() == (pytest.approx(index.n - 1, abs=TOLERANCE))


def test_a_scoring_returning_the_wrong_shape_or_dtype_is_refused() -> None:
    index = SpanIndex(3)
    vectors, leaves, _ = scored(index, seed=800)
    channel = marginal_channel(composer(3, 3))

    with pytest.raises(ValueError, match="one log weight each"):
        channel.sweep_scoring(
            vectors,
            leaves,
            lambda left, _: left.sum(dim=(-1, -2)),
            index,
            recompute=True,
        )

    with pytest.raises(TypeError, match="floating point"):
        channel.sweep_scoring(
            vectors,
            leaves,
            lambda left, _: left.sum(dim=-1).long(),
            index,
            recompute=True,
        )


def test_the_scoring_sweep_refuses_what_the_plain_sweep_refuses() -> None:
    index = SpanIndex(3)
    vectors, leaves, splits = scored(index, seed=900)
    score, _ = lookup(index, splits)
    channel = marginal_channel(composer(3, 3))

    with pytest.raises(ValueError, match="that many leaf vectors"):
        channel.sweep_scoring(vectors[:, :-1], leaves, score, index, recompute=True)

    with pytest.raises(ValueError, match="items came with scores for"):
        channel.sweep_scoring(
            vectors.expand(2, index.n, 3), leaves, score, index, recompute=True
        )

    with pytest.raises(ValueError, match="at least one"):
        channel.sweep_scoring(
            torch.zeros(1, 0, 3), torch.zeros(1, 0), score, SpanIndex(0), recompute=True
        )


def bilinear_scoring(seed: int, width: int) -> Composition:
    generator = torch.Generator().manual_seed(seed)
    form = torch.randn(width, width, generator=generator, dtype=torch.float64)

    def score(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return torch.tanh(torch.einsum("ispw,wv,ispv->isp", left, form, right))

    return score


def held_bytes(run: Callable[[], torch.Tensor]) -> tuple[torch.Tensor, int]:
    """Return what a forward pass returns and the bytes autograd held for backward."""
    held = 0

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        nonlocal held
        held += tensor.numel() * tensor.element_size()

        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        result = run()

    return result, held


def checkpointed_sweep(
    threshold: int | None, *, recompute: bool = True
) -> tuple[ScoredChart, int, tuple[torch.Tensor, torch.Tensor]]:
    index = SpanIndex(9)
    vectors, leaves, _ = scored(index, seed=900, batch=2, width=6)
    vectors.requires_grad_()
    leaves.requires_grad_()
    channel = marginal_channel(composer(9, 6), checkpoint_from_width=threshold)
    score = bilinear_scoring(9, 6)
    swept: list[ScoredChart] = []

    def run() -> torch.Tensor:
        chart = channel.sweep_scoring(
            vectors, leaves, score, index, recompute=recompute
        )
        swept.append(chart)

        return chart.chart.inside.sum() + chart.chart.cells.square().sum()

    total, held = held_bytes(run)
    total.backward()

    assert vectors.grad is not None
    assert leaves.grad is not None

    return swept[0], held, (vectors.grad, leaves.grad)


@pytest.mark.parametrize("threshold", [2, 5, 9])
def test_checkpointed_widths_hold_fewer_activations_and_change_no_value(
    threshold: int,
) -> None:
    held_chart, held, held_grads = checkpointed_sweep(None)
    chart, recomputed, grads = checkpointed_sweep(threshold)

    assert recomputed < held
    assert torch.equal(chart.chart.inside, held_chart.chart.inside)
    assert torch.equal(chart.chart.cells, held_chart.chart.cells)
    assert torch.equal(chart.chart.alpha, held_chart.chart.alpha)
    assert torch.equal(chart.split_scores, held_chart.split_scores)
    assert all(
        torch.allclose(found, expected, rtol=0.0, atol=1e-12)
        for found, expected in zip(grads, held_grads, strict=True)
    )


@pytest.mark.parametrize(
    ("threshold", "recomposed"), [(None, 0), (9, 1), (5, 5), (2, 8)]
)
def test_a_sweep_over_given_split_scores_recomposes_from_the_declared_width(
    threshold: int | None, recomposed: int
) -> None:
    index = SpanIndex(9)
    vectors, leaves, splits = scored(index, seed=902, batch=2, width=6)
    vectors.requires_grad_()
    compose = composer(9, 6)
    calls = 0

    def counted(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1

        return compose(left, right)

    reference = marginal_channel(compose).sweep(vectors, leaves, splits, index).cells
    (expected,) = torch.autograd.grad(reference.square().sum(), vectors)
    channel = marginal_channel(counted, checkpoint_from_width=threshold)
    cells = channel.sweep(vectors, leaves, splits, index).cells
    (gradient,) = torch.autograd.grad(cells.square().sum(), vectors)

    assert calls == index.n - 1 + recomposed
    assert torch.equal(cells, reference)
    assert torch.allclose(gradient, expected, rtol=0.0, atol=1e-12)


def test_a_lower_threshold_holds_no_more_than_a_higher_one() -> None:
    held = [checkpointed_sweep(threshold)[1] for threshold in (8, 5, 2)]

    assert held[0] > held[1] > held[2]


def test_a_scoring_that_may_not_run_again_is_never_checkpointed() -> None:
    _, held, _ = checkpointed_sweep(None)
    _, unrepeated, _ = checkpointed_sweep(2, recompute=False)

    assert unrepeated == held


def test_a_sweep_without_gradients_is_the_same_under_any_threshold() -> None:
    index = SpanIndex(6)
    vectors, leaves, _ = scored(index, seed=901, batch=2, width=4)
    score = bilinear_scoring(6, 4)

    with torch.no_grad():
        charts = [
            marginal_channel(
                composer(6, 4), checkpoint_from_width=threshold
            ).sweep_scoring(vectors, leaves, score, index, recompute=True)
            for threshold in (None, 2)
        ]

    assert torch.equal(charts[0].chart.cells, charts[1].chart.cells)
    assert torch.equal(charts[0].chart.inside, charts[1].chart.inside)


def test_checkpointing_below_the_first_composed_width_is_refused() -> None:
    with pytest.raises(ValueError, match="at least two"):
        marginal_channel(composer(3, 3), checkpoint_from_width=1)

    assert (
        marginal_channel(composer(3, 3), checkpoint_from_width=3).checkpoint_from_width
        == 3
    )
    assert marginal_channel(composer(3, 3)).checkpoint_from_width is None
