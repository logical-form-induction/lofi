import math

import pytest
import torch

from lofi.deduce import Cell, SpanIndex
from lofi.nn.parsers._chart import TensorChart


def numbered_chart(n: int) -> TensorChart:
    index = SpanIndex(n)
    chart = TensorChart(index, torch.zeros(()), batch=1, width=1)

    for i, j in index.spans_by_length():
        chart.write(i, j, torch.full((1, 1), float(index.span_id(i, j))))

    return chart


def enumerate_trees(
    index: SpanIndex, leaves: list[float], edges: dict[tuple[int, int, int], float]
) -> dict[Cell, list[float]]:
    totals: dict[Cell, list[float]] = {(i, i + 1): [leaves[i]] for i in range(index.n)}

    for i, j in index.spans_by_length(min_length=2):
        scores = [
            left + right + edges[(i, j, k)]
            for k in range(i + 1, j)
            for left in totals[(i, k)]
            for right in totals[(k, j)]
        ]
        totals[(i, j)] = scores

    return totals


def sum_product_through_the_chart(
    index: SpanIndex, leaves: list[float], edges: dict[tuple[int, int, int], float]
) -> torch.Tensor:
    chart = TensorChart(index, torch.zeros((), dtype=torch.float64), batch=1, width=1)

    for i in range(index.n):
        chart.write(i, i + 1, torch.full((1, 1), leaves[i], dtype=torch.float64))

    for i, j in index.spans_by_length(min_length=2):
        left, right = chart.read_children(i, j)
        alternatives = torch.stack(
            [
                left[k - i - 1, 0, 0]
                + right[k - i - 1, 0, 0]
                + torch.tensor(edges[(i, j, k)], dtype=torch.float64)
                for k in index.splits(i, j)
            ]
        )
        chart.write(i, j, torch.logsumexp(alternatives, dim=0).reshape(1, 1))

    return chart.read(0, index.n)[0, 0]


def scored(
    index: SpanIndex, seed: int
) -> tuple[list[float], dict[tuple[int, int, int], float]]:
    generator = torch.Generator().manual_seed(seed)
    draws = torch.randn(64, generator=generator, dtype=torch.float64).tolist()
    leaves = [draws.pop() for _ in range(index.n)]
    edges = {
        (i, j, k): draws.pop()
        for i, j in index.spans_by_length(min_length=2)
        for k in index.splits(i, j)
    }

    return (leaves, edges)


def test_a_chart_takes_its_placement_and_precision_from_the_input() -> None:
    chart = TensorChart(
        SpanIndex(3), torch.zeros((), dtype=torch.float64), batch=2, width=4
    )

    assert chart.dtype is torch.float64
    assert chart.read(0, 3).dtype is torch.float64
    assert chart.read(0, 3).device == torch.zeros(()).device


def test_a_chart_follows_an_input_onto_a_second_device() -> None:
    chart = TensorChart(SpanIndex(3), torch.zeros((), device="meta"), batch=2, width=4)
    left, right = chart.read_children(0, 3)

    assert chart.read(0, 3).device.type == "meta"
    assert left.device.type == "meta"
    assert right.device.type == "meta"


def test_a_chart_reports_the_shape_of_what_it_holds() -> None:
    index = SpanIndex(5)
    chart = TensorChart(index, torch.zeros(()), batch=3, width=7)

    assert chart.index == index
    assert chart.batch == 3
    assert chart.width == 7
    assert chart.read(0, 5).shape == (3, 7)


def test_a_cell_nothing_has_written_reads_as_zeros() -> None:
    chart = TensorChart(SpanIndex(3), torch.zeros(()), batch=2, width=4)

    assert torch.equal(chart.read(1, 3), torch.zeros(2, 4))


def test_a_written_cell_reads_back() -> None:
    chart = TensorChart(SpanIndex(3), torch.zeros(()), batch=2, width=2)
    value = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    chart.write(1, 3, value)

    assert torch.equal(chart.read(1, 3), value)
    assert torch.equal(chart.read(0, 2), torch.zeros(2, 2))


def test_read_hands_back_a_copy_rather_than_the_store() -> None:
    chart = TensorChart(SpanIndex(3), torch.zeros(()), batch=1, width=2)
    chart.write(0, 3, torch.tensor([[1.0, 2.0]]))

    got = chart.read(0, 3)
    got.add_(100.0)

    assert torch.equal(chart.read(0, 3), torch.tensor([[1.0, 2.0]]))


def test_read_children_hands_back_copies_rather_than_the_store() -> None:
    chart = numbered_chart(3)

    left, right = chart.read_children(0, 3)
    left.add_(100.0)
    right.add_(100.0)

    assert chart.read(0, 1).item() == 0.0
    assert chart.read(1, 3).item() == 4.0


def test_the_write_a_caller_keeps_is_no_longer_the_cell() -> None:
    chart = TensorChart(SpanIndex(3), torch.zeros(()), batch=1, width=1)
    value = torch.zeros(1, 1)

    chart.write(0, 3, value)
    value.add_(5.0)

    assert chart.read(0, 3).item() == 0.0


def test_read_children_gathers_the_cells_the_span_index_names() -> None:
    index = SpanIndex(4)
    chart = numbered_chart(4)
    root = index.span_id(0, 4)

    left, right = chart.read_children(0, 4)

    assert [left[c, 0, 0].item() for c in range(index.max_splits)] == [
        float(child) for child in index.left_child_ids[root]
    ]
    assert [right[c, 0, 0].item() for c in range(index.max_splits)] == [
        float(child) for child in index.right_child_ids[root]
    ]


def test_a_padding_split_reads_the_cell_its_padding_id_names() -> None:
    index = SpanIndex(4)
    chart = numbered_chart(4)
    real = index.split_mask[index.span_id(1, 3)]

    left, _ = chart.read_children(1, 3)

    assert list(real) == [True, False, False]
    assert left[0, 0, 0].item() == float(index.span_id(1, 2))
    assert left[1, 0, 0].item() == 0.0


def test_a_gradient_reaches_a_value_written_into_a_cell() -> None:
    chart = TensorChart(SpanIndex(2), torch.zeros(()), batch=1, width=2)
    value = torch.tensor([[1.0, 2.0]], requires_grad=True)

    chart.write(0, 2, value * 3.0)
    chart.read(0, 2).sum().backward()

    assert value.grad is not None
    assert value.grad.tolist() == [[3.0, 3.0]]


def test_a_gradient_reaches_a_cell_read_back_through_a_gather() -> None:
    chart = TensorChart(SpanIndex(3), torch.zeros(()), batch=1, width=1)
    leaf = torch.tensor([[2.0]], requires_grad=True)

    chart.write(0, 1, leaf)
    chart.write(1, 2, torch.ones(1, 1))
    chart.write(2, 3, torch.ones(1, 1))
    left, _ = chart.read_children(0, 3)
    left.sum().backward()

    assert leaf.grad is not None
    assert leaf.grad.item() == 1.0


def test_a_multiwidth_backward_matches_the_same_recurrence_without_a_store() -> None:
    index = SpanIndex(5)
    leaves = torch.randn(index.n, dtype=torch.float64, requires_grad=True)
    reference_leaves = leaves.detach().clone().requires_grad_()
    chart = TensorChart(index, leaves, batch=1, width=1)
    reference = {(i, i + 1): reference_leaves[i].reshape(1, 1) for i in range(index.n)}

    for i in range(index.n):
        chart.write(i, i + 1, leaves[i].reshape(1, 1))

    for width in range(2, index.n + 1):
        for i in range(index.n - width + 1):
            j = i + width
            left, right = chart.read_children(i, j)
            chart.write(i, j, (left[: width - 1] + right[: width - 1]).sum(0))
            reference[i, j] = sum(
                (reference[i, k] + reference[k, j] for k in range(i + 1, j)),
                start=torch.zeros(1, 1, dtype=torch.float64),
            )

    found = chart.read(0, index.n)
    expected = reference[0, index.n]
    found_gradient = torch.autograd.grad(found.sum(), leaves)[0]
    expected_gradient = torch.autograd.grad(expected.sum(), reference_leaves)[0]

    assert torch.allclose(found, expected, rtol=0.0, atol=1e-12)
    assert torch.allclose(found_gradient, expected_gradient, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_a_sweep_driven_through_the_chart_matches_direct_enumeration(n: int) -> None:
    index = SpanIndex(n)
    leaves, edges = scored(index, seed=n)

    got = sum_product_through_the_chart(index, leaves, edges)
    trees = enumerate_trees(index, leaves, edges)[(0, n)]

    assert got.item() == pytest.approx(math.log(sum(math.exp(s) for s in trees)))


@pytest.mark.parametrize(("i", "j"), [(2, 1), (1, 1), (0, 4), (-1, 2)])
def test_a_span_the_input_does_not_have_is_refused(i: int, j: int) -> None:
    chart = TensorChart(SpanIndex(3), torch.zeros(()), batch=1, width=1)

    with pytest.raises(IndexError, match="is not a span"):
        chart.read(i, j)

    with pytest.raises(IndexError, match="is not a span"):
        chart.read_children(i, j)

    with pytest.raises(IndexError, match="is not a span"):
        chart.write(i, j, torch.zeros(1, 1))


def test_a_value_that_is_not_the_shape_of_a_cell_is_refused() -> None:
    chart = TensorChart(SpanIndex(3), torch.zeros(()), batch=2, width=4)

    with pytest.raises(ValueError, match=r"shape \(2, 4\)"):
        chart.write(0, 3, torch.zeros(2, 3))


def test_an_input_with_no_spans_is_refused() -> None:
    with pytest.raises(ValueError, match="length at least one"):
        TensorChart(SpanIndex(0), torch.zeros(()), batch=1, width=1)


def test_an_integer_store_is_refused() -> None:
    with pytest.raises(TypeError, match="floating point"):
        TensorChart(SpanIndex(3), torch.zeros((), dtype=torch.long), batch=1, width=1)


@pytest.mark.parametrize(("batch", "width"), [(0, 1), (1, 0), (-1, 1)])
def test_a_chart_holding_nothing_is_refused(batch: int, width: int) -> None:
    with pytest.raises(ValueError, match="at least one"):
        TensorChart(SpanIndex(3), torch.zeros(()), batch=batch, width=width)


def test_a_chart_holds_only_the_fields_it_was_built_with() -> None:
    chart = TensorChart(SpanIndex(3), torch.zeros(()), batch=1, width=1)

    assert not hasattr(chart, "__dict__")
