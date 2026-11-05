import pytest
import torch

from lofi.nn._values import Encoded, LogDistribution
from lofi.nn.loss._likelihood import live_rows, negative_log_likelihood, rows_of


def weights(mask: torch.Tensor | None = None) -> LogDistribution:
    scores = torch.tensor([-1.0, -2.0, -4.0], dtype=torch.float64).requires_grad_()
    scored = scores if mask is None else scores.masked_fill(~mask, -torch.inf)

    return LogDistribution(scored, ("term",), mask)


def test_the_likelihood_sums_every_row_where_nothing_is_masked() -> None:
    total, count = negative_log_likelihood(weights(), terms=3, what="the terms")

    assert total.item() == 7.0
    assert count == 3
    assert total.grad_fn is not None


def test_a_masked_row_is_neither_summed_nor_counted() -> None:
    mask = torch.tensor([True, False, True])

    total, count = negative_log_likelihood(weights(mask), terms=3, what="the terms")

    assert total.item() == 5.0
    assert count == 2
    assert torch.isfinite(total)


def test_a_batch_whose_rows_are_all_masked_counts_nothing() -> None:
    mask = torch.zeros(3, dtype=torch.bool)

    total, count = negative_log_likelihood(weights(mask), terms=3, what="the terms")

    assert total.item() == 0.0
    assert count == 0


def test_the_gradient_of_a_masked_row_is_zero() -> None:
    mask = torch.tensor([True, False, True])
    scores = torch.tensor([-1.0, -2.0, -4.0], dtype=torch.float64).requires_grad_()
    scored = LogDistribution(scores.masked_fill(~mask, -torch.inf), ("term",), mask)

    total, _ = negative_log_likelihood(scored, terms=3, what="the terms")
    total.backward()

    assert scores.grad is not None
    assert scores.grad.tolist() == [-1.0, 0.0, -1.0]


@pytest.mark.parametrize(
    ("axes", "shape"),
    [(("term",), (2,)), (("cell",), (3,)), (("term", "slot"), (3, 1))],
)
def test_weights_that_are_not_one_per_term_are_refused(
    axes: tuple[str, ...], shape: tuple[int, ...]
) -> None:
    scored = LogDistribution(torch.zeros(shape, dtype=torch.float64), axes)

    with pytest.raises(ValueError, match="3 terms"):
        negative_log_likelihood(scored, terms=3, what="the terms")


def test_rows_are_taken_in_the_order_given_with_their_mask() -> None:
    encoded = Encoded(
        torch.arange(6, dtype=torch.float64).reshape(3, 2),
        ("term", "width"),
        torch.tensor([True, False, True]),
    )

    taken = rows_of(encoded, torch.tensor([2, 0]))

    assert isinstance(taken, Encoded)
    assert taken.tensor.tolist() == [[4.0, 5.0], [0.0, 1.0]]
    assert taken.axes == ("term", "width")
    assert taken.mask is not None
    assert taken.mask.tolist() == [True, True]


def test_rows_of_an_unmasked_batch_stay_unmasked() -> None:
    encoded = Encoded(torch.zeros(3, 2), ("term", "width"))

    assert rows_of(encoded, torch.tensor([1])).mask is None


def test_live_rows_are_the_rows_every_mask_keeps() -> None:
    first = torch.tensor([True, True, False])
    second = torch.tensor([True, False, True])

    alone = live_rows((first, None))
    both = live_rows((first, second))

    assert live_rows((None, None)) is None
    assert live_rows(()) is None
    assert alone is not None
    assert alone.tolist() == first.tolist()
    assert both is not None
    assert both.tolist() == [True, False, False]
