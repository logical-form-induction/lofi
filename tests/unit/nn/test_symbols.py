from typing import Final

import pytest
import torch

from lofi.nn.symbols import SymbolTable

SYMBOLS: Final = 5
WIDTH: Final = 4
CAP: Final = 1.5


def build(cap: float | None = CAP) -> SymbolTable:
    torch.manual_seed(11)

    return SymbolTable(symbols=SYMBOLS, width=WIDTH, cap=cap)


def planted(cap: float | None = CAP) -> SymbolTable:
    subject = build(cap)

    with torch.no_grad():
        subject.weight[0] = 0.0
        subject.weight[1] = torch.tensor([0.3, 0.0, 0.4, 0.0])
        subject.weight[2] = torch.tensor([3.0, 0.0, 4.0, 0.0])
        subject.weight[3] = torch.tensor([0.0, -1.5, 0.0, 0.0])
        subject.weight[4] = torch.tensor([-30.0, 40.0, 0.0, 0.0])

    return subject


def test_without_a_cap_the_rows_are_the_weight_itself() -> None:
    subject = planted(cap=None)

    assert subject.cap is None
    assert subject.rows is subject.weight


def test_a_row_past_the_cap_is_scaled_onto_it_and_one_within_it_is_not() -> None:
    subject = planted()
    rows = subject.rows

    assert torch.equal(rows[0], torch.zeros(WIDTH))
    assert torch.equal(rows[1], subject.weight[1])
    assert torch.allclose(rows[2], torch.tensor([0.9, 0.0, 1.2, 0.0]))
    assert torch.equal(rows[3], subject.weight[3])
    assert torch.allclose(rows[4], torch.tensor([-0.9, 1.2, 0.0, 0.0]))
    assert bool(rows.norm(dim=1).le(CAP + 1e-6).all())


def test_the_projection_changes_nothing_in_place() -> None:
    subject = planted()
    before = subject.weight.detach().clone()
    first = subject.rows
    second = subject.rows

    assert torch.equal(first, second)
    assert torch.equal(subject.weight, before)


def test_gradient_reaches_every_row_and_is_finite_at_a_zero_row() -> None:
    subject = planted()
    subject.rows.sum().backward()
    gradient = subject.weight.grad

    assert gradient is not None
    assert bool(gradient.isfinite().all())
    assert bool(gradient[1].ne(0.0).any())
    assert bool(gradient[2].ne(0.0).any())


def test_scaling_a_row_past_the_cap_further_does_not_move_what_is_read() -> None:
    subject = planted()
    once = subject.rows[2].detach().clone()

    with torch.no_grad():
        subject.weight[2] *= 100.0

    assert torch.allclose(subject.rows[2], once)


def test_the_table_reports_its_size_width_and_cap() -> None:
    subject = build()

    assert subject.symbols == SYMBOLS
    assert subject.width == WIDTH
    assert subject.cap == CAP
    assert tuple(subject.weight.shape) == (SYMBOLS, WIDTH)
    assert subject.extra_repr() == f"symbols={SYMBOLS}, width={WIDTH}, cap={CAP}"


@pytest.mark.parametrize("cap", [0.0, -1.0, float("inf"), float("nan")])
def test_a_cap_must_be_finite_and_above_zero(cap: float) -> None:
    with pytest.raises(ValueError, match="norm cap"):
        SymbolTable(symbols=SYMBOLS, width=WIDTH, cap=cap)


def test_a_cap_must_be_a_number() -> None:
    with pytest.raises(TypeError, match="norm cap"):
        SymbolTable(symbols=SYMBOLS, width=WIDTH, cap=True)


@pytest.mark.parametrize(
    ("symbols", "width", "message"),
    [(0, WIDTH, "size"), (SYMBOLS, 0, "width")],
)
def test_the_size_and_the_width_must_be_positive(
    symbols: int, width: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        SymbolTable(symbols=symbols, width=width)
