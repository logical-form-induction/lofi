from typing import TYPE_CHECKING, Final, override

import pytest
import torch
from hypothesis import strategies as st

from lofi._checks import LawViolationError
from lofi.train._optimization import Optimization, OptInSet, Rewrite
from lofi.train.laws import (
    EvaluationFixture,
    autograd_status_is_invariant_across_opt_ins,
    check_evaluation_regime_laws,
    shape_and_dtype_are_invariant_across_opt_ins,
    the_reference_regime_is_declared,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor

type Row = tuple[float, ...]

_MEMO: Final = Optimization("memo", Rewrite.VALUE)

_REGIMES: Final = (OptInSet(), OptInSet([_MEMO]))

_EXAMPLES: Final = 20

_ROWS: Final = st.lists(
    st.floats(min_value=-4.0, max_value=4.0, allow_nan=False, allow_infinity=False),
    min_size=1,
    max_size=4,
).map(tuple)


class Scale(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))
        self._memo: dict[Row, Tensor] = {}

    def evaluate(self, values: Row, opt_ins: OptInSet) -> Tensor:
        """Return the scaled row, in evaluation mode, under one set of rewrites.

        Parameters
        ----------
        values : Row
            The numbers to scale.
        opt_ins : OptInSet
            The enabled value rewrites.
        """
        self.eval()

        if _MEMO not in opt_ins:
            return self.compute(values)

        cached = self._memo.get(values)

        if cached is None:
            computed = self.compute(values)
            self._memo[values] = computed

            return computed

        return self.serve(cached)

    def compute(self, values: Row) -> Tensor:
        """Return the scaled row, computed from the weight and the input.

        Parameters
        ----------
        values : Row
            The numbers to scale.
        """
        return self.weight * torch.tensor(values)

    def serve(self, cached: Tensor) -> Tensor:
        """Return what the memo stored for this input.

        Parameters
        ----------
        cached : Tensor
            The tensor the memo holds.
        """
        return cached


class DetachesOnAHit(Scale):
    @override
    def serve(self, cached: Tensor) -> Tensor:
        return cached.detach()


class DetachesWhileMemoizing(Scale):
    @override
    def evaluate(self, values: Row, opt_ins: OptInSet) -> Tensor:
        returned = super().evaluate(values, opt_ins)

        return returned.detach() if _MEMO in opt_ins else returned


class TrimsOnAHit(Scale):
    @override
    def serve(self, cached: Tensor) -> Tensor:
        return cached[:1]


class WidensOnAHit(Scale):
    @override
    def serve(self, cached: Tensor) -> Tensor:
        return cached.double()


def _fixture(
    component: Scale, opt_ins: Sequence[OptInSet] = _REGIMES
) -> EvaluationFixture[Row]:
    return EvaluationFixture(component.evaluate, _ROWS, opt_ins, _EXAMPLES)


def test_a_memo_that_serves_what_it_stored_passes_every_law() -> None:
    check_evaluation_regime_laws(_fixture(Scale()))


def test_a_component_that_offers_no_rewrite_passes_every_law() -> None:
    check_evaluation_regime_laws(_fixture(Scale(), (OptInSet(),)))


def test_a_memo_that_detaches_on_a_hit_is_caught() -> None:
    with pytest.raises(LawViolationError, match="autograd status"):
        autograd_status_is_invariant_across_opt_ins(_fixture(DetachesOnAHit()))


def test_a_component_that_stops_differentiating_when_memoizing_is_caught() -> None:
    with pytest.raises(LawViolationError, match="autograd status"):
        autograd_status_is_invariant_across_opt_ins(_fixture(DetachesWhileMemoizing()))


def test_a_memo_that_returns_less_than_it_stored_is_caught() -> None:
    with pytest.raises(LawViolationError, match="shape and dtype"):
        shape_and_dtype_are_invariant_across_opt_ins(_fixture(TrimsOnAHit()))


def test_a_memo_that_changes_precision_on_a_hit_is_caught() -> None:
    with pytest.raises(LawViolationError, match="shape and dtype"):
        shape_and_dtype_are_invariant_across_opt_ins(_fixture(WidensOnAHit()))


def test_a_memo_that_only_narrows_still_differentiates() -> None:
    autograd_status_is_invariant_across_opt_ins(_fixture(TrimsOnAHit()))


def test_a_component_declaring_no_unoptimized_regime_is_caught() -> None:
    with pytest.raises(LawViolationError, match="reference regime"):
        the_reference_regime_is_declared(_fixture(Scale(), (OptInSet([_MEMO]),)))


def test_the_detached_hit_and_the_differentiable_miss_agree_in_value() -> None:
    component = DetachesOnAHit()
    row: Row = (1.0, -2.0)

    miss = component.evaluate(row, OptInSet([_MEMO]))
    hit = component.evaluate(row, OptInSet([_MEMO]))

    assert torch.equal(miss, hit)
    assert miss.requires_grad
    assert not hit.requires_grad
