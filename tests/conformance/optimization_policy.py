from typing import TYPE_CHECKING, Final

import pytest

from lofi.train.laws import (
    a_nonfinite_loss_is_reported_without_a_step,
    an_amp_step_matches_the_recorded_order,
    gradient_accumulation_delays_the_optimizer_step,
    mixed_precision_off_skips_scaler_operations,
    the_policy_owns_gradient_zeroing,
    the_step_scale_is_read_before_update,
)

if TYPE_CHECKING:
    from lofi.train.laws import OptimizationFixture

__all__ = [
    "OPTIMIZATION_POLICY_CONFORMANCE_LAWS",
    "OptimizationPolicyConformance",
]


class OptimizationPolicyConformance:
    pytestmark = pytest.mark.conformance

    def test_an_amp_step_matches_the_recorded_order[LossT](
        self,
        conformance: OptimizationFixture[LossT],
    ) -> None:
        an_amp_step_matches_the_recorded_order(conformance)

    def test_the_step_scale_is_read_before_update[LossT](
        self,
        conformance: OptimizationFixture[LossT],
    ) -> None:
        the_step_scale_is_read_before_update(conformance)

    def test_mixed_precision_off_skips_scaler_operations[LossT](
        self,
        conformance: OptimizationFixture[LossT],
    ) -> None:
        mixed_precision_off_skips_scaler_operations(conformance)

    def test_a_nonfinite_loss_is_reported_without_a_step[LossT](
        self,
        conformance: OptimizationFixture[LossT],
    ) -> None:
        a_nonfinite_loss_is_reported_without_a_step(conformance)

    def test_gradient_accumulation_delays_the_optimizer_step[LossT](
        self,
        conformance: OptimizationFixture[LossT],
    ) -> None:
        gradient_accumulation_delays_the_optimizer_step(conformance)

    def test_the_policy_owns_gradient_zeroing[LossT](
        self,
        conformance: OptimizationFixture[LossT],
    ) -> None:
        the_policy_owns_gradient_zeroing(conformance)


OPTIMIZATION_POLICY_CONFORMANCE_LAWS: Final = (
    OptimizationPolicyConformance.test_an_amp_step_matches_the_recorded_order,
    OptimizationPolicyConformance.test_the_step_scale_is_read_before_update,
    OptimizationPolicyConformance.test_mixed_precision_off_skips_scaler_operations,
    OptimizationPolicyConformance.test_a_nonfinite_loss_is_reported_without_a_step,
    OptimizationPolicyConformance.test_gradient_accumulation_delays_the_optimizer_step,
    OptimizationPolicyConformance.test_the_policy_owns_gradient_zeroing,
)
