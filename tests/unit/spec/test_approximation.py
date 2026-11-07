import json

import didactic.api as dx
import pytest

from lofi.spec.approximation import (
    ApproximationPlan,
    ApproximationRequest,
    ApproximationSpec,
)


def test_one_run_bound_resolves_every_consumer_and_records_only_overrides() -> None:
    plan = ApproximationPlan(
        ApproximationSpec(residual_mass=0.1),
        {
            "model.readouts.types": ApproximationRequest(None),
            "loss.terms.0": ApproximationRequest(0.01),
        },
    )

    assert plan.bound("model.readouts.types") == pytest.approx(0.1)
    assert plan.bound("loss.terms.0") == pytest.approx(0.01)
    assert plan.measurements()["approximation-residual-mass"] == pytest.approx(0.1)
    assert json.loads(str(plan.measurements()["approximation-overrides"])) == {
        "loss.terms.0": 0.01
    }


def test_a_consumer_without_any_bound_is_refused_by_structural_path() -> None:
    with pytest.raises(ValueError, match=r"model\.readouts\.types.*residual_mass"):
        ApproximationPlan(
            None,
            {"model.readouts.types": ApproximationRequest(None)},
        )


@pytest.mark.parametrize("value", [-0.1, 1.0, float("inf"), float("nan")])
def test_a_residual_mass_bound_is_a_finite_fraction_below_one(value: float) -> None:
    with pytest.raises(dx.ValidationError, match="residual mass"):
        ApproximationSpec(residual_mass=value)
