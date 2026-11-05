"""Tests for declared proof-service budgets."""

import didactic.api as dx
import pytest

from lofi.lf.provers import ProofBudget
from lofi.spec.provers import ProofBudgetSpec


def test_a_proof_budget_declaration_builds_the_interior_budget() -> None:
    """Every declared bound reaches its sole interior owner."""
    declared = ProofBudgetSpec(
        calls_per_step=0,
        wall_per_step=3.0,
        cpu_seconds=7.5,
    )

    assert declared.build() == ProofBudget(
        calls_per_step=0,
        wall_per_step=3.0,
        cpu_seconds=7.5,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("calls_per_step", -1),
        ("wall_per_step", -1.0),
        ("wall_per_step", float("nan")),
        ("cpu_seconds", -1.0),
        ("cpu_seconds", float("inf")),
    ],
)
def test_a_proof_budget_declaration_refuses_an_invalid_bound(
    field: str, value: float
) -> None:
    """A boundary refuses negative or non-finite resource bounds."""
    with pytest.raises(dx.ValidationError):
        ProofBudgetSpec.model_validate({field: value})
