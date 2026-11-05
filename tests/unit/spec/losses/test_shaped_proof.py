"""Tests for the shaped proof-credit declaration."""

import didactic.api as dx
import pytest

from lofi.lf.provers import ProofBudget
from lofi.spec import Dependency
from lofi.spec.components.data import TrainingBatchKind
from lofi.spec.components.loss import LossSpec
from lofi.spec.losses.shaped_proof import (
    ProofShapingSettings,
    ProofShapingSpec,
    ShapedProofLossSpec,
)
from lofi.spec.provers import ProofBudgetSpec


def declaration() -> ShapedProofLossSpec:
    """Return one complete shaped proof-credit declaration."""
    return ShapedProofLossSpec(
        readout="logical-forms",
        shaping=ProofShapingSpec(
            ill_formed=0.0,
            well_formed=0.25,
            type_checks=0.5,
            proved=1.0,
            length=0.05,
        ),
        budget=ProofBudgetSpec(
            calls_per_step=8,
            wall_per_step=2.0,
            cpu_seconds=10.0,
        ),
    )


def test_the_shaped_proof_variant_declares_readout_credit_and_budget() -> None:
    declared = LossSpec.model_validate(declaration().model_dump())

    assert isinstance(declared, ShapedProofLossSpec)
    settings = declared.settings()

    assert declared == declaration()
    assert declared.dependencies() == (
        Dependency("logical-forms", "shaped-proof-readout"),
    )
    assert declared.required_training_batches() == frozenset(
        {TrainingBatchKind.GOLD_FORMS}
    )
    assert settings.readout == "logical-forms"
    assert settings.shaping == ProofShapingSettings(0.0, 0.25, 0.5, 1.0, 0.05)
    assert settings.budget == ProofBudget(
        calls_per_step=8,
        wall_per_step=2.0,
        cpu_seconds=10.0,
    )


def test_a_shaped_proof_readout_is_named() -> None:
    with pytest.raises(dx.ValidationError, match="readout"):
        declaration().with_(readout="")


def test_a_shaped_proof_length_cost_is_nonnegative() -> None:
    with pytest.raises(dx.ValidationError, match="length cost"):
        ProofShapingSpec(
            ill_formed=0.0,
            well_formed=0.25,
            type_checks=0.5,
            proved=1.0,
            length=-0.1,
        )


def test_shaped_proof_credits_never_decrease() -> None:
    with pytest.raises(dx.ValidationError, match="do not decrease"):
        ProofShapingSpec(
            ill_formed=0.0,
            well_formed=0.25,
            type_checks=0.1,
            proved=1.0,
            length=0.0,
        )
