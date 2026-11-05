import copy

import pytest
import torch
from tests.verdicts import proved

from lofi.control_variate import MovingControlVariate
from lofi.lf import THREE_WAY, InferenceCell
from lofi.lf.provers import ServiceReport
from lofi.nn._values import LogDistribution
from lofi.nn.grounding import (
    CellMass,
    EvidenceLedger,
    GradientPath,
    GroundedBatch,
    GroundingTelemetry,
)


def test_a_grounded_batch_carries_one_distribution_and_one_decision_per_pair() -> None:
    probabilities = torch.tensor([[0.5, 0.25, 0.25], [0.2, 0.3, 0.5]]).log()
    decided = torch.tensor([True, False])
    baseline = MovingControlVariate(decay=0.5, initial=0.0)
    batch = GroundedBatch(
        LogDistribution(
            probabilities,
            ("pair", "label"),
            decided,
            normalized_over="label",
        ),
        labels=THREE_WAY.labels,
        complete=False,
        residual=torch.tensor([0.1, 1.0], requires_grad=True),
        decided=decided,
        cells=CellMass(torch.tensor([[0.5, 0.25, 0.25, 0.0], [0.0, 0.0, 0.0, 0.0]])),
        gradient_path=GradientPath.SCORE_FUNCTION,
        baseline=baseline,
        auxiliary=None,
        ledger=EvidenceLedger.from_evidence((proved("p"),)),
        variance=torch.tensor([0.2, 0.0], requires_grad=True),
        reward=None,
    )

    assert batch.log_probabilities.tensor is probabilities
    assert not batch.complete
    assert batch.decided is decided
    assert not batch.residual.requires_grad
    assert batch.cells.order == tuple(InferenceCell)
    assert batch.gradient_path is GradientPath.SCORE_FUNCTION
    assert batch.ledger.calls == 1
    assert batch.variance is not None
    assert not batch.variance.requires_grad
    batch.observe_payments(torch.tensor([0.5, 1.0]))
    assert baseline.value == pytest.approx(0.375)
    assert copy.copy(batch) is batch


def test_grounded_batch_rejects_disagreeing_pair_axes() -> None:
    with pytest.raises(ValueError, match="one residual per pair"):
        GroundedBatch(
            LogDistribution(
                torch.tensor([[0.5, 0.25, 0.25]]).log(),
                ("pair", "label"),
                normalized_over="label",
            ),
            labels=THREE_WAY.labels,
            complete=True,
            residual=torch.zeros(2),
            decided=torch.ones(1, dtype=torch.bool),
            cells=CellMass(torch.tensor([[0.5, 0.25, 0.25, 0.0]])),
            gradient_path=GradientPath.PATHWISE,
            baseline=None,
            auxiliary=None,
            ledger=EvidenceLedger.empty(),
            variance=None,
            reward=None,
        )


def test_gradient_paths_include_the_mixed_estimator() -> None:
    assert tuple(GradientPath) == (
        GradientPath.PATHWISE,
        GradientPath.SCORE_FUNCTION,
        GradientPath.MIXED,
    )


def test_an_evidence_ledger_preserves_the_service_step_telemetry() -> None:
    ledger = EvidenceLedger.from_service(
        (proved("p"),),
        ServiceReport(3, 1, {}, 0.25, 4, 2.0, 2),
    )

    assert ledger.calls == 3
    assert ledger.memo_hits == 1
    assert ledger.memo_hit_rate == 0.25
    assert ledger.questions == 4
    assert ledger.wall_seconds == 2.0
    assert ledger.questions_per_second == 2.0
    assert ledger.process_starts == 2


def test_an_evidence_ledger_preserves_custom_abstention_causes() -> None:
    ledger = EvidenceLedger.from_service(
        (),
        ServiceReport(2, 0, {"license-expired": 2}, 0.0, 2, 0.5, None),
    )

    assert ledger.abstentions["license-expired"] == 2
    assert ledger.abstentions["gave-up"] == 0


def test_grounding_telemetry_checks_and_preserves_out_of_fragment_share() -> None:
    def telemetry(share: float) -> GroundingTelemetry:
        return GroundingTelemetry(
            GradientPath.PATHWISE,
            {"out-of-fragment": 1},
            calls=1,
            questions=4,
            questions_per_second=8.0,
            memo_hits=0,
            memo_hit_rate=0.0,
            cpu_seconds=0.1,
            wall_seconds=0.5,
            process_starts=None,
            label_agreement=0.75,
            degenerate_mass=0.0,
            malformed_mass=0.0,
            reward_mean=None,
            variance=None,
            verified_share=0.5,
            out_of_fragment_share=share,
        )

    assert telemetry(0.25).out_of_fragment_share == 0.25

    with pytest.raises(ValueError, match="probability"):
        telemetry(1.25)
