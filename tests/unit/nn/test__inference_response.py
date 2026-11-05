"""Behavioral tests for declared inference response models."""

import pytest
import torch

from lofi.lf import THREE_WAY, InferenceCell, InferenceLabel
from lofi.nn._inference_response import JUDGMENT_PLUGIN, resolve_response
from lofi.nn._values import InferenceSignal, LogDistribution
from lofi.nn.grounding import CellMass, EvidenceLedger, GradientPath, GroundedBatch
from lofi.spec.losses import JudgmentResponseSettings
from lofi.spec.losses.inference import (
    InferenceResponseDeclaration,
    LabelResponseSettings,
)


class Judgments:
    judgments = ((0, 0), (1, 1))


def grounded() -> GroundedBatch:
    probabilities = torch.tensor(
        [[0.5, 0.25, 0.25], [0.1, 0.2, 0.7]], dtype=torch.float64
    )
    decided = torch.tensor([True, False])
    return GroundedBatch(
        LogDistribution(
            probabilities.log(),
            ("pair", "label"),
            decided,
            normalized_over="label",
        ),
        labels=THREE_WAY.labels,
        complete=False,
        residual=torch.tensor([0.1, 0.4], dtype=torch.float64),
        decided=decided,
        cells=CellMass(torch.zeros(2, len(tuple(InferenceCell)), dtype=torch.float64)),
        gradient_path=GradientPath.SCORE_FUNCTION,
        baseline=None,
        auxiliary=None,
        ledger=EvidenceLedger.empty(),
        variance=torch.zeros(2, dtype=torch.float64),
        reward=None,
    )


def test_a_judgment_response_marginalizes_labels_through_its_instrument() -> None:
    response = JUDGMENT_PLUGIN.construct(
        JudgmentResponseSettings(
            (
                (InferenceLabel.ENTAILS, (0.8, 0.2)),
                (InferenceLabel.CONTRADICTS, (0.1, 0.9)),
                (InferenceLabel.NEUTRAL, (0.5, 0.5)),
            )
        )
    )

    result = response.evaluate(InferenceSignal.REWARD, grounded(), Judgments())

    assert result.pair_positions.tolist() == [0, 1]
    assert result.observed.tolist() == [True, False]
    assert result.rewards is not None
    assert result.rewards.tolist() == pytest.approx([0.55, 0.55])
    assert result.payments.tolist() == pytest.approx([0.55, 0.55])
    assert result.costs.tolist() == pytest.approx((-result.rewards).tolist())


def test_a_judgment_level_outside_the_declared_instrument_is_refused() -> None:
    response = JUDGMENT_PLUGIN.construct(
        JudgmentResponseSettings(
            (
                (InferenceLabel.ENTAILS, (1.0, 0.0)),
                (InferenceLabel.CONTRADICTS, (0.0, 1.0)),
                (InferenceLabel.NEUTRAL, (0.5, 0.5)),
            )
        )
    )

    class Outside:
        judgments = ((0, 2),)

    with pytest.raises(ValueError, match="outside"):
        response.evaluate(InferenceSignal.REWARD, grounded(), Outside())


def test_a_label_response_predicts_each_label_with_certainty() -> None:
    response = resolve_response(
        InferenceResponseDeclaration("label", LabelResponseSettings())
    )
    labels = THREE_WAY.labels

    assert response.response_rows(labels) == tuple(
        tuple(float(column == row) for column in range(len(labels)))
        for row in range(len(labels))
    )


def test_a_judgment_response_predicts_each_label_through_its_instrument() -> None:
    rows = ((0.8, 0.2), (0.5, 0.5), (0.1, 0.9))
    labels = THREE_WAY.labels
    response = resolve_response(
        InferenceResponseDeclaration(
            "judgment",
            JudgmentResponseSettings(tuple(zip(labels, rows, strict=True))),
        )
    )

    assert response.response_rows(labels) == rows

    with pytest.raises(ValueError, match="label inventory exactly"):
        response.response_rows(labels[:2])
