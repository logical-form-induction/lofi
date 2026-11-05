"""Inference relations derived from the two entailment questions."""

from typing import cast

import pytest

from lofi.lf import RELATION_LABELS, Abstain, CounterSatisfiable, Proof, Proved, Refuted
from lofi.lf._relation import (
    CLASSICAL,
    STRICT,
    Decided,
    InferenceCell,
    InferenceLabel,
    LabelStatus,
    Reward,
)


def _proved(text: str) -> Proved:
    return Proved(Proof("fixture", text))


def _refuted(text: str) -> Refuted:
    return Refuted(CounterSatisfiable("fixture", "CounterSatisfiable", text))


@pytest.mark.parametrize(
    ("positive", "negative", "cell"),
    [
        (_proved("positive"), _refuted("negative"), InferenceCell.ENTAILS),
        (_refuted("positive"), _proved("negative"), InferenceCell.CONTRADICTS),
        (_refuted("positive"), _refuted("negative"), InferenceCell.NEUTRAL),
        (_proved("positive"), _proved("negative"), InferenceCell.DEGENERATE),
    ],
)
def test_two_certified_questions_select_one_relation_cell(
    positive: Proved | Refuted,
    negative: Proved | Refuted,
    cell: InferenceCell,
) -> None:
    decided = Decided.from_verdicts(positive, negative)

    assert decided.cells == frozenset({cell})
    assert decided.complete


def test_an_abstention_retains_every_cell_consistent_with_the_evidence() -> None:
    positive_open = Decided.from_verdicts(
        Abstain("timed out", cause="timeout"), _refuted("negative")
    )
    both_open = Decided.from_verdicts(Abstain("first"), Abstain("second"))

    assert positive_open.cells == frozenset(
        {InferenceCell.ENTAILS, InferenceCell.NEUTRAL}
    )
    assert both_open.cells == frozenset(InferenceCell)
    assert not positive_open.complete
    assert not both_open.complete


def test_the_classical_map_counts_an_inconsistent_premise_as_entailing() -> None:
    inconsistent = Decided.from_verdicts(_proved("positive"), _proved("negative"))

    assert CLASSICAL.status(InferenceLabel.ENTAILS, inconsistent) is LabelStatus.HOLDS
    assert (
        CLASSICAL.status(InferenceLabel.CONTRADICTS, inconsistent) is LabelStatus.HOLDS
    )
    assert CLASSICAL.label(inconsistent) is InferenceLabel.ENTAILS


def test_the_strict_map_assigns_no_label_to_an_inconsistent_premise() -> None:
    inconsistent = Decided.from_verdicts(_proved("positive"), _proved("negative"))

    assert STRICT.label(inconsistent) is None
    assert all(
        STRICT.status(label, inconsistent) is LabelStatus.FAILS
        for label in RELATION_LABELS
    )


def test_partial_evidence_makes_a_label_open_until_its_status_is_settled() -> None:
    open_entailment = Decided.from_verdicts(_proved("positive"), Abstain("second"))

    assert STRICT.status(InferenceLabel.ENTAILS, open_entailment) is LabelStatus.OPEN
    assert STRICT.status(InferenceLabel.NEUTRAL, open_entailment) is LabelStatus.FAILS


@pytest.mark.parametrize(
    ("label", "decided", "value"),
    [
        (
            InferenceLabel.ENTAILS,
            Decided.from_verdicts(_proved("positive"), _refuted("negative")),
            1.0,
        ),
        (
            InferenceLabel.CONTRADICTS,
            Decided.from_verdicts(_proved("positive"), _refuted("negative")),
            0.0,
        ),
        (
            InferenceLabel.ENTAILS,
            Decided.from_verdicts(_proved("positive"), Abstain("negative")),
            None,
        ),
    ],
)
def test_a_reward_is_built_only_from_decided_status(
    label: InferenceLabel, decided: Decided, value: float | None
) -> None:
    reward = Reward.from_decided(label, decided, mapping=STRICT)

    assert reward.label is label
    assert reward.decided is decided
    assert reward.value == value


def test_relation_values_refuse_untyped_or_empty_inputs() -> None:
    with pytest.raises(TypeError, match="InferenceLabel"):
        Reward.from_decided(
            cast("InferenceLabel", "entails"), Decided.all(), mapping=STRICT
        )

    with pytest.raises(ValueError, match="at least one"):
        Decided(frozenset())

    with pytest.raises(ValueError, match="is not mapped"):
        CLASSICAL.status(InferenceLabel.NON_ENTAILS, Decided.all())
