import pytest
import torch
from tests.verdicts import proved, refuted

from lofi.nn.grounding import EvidenceLedger, PairScorer, Scores


class Scorer:
    scorer_kind = "fixture"
    source_identity = "fixture"
    fragment = object()
    differentiable = False
    certified = True
    proof_metadata = None

    def score(self, questions: object) -> Scores[object]:
        del questions
        return Scores(
            torch.tensor([1.0, 0.0]),
            torch.tensor([True, True]),
            (proved("p"), refuted("c")),
            EvidenceLedger.empty(),
        )

    def open_step(self) -> None:
        pass


def test_a_pair_scorer_returns_aligned_values_decisions_and_evidence() -> None:
    scorer = Scorer()
    scores = scorer.score(object())

    assert isinstance(scorer, PairScorer)
    assert scores.values.tolist() == [1.0, 0.0]
    assert scores.decided.tolist() == [True, True]
    assert len(scores.evidence) == 2


def test_scores_reject_a_value_outside_the_unit_interval() -> None:
    with pytest.raises(ValueError, match="unit interval"):
        Scores(
            torch.tensor([1.1]),
            torch.tensor([True]),
            (proved("p"),),
            EvidenceLedger.empty(),
        )
