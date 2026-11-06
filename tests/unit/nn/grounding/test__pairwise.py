from types import MappingProxyType
from typing import TYPE_CHECKING, Final, override

import pytest
import torch
from tests.language import language
from tests.verdicts import proved, refuted

from lofi._types import ReadoutId
from lofi.control_variate import MovingControlVariate
from lofi.lf import CLASSICAL, RTE, THREE_WAY, ArgumentRenaming, InferenceLabel
from lofi.nn.forms import Estimate, WeightedTerms
from lofi.nn.grounding import (
    EvidenceLedger,
    GradientPath,
    GroundingInput,
    GroundingQuery,
    Marginal,
    PairScorer,
    PairwiseGrounding,
    Phase,
    Scores,
)
from lofi.reward import InferenceReward

if TYPE_CHECKING:
    from lofi.lf import Verdict
    from lofi.lf.provers import Questions
    from lofi.terms import Term

LANGUAGE: Final = language(("rain", "wet", "cold"))
RAIN: Final = LANGUAGE.constant("rain")
WET: Final = LANGUAGE.constant("wet")
COLD: Final = LANGUAGE.constant("cold")
NOT: Final = LANGUAGE.constant("not")


class Pairs:
    premises = (ReadoutId("premise"),)
    hypotheses = (ReadoutId("hypothesis"),)
    alignments = ((ArgumentRenaming.identity(),),)
    labels = (InferenceLabel.ENTAILS,)


class TwoPairs:
    premises = (ReadoutId("first-premise"), ReadoutId("second-premise"))
    hypotheses = (ReadoutId("first-hypothesis"), ReadoutId("second-hypothesis"))
    alignments = ((ArgumentRenaming.identity(),),) * 2
    labels = (InferenceLabel.ENTAILS,) * 2


class JudgmentPairs:
    premises = (ReadoutId("premise"),)
    hypotheses = (ReadoutId("hypothesis"),)
    alignments = ((ArgumentRenaming.identity(),),)

    @property
    def labels(self) -> tuple[InferenceLabel, ...]:
        msg = "full judgment grounding must not read proof labels"
        raise AssertionError(msg)


class LogicalScorer:
    scorer_kind = "fixture"
    source_identity = "fixture"
    fragment = LANGUAGE.profile.fragment
    differentiable = False
    certified = True
    proof_metadata = None

    def score(self, questions: Questions) -> Scores[Verdict]:
        values: list[float] = []
        evidence = []

        for premise, goal in zip(questions.premises, questions.goals, strict=True):
            holds = premise == RAIN and goal == WET
            values.append(float(holds))
            evidence.append(proved("rain entails wet") if holds else refuted("no"))

        return Scores(
            torch.tensor(values),
            torch.ones(len(values), dtype=torch.bool),
            tuple(evidence),
            EvidenceLedger.from_evidence(evidence),
        )

    def open_step(self) -> None:
        pass


class ExplicitTwoQuestionScorer(LogicalScorer):
    positive_questions_only = False

    def __init__(self) -> None:
        self.received: Questions | None = None

    def validate_labels(self, labels: tuple[InferenceLabel, ...]) -> None:
        del labels
        msg = "a false positive-only capability must not validate labels"
        raise AssertionError(msg)

    @override
    def score(self, questions: Questions) -> Scores[Verdict]:
        self.received = questions
        return super().score(questions)


def terms(*forms: Term, estimate: Estimate = Estimate.EXACT) -> WeightedTerms:
    weight = torch.full((len(forms),), 1.0 / len(forms)).log().requires_grad_()
    return WeightedTerms(forms, weight, torch.tensor(-torch.inf), estimate)


def weighted_terms(
    forms: tuple[Term, ...], probabilities: tuple[float, ...]
) -> WeightedTerms:
    weights = torch.tensor(probabilities, dtype=torch.float64).log().requires_grad_()
    return WeightedTerms(forms, weights, torch.tensor(-torch.inf), Estimate.SAMPLED)


def grounding(*, baseline: MovingControlVariate | None = None) -> PairwiseGrounding:
    scorer = LogicalScorer()
    assert isinstance(scorer, PairScorer)
    return PairwiseGrounding(
        LANGUAGE.store,
        scorer,
        negation=NOT,
        application=LANGUAGE.calculus.app,
        labels=THREE_WAY.labels,
        label_map=CLASSICAL,
        contradiction="negated-hypothesis",
        composition=Marginal(),
        baseline=baseline,
        reward=None,
    )


def test_pairwise_grounding_batches_pairs_and_publishes_a_label_distribution() -> None:
    batch = GroundingInput(
        MappingProxyType(
            {
                ReadoutId("premise"): terms(RAIN),
                ReadoutId("hypothesis"): terms(WET),
            }
        ),
        choice_rows=MappingProxyType({}),
        pairs=Pairs(),
        query=GroundingQuery.FULL,
        sentences=None,
        phase=Phase.EVALUATE,
    )

    result = grounding().ground(batch)

    assert result.complete
    assert result.decided.tolist() == [True]
    assert result.gradient_path is GradientPath.PATHWISE
    assert result.log_probabilities.tensor.exp()[0].tolist() == pytest.approx(
        [1.0, 0.0, 0.0]
    )
    assert result.residual.tolist() == pytest.approx([0.0])
    assert grounding().grounding_metadata.scorer == "fixture"
    assert grounding().grounding_metadata.source == "fixture"


def test_pairwise_grounding_publishes_the_declared_binary_labeling() -> None:
    subject = PairwiseGrounding(
        LANGUAGE.store,
        LogicalScorer(),
        negation=NOT,
        application=LANGUAGE.calculus.app,
        labels=RTE.labels,
        label_map=RTE.label_map(CLASSICAL),
        contradiction="negated-hypothesis",
        composition=Marginal(),
        baseline=None,
        reward=None,
    )
    batch = GroundingInput(
        MappingProxyType(
            {
                ReadoutId("premise"): terms(COLD),
                ReadoutId("hypothesis"): terms(WET),
            }
        ),
        choice_rows=MappingProxyType({}),
        pairs=Pairs(),
        query=GroundingQuery.FULL,
        sentences=None,
        phase=Phase.EVALUATE,
    )

    result = subject.ground(batch)

    assert result.labels == RTE.labels
    assert result.log_probabilities.tensor.exp()[0].tolist() == pytest.approx(
        [0.0, 1.0]
    )


def test_false_positive_only_capability_keeps_both_entailment_questions() -> None:
    scorer = ExplicitTwoQuestionScorer()
    subject = PairwiseGrounding(
        LANGUAGE.store,
        scorer,
        negation=NOT,
        application=LANGUAGE.calculus.app,
        labels=RTE.labels,
        label_map=RTE.label_map(CLASSICAL),
        contradiction="negated-hypothesis",
        composition=Marginal(),
        baseline=None,
        reward=None,
    )
    batch = GroundingInput(
        MappingProxyType(
            {
                ReadoutId("premise"): terms(RAIN),
                ReadoutId("hypothesis"): terms(WET),
            }
        ),
        choice_rows=MappingProxyType({}),
        pairs=Pairs(),
        query=GroundingQuery.FULL,
        sentences=None,
        phase=Phase.EVALUATE,
    )

    subject.ground(batch)

    assert scorer.received is not None
    assert scorer.received.premises == (RAIN, RAIN)
    assert scorer.received.goals == (WET, LANGUAGE.read("not(wet)"))


def test_sampled_forms_select_the_score_function_path() -> None:
    batch = GroundingInput(
        MappingProxyType(
            {
                ReadoutId("premise"): terms(RAIN, COLD, estimate=Estimate.SAMPLED),
                ReadoutId("hypothesis"): terms(WET, COLD, estimate=Estimate.SAMPLED),
            }
        ),
        choice_rows=MappingProxyType({}),
        pairs=Pairs(),
        query=GroundingQuery.OBSERVED_LABEL,
        sentences=None,
        phase=Phase.TRAIN,
    )

    result = grounding(baseline=MovingControlVariate(decay=0.9, initial=0.0)).ground(
        batch
    )

    assert not result.complete
    assert result.gradient_path is GradientPath.SCORE_FUNCTION
    result.log_probabilities.tensor[0, 0].backward()
    assert batch.forms[ReadoutId("premise")].log_weights.grad is not None


def test_full_response_grounding_asks_both_questions_without_reading_labels() -> None:
    batch = GroundingInput(
        MappingProxyType(
            {
                ReadoutId("premise"): terms(RAIN, estimate=Estimate.SAMPLED),
                ReadoutId("hypothesis"): terms(WET, estimate=Estimate.SAMPLED),
            }
        ),
        choice_rows=MappingProxyType({}),
        pairs=JudgmentPairs(),
        query=GroundingQuery.FULL,
        sentences=None,
        phase=Phase.TRAIN,
    )

    result = grounding(baseline=MovingControlVariate(decay=0.5, initial=0.0)).ground(
        batch
    )

    assert result.ledger.questions == 2
    assert result.decided.tolist() == [True]


def test_joint_inconsistency_checks_both_directions() -> None:
    subject = PairwiseGrounding(
        LANGUAGE.store,
        LogicalScorer(),
        negation=NOT,
        application=LANGUAGE.calculus.app,
        labels=THREE_WAY.labels,
        label_map=CLASSICAL,
        contradiction="joint-inconsistency",
        composition=Marginal(),
        baseline=None,
        reward=None,
    )
    batch = GroundingInput(
        MappingProxyType(
            {
                ReadoutId("premise"): terms(RAIN),
                ReadoutId("hypothesis"): terms(WET),
            }
        ),
        choice_rows=MappingProxyType({}),
        pairs=JudgmentPairs(),
        query=GroundingQuery.FULL,
        sentences=None,
        phase=Phase.TRAIN,
    )

    result = subject.ground(batch)

    assert result.ledger.questions == 3


def test_sampled_forms_require_a_declared_moving_baseline() -> None:
    batch = GroundingInput(
        MappingProxyType(
            {
                ReadoutId("premise"): terms(RAIN, estimate=Estimate.SAMPLED),
                ReadoutId("hypothesis"): terms(WET, estimate=Estimate.SAMPLED),
            }
        ),
        choice_rows=MappingProxyType({}),
        pairs=Pairs(),
        query=GroundingQuery.OBSERVED_LABEL,
        sentences=None,
        phase=Phase.TRAIN,
    )

    with pytest.raises(ValueError, match="requires a control variate"):
        grounding().ground(batch)


def test_two_pair_draw_rewards_and_baseline_match_the_hand_computation() -> None:
    baseline = MovingControlVariate(decay=0.5, initial=0.2)
    reward = InferenceReward(
        proved=2.0,
        refuted=-1.0,
        contradiction="negated-hypothesis",
    )
    subject = PairwiseGrounding(
        LANGUAGE.store,
        LogicalScorer(),
        negation=NOT,
        application=LANGUAGE.calculus.app,
        labels=THREE_WAY.labels,
        label_map=CLASSICAL,
        contradiction="negated-hypothesis",
        composition=Marginal(),
        baseline=baseline,
        reward=reward,
    )
    batch = GroundingInput(
        MappingProxyType(
            {
                ReadoutId("first-premise"): weighted_terms((RAIN, COLD), (0.75, 0.25)),
                ReadoutId("first-hypothesis"): weighted_terms((WET,), (1.0,)),
                ReadoutId("second-premise"): weighted_terms((RAIN, COLD), (0.25, 0.75)),
                ReadoutId("second-hypothesis"): weighted_terms((WET,), (1.0,)),
            }
        ),
        choice_rows=MappingProxyType({}),
        pairs=TwoPairs(),
        query=GroundingQuery.OBSERVED_LABEL,
        sentences=None,
        phase=Phase.TRAIN,
    )

    result = subject.ground(batch)

    # Each draw table pays one for rain -> wet and zero for cold -> wet. The
    # classical three-label normalization maps the two weighted tables to the
    # following observed-label probabilities and declared affine rewards.
    label_probabilities = (0.5, 1.0 / 6.0)
    hand_rewards = tuple(reward.pay(one) for one in label_probabilities)
    result.observe_payments(torch.tensor(hand_rewards, dtype=torch.float64))
    assert result.log_probabilities.tensor[:, 0].exp().tolist() == pytest.approx(
        label_probabilities
    )
    assert tuple(reward.pay(one) for one in label_probabilities) == pytest.approx(
        (0.5, -0.5)
    )
    assert baseline.value == pytest.approx(
        0.5 * 0.2 + 0.5 * sum(hand_rewards) / len(hand_rewards)
    )
