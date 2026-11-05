from typing import Final, NamedTuple

import pytest
import torch
from tests.language import language

from lofi._types import ReadoutId
from lofi.control_variate import MovingControlVariate
from lofi.deduce import SpanIndex
from lofi.lf import THREE_WAY, ArgumentRenaming, InferenceLabel
from lofi.nn._inference_response import LABEL_PLUGIN
from lofi.nn._values import InferenceSignal, LogDistribution
from lofi.nn.forms import Estimate, WeightedTerms
from lofi.nn.grounding import CellMass, EvidenceLedger, GradientPath, GroundedBatch
from lofi.nn.loss._inference import InferenceObjective
from lofi.nn.loss._parser import SPAN_AXES
from lofi.nn.loss._values import CellChart, ObjectiveOutputs, RecordingOutputs
from lofi.reward import InferenceReward
from lofi.spec.losses import LabelResponseSettings

LANGUAGE: Final = language(("rain",))
TERM: Final = LANGUAGE.constant("rain")
SPANS: Final = SpanIndex(1)


class Pairs(NamedTuple):
    premises: tuple[ReadoutId, ...]
    hypotheses: tuple[ReadoutId, ...]
    alignments: tuple[tuple[ArgumentRenaming, ...], ...]
    labels: tuple[InferenceLabel, ...]


class PaysRain:
    def credit(self, form: object, /) -> float:
        return 1.0 if form == TERM else 0.0


def prediction(
    *,
    decided: tuple[bool, ...] = (True, True),
    auxiliary: torch.Tensor | None = None,
    baseline: MovingControlVariate | None = None,
    reward: InferenceReward | None = None,
) -> GroundedBatch:
    probabilities = torch.tensor(
        [[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]], dtype=torch.float64
    ).log()
    mask = torch.tensor(decided)
    ledger = EvidenceLedger(
        {"verdict": 3},
        {"timeout": 1},
        calls=3,
        memo_hits=1,
        cpu_seconds=0.5,
        questions=4,
        wall_seconds=2.0,
        process_starts=2,
        verified_share=0.75,
    )
    return GroundedBatch(
        LogDistribution(
            probabilities,
            ("pair", "label"),
            mask,
            normalized_over="label",
        ),
        labels=THREE_WAY.labels,
        complete=True,
        residual=torch.tensor([0.1, 0.2], dtype=torch.float64),
        decided=mask,
        cells=CellMass(
            torch.tensor(
                [[0.7, 0.1, 0.1, 0.1], [0.1, 0.2, 0.5, 0.2]],
                dtype=torch.float64,
            )
        ),
        gradient_path=GradientPath.SCORE_FUNCTION,
        baseline=baseline,
        auxiliary=auxiliary,
        ledger=ledger,
        variance=torch.tensor([0.2, 0.4], dtype=torch.float64),
        reward=reward,
    )


def forms() -> dict[ReadoutId, WeightedTerms]:
    weights = torch.tensor([0.6], dtype=torch.float64).log().requires_grad_()
    return {
        ReadoutId("premise/0"): WeightedTerms(
            (TERM,), weights, weights.new_tensor(0.4).log(), Estimate.TRUNCATED
        )
    }


def outputs(
    grounded: GroundedBatch | None = None,
    *,
    labels: tuple[InferenceLabel, ...] = (
        InferenceLabel.ENTAILS,
        InferenceLabel.NEUTRAL,
    ),
    published_forms: dict[ReadoutId, WeightedTerms] | None = None,
) -> ObjectiveOutputs[Pairs]:
    pairs = len(labels)
    sides = tuple(ReadoutId(f"side/{position}") for position in range(pairs))
    return ObjectiveOutputs(
        spans=SPANS,
        chart=CellChart(LogDistribution(torch.zeros(pairs, 1), SPAN_AXES)),
        batch=Pairs(
            sides,
            sides,
            ((ArgumentRenaming.identity(),),) * pairs,
            labels,
        ),
        forms=published_forms,
        entailment=prediction() if grounded is None else grounded,
    )


def objective(
    signal: InferenceSignal = InferenceSignal.LIKELIHOOD,
    *,
    auxiliary_weight: float = 0.0,
    credit: PaysRain | None = None,
) -> InferenceObjective:
    return InferenceObjective(
        signal,
        LABEL_PLUGIN.construct(LabelResponseSettings()),
        auxiliary_weight=auxiliary_weight,
        credit=credit,
        name="inference",
    )


def test_likelihood_is_the_mean_negative_log_probability_of_decided_labels() -> None:
    term = objective().evaluate(outputs())

    assert term.count == 2
    assert float(term.value) == pytest.approx(
        -(torch.tensor(0.7).log() + torch.tensor(0.6).log()).item() / 2.0
    )
    assert term.residual is not None
    assert float(term.residual) == pytest.approx(0.15)


def test_reward_is_the_negative_mean_probability_of_decided_labels() -> None:
    baseline = MovingControlVariate(decay=0.5, initial=0.0)
    term = objective(InferenceSignal.REWARD).evaluate(
        outputs(
            prediction(
                baseline=baseline,
                reward=InferenceReward(
                    proved=2.0,
                    refuted=-0.5,
                    contradiction="negated-hypothesis",
                ),
            )
        )
    )

    paid = (-0.5 + 2.5 * 0.7, -0.5 + 2.5 * 0.6)
    assert float(term.value) == pytest.approx(-sum(paid) / 2.0)
    assert float(term.diagnostics["reward_mean"]) == pytest.approx(sum(paid) / 2.0)
    assert baseline.value == pytest.approx(sum(paid) / 4.0)


def test_reward_refuses_a_grounding_without_declared_payoffs() -> None:
    with pytest.raises(ValueError, match="declared proof payoffs"):
        objective(InferenceSignal.REWARD).evaluate(outputs())


def test_undecided_pairs_enter_neither_the_value_nor_the_count() -> None:
    term = objective().evaluate(outputs(prediction(decided=(True, False))))

    assert term.count == 1
    assert float(term.value) == pytest.approx(-torch.tensor(0.7).log().item())
    assert float(term.diagnostics["abstention_rate"]) == pytest.approx(0.5)


def test_a_fully_undecided_batch_returns_a_graph_connected_zero() -> None:
    grounded = prediction(decided=(False, False))
    grounded.log_probabilities.tensor.requires_grad_()

    term = objective().evaluate(outputs(grounded))

    assert term.count == 0
    assert float(term.value.detach()) == 0.0
    assert term.value.grad_fn is not None
    assert term.residual is not None
    assert float(term.residual) == 1.0


def test_auxiliary_weight_is_applied_once() -> None:
    auxiliary = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    plain = objective().evaluate(outputs())
    weighted = objective(auxiliary_weight=0.25).evaluate(
        outputs(prediction(auxiliary=auxiliary))
    )

    assert float((weighted.value - plain.value).detach()) == pytest.approx(0.5)
    assert torch.autograd.grad(weighted.value, auxiliary) == (torch.tensor(0.25),)


def test_optional_form_credit_adds_forms_to_the_declared_inputs() -> None:
    plain = objective()
    shaped = objective(credit=PaysRain())
    published = outputs(published_forms=forms())
    plain_value = plain.evaluate(published).value
    shaped_value = shaped.evaluate(published).value

    assert plain.requires == frozenset({"batch", "entailment"})
    assert shaped.requires == frozenset({"batch", "entailment", "forms"})
    assert float((plain_value - shaped_value).detach()) == pytest.approx(0.6)


def test_the_objective_reads_only_its_declared_outputs() -> None:
    subject = objective()
    view = RecordingOutputs(
        outputs(), objective=subject.name, requires=subject.requires
    )

    subject.evaluate(view)

    assert view.touched == frozenset({"batch", "entailment"})


def test_grounding_telemetry_is_carried_into_step_diagnostics() -> None:
    diagnostics = objective().evaluate(outputs()).diagnostics

    assert float(diagnostics["calls"]) == 3.0
    assert float(diagnostics["questions"]) == 4.0
    assert float(diagnostics["questions_per_second"]) == 2.0
    assert float(diagnostics["memo_hits"]) == 1.0
    assert float(diagnostics["memo_hit_rate"]) == 0.25
    assert float(diagnostics["cpu_seconds"]) == 0.5
    assert float(diagnostics["wall_seconds"]) == 2.0
    assert float(diagnostics["process_starts"]) == 2.0
    assert float(diagnostics["abstention/timeout"]) == 1.0
    assert float(diagnostics["degenerate_mass"]) == pytest.approx(0.15)
    assert float(diagnostics["variance"]) == pytest.approx(0.3)
    assert float(diagnostics["verified_share"]) == 0.75


def test_a_label_count_that_disagrees_with_the_grounding_is_refused() -> None:
    with pytest.raises(ValueError, match="one label per grounded pair"):
        objective().evaluate(outputs(labels=(InferenceLabel.ENTAILS,)))


@pytest.mark.parametrize("weight", [-0.1, float("inf"), float("nan")])
def test_auxiliary_weights_are_finite_and_nonnegative(weight: float) -> None:
    with pytest.raises(ValueError, match="auxiliary weight"):
        objective(auxiliary_weight=weight)
