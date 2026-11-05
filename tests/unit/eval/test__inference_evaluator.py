from typing import TYPE_CHECKING, NamedTuple, cast

import pytest
import torch

from lofi._types import ItemId, JsonValue
from lofi.data import Categorical
from lofi.eval import InferenceEvaluationContext, StratifiedMetric
from lofi.eval._inference_evaluator import InferenceEvaluator, InferencePrediction
from lofi.eval._inference_proof import InferenceProof
from lofi.lf import RTE, THREE_WAY, InferenceCell, InferenceLabel, InferenceLabeling
from lofi.spec.components.run import ParseSpec
from lofi.spec.evaluators.inference import InferenceEvaluatorSpec

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi._types import ReadoutId
    from lofi.data.synthetic import InterpretationFunction
    from lofi.nn.loss._values import ObjectiveOutputs
    from lofi.spec.grammar import GrammarAssembly


class Batch(NamedTuple):
    pair_ids: tuple[ItemId, ...]
    labels: tuple[InferenceLabel, ...]
    item_ids: tuple[ItemId, ...] = ()
    words: tuple[tuple[str, ...], ...] = ()
    premises: tuple[ReadoutId, ...] = ()
    hypotheses: tuple[ReadoutId, ...] = ()


class Distribution(NamedTuple):
    tensor: torch.Tensor


class Grounding(NamedTuple):
    log_probabilities: Distribution
    decided: torch.Tensor
    labels: tuple[InferenceLabel, ...]


class Outputs(NamedTuple):
    batch: Batch
    entailment: Grounding | None


def evaluator(
    reduction: JsonValue = None,
    *,
    coverage: str = "zero",
    tokens: Mapping[ItemId, tuple[tuple[str, ...], tuple[str, ...]]] | None = None,
    proof: JsonValue = None,
    labeling: InferenceLabeling = THREE_WAY,
) -> InferenceEvaluator:
    spec = InferenceEvaluatorSpec.model_validate(
        {
            "kind": "inference",
            "interval": {"replicates": 50, "level": 0.95},
            "reduction": reduction,
            "partitions": {"fit_on": "dev", "score_on": "test"},
            "bins": 2,
            "proof": proof,
            "conventions": {
                "punctuation": "keep",
                "unary_chains": "keep",
                "trivial_spans": "keep",
                "minimum_length": 0,
                "aggregation": "corpus",
                "coverage": coverage,
                "self_arcs": "allow",
                "gold_derivation": "canonical",
            },
        }
    )
    context = InferenceEvaluationContext(
        cast("GrammarAssembly", object()),
        tuple(label.value for label in labeling.labels),
        interpretation=cast("InterpretationFunction", object()),
        forms={},
        tokens={} if tokens is None else tokens,
        parse=ParseSpec(retain="forest", reachability="all").build(),
    )

    return InferenceEvaluator(context, spec.settings())


def test_dataset_joins_pair_occurrences_to_the_shared_response_scale() -> None:
    built = evaluator()
    batch = Batch(
        (ItemId("inference-pair/2"), ItemId("inference-pair/7")),
        (InferenceLabel.ENTAILS, InferenceLabel.NEUTRAL),
    )

    dataset = built.dataset((batch,))

    assert dataset.ids == batch.pair_ids
    assert tuple(annotation.value.label for annotation in dataset.annotations) == (
        "entails",
        "neutral",
    )
    assert all(
        annotation.value.label_set
        == frozenset(label.value for label in THREE_WAY.labels)
        for annotation in dataset.annotations
    )


def test_prediction_uses_enum_column_order_and_keeps_explicit_abstentions() -> None:
    built = evaluator()
    batch = Batch(
        (ItemId("inference-pair/2"), ItemId("inference-pair/7")),
        (InferenceLabel.ENTAILS, InferenceLabel.NEUTRAL),
    )
    outputs = Outputs(
        batch,
        Grounding(
            Distribution(torch.tensor([[0.0, -1.0, -2.0], [-2.0, -1.0, 0.0]])),
            torch.tensor([True, False]),
            THREE_WAY.labels,
        ),
    )

    predicted = built.predict(cast("ObjectiveOutputs[Batch]", outputs))
    assert {key: value.label for key, value in predicted.items()} == {
        ItemId("inference-pair/2"): "entails",
        ItemId("inference-pair/7"): None,
    }


def test_binary_data_inventory_controls_prediction_and_metric_columns() -> None:
    built = evaluator(labeling=RTE)
    batch = Batch(
        (ItemId("inference-pair/2"), ItemId("inference-pair/7")),
        (InferenceLabel.ENTAILS, InferenceLabel.NON_ENTAILS),
    )
    outputs = Outputs(
        batch,
        Grounding(
            Distribution(torch.tensor([[0.0, -1.0], [-2.0, 0.0]])),
            torch.tensor([True, True]),
            RTE.labels,
        ),
    )

    predicted = built.predict(cast("ObjectiveOutputs[Batch]", outputs))

    assert {key: value.label for key, value in predicted.items()} == {
        ItemId("inference-pair/2"): "entails",
        ItemId("inference-pair/7"): "non-entails",
    }
    assert tuple(metric.name for metric in built.metrics)[:2] == (
        "inference-label-2-way",
        "inference-label-2-way-macro-f1",
    )


def test_every_declared_metric_is_a_distinct_reportable_accumulator() -> None:
    built = evaluator()
    labels = THREE_WAY.labels
    scale = frozenset(label.value for label in labels)
    predictions = (
        "entails",
        "contradicts",
        "neutral",
        "neutral",
        "entails",
        "contradicts",
    )
    gold = (
        InferenceLabel.ENTAILS,
        InferenceLabel.CONTRADICTS,
        InferenceLabel.NEUTRAL,
        InferenceLabel.ENTAILS,
        InferenceLabel.CONTRADICTS,
        InferenceLabel.NEUTRAL,
    )

    for metric in built.metrics:
        for index, (predicted, observed) in enumerate(
            zip(predictions, gold, strict=True)
        ):
            metric.update(
                InferencePrediction(predicted, (0.4, 0.3, 0.3), None),
                Categorical(observed.value, scale),
                key=ItemId(str(index)),
            )

        result = metric.compute()
        assert result.value is not None
        assert result.n == sum(
            item.value is not None for item in metric.per_item().values()
        )

    assert len(built.metrics) == 15
    assert len({metric.name for metric in built.metrics}) == 15
    assert all("micro" not in metric.name for metric in built.metrics)


@pytest.mark.parametrize(
    ("coverage", "accuracy", "scored"),
    [("zero", 0.5, 2), ("ignore", 1.0, 1)],
)
def test_abstention_is_reported_and_obeys_the_coverage_policy(
    coverage: str,
    accuracy: float,
    scored: int,
) -> None:
    built = evaluator(coverage=coverage)
    scale = frozenset(label.value for label in THREE_WAY.labels)
    predictions = ("entails", None)
    gold = (InferenceLabel.ENTAILS, InferenceLabel.NEUTRAL)

    for metric in built.metrics:
        for index, (predicted, observed) in enumerate(
            zip(predictions, gold, strict=True)
        ):
            metric.update(
                InferencePrediction(predicted, (0.4, 0.3, 0.3), None),
                Categorical(observed.value, scale),
                key=ItemId(str(index)),
            )

    by_name = {metric.name: metric.compute() for metric in built.metrics}

    assert by_name["inference-label-3-way"].value == accuracy
    assert by_name["inference-label-3-way"].n == scored
    assert by_name["inference-label-3-way-abstention"].value == 0.5
    assert by_name["inference-label-3-way-abstention"].n == 2


def test_target_rows_stratify_resamples_by_their_defining_support() -> None:
    built = evaluator(labeling=RTE)
    scale = frozenset(label.value for label in RTE.labels)
    pairs = (
        ("entails", InferenceLabel.ENTAILS),
        ("non-entails", InferenceLabel.ENTAILS),
        ("entails", InferenceLabel.NON_ENTAILS),
        ("non-entails", InferenceLabel.NON_ENTAILS),
    )
    for metric in built.metrics:
        for index, (predicted, observed) in enumerate(pairs):
            metric.update(
                InferencePrediction(predicted, (0.5, 0.5), None),
                Categorical(observed.value, scale),
                key=ItemId(str(index)),
            )

    by_name = {metric.name: metric for metric in built.metrics}
    precision = by_name["inference-label-2-way-entails-precision"]
    recall = by_name["inference-label-2-way-entails-recall"]
    f1 = by_name["inference-label-2-way-entails-f1"]
    accuracy = by_name["inference-label-2-way"]

    assert isinstance(precision, StratifiedMetric)
    assert precision.strata() == {
        ItemId("0"): "target-support",
        ItemId("1"): "other",
        ItemId("2"): "target-support",
        ItemId("3"): "other",
    }
    assert recall.strata() == {
        ItemId("0"): "target-support",
        ItemId("1"): "target-support",
        ItemId("2"): "other",
        ItemId("3"): "other",
    }
    assert f1.strata() == {
        ItemId("0"): "target-support",
        ItemId("1"): "target-support",
        ItemId("2"): "target-support",
        ItemId("3"): "other",
    }
    assert set(accuracy.strata().values()) == {"all"}
    assert {key: result.value for key, result in precision.per_item().items()} == {
        ItemId("0"): 1.0,
        ItemId("1"): None,
        ItemId("2"): 0.0,
        ItemId("3"): None,
    }
    assert recall.compute().n == 2
    assert f1.compute().n == 3


def test_missing_grounding_output_is_named() -> None:
    built = evaluator()
    batch = Batch((ItemId("inference-pair/2"),), (InferenceLabel.ENTAILS,))

    with pytest.raises(ValueError, match="entailment"):
        built.predict(cast("ObjectiveOutputs[Batch]", Outputs(batch, None)))


def test_controls_fit_development_and_remain_frozen_on_changed_test_labels() -> None:
    scale = frozenset(label.value for label in THREE_WAY.labels)
    dev = {
        ItemId("dev-e"): "entails",
        ItemId("dev-c"): "contradicts",
        ItemId("dev-n"): "neutral",
    }
    test = {
        ItemId("test-e"): "entails",
        ItemId("test-c"): "contradicts",
        ItemId("test-n"): "neutral",
    }
    tokens = {key: (("premise",), (label,)) for key, label in {**dev, **test}.items()}
    built = evaluator(tokens=tokens)
    forecast = InferencePrediction("entails", (0.4, 0.3, 0.3), None)
    head = built.fit_alignment(
        dict.fromkeys(dev, forecast),
        {key: Categorical(label, scale) for key, label in dev.items()},
    )
    scored = built.score_alignment(
        dict.fromkeys(test, forecast),
        {key: Categorical(label, scale) for key, label in test.items()},
        seed=0,
    )
    assert scored.rows["hypothesis-only"].value == 1.0
    assert scored.rows["majority-class"].value == pytest.approx(1 / 3)
    assert head.fitted == len(dev)
    with pytest.raises(ValueError, match=r"already.*frozen"):
        built.fit_alignment(
            dict.fromkeys(test, forecast),
            {key: Categorical(label, scale) for key, label in test.items()},
        )
    with pytest.raises(ValueError, match="fitting occurrences"):
        built.score_alignment(
            dict.fromkeys(dev, forecast),
            {key: Categorical(label, scale) for key, label in dev.items()},
            seed=0,
        )


def test_calibration_reduces_and_conditions_the_published_label_mass() -> None:
    built = evaluator(
        {
            "mapping": {
                "entails": "entails",
                "contradicts": "not-entails",
                "neutral": "not-entails",
            }
        }
    )
    scale = frozenset(label.value for label in THREE_WAY.labels)
    prediction = InferencePrediction("neutral", (0.05, 0.2, 0.25), None)
    for metric in built.metrics:
        metric.update(prediction, Categorical("neutral", scale), key=ItemId("test"))
    rows = {metric.name: metric.compute() for metric in built.metrics}
    assert rows["entailment-calibration-brier"].value == pytest.approx(0.02)


def test_proof_rows_keep_reference_labels_beam_recovery_and_abstentions_distinct() -> (
    None
):
    built = evaluator(proof={"budget": {"calls_per_step": 8}, "depth": 2})
    scale = frozenset(label.value for label in THREE_WAY.labels)
    proof = InferenceProof(
        InferenceCell.ENTAILS,
        (InferenceCell.NEUTRAL, InferenceCell.ENTAILS),
        6,
        0,
        0.01,
        6,
        0.02,
        (("timeout", 0),),
    )
    prediction = InferencePrediction("entails", (0.7, 0.1, 0.2), proof)
    for metric in built.metrics:
        metric.update(prediction, Categorical("contradicts", scale), key=ItemId("test"))
    rows = {metric.name: metric.compute() for metric in built.metrics}
    assert rows["inference-label-3-way"].value == 0.0
    assert rows["entailment-preserving-rate"].value == 0.0
    assert rows["entailment-preserving-rate-at-2"].value == 1.0
    assert rows["relaxation-gap"].value == pytest.approx(0.7)
    assert rows["proof-calls"].value == 6
    assert rows["proof-questions"].value == 6


@pytest.mark.parametrize("cell", [None, InferenceCell.DEGENERATE])
def test_proof_abstention_and_both_proved_are_explicit_undefined_rates(
    cell: InferenceCell | None,
) -> None:
    built = evaluator(proof={"budget": {"calls_per_step": 0}, "depth": 2})
    scale = frozenset(label.value for label in THREE_WAY.labels)
    proof = InferenceProof(
        InferenceCell.ENTAILS,
        (cell,),
        0,
        0,
        0.0,
        2,
        0.0,
        (("budget-spent", 2),),
    )
    prediction = InferencePrediction(None, (0.0, 0.0, 0.0), proof)
    for metric in built.metrics:
        metric.update(prediction, Categorical("entails", scale), key=ItemId("test"))
    rows = {metric.name: metric.compute() for metric in built.metrics}
    assert rows["entailment-preserving-rate"].value is None
    assert rows["entailment-preserving-rate-at-2"].value is None
    assert rows["relaxation-gap"].value is None
    assert rows["entailment-preserving-abstention"].value == float(cell is None)
    assert rows["entailment-preserving-both"].value == float(
        cell is InferenceCell.DEGENERATE
    )
    assert rows["proof-abstentions"].components["budget-spent"] == 2
