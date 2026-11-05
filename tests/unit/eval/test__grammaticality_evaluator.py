from typing import NamedTuple, cast

import pytest
import torch

from lofi._types import ItemId, ReadoutId
from lofi.deduce import SpanIndex
from lofi.eval._boundary import decode_root_mass
from lofi.eval._grammaticality_evaluator import GrammaticalityEvaluator
from lofi.nn._values import LogDistribution
from lofi.nn.loss._readout import MASS_AXES
from lofi.nn.loss._values import ObjectiveOutputs, SpanReadout
from lofi.spec.evaluators.conventions import ConventionSettings
from lofi.spec.evaluators.grammaticality import GrammaticalityEvaluatorSettings


class Batch(NamedTuple):
    item_ids: tuple[ItemId, ...]
    words: tuple[tuple[str, ...], ...]
    lengths: tuple[int, ...]
    labels: tuple[bool, ...]


class Outputs(NamedTuple):
    batch: Batch
    readouts: dict[ReadoutId, SpanReadout]


def settings() -> GrammaticalityEvaluatorSettings:
    return GrammaticalityEvaluatorSettings(
        "roots",
        "root-mass",
        0.1,
        2,
        ConventionSettings(
            "keep", "keep", "keep", 0, "sentence", "zero", "allow", "canonical"
        ),
        (
            "grammaticality-roc-auc",
            "grammaticality-pr-auc",
            "grammaticality-accuracy-at-youden",
            "grammaticality-accuracy-at-half",
            "grammaticality-mcc",
            "grammaticality-noise-adjusted-auc",
            "grammaticality-auc-ceiling",
            "grammaticality-calibration-ece",
            "grammaticality-calibration-brier",
        ),
    )


def test_dataset_keeps_both_grammatical_and_corrupted_items() -> None:
    batch = Batch(
        (ItemId("good"), ItemId("bad")),
        (("a", "b"), ("b", "a")),
        (2, 2),
        (True, False),
    )
    dataset = GrammaticalityEvaluator(settings()).dataset((batch,))

    assert dataset.ids == batch.item_ids
    assert tuple(annotation.value for annotation in dataset.annotations) == (
        True,
        False,
    )


def test_prediction_reads_each_items_sentence_spanning_mass() -> None:
    batch = Batch(
        (ItemId("good"), ItemId("bad")),
        (("a", "b"), ("b", "a")),
        (2, 2),
        (True, False),
    )
    spans = SpanIndex(2)
    probabilities = torch.full((2, spans.num_spans, 2), 0.5)
    root = spans.span_id(0, 2)
    probabilities[0, root] = torch.tensor([0.9, 0.1])
    probabilities[1, root] = torch.tensor([0.2, 0.8])
    readout = SpanReadout(
        spans,
        LogDistribution(probabilities.log(), MASS_AXES, normalized_over="outcome"),
    )

    evaluator = GrammaticalityEvaluator(settings())
    predicted = evaluator.predict(
        cast(
            "ObjectiveOutputs[Batch]",
            Outputs(batch, {ReadoutId("root-mass"): readout}),
        )
    )

    assert predicted == pytest.approx({ItemId("good"): 0.9, ItemId("bad"): 0.2})
    [artifact] = evaluator.artifacts()
    published = decode_root_mass(artifact.content)
    assert artifact.name == "root-mass.json"
    assert published.readout == "root-mass"
    assert tuple(item.item_id for item in published.items) == ("good", "bad")
    assert tuple(item.log_mass for item in published.items) == pytest.approx(
        (probabilities[0, root, 0].log().item(), probabilities[1, root, 0].log().item())
    )
    assert tuple(item.log_complement for item in published.items) == pytest.approx(
        (probabilities[0, root, 1].log().item(), probabilities[1, root, 1].log().item())
    )


def test_every_battery_row_computes_over_two_classes() -> None:
    evaluator = GrammaticalityEvaluator(settings())
    entries = ((0.9, True), (0.7, True), (0.3, False), (0.1, False))

    for metric in evaluator.metrics:
        for position, (predicted, gold) in enumerate(entries):
            metric.update(predicted, gold, key=ItemId(str(position)))
        assert metric.compute().value is not None
        assert metric.strata() == {
            ItemId("0"): "grammatical",
            ItemId("1"): "grammatical",
            ItemId("2"): "ungrammatical",
            ItemId("3"): "ungrammatical",
        }
