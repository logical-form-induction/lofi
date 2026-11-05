"""Tests for held-out string and derivational likelihood evaluation."""

from __future__ import annotations

from math import log
from typing import TYPE_CHECKING, NamedTuple, cast

import pytest
import torch

from lofi._types import ItemId, ReadoutId
from lofi.data import (
    ByteSpan,
    Derivation,
    DerivationNode,
    Example,
    GoldDerivation,
    Item,
    Token,
    TokenSpan,
)
from lofi.deduce import SpanIndex
from lofi.eval._held_out_likelihood_evaluator import (
    PLUGIN,
    HeldOutLikelihoodEvaluator,
)
from lofi.nn._values import (
    DerivationScoreChart,
    DerivationSetScore,
    LogDistribution,
    NormalFormDerivationScores,
)
from lofi.nn.loss._readout import MASS_AXES
from lofi.nn.loss._values import ObjectiveOutputs, SpanReadout
from lofi.spec.evaluators.conventions import ConventionSettings
from lofi.spec.evaluators.held_out_likelihood import (
    HeldOutLikelihoodMeasurement,
    HeldOutLikelihoodSettings,
    HeldOutLikelihoodSpec,
)
from lofi.spec.losses.simplicity import UniformSymbolCostSpec
from lofi.terms import (
    Operator,
    OperatorIndex,
    PreorderArityLayout,
    Signature,
    TermCodec,
)

if TYPE_CHECKING:
    from lofi.eval import HeldOutData, HeldOutEvaluationContext, Metric
    from lofi.terms import Term


class _NormalForm:
    name = "eisner"


class _Grammar(NamedTuple):
    normal_form: _NormalForm


class _Syntax:
    def parse(self, store: object, text: str, /) -> str:
        del store
        return f"term:{text}"


class _LanguageCodec:
    def at(self, sort: str, /) -> TermCodec:
        if sort != "Cat":
            msg = f"no sort {sort!r}"
            raise ValueError(msg)

        return CODEC


class _Language(NamedTuple):
    codec: _LanguageCodec
    category_sort: str


class _Assembly(NamedTuple):
    grammar: _Grammar
    syntax: _Syntax
    store: object
    language: _Language


class _Context(NamedTuple):
    grammar: _Assembly


class _Batch(NamedTuple):
    item_ids: tuple[ItemId, ...]
    words: tuple[tuple[str, ...], ...]


class _Outputs(NamedTuple):
    batch: _Batch
    spans: SpanIndex
    readouts: dict[ReadoutId, SpanReadout]
    derivations: DerivationScoreChart[Term] | None


class _Chart:
    normal_form = "eisner"

    def score(
        self, item: int, derivation: Derivation[Term], /
    ) -> NormalFormDerivationScores:
        del derivation
        if item != 0:
            msg = f"no item {item}"
            raise IndexError(msg)

        return NormalFormDerivationScores(
            DerivationSetScore("eisner", -1.0, -0.5),
            DerivationSetScore("none", -1.0, 0.0),
        )


_CHART = _Chart()
_DERIVATION_CHART = cast("DerivationScoreChart[Term]", _CHART)

SIGNATURE = Signature(
    "held-out-likelihood",
    ("Cat",),
    (
        Operator("A", "Cat", family="primitive"),
        Operator("B", "Cat", family="primitive"),
    ),
)
CODEC = TermCodec(OperatorIndex(SIGNATURE), PreorderArityLayout(1))


def settings(
    measurements: tuple[HeldOutLikelihoodMeasurement, ...] = (
        "string",
        "derivation",
        "description-length",
    ),
) -> HeldOutLikelihoodSettings:
    """Return held-out likelihood settings under canonical conventions."""
    return HeldOutLikelihoodSettings(
        "types",
        "sentences",
        UniformSymbolCostSpec(cost=2.0),
        ConventionSettings(
            "keep", "keep", "keep", 0, "sentence", "zero", "allow", "canonical"
        ),
        measurements,
    )


def context() -> HeldOutEvaluationContext:
    """Return a minimal held-out grammar boundary."""
    held = _Context(
        _Assembly(
            _Grammar(_NormalForm()),
            _Syntax(),
            object(),
            _Language(_LanguageCodec(), "Cat"),
        )
    )
    return cast("HeldOutEvaluationContext", held)


def evaluator() -> HeldOutLikelihoodEvaluator:
    """Return an evaluator over a minimal grammar boundary."""
    return HeldOutLikelihoodEvaluator(context(), settings())


def gold() -> Derivation[str]:
    """Return a two-token gold derivation."""
    return Derivation(
        2,
        {
            0: DerivationNode(TokenSpan(0, 1), "A"),
            1: DerivationNode(TokenSpan(1, 2), "B"),
            2: DerivationNode(TokenSpan(0, 2), "S", rule="combine", children=(0, 1)),
        },
    )


def outputs(
    *, chart: DerivationScoreChart[Term] | None = _DERIVATION_CHART
) -> _Outputs:
    """Return root mass, lexical terms and an optional derivation chart."""
    spans = SpanIndex(2)
    probabilities = torch.full((1, spans.num_spans, 2), 0.5, dtype=torch.float64)
    probabilities[0, spans.span_id(0, 2)] = torch.tensor([0.25, 0.75])
    mass = SpanReadout(
        spans,
        LogDistribution(probabilities.log(), MASS_AXES, normalized_over="outcome"),
    )
    term_probabilities = torch.zeros(
        1, spans.num_spans, CODEC.capacity, CODEC.n_symbols, dtype=torch.float64
    )
    term_probabilities[0, spans.span_id(0, 1), 0] = torch.tensor([0.25, 0.75, 0.0])
    term_probabilities[0, spans.span_id(1, 2), 0] = torch.tensor([0.5, 0.0, 0.5])
    term_probabilities[0, spans.span_id(0, 2), 0] = torch.tensor([0.0, 1.0, 0.0])
    terms = SpanReadout(
        spans,
        LogDistribution(
            term_probabilities.log(),
            ("item", "span", "slot", "symbol"),
            normalized_over="symbol",
        ),
    )

    return _Outputs(
        _Batch((ItemId("one"),), (("a", "b"),)),
        spans,
        {ReadoutId("sentences"): mass, ReadoutId("types"): terms},
        chart,
    )


class _Partitions:
    """Retained data holding one gold derivation under every split."""

    def derivations(self, split: str, /) -> tuple[Example[GoldDerivation[str]], ...]:
        del split
        tokens = (Token(0, "a", ByteSpan(0, 1)), Token(1, "b", ByteSpan(2, 3)))
        item = Item(ItemId("one"), tokens, text="a b")
        return (Example(item, GoldDerivation(gold())),)


def test_prediction_precedes_annotation_and_all_rows_use_the_published_values() -> None:
    built = evaluator()
    predicted = built.predict(cast("ObjectiveOutputs[_Batch]", outputs()))
    annotated = built.annotate(cast("Item", object()), gold())
    expected = (-log(4.0), -log(2.0), -0.5, -0.25, -1.0, -0.5, 2.5)

    partition = built.held_out(cast("HeldOutData", _Partitions()), "test")
    assert partition.annotations[0].value == annotated
    assert tuple(node.label for node in annotated.nodes.values()) == (
        "term:A",
        "term:B",
        "term:S",
    )
    for metric, value in zip(built.metrics, expected, strict=True):
        metric.update(predicted[ItemId("one")], annotated, key=ItemId("one"))
        assert metric.compute().value == pytest.approx(value)
        assert metric.conventions is built.conventions

    description = built.metrics[-1]
    result = description.compute()
    per_item = description.per_item()[ItemId("one")]

    assert result.n == 1
    assert result.components == {"cells": 2.0}
    assert per_item.value == pytest.approx(2.5)
    assert per_item.n == 1


def test_prediction_requires_the_tagged_probabilistic_derivation_chart() -> None:
    with pytest.raises(ValueError, match="tagged log-sum-exp chart"):
        evaluator().predict(cast("ObjectiveOutputs[_Batch]", outputs(chart=None)))


def test_string_and_description_measurements_do_not_score_gold_derivations() -> None:
    built = HeldOutLikelihoodEvaluator(
        context(), settings(("string", "description-length"))
    )
    predicted = built.predict(cast("ObjectiveOutputs[_Batch]", outputs(chart=None)))

    partition = built.held_out(cast("HeldOutData", _Partitions()), "test")
    assert partition.annotations[0].value == gold()
    assert tuple(metric.name for metric in built.metrics) == (
        "string-log-likelihood-per-sentence",
        "string-log-likelihood-per-token",
        "description-length-nats",
    )
    surface_metrics = cast("tuple[Metric[object, Derivation[str]], ...]", built.metrics)
    for metric in surface_metrics:
        metric.update(predicted[ItemId("one")], gold(), key=ItemId("one"))

    assert tuple(metric.compute().value for metric in built.metrics) == pytest.approx(
        (-log(4.0), -log(2.0), 2.5)
    )


def test_the_plugin_builds_the_rows_its_declaration_promises() -> None:
    declaration = HeldOutLikelihoodSpec.model_validate(
        {
            "kind": "held-out-likelihood",
            "interval": {"replicates": 50, "level": 0.95},
            "readout": "types",
            "output": "sentences",
            "measurements": ["string", "derivation", "description-length"],
            "symbol_cost": {"kind": "uniform", "cost": 2.0},
            "conventions": settings().conventions._asdict(),
        }
    )

    built = PLUGIN.construct(declaration, context())

    assert PLUGIN.spec is HeldOutLikelihoodSpec
    assert declaration.metrics() == tuple(metric.name for metric in built.metrics)
