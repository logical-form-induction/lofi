from __future__ import annotations

from math import log
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import pytest
import torch
from tests.conformance.evaluator import (
    CONFIG,
    HeldOutPass,
    held_out_run,
    packaged_held_out_pass,
)

from lofi._types import ItemId, ReadoutId
from lofi.data import AttachmentTree
from lofi.deduce import FREE_ROOTS, DerivationCountRow, parse
from lofi.eval._attachment import LengthBand
from lofi.eval._bracketing import Bracketing, Constituent, GoldBracketing
from lofi.eval._dependencies import DependencyPrediction
from lofi.eval._structural_diagnostics import (
    PLUGIN,
    StructuralGold,
    StructuralPrediction,
    _structural_rows,
    _StructuralMetric,
)
from lofi.eval._values import ConventionVector
from lofi.nn.forms import Estimate, WeightedTerms
from lofi.nn.loss._values import ObjectiveOutputs, SpanTerms
from lofi.spec import PACKAGED_CONFIG_ROOT, RunSpecCodec
from lofi.spec.evaluators.structural_diagnostics import StructuralDiagnosticsSpec

if TYPE_CHECKING:
    from lofi.eval import HeldOutItems

CONVENTIONS = ConventionVector(
    punctuation="keep",
    unary_chains="keep",
    trivial_spans="keep",
    minimum_length=0,
    aggregation="sentence",
    coverage="zero",
    self_arcs="allow",
    gold_derivation="canonical",
)
GOLD = StructuralGold(
    GoldBracketing(
        4,
        (
            Constituent(0, 4, label="S", depth=0),
            Constituent(0, 2, label="NP", depth=1),
            Constituent(2, 4, label="VP", depth=1),
        ),
        punctuation=(),
    ),
    AttachmentTree(("a", "b", "c", "d"), (0, 1, 2, 3)),
    DependencyPrediction(()),
)


def _declaration(
    *,
    derivation_counts: bool,
    root_mode: str = "goal",
    labels: tuple[str, ...] = (),
    lengths: tuple[int, ...] = (),
    depths: tuple[int, ...] = (),
) -> StructuralDiagnosticsSpec:
    return StructuralDiagnosticsSpec.model_validate(
        {
            "kind": "structural-diagnostics",
            "interval": {"replicates": 20, "level": 0.95},
            "readout": "category-terms",
            "astar": {
                "root_mode": {"kind": root_mode},
                "proposer": {"kind": "supertag-beam"},
            },
            "decode": {"kind": "viterbi"},
            "derivation_counts": derivation_counts,
            "dependency_interpretation": "dependency",
            "labels": list(labels),
            "lengths": list(lengths),
            "depths": list(depths),
            "attachment_band": "all",
            "attachment_limit": None,
            "attachment_direction": "directed",
            "dependency_mode": "labeled-directed",
            "dependency_measure": "f-score",
            "conventions": dict(CONVENTIONS.factors),
        }
    )


def _gold_lexical_output(
    held_out: HeldOutPass | None = None,
) -> ObjectiveOutputs[HeldOutItems]:
    held_out = packaged_held_out_pass() if held_out is None else held_out
    outputs = held_out.published[0]
    item_id = outputs.batch.item_ids[0]
    derivation = next(
        derivation for item, derivation in held_out.gold if item.item_id == item_id
    )
    cells: dict[tuple[int, int], WeightedTerms] = {}
    for position, leaf in enumerate(derivation.leaves()):
        category = held_out.grammar.syntax.parse(held_out.grammar.store, leaf.label)
        span = outputs.spans.span_id(position, position + 1)
        cells[0, span] = WeightedTerms(
            (category,),
            torch.tensor([-log(1.0)]),
            torch.tensor(-torch.inf),
            Estimate.EXACT,
        )
    return ObjectiveOutputs(
        spans=outputs.spans,
        chart=outputs.chart,
        batch=outputs.batch,
        terms={ReadoutId("category-terms"): SpanTerms(outputs.spans, cells)},
    )


def prediction(bracketing: Bracketing) -> StructuralPrediction:
    return StructuralPrediction(
        bracketing,
        GOLD.attachment,
        GOLD.dependency,
        3,
        6,
        0.5,
    )


def _rows(
    *,
    derivation_counts: bool = False,
    labels: tuple[str, ...] = (),
    lengths: tuple[int, ...] = (),
    depths: tuple[int, ...] = (),
) -> dict[str, _StructuralMetric]:
    settings = _declaration(
        derivation_counts=derivation_counts,
        labels=labels,
        lengths=lengths,
        depths=depths,
    ).settings()
    return {
        name: _StructuralMetric(name, reading, CONVENTIONS)
        for name, reading in _structural_rows(settings, LengthBand("all", None))
    }


def test_structural_rows_select_label_length_and_depth_breakdowns() -> None:
    pred = prediction(Bracketing(4, ((0, 4), (0, 2))))
    expected = {
        "bracketing-recall-label:S": 1.0,
        "bracketing-recall-label:VP": 0.0,
        "bracketing-recall-length:2": 0.5,
        "bracketing-recall-depth:1": 0.5,
    }
    rows = _rows(labels=("S", "VP"), lengths=(2,), depths=(1,))
    for name, value in expected.items():
        metric = rows[name]
        metric.update(pred, GOLD, key=ItemId("one"))
        assert metric.compute().value == value


def test_all_three_bracketing_baselines_and_normal_form_counts_report() -> None:
    names = (
        "bracketing-baseline:left",
        "bracketing-baseline:right",
        "bracketing-baseline:balanced",
        "normal-form-derivation-count",
        "unconstrained-derivation-count",
        "normal-form-retention",
    )
    rows = _rows(derivation_counts=True)
    for name in names:
        metric = rows[name]
        metric.update(prediction(Bracketing(4, ())), GOLD, key=ItemId("one"))
        assert metric.compute().value is not None


def test_the_disabled_census_never_requests_a_derivation_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    held_out = packaged_held_out_pass()
    evaluator = PLUGIN.construct(
        _declaration(derivation_counts=False), held_out.context
    )

    def refused(*args: object, **kwargs: object) -> None:
        del args, kwargs
        msg = "the exact census was requested"
        raise RuntimeError(msg)

    monkeypatch.setattr(
        "lofi.eval._structural_diagnostics.derivation_count_report", refused
    )
    prediction = next(iter(evaluator.predict(_gold_lexical_output()).values()))

    assert prediction.normal_form_count is None
    assert prediction.unconstrained_count is None
    assert prediction.retention is None
    assert not any("derivation-count" in metric.name for metric in evaluator.metrics)


def test_the_enabled_census_reports_exact_counts_on_a_tiny_gold_frontier() -> None:
    held_out = packaged_held_out_pass()
    evaluator = PLUGIN.construct(_declaration(derivation_counts=True), held_out.context)
    prediction = next(iter(evaluator.predict(_gold_lexical_output()).values()))

    assert prediction.normal_form_count is not None
    assert prediction.unconstrained_count is not None
    assert prediction.normal_form_count <= prediction.unconstrained_count
    conventions = dict(evaluator.conventions.factors)
    assert conventions["derivation_count_parse"] == "exhaustive"
    assert conventions["derivation_count_semiring"] == "counting"
    assert conventions["derivation_count_uses_run_parse"] is False
    assert {metric.name for metric in evaluator.metrics}.issuperset(
        {
            "normal-form-derivation-count",
            "unconstrained-derivation-count",
            "normal-form-retention",
        }
    )


def test_the_exact_census_uses_the_declared_free_root_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    held_out = packaged_held_out_pass()
    evaluator = PLUGIN.construct(
        _declaration(derivation_counts=True, root_mode="free"),
        held_out.context,
    )
    goals: list[object] = []

    def record(
        inventories: object,
        sentences: object,
        *,
        source: object,
        goal: object,
        proposer: object,
        unary_depth: int,
        threads: int,
    ) -> tuple[DerivationCountRow, ...]:
        del inventories, source, proposer, unary_depth, threads
        goals.append(goal)
        words = cast("tuple[tuple[str, ...], ...]", sentences)[0]
        return (DerivationCountRow("recording", 0, words, 1, 1, 1.0),)

    monkeypatch.setattr(
        "lofi.eval._structural_diagnostics.derivation_count_report", record
    )
    evaluator.predict(_gold_lexical_output())

    assert goals == [FREE_ROOTS.goal(held_out.grammar.grammar.goal)]


def test_the_grammar_s_declared_threads_reach_the_census_parses() -> None:
    run_spec = RunSpecCodec(search_path=(PACKAGED_CONFIG_ROOT,))(
        CONFIG,
        overrides=("data.datasets.train.dataset.grammar.budget.threads=2",),
    )
    held_out = held_out_run(run_spec)
    evaluator = PLUGIN.construct(_declaration(derivation_counts=True), held_out.context)

    with patch("lofi.deduce._report.parse", wraps=parse) as parser:
        evaluator.predict(_gold_lexical_output(held_out))

    assert parser.call_count == 2
    assert {call.kwargs["threads"] for call in parser.call_args_list} == {2}


def test_the_rows_refuse_a_declaration_whose_names_they_do_not_compute() -> None:
    settings = _declaration(derivation_counts=False).settings()
    drifted = settings._replace(metrics=(*settings.metrics, "bracketing-recall-x:1"))

    with pytest.raises(ValueError, match="which no structural reading computes"):
        _structural_rows(drifted, LengthBand("all", None))
