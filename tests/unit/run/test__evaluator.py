from typing import TYPE_CHECKING, cast

import pytest

from lofi._types import ItemId
from lofi.data import Annotation, EvalDataset, Item, Token
from lofi.eval import (
    NO_ENUMERATION_BUDGET,
    Approximation,
    ConventionVector,
    MetricResult,
    Report,
    ReportIdentity,
    Resampling,
)
from lofi.run._evaluator import Evaluator
from lofi.run._sinks import NullSink
from lofi.spec import IntervalSettings
from lofi.stats import Bootstrap, Interval

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from lofi._types import JsonValue
    from lofi.formalism import Grammar
    from lofi.spec import RunSpec


_ITEMS = Resampling("item")


class ExactMatch:
    def __init__(
        self,
        name: str,
        *,
        conventions: ConventionVector | None,
        interval: Interval | None,
    ) -> None:
        self._name = name
        self._conventions = conventions
        self._interval = interval
        self.seen: list[ItemId] = []
        self._hits: dict[ItemId, bool] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def conventions(self) -> ConventionVector | None:
        return self._conventions

    @property
    def order_invariant(self) -> bool:
        return True

    def reset(self) -> None:
        self.seen = []
        self._hits = {}

    def update(self, pred: str, gold: str, *, key: ItemId) -> None:
        self.seen.append(key)
        self._hits[key] = pred == gold

    def compute(self) -> MetricResult:
        if not self._hits:
            return MetricResult(None, 0, _ITEMS)

        matched = sum(self._hits.values())

        return MetricResult(
            matched / len(self._hits), len(self._hits), _ITEMS, ci=self._interval
        )

    def per_item(self) -> Mapping[ItemId, MetricResult]:
        return {
            key: MetricResult(float(hit), 1, _ITEMS) for key, hit in self._hits.items()
        }


class NormalFormExactMatch(ExactMatch):
    def __init__(self, normal_form: str) -> None:
        super().__init__("exact-match", conventions=None, interval=None)
        self._normal_form = normal_form

    @property
    def normal_form(self) -> str:
        return self._normal_form


class RecordingSink:
    def __init__(self) -> None:
        self.written: list[Report] = []

    @property
    def name(self) -> str:
        return "recording"

    def emit(self, report: Report) -> None:
        self.written.append(report)


def _conventions(*, minimum_length: int = 3) -> ConventionVector:
    return ConventionVector(
        punctuation="strip",
        unary_chains="collapse",
        trivial_spans="drop",
        minimum_length=minimum_length,
        aggregation="sentence",
        coverage="zero",
        self_arcs="forbid",
        gold_derivation="canonical",
    )


class _Config:
    def model_dump(self) -> dict[str, JsonValue]:
        return {"run": "evaluator-test"}


class _Grammar:
    fingerprint = "grammar:evaluator-test"


def _evaluator(
    dataset: EvalDataset[str],
    metrics: Iterable[ExactMatch],
    *,
    sink: RecordingSink | None = None,
    fallback_conventions: ConventionVector | None = None,
    interval: IntervalSettings | None = None,
) -> Evaluator[str, str]:
    return Evaluator(
        dataset,
        metrics,
        run_spec=cast("RunSpec", _Config()),
        grammar=cast("Grammar", _Grammar()),
        freeze_id="freeze/1",
        fallback_conventions=(
            _conventions() if fallback_conventions is None else fallback_conventions
        ),
        suite="grammar-comparison",
        task="acceptability",
        dataset_name="megaacceptability",
        split="test",
        seed=7,
        approximation=Approximation(NO_ENUMERATION_BUDGET, residual=None),
        normal_form="eisner",
        sink=sink,
        interval=interval,
    )


def _dataset(gold: Mapping[str, str]) -> EvalDataset[str]:
    items = [Item(ItemId(key), (Token(0, "kim"), Token(1, "slept"))) for key in gold]
    annotations = [Annotation(ItemId(key), value) for key, value in gold.items()]

    return EvalDataset(items, annotations)


def _predictions(predicted: Mapping[str, str]) -> dict[ItemId, str]:
    return {ItemId(key): value for key, value in predicted.items()}


def test_the_rows_come_back_one_per_metric_in_the_order_composed() -> None:
    evaluator = _evaluator(
        _dataset({"s-1": "yes", "s-2": "no"}),
        [
            ExactMatch("exact-match", conventions=None, interval=None),
            ExactMatch(
                "agreement", conventions=_conventions(minimum_length=0), interval=None
            ),
        ],
    )
    report = evaluator.evaluate(_predictions({"s-1": "yes", "s-2": "yes"}))

    assert evaluator.metric_names == ("exact-match", "agreement")
    assert [row.metric for row in report] == ["exact-match", "agreement"]


def test_a_row_carries_the_metric_number_beside_the_shared_columns() -> None:
    evaluator = _evaluator(
        _dataset({"s-1": "yes", "s-2": "no"}),
        [
            ExactMatch(
                "exact-match", conventions=None, interval=Interval(0.2, 0.8, 0.95)
            )
        ],
    )
    (row,) = evaluator.evaluate(_predictions({"s-1": "yes", "s-2": "yes"})).rows

    assert row.value == 0.5
    assert row.n == 2
    assert (row.ci_lo, row.ci_hi, row.ci_level) == (0.2, 0.8, 0.95)
    assert row.suite == "grammar-comparison"
    assert row.split == "test"
    assert row.seed == 7
    assert len(row.config_hash) == 64
    assert row.grammar_fingerprint == "grammar:evaluator-test"
    assert row.normal_form == "eisner"


def test_the_conventions_a_metric_ran_under_reach_the_row_as_a_hash() -> None:
    evaluator = _evaluator(
        _dataset({"s-1": "yes"}),
        [
            ExactMatch(
                "agreement", conventions=_conventions(minimum_length=0), interval=None
            ),
            ExactMatch("exact-match", conventions=None, interval=None),
        ],
    )
    scored, unscored = evaluator.evaluate(_predictions({"s-1": "yes"})).rows

    assert scored.identity == ReportIdentity(
        _Config(),
        _Grammar(),
        _conventions(minimum_length=0),
        freeze_id="freeze/1",
    )
    assert unscored.identity == ReportIdentity(
        _Config(), _Grammar(), _conventions(), freeze_id="freeze/1"
    )
    assert scored.convention_hash != unscored.convention_hash


def test_an_empty_dataset_returns_a_report_rather_than_raising() -> None:
    evaluator = _evaluator(
        _dataset({}),
        [ExactMatch("exact-match", conventions=None, interval=None)],
    )
    report = evaluator.evaluate({})

    (row,) = report.rows
    assert row.value is None
    assert row.n == 0
    assert report.unpredicted == ()
    assert report.unmatched == ()


def test_predictions_that_miss_and_overshoot_are_reported_not_dropped() -> None:
    metric = ExactMatch("exact-match", conventions=None, interval=None)
    evaluator = _evaluator(
        _dataset({"s-1": "yes", "s-2": "no", "s-3": "yes"}), [metric]
    )
    report = evaluator.evaluate(_predictions({"s-1": "yes", "s-9": "no"}))

    (row,) = report.rows
    assert row.n == 1
    assert report.unpredicted == (ItemId("s-2"), ItemId("s-3"))
    assert report.unmatched == (ItemId("s-9"),)
    assert metric.seen == [ItemId("s-1")]


def test_a_metric_sees_the_items_in_the_order_the_set_carries_them() -> None:
    metric = ExactMatch("exact-match", conventions=None, interval=None)
    evaluator = _evaluator(
        _dataset({"s-3": "yes", "s-1": "no", "s-2": "yes"}), [metric]
    )
    evaluator.evaluate(_predictions({"s-1": "no", "s-2": "yes", "s-3": "yes"}))

    assert metric.seen == [ItemId("s-3"), ItemId("s-1"), ItemId("s-2")]


def test_each_evaluation_scores_only_the_predictions_it_was_handed() -> None:
    metric = ExactMatch("exact-match", conventions=None, interval=None)
    evaluator = _evaluator(_dataset({"s-1": "yes", "s-2": "no"}), [metric])

    first = evaluator.evaluate(_predictions({"s-1": "yes", "s-2": "no"}))
    second = evaluator.evaluate(_predictions({"s-1": "no"}))

    assert first.rows[0].value == 1.0
    assert second.rows[0].value == 0.0
    assert second.rows[0].n == 1
    assert set(metric.per_item()) == {ItemId("s-1")}


def test_the_report_returned_is_the_report_the_sink_was_handed() -> None:
    sink = RecordingSink()
    evaluator = _evaluator(
        _dataset({"s-1": "yes"}),
        [ExactMatch("exact-match", conventions=None, interval=None)],
        sink=sink,
    )
    report = evaluator.evaluate(_predictions({"s-1": "yes"}))

    assert sink.written == [report]
    assert evaluator.sink is sink


def test_an_evaluation_with_no_sink_configured_still_returns_its_report() -> None:
    evaluator = _evaluator(
        _dataset({"s-1": "yes"}),
        [ExactMatch("exact-match", conventions=None, interval=None)],
    )
    report = evaluator.evaluate(_predictions({"s-1": "yes"}))

    assert isinstance(evaluator.sink, NullSink)
    assert evaluator.sink.name == "null"
    assert report.rows[0].value == 1.0


def test_an_evaluator_composing_no_metrics_is_refused() -> None:
    with pytest.raises(ValueError, match="composes no metrics"):
        _evaluator(_dataset({"s-1": "yes"}), [])


def test_an_evaluator_requires_explicit_fallback_conventions() -> None:
    constructor = cast("Callable[..., object]", Evaluator)

    with pytest.raises(TypeError, match="fallback_conventions"):
        constructor(
            _dataset({"s-1": "yes"}),
            [ExactMatch("exact-match", conventions=None, interval=None)],
            run_spec=_Config(),
            grammar=_Grammar(),
            freeze_id="freeze/1",
            suite="grammar-comparison",
            task="acceptability",
            dataset_name="megaacceptability",
            split="test",
            seed=7,
        )


def test_two_metrics_reporting_one_name_are_refused() -> None:
    with pytest.raises(ValueError, match="report the name"):
        _evaluator(
            _dataset({"s-1": "yes"}),
            [
                ExactMatch("exact-match", conventions=None, interval=None),
                ExactMatch("exact-match", conventions=_conventions(), interval=None),
            ],
        )


def test_one_metric_name_may_report_distinct_normal_form_rows() -> None:
    evaluator = _evaluator(
        _dataset({"s-1": "yes"}),
        [NormalFormExactMatch("eisner"), NormalFormExactMatch("none")],
    )

    rows = evaluator.evaluate(_predictions({"s-1": "yes"})).rows

    assert [row.metric for row in rows] == ["exact-match", "exact-match"]
    assert [row.normal_form for row in rows] == ["eisner", "none"]


def test_an_evaluator_repr_names_what_it_scores_and_where_it_reports() -> None:
    evaluator = _evaluator(
        _dataset({"s-1": "yes", "s-2": "no"}),
        [ExactMatch("exact-match", conventions=None, interval=None)],
    )

    assert repr(evaluator) == (
        "Evaluator(2 items, metrics=('exact-match',), sink='null')"
    )


def _graded(count: int) -> tuple[dict[str, str], dict[ItemId, str]]:
    gold = {f"s-{index}": "yes" for index in range(count)}
    predicted = _predictions(
        {key: "yes" if index % 3 else "no" for index, key in enumerate(gold)}
    )

    return gold, predicted


def test_a_declared_interval_resamples_the_items_the_metric_scored() -> None:
    gold, predicted = _graded(12)
    metric = ExactMatch("exact-match", conventions=None, interval=None)
    evaluator = _evaluator(
        _dataset(gold), [metric], interval=IntervalSettings(200, 0.9)
    )

    (row,) = evaluator.evaluate(predicted).rows
    hits = [float(predicted[ItemId(key)] == value) for key, value in gold.items()]
    expected = (
        Bootstrap(unit="item", replicates=200, seed=7, level=0.9)
        .interval(hits, lambda sample: sum(sample) / len(sample))
        .bca
    )

    assert row.value == pytest.approx(8 / 12)
    assert (row.ci_lo, row.ci_hi, row.ci_level) == (
        pytest.approx(expected.low),
        pytest.approx(expected.high),
        0.9,
    )
    assert row.ci_lo is not None
    assert row.ci_hi is not None
    assert row.ci_lo < row.ci_hi
    assert set(metric.per_item()) == {ItemId(key) for key in gold}


def test_the_interval_follows_the_declared_replicates_and_level() -> None:
    gold, predicted = _graded(12)
    rows = [
        _evaluator(
            _dataset(gold),
            [ExactMatch("exact-match", conventions=None, interval=None)],
            interval=settings,
        )
        .evaluate(predicted)
        .rows[0]
        for settings in (IntervalSettings(200, 0.9), IntervalSettings(200, 0.5))
    ]

    assert rows[0].ci_level == 0.9
    assert rows[1].ci_level == 0.5
    assert rows[1].ci_lo is not None
    assert rows[0].ci_lo is not None
    assert rows[1].ci_lo >= rows[0].ci_lo


def test_no_interval_is_drawn_unless_declared_or_over_one_item() -> None:
    gold, predicted = _graded(12)
    undeclared = _evaluator(
        _dataset(gold), [ExactMatch("exact-match", conventions=None, interval=None)]
    )
    single = _evaluator(
        _dataset({"s-0": "yes"}),
        [ExactMatch("exact-match", conventions=None, interval=None)],
        interval=IntervalSettings(200, 0.9),
    )

    assert undeclared.evaluate(predicted).rows[0].ci is None
    assert single.evaluate(_predictions({"s-0": "yes"})).rows[0].ci is None
