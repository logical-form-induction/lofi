"""Ordered evaluation of predictions.

Each metric's value is computed over every scored item. Where an interval is
declared, the metric is recomputed over bootstrap resamples of the items it
scored, each resampled item scored again under a fresh key, and the
bias-corrected and accelerated interval of those recomputations is reported
beside the value. An item the metric leaves undefined is not resampled, since
the value does not rest on it; a value resting on fewer than two items has no
interval.
"""

from typing import TYPE_CHECKING, override

from lofi._types import ItemId
from lofi.eval import (
    Approximation,
    ControlRequired,
    ConventionVector,
    MetricResult,
    NormalFormMetric,
    Report,
    ReportIdentity,
    ReportRow,
    Resampling,
    StratifiedMetric,
    report,
)
from lofi.run._sinks import NullSink
from lofi.stats import Bootstrap, Interval

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from lofi.data import EvalDataset, Pairing
    from lofi.eval import Metric
    from lofi.eval._alignment import Split
    from lofi.formalism import Grammar
    from lofi.run._ports import ReportSink
    from lofi.spec import IntervalSettings, RunSpec


__all__ = ["Evaluator"]


class Evaluator[G, P]:
    """Evaluates one dataset with ordered metrics.

    Parameters
    ----------
    dataset : EvalDataset[G]
        Evaluation dataset.
    metrics : Iterable[Metric[P, G]]
        Metrics in report order.
    run_spec : RunSpec
        Validated run specification.
    grammar : Grammar
        Grammar used to produce predictions.
    freeze_id : str
        Freeze every row is reported against.
    fallback_conventions : ConventionVector
        Conventions for metrics that declare none.
    suite : str
        Evaluation suite name.
    task : str
        Evaluation task name.
    dataset_name : str
        Published dataset name.
    split : Split
        Evaluated dataset split.
    seed : int
        Run seed.
    normal_form : str | None
        Normal form name, if applicable.
    sink : ReportSink | None
        Report destination, or ``None`` to discard output.
    interval : IntervalSettings | None
        The bootstrap each value's interval is drawn by, seeded by the run
        seed, or None to report values without intervals.

    Raises
    ------
    TypeError
        If run identity inputs have invalid types.
    ValueError
        If identity metadata is invalid, no metrics are supplied, or names repeat.
    """

    __slots__ = (
        "_approximation",
        "_dataset",
        "_dataset_name",
        "_fallback_conventions",
        "_freeze_id",
        "_grammar",
        "_interval",
        "_metrics",
        "_normal_form",
        "_run_spec",
        "_seed",
        "_sink",
        "_split",
        "_suite",
        "_task",
    )

    def __init__(
        self,
        dataset: EvalDataset[G],
        metrics: Iterable[Metric[P, G]],
        *,
        run_spec: RunSpec,
        grammar: Grammar,
        freeze_id: str,
        fallback_conventions: ConventionVector,
        suite: str,
        task: str,
        dataset_name: str,
        split: Split,
        seed: int,
        approximation: Approximation,
        normal_form: str | None = None,
        sink: ReportSink | None = None,
        interval: IntervalSettings | None,
    ) -> None:
        self._dataset = dataset
        self._metrics = tuple(metrics)
        self._run_spec = run_spec
        self._grammar = grammar
        self._freeze_id = freeze_id
        self._fallback_conventions = fallback_conventions
        self._suite = suite
        self._task = task
        self._dataset_name = dataset_name
        self._split = split
        self._seed = seed
        self._normal_form = normal_form
        self._approximation = approximation
        self._sink: ReportSink = NullSink() if sink is None else sink
        self._interval = interval

        ReportIdentity(run_spec, grammar, fallback_conventions, freeze_id=freeze_id)

        if not self._metrics:
            msg = "an evaluator composes no metrics"
            raise ValueError(msg)

        seen: set[tuple[str, str | None]] = set()

        for metric in self._metrics:
            normal_form = (
                metric.normal_form
                if isinstance(metric, NormalFormMetric)
                else self._normal_form
            )
            identity = (metric.name, normal_form)

            if identity in seen:
                msg = (
                    f"two metrics report the name {metric.name!r} under "
                    f"normal form {normal_form!r}"
                )
                raise ValueError(msg)

            seen.add(identity)

    @property
    def metric_names(self) -> tuple[str, ...]:
        """Return metric names in report order."""
        return tuple(metric.name for metric in self._metrics)

    @property
    def sink(self) -> ReportSink:
        """Return the report sink."""
        return self._sink

    def evaluate(self, predictions: Mapping[ItemId, P]) -> Report:
        """Evaluate predictions and emit the report.

        Parameters
        ----------
        predictions : Mapping[ItemId, P]
            Predictions keyed by item identifier.

        Returns
        -------
        Report
            Metric rows and unmatched item identifiers.
        """
        join = self._dataset.join(predictions)
        rows: list[ReportRow] = []

        for metric in self._metrics:
            metric.reset()

            for pairing in join.pairs:
                metric.update(
                    pairing.prediction, pairing.gold.value, key=pairing.item_id
                )

            result = metric.compute()
            ci = (
                result.ci
                if self._interval is None or result.value is None
                else self._bootstrapped(metric, join.pairs, self._interval)
            )
            conventions = metric.conventions or self._fallback_conventions
            normal_form = (
                metric.normal_form
                if isinstance(metric, NormalFormMetric)
                else self._normal_form
            )
            rows.append(
                ReportRow(
                    identity=ReportIdentity(
                        self._run_spec,
                        self._grammar,
                        conventions,
                        freeze_id=self._freeze_id,
                    ),
                    method=self._task,
                    metric=metric.name,
                    result=MetricResult(
                        result.value,
                        result.n,
                        Resampling(result.resampling.unit, result.resampling.seed),
                        components=result.components,
                        ci=ci,
                    ),
                    approximation=self._approximation,
                    suite=self._suite,
                    task=self._task,
                    dataset=self._dataset_name,
                    split=self._split,
                    seed=self._seed,
                    normal_form=normal_form,
                    required_control=(
                        metric.required_control
                        if isinstance(metric, ControlRequired)
                        else None
                    ),
                )
            )

        completed = report(rows, unpredicted=join.unpredicted, unmatched=join.unmatched)
        self._sink.emit(completed)

        return completed

    def _bootstrapped(
        self,
        metric: Metric[P, G],
        pairs: Sequence[Pairing[G, P]],
        interval: IntervalSettings,
        /,
    ) -> Interval | None:
        """Return the bootstrap interval of a metric over the items it scored.

        The metric is left holding every pair again once the interval is drawn.

        Parameters
        ----------
        metric : Metric[P, G]
            The metric, holding every pair.
        pairs : Sequence[Pairing[G, P]]
            Every prediction joined to its gold analysis.
        interval : IntervalSettings
            The replicate budget and coverage.

        Returns
        -------
        Interval | None
            The bias-corrected and accelerated interval, or None where fewer
            than two items carry a defined value.
        """
        defined = metric.per_item()
        units = tuple(
            pairing for pairing in pairs if defined[pairing.item_id].value is not None
        )

        if len(units) < 2:
            return None

        strata: tuple[str, ...] | None = None
        if isinstance(metric, StratifiedMetric):
            declared = metric.strata()
            strata = tuple(declared[pairing.item_id] for pairing in units)
            counts: dict[str, int] = {}
            for label in strata:
                counts[label] = counts.get(label, 0) + 1
            if any(count < 2 for count in counts.values()):
                return None

        def statistic(sample: Sequence[Pairing[G, P]]) -> float:
            metric.reset()

            for index, pairing in enumerate(sample):
                metric.update(
                    pairing.prediction, pairing.gold.value, key=ItemId(str(index))
                )

            value = metric.compute().value

            return float("nan") if value is None else value

        drawn = Bootstrap(
            unit=metric.compute().resampling.unit,
            replicates=interval.replicates,
            seed=self._seed,
            level=interval.level,
            strata=strata,
        ).interval(units, statistic)
        metric.reset()

        for pairing in pairs:
            metric.update(pairing.prediction, pairing.gold.value, key=pairing.item_id)

        return Interval(drawn.bca.low, drawn.bca.high, drawn.bca.level)

    @override
    def __repr__(self) -> str:
        """Return a summary naming the metrics and where the report goes."""
        return (
            f"Evaluator({len(self._dataset)} items, "
            f"metrics={self.metric_names!r}, sink={self._sink.name!r})"
        )
