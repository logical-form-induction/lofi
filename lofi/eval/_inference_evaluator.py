"""Classification, artifact controls, and proof reproduction for inference."""

import json
from math import fsum
from statistics import fmean
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, NamedTuple, cast

from lofi.data import Annotation, Categorical, EvalDataset, Item
from lofi.eval._artifact import Hypothesis, HypothesisFeaturizer, LabelHead
from lofi.eval._calibration import (
    Forecast,
    brier_score,
    expected_calibration_error,
)
from lofi.eval._inference import InferenceLabelMetric, LabelReduction
from lofi.eval._inference_proof import (
    InferenceProof,
    open_inference_proof_step,
    prove_inference_beam,
)
from lofi.eval._values import ConventionVector, MetricResult, Resampling
from lofi.lf import InferenceCell, InferenceLabel
from lofi.registry import ComponentPlugin
from lofi.spec.evaluators.inference import (
    InferenceEvaluatorSettings,
    InferenceEvaluatorSpec,
    InferencePartitions,
    inference_evaluation_metrics,
)
from lofi.stats import Bootstrap

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from lofi._types import ItemId
    from lofi.eval import HeldOutItems, InferenceEvaluationContext, InferenceItems
    from lofi.eval._ports import HeldOutData
    from lofi.nn.loss._values import ObjectiveOutputs

__all__ = ["PLUGIN", "InferenceEvaluator", "InferencePrediction"]


class InferencePrediction(NamedTuple):
    """One published label forecast and optional proof reproduction."""

    label: str | None
    probabilities: tuple[float, ...]
    proof: InferenceProof | None


class _ControlScore(NamedTuple):
    """Control scores emitted beside the model's test rows."""

    rows: Mapping[str, MetricResult]


type _PairValue = Callable[[InferencePrediction], float | None]
"""Reads one pair's contribution to a row, absent where the row is undefined."""


class _Calibration(NamedTuple):
    """A calibration row over decided pairs' forecasts.

    Parameters
    ----------
    score : Callable[[Sequence[Forecast]], MetricResult]
        The calibration score over the decided forecasts.
    minimum : int
        The fewest decided forecasts the score is defined over.
    """

    score: Callable[[Sequence[Forecast]], MetricResult]
    minimum: int


class _PairRow(NamedTuple):
    """A row aggregated from one value per pair.

    Parameters
    ----------
    value : _PairValue
        One pair's contribution.
    total : bool
        Whether the row reports the sum of the contributions rather than their
        mean.
    components : Callable[[Iterable[InferencePrediction]], Mapping[str, float]]
        The breakdown reported beside the value.
    """

    value: _PairValue
    total: bool
    components: Callable[[Iterable[InferencePrediction]], Mapping[str, float]]


def _no_components(
    predictions: Iterable[InferencePrediction], /
) -> Mapping[str, float]:
    """Return the empty breakdown of a scalar row."""
    del predictions
    return MappingProxyType({})


def _abstention_causes(
    predictions: Iterable[InferencePrediction], /
) -> Mapping[str, float]:
    """Return proof abstentions counted by their cause."""
    counted: dict[str, float] = {}
    for pred in predictions:
        if pred.proof is not None:
            for cause, count in pred.proof.abstentions:
                counted[cause] = counted.get(cause, 0.0) + count
    return MappingProxyType(counted)


class _ProofCells(NamedTuple):
    """The first candidate cell and the reference cell of one proved pair."""

    first: InferenceCell | None
    reference: InferenceCell | None

    @property
    def settled(self) -> bool:
        """Return whether the first candidate decided a nondegenerate cell."""
        return self.first is not None and self.first is not InferenceCell.DEGENERATE

    @property
    def reference_settled(self) -> bool:
        """Return whether the reference decided a nondegenerate cell."""
        reference = self.reference
        return reference is not None and reference is not InferenceCell.DEGENERATE


def _cells(proof: InferenceProof, /) -> _ProofCells:
    """Return the first candidate and reference cells of one proof."""
    return _ProofCells(
        None if not proof.candidates else proof.candidates[0], proof.reference
    )


def _proved(value: Callable[[InferenceProof], float | None], /) -> _PairValue:
    """Return a contribution read off a pair's proof, absent without one."""
    return lambda pred: None if pred.proof is None else value(pred.proof)


def _preserving_rate(proof: InferenceProof, /) -> float | None:
    """Return whether the first candidate reproduces a settled reference."""
    cells = _cells(proof)
    if not (cells.settled and cells.reference_settled):
        return None
    return float(cells.first == cells.reference)


def _preserving_rate_in_beam(proof: InferenceProof, /) -> float | None:
    """Return whether any decided candidate of the beam reproduces the reference."""
    reference = proof.reference
    decided = tuple(
        cell
        for cell in proof.candidates
        if cell is not None and cell is not InferenceCell.DEGENERATE
    )
    settled = _cells(proof).reference_settled
    if reference in decided and settled:
        return 1.0
    unresolved = any(cell is None for cell in proof.candidates)
    return 0.0 if decided and settled and not unresolved else None


def _preserving_coverage(proof: InferenceProof, /) -> float | None:
    """Return whether both the candidate and the reference settled."""
    cells = _cells(proof)
    return float(cells.settled and cells.reference_settled)


def _preserving_abstention(proof: InferenceProof, /) -> float | None:
    """Return whether the candidate or the reference went undecided."""
    cells = _cells(proof)
    return float(cells.first is None or cells.reference is None)


def _preserving_both(proof: InferenceProof, /) -> float | None:
    """Return whether the candidate or the reference was degenerate."""
    cells = _cells(proof)
    return float(
        cells.first is InferenceCell.DEGENERATE
        or cells.reference is InferenceCell.DEGENERATE
    )


def _relaxation_gap(native_labels: tuple[str, ...], /) -> _PairValue:
    """Return the gap between forecast entailment mass and the proved cell."""
    entails = native_labels.index(InferenceLabel.ENTAILS.value)

    def value(pred: InferencePrediction, /) -> float | None:
        proof = pred.proof
        if proof is None:
            return None
        cells = _cells(proof)
        if not cells.settled:
            return None
        return abs(
            pred.probabilities[entails] - float(cells.first is InferenceCell.ENTAILS)
        )

    return value


def _mean(value: _PairValue, /) -> _PairRow:
    """Return a row averaging one contribution over its defined pairs."""
    return _PairRow(value, total=False, components=_no_components)


def _sum(
    value: _PairValue,
    components: Callable[
        [Iterable[InferencePrediction]], Mapping[str, float]
    ] = _no_components,
    /,
) -> _PairRow:
    """Return a row summing one contribution over its defined pairs."""
    return _PairRow(value, total=True, components=components)


def _calibration_score(
    score: Callable[[Sequence[Forecast]], MetricResult], /
) -> Callable[[Sequence[Forecast]], MetricResult]:
    """Return a calibration score as a pair-resampled row."""

    def scored(forecasts: Sequence[Forecast], /) -> MetricResult:
        result = score(forecasts)
        return MetricResult(
            result.value, result.n, Resampling("pair"), components=result.components
        )

    return scored


def _inference_rows(
    native_labels: tuple[str, ...], bins: int, depth: int | None, /
) -> tuple[tuple[str, _PairRow | _Calibration], ...]:
    """Return every declared forecast and proof row beside its reading.

    Parameters
    ----------
    native_labels : tuple[str, ...]
        The response scale the forecasts are over.
    bins : int
        The calibration bin count.
    depth : int | None
        The declared joint beam depth of proof evaluation, or None when no
        proof is declared.

    Returns
    -------
    tuple[tuple[str, _PairRow | _Calibration], ...]
        One named reading per row, in report order.
    """
    rows: tuple[tuple[str, _PairRow | _Calibration], ...] = (
        ("entailment-coverage", _mean(lambda pred: float(pred.label is not None))),
        (
            "entailment-calibration-ece",
            _Calibration(
                _calibration_score(
                    lambda forecasts: expected_calibration_error(forecasts, bins=bins)
                ),
                bins,
            ),
        ),
        (
            "entailment-calibration-brier",
            _Calibration(_calibration_score(brier_score), 1),
        ),
    )
    if depth is None:
        return rows
    return (
        *rows,
        ("entailment-preserving-rate", _mean(_proved(_preserving_rate))),
        (
            f"entailment-preserving-rate-at-{depth}",
            _mean(_proved(_preserving_rate_in_beam)),
        ),
        ("entailment-preserving-coverage", _mean(_proved(_preserving_coverage))),
        ("entailment-preserving-abstention", _mean(_proved(_preserving_abstention))),
        ("entailment-preserving-both", _mean(_proved(_preserving_both))),
        ("relaxation-gap", _mean(_relaxation_gap(native_labels))),
        ("proof-calls", _sum(_proved(lambda proof: float(proof.calls)))),
        ("proof-memo-hits", _sum(_proved(lambda proof: float(proof.memo_hits)))),
        ("proof-cpu-seconds", _sum(_proved(lambda proof: proof.cpu_seconds))),
        ("proof-questions", _sum(_proved(lambda proof: float(proof.questions)))),
        ("proof-wall-seconds", _sum(_proved(lambda proof: proof.wall_seconds))),
        (
            "proof-abstentions",
            _sum(
                _proved(
                    lambda proof: float(sum(count for _, count in proof.abstentions))
                ),
                _abstention_causes,
            ),
        ),
    )


class _InferenceMetric:
    """Accumulate one inference row over rich pair predictions."""

    __slots__ = (
        "_classes",
        "_conventions",
        "_items",
        "_label",
        "_name",
        "_native_labels",
        "_reading",
        "_reduction",
    )

    def __init__(
        self,
        name: str,
        classes: tuple[str, ...],
        native_labels: tuple[str, ...],
        conventions: ConventionVector,
        *,
        reading: InferenceLabelMetric | _PairRow | _Calibration,
        reduction: LabelReduction | None,
    ) -> None:
        self._name = name
        self._classes = classes
        self._native_labels = native_labels
        self._conventions = conventions
        self._label = reading if isinstance(reading, InferenceLabelMetric) else None
        self._reading = reading
        self._reduction = reduction
        self._items: dict[ItemId, tuple[InferencePrediction, Categorical]] = {}

    @property
    def name(self) -> str:
        """Return this row's declared name."""
        return self._name

    @property
    def conventions(self) -> ConventionVector:
        """Return the common scoring conventions."""
        return self._conventions

    @property
    def order_invariant(self) -> bool:
        """Return whether item insertion order affects this row."""
        return True

    def reset(self) -> None:
        """Clear accumulated pair occurrences."""
        self._items = {}
        if self._label is not None:
            self._label.reset()

    def update(
        self, pred: InferencePrediction, gold: Categorical, *, key: ItemId
    ) -> None:
        """Accumulate one pair occurrence and refuse conflicting duplicates."""
        standing = self._items.get(key)
        if standing is not None and standing != (pred, gold):
            msg = f"inference pair {key!r} has conflicting predictions"
            raise ValueError(msg)
        self._items[key] = (pred, gold)
        if self._label is not None:
            self._label.update(pred.label, gold, key=key)

    def _forecast(
        self, key: ItemId, pred: InferencePrediction, gold: Categorical
    ) -> Forecast:
        """Return a conditional forecast on the declared scored label scale."""
        probabilities = dict.fromkeys(self._classes, 0.0)
        for label, mass in zip(self._native_labels, pred.probabilities, strict=True):
            image = label if self._reduction is None else self._reduction.apply(label)
            probabilities[image] += mass
        total = fsum(probabilities.values())
        observed = cast("str", gold.label)
        image = observed if self._reduction is None else self._reduction.apply(observed)
        return Forecast(
            key,
            tuple(probabilities[label] / total for label in self._classes),
            self._classes.index(image),
        )

    def compute(self) -> MetricResult:
        """Compute the row over its defined pair occurrences."""
        reading = self._reading
        if isinstance(reading, InferenceLabelMetric):
            return reading.compute()
        scheme = Resampling("pair")
        if isinstance(reading, _Calibration):
            forecasts = tuple(
                self._forecast(key, pred, gold)
                for key, (pred, gold) in self._items.items()
                if pred.label is not None
            )
            if len(forecasts) < reading.minimum:
                return MetricResult(None, len(forecasts), scheme)
            return reading.score(forecasts)
        values = tuple(
            value
            for pred, _ in self._items.values()
            if (value := reading.value(pred)) is not None
        )
        if not values:
            return MetricResult(None, 0, scheme)
        total = fsum(values)
        return MetricResult(
            total if reading.total else total / len(values),
            len(values),
            scheme,
            components=dict(
                reading.components(pred for pred, _ in self._items.values())
            ),
        )

    def per_item(self) -> Mapping[ItemId, MetricResult]:
        """Return the pair occurrences on which the row is defined."""
        reading = self._reading
        if isinstance(reading, InferenceLabelMetric):
            return reading.per_item()
        scheme = Resampling("pair")
        return MappingProxyType(
            {
                key: MetricResult(
                    (1.0 if pred.label is not None else None)
                    if isinstance(reading, _Calibration)
                    else reading.value(pred),
                    1,
                    scheme,
                )
                for key, (pred, _) in self._items.items()
            }
        )

    def strata(self) -> Mapping[ItemId, str]:
        """Return label support strata, or one common stratum for other rows."""
        if self._label is not None:
            return self._label.strata()
        return MappingProxyType(dict.fromkeys(self._items, "all"))


class InferenceEvaluator:
    """Score inference forecasts beside development-fitted artifact controls."""

    __slots__ = (
        "_classes",
        "_context",
        "_conventions",
        "_fitted_ids",
        "_head",
        "_metrics",
        "_settings",
    )

    def __init__(
        self,
        context: InferenceEvaluationContext,
        settings: InferenceEvaluatorSettings,
        /,
    ) -> None:
        self._context = context
        self._settings = settings
        self._head: LabelHead | None = None
        self._fitted_ids: frozenset[ItemId] = frozenset()
        reduction = (
            None
            if settings.reduction is None
            else LabelReduction(dict(settings.reduction))
        )
        native_labels = context.labels
        classes = native_labels if reduction is None else reduction.images
        self._classes = classes
        self._conventions = ConventionVector.from_settings(
            settings.conventions
        ).extended(
            {
                "response_scale": json.dumps(classes, separators=(",", ":")),
                "label_reduction": None
                if settings.reduction is None
                else json.dumps(
                    dict(settings.reduction), sort_keys=True, separators=(",", ":")
                ),
                "calibration_bins": settings.bins,
                "control_fit_on": settings.partitions.fit_on,
                "score_on": settings.partitions.score_on,
                "calibration_conditioning": "decided represented label mass",
            }
        )
        measures = (
            ("accuracy", None),
            ("macro-f1", None),
            *(("precision", label) for label in classes),
            *(("recall", label) for label in classes),
            *(("f1", label) for label in classes),
            ("abstention", None),
        )
        labels = {
            name: InferenceLabelMetric(
                name,
                frozenset(context.labels),
                measure=measure,
                target=target,
                reduction=reduction,
                conventions=self._conventions,
            )
            for name, (measure, target) in zip(
                inference_evaluation_metrics(classes), measures, strict=True
            )
        }
        readings: tuple[
            tuple[str, InferenceLabelMetric | _PairRow | _Calibration], ...
        ] = (
            *labels.items(),
            *_inference_rows(
                native_labels,
                settings.bins,
                None if settings.proof is None else settings.proof.depth,
            ),
        )
        self._metrics = tuple(
            _InferenceMetric(
                name,
                classes,
                native_labels,
                self._conventions,
                reading=reading,
                reduction=reduction,
            )
            for name, reading in readings
        )
        if settings.proof is not None and context.interpretation is None:
            msg = "inference proof evaluation requires the retained interpretation"
            raise ValueError(msg)

    @property
    def conventions(self) -> ConventionVector:
        """Return declared response and scoring conventions."""
        return self._conventions

    @property
    def metrics(self) -> tuple[_InferenceMetric, ...]:
        """Return forecast and proof rows in declared order."""
        return self._metrics

    @property
    def alignment(self) -> InferencePartitions:
        """Return the exact control fitting and scoring partitions."""
        return self._settings.partitions

    def held_out(self, data: HeldOutData, split: str, /) -> EvalDataset[Categorical]:
        """Return a retained partition's labeled inference pairs."""
        return self.dataset(data.pair_batches(split))

    def dataset(self, batches: Sequence[InferenceItems], /) -> EvalDataset[Categorical]:
        """Join pair occurrences to their declared gold labels."""
        items: list[Item] = []
        annotations: list[Annotation[Categorical]] = []
        for batch in batches:
            for pair_id, label in zip(batch.pair_ids, batch.labels, strict=True):
                items.append(Item(pair_id, ()))
                annotations.append(
                    Annotation(
                        pair_id,
                        Categorical(label.value, frozenset(self._context.labels)),
                    )
                )
        return EvalDataset(items, annotations)

    def fit_alignment(
        self,
        predictions: Mapping[ItemId, InferencePrediction],
        gold: Mapping[ItemId, Categorical],
        /,
    ) -> LabelHead:
        """Fit and freeze hypothesis-only counts using development labels."""
        if self._head is not None:
            msg = "inference controls have already been fitted and frozen"
            raise ValueError(msg)
        if predictions.keys() != gold.keys():
            msg = (
                "inference controls require identical development prediction "
                "and gold inventories"
            )
            raise ValueError(msg)
        keys = tuple(sorted(gold))
        features = HypothesisFeaturizer()
        labels = tuple(self._reduced(cast("str", gold[key].label)) for key in keys)
        head = LabelHead(
            tuple(features(Hypothesis(self._context.tokens[key][1])) for key in keys),
            labels,
            inventory=frozenset(self._classes),
        )
        self._head = head
        self._fitted_ids = frozenset(keys)
        return head

    def _reduced(self, label: str) -> str:
        """Return the declared scored class of one native inference label."""
        reduction = self._settings.reduction
        return label if reduction is None else dict(reduction)[label]

    def score_alignment(
        self,
        predictions: Mapping[ItemId, InferencePrediction],
        gold: Mapping[ItemId, Categorical],
        /,
        *,
        seed: int,
    ) -> _ControlScore:
        """Score frozen controls on the disjoint test occurrence inventory."""
        head = self._head
        if head is None:
            msg = "inference controls must be fitted on development before test scoring"
            raise ValueError(msg)
        if self._fitted_ids.intersection(gold):
            msg = "inference controls cannot score their fitting occurrences"
            raise ValueError(msg)
        if predictions.keys() != gold.keys() or not gold:
            msg = "inference controls require identical nonempty test inventories"
            raise ValueError(msg)
        features = HypothesisFeaturizer()
        count = len(gold)
        majority = tuple(
            float(head.majority == self._reduced(cast("str", observed.label)))
            for observed in gold.values()
        )
        hypothesis = tuple(
            float(
                head.predict(features(Hypothesis(self._context.tokens[key][1])))
                == self._reduced(cast("str", observed.label))
            )
            for key, observed in gold.items()
        )
        bootstrap = Bootstrap(
            unit="pair",
            seed=seed,
            replicates=self._settings.interval.replicates,
            level=self._settings.interval.level,
        )
        return _ControlScore(
            MappingProxyType(
                {
                    name: MetricResult(
                        fmean(values),
                        count,
                        Resampling("pair", seed),
                        ci=None if count < 2 else bootstrap.interval(values, fmean).bca,
                    )
                    for name, values in (
                        ("majority-class", majority),
                        ("hypothesis-only", hypothesis),
                    )
                }
            )
        )

    def predict[BatchT: HeldOutItems](
        self, outputs: ObjectiveOutputs[BatchT], /
    ) -> Mapping[ItemId, InferencePrediction]:
        """Read published label probabilities and, when declared, form beams."""
        grounded = outputs.entailment
        if grounded is None:
            msg = "inference evaluation reads entailment, and none was published"
            raise ValueError(msg)
        batch = cast("InferenceItems", outputs.batch)
        probabilities = grounded.log_probabilities.tensor.detach().exp().tolist()
        decided = grounded.decided.tolist()
        classes = tuple(label.value for label in grounded.labels)
        if classes != self._context.labels:
            msg = (
                f"grounding labels {classes!r} disagree with the data's "
                f"declared labels {self._context.labels!r}"
            )
            raise ValueError(msg)
        proof_settings = self._settings.proof
        interpretation = self._context.interpretation
        if proof_settings is not None and interpretation is not None:
            open_inference_proof_step(interpretation.service)
        predictions: dict[ItemId, InferencePrediction] = {}
        for row, (key, values, settled) in enumerate(
            zip(batch.pair_ids, probabilities, decided, strict=True)
        ):
            proof = None
            if proof_settings is not None and interpretation is not None:
                proof = prove_inference_beam(
                    interpretation,
                    proof_settings,
                    self._context.forms[key],
                    outputs.forms[batch.premises[row]],
                    outputs.forms[batch.hypotheses[row]],
                )
            label = (
                classes[max(range(len(values)), key=values.__getitem__)]
                if settled
                else None
            )
            predictions[key] = InferencePrediction(label, tuple(values), proof)
        return MappingProxyType(predictions)


def _build(
    settings: InferenceEvaluatorSettings, context: InferenceEvaluationContext, /
) -> InferenceEvaluator:
    """Build the unified inference evaluator in its retained pair context."""
    return InferenceEvaluator(context, settings)


PLUGIN: Final = ComponentPlugin(spec=InferenceEvaluatorSpec, build=_build)
