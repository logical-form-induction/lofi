"""Held-out grammaticality and calibration over declared root mass."""

from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, override

from lofi._types import ReadoutId
from lofi.data import Annotation, EvalDataset, Item, Token
from lofi.eval._boundary import (
    RootMassDocument,
    RootMassItemDocument,
    encode_root_mass,
)
from lofi.eval._calibration import Forecast, brier_score, expected_calibration_error
from lofi.eval._ports import EvaluationArtifact
from lofi.eval._values import ConventionVector, MetricResult, Resampling
from lofi.eval.grammaticality import grammaticality_report
from lofi.nn.loss._readout import (
    MASS_AXES,
    log_complement,
    log_mass,
)
from lofi.registry import ComponentPlugin
from lofi.spec.evaluators.grammaticality import (
    GrammaticalityEvaluatorSettings,
    GrammaticalityEvaluatorSpec,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lofi._types import ItemId
    from lofi.eval import HeldOutEvaluationContext
    from lofi.eval._ports import GrammaticalityItems, HeldOutData, HeldOutItems
    from lofi.nn.loss._values import ObjectiveOutputs

__all__ = ["PLUGIN", "GrammaticalityEvaluator"]


type _Measure = Literal[
    "roc_auc",
    "pr_auc",
    "accuracy_at_youden",
    "accuracy_at_half",
    "mcc",
    "noise_adjusted_auc",
    "auc_ceiling",
    "ece",
    "brier",
]


class _GrammaticalityMetric:
    """Accumulate one row of the grammaticality battery."""

    __slots__ = ("_bins", "_conventions", "_flip_rate", "_items", "_measure", "_name")

    def __init__(
        self,
        name: str,
        measure: _Measure,
        *,
        flip_rate: float,
        bins: int,
        conventions: ConventionVector,
    ) -> None:
        self._name = name
        self._measure = measure
        self._flip_rate = flip_rate
        self._bins = bins
        self._conventions = conventions
        self._items: dict[ItemId, tuple[float, bool]] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def conventions(self) -> ConventionVector:
        return self._conventions

    @property
    def order_invariant(self) -> bool:
        return True

    def reset(self) -> None:
        self._items = {}

    def update(self, pred: float, gold: Literal[False, True], *, key: ItemId) -> None:
        value = (pred, gold)
        standing = self._items.get(key)
        if standing is not None and standing != value:
            msg = f"item {key!r} has two grammaticality predictions"
            raise ValueError(msg)
        self._items[key] = value

    def compute(self) -> MetricResult:
        scheme = Resampling("sentence")
        if not self._items:
            return MetricResult(None, 0, scheme)
        scores = tuple(score for score, _ in self._items.values())
        labels = tuple(label for _, label in self._items.values())
        if self._measure == "ece":
            if len(scores) < self._bins:
                return MetricResult(None, len(scores), scheme)
            result = expected_calibration_error(
                (
                    Forecast(key, (1.0 - score, score), int(label))
                    for key, (score, label) in self._items.items()
                ),
                category=1,
                bins=self._bins,
            )
            return MetricResult(
                result.value, result.n, scheme, components=result.components
            )
        if self._measure == "brier":
            result = brier_score(
                (
                    Forecast(key, (1.0 - score, score), int(label))
                    for key, (score, label) in self._items.items()
                ),
                category=1,
            )
            return MetricResult(
                result.value, result.n, scheme, components=result.components
            )
        if all(labels) or not any(labels):
            return MetricResult(None, len(labels), scheme)
        report = grammaticality_report(labels, scores, flip_rate=self._flip_rate)
        return MetricResult(float(getattr(report, self._measure)), report.n, scheme)

    def per_item(self) -> Mapping[ItemId, MetricResult]:
        """Return one defined resampling unit per labeled sentence."""
        return MappingProxyType(
            {key: MetricResult(1.0, 1, Resampling("sentence")) for key in self._items}
        )

    def strata(self) -> Mapping[ItemId, str]:
        """Return each sentence's observed grammaticality class."""
        return MappingProxyType(
            {
                key: "grammatical" if label else "ungrammatical"
                for key, (_, label) in self._items.items()
            }
        )


class GrammaticalityEvaluator:
    """Score every labeled held-out item from a published root-mass readout."""

    __slots__ = ("_conventions", "_metrics", "_readout", "_root_mass")

    def __init__(self, settings: GrammaticalityEvaluatorSettings, /) -> None:
        self._readout = ReadoutId(settings.output)
        self._conventions = ConventionVector.from_settings(
            settings.conventions
        ).extended(
            {
                "readout": settings.readout,
                "readout_output": settings.output,
                "flip_rate": settings.flip_rate,
                "bins": settings.bins,
            }
        )
        measures: tuple[_Measure, ...] = (
            "roc_auc",
            "pr_auc",
            "accuracy_at_youden",
            "accuracy_at_half",
            "mcc",
            "noise_adjusted_auc",
            "auc_ceiling",
            "ece",
            "brier",
        )
        self._metrics = tuple(
            _GrammaticalityMetric(
                name,
                measure,
                flip_rate=settings.flip_rate,
                bins=settings.bins,
                conventions=self._conventions,
            )
            for name, measure in zip(settings.metrics, measures, strict=True)
        )
        self._root_mass: dict[ItemId, tuple[float, float]] = {}

    @property
    def conventions(self) -> ConventionVector:
        return self._conventions

    @property
    def metrics(self) -> tuple[_GrammaticalityMetric, ...]:
        return self._metrics

    def held_out(self, data: HeldOutData, split: str, /) -> EvalDataset[bool]:
        """Return a retained partition's grammaticality judgments."""
        return self.dataset(data.judgment_batches(split))

    def dataset(self, batches: Sequence[GrammaticalityItems], /) -> EvalDataset[bool]:
        items: list[Item] = []
        annotations: list[Annotation[bool]] = []
        for batch in batches:
            rows = zip(batch.item_ids, batch.words, batch.labels, strict=True)
            for item_id, words, label in rows:
                item = Item(
                    item_id, tuple(Token(at, word) for at, word in enumerate(words))
                )
                items.append(item)
                annotations.append(Annotation(item_id, label))
        return EvalDataset(items, annotations)

    def predict[BatchT: HeldOutItems](
        self, outputs: ObjectiveOutputs[BatchT], /
    ) -> Mapping[ItemId, float]:
        readout = outputs.readouts.get(self._readout)
        if readout is None:
            offered = ", ".join(sorted(outputs.readouts)) or "nothing"
            msg = (
                f"grammaticality reads {self._readout!r}, and the pass "
                f"published {offered}"
            )
            raise ValueError(msg)
        if readout.axes != MASS_AXES:
            msg = (
                f"grammaticality reads a mass pair over {MASS_AXES}, not {readout.axes}"
            )
            raise ValueError(msg)
        cells = readout.gather(
            [(row, 0, len(words)) for row, words in enumerate(outputs.batch.words)]
        )
        masses = log_mass(cells).detach()
        complements = log_complement(cells).detach()
        pairs = zip(
            outputs.batch.item_ids,
            masses.tolist(),
            complements.tolist(),
            strict=True,
        )
        for item_id, mass, complement in pairs:
            value = (mass, complement)
            standing = self._root_mass.get(item_id)
            if standing is not None and standing != value:
                msg = f"item {item_id!r} has two published root masses"
                raise ValueError(msg)
            self._root_mass[item_id] = value

        probabilities = masses.exp().tolist()
        return MappingProxyType(
            dict(zip(outputs.batch.item_ids, probabilities, strict=True))
        )

    def artifacts(self) -> tuple[EvaluationArtifact, ...]:
        """Return the published root masses in held-out item order."""
        document = RootMassDocument(
            readout=self._readout,
            items=tuple(
                RootMassItemDocument(
                    item_id=item_id,
                    log_mass=mass,
                    log_complement=complement,
                )
                for item_id, (mass, complement) in self._root_mass.items()
            ),
        )
        return (EvaluationArtifact("root-mass.json", encode_root_mass(document)),)

    @override
    def __repr__(self) -> str:
        return f"GrammaticalityEvaluator(readout={self._readout!r})"


def _build(
    settings: GrammaticalityEvaluatorSettings,
    context: HeldOutEvaluationContext,
    /,
) -> GrammaticalityEvaluator:
    del context
    return GrammaticalityEvaluator(settings)


PLUGIN: Final = ComponentPlugin(spec=GrammaticalityEvaluatorSpec, build=_build)
