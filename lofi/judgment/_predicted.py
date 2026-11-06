"""Judgment datasets derived from a trained inference point's predictions.

A predicted source's items are the bead items its declared item-construction
document constructs. Each item is read by the trained point through the
declared mapping from the point's input roles to the item's elements, and each
row's factors are read from the item by the declared attribute expressions.
"""

from math import fsum, isfinite
from types import MappingProxyType
from typing import TYPE_CHECKING, final

from lofi._frozen import FrozenValue
from lofi.judgment._attributes import ItemFactors, item_key
from lofi.judgment._design import JudgmentRow, RowColumns
from lofi.judgment._items import construct_items
from lofi.judgment._spec import DesignProvenance, decode_item_construction
from lofi.judgment.bead import InferenceText, PointInputs, TrainedPointPredictor
from lofi.registry import Registry, RegistryError

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path

    from bead.items.item import Item

    from lofi.lf import InferenceLabel
    from lofi.spec.judgment_sources import PredictedJudgmentSourceSettings

__all__ = [
    "PREDICTED_JUDGMENT_SOURCE_PLUGIN",
    "PredictedJudgment",
    "PredictedJudgmentSource",
    "PredictedJudgmentSourcePlugin",
    "predicted_judgment_source_for",
]


def _probability_rows(
    rows: Iterable[Iterable[float]], width: int, name: str, /
) -> tuple[tuple[float, ...], ...]:
    held = tuple(tuple(float(value) for value in row) for row in rows)
    for row in held:
        if len(row) != width:
            msg = f"producer {name} probabilities have width {len(row)}, not {width}"
            raise ValueError(msg)
        if (
            any(not isfinite(value) or value < 0.0 or value > 1.0 for value in row)
            or abs(fsum(row) - 1.0) > 1e-6
        ):
            msg = f"producer {name} row is not a probability distribution: {row!r}"
            raise ValueError(msg)
    return held


@final
class PredictedJudgment(FrozenValue):
    """One producer point's exact response and relation distributions."""

    _noun = "predicted judgment"
    __slots__ = ("_item_id", "_relations", "_responses")

    def __init__(
        self,
        item_id: str,
        responses: tuple[float, ...],
        relations: Mapping[InferenceLabel, float],
    ) -> None:
        object.__setattr__(self, "_item_id", item_id)
        object.__setattr__(self, "_responses", responses)
        object.__setattr__(self, "_relations", MappingProxyType(dict(relations)))

    @property
    def item_id(self) -> str:
        """Return the source item's preserved identity."""
        return self._item_id

    @property
    def response_probabilities(self) -> tuple[float, ...]:
        """Return exact predicted response probabilities in declared label order."""
        return self._responses

    @property
    def relation_probabilities(self) -> Mapping[InferenceLabel, float]:
        """Return exact predicted NLI probabilities by producer label."""
        return self._relations


@final
class PredictedJudgmentSource(FrozenValue):
    """Serve a trained point's modal responses as a judgment design.

    Exact response and NLI probabilities remain published alongside the hard
    modal rows consumed by the existing response-model fitting boundary.
    """

    _noun = "predicted judgment source"
    __slots__ = ("_factors", "_predictions", "_rows", "_settings", "_skips")

    def __init__(
        self,
        settings: PredictedJudgmentSourceSettings,
        items: Iterable[Item],
        predictor: TrainedPointPredictor,
        /,
    ) -> None:
        held = tuple(items)
        if not held:
            msg = "a predicted judgment source needs items"
            raise ValueError(msg)
        factors = ItemFactors(settings.factors, held)
        inputs = PointInputs(settings.inputs, roles=InferenceText.ROLES)
        if predictor.response_levels != len(settings.labels):
            msg = (
                f"producer predicts {predictor.response_levels} response levels, "
                f"not the declared {len(settings.labels)}"
            )
            raise ValueError(msg)
        texts = tuple(InferenceText.of(item, inputs) for item in held)
        labels = predictor.relation_labels
        if len(labels) < 2 or len(set(labels)) != len(labels):
            msg = f"producer relation labels are not distinct: {labels!r}"
            raise ValueError(msg)
        responses = _probability_rows(
            predictor.predict_responses(texts), len(settings.labels), "response"
        )
        relations = _probability_rows(
            predictor.predict_relations(texts), len(labels), "relation"
        )
        if len(responses) != len(held) or len(relations) != len(held):
            msg = "producer predictions do not cover the declared item set"
            raise ValueError(msg)
        predictions = tuple(
            PredictedJudgment(
                item_key(item),
                tuple(float(value) for value in response),
                {
                    label: float(value)
                    for label, value in zip(labels, relation, strict=True)
                },
            )
            for item, response, relation in zip(held, responses, relations, strict=True)
        )
        rows = tuple(
            JudgmentRow(
                settings.participant,
                prediction.item_id,
                max(
                    range(len(prediction.response_probabilities)),
                    key=prediction.response_probabilities.__getitem__,
                ),
                RowColumns(factors.levels(prediction.item_id), {}),
            )
            for prediction in predictions
        )
        object.__setattr__(self, "_settings", settings)
        object.__setattr__(self, "_predictions", predictions)
        object.__setattr__(self, "_rows", rows)
        object.__setattr__(self, "_factors", factors.names)
        object.__setattr__(self, "_skips", MappingProxyType({}))

    @property
    def response_levels(self) -> int:
        """Return the declared response width."""
        return len(self._settings.labels)

    @property
    def factors(self) -> tuple[str, ...]:
        """Return the declared factors every row records, in column order."""
        return self._factors

    @property
    def skips(self) -> Mapping[str, int]:
        """Return the empty disposition count: each prediction becomes a row."""
        return self._skips

    @property
    def predictions(self) -> tuple[PredictedJudgment, ...]:
        """Return exact producer distributions in item order."""
        return self._predictions

    def rows(self) -> tuple[JudgmentRow, ...]:
        """Return one modal pseudo-judgment per item."""
        return self._rows

    def provenance(self, tag: str, /) -> DesignProvenance:
        """Return manifest provenance including the exact producer identity."""
        return DesignProvenance(
            tag=tag,
            checkpoint=None,
            checkpoint_dtype=None,
            unigram_digest=None,
            corpus_digests={},
            seed=None,
            participants=None,
            items=None,
            levels=None,
            population_digest=None,
            producer_run=self._settings.producer_run,
            producer_point=self._settings.producer_point,
        )


@final
class PredictedJudgmentSourcePlugin:
    """Construct the current trained-point judgment source."""

    __slots__ = ()

    def construct(
        self,
        settings: PredictedJudgmentSourceSettings,
        root: Path,
        predictor: TrainedPointPredictor,
        /,
    ) -> PredictedJudgmentSource:
        """Return predicted judgments over the declared item construction.

        Parameters
        ----------
        settings : PredictedJudgmentSourceSettings
            The resolved declaration.
        root : Path
            The directory the declared item-construction document is relative
            to.
        predictor : TrainedPointPredictor
            The producer point.
        """
        path = root / settings.item_set
        if not path.is_file():
            msg = f"the declared item set {settings.item_set!r} is no file under {root}"
            raise ValueError(msg)
        items = construct_items(
            decode_item_construction(path.read_bytes()), path.parent
        )
        return PredictedJudgmentSource(settings, items, predictor)


PREDICTED_JUDGMENT_SOURCE_PLUGIN = PredictedJudgmentSourcePlugin()
"""The registered current trained-point judgment source."""


def predicted_judgment_source_for(
    settings: PredictedJudgmentSourceSettings,
    root: Path,
    predictor: TrainedPointPredictor,
    /,
) -> PredictedJudgmentSource:
    """Construct the installed predicted source for resolved settings."""
    member = Registry[object]("judgment_source").get("predicted")
    if not isinstance(member, PredictedJudgmentSourcePlugin):
        msg = "judgment-source entry 'predicted' is not a predicted-source plugin"
        raise RegistryError(msg, family="judgment_source")
    return member.construct(settings, root, predictor)
