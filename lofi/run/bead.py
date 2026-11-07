"""Serve trained LoFI points through bead's structural predictor protocol."""

from __future__ import annotations

import shutil
from math import fsum
from pathlib import Path
from tempfile import mkdtemp
from typing import TYPE_CHECKING, Self, final

import numpy as np
from bead.items import Item, ItemTemplate, TaskType
from bead.items.validation import validate_item
from bead.protocol import ResponseEncoding, scale_type_to_task_type

from lofi._checks import ensure_finite, ensure_name
from lofi.judgment.bead import (
    InferenceText,
    PointInputs,
    TrainedPointPredictor,
)
from lofi.lf import InferenceLabel
from lofi.run._boundary.bead import (
    BeadPredictionDocument,
    BeadServingSpec,
    LoFIPredictorManifest,
    decode_serving,
)
from lofi.run.hub import load_hub_bundle, restore_trained_point

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from lofi.train import PreparedTraining

__all__ = [
    "BeadPredictionOperation",
    "LoFIPredictor",
    "predict_bead_items",
]

_MANIFEST = "manifest.json"
_HUB_BUNDLE = "hub"
_FORMAT = "lofi-bead-predictor"


def _inputs(serving: BeadServingSpec, /) -> PointInputs:
    """Return the declared role mapping, checked against the point's roles."""
    return PointInputs(serving.inputs, roles=InferenceText.ROLES)


def _probability_rows(
    rows: Sequence[Sequence[float]],
    /,
    *,
    count: int,
    width: int,
    source: str,
) -> np.ndarray:
    """Validate detached probability rows once at the serving boundary."""
    held = tuple(
        tuple(ensure_finite(f"a {source} probability", value) for value in row)
        for row in rows
    )
    if len(held) != count:
        msg = f"{source} returned {len(held)} rows for {count} bead items"
        raise ValueError(msg)

    for row in held:
        if len(row) != width:
            msg = f"{source} returned width {len(row)}, not {width}"
            raise ValueError(msg)
        if (
            any(value < 0.0 or value > 1.0 for value in row)
            or not abs(fsum(row) - 1.0) <= 1e-6
        ):
            msg = f"a {source} row is not a probability distribution: {row!r}"
            raise ValueError(msg)

    if held:
        return np.asarray(held, dtype=np.float64)
    return np.empty((0, width), dtype=np.float64)


def _responses(
    point: TrainedPointPredictor,
    encoding: ResponseEncoding,
    items: Sequence[Item],
    inputs: PointInputs,
    /,
) -> np.ndarray:
    """Return checked response probabilities for real bead items."""
    pairs = tuple(InferenceText.of(item, inputs) for item in items)
    return _probability_rows(
        point.predict_responses(pairs),
        count=len(items),
        width=encoding.n_levels,
        source="trained point",
    )


def _prediction(prepared: PreparedTraining, /) -> TrainedPointPredictor:
    """Return the prediction capability a grounded run published."""
    prediction = prepared.prediction

    if prediction is None:
        msg = "the trained point publishes no inference prediction capability"
        raise ValueError(msg)

    return prediction


@final
class LoFIPredictor:
    """Expose one Hub-bundled trained point through bead serving."""

    __slots__ = (
        "_hub_bundle",
        "_inputs",
        "_model_name",
        "_model_version",
        "_point",
        "_response_encoding",
    )

    def __init__(
        self,
        model_name: str,
        model_version: str,
        response_encoding: ResponseEncoding,
        inputs: PointInputs,
        point: TrainedPointPredictor,
        hub_bundle: Path,
        /,
    ) -> None:
        self._model_name = ensure_name("a bead model name", model_name)
        self._model_version = ensure_name("a bead model version", model_version)
        if point.response_levels != response_encoding.n_levels:
            msg = (
                f"trained point predicts {point.response_levels} response levels for "
                f"bead encoding {response_encoding.name!r}, which declares "
                f"{response_encoding.n_levels}"
            )
            raise ValueError(msg)
        self._response_encoding = response_encoding
        self._inputs = inputs
        self._point = point
        self._hub_bundle = hub_bundle

    @property
    def model_name(self) -> str:
        """Return the stable Hub release name."""
        return self._model_name

    @property
    def model_version(self) -> str:
        """Return the released grammar identity."""
        return self._model_version

    @classmethod
    def from_hub_bundle(
        cls,
        source: str | Path,
        serving: BeadServingSpec,
        /,
    ) -> Self:
        """Adapt one unchanged LoFI Hub bundle to bead serving.

        Raises
        ------
        ValueError
            If the serving declaration maps other input roles than the point
            publishes.
        """
        bundle = Path(source)
        loaded = load_hub_bundle(bundle)
        return cls(
            loaded.release.model_name,
            loaded.release.grammar_identity,
            serving.response_encoding,
            _inputs(serving),
            _prediction(loaded.prepared),
            bundle,
        )

    @property
    def response_encoding(self) -> ResponseEncoding:
        """Return the bead encoding that names probability columns."""
        return self._response_encoding

    @property
    def inputs(self) -> PointInputs:
        """Return the element each of the point's input roles reads."""
        return self._inputs

    @property
    def supported_task_types(self) -> tuple[TaskType, ...]:
        """Return the task kind bead derives from the response scale."""
        return (scale_type_to_task_type(self._response_encoding.scale_type),)

    def validate_item_compatibility(
        self, item: Item, item_template: ItemTemplate
    ) -> None:
        """Validate the LoFI inference and response shapes of one bead item."""
        (supported,) = self.supported_task_types
        if item_template.task_type != supported:
            msg = (
                f"the trained point supports {supported!r}, "
                f"not {item_template.task_type!r}"
            )
            raise ValueError(msg)
        errors = validate_item(item, item_template)
        if errors:
            msg = (
                f"the bead item is incompatible with its template: {'; '.join(errors)}"
            )
            raise ValueError(msg)
        InferenceText.of(item, self._inputs)

    def predict_proba(
        self,
        items: list[Item],
        participant_ids: list[str] | None = None,
    ) -> np.ndarray:
        """Return one declared response distribution per bead item."""
        if participant_ids is not None:
            if len(participant_ids) != len(items):
                msg = (
                    f"prediction has {len(items)} items and "
                    f"{len(participant_ids)} participant identifiers"
                )
                raise ValueError(msg)
            for participant in participant_ids:
                ensure_name("a bead participant identifier", participant)

        return _responses(self._point, self._response_encoding, items, self._inputs)

    def compute_nli(self, premise: str, hypothesis: str) -> dict[str, float]:
        """Return bead's three named NLI probabilities for one text pair."""
        predicted = _probability_rows(
            self._point.predict_relations(
                (InferenceText("direct-nli", premise, hypothesis),)
            ),
            count=1,
            width=3,
            source="trained point NLI",
        )
        labels = self._point.relation_labels
        expected = (
            InferenceLabel.ENTAILS,
            InferenceLabel.NEUTRAL,
            InferenceLabel.CONTRADICTS,
        )
        if len(labels) != 3 or set(labels) != set(expected):
            msg = (
                "three-way NLI requires entailment, neutral and contradiction: "
                f"{labels!r}"
            )
            raise ValueError(msg)
        row = predicted[0]
        return {
            "entailment": float(row[labels.index(InferenceLabel.ENTAILS)]),
            "neutral": float(row[labels.index(InferenceLabel.NEUTRAL)]),
            "contradiction": float(row[labels.index(InferenceLabel.CONTRADICTS)]),
        }

    def save(self, path: str | Path) -> None:
        """Wrap the unchanged Hub bundle in one native bead artifact."""
        destination = Path(path)
        if destination.exists():
            msg = f"bead predictor destination already exists: {destination}"
            raise FileExistsError(msg)
        if not destination.parent.is_dir():
            msg = f"bead predictor parent is not a directory: {destination.parent}"
            raise FileNotFoundError(msg)

        source = self._hub_bundle.resolve()
        target = destination.resolve()
        if target.is_relative_to(source):
            msg = (
                "bead predictor destination cannot be inside its source Hub bundle: "
                f"{destination}"
            )
            raise ValueError(msg)

        staging = Path(mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
        try:
            shutil.copytree(self._hub_bundle, staging / _HUB_BUNDLE)
            manifest = LoFIPredictorManifest(
                format=_FORMAT,
                hub_bundle=_HUB_BUNDLE,
                serving=BeadServingSpec(
                    response_encoding=self._response_encoding,
                    inputs=dict(self._inputs.elements),
                ),
            )
            (staging / _MANIFEST).write_text(
                manifest.model_dump_json(), encoding="utf-8", newline="\n"
            )
            staging.rename(destination)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Restore one predictor from its strict native wrapper artifact."""
        source = Path(path)
        if not source.is_dir():
            msg = f"bead predictor artifact is not a directory: {source}"
            raise FileNotFoundError(msg)

        entries = frozenset(child.name for child in source.iterdir())
        expected = frozenset({_MANIFEST, _HUB_BUNDLE})
        if entries != expected:
            missing = sorted(expected - entries)
            extra = sorted(entries - expected)
            msg = f"bead predictor wrapper has missing {missing} and extra {extra}"
            raise ValueError(msg)

        manifest = LoFIPredictorManifest.model_validate_json(
            (source / _MANIFEST).read_text(encoding="utf-8")
        )
        return cls.from_hub_bundle(source / manifest.hub_bundle, manifest.serving)


@final
class BeadPredictionOperation:
    """Read bead documents and score them through one restored LoFI point."""

    __slots__ = ("_restore",)

    def __init__(self, restore: Callable[[Path], TrainedPointPredictor], /) -> None:
        self._restore = restore

    def __call__(
        self,
        run: Path,
        serving: Path,
        items: Path,
        destination: Path,
        /,
    ) -> int:
        """Write one prediction record per bead item and return the row count.

        The serving document declares the response encoding and the element
        each of the point's input roles reads.

        Raises
        ------
        ValueError
            If the serving document is refused, maps other input roles than the
            point publishes, or an item renders a mapped element nowhere.
        """
        if destination.exists():
            msg = f"the bead prediction rows file {destination} exists"
            raise FileExistsError(msg)

        declaration = decode_serving(serving.read_bytes())
        inputs = _inputs(declaration)
        declared = declaration.response_encoding
        judged = [
            Item.model_validate_json(stripped)
            for line in items.read_text(encoding="utf-8").splitlines()
            if (stripped := line.strip())
        ]
        probabilities = _responses(self._restore(run), declared, judged, inputs)
        records = _prediction_rows(declared, judged, probabilities)
        text = "".join(f"{record.model_dump_json()}\n" for record in records)

        with destination.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)

        return len(records)


def _prediction_rows(
    encoding: ResponseEncoding,
    items: Sequence[Item],
    probabilities: np.ndarray,
    /,
) -> tuple[BeadPredictionDocument, ...]:
    """Lower checked probabilities to bead prediction records."""
    records = []
    for item, row in zip(items, probabilities, strict=True):
        paired = tuple(zip(encoding.labels, row, strict=True))
        predicted, confidence = max(paired, key=lambda pair: pair[1])
        records.append(
            BeadPredictionDocument(
                item_id=str(item.id),
                probabilities={label: float(value) for label, value in paired},
                predicted_class=predicted,
                confidence=float(confidence),
            )
        )
    return tuple(records)


def _restore_prediction(run: Path, /) -> TrainedPointPredictor:
    """Restore one run and return its published inference prediction site."""
    return _prediction(restore_trained_point(run))


predict_bead_items = BeadPredictionOperation(_restore_prediction)
"""The concrete bead prediction operation supplied by the process entry point."""
