"""Serving-boundary tests for the LoFI bead predictor."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, NamedTuple, cast
from uuid import UUID

import numpy as np
import pytest
from bead.items import (
    Item,
    ItemElement,
    ItemTemplate,
    PresentationSpec,
    TaskSpec,
    TaskType,
)
from bead.protocol import ResponseEncoding, ScaleType
from bead.protocol.anchor import SemanticPoles
from bead.serving import (
    NLICase,
    PredictionCase,
    PredictorRegistry,
    ServingPredictor,
    run_predictor_conformance,
)

from lofi.judgment import InferenceText
from lofi.lf import InferenceLabel
from lofi.run import bead as serving
from lofi.run._boundary.bead import BeadServingSpec
from lofi.run.bead import BeadPredictionOperation, LoFIPredictor

INPUTS = {"premise": "premise", "hypothesis": "hypothesis"}

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


class Point:
    """A restored fixed point that records the text it scores."""

    def __init__(
        self,
        response_rows: tuple[tuple[float, ...], ...] = (
            (0.7, 0.2, 0.1),
            (0.1, 0.3, 0.6),
        ),
    ) -> None:
        self.response_rows = response_rows
        self.seen: list[tuple[InferenceText, ...]] = []

    @property
    def response_levels(self) -> int:
        return len(self.response_rows[0])

    @property
    def relation_labels(self) -> tuple[InferenceLabel, ...]:
        return (
            InferenceLabel.ENTAILS,
            InferenceLabel.CONTRADICTS,
            InferenceLabel.NEUTRAL,
        )

    def predict_responses(
        self, pairs: Sequence[InferenceText], /
    ) -> tuple[tuple[float, ...], ...]:
        held = tuple(pairs)
        self.seen.append(held)
        return self.response_rows[: len(held)]

    def predict_relations(
        self, pairs: Sequence[InferenceText], /
    ) -> tuple[tuple[float, ...], ...]:
        held = tuple(pairs)
        self.seen.append(held)
        return ((0.8, 0.05, 0.15),)[: len(held)]


class Release(NamedTuple):
    """The Hub release fields used by bead adaptation."""

    model_name: str
    grammar_identity: str


class Prepared(NamedTuple):
    """The prepared prediction site used by bead adaptation."""

    prediction: Point


class Loaded(NamedTuple):
    """A loaded Hub bundle fixture."""

    release: Release
    prepared: Prepared


class Restore:
    """Restore one fixture point while recording the requested run."""

    def __init__(self, point: Point) -> None:
        self.point = point
        self.runs: list[Path] = []

    def __call__(self, source: Path, /) -> Point:
        self.runs.append(source)
        return self.point


def _encoding() -> ResponseEncoding:
    """Return the real bead encoding used by the predictor fixtures."""
    return ResponseEncoding(
        name="inference",
        n_levels=3,
        scale_type=ScaleType.ORDINAL,
        labels=("entailment", "neutral", "contradiction"),
        semantic_poles=SemanticPoles(low="contradiction", high="entailment"),
    )


def _template(task_type: TaskType = "ordinal_scale") -> ItemTemplate:
    """Return one real bead inference template."""
    return ItemTemplate(
        name="inference",
        judgment_type="inference",
        task_type=task_type,
        task_spec=TaskSpec(prompt="Does the premise entail the hypothesis?"),
        presentation_spec=PresentationSpec(),
        elements=(
            ItemElement(element_type="text", element_name="premise"),
            ItemElement(element_type="text", element_name="hypothesis"),
        ),
    )


def _item(
    template: ItemTemplate,
    identifier: int,
    premise: str,
    hypothesis: str,
) -> Item:
    """Return one real bead inference item bound to its template."""
    return Item(
        id=UUID(int=identifier),
        item_template_id=template.id,
        rendered_elements={"premise": premise, "hypothesis": hypothesis},
    )


def _loaded(point: Point) -> Loaded:
    """Return the fixed loaded-Hub result for one point."""
    return Loaded(Release("lofi-inference", "grammar-17"), Prepared(point))


def test_a_hub_point_implements_bead_serving_with_real_items(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    point = Point()
    monkeypatch.setattr(serving, "load_hub_bundle", lambda _source: _loaded(point))
    predictor = LoFIPredictor.from_hub_bundle(
        tmp_path, BeadServingSpec(response_encoding=_encoding(), inputs=INPUTS)
    )
    template = _template()
    items = [
        _item(template, 1, "Every dog barked.", "A dog barked."),
        _item(template, 2, "No cat slept.", "A cat slept."),
    ]

    predictor.validate_item_compatibility(items[0], template)
    probabilities = predictor.predict_proba(items)
    nli = predictor.compute_nli("Every dog barked.", "A dog barked.")

    assert isinstance(predictor, ServingPredictor)
    assert predictor.model_name == "lofi-inference"
    assert predictor.model_version == "grammar-17"
    assert predictor.response_encoding.semantic_poles is not None
    assert predictor.response_encoding.semantic_poles.as_tuple() == (
        "contradiction",
        "entailment",
    )
    assert predictor.supported_task_types == ("ordinal_scale",)
    np.testing.assert_allclose(probabilities, ((0.7, 0.2, 0.1), (0.1, 0.3, 0.6)))
    assert nli == pytest.approx(
        {"entailment": 0.8, "neutral": 0.15, "contradiction": 0.05}
    )
    assert point.seen[0] == (
        InferenceText(str(items[0].id), "Every dog barked.", "A dog barked."),
        InferenceText(str(items[1].id), "No cat slept.", "A cat slept."),
    )

    with pytest.raises(ValueError, match="supports 'ordinal_scale', not 'categorical'"):
        predictor.validate_item_compatibility(items[0], _template("categorical"))
    with pytest.raises(ValueError, match="Item template ID mismatch"):
        predictor.validate_item_compatibility(items[0], _template())
    with pytest.raises(ValueError, match="2 participant identifiers"):
        predictor.predict_proba(items[:1], ["participant-1", "participant-2"])


@pytest.mark.parametrize(
    ("scale_type", "labels", "task_type", "response_row"),
    [
        (ScaleType.BINARY, ("no", "yes"), "binary", (0.4, 0.6)),
        (
            ScaleType.ORDINAL,
            ("low", "middle", "high"),
            "ordinal_scale",
            (0.2, 0.3, 0.5),
        ),
        (
            ScaleType.NOMINAL,
            ("entailment", "neutral", "contradiction"),
            "categorical",
            (0.7, 0.2, 0.1),
        ),
        (
            ScaleType.FORCED_CHOICE,
            ("first", "second"),
            "forced_choice",
            (0.8, 0.2),
        ),
    ],
)
def test_each_response_family_passes_bead_serving_conformance(
    scale_type: ScaleType,
    labels: tuple[str, ...],
    task_type: TaskType,
    response_row: tuple[float, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    point = Point((response_row,))
    monkeypatch.setattr(serving, "load_hub_bundle", lambda _source: _loaded(point))
    encoding = ResponseEncoding(
        name="inference",
        n_levels=len(labels),
        scale_type=scale_type,
        labels=labels,
    )
    predictor = LoFIPredictor.from_hub_bundle(
        tmp_path, BeadServingSpec(response_encoding=encoding, inputs=INPUTS)
    )
    template = _template(task_type)
    item = _item(template, 1, "Every dog barked.", "A dog barked.")

    report = run_predictor_conformance(
        predictor,
        prediction_cases=[PredictionCase(item, template)],
        response_encoding=encoding,
        nli_cases=[NLICase("Every dog barked.", "A dog barked.")],
    )

    assert report.n_levels == len(labels)
    assert report.checks == (
        "identity",
        "task_compatibility",
        "predict_proba",
        "persistence",
        "nli",
    )


def test_native_persistence_wraps_one_unchanged_hub_bundle_and_loads_by_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    point = Point()
    sources: list[Path] = []

    def load(source: Path) -> Loaded:
        sources.append(source)
        return _loaded(point)

    monkeypatch.setattr(serving, "load_hub_bundle", load)
    hub = tmp_path / "source-hub"
    (hub / "states").mkdir(parents=True)
    (hub / "manifest.json").write_bytes(b"hub manifest bytes\n")
    (hub / "states" / "000000.pt").write_bytes(b"tensor bytes\x00\x01")
    predictor = LoFIPredictor.from_hub_bundle(
        hub, BeadServingSpec(response_encoding=_encoding(), inputs=INPUTS)
    )
    artifact = tmp_path / "bead-predictor"

    predictor.save(artifact)
    registry = PredictorRegistry[ServingPredictor]()
    registry.register("lofi", LoFIPredictor.load)
    restored = registry.load("lofi", artifact)

    assert registry.list_factories() == ("lofi",)
    assert sources == [hub, artifact / "hub"]
    assert (artifact / "hub" / "manifest.json").read_bytes() == (
        hub / "manifest.json"
    ).read_bytes()
    assert (artifact / "hub" / "states" / "000000.pt").read_bytes() == (
        hub / "states" / "000000.pt"
    ).read_bytes()
    assert restored.response_encoding.semantic_poles is not None
    assert restored.response_encoding.semantic_poles.as_tuple() == (
        "contradiction",
        "entailment",
    )

    (artifact / "unexpected.txt").write_text(
        "not declared", encoding="utf-8", newline="\n"
    )
    with pytest.raises(ValueError, match=r"extra \['unexpected.txt'\]"):
        LoFIPredictor.load(artifact)


def test_native_persistence_refuses_a_destination_inside_its_source_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hub = tmp_path / "source-hub"
    hub.mkdir()
    monkeypatch.setattr(serving, "load_hub_bundle", lambda _source: _loaded(Point()))
    predictor = LoFIPredictor.from_hub_bundle(
        hub, BeadServingSpec(response_encoding=_encoding(), inputs=INPUTS)
    )

    with pytest.raises(ValueError, match="cannot be inside its source Hub bundle"):
        predictor.save(hub / "bead-predictor")

    assert tuple(hub.iterdir()) == ()


def _write_serving(path: Path, inputs: dict[str, str] = INPUTS) -> None:
    """Write one serving document over the current bead response encoding."""
    serving_document = BeadServingSpec(response_encoding=_encoding(), inputs=inputs)
    path.write_text(serving_document.model_dump_json(), encoding="utf-8", newline="\n")


def _write_items(path: Path) -> tuple[UUID, UUID]:
    """Write two current bead item documents."""
    template = _template()
    items = (
        _item(template, 1, "Every dog barked.", "A dog barked."),
        _item(template, 2, "No cat slept.", "A cat slept."),
    )
    path.write_text(
        "".join(f"{item.model_dump_json()}\n" for item in items),
        encoding="utf-8",
        newline="\n",
    )
    return items[0].id, items[1].id


def test_current_bead_documents_score_through_the_command_boundary(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    document = tmp_path / "serving.json"
    items = tmp_path / "items.jsonl"
    destination = tmp_path / "predictions.jsonl"
    point = Point()
    restore = Restore(point)
    _write_serving(document)
    identifiers = _write_items(items)

    written = BeadPredictionOperation(restore)(run, document, items, destination)

    records = tuple(
        cast("dict[str, object]", json.loads(line))
        for line in destination.read_text(encoding="utf-8").splitlines()
    )
    assert written == 2
    assert restore.runs == [run]
    assert records == (
        {
            "item_id": str(identifiers[0]),
            "probabilities": {
                "entailment": 0.7,
                "neutral": 0.2,
                "contradiction": 0.1,
            },
            "predicted_class": "entailment",
            "confidence": 0.7,
        },
        {
            "item_id": str(identifiers[1]),
            "probabilities": {
                "entailment": 0.1,
                "neutral": 0.3,
                "contradiction": 0.6,
            },
            "predicted_class": "contradiction",
            "confidence": 0.6,
        },
    )


def test_bead_prediction_refuses_an_existing_destination_before_restoring(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "predictions.jsonl"
    destination.write_text("keep\n", encoding="utf-8", newline="\n")
    restore = Restore(Point())

    with pytest.raises(FileExistsError, match=str(destination)):
        BeadPredictionOperation(restore)(
            tmp_path / "run",
            tmp_path / "serving.json",
            tmp_path / "items.jsonl",
            destination,
        )

    assert destination.read_text(encoding="utf-8") == "keep\n"
    assert restore.runs == []
    assert "load_bead_model" not in serving.__all__


def test_a_serving_document_mapping_other_roles_than_the_point_is_refused(
    tmp_path: Path,
) -> None:
    document = tmp_path / "serving.json"
    _write_serving(document, {"words": "premise"})
    _write_items(tmp_path / "items.jsonl")

    with pytest.raises(ValueError, match="reads the input roles"):
        BeadPredictionOperation(Restore(Point()))(
            tmp_path / "run", document, tmp_path / "items.jsonl", tmp_path / "out"
        )
