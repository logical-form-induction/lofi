"""A trained logical-form point conforms to bead's serving protocol."""

import json
from typing import TYPE_CHECKING, Final

import numpy as np
from bead.items import (
    Item,
    ItemElement,
    ItemTemplate,
    PresentationSpec,
    TaskSpec,
)
from bead.protocol import ResponseEncoding, ScaleType
from bead.serving import NLICase, PredictionCase, run_predictor_conformance
from tests.acceptance.driver import train

from lofi.judgment import InferenceText
from lofi.run._boundary.bead import BeadServingSpec
from lofi.run.bead import LoFIPredictor
from lofi.run.hub import export_trained_point, load_hub_bundle

if TYPE_CHECKING:
    from pathlib import Path

RELEASE: Final[dict[str, object]] = {
    "model_name": "lofi-inference",
    "datasets": ["lofi/synthetic-inference"],
    "languages": ["en"],
    "source_method": "generated",
    "license_expression": "CC-BY-4.0",
    "licenses": [{"identifier": "CC-BY-4.0", "applies_to": "generated-corpus"}],
    "eprint_refs": ["at://did:plc:lofi/pub.layers.eprint.eprint/1"],
    "code_uri": "https://github.com/logical-form-induction/lofi",
    "command": "lofi run --config inference-point.yaml --experiment-seed 5",
    "environment": "uv",
}


def _case() -> tuple[ItemTemplate, Item]:
    """Return one real bead inference case for a held-out text pair."""
    template = ItemTemplate(
        name="inference",
        judgment_type="inference",
        task_type="categorical",
        task_spec=TaskSpec(prompt="What relation holds?"),
        presentation_spec=PresentationSpec(),
        elements=(
            ItemElement(element_type="text", element_name="context"),
            ItemElement(element_type="text", element_name="claim"),
        ),
    )
    item = Item(
        item_template_id=template.id,
        rendered_elements={
            "context": "Every dog barked.",
            "claim": "A dog barked.",
        },
    )
    return template, item


def test_a_trained_inference_point_passes_bead_serving_conformance(
    tmp_path: Path,
) -> None:
    trained = train("inference-point.yaml", tmp_path / "experiment")
    release = tmp_path / "release.json"
    release.write_text(json.dumps(RELEASE), encoding="utf-8", newline="\n")
    bundle = tmp_path / "hub"
    export_trained_point(trained.directory, release, bundle)
    encoding = ResponseEncoding(
        name="inference",
        n_levels=3,
        scale_type=ScaleType.NOMINAL,
        labels=("entailment", "neutral", "contradiction"),
    )
    predictor = LoFIPredictor.from_hub_bundle(
        bundle,
        BeadServingSpec(
            response_encoding=encoding,
            inputs={"premise": "context", "hypothesis": "claim"},
        ),
    )
    template, item = _case()
    pair = InferenceText(
        str(item.id),
        item.rendered_elements["context"],
        item.rendered_elements["claim"],
    )
    loaded = load_hub_bundle(bundle)
    point = loaded.prepared.prediction
    if point is None:
        msg = "the trained inference point published no prediction capability"
        raise AssertionError(msg)

    expected = np.asarray(point.predict_responses((pair,)), dtype=np.float64)
    found = predictor.predict_proba([item])
    report = run_predictor_conformance(
        predictor,
        prediction_cases=[PredictionCase(item, template)],
        response_encoding=encoding,
        nli_cases=[NLICase(pair.premise, pair.hypothesis)],
    )

    np.testing.assert_allclose(found, expected)
    assert report.model_name == "lofi-inference"
    assert report.n_items == 1
    assert report.checks == (
        "identity",
        "task_compatibility",
        "predict_proba",
        "persistence",
        "nli",
    )
