"""Boundary tests for bead predictor artifacts and prediction rows."""

import didactic.api as dx
import pytest
from bead.protocol import ResponseEncoding, ScaleType
from bead.protocol.anchor import SemanticPoles

from lofi.run._boundary.bead import (
    BeadPredictionDocument,
    BeadServingSpec,
    LoFIPredictorManifest,
    decode_serving,
)


def test_a_predictor_manifest_round_trips_the_real_response_encoding() -> None:
    encoding = ResponseEncoding(
        name="inference",
        n_levels=3,
        scale_type=ScaleType.ORDINAL,
        labels=("contradiction", "neutral", "entailment"),
        semantic_poles=SemanticPoles(low="contradiction", high="entailment"),
    )
    manifest = LoFIPredictorManifest(
        format="lofi-bead-predictor",
        hub_bundle="hub",
        serving=BeadServingSpec(
            response_encoding=encoding, inputs={"premise": "first", "hypothesis": "b"}
        ),
    )

    restored = LoFIPredictorManifest.model_validate_json(manifest.model_dump_json())
    served = restored.serving

    assert served.inputs == {"premise": "first", "hypothesis": "b"}
    assert served.response_encoding.labels == encoding.labels
    assert served.response_encoding.scale_type is ScaleType.ORDINAL
    assert served.response_encoding.semantic_poles is not None
    assert served.response_encoding.semantic_poles.as_tuple() == (
        "contradiction",
        "entailment",
    )


def test_a_predictor_manifest_and_prediction_row_have_one_closed_shape() -> None:
    with pytest.raises(dx.ValidationError, match="unknown field 'provider'"):
        LoFIPredictorManifest.model_validate(
            {
                "format": "lofi-bead-predictor",
                "hub_bundle": "hub",
                "serving": {
                    "response_encoding": {
                        "name": "inference",
                        "n_levels": 3,
                        "scale_type": "nominal",
                        "labels": ["entailment", "neutral", "contradiction"],
                    },
                    "inputs": {"premise": "a", "hypothesis": "b"},
                },
                "provider": "lofi",
            }
        )

    with pytest.raises(dx.ValidationError, match="unknown field 'metadata'"):
        BeadPredictionDocument.model_validate(
            {
                "item_id": "00000000-0000-0000-0000-000000000001",
                "probabilities": {
                    "entailment": 0.8,
                    "neutral": 0.1,
                    "contradiction": 0.1,
                },
                "predicted_class": "entailment",
                "confidence": 0.8,
                "metadata": {},
            }
        )


def test_a_serving_document_names_every_input_role_and_element() -> None:
    document = (
        b'{"response_encoding": {"name": "q", "n_levels": 2, "scale_type": '
        b'"binary", "labels": ["no", "yes"]}, "inputs": {"role": " "}}'
    )

    with pytest.raises(dx.ValidationError, match="named"):
        decode_serving(document)
