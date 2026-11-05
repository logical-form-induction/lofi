import json

import pytest

from lofi.eval._boundary import (
    RootMassDocument,
    RootMassItemDocument,
    ValidationError,
    decode_comparison_report,
    decode_manifest,
    decode_metrics,
    decode_object,
    decode_predictions,
    decode_preregistration,
    decode_protocol_register,
    decode_root_mass,
    decode_rows,
    encode_root_mass,
)


def test_analysis_documents_decode_to_typed_boundary_values() -> None:
    preregistration = decode_preregistration(
        json.dumps(
            {
                "grid": {"grammar": ["ccg", "ab"]},
                "hypotheses": [
                    {
                        "name": "ccg-over-ab",
                        "left": "ccg",
                        "right": "ab",
                        "family": "grammar",
                        "margin": 0.01,
                    }
                ],
            }
        )
    )
    register = decode_protocol_register(
        json.dumps(
            {
                "protocols": [
                    {
                        "name": "held-out",
                        "dataset": "test",
                        "falsifier": "the systems do not separate",
                        "smoke_test": False,
                    }
                ]
            }
        )
    )
    report = decode_comparison_report(
        {
            "comparisons": [
                {
                    "protocol": "held-out",
                    "falsifier": "the systems do not separate",
                    "smoke_test": False,
                    "statistic": 2.0,
                }
            ]
        }
    )

    assert preregistration.grid == {"grammar": ("ccg", "ab")}
    assert preregistration.hypotheses[0].margin == 0.01
    assert register.protocols[0].reason is None
    assert report.comparisons[0].protocol == "held-out"


def test_prediction_documents_use_named_span_and_constituent_fields() -> None:
    predictions = decode_predictions(
        json.dumps(
            {
                "system": "parser",
                "budget": {"depth": 3, "beam": 5, "mass": None},
                "items": [
                    {
                        "item_id": "s1",
                        "split": "test",
                        "length": 2,
                        "punctuation": [],
                        "gold": [{"start": 0, "end": 2, "label": "S", "depth": 0}],
                        "spans": [{"start": 0, "end": 2, "label": "c0"}],
                    }
                ],
            }
        )
    )

    assert predictions.items[0].gold[0].label == "S"
    assert predictions.items[0].spans is not None
    assert predictions.items[0].spans[0].label == "c0"

    with pytest.raises(ValidationError):
        decode_predictions(
            '{"system":"parser","budget":{"depth":3,"beam":5,"mass":null},'
            '"items":[{"item_id":"s1","split":"test","length":2,'
            '"punctuation":[],"gold":[[0,2,"S",0]],"spans":null}]}'
        )


def test_run_observation_documents_project_only_the_reducer_inputs() -> None:
    assert decode_object('{"unused": true}') == decode_object("{}")
    assert decode_manifest('{"status":"completed","unused":true}').status == "completed"

    metrics = decode_metrics(
        json.dumps(
            {
                "stage": "closed",
                "point": {"seed": 7, "unused": True},
                "fitting": {
                    "optimizer_steps": 4,
                    "epochs": [
                        {
                            "steps": [
                                {
                                    "metrics": {
                                        "grounded/cpu_seconds": 1.5,
                                        "grounded/wall_seconds": 2.0,
                                    }
                                }
                            ]
                        }
                    ],
                    "steps": [],
                },
            }
        )
    )
    rows = decode_rows('{"rows":[{"metric":"entailment-label-accuracy","value":0.75}]}')

    assert metrics.point.seed == 7
    assert metrics.fitting.epochs[0].steps[0].metrics["grounded/cpu_seconds"] == 1.5
    assert rows.rows[0].value == 0.75


def test_root_mass_artifacts_round_trip_the_published_log_pair() -> None:
    document = RootMassDocument(
        readout="root-mass",
        items=(
            RootMassItemDocument(item_id="item-1", log_mass=-0.25, log_complement=-1.5),
        ),
    )

    assert decode_root_mass(encode_root_mass(document)) == document


def test_root_mass_artifacts_require_distinct_items_and_log_probabilities() -> None:
    repeated = (
        '{"readout":"root-mass","items":['
        '{"item_id":"item-1","log_mass":-0.25,"log_complement":-1.5},'
        '{"item_id":"item-1","log_mass":-0.5,"log_complement":-1.0}]}'
    )

    with pytest.raises(ValidationError, match="repeats"):
        decode_root_mass(repeated.encode())

    with pytest.raises(ValidationError, match="exceeds zero"):
        decode_root_mass(
            b'{"readout":"root-mass","items":['
            b'{"item_id":"item-1","log_mass":0.25,"log_complement":-1.5}]}'
        )
