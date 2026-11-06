"""Tests for active-learning acquisition, input and stopping declarations."""

import json
from typing import TYPE_CHECKING

import didactic.api as dx
import pytest

from lofi.spec.acquisition import (
    AgreementStopSettings,
    AsymptoteStopSettings,
    BeadCollectorSettings,
    BeadInitialSettings,
    CollectedInitialSettings,
    CrossValidationSettings,
    PointReadoutSettings,
    decode_active_learning,
)
from lofi.spec.held_out import HeldOutParticipantPolicy

if TYPE_CHECKING:
    from lofi._types import JsonValue


COLLECTED: dict[str, JsonValue] = {
    "kind": "collected",
    "stratify_by": "item['structure']",
    "items_per_stratum": 4,
}


def a_document(**changed: JsonValue) -> bytes:
    """Return an active-learning document with some fields replaced."""
    document: dict[str, JsonValue] = {
        "acquisition": {
            "kind": "rc2.1",
            "batch_size": 500,
            "allocate_by": "item['structure']",
            "constraints": [
                {"constraint_type": "uniqueness", "property_expression": "item['verb']"}
            ],
            "share_tolerance": 0.02,
        },
        "stopping": {"kind": "agreement", "threshold": 0.8, "basis": "lower-bound"},
        "max_rounds": 12,
        "confidence": 0.95,
        "bootstrap_samples": 200,
        "pool": "items.json",
        "factors": {"structure": "item['structure']", "verb": "item['verb']"},
        "initial": {"kind": "bead", "responses": "initial.jsonl"},
        "encoding": "encoding.json",
        "fit": "fit.json",
        "predictors": {
            "run": "runs/trained",
            "inputs": {"words": "sentence"},
            "tokenizer": {
                "backend": "whitespace",
                "language": "en",
                "model_name": None,
            },
            "readouts": {
                "score_complement": {
                    "readout": "spans",
                    "measure": "log-complement",
                },
                "score": {"readout": "spans", "measure": "log-mass"},
            },
        },
        "collector": {"kind": "bead"},
        "cross_validation": {
            "held_out_unit": "item",
            "splitter": "kfold",
            "folds": 3,
            "coverage_threshold": 0.0,
            "unseen_participant": "plug_in",
        },
    }
    document.update(changed)
    return json.dumps(document).encode()


def test_active_document_resolves_batch_and_quantitative_stopping_controls() -> None:
    settings = decode_active_learning(a_document()).settings()

    assert settings.acquisition.batch_size == 500
    assert settings.acquisition.allocate_by == "item['structure']"
    assert tuple(
        constraint.constraint_type for constraint in settings.acquisition.constraints
    ) == ("uniqueness",)
    assert settings.acquisition.share_tolerance == pytest.approx(0.02)
    assert settings.stopping == AgreementStopSettings(0.8, "lower-bound")
    assert settings.max_rounds == 12


def test_active_document_resolves_its_pool_judgments_fit_point_and_collector() -> None:
    settings = decode_active_learning(a_document()).settings()

    assert settings.pool == "items.json"
    assert dict(settings.factors) == {
        "structure": "item['structure']",
        "verb": "item['verb']",
    }
    assert settings.initial == BeadInitialSettings("bead", "initial.jsonl")
    assert settings.encoding == "encoding.json"
    assert settings.fit == "fit.json"
    assert settings.predictors.run == "runs/trained"
    assert dict(settings.predictors.inputs) == {"words": "sentence"}
    assert settings.predictors.tokenizer.backend == "whitespace"
    assert tuple(settings.predictors.readouts.items()) == (
        ("score", PointReadoutSettings("spans", "log-mass")),
        ("score_complement", PointReadoutSettings("spans", "log-complement")),
    )
    assert settings.collector == BeadCollectorSettings("bead")


def test_a_collected_start_resolves_the_items_drawn_from_each_stratum() -> None:
    settings = decode_active_learning(a_document(initial=COLLECTED)).settings()

    assert settings.initial == CollectedInitialSettings(
        "collected", "item['structure']", 4
    )


@pytest.mark.parametrize(
    "changed",
    [
        {"collector": {"kind": "registered-design", "design": "toy", "join": None}},
        {"initial": COLLECTED},
    ],
)
def test_one_bead_reader_requires_the_shared_encoding(
    changed: dict[str, JsonValue],
) -> None:
    settings = decode_active_learning(a_document(**changed)).settings()

    assert settings.encoding == "encoding.json"

    with pytest.raises(dx.ValidationError, match="declares no response encoding"):
        decode_active_learning(a_document(**changed, encoding=None))


def test_an_encoding_without_a_bead_reader_is_refused() -> None:
    simulated = {
        "initial": COLLECTED,
        "collector": {"kind": "registered-design", "design": "toy", "join": None},
    }

    settings = decode_active_learning(a_document(**simulated, encoding=None)).settings()

    assert settings.encoding is None

    with pytest.raises(dx.ValidationError, match="no bead reader"):
        decode_active_learning(a_document(**simulated))


def test_active_document_resolves_its_own_cross_validation() -> None:
    settings = decode_active_learning(a_document()).settings()

    assert settings.cross_validation == CrossValidationSettings(
        "item", "kfold", 3, 0.0, HeldOutParticipantPolicy.PLUG_IN
    )
    assert type(settings.cross_validation.unseen_participant) is (
        HeldOutParticipantPolicy
    )


def test_a_cross_validation_may_hold_out_one_fold_per_level_without_a_rule() -> None:
    settings = decode_active_learning(
        a_document(
            cross_validation={
                "held_out_unit": "participant",
                "splitter": "leave-one-group-out",
                "folds": None,
                "coverage_threshold": 1.0,
                "unseen_participant": None,
            }
        )
    ).settings()

    assert settings.cross_validation == CrossValidationSettings(
        "participant", "leave-one-group-out", None, 1.0, None
    )


def test_a_cross_validation_seed_is_not_declared_in_the_document() -> None:
    with pytest.raises(dx.ValidationError, match="seed"):
        decode_active_learning(
            a_document(
                cross_validation={
                    "held_out_unit": "item",
                    "splitter": "kfold",
                    "folds": 3,
                    "seed": 5,
                    "coverage_threshold": 0.0,
                    "unseen_participant": "plug_in",
                }
            )
        )


def test_a_run_seed_is_not_declared_in_the_document() -> None:
    with pytest.raises(dx.ValidationError, match="bootstrap_seed"):
        decode_active_learning(a_document(bootstrap_seed=17))


@pytest.mark.parametrize(
    ("changed", "reason"),
    [
        ({"pool": ""}, "active-learning document"),
        ({"fit": ""}, "active-learning document"),
        ({"initial": {"kind": "bead", "responses": ""}}, "initial-judgment document"),
        ({"initial": {"responses": "initial.jsonl"}}, "kind"),
        (
            {
                "initial": {
                    "kind": "bead",
                    "responses": "initial.jsonl",
                    "encoding": "encoding.json",
                }
            },
            "encoding",
        ),
        ({"collector": {"kind": "bead", "encoding": "encoding.json"}}, "encoding"),
        ({"encoding": ""}, "response encoding"),
        (
            {"initial": {**COLLECTED, "items_per_stratum": 0}},
            "items per stratum",
        ),
        ({"initial": {**COLLECTED, "stratify_by": ""}}, "stratification"),
        ({"initial": {**COLLECTED, "responses": "initial.jsonl"}}, "responses"),
        (
            {
                "predictors": {
                    "run": "runs/trained",
                    "inputs": {"words": "e"},
                    "tokenizer": {
                        "backend": "whitespace",
                        "language": "en",
                        "model_name": None,
                    },
                    "readouts": {},
                }
            },
            "at least one",
        ),
        (
            {
                "predictors": {
                    "run": "runs/trained",
                    "inputs": {"words": ""},
                    "tokenizer": {
                        "backend": "whitespace",
                        "language": "en",
                        "model_name": None,
                    },
                    "readouts": {"score": {"readout": "spans", "measure": "log-mass"}},
                }
            },
            "element",
        ),
        (
            {
                "predictors": {
                    "run": "runs/trained",
                    "inputs": {"words": "sentence"},
                    "tokenizer": {
                        "backend": "whitespace",
                        "language": "en",
                        "model_name": None,
                    },
                    "readouts": {"score": {"readout": "spans", "measure": "mass"}},
                }
            },
            "measure",
        ),
        ({"factors": {"structure": ""}}, "expression"),
        (
            {
                "acquisition": {
                    "kind": "rc2.1",
                    "allocate_by": "",
                    "constraints": [],
                    "share_tolerance": 0.1,
                }
            },
            "allocation expression",
        ),
        (
            {
                "acquisition": {
                    "kind": "rc2.1",
                    "allocate_by": "item['structure']",
                    "constraints": [],
                    "frame_tolerance": 0.1,
                }
            },
            "frame_tolerance",
        ),
        (
            {
                "acquisition": {
                    "kind": "rc2.1",
                    "batch_size": 4,
                    "allocate_by": "item['structure']",
                    "constraints": [
                        {
                            "constraint_type": "size",
                            "min_size": None,
                            "max_size": 3,
                            "exact_size": None,
                        }
                    ],
                    "share_tolerance": 0.1,
                }
            },
            "batch of 4",
        ),
        ({"collector": {"kind": "telepathy"}}, "telepathy"),
        (
            {
                "cross_validation": {
                    "held_out_unit": "",
                    "splitter": "kfold",
                    "folds": 3,
                    "coverage_threshold": 0.0,
                    "unseen_participant": None,
                }
            },
            "held-out unit",
        ),
        (
            {
                "cross_validation": {
                    "held_out_unit": "item",
                    "splitter": "kfold",
                    "folds": 1,
                    "coverage_threshold": 0.0,
                    "unseen_participant": None,
                }
            },
            "fold count",
        ),
        (
            {
                "cross_validation": {
                    "held_out_unit": "item",
                    "splitter": "kfold",
                    "folds": 3,
                    "coverage_threshold": 1.5,
                    "unseen_participant": None,
                }
            },
            "coverage threshold",
        ),
        (
            {
                "cross_validation": {
                    "held_out_unit": "item",
                    "splitter": "kfold",
                    "folds": 3,
                    "coverage_threshold": 0.0,
                    "unseen_participant": "guess",
                }
            },
            "unseen_participant",
        ),
    ],
)
def test_a_malformed_input_declaration_is_refused_at_the_boundary(
    changed: dict[str, JsonValue], reason: str
) -> None:
    with pytest.raises(dx.ValidationError, match=reason):
        decode_active_learning(a_document(**changed))


def test_asymptote_requires_a_multi_round_window_and_nonnegative_gain() -> None:
    declaration = decode_active_learning(
        a_document(stopping={"kind": "asymptote", "window": 3, "maximum_gain": 0.01})
    )

    assert declaration.settings().stopping == AsymptoteStopSettings(3, 0.01)

    with pytest.raises(dx.ValidationError, match="fewer than two"):
        decode_active_learning(
            a_document(
                stopping={"kind": "asymptote", "window": 1, "maximum_gain": 0.01}
            )
        )
