"""Tests for the boundary records of an active-learning run."""

import json
from typing import TYPE_CHECKING

import pytest
from tests.unit.spec.acquisition.test_active import a_document

from lofi.run._boundary.active import (
    ACTIVE_LEARNING_METRICS,
    ACTIVE_LEARNING_ROLE,
    ActiveLearningManifest,
    compose_active_learning,
    decode_active_learning_document,
    decode_active_learning_manifest,
    encode_active_learning_manifest,
)
from lofi.spec.acquisition import decode_active_learning

if TYPE_CHECKING:
    from pathlib import Path

    from lofi._types import JsonValue


def _manifest(**changed: JsonValue) -> dict[str, JsonValue]:
    document: dict[str, JsonValue] = {
        "document": json.loads(a_document()),
        "status": "awaiting-responses",
        "rounds": 1,
        "awaiting": 2,
        "stop_reason": None,
        "initial_judgments": 24,
        "initial_skipped": 0,
        "source_digests": {"items.json": "sha256:" + "0" * 64},
        "seed": 3,
        "seed_derivation": None,
        "package_versions": {"lofi": "0.1.0"},
        "git_revision": "0" * 40,
    }
    document.update(changed)
    return document


def test_a_manifest_round_trips_through_its_deterministic_encoding() -> None:
    manifest = decode_active_learning_manifest(json.dumps(_manifest()).encode())

    encoded = encode_active_learning_manifest(manifest)

    assert decode_active_learning_manifest(encoded) == manifest
    assert encoded == encode_active_learning_manifest(
        decode_active_learning_manifest(encoded)
    )
    assert manifest.kind == "active-learning"


@pytest.mark.parametrize(
    "changed",
    [
        {"status": "completed", "stop_reason": "max-rounds"},
        {"status": "awaiting-responses", "awaiting": None},
    ],
)
def test_exactly_an_awaiting_manifest_names_its_awaited_round(
    changed: dict[str, JsonValue],
) -> None:
    with pytest.raises(ValueError, match="names awaited round"):
        decode_active_learning_manifest(json.dumps(_manifest(**changed)).encode())


def test_a_completed_manifest_names_no_awaited_round() -> None:
    manifest = ActiveLearningManifest.model_validate_json(
        json.dumps(
            _manifest(status="completed", awaiting=None, stop_reason="pool-exhausted")
        )
    )

    assert manifest.awaiting is None
    assert manifest.stop_reason == "pool-exhausted"


def test_a_manifest_awaiting_round_zero_names_it() -> None:
    manifest = decode_active_learning_manifest(
        json.dumps(_manifest(rounds=0, awaiting=0)).encode()
    )

    assert manifest.awaiting == 0


@pytest.mark.parametrize(
    "changed",
    [
        {"status": "completed", "awaiting": None, "stop_reason": None},
        {"stop_reason": "agreement"},
        {"status": "completed", "awaiting": None, "stop_reason": "telepathy"},
    ],
)
def test_exactly_a_completed_manifest_records_why_it_stopped(
    changed: dict[str, JsonValue],
) -> None:
    with pytest.raises(ValueError, match="stop"):
        decode_active_learning_manifest(json.dumps(_manifest(**changed)).encode())


def test_a_resolved_document_and_a_document_file_validate_alike(
    tmp_path: Path,
) -> None:
    path = tmp_path / "active.json"
    path.write_bytes(a_document())

    composed = compose_active_learning(path)

    assert composed == decode_active_learning(a_document())
    assert decode_active_learning_document(json.loads(a_document())) == composed


def test_every_round_reports_its_metrics_under_one_role() -> None:
    assert ACTIVE_LEARNING_ROLE == "active-learning"
    assert ACTIVE_LEARNING_METRICS == (
        "agreement",
        "rank-correlation",
        "batch-variance",
        "stratum-share",
    )
