from __future__ import annotations

from typing import TYPE_CHECKING

import didactic.api as dx
import pytest

from lofi.run._boundary.external_probe import ExternalProbeSpec

if TYPE_CHECKING:
    from lofi._types import JsonValue


def _document(*, checkpoint_root: str, lengths: list[int]) -> dict[str, JsonValue]:
    length_values: list[JsonValue] = [*lengths]

    return {
        "model": {
            "model": "diora-test",
            "checkpoint": {
                "kind": "url",
                "url": "https://example.test/release.zip",
                "sha256": "0" * 64,
            },
            "sites": [],
        },
        "backend": {"kind": "released-mlp", "member": "model.pt"},
        "artifacts": {"root": checkpoint_root},
        "stimulus_lengths": length_values,
    }


def test_an_external_probe_document_requires_an_absolute_checkpoint_root() -> None:
    with pytest.raises(dx.ValidationError, match="relative"):
        ExternalProbeSpec.model_validate(
            _document(checkpoint_root="relative", lengths=[3])
        )


def test_an_external_probe_document_requires_a_stimulus() -> None:
    with pytest.raises(dx.ValidationError, match="at least one"):
        ExternalProbeSpec.model_validate(
            _document(checkpoint_root="/checkpoints", lengths=[])
        )


def test_an_external_probe_document_declares_its_backend_member() -> None:
    document = _document(checkpoint_root="/checkpoints", lengths=[3])
    settings = ExternalProbeSpec.model_validate(document).settings()

    assert repr(settings.factory) == "ReleasedDioraCheckpointFactory('model.pt')"

    with pytest.raises(dx.ValidationError, match="backend"):
        ExternalProbeSpec.model_validate(
            {key: value for key, value in document.items() if key != "backend"}
        )

    with pytest.raises(dx.ValidationError, match="member"):
        ExternalProbeSpec.model_validate(
            {**document, "backend": {"kind": "released-mlp", "member": ""}}
        )
