from hashlib import sha256
from pathlib import Path
from typing import Final

import didactic.api as dx
import pytest

from lofi._artifacts import PinnedArchive, PinnedCheckpoint
from lofi.spec.artifacts import (
    ArtifactCacheSettings,
    ArtifactCacheSpec,
    HubSourceSpec,
    PretrainedSourceSpec,
    UrlSourceSpec,
    source_artifact,
    source_spec,
)

DIGEST: Final = sha256(b"release").hexdigest()


@pytest.mark.parametrize(
    ("declared", "artifact"),
    [
        (
            {"kind": "hub", "repository": "an-org/a-model", "revision": "a" * 40},
            PinnedCheckpoint("an-org/a-model", "a" * 40),
        ),
        (
            {"kind": "url", "url": "https://example.test/a.zip", "sha256": DIGEST},
            PinnedArchive("https://example.test/a.zip", DIGEST),
        ),
    ],
)
def test_each_source_kind_names_its_pinned_artifact_and_round_trips(
    declared: dict[str, str], artifact: PinnedCheckpoint | PinnedArchive
) -> None:
    spec = PretrainedSourceSpec.model_validate(declared)

    assert source_artifact(spec) == artifact
    assert source_spec(artifact) == spec


def test_a_source_kind_is_one_of_the_registered_variants() -> None:
    assert isinstance(
        source_spec(PinnedArchive("https://a.test/x", DIGEST)), UrlSourceSpec
    )
    assert isinstance(
        source_spec(PinnedCheckpoint("an-org/a-model", "a" * 40)), HubSourceSpec
    )

    with pytest.raises(dx.ValidationError):
        PretrainedSourceSpec.model_validate({"kind": "ftp", "url": "x"})


@pytest.mark.parametrize(
    "declared",
    [
        {"kind": "hub", "repository": "an-org/a-model", "revision": "main"},
        {"kind": "url", "url": "https://example.test/a.zip", "sha256": "abc"},
        {"kind": "url", "url": "file:///a.zip", "sha256": DIGEST},
    ],
)
def test_a_source_that_pins_nothing_is_refused(declared: dict[str, str]) -> None:
    with pytest.raises(dx.ValidationError):
        PretrainedSourceSpec.model_validate(declared)


def test_a_cache_root_is_absolute_and_online_unless_declared_offline() -> None:
    declared = ArtifactCacheSpec(root=str(Path("/data/artifacts").absolute()))

    assert declared.settings() == ArtifactCacheSettings(
        Path("/data/artifacts").absolute(), offline=False
    )
    assert (
        ArtifactCacheSpec(root=str(Path("/data/artifacts").absolute()), offline=True)
        .settings()
        .offline
    )

    with pytest.raises(dx.ValidationError, match="absolute"):
        ArtifactCacheSpec(root="artifacts")


def test_a_hub_consumer_refuses_an_archive_by_name() -> None:
    archive = source_artifact(
        UrlSourceSpec(url="https://example.test/a.zip", sha256=DIGEST)
    )
    hub = PinnedCheckpoint("an-org/a-model", "a" * 40)

    assert hub.hub_checkpoint("an encoder") is hub

    with pytest.raises(ValueError, match="an encoder loads a hub repository"):
        archive.hub_checkpoint("an encoder")
