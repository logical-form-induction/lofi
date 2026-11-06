from __future__ import annotations

from typing import TYPE_CHECKING

from lofi.nn import _snapshots
from lofi.nn._snapshots import HubSnapshotTransport, artifact_fetcher
from lofi.spec.artifacts import ArtifactCacheSettings

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_a_snapshot_downloads_the_pinned_revision_into_the_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[tuple[str, str, Path]] = []

    def download(repository: str, *, revision: str, local_dir: Path) -> str:
        requests.append((repository, revision, local_dir))
        return str(local_dir)

    monkeypatch.setattr(_snapshots, "snapshot_download", download)

    HubSnapshotTransport().snapshot("an-org/a-model", "a" * 40, tmp_path)

    assert requests == [("an-org/a-model", "a" * 40, tmp_path)]
    assert repr(HubSnapshotTransport()) == "HubSnapshotTransport()"


def test_a_declared_cache_names_the_fetcher_root_and_mode(tmp_path: Path) -> None:
    fetcher = artifact_fetcher(ArtifactCacheSettings(tmp_path, offline=True))

    assert fetcher.root == tmp_path
    assert fetcher.offline
