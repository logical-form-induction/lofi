from __future__ import annotations

from hashlib import sha256
from typing import TYPE_CHECKING, Final

import pytest

from lofi._artifacts import (
    ArtifactFetcher,
    HttpArchiveTransport,
    PinnedArchive,
    PinnedCheckpoint,
)

if TYPE_CHECKING:
    from pathlib import Path

PAYLOAD: Final = b"released artifact"
DIGEST: Final = sha256(PAYLOAD).hexdigest()
URL: Final = "https://example.test/release.zip"
HUB: Final = PinnedCheckpoint("an-org/a-model", "a" * 40)
ARCHIVE: Final = PinnedArchive(URL, DIGEST)


class _Archives:
    def __init__(self, payload: bytes = PAYLOAD) -> None:
        self.payload = payload
        self.requests: list[str] = []

    def download(self, url: str, target: Path, /) -> None:
        self.requests.append(url)
        target.write_bytes(self.payload)


class _Snapshots:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.requests: list[tuple[str, str]] = []

    def snapshot(self, repository: str, revision: str, target: Path, /) -> None:
        self.requests.append((repository, revision))
        (target / "config.json").write_text("{}", encoding="utf-8")

        if self.fail:
            msg = "the hub refused the request"
            raise OSError(msg)


def a_fetcher(
    root: Path,
    *,
    archives: _Archives | None = None,
    snapshots: _Snapshots | None = None,
    offline: bool = False,
) -> ArtifactFetcher:
    return ArtifactFetcher(
        root,
        archives=_Archives() if archives is None else archives,
        snapshots=_Snapshots() if snapshots is None else snapshots,
        offline=offline,
    )


def test_a_pinned_checkpoint_records_its_repository_and_commit() -> None:
    assert HUB.repository == "an-org/a-model"
    assert HUB.revision == "a" * 40
    assert HUB.identity == f"an-org/a-model@{'a' * 40}"
    assert PinnedCheckpoint(HUB.repository, HUB.revision) == HUB
    assert hash(HUB) == hash(PinnedCheckpoint(HUB.repository, HUB.revision))
    assert PinnedCheckpoint(HUB.repository, "b" * 40) != HUB
    assert HUB.declaration() == {
        "kind": "hub",
        "repository": "an-org/a-model",
        "revision": "a" * 40,
    }
    assert repr(HUB) == (
        f"PinnedCheckpoint(repository='an-org/a-model', revision='{'a' * 40}')"
    )


@pytest.mark.parametrize(
    "revision", ["main", "v1.0", "0123456789ABCDEF0123456789ABCDEF01234567", "0" * 39]
)
def test_a_checkpoint_pinned_to_anything_but_a_commit_is_refused(
    revision: str,
) -> None:
    with pytest.raises(ValueError, match="commit"):
        PinnedCheckpoint("an-org/a-model", revision)


def test_a_pinned_archive_records_its_url_and_digest() -> None:
    assert ARCHIVE.url == URL
    assert ARCHIVE.sha256 == DIGEST
    assert ARCHIVE.identity == f"{URL}@sha256:{DIGEST}"
    assert PinnedArchive(URL, DIGEST) == ARCHIVE
    assert hash(ARCHIVE) == hash(PinnedArchive(URL, DIGEST))
    assert PinnedArchive("https://example.test/other.zip", DIGEST) != ARCHIVE
    assert ARCHIVE.declaration() == {"kind": "url", "url": URL, "sha256": DIGEST}
    assert ARCHIVE != HUB


@pytest.mark.parametrize(
    ("url", "digest", "match"),
    [
        ("ftp://example.test/release.zip", DIGEST, "HTTP"),
        ("release.zip", DIGEST, "HTTP"),
        (URL, "abcd", "SHA-256"),
        (URL, DIGEST.upper(), "SHA-256"),
    ],
)
def test_an_archive_needs_an_absolute_http_url_and_a_sha256(
    url: str, digest: str, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        PinnedArchive(url, digest)


def test_an_archive_is_fetched_once_and_then_read_offline(tmp_path: Path) -> None:
    archives = _Archives()

    first = a_fetcher(tmp_path, archives=archives).fetch(ARCHIVE)
    second = a_fetcher(tmp_path, archives=archives, offline=True).fetch(ARCHIVE)

    assert first == second == tmp_path / "archives" / DIGEST
    assert first.read_bytes() == PAYLOAD
    assert archives.requests == [URL]


def test_an_archive_with_another_digest_is_refused_and_not_cached(
    tmp_path: Path,
) -> None:
    fetcher = a_fetcher(tmp_path, archives=_Archives(b"not the release"))

    with pytest.raises(ValueError, match="SHA-256"):
        fetcher.fetch(ARCHIVE)

    assert not (tmp_path / "archives" / DIGEST).exists()
    assert list((tmp_path / "archives").iterdir()) == []


def test_a_cached_archive_with_another_digest_is_refused(tmp_path: Path) -> None:
    cached = tmp_path / "archives" / DIGEST
    cached.parent.mkdir()
    cached.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="cached archive"):
        a_fetcher(tmp_path).fetch(ARCHIVE)


def test_a_hub_revision_is_snapshotted_once_and_then_read_offline(
    tmp_path: Path,
) -> None:
    snapshots = _Snapshots()

    first = a_fetcher(tmp_path, snapshots=snapshots).fetch(HUB)
    second = a_fetcher(tmp_path, snapshots=snapshots, offline=True).fetch(HUB)

    assert first == second == tmp_path / "snapshots" / "an-org%2Fa-model" / ("a" * 40)
    assert (first / "config.json").is_file()
    assert snapshots.requests == [("an-org/a-model", "a" * 40)]


def test_a_failed_snapshot_leaves_no_cache_entry(tmp_path: Path) -> None:
    with pytest.raises(OSError, match="refused"):
        a_fetcher(tmp_path, snapshots=_Snapshots(fail=True)).fetch(HUB)

    assert list((tmp_path / "snapshots" / "an-org%2Fa-model").iterdir()) == []


@pytest.mark.parametrize("artifact", [HUB, ARCHIVE])
def test_offline_fetching_refuses_an_absent_artifact(
    tmp_path: Path, artifact: PinnedCheckpoint | PinnedArchive
) -> None:
    with pytest.raises(FileNotFoundError, match="offline"):
        a_fetcher(tmp_path, offline=True).fetch(artifact)


def test_the_cache_root_is_absolute(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute"):
        a_fetcher(tmp_path.relative_to(tmp_path.anchor))

    fetcher = a_fetcher(tmp_path, offline=True)

    assert fetcher.root == tmp_path
    assert fetcher.offline


def test_the_http_transport_refuses_a_url_it_cannot_reach(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="HTTP"):
        HttpArchiveTransport().download("ftp://example.test/a", tmp_path / "a")

    assert repr(HttpArchiveTransport()) == "HttpArchiveTransport()"
