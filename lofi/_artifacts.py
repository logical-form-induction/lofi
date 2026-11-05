"""Pinned pretrained artifacts and the one fetcher that materializes them.

A pretrained artifact is named either by a model-hub repository pinned to one
commit or by an absolute HTTP(S) URL pinned to the SHA-256 digest of its bytes.
Both variants are immutable identities, and one fetcher resolves either to a
verified local path under a cache root: a hub revision to a snapshot directory,
an archive URL to a digest-checked file. The fetcher reaches the network only
through the transports it is given, so the transports a run installs decide
where bytes come from.
"""

from __future__ import annotations

from hashlib import sha256
from http.client import HTTPConnection, HTTPSConnection
from shutil import rmtree
from typing import TYPE_CHECKING, Final, Protocol, cast, final, override
from urllib.parse import quote, urlsplit

from lofi._checks import ensure_commit, ensure_name
from lofi._frozen import FrozenValue

if TYPE_CHECKING:
    from pathlib import Path

    from lofi._types import JsonValue

__all__ = [
    "ArchiveTransport",
    "ArtifactFetcher",
    "HttpArchiveTransport",
    "PinnedArchive",
    "PinnedCheckpoint",
    "PretrainedArtifact",
    "SnapshotTransport",
]

_BLOCK: Final = 1024 * 1024
"""How many bytes one streamed read or digest update covers."""

_HEX: Final = frozenset("0123456789abcdef")


class ArchiveTransport(Protocol):
    """Writes the complete bytes at one URL to a caller-owned path."""

    def download(self, url: str, target: Path, /) -> None:
        """Write the artifact at ``url`` to ``target``."""
        ...


class SnapshotTransport(Protocol):
    """Writes one hub repository at one commit into a caller-owned directory."""

    def snapshot(self, repository: str, revision: str, target: Path, /) -> None:
        """Write every file of ``repository`` at ``revision`` under ``target``."""
        ...


class _Materializer(Protocol):
    """The two materializations a fetcher performs, one per artifact variant."""

    def archive(self, artifact: PinnedArchive, /) -> Path:
        """Return the verified local file of an archive artifact."""
        ...

    def snapshot(self, artifact: PinnedCheckpoint, /) -> Path:
        """Return the local snapshot directory of a hub artifact."""
        ...


@final
class PinnedCheckpoint(FrozenValue):
    """A released model repository pinned to one commit.

    Parameters
    ----------
    repository : str
        Hugging Face repository identifier.
    revision : str
        Full lowercase hexadecimal commit hash of the release.

    Raises
    ------
    TypeError
        If a field is not a string.
    ValueError
        If the repository is empty or the revision is not a full commit hash.
    """

    __slots__ = ("_repository", "_revision")

    _noun = "a pinned checkpoint"

    def __init__(self, repository: str, revision: str) -> None:
        object.__setattr__(
            self, "_repository", ensure_name("a checkpoint's repository", repository)
        )
        object.__setattr__(
            self, "_revision", ensure_commit("a checkpoint's revision", revision)
        )

    @property
    def repository(self) -> str:
        """Return the repository identifier."""
        return self._repository

    @property
    def revision(self) -> str:
        """Return the commit the repository is pinned to."""
        return self._revision

    @property
    def identity(self) -> str:
        """Return the repository and commit as one ``repository@commit`` string."""
        return f"{self._repository}@{self._revision}"

    def declaration(self) -> dict[str, JsonValue]:
        """Return the source declaration that names this artifact."""
        return {
            "kind": "hub",
            "repository": self._repository,
            "revision": self._revision,
        }

    def materialize(self, fetcher: _Materializer, /) -> Path:
        """Return the local snapshot directory the fetcher holds for this revision."""
        return fetcher.snapshot(self)

    def hub_checkpoint(self, consumer: str, /) -> PinnedCheckpoint:
        """Return this revision, which every hub-loading consumer reads.

        Parameters
        ----------
        consumer : str
            What loads the artifact, named in the refusal an archive gives.
        """
        del consumer

        return self

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another checkpoint names the same repository and commit."""
        if type(other) is not PinnedCheckpoint:
            return NotImplemented

        return (self._repository, self._revision) == (
            other.repository,
            other.revision,
        )

    @override
    def __hash__(self) -> int:
        """Return a hash of the repository and commit."""
        return hash((PinnedCheckpoint, self._repository, self._revision))

    @override
    def __repr__(self) -> str:
        """Return the repository and commit."""
        return (
            f"PinnedCheckpoint(repository={self._repository!r}, "
            f"revision={self._revision!r})"
        )


@final
class PinnedArchive(FrozenValue):
    """A released artifact fetched from one URL and pinned to its digest.

    Parameters
    ----------
    url : str
        The absolute HTTP or HTTPS URL the artifact is fetched from.
    sha256 : str
        The artifact's 64-digit lowercase hexadecimal SHA-256 digest.

    Raises
    ------
    TypeError
        If a field is not a string.
    ValueError
        If the URL is not absolute HTTP(S) or the digest is malformed.
    """

    __slots__ = ("_sha256", "_url")

    _noun = "a pinned archive"

    def __init__(self, url: str, sha256: str) -> None:
        source = ensure_name("an archive's URL", url)
        parsed = urlsplit(source)

        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            msg = f"an archive is fetched from an absolute HTTP(S) URL, not {url!r}"
            raise ValueError(msg)

        digest = ensure_name("an archive's SHA-256", sha256)

        if len(digest) != 64 or not frozenset(digest) <= _HEX:
            msg = (
                "an archive's SHA-256 is 64 lowercase hexadecimal digits, "
                f"not {sha256!r}"
            )
            raise ValueError(msg)

        object.__setattr__(self, "_url", source)
        object.__setattr__(self, "_sha256", digest)

    @property
    def url(self) -> str:
        """Return where the artifact is fetched from."""
        return self._url

    @property
    def sha256(self) -> str:
        """Return the digest the fetched bytes must have."""
        return self._sha256

    @property
    def identity(self) -> str:
        """Return the URL and digest as one ``url@sha256:digest`` string."""
        return f"{self._url}@sha256:{self._sha256}"

    def declaration(self) -> dict[str, JsonValue]:
        """Return the source declaration that names this artifact."""
        return {"kind": "url", "url": self._url, "sha256": self._sha256}

    def materialize(self, fetcher: _Materializer, /) -> Path:
        """Return the verified local file the fetcher holds for this archive."""
        return fetcher.archive(self)

    def hub_checkpoint(self, consumer: str, /) -> PinnedCheckpoint:
        """Refuse an archive where a consumer loads a hub repository.

        Parameters
        ----------
        consumer : str
            What loads the artifact, named in the refusal.

        Raises
        ------
        ValueError
            Always, since an archive is one file rather than a repository.
        """
        msg = (
            f"{consumer} loads a hub repository at one commit, and the archive "
            f"{self.identity!r} is not one"
        )
        raise ValueError(msg)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another archive names the same URL and digest."""
        if type(other) is not PinnedArchive:
            return NotImplemented

        return (self._url, self._sha256) == (other.url, other.sha256)

    @override
    def __hash__(self) -> int:
        """Return a hash of the URL and digest."""
        return hash((PinnedArchive, self._url, self._sha256))

    @override
    def __repr__(self) -> str:
        """Return the URL and digest."""
        return f"PinnedArchive(url={self._url!r}, sha256={self._sha256!r})"


type PretrainedArtifact = PinnedCheckpoint | PinnedArchive
"""One pinned pretrained artifact: a hub revision or a digest-pinned archive."""


def _digest(path: Path, /) -> str:
    """Return one file's SHA-256 digest without holding it in memory."""
    hashed = sha256()

    with path.open("rb") as stream:
        while block := stream.read(_BLOCK):
            hashed.update(block)

    return hashed.hexdigest()


class _HttpResponse(Protocol):
    """The part of an HTTP response the archive transport reads."""

    status: int

    def read(self, amount: int, /) -> bytes:
        """Return up to ``amount`` further bytes of the body."""
        ...

    def close(self) -> None:
        """Release the response."""
        ...


class HttpArchiveTransport:
    """Streams an archive through Python's HTTP client."""

    __slots__ = ()

    def download(self, url: str, target: Path, /) -> None:
        """Stream the complete body at ``url`` to ``target``.

        Raises
        ------
        ValueError
            If the URL is not HTTP(S) or names no host.
        OSError
            If the server answers with a status outside 2xx.
        """
        parsed = urlsplit(url)
        host = parsed.hostname

        if parsed.scheme not in {"http", "https"} or host is None:
            msg = f"an archive is downloaded over HTTP(S) from a host, not {url!r}"
            raise ValueError(msg)

        connection_type = (
            HTTPSConnection if parsed.scheme == "https" else HTTPConnection
        )
        connection = connection_type(host, parsed.port)
        path = parsed.path or "/"

        if parsed.query:
            path = f"{path}?{parsed.query}"

        try:
            connection.request("GET", path)
            response = cast("_HttpResponse", connection.getresponse())

            if not 200 <= response.status < 300:
                response.close()
                msg = f"the archive request for {url!r} returned HTTP {response.status}"
                raise OSError(msg)

            with target.open("wb") as stream:
                while block := response.read(_BLOCK):
                    stream.write(block)

            response.close()

        finally:
            connection.close()

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this transport."""
        return "HttpArchiveTransport()"


class ArtifactFetcher:
    """Resolves pinned artifacts to verified local paths under one cache root.

    An archive is cached at ``archives/<sha256>`` and is accepted only when its
    bytes have the declared digest, both when it is fetched and when a cached
    copy is reused. A hub revision is cached at
    ``snapshots/<quoted repository>/<commit>``; a commit is immutable, so a
    complete snapshot directory is reused as it stands. Each is written beside
    its target and renamed into place, so a failed fetch leaves no entry.

    Parameters
    ----------
    root : Path
        The absolute cache directory.
    archives : ArchiveTransport
        How an absent archive is downloaded.
    snapshots : SnapshotTransport
        How an absent hub revision is downloaded.
    offline : bool
        Whether an artifact absent from the cache is refused rather than fetched.

    Raises
    ------
    ValueError
        If the root is not absolute.
    """

    __slots__ = ("_archives", "_offline", "_root", "_snapshots")

    def __init__(
        self,
        root: Path,
        *,
        archives: ArchiveTransport,
        snapshots: SnapshotTransport,
        offline: bool,
    ) -> None:
        if not root.is_absolute():
            msg = f"an artifact cache root is an absolute directory, not {root}"
            raise ValueError(msg)

        self._root = root
        self._archives = archives
        self._snapshots = snapshots
        self._offline = offline

    @property
    def root(self) -> Path:
        """Return the cache directory."""
        return self._root

    @property
    def offline(self) -> bool:
        """Return whether an absent artifact is refused rather than fetched."""
        return self._offline

    def fetch(self, artifact: PretrainedArtifact, /) -> Path:
        """Return the verified local path of one pinned artifact.

        Raises
        ------
        FileNotFoundError
            If the artifact is absent from the cache in offline mode.
        ValueError
            If a cached or downloaded archive has another digest.
        """
        return artifact.materialize(self)

    def _absent(self, artifact: PretrainedArtifact, target: Path, /) -> None:
        """Refuse fetching an absent artifact in offline mode."""
        if self._offline:
            msg = f"artifact {artifact.identity!r} is absent from {target} offline"
            raise FileNotFoundError(msg)

    def archive(self, artifact: PinnedArchive, /) -> Path:
        """Return the digest-checked local file of an archive artifact."""
        target = self._root / "archives" / artifact.sha256

        if target.exists():
            found = _digest(target)

            if found != artifact.sha256:
                msg = (
                    f"cached archive {target} has SHA-256 {found}, not the "
                    f"declared {artifact.sha256}"
                )
                raise ValueError(msg)

            return target

        self._absent(artifact, target)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(f".{target.name}.partial")

        try:
            self._archives.download(artifact.url, partial)
            found = _digest(partial)

            if found != artifact.sha256:
                msg = (
                    f"the archive downloaded from {artifact.url!r} has SHA-256 "
                    f"{found}, not the declared {artifact.sha256}"
                )
                raise ValueError(msg)

            partial.replace(target)

        finally:
            partial.unlink(missing_ok=True)

        return target

    def snapshot(self, artifact: PinnedCheckpoint, /) -> Path:
        """Return the local snapshot directory of a hub artifact."""
        target = (
            self._root
            / "snapshots"
            / quote(artifact.repository, safe="")
            / artifact.revision
        )

        if target.is_dir():
            return target

        self._absent(artifact, target)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(f".{target.name}.partial")
        rmtree(partial, ignore_errors=True)
        partial.mkdir()

        try:
            self._snapshots.snapshot(artifact.repository, artifact.revision, partial)
            partial.replace(target)

        finally:
            rmtree(partial, ignore_errors=True)

        return target

    @override
    def __repr__(self) -> str:
        """Return the cache root and offline policy."""
        return f"ArtifactFetcher(root={self._root!r}, offline={self._offline!r})"
