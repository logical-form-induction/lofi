"""Hub snapshots written through the Hugging Face Hub client.

The fetcher a declared artifact cache describes downloads archives over HTTP(S)
and hub revisions through this transport, so every pretrained download a run
or a document declares goes through one fetcher.
"""

from typing import TYPE_CHECKING, override

from huggingface_hub import snapshot_download

from lofi._artifacts import ArtifactFetcher, HttpArchiveTransport

if TYPE_CHECKING:
    from pathlib import Path

    from lofi.spec.artifacts import ArtifactCacheSettings

__all__ = ["HubSnapshotTransport", "artifact_fetcher"]


class HubSnapshotTransport:
    """Writes one hub repository at one commit into a directory."""

    __slots__ = ()

    def snapshot(self, repository: str, revision: str, target: Path, /) -> None:
        """Write every file of ``repository`` at ``revision`` under ``target``."""
        snapshot_download(repository, revision=revision, local_dir=target)

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this transport."""
        return "HubSnapshotTransport()"


def artifact_fetcher(cache: ArtifactCacheSettings, /) -> ArtifactFetcher:
    """Return the fetcher of one declared artifact cache.

    Parameters
    ----------
    cache : ArtifactCacheSettings
        The cache root and whether an absent artifact is fetched.

    Returns
    -------
    ArtifactFetcher
        A fetcher downloading archives over HTTP(S) and hub revisions through
        the Hub client.
    """
    return ArtifactFetcher(
        cache.root,
        archives=HttpArchiveTransport(),
        snapshots=HubSnapshotTransport(),
        offline=cache.offline,
    )
