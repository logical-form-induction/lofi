"""Declared sources of pinned pretrained artifacts."""

from pathlib import Path
from typing import Literal, NamedTuple, Protocol, Self, cast

import didactic.api as dx

from lofi._artifacts import PinnedArchive, PinnedCheckpoint, PretrainedArtifact

__all__ = [
    "ArtifactCacheSettings",
    "ArtifactCacheSpec",
    "HubSourceSpec",
    "PretrainedSourceSpec",
    "UrlSourceSpec",
    "source_artifact",
    "source_spec",
]


class _SourceDeclaration(Protocol):
    """A declaration that names one pinned pretrained artifact."""

    def artifact(self) -> PretrainedArtifact:
        """Return the pinned artifact the declaration names."""
        ...


class ArtifactCacheSettings(NamedTuple):
    """Where fetched artifacts are cached and whether an absent one is fetched.

    Attributes
    ----------
    root : Path
        The absolute directory fetched artifacts are cached under.
    offline : bool
        Whether an artifact absent from the cache is refused rather than fetched.
    """

    root: Path
    offline: bool


class ArtifactCacheSpec(dx.Model, extra="forbid"):
    """Declare the cache every pinned artifact of one document is fetched into.

    Parameters
    ----------
    root : str
        The absolute directory fetched artifacts are cached under.
    offline : bool
        Whether an artifact absent from the cache is refused rather than fetched.
    """

    root: str
    offline: bool = False

    @dx.validates("root")
    @classmethod
    def _root_is_absolute(cls, value: str) -> str:
        """Refuse a cache root that depends on the working directory."""
        if not Path(value).is_absolute():
            msg = f"an artifact cache root is an absolute directory, not {value!r}"
            raise ValueError(msg)

        return value

    def settings(self) -> ArtifactCacheSettings:
        """Return the cache root and fetching mode."""
        return ArtifactCacheSettings(Path(self.root), self.offline)


class PretrainedSourceSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declares where one pinned pretrained artifact is fetched from."""


class HubSourceSpec(PretrainedSourceSpec):
    """Declare a model hub repository pinned to one commit.

    Parameters
    ----------
    repository : str
        The hub repository identifier.
    revision : str
        The full lowercase hexadecimal commit the repository is read at.
    """

    kind: Literal["hub"] = "hub"
    repository: str
    revision: str

    @dx.model_validator()
    def _pins(self) -> Self:
        """Refuse a repository or revision no artifact is named by."""
        self.artifact()

        return self

    def artifact(self) -> PretrainedArtifact:
        """Return the pinned hub revision."""
        return PinnedCheckpoint(self.repository, self.revision)


class UrlSourceSpec(PretrainedSourceSpec):
    """Declare an archive fetched from one URL and pinned to its digest.

    Parameters
    ----------
    url : str
        The absolute HTTP or HTTPS URL the archive is fetched from.
    sha256 : str
        The archive's lowercase hexadecimal SHA-256 digest.
    """

    kind: Literal["url"] = "url"
    url: str
    sha256: str

    @dx.model_validator()
    def _pins(self) -> Self:
        """Refuse a URL or digest no artifact is named by."""
        self.artifact()

        return self

    def artifact(self) -> PretrainedArtifact:
        """Return the digest-pinned archive."""
        return PinnedArchive(self.url, self.sha256)


def source_artifact(declaration: PretrainedSourceSpec, /) -> PretrainedArtifact:
    """Return the pinned artifact a validated source declaration names."""
    return cast("_SourceDeclaration", declaration).artifact()


def source_spec(artifact: PretrainedArtifact, /) -> PretrainedSourceSpec:
    """Return the source declaration that names one pinned artifact."""
    return PretrainedSourceSpec.model_validate(artifact.declaration())
