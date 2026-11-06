"""The declared boundary of a released external-model probe."""

from typing import TYPE_CHECKING, Literal, Protocol, Self, cast

import didactic.api as dx

from lofi import probe
from lofi._checks import ensure_name, ensure_positive
from lofi._frozen import FrozenValue
from lofi.run._diora import ReleasedDioraCheckpointFactory
from lofi.spec.artifacts import ArtifactCacheSpec
from lofi.spec.components.run import ProbeSpec

if TYPE_CHECKING:
    from lofi.spec.artifacts import ArtifactCacheSettings

__all__ = [
    "DioraBackendDecl",
    "ExternalProbeSettings",
    "ExternalProbeSpec",
    "ReleasedMlpBackendDecl",
]


class _BackendDeclaration(Protocol):
    """A declaration that builds the DIORA backend of a fetched artifact."""

    def factory(self) -> probe.DioraCheckpointFactory:
        """Return the factory that builds the backend from the fetched path."""
        ...


class DioraBackendDecl(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declares which DIORA architecture a released artifact stores."""


class ReleasedMlpBackendDecl(DioraBackendDecl):
    """Declare the MLP DIORA of the released implementation.

    Parameters
    ----------
    member : str
        The relative path of the stored state file inside the fetched artifact:
        a member of a zip archive, or a file of a snapshot directory.
    """

    kind: Literal["released-mlp"] = "released-mlp"
    member: str

    @dx.validates("member")
    @classmethod
    def _member_is_named(cls, value: str) -> str:
        """Refuse an empty member path."""
        return ensure_name("a released DIORA state member", value)

    def factory(self) -> probe.DioraCheckpointFactory:
        """Return the factory reading the declared state member."""
        return ReleasedDioraCheckpointFactory(self.member)


class ExternalProbeSettings(FrozenValue):
    """Lowered settings for one released-model probe."""

    __slots__ = ("_cache", "_factory", "_lengths", "_model")

    _noun = "external probe settings"

    def __init__(
        self,
        model: probe.ExternalSiteMap,
        factory: probe.DioraCheckpointFactory,
        cache: ArtifactCacheSettings,
        lengths: tuple[int, ...],
    ) -> None:
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_factory", factory)
        object.__setattr__(self, "_cache", cache)
        object.__setattr__(
            self,
            "_lengths",
            tuple(
                ensure_positive("an external stimulus length", length)
                for length in lengths
            ),
        )

        if not self._lengths:
            msg = "an external probe declares at least one stimulus length"
            raise ValueError(msg)

    @property
    def model(self) -> probe.ExternalSiteMap:
        """Return the lowered external-model declaration."""
        return self._model

    @property
    def factory(self) -> probe.DioraCheckpointFactory:
        """Return the declared backend built from the fetched artifact."""
        return self._factory

    @property
    def cache(self) -> ArtifactCacheSettings:
        """Return the cache fetched pinned artifacts are kept in."""
        return self._cache

    @property
    def lengths(self) -> tuple[int, ...]:
        """Return the lengths of the declared synthetic stimuli."""
        return self._lengths


class ExternalProbeSpec(ProbeSpec):
    """Declare split posterior diagnostics over a released DIORA checkpoint.

    The stimulus lengths describe deterministic all-zero input vectors. Their
    feature width is derived from the released model rather than repeated in the
    document. These synthetic stimuli exercise the released chart and its sites;
    empirical corpus inputs belong to the experiment that consumes the probe rows.

    Parameters
    ----------
    model : ExternalModelSpec
        The released model, its pinned checkpoint source and its sites.
    backend : DioraBackendDecl
        The architecture the released artifact stores.
    artifacts : ArtifactCacheSpec
        The cache the released checkpoint is fetched into.
    stimulus_lengths : tuple[int, ...]
        The sentence lengths of the stimuli, one pass each.
    """

    kind: Literal["external"] = "external"
    model: probe.ExternalModelSpec
    backend: DioraBackendDecl
    artifacts: ArtifactCacheSpec
    stimulus_lengths: tuple[int, ...]

    @dx.model_validator()
    def _lowers(self) -> Self:
        """Refuse a declaration that cannot lower to runtime settings."""
        self.settings()

        return self

    def settings(self) -> ExternalProbeSettings:
        """Return the lowered model, backend, checkpoint root and stimuli."""
        return ExternalProbeSettings(
            probe.ExternalSiteMapIso().forward(self.model),
            cast("_BackendDeclaration", self.backend).factory(),
            self.artifacts.settings(),
            self.stimulus_lengths,
        )
