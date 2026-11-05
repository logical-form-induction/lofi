"""Declared held-out label alignment and its uncertainty summaries."""

from typing import Literal, NamedTuple, Protocol, Self

import didactic.api as dx

from lofi._checks import ensure_name
from lofi.spec.components.run import BootstrapIntervalSpec

__all__ = [
    "AlignmentDeclaration",
    "AlignmentKind",
    "AlignmentSettings",
    "AlignmentSpec",
    "AlignmentTieBreak",
    "IdentityAlignmentSpec",
    "ManyToOneAlignmentSpec",
    "OneToOneAlignmentSpec",
]

type AlignmentKind = Literal["identity", "many-to-one", "one-to-one"]
"""A declared relationship between induced and gold labels."""

type AlignmentTieBreak = Literal["least-gold-label", "largest-gold-marginal"]
"""A total order for equally scoring label assignments."""


class AlignmentSettings(NamedTuple):
    """The validated alignment settings consumed below the document boundary."""

    kind: AlignmentKind
    tie_break: AlignmentTieBreak | None
    fit_on: str
    score_on: str
    floor_replicates: int
    floor_level: float
    stability_replicates: int
    stability_level: float


class AlignmentDeclaration(Protocol):
    """A validated declaration that resolves an alignment policy."""

    def settings(self) -> AlignmentSettings:
        """Return the immutable interior alignment settings."""
        ...


class AlignmentSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declare a label correspondence fitted on development and frozen for test."""

    fit_on: str
    score_on: str
    random_floor: BootstrapIntervalSpec
    stability: BootstrapIntervalSpec

    @dx.model_validator()
    def _partitions_are_disjoint(self) -> Self:
        """Return two named partitions with no fitting-to-scoring leakage."""
        ensure_name("an alignment fitting partition", self.fit_on)
        ensure_name("an alignment scoring partition", self.score_on)
        if self.fit_on == self.score_on:
            msg = "alignment fitting and scoring partitions are disjoint"
            raise ValueError(msg)
        return self

    def _settings(
        self,
        kind: AlignmentKind,
        tie_break: AlignmentTieBreak | None,
    ) -> AlignmentSettings:
        """Return the common settings with an explicit policy identity."""
        return AlignmentSettings(
            kind=kind,
            tie_break=tie_break,
            fit_on=self.fit_on,
            score_on=self.score_on,
            floor_replicates=self.random_floor.replicates,
            floor_level=self.random_floor.level,
            stability_replicates=self.stability.replicates,
            stability_level=self.stability.level,
        )


class IdentityAlignmentSpec(AlignmentSpec):
    """Declare literal label identity, with no tie to settle."""

    kind: Literal["identity"] = "identity"

    def settings(self) -> AlignmentSettings:
        """Return the explicit identity settings."""
        return self._settings(self.kind, None)


class ManyToOneAlignmentSpec(AlignmentSpec):
    """Declare an independent maximum-count gold image for every induced label."""

    kind: Literal["many-to-one"] = "many-to-one"
    tie_break: AlignmentTieBreak

    def settings(self) -> AlignmentSettings:
        """Return the many-to-one settings."""
        return self._settings(self.kind, self.tie_break)


class OneToOneAlignmentSpec(AlignmentSpec):
    """Declare a maximum-count injective assignment between label inventories."""

    kind: Literal["one-to-one"] = "one-to-one"
    tie_break: AlignmentTieBreak

    def settings(self) -> AlignmentSettings:
        """Return the one-to-one settings."""
        return self._settings(self.kind, self.tie_break)
