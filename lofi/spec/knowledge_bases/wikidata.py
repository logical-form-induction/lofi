"""Boundary declaration of one local Wikidata snapshot."""

from typing import Literal

import didactic.api as dx

from lofi._checks import ensure_name
from lofi.lf.models import WikidataSettings
from lofi.spec._interpolation import lower_package_resource

__all__ = ["KnowledgeBaseSpec", "WikidataKnowledgeBaseSpec"]


class KnowledgeBaseSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declare an external interpretation and its local representation."""


class WikidataKnowledgeBaseSpec(KnowledgeBaseSpec):
    """Declare a local Wikidata fact snapshot and vocabulary-id mapping."""

    kind: Literal["wikidata"] = "wikidata"
    snapshot: str
    mapping: str
    world: Literal["open", "closed"]
    unmapped: str

    @dx.validates("snapshot", "mapping")
    @classmethod
    def _path_is_named(cls, value: str) -> str:
        """Return a nonempty path or package-resource reference."""
        return ensure_name("a knowledge-base path", value)

    @dx.validates("unmapped")
    @classmethod
    def _unmapped_policy_is_named(cls, value: str) -> str:
        """Require an installed missing-constant policy name."""
        return ensure_name("an unmapped-constant policy", value)

    def settings(self) -> WikidataSettings:
        """Resolve package resources and discard the boundary model."""
        return WikidataSettings(
            lower_package_resource(self.snapshot),
            lower_package_resource(self.mapping),
            self.world,
            self.unmapped,
        )
