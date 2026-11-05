"""Boundary declaration of a knowledge-base-backed structure source."""

from typing import Literal, Protocol, cast

import didactic.api as dx

from lofi.lf.models import GivenSettings
from lofi.spec.knowledge_bases import KnowledgeBaseSpec
from lofi.spec.structures.sampled import StructureSourceSpec

__all__ = ["GivenStructureSourceSpec"]


class _KnowledgeBaseDeclaration(Protocol):
    """Resolve any registered knowledge-base declaration."""

    @property
    def kind(self) -> str: ...

    def settings(self) -> object: ...


class GivenStructureSourceSpec(StructureSourceSpec):
    """Declare one read-only intended interpretation from a knowledge base."""

    kind: Literal["given"] = "given"
    knowledge_base: KnowledgeBaseSpec
    full_function_space_cap: int
    henkin_pool: int

    @dx.model_validator()
    def _frame_limits_are_positive(self) -> GivenStructureSourceSpec:
        """Validate frame limits once at the structure-source boundary."""
        if self.full_function_space_cap < 1 or self.henkin_pool < 1:
            msg = "function-space cap and Henkin pool are positive"
            raise ValueError(msg)

        return self

    def settings(self) -> GivenSettings:
        """Return immutable interior settings for the given source."""
        knowledge_base = cast("_KnowledgeBaseDeclaration", self.knowledge_base)
        return GivenSettings(
            knowledge_base.kind,
            knowledge_base.settings(),
            full_function_space_cap=self.full_function_space_cap,
            henkin_pool=self.henkin_pool,
        )
