"""Declarations for systems fitted as external baselines."""

from typing import TYPE_CHECKING

import didactic.api as dx

from lofi.spec.components.base import ComponentSpec

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi._types import JsonValue

__all__ = ["BaselineSystemSpec"]


class BaselineSystemSpec(
    ComponentSpec, dx.TaggedUnion, discriminator="kind", extra="forbid"
):
    """Declare one independently fitted comparison system."""

    @classmethod
    def family(cls) -> str:
        """Return the family whose entry point group lists the plugins."""
        return "baseline_system"

    def metrics(self) -> tuple[str, ...]:
        """Return the metrics the fitted system reports, in row order."""
        return ()

    def conventions(self) -> Mapping[str, JsonValue]:
        """Return the scoring conventions the fitted system declares."""
        return {}
