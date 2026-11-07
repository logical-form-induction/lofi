"""Declarations of the weights that mix a span's candidate splits."""

from typing import Literal, NamedTuple, Protocol, cast

import didactic.api as dx

from lofi.spec.span_functions import (
    DeclaredSpanFunction,
    SpanFunctionDeclaration,
    SpanFunctionSpec,
    default_span_function,
)

__all__ = [
    "AttendSplitWeigherSettings",
    "AttendSplitWeigherSpec",
    "DeclaredSplitWeigher",
    "MarginalSplitWeigherSettings",
    "MarginalSplitWeigherSpec",
    "SplitWeigherDeclaration",
    "SplitWeigherSettings",
    "SplitWeigherSpec",
]


class MarginalSplitWeigherSettings(NamedTuple):
    """Resolved settings of exact conditional split posterior weighting."""


class AttendSplitWeigherSettings(NamedTuple):
    """Resolved settings of learned child pair attention weighting."""

    score: SpanFunctionDeclaration


type SplitWeigherSettings = MarginalSplitWeigherSettings | AttendSplitWeigherSettings
"""Resolved settings of one split-weigher variant."""


class SplitWeigherDeclaration(NamedTuple):
    """One resolved split-weigher tag and its immutable settings."""

    kind: str
    settings: SplitWeigherSettings


class DeclaredSplitWeigher(Protocol):
    """A split-weigher variant that lowers at the parser boundary."""

    def declaration(self) -> SplitWeigherDeclaration:
        """Return the registered tag and immutable interior settings."""
        ...


class SplitWeigherSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declare how a neural chart mixes the candidate splits of a span.

    Each variant states ``reads_children``: whether its weights are scored from
    the child vectors, which a span-input controller requires.
    """

    reads_children = False


class MarginalSplitWeigherSpec(SplitWeigherSpec):
    """Mix splits by their exact conditional posterior under the chart."""

    kind: Literal["marginal"] = "marginal"

    def declaration(self) -> SplitWeigherDeclaration:
        """Return the marginal plugin declaration."""
        return SplitWeigherDeclaration(self.kind, MarginalSplitWeigherSettings())


class AttendSplitWeigherSpec(SplitWeigherSpec):
    """Mix splits by the learned child pair scores alone."""

    kind: Literal["attend"] = "attend"
    reads_children = True
    score: SpanFunctionSpec = dx.field(default_factory=default_span_function)

    def declaration(self) -> SplitWeigherDeclaration:
        """Return the attention plugin declaration."""
        score = cast("DeclaredSpanFunction", self.score)

        return SplitWeigherDeclaration(
            self.kind, AttendSplitWeigherSettings(score.declaration())
        )


_DEFAULT_SPLIT_WEIGHER = dx.field(default_factory=MarginalSplitWeigherSpec)
"""The shared default field of neural chart declarations."""
