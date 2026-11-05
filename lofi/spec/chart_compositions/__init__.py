"""Declarations of neural chart-cell composition families."""

from typing import Final, Literal, NamedTuple, Protocol, cast

import didactic.api as dx

from lofi.spec.span_functions import (
    DeclaredSpanFunction,
    SpanFunctionDeclaration,
    SpanFunctionSpec,
    default_span_function,
)

__all__ = [
    "ChartCompositionDeclaration",
    "ChartCompositionSettings",
    "ChartCompositionSpec",
    "DeclaredChartComposition",
    "GatedResidualCompositionSettings",
    "GatedResidualCompositionSpec",
    "MeanFieldCompositionSettings",
    "MeanFieldCompositionSpec",
    "default_chart_composition",
]


class MeanFieldCompositionSettings(NamedTuple):
    """Resolved functions of direct mean-field cell composition."""

    leaf: SpanFunctionDeclaration
    cell: SpanFunctionDeclaration


class GatedResidualCompositionSettings(NamedTuple):
    """Resolved functions of gated residual cell composition."""

    leaf: SpanFunctionDeclaration
    candidate: SpanFunctionDeclaration
    gate: SpanFunctionDeclaration


type ChartCompositionSettings = (
    MeanFieldCompositionSettings | GatedResidualCompositionSettings
)
"""Resolved settings of one chart-composition variant."""


class ChartCompositionDeclaration(NamedTuple):
    """One registered chart-composition tag and its immutable settings."""

    kind: str
    settings: ChartCompositionSettings


class DeclaredChartComposition(Protocol):
    """A chart-composition declaration that lowers at the parser boundary."""

    def declaration(self) -> ChartCompositionDeclaration:
        """Return the registered tag and immutable interior settings."""
        ...


class ChartCompositionSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declare how leaves enter a chart and child cells make a parent."""


_DEFAULT_LEAF: Final = dx.field(default_factory=default_span_function)
"""The shared leaf-function field of chart-composition declarations."""


class MeanFieldCompositionSpec(ChartCompositionSpec):
    """Compose each child pair directly before mixing over candidate splits."""

    kind: Literal["mean-field"] = "mean-field"
    leaf: SpanFunctionSpec = _DEFAULT_LEAF
    cell: SpanFunctionSpec = dx.field(default_factory=default_span_function)

    def declaration(self) -> ChartCompositionDeclaration:
        """Return the direct mean-field composition declaration."""
        leaf = cast("DeclaredSpanFunction", self.leaf)
        cell = cast("DeclaredSpanFunction", self.cell)

        return ChartCompositionDeclaration(
            self.kind,
            MeanFieldCompositionSettings(leaf.declaration(), cell.declaration()),
        )


class GatedResidualCompositionSpec(ChartCompositionSpec):
    """Gate a composed candidate against the mean of its two child cells."""

    kind: Literal["gated-residual"] = "gated-residual"
    leaf: SpanFunctionSpec = _DEFAULT_LEAF
    candidate: SpanFunctionSpec = dx.field(default_factory=default_span_function)
    gate: SpanFunctionSpec = dx.field(default_factory=default_span_function)

    def declaration(self) -> ChartCompositionDeclaration:
        """Return the gated residual composition declaration."""
        leaf = cast("DeclaredSpanFunction", self.leaf)
        candidate = cast("DeclaredSpanFunction", self.candidate)
        gate = cast("DeclaredSpanFunction", self.gate)

        return ChartCompositionDeclaration(
            self.kind,
            GatedResidualCompositionSettings(
                leaf.declaration(), candidate.declaration(), gate.declaration()
            ),
        )


def default_chart_composition() -> ChartCompositionSpec:
    """Return the direct mean-field composition used by a default chart."""
    return MeanFieldCompositionSpec()
