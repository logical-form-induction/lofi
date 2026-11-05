"""Registered neural chart-cell composition families."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, final, override, runtime_checkable

import torch
from torch import nn

from lofi._frozen import FrozenValue
from lofi.nn.parsers.functions import SpanFunctionModule, build_span_function
from lofi.registry import Registry, RegistryError
from lofi.spec.chart_compositions import (
    ChartCompositionDeclaration,
    GatedResidualCompositionSettings,
    MeanFieldCompositionSettings,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

    from lofi.judge import ActionSet, RuleSet

__all__ = [
    "GATED_RESIDUAL_PLUGIN",
    "MEAN_FIELD_PLUGIN",
    "ChartComposition",
    "ChartCompositionModule",
    "ChartCompositionPlugin",
    "GatedResidualComposition",
    "MeanFieldComposition",
    "action_arities",
    "build_chart_composition",
]


@runtime_checkable
class ChartComposition(Protocol):
    """Project leaves and compose paired chart cells at one declared width."""

    @property
    def input_width(self) -> int:
        """Return the width of each pooled leaf vector."""
        ...

    @property
    def output_width(self) -> int:
        """Return the width of every chart cell."""
        ...

    def leaf(self, values: Tensor, /) -> Tensor:
        """Return the initial chart cells projected from pooled leaves."""
        ...

    def compose(self, left: Tensor, right: Tensor, /) -> Tensor:
        """Return one parent cell per pair of child cells."""
        ...


class ChartCompositionModule(nn.Module):
    """Hold the declared leaf projection shared by chart-composition variants."""

    def __init__(self, leaf: SpanFunctionModule, /, *, width: int) -> None:
        super().__init__()

        if len(leaf.input_widths) != 1:
            msg = "a chart leaf function reads exactly one pooled vector"
            raise ValueError(msg)

        if leaf.output_width != width:
            msg = (
                f"a chart leaf function returns width {leaf.output_width}, "
                f"not the chart width {width}"
            )
            raise ValueError(msg)

        self._leaf = leaf
        self._width = width

    @property
    def input_width(self) -> int:
        """Return the width of each pooled leaf vector."""
        return self._leaf.input_widths[0]

    @property
    def output_width(self) -> int:
        """Return the width of every chart cell."""
        return self._width

    def leaf(self, values: Tensor, /) -> Tensor:
        """Return the initial chart cells projected from pooled leaves."""
        return self._leaf(values)

    def compose(self, left: Tensor, right: Tensor, /) -> Tensor:
        """Return one parent cell per pair through this module's forward pass."""
        return self(left, right)


@final
class MeanFieldComposition(ChartCompositionModule):
    """Compose each child pair directly before the chart mixes its splits."""

    def __init__(
        self, leaf: SpanFunctionModule, cell: SpanFunctionModule, /, *, width: int
    ) -> None:
        super().__init__(leaf, width=width)
        _require_cell_function("mean-field cell", cell, width)
        self._cell = cell

    @override
    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        """Return the declared direct composition of each child pair."""
        return self._cell(left, right)


@final
class GatedResidualComposition(ChartCompositionModule):
    """Mix a candidate parent with the arithmetic mean of its children."""

    def __init__(
        self,
        leaf: SpanFunctionModule,
        candidate: SpanFunctionModule,
        gate: SpanFunctionModule,
        /,
        *,
        width: int,
    ) -> None:
        super().__init__(leaf, width=width)
        _require_cell_function("gated residual candidate", candidate, width)
        _require_cell_function("gated residual gate", gate, width)
        self._candidate = candidate
        self._gate = gate

    @override
    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        """Return the learned mixture of a candidate and the child residual."""
        candidate = self._candidate(left, right)
        gate = torch.sigmoid(self._gate(left, right))
        residual = (left + right) * 0.5

        return gate * candidate + (1.0 - gate) * residual


def _require_cell_function(
    slot: str, function: SpanFunctionModule, width: int, /
) -> None:
    """Refuse a cell function that does not map two cells back to one cell."""
    if function.input_widths != (width, width):
        msg = (
            f"the {slot} function reads widths {function.input_widths}, "
            f"not {(width, width)}"
        )
        raise ValueError(msg)

    if function.output_width != width:
        msg = f"the {slot} function returns width {function.output_width}, not {width}"
        raise ValueError(msg)


class ChartCompositionPlugin[S](FrozenValue):
    """Build one chart composition from its resolved settings."""

    __slots__ = ("_build", "_settings")
    _noun = "chart-composition plugin"

    def __init__(
        self,
        *,
        settings: type[S],
        build: Callable[[S, int, int], ChartCompositionModule],
    ) -> None:
        object.__setattr__(self, "_settings", settings)
        object.__setattr__(self, "_build", build)

    def construct(
        self, settings: object, input_width: int, output_width: int, /
    ) -> ChartCompositionModule:
        """Build a composition after checking the registered settings type."""
        if not isinstance(settings, self._settings):
            msg = (
                f"a chart-composition plugin reads {self._settings.__name__}, "
                f"not {type(settings).__name__}"
            )
            raise TypeError(msg)

        return self._build(settings, input_width, output_width)


def build_chart_composition(
    declaration: ChartCompositionDeclaration,
    input_width: int,
    output_width: int,
    /,
) -> ChartCompositionModule:
    """Build one chart composition through its installed plugin."""
    member = Registry[object]("chart_composition").get(declaration.kind)

    if not isinstance(member, ChartCompositionPlugin):
        msg = (
            f"the chart_composition entry {declaration.kind!r} names {member!r}, "
            "not a chart-composition plugin"
        )
        raise RegistryError(msg, family="chart_composition")

    return member.construct(declaration.settings, input_width, output_width)


def action_arities[T](actions: ActionSet, rules: RuleSet[T], /) -> tuple[int, ...]:
    """Return the distinct rule arities named by composing actions."""
    by_name = {rule.name: rule.arity for rule in rules}
    found: set[int] = set()

    for action in actions.composing:
        rule = action.rule

        if rule is None:
            msg = f"composing action {action.name!r} names no rule"
            raise ValueError(msg)

        arity = by_name.get(rule)

        if arity is None:
            msg = f"composing action {action.name!r} names unknown rule {rule!r}"
            raise ValueError(msg)

        found.add(arity)

    return tuple(sorted(found))


def _mean_field(
    settings: MeanFieldCompositionSettings, input_width: int, output_width: int, /
) -> MeanFieldComposition:
    """Build direct mean-field cell composition."""
    return MeanFieldComposition(
        build_span_function(settings.leaf, (input_width,), output_width),
        build_span_function(settings.cell, (output_width, output_width), output_width),
        width=output_width,
    )


def _gated_residual(
    settings: GatedResidualCompositionSettings,
    input_width: int,
    output_width: int,
    /,
) -> GatedResidualComposition:
    """Build gated residual cell composition."""
    inputs = (output_width, output_width)

    return GatedResidualComposition(
        build_span_function(settings.leaf, (input_width,), output_width),
        build_span_function(settings.candidate, inputs, output_width),
        build_span_function(settings.gate, inputs, output_width),
        width=output_width,
    )


MEAN_FIELD_PLUGIN = ChartCompositionPlugin(
    settings=MeanFieldCompositionSettings, build=_mean_field
)
GATED_RESIDUAL_PLUGIN = ChartCompositionPlugin(
    settings=GatedResidualCompositionSettings, build=_gated_residual
)
