"""Wiring declared metric sinks and constraint policies."""

from typing import TYPE_CHECKING, cast

from lofi.data._selection_factory import build_selection_policy
from lofi.registry import plugin_for
from lofi.run._sinks import NullSink

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi.data import SelectionPolicy, Weighted
    from lofi.registry import ComponentPlugin
    from lofi.run._ports import MetricSink
    from lofi.run._sinks import SinkContext
    from lofi.spec import (
        ConfiguredSpec,
        ConstraintPolicySpec,
        SinkSpec,
    )

__all__ = ["resolve_constraint_policy", "resolve_metric_sinks"]


def resolve_constraint_policy(
    declaration: ConstraintPolicySpec, /
) -> SelectionPolicy[str, str | Weighted[str]]:
    """Return the runtime policy a validated declaration describes.

    Parameters
    ----------
    declaration : ConstraintPolicySpec
        Validated constraint-selection declaration.
    """
    return build_selection_policy(declaration.settings())


def resolve_metric_sinks(
    specs: Mapping[str, SinkSpec],
    context: SinkContext,
    /,
) -> tuple[MetricSink, ...]:
    """Build the declared metric sinks of one point, or the null sink.

    Each declaration is built through the plugin its kind names in the sink
    family, in role order.

    Parameters
    ----------
    specs : Mapping[str, SinkSpec]
        The validated sink declarations, by role.
    context : SinkContext
        The point the sinks receive metrics for.

    Returns
    -------
    tuple[MetricSink, ...]
        One sink per declared role, or a lone ``NullSink`` when none is declared,
        so a run that declares no sink loads and contacts no telemetry backend.

    Raises
    ------
    UnknownTagError
        If the sink family lists no plugin under a declaration's kind.
    MissingTelemetryBackendError
        If a declared sink's client package is not installed.
    """
    if not specs:
        return (NullSink(),)

    built: list[MetricSink] = []

    for role in sorted(specs):
        declaration = cast("ConfiguredSpec[object]", specs[role])
        plugin = cast(
            "ComponentPlugin[object, SinkContext, MetricSink]",
            plugin_for(declaration),
        )
        built.append(plugin.construct(declaration, context))

    return tuple(built)
