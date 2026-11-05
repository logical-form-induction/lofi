"""The installed collectors of an active-learning run.

Each plugin pairs a collector declaration with a builder that loads the
collector implementations, and with them the judgments capability, only when a
run builds its collector. Discovering the plugins therefore imports no
collector implementation and no quivers, and the builders name the built
collectors through the module that defines them.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Final, Protocol, cast

from lofi.registry import ComponentPlugin
from lofi.spec.acquisition import (
    BeadCollectorSpec,
    DrawnEffectsCollectorSpec,
    PredictorTruthCollectorSpec,
    RegisteredDesignCollectorSpec,
)

if TYPE_CHECKING:
    import lofi.run._active_collection
    from lofi.judgment import CollectorContext
    from lofi.registry import Builder
    from lofi.spec.acquisition import (
        BeadCollectorSettings,
        DrawnEffectsCollectorSettings,
        PredictorTruthCollectorSettings,
        RegisteredDesignCollectorSettings,
    )

__all__ = [
    "BEAD_COLLECTOR_PLUGIN",
    "DRAWN_EFFECTS_COLLECTOR_PLUGIN",
    "PREDICTOR_TRUTH_COLLECTOR_PLUGIN",
    "REGISTERED_DESIGN_COLLECTOR_PLUGIN",
]


class _Collection(Protocol):
    """The collector builders loaded when a run builds its collector."""

    bead_collector: Builder[
        BeadCollectorSettings,
        CollectorContext,
        lofi.run._active_collection.BeadCollector,
    ]
    predictor_truth_collector: Builder[
        PredictorTruthCollectorSettings,
        CollectorContext,
        lofi.run._active_collection.SimulatedCollector,
    ]
    drawn_effects_collector: Builder[
        DrawnEffectsCollectorSettings,
        CollectorContext,
        lofi.run._active_collection.SimulatedCollector,
    ]
    registered_design_collector: Builder[
        RegisteredDesignCollectorSettings,
        CollectorContext,
        lofi.run._active_collection.SimulatedCollector,
    ]


def _collection() -> _Collection:
    """Return the collector implementations at the build site that needs them."""
    return cast("_Collection", import_module("lofi.run._active_collection"))


def _bead(
    settings: BeadCollectorSettings, context: CollectorContext, /
) -> lofi.run._active_collection.BeadCollector:
    """Build the bead collector of one run."""
    return _collection().bead_collector(settings, context)


def _predictor_truth(
    settings: PredictorTruthCollectorSettings, context: CollectorContext, /
) -> lofi.run._active_collection.SimulatedCollector:
    """Build the collector whose true item effects are a predictor column."""
    return _collection().predictor_truth_collector(settings, context)


def _drawn_effects(
    settings: DrawnEffectsCollectorSettings, context: CollectorContext, /
) -> lofi.run._active_collection.SimulatedCollector:
    """Build the collector over a known-effects design with drawn item scores."""
    return _collection().drawn_effects_collector(settings, context)


def _registered_design(
    settings: RegisteredDesignCollectorSettings, context: CollectorContext, /
) -> lofi.run._active_collection.SimulatedCollector:
    """Build the collector answering from a registered rating design."""
    return _collection().registered_design_collector(settings, context)


BEAD_COLLECTOR_PLUGIN: Final = ComponentPlugin(spec=BeadCollectorSpec, build=_bead)
"""The installed collector that fields every batch through bead."""

PREDICTOR_TRUTH_COLLECTOR_PLUGIN: Final = ComponentPlugin(
    spec=PredictorTruthCollectorSpec, build=_predictor_truth
)
"""The installed collector whose true item effects are a predictor column."""

DRAWN_EFFECTS_COLLECTOR_PLUGIN: Final = ComponentPlugin(
    spec=DrawnEffectsCollectorSpec, build=_drawn_effects
)
"""The installed collector over a known-effects design with drawn item scores."""

REGISTERED_DESIGN_COLLECTOR_PLUGIN: Final = ComponentPlugin(
    spec=RegisteredDesignCollectorSpec, build=_registered_design
)
"""The installed collector answering from a registered rating design."""
