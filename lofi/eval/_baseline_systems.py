"""The installed baseline systems whose implementations read a corpus codec.

Each plugin pairs a baseline declaration with a builder that loads the system
implementation, and with it the corpus codecs, only when a run builds the
baseline. Discovering the plugins therefore imports no baseline implementation,
and the builders name the built operations through the module that defines them.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Final, Protocol, cast

from lofi.registry import ComponentPlugin
from lofi.spec.baselines.bisk_hockenmaier import BiskHockenmaierBaselineSpec

if TYPE_CHECKING:
    import lofi.eval._bisk_hockenmaier_baseline
    from lofi.eval._ports import BaselineBuildContext
    from lofi.spec.baselines.bisk_hockenmaier import BiskHockenmaierBaselineSettings

__all__ = ["BISK_HOCKENMAIER_PLUGIN"]


class _BiskHockenmaier(Protocol):
    """The baseline implementation loaded when a run builds the baseline."""

    def build_bisk_hockenmaier_baseline(
        self,
        settings: BiskHockenmaierBaselineSettings,
        context: BaselineBuildContext,
        /,
    ) -> lofi.eval._bisk_hockenmaier_baseline.BiskHockenmaierOperation:
        """Bind a validated baseline declaration to its run resources."""
        ...


def _bisk_hockenmaier(
    settings: BiskHockenmaierBaselineSettings, context: BaselineBuildContext, /
) -> lofi.eval._bisk_hockenmaier_baseline.BiskHockenmaierOperation:
    """Build the Bisk--Hockenmaier baseline of one run."""
    module = cast(
        "_BiskHockenmaier", import_module("lofi.eval._bisk_hockenmaier_baseline")
    )
    return module.build_bisk_hockenmaier_baseline(settings, context)


BISK_HOCKENMAIER_PLUGIN: Final = ComponentPlugin(
    spec=BiskHockenmaierBaselineSpec, build=_bisk_hockenmaier
)
"""The installed Bisk--Hockenmaier baseline system."""
