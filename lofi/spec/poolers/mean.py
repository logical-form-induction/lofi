"""Mean-subword pooler declaration."""

from typing import TYPE_CHECKING, Literal, NamedTuple, override

from lofi.spec.components.model import SubwordPoolerSpec
from lofi.spec.poolers._dependencies import ENCODER_DEPENDENCY

if TYPE_CHECKING:
    from lofi.spec._ports import Dependency

__all__ = ["MeanSubwordPoolerSettings", "MeanSubwordPoolerSpec"]


class MeanSubwordPoolerSettings(NamedTuple):
    """The settings of a pooler that takes none."""


class MeanSubwordPoolerSpec(SubwordPoolerSpec):
    """Declares a pooler that averages the rows each word reached.

    Attributes
    ----------
    kind : Literal["mean"]
        Registry tag.
    """

    kind: Literal["mean"] = "mean"

    @override
    def dependencies(self) -> tuple[Dependency, ...]:
        """Return the token encoder this pooler gathers rows from."""
        return ENCODER_DEPENDENCY

    def settings(self) -> MeanSubwordPoolerSettings:
        """Return the settings the plugin builds this component from."""
        return MeanSubwordPoolerSettings()
