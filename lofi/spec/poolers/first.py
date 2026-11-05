"""First-subword pooler declaration."""

from typing import TYPE_CHECKING, Literal, NamedTuple, override

from lofi.spec.components.model import SubwordPoolerSpec
from lofi.spec.poolers._dependencies import ENCODER_DEPENDENCY

if TYPE_CHECKING:
    from lofi.spec._ports import Dependency

__all__ = ["FirstSubwordPoolerSettings", "FirstSubwordPoolerSpec"]


class FirstSubwordPoolerSettings(NamedTuple):
    """The settings of a pooler that takes none."""


class FirstSubwordPoolerSpec(SubwordPoolerSpec):
    """Declares a pooler that reads each word off the row of its first subword.

    Attributes
    ----------
    kind : Literal["first"]
        Registry tag.
    """

    kind: Literal["first"] = "first"

    @override
    def dependencies(self) -> tuple[Dependency, ...]:
        """Return the token encoder this pooler gathers rows from."""
        return ENCODER_DEPENDENCY

    def settings(self) -> FirstSubwordPoolerSettings:
        """Return the settings the plugin builds this component from."""
        return FirstSubwordPoolerSettings()
