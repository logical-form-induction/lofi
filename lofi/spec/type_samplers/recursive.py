"""Recursive type-sampler declaration."""

from typing import Literal, NamedTuple

import didactic.api as dx

from lofi._checks import ensure_positive
from lofi.spec.components.trainer import TypeSamplerSpec

__all__ = ["RecursiveTypeSamplerSettings", "RecursiveTypeSamplerSpec"]


class RecursiveTypeSamplerSettings(NamedTuple):
    """The settings a recursive type sampler is built from.

    Attributes
    ----------
    max_size : int
        Inclusive node bound for the candidate inventory.
    """

    max_size: int


class RecursiveTypeSamplerSpec(TypeSamplerSpec):
    """Declares recursive sampling with a finite candidate inventory.

    Attributes
    ----------
    kind : Literal["recursive"]
        Registry tag.
    max_size : int
        Inclusive node bound for the candidate inventory.
    """

    kind: Literal["recursive"] = "recursive"
    max_size: int

    @dx.validates("max_size")
    @classmethod
    def _size_is_positive(cls, value: int) -> int:
        """Validate the candidate-inventory node bound.

        Parameters
        ----------
        value : int
            Bound to validate.
        """
        return ensure_positive("max_size", value)

    def settings(self) -> RecursiveTypeSamplerSettings:
        """Return the settings the plugin builds this sampler from."""
        return RecursiveTypeSamplerSettings(max_size=self.max_size)
