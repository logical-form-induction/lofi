"""Learned token encoder declaration."""

from typing import Literal, NamedTuple

import didactic.api as dx

from lofi._checks import ensure_finite
from lofi.spec.components.model import TokenEncoderSpec

__all__ = ["LearnedTokenEncoderSettings", "LearnedTokenEncoderSpec"]


class LearnedTokenEncoderSettings(NamedTuple):
    """The settings a learned token encoder is built from.

    Attributes
    ----------
    width : int
        The number of features of every row.
    """

    width: int
    init_std: float


class LearnedTokenEncoderSpec(TokenEncoderSpec):
    """Declares a token encoder whose rows are learned from random initialization.

    Attributes
    ----------
    kind : Literal["learned"]
        Registry tag.
    width : int
        The number of features of every row, which is also the encoder's output
        width.
    """

    kind: Literal["learned"] = "learned"
    width: int
    init_std: float | None = None

    __axioms__ = (dx.axiom("width > 0", message="a row of no features holds nothing"),)

    @dx.validates("init_std")
    @classmethod
    def _init_std_is_positive_or_absent(cls, value: float | None) -> float | None:
        """Validate an explicitly declared initialization scale."""
        if value is None:
            return None

        checked = ensure_finite("an initialization standard deviation", value)

        if checked <= 0:
            msg = "an initialization standard deviation is positive"
            raise ValueError(msg)

        return checked

    def settings(self) -> LearnedTokenEncoderSettings:
        """Return the settings the plugin builds this component from."""
        return LearnedTokenEncoderSettings(
            width=self.width,
            init_std=self.width**-0.5 if self.init_std is None else self.init_std,
        )
