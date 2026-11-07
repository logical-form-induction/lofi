"""Interannotator correlation ceiling declaration."""

from typing import Literal, NamedTuple, Self

import didactic.api as dx

from lofi._checks import ensure_positive

__all__ = [
    "InterannotatorCoefficientName",
    "InterannotatorSettings",
    "InterannotatorSpec",
]

type InterannotatorCoefficientName = Literal["spearman", "pearson"]
"""A correlation coefficient a ceiling may report."""


class InterannotatorSettings(NamedTuple):
    """The checked coefficients and bootstrap of an interannotator ceiling."""

    coefficients: tuple[InterannotatorCoefficientName, ...]
    replicates: int
    level: float


class InterannotatorSpec(dx.Model, extra="forbid"):
    """Declare an interannotator correlation ceiling over participant pairs.

    Participants who responded to exactly the same items form the pairs, and
    each pair's correlation is one unit of a participant pair bootstrap.

    Parameters
    ----------
    coefficients : tuple[InterannotatorCoefficientName, ...]
        The coefficients reported, one row each in declaration order.
    replicates : int
        The number of bootstrap resamples.
    level : float
        The coverage the bias-corrected interval aims at.
    """

    coefficients: tuple[InterannotatorCoefficientName, ...]
    replicates: int
    level: float

    @dx.validates("replicates")
    @classmethod
    def _replicates_are_positive(cls, value: int) -> int:
        """Return a positive replicate count."""
        return ensure_positive("an interannotator bootstrap replicate count", value)

    @dx.model_validator()
    def _the_ceiling_is_well_declared(self) -> Self:
        """Return distinct coefficients and a coverage strictly inside (0, 1)."""
        if not self.coefficients or len(set(self.coefficients)) != len(
            self.coefficients
        ):
            msg = "an interannotator ceiling declares one or more distinct coefficients"
            raise ValueError(msg)

        if not 0.0 < self.level < 1.0:
            msg = f"an interannotator interval level lies in (0, 1), not {self.level}"
            raise ValueError(msg)

        return self

    def settings(self) -> InterannotatorSettings:
        """Return the immutable ceiling settings."""
        return InterannotatorSettings(self.coefficients, self.replicates, self.level)
