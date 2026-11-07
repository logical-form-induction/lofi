"""Declared payoffs for proof-grounded inference."""

from __future__ import annotations

from typing import Final, Literal, final, override

import didactic.api as dx

from lofi._checks import ensure_finite
from lofi._frozen import FrozenValue

__all__ = [
    "CONTRADICTION_DEFINITIONS",
    "ContradictionDefinition",
    "InferenceReward",
    "RewardSpec",
]

type ContradictionDefinition = Literal["negated-hypothesis", "joint-inconsistency"]
"""How a pairwise proof treatment establishes contradiction."""

CONTRADICTION_DEFINITIONS: Final[tuple[ContradictionDefinition, ...]] = (
    "negated-hypothesis",
    "joint-inconsistency",
)
"""The contradiction treatments accepted at the declared reward boundary."""


@final
class InferenceReward(FrozenValue):
    """Map a decided label probability to its declared proof payoff."""

    _noun = "inference reward"
    __slots__ = ("_contradiction", "_proved", "_refuted")

    def __init__(
        self,
        *,
        proved: float,
        refuted: float,
        contradiction: ContradictionDefinition,
    ) -> None:
        held_proved = ensure_finite("a proved payoff", proved)
        held_refuted = ensure_finite("a refuted payoff", refuted)

        if held_proved <= held_refuted:
            msg = (
                "a proved payoff is greater than a refuted payoff, not "
                f"{held_proved!r} <= {held_refuted!r}"
            )
            raise ValueError(msg)

        if contradiction not in CONTRADICTION_DEFINITIONS:
            msg = f"unknown contradiction definition {contradiction!r}"
            raise ValueError(msg)

        object.__setattr__(self, "_proved", held_proved)
        object.__setattr__(self, "_refuted", held_refuted)
        object.__setattr__(self, "_contradiction", contradiction)

    @property
    def proved(self) -> float:
        """Return the payment at probability one."""
        return self._proved

    @property
    def refuted(self) -> float:
        """Return the payment at probability zero."""
        return self._refuted

    @property
    def contradiction(self) -> ContradictionDefinition:
        """Return the declared contradiction proof treatment."""
        return self._contradiction

    def pay(self, probability: float, /) -> float:
        """Return the affine expected payoff of a decided label probability."""
        held = ensure_finite("a decided label probability", probability)

        if not 0.0 <= held <= 1.0:
            msg = f"a decided label probability lies in [0, 1], not {held!r}"
            raise ValueError(msg)

        return self._refuted + (self._proved - self._refuted) * held

    def probability_baseline(self, value: float, /) -> float:
        """Return a reward baseline expressed on the unit-probability scale."""
        held = ensure_finite("an inference reward baseline", value)
        return (held - self._refuted) / (self._proved - self._refuted)

    @override
    def __eq__(self, other: object) -> bool:
        return (
            type(other) is InferenceReward
            and self._proved == other._proved
            and self._refuted == other._refuted
            and self._contradiction == other._contradiction
        )

    @override
    def __hash__(self) -> int:
        return hash((InferenceReward, self._proved, self._refuted, self._contradiction))

    @override
    def __repr__(self) -> str:
        return (
            f"InferenceReward(proved={self._proved!r}, refuted={self._refuted!r}, "
            f"contradiction={self._contradiction!r})"
        )


class RewardSpec(dx.Model, extra="forbid"):
    """Declare proof payoffs, contradiction semantics, and abstention treatment.

    ``contradiction`` chooses either directed entailment of the negated
    hypothesis or symmetric joint inconsistency. An abstention contributes no
    payment and is excluded from the objective mean.
    """

    proved: float
    refuted: float
    contradiction: ContradictionDefinition
    abstention: Literal["exclude"]

    @dx.model_validator()
    def _payoffs_are_ordered(self) -> RewardSpec:
        """Validate and return the declaration through its runtime value."""
        self.build()
        return self

    def build(self) -> InferenceReward:
        """Return the immutable runtime payoff."""
        return InferenceReward(
            proved=self.proved,
            refuted=self.refuted,
            contradiction=self.contradiction,
        )
