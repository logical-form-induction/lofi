"""Derivation-entropy objective declaration."""

from typing import Literal, NamedTuple

from lofi.spec.components.loss import LossSpec

__all__ = [
    "DerivationEntropyDirection",
    "DerivationEntropyLossSettings",
    "DerivationEntropyLossSpec",
]

type DerivationEntropyDirection = Literal["minimize", "maximize"]
"""Whether fitting concentrates or spreads the tree posterior."""


class DerivationEntropyLossSettings(NamedTuple):
    """Resolved direction of a derivation-entropy objective."""

    direction: DerivationEntropyDirection


class DerivationEntropyLossSpec(LossSpec):
    """Declare a regularizer over the parser's tree-distribution entropy."""

    kind: Literal["derivation-entropy"] = "derivation-entropy"
    direction: DerivationEntropyDirection

    def settings(self) -> DerivationEntropyLossSettings:
        """Return whether the optimizer minimizes or maximizes entropy."""
        return DerivationEntropyLossSettings(self.direction)

    def required_chart_outputs(self) -> frozenset[str]:
        """Return the parser-owned entropy consumed by this objective."""
        return frozenset({"entropy"})
