"""Declared normalization for logical-form boundaries."""

from typing import TYPE_CHECKING

import didactic.api as dx

from lofi.interpret import NormalizationPolicy
from lofi.terms import EtaPolicy, Normalization, NormalizationOrder

if TYPE_CHECKING:
    from lofi._types import JsonValue

__all__ = ["NormalizationSpec"]


class NormalizationSpec(dx.Model, extra="forbid"):
    """Declare the equations and traversal used at an LF boundary."""

    order: NormalizationOrder = "leftmost-outermost"
    policy: NormalizationPolicy = "eager"
    eta: EtaPolicy = "off"
    associative_commutative: tuple[str, ...] = ()

    @dx.validates("associative_commutative")
    @classmethod
    def _symbols_are_distinct(cls, names: tuple[str, ...]) -> tuple[str, ...]:
        """Return a declaration with no repeated quotient symbol."""
        if len(set(names)) != len(names):
            msg = "associative-commutative symbols are distinct"
            raise ValueError(msg)
        return names

    def build(self, *, fuel: int) -> Normalization:
        """Build normalization within an owner-supplied reduction budget."""
        return Normalization(
            fuel=fuel,
            order=self.order,
            eta=self.eta,
            associative_commutative=self.associative_commutative,
        )

    def conventions(self) -> dict[str, JsonValue]:
        """Return the complete normalization convention vector."""
        return {
            "order": self.order,
            "policy": self.policy,
            "eta": self.eta,
            "associative_commutative": list(self.associative_commutative),
        }
