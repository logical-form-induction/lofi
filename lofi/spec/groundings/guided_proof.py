"""Learned guided symbolic proof-search declarations."""

from typing import TYPE_CHECKING, Literal, NamedTuple, override

import didactic.api as dx

from lofi._checks import ensure_name, ensure_positive
from lofi.lf import SequentLimits
from lofi.spec.components.model import PairScorerSpec
from lofi.spec.provers import ProofBudgetSpec

if TYPE_CHECKING:
    from lofi.lf.provers import ProofBudget

__all__ = ["GuidedProofScorerSettings", "GuidedProofScorerSpec"]


class GuidedProofScorerSettings(NamedTuple):
    """Resolved guided search, encoder, and verification settings."""

    readout: str
    order: str
    limits: SequentLimits
    hidden: int
    verification: ProofBudget


class GuidedProofScorerSpec(PairScorerSpec):
    """Declare encoder-guided signed-tableau proof search."""

    kind: Literal["guided-proof"] = "guided-proof"
    readout: str
    order: str
    steps: int
    depth: int
    top_k: int
    instantiations: int
    hidden: int
    verification: ProofBudgetSpec

    @dx.validates("readout")
    @classmethod
    def _readout_is_named(cls, value: str) -> str:
        return ensure_name("a guided-proof readout", value)

    @dx.validates("order")
    @classmethod
    def _order_is_served(cls, value: str) -> str:
        held = ensure_name("a guided-proof order", value)
        if held not in {"first", "second", "higher"}:
            msg = f"guided-proof order is first, second or higher, not {held!r}"
            raise ValueError(msg)
        return held

    @dx.validates("steps", "depth", "top_k", "instantiations", "hidden")
    @classmethod
    def _resource_is_positive(cls, value: int) -> int:
        return ensure_positive("a guided-proof resource", value)

    def settings(self) -> GuidedProofScorerSettings:
        """Resolve each search and proof-service resource once."""
        return GuidedProofScorerSettings(
            self.readout,
            self.order,
            SequentLimits(
                steps=self.steps,
                depth=self.depth,
                top_k=self.top_k,
                instantiations=self.instantiations,
            ),
            self.hidden,
            self.verification.build(),
        )

    @override
    def checkpointed(self) -> bool:
        """Return true because the learned step scorer is stored state."""
        return True
