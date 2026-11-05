"""Differentiable proof-search scoring declarations."""

from typing import TYPE_CHECKING, Literal, NamedTuple, Self, override

import didactic.api as dx

from lofi._checks import ensure_finite, ensure_name, ensure_positive
from lofi.spec.components.model import PairScorerSpec
from lofi.spec.provers import ProofBudgetSpec

if TYPE_CHECKING:
    from lofi.lf.provers import ProofBudget

__all__ = [
    "NeuralProofScorerSettings",
    "NeuralProofScorerSpec",
    "RuleTemplate",
    "SimilarityKernelSettings",
    "SimilarityKernelSpec",
]

type RuleTemplate = Literal[
    "identity",
    "conjunction-elimination",
    "conjunction-introduction",
    "modus-ponens",
]
type SimilarityKernel = str
"""The installed ``lofi.similarity_kernel`` entry soft unification scores with."""


class SimilarityKernelSettings(NamedTuple):
    """Resolved soft-unification similarity kernel."""

    kind: SimilarityKernel
    temperature: float


class NeuralProofScorerSettings(NamedTuple):
    """Resolved bounded neural proof-search settings."""

    order: str
    instantiations: int
    templates: tuple[RuleTemplate, ...]
    depth: int
    width: int
    similarity: SimilarityKernelSettings
    verification: ProofBudget


class SimilarityKernelSpec(dx.Model, extra="forbid"):
    """Declare the choice-row similarity used by soft unification."""

    kind: str
    temperature: float = 1.0

    @dx.validates("kind")
    @classmethod
    def _kernel_is_named(cls, value: str) -> str:
        return ensure_name("a similarity kernel", value)

    @dx.validates("temperature")
    @classmethod
    def _positive_temperature(cls, value: float) -> float:
        held = ensure_finite("similarity temperature", value)

        if held <= 0.0:
            msg = "similarity temperature is positive"
            raise ValueError(msg)

        return held

    def settings(self) -> SimilarityKernelSettings:
        """Return the tensor-free interior kernel settings."""
        return SimilarityKernelSettings(self.kind, self.temperature)


class NeuralProofScorerSpec(PairScorerSpec):
    """Declare type-directed neural backward chaining and proof verification."""

    kind: Literal["neural-proof"] = "neural-proof"
    order: str
    instantiations: int
    templates: tuple[RuleTemplate, ...]
    depth: int
    width: int
    similarity: SimilarityKernelSpec
    verification: ProofBudgetSpec

    @dx.validates("depth", "width", "instantiations")
    @classmethod
    def _positive_resource(cls, value: int) -> int:
        return ensure_positive("a neural proof resource", value)

    @dx.validates("order")
    @classmethod
    def _served_order(cls, value: str) -> str:
        held = ensure_name("a neural proof order", value)
        if held not in {"first", "second"}:
            msg = f"neural proof order is first or second, not {held!r}"
            raise ValueError(msg)
        return held

    @dx.model_validator()
    def _distinct_templates(self) -> Self:
        if not self.templates:
            msg = "a neural proof search declares at least one rule template"
            raise ValueError(msg)

        if len(set(self.templates)) != len(self.templates):
            msg = "a neural proof search cannot declare a rule template twice"
            raise ValueError(msg)

        return self

    def settings(self) -> NeuralProofScorerSettings:
        """Resolve the proof search and verification resources once."""
        return NeuralProofScorerSettings(
            self.order,
            self.instantiations,
            self.templates,
            self.depth,
            self.width,
            self.similarity.settings(),
            self.verification.build(),
        )

    @override
    def checkpointed(self) -> bool:
        """Return true because the choice embeddings are learned state."""
        return True
