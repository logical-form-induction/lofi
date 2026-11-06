"""Expected-form and sentence-vector grounding declaration."""

from typing import TYPE_CHECKING, Literal, NamedTuple, Self, cast, override

import didactic.api as dx

from lofi._checks import ensure_name, ensure_positive
from lofi.spec.components.model import GroundingSpec
from lofi.spec.groundings._policies import (
    _DEFAULT_GROUNDING_COMPOSITION,
    CompositionDeclaration,
    CompositionRuleSpec,
    CompositionSettings,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi.spec.components.model import ReadoutSpec

__all__ = ["ExpectedEmbeddingSettings", "ExpectedEmbeddingSpec"]


class ExpectedEmbeddingSettings(NamedTuple):
    """Resolved expected-embedding grounding settings."""

    readout: str
    vectors: Literal["expected-form", "sentence"]
    expectation: Literal["enumerate", "soft"]
    composition: CompositionSettings
    hidden: int


class ExpectedEmbeddingSpec(GroundingSpec):
    """Declare a label head over expected-form or sentence vectors."""

    kind: Literal["expected-embedding"] = "expected-embedding"
    readout: str
    vectors: Literal["expected-form", "sentence"]
    expectation: Literal["enumerate", "soft"]
    composition: CompositionRuleSpec = _DEFAULT_GROUNDING_COMPOSITION
    hidden: int

    @dx.validates("readout")
    @classmethod
    def _readout_is_named(cls, readout: str) -> str:
        """Return the named form readout role."""
        return ensure_name("an expected-embedding readout", readout)

    @dx.validates("hidden")
    @classmethod
    def _hidden_is_positive(cls, hidden: int) -> int:
        """Return a positive hidden width."""
        return ensure_positive("an entailment head hidden width", hidden)

    @dx.model_validator()
    def _sentence_vectors_have_no_soft_term_expectation(self) -> Self:
        """Refuse a term-distribution route when sentence vectors are selected."""
        if self.vectors == "sentence" and self.expectation != "enumerate":
            msg = "sentence-vector grounding has no soft term expectation"
            raise ValueError(msg)

        return self

    def settings(self) -> ExpectedEmbeddingSettings:
        """Return the grounding's interior settings."""
        composition = cast("CompositionDeclaration", self.composition)
        return ExpectedEmbeddingSettings(
            self.readout,
            self.vectors,
            self.expectation,
            composition.settings(),
            self.hidden,
        )

    @override
    def form_readouts(self) -> tuple[str, str] | None:
        """Return the shared form readout when grounding expected forms."""
        if self.vectors == "sentence":
            return None

        return self.readout, self.readout

    @override
    def supported_signals(self) -> frozenset[str]:
        """Return the likelihood signal supported by direct classification."""
        return frozenset({"likelihood"})

    @override
    def validate_readouts(self, readouts: Mapping[str, ReadoutSpec], /) -> None:
        """Require a complete term readout only for expected-form grounding."""
        if self.vectors == "sentence":
            return

        declaration = readouts.get(self.readout)

        if declaration is None:
            msg = f"expected-form grounding names absent readout {self.readout!r}"
            raise ValueError(msg)

        estimate = declaration.estimate_kind()

        if self.expectation == "enumerate" and estimate is None:
            msg = (
                f"expected-form grounding reads no complete terms from {self.readout!r}"
            )
            raise ValueError(msg)

        if self.expectation == "soft" and estimate is not None:
            msg = (
                f"soft expected-form grounding requires marginal rows from "
                f"{self.readout!r}, not {estimate} complete terms"
            )
            raise ValueError(msg)
