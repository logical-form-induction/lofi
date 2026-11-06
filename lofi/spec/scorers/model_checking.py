"""Scored finite-model checking declaration."""

from typing import Literal, NamedTuple, Protocol, cast

import didactic.api as dx

from lofi._checks import ensure_finite
from lofi.spec.components.model import PairScorerSpec
from lofi.spec.fuzzy_logic import FuzzyLogicSettings, FuzzyLogicSpec
from lofi.spec.structures import StructureSourceSpec

__all__ = ["ModelCheckingScorerSettings", "ModelCheckingScorerSpec"]


class _SourceDeclaration(Protocol):
    """Resolve any registered structure-source declaration."""

    @property
    def kind(self) -> str: ...

    def settings(self) -> object: ...


class ModelCheckingScorerSettings(NamedTuple):
    """Resolved model checking policy and structure-source declaration."""

    source_kind: str
    source: object
    logic: FuzzyLogicSettings
    truth_threshold: float
    abstain_band: float
    premise_floor: float
    unsatisfied_premise: Literal["undecided", "zero"]


class ModelCheckingScorerSpec(PairScorerSpec):
    """Declare conditional satisfaction over finite structures."""

    kind: Literal["model-checking"] = "model-checking"
    source: StructureSourceSpec
    logic: FuzzyLogicSpec
    truth_threshold: float
    abstain_band: float
    premise_floor: float
    unsatisfied_premise: Literal["undecided", "zero"]

    @dx.validates("truth_threshold", "abstain_band", "premise_floor")
    @classmethod
    def _probability(cls, value: float) -> float:
        """Return a finite unit interval policy value."""
        held = ensure_finite("a model-checking threshold", value)

        if not 0.0 <= held <= 1.0:
            msg = "a model-checking threshold lies in the unit interval"
            raise ValueError(msg)

        return held

    @dx.model_validator()
    def _abstention_band_fits(self) -> ModelCheckingScorerSpec:
        """Keep the abstention band inside the unit interval."""
        if self.abstain_band > min(self.truth_threshold, 1.0 - self.truth_threshold):
            msg = "the abstention band crosses a unit-interval boundary"
            raise ValueError(msg)

        return self

    def settings(self) -> ModelCheckingScorerSettings:
        """Resolve immutable interior settings and discard boundary models."""
        source = cast("_SourceDeclaration", self.source)
        return ModelCheckingScorerSettings(
            source.kind,
            source.settings(),
            self.logic.settings(),
            self.truth_threshold,
            self.abstain_band,
            self.premise_floor,
            self.unsatisfied_premise,
        )
