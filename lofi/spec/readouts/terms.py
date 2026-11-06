"""Term readout declaration over one represented space."""

import enum
from typing import Literal, NamedTuple, cast, override

import didactic.api as dx

from lofi._checks import ensure_name
from lofi.spec.approximation import ApproximationRequest, residual_mass_bound
from lofi.spec.components.model import ReadoutSpec
from lofi.spec.decoding import DecodingDeclaration, DecodingSettings, DecodingSpec

__all__ = [
    "DecodedMeaningSettings",
    "DecodedMeaningSpec",
    "MeaningInductionSettings",
    "MeaningInductionSpec",
    "ProofPushforwardSettings",
    "ProofTermMeaningSettings",
    "ProofTermMeaningSpec",
    "ReadoutMass",
    "ReadoutSource",
    "TargetSource",
    "TermsReadoutSettings",
    "TermsReadoutSpec",
]


class ProofPushforwardSettings(NamedTuple):
    """Resolved weighting and residual bound of a proof-action pushforward."""

    weighting: str
    residual_mass: float


class DecodedMeaningSettings(NamedTuple):
    """Resolved direct-decoding meaning induction."""

    kind: Literal["decoded"]

    def chart_outputs(self) -> frozenset[str]:
        """Return the chart outputs direct decoding reads, of which there are none."""
        return frozenset()

    def pushforward(self) -> ProofPushforwardSettings | None:
        """Return nothing, since direct decoding pushes no lexical form."""
        return None


class ProofTermMeaningSettings(NamedTuple):
    """Resolved proof term meaning induction."""

    kind: Literal["proof-term"]
    weighting: str
    residual_mass: float

    def chart_outputs(self) -> frozenset[str]:
        """Return the split weights and proof actions the induction pushes through."""
        return frozenset({"alpha", "actions"})

    def pushforward(self) -> ProofPushforwardSettings:
        """Return the weighting and bound lexical forms are pushed through under."""
        return ProofPushforwardSettings(self.weighting, self.residual_mass)


type MeaningInductionSettings = DecodedMeaningSettings | ProofTermMeaningSettings
"""Resolved settings of one complete-meaning induction route."""


class MeaningInductionSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declare how complete logical forms are induced at internal spans."""

    def settings(self) -> MeaningInductionSettings:
        """Return the resolved meaning-induction settings."""
        msg = f"{type(self).__name__} is the meaning-induction union, not a route"
        raise TypeError(msg)


class DecodedMeaningSpec(MeaningInductionSpec):
    """Decode a complete logical form independently at every span."""

    kind: Literal["decoded"] = "decoded"

    @override
    def settings(self) -> DecodedMeaningSettings:
        """Return the direct-decoding settings."""
        return DecodedMeaningSettings(self.kind)


class ProofTermMeaningSpec(MeaningInductionSpec):
    """Push lexical forms through semantic proof actions.

    ``residual_mass`` bounds the truncation residual of each derived phrase cell.
    A derived cell keeps only the mass its children's frontiers retain, so its
    residual compounds the omissions of every word beneath it; word decoding
    keeps the readout's own residual mass bound.
    """

    kind: Literal["proof-term"] = "proof-term"
    weighting: str = "distinct"
    residual_mass: float

    @dx.validates("weighting")
    @classmethod
    def _weighting_is_named(cls, value: str) -> str:
        """Return a nonempty installed meaning-weighting name."""
        return ensure_name("a meaning weighting", value)

    @dx.validates("residual_mass")
    @classmethod
    def _residual_mass_is_a_bound(cls, value: float) -> float:
        """Return the derived-cell residual mass bound."""
        return residual_mass_bound(value)

    @override
    def settings(self) -> ProofTermMeaningSettings:
        """Return the proof term induction settings."""
        return ProofTermMeaningSettings(self.kind, self.weighting, self.residual_mass)


class TargetSource(enum.StrEnum):
    """Names a target set supplied by the grammar boundary."""

    GRAMMAR_GOAL = "grammar-goal"


class ReadoutSource(enum.StrEnum):
    """Names the model value a term readout decodes."""

    INSIDE = "inside"
    SPAN = "span"


class ReadoutMass(enum.StrEnum):
    """Names how a readout computes a declared target's mass."""

    FACTORIZED = "factorized"
    COUPLED = "coupled"


class TermsReadoutSettings(NamedTuple):
    """Resolved settings of a term readout."""

    space: str
    reads: ReadoutSource
    decoding: DecodingSettings | None
    targets: TargetSource | None
    output: str | None
    mass: ReadoutMass
    meanings: MeaningInductionSettings


class TermsReadoutSpec(ReadoutSpec):
    """Declare terms read from one represented space at each chart span.

    Without a decoding strategy, the readout publishes the decoder's marginal
    per-choice rows and optional target mass for objectives such as coherence. A
    decoding strategy publishes weighted complete terms and their residual mass.
    Complete meanings are either decoded at each span or constructed at internal
    spans by pushing lexical forms through the parser's semantic proof actions.
    """

    kind: Literal["terms"] = "terms"
    reads: ReadoutSource
    decoding: DecodingSpec | None = None
    targets: TargetSource | None = None
    output: str | None = None
    mass: ReadoutMass = ReadoutMass.FACTORIZED
    meanings: MeaningInductionSpec = DecodedMeaningSpec()

    @dx.validates("output")
    @classmethod
    def _output_is_named(cls, value: str | None) -> str | None:
        """Return a nonempty optional target mass output name."""
        return None if value is None else ensure_name("a term readout output", value)

    @dx.model_validator()
    def _target_route(self) -> TermsReadoutSpec:
        """Require target mass to be computed from marginal rows."""
        if self.targets is None and self.output is not None:
            msg = "a target-mass output needs a target source"
            raise ValueError(msg)

        if self.targets is not None and self.decoding is not None:
            msg = "target mass is computed from marginal rows, not decoded terms"
            raise ValueError(msg)

        pushed = self.meanings.settings().pushforward()

        if pushed is not None and self.decoding is None:
            msg = "proof-term meanings require a lexical decoding strategy"
            raise ValueError(msg)

        return self

    def settings(self) -> TermsReadoutSettings:
        """Return the resolved readout settings."""
        decoding = self.decoding
        return TermsReadoutSettings(
            space=self.space,
            reads=self.reads,
            decoding=(
                None
                if decoding is None
                else cast("DecodingDeclaration", decoding).settings()
            ),
            targets=self.targets,
            output=self.output,
            mass=self.mass,
            meanings=self.meanings.settings(),
        )

    @override
    def published_masses(self) -> frozenset[str]:
        """Return the ids of the target mass pairs this readout publishes.

        The readout publishes the mass of every target source the grammar
        supplies under the source's name, except its own declared target
        source, whose mass it publishes under ``output`` when one is declared.
        """
        sources = {str(source) for source in TargetSource}

        if self.targets is None:
            return frozenset(sources)

        own = str(self.targets)

        return frozenset(
            (sources - {own}) | {own if self.output is None else self.output}
        )

    def required_chart_outputs(self) -> frozenset[str]:
        """Return the chart vectors and weights this readout consumes."""
        required = {"cells" if self.reads is ReadoutSource.INSIDE else "spans"}

        if self.targets is not None:
            required.update({"alpha", "mu"})

        if self.meanings.settings().kind == "proof-term":
            required.update({"alpha", "actions"})

        return frozenset(required)

    def approximation_request(self) -> ApproximationRequest | None:
        """Return the residual budget requested by exhaustive decoding."""
        decoding = self.decoding

        if decoding is None:
            return None

        settings = cast("DecodingDeclaration", decoding).settings()

        if settings.strategy != "exhaustive":
            return None

        return ApproximationRequest(settings.residual_mass)

    @override
    def estimate_kind(self) -> str | None:
        """Return whether complete forms are sampled or enumerated."""
        decoding = self.decoding

        if decoding is None:
            return None

        settings = cast("DecodingDeclaration", decoding).settings()
        return "sampled" if settings.strategy == "sample" else "enumerated"
