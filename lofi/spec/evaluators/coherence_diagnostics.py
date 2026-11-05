"""Held-out coherence and collapse diagnostic declaration."""

from typing import TYPE_CHECKING, Literal, NamedTuple, override

import didactic.api as dx

from lofi._checks import ensure_finite, ensure_name
from lofi.spec._ports import Dependency
from lofi.spec.components.run import EvaluatorSpec, ParseSpec
from lofi.spec.evaluators.conventions import ConventionSettings, ConventionSpec
from lofi.spec.evaluators.decode import (
    AStarSpec,
    DecodeSpec,
    DerivationDecision,
    astar_parse_refusal,
    decode_decision,
)

if TYPE_CHECKING:
    from lofi.deduce import AStarOptions
    from lofi.spec._errors import DeclarationError

__all__ = ["CoherenceDiagnosticsSettings", "CoherenceDiagnosticsSpec"]


_METRICS = (
    "coherence-loss-per-token",
    "coherence-loss-by-action",
    "coherence-loss-by-span-length",
    "controller-entropy-per-span",
    "decoder-entropy-per-span",
    "type-distribution-collapse-rate",
    "coherence-accuracy-correlation",
    "combinator-use-jsd",
    "root-mass-agreement",
)


class CoherenceDiagnosticsSettings(NamedTuple):
    """Resolved settings for held-out coherence diagnostics."""

    objective: str
    readout: str
    root_mass: str
    collapse_threshold: float
    astar: AStarOptions
    decode: DerivationDecision
    conventions: ConventionSettings
    metrics: tuple[str, ...]


class CoherenceDiagnosticsSpec(EvaluatorSpec):
    """Declare objective, entropy, collapse, usage, and mass-agreement rows."""

    kind: Literal["coherence-diagnostics"] = "coherence-diagnostics"
    objective: str
    readout: str
    root_mass: str
    collapse_threshold: float
    astar: AStarSpec
    decode: DecodeSpec
    conventions: ConventionSpec

    @dx.validates("objective", "readout", "root_mass")
    @classmethod
    def _role_is_named(cls, value: str) -> str:
        """Return a named objective role, readout role, or mass output."""
        return ensure_name("a coherence diagnostic role", value)

    @dx.validates("collapse_threshold")
    @classmethod
    def _threshold_is_a_probability(cls, value: float) -> float:
        """Return a finite strict probability threshold."""
        threshold = ensure_finite("a collapse threshold", value)
        if not 0.0 < threshold < 1.0:
            msg = f"a collapse threshold lies strictly between zero and one: {value!r}"
            raise ValueError(msg)
        return threshold

    def parse_refusal(self, parse: ParseSpec, /) -> DeclarationError | None:
        """Return why the run parse cannot serve the declared A* decode."""
        return astar_parse_refusal(self.astar, parse, marginals=True)

    @override
    def dependencies(self) -> tuple[Dependency, ...]:
        """Return the complete-term readout used for entropy and symbolic decode."""
        return (Dependency(self.readout, "terms-readout"),)

    @override
    def metrics(self, labels: tuple[str, ...] = (), /) -> tuple[str, ...]:
        """Return every held-out coherence diagnostic row."""
        del labels
        return _METRICS

    def settings(self) -> CoherenceDiagnosticsSettings:
        """Return immutable evaluator settings."""
        return CoherenceDiagnosticsSettings(
            self.objective,
            self.readout,
            self.root_mass,
            self.collapse_threshold,
            self.astar.build(),
            decode_decision(self.decode),
            self.conventions.settings(),
            self.metrics(),
        )
