"""Declared bracketing, dependency, attachment, and normal form reports."""

from typing import TYPE_CHECKING, Literal, NamedTuple, cast, override

import didactic.api as dx

from lofi._checks import ensure_name, ensure_non_negative
from lofi.spec._ports import Dependency
from lofi.spec.components.run import EvaluatorSpec, ParseSpec
from lofi.spec.evaluators.conventions import ConventionSettings, ConventionSpec
from lofi.spec.evaluators.decode import (
    AStarSpec,
    DecodeMethodDeclaration,
    DecodeSpec,
    DerivationDecision,
    astar_parse_refusal,
    decode_decision,
)

if TYPE_CHECKING:
    from lofi.deduce import AStarOptions
    from lofi.spec._errors import DeclarationError

__all__ = ["StructuralDiagnosticsSettings", "StructuralDiagnosticsSpec"]

type AttachmentDirection = Literal["directed", "undirected"]
type DependencyMode = Literal[
    "labeled-directed",
    "labeled-undirected",
    "unlabeled-directed",
    "unlabeled-undirected",
]
type DependencyMeasure = Literal[
    "precision", "recall", "f-score", "sentence-exact", "coverage"
]


class StructuralDiagnosticsSettings(NamedTuple):
    """Resolved structural diagnostic settings."""

    readout: str
    astar: AStarOptions
    decode: DerivationDecision
    derivation_counts: bool
    dependency_interpretation: str
    labels: tuple[str, ...]
    lengths: tuple[int, ...]
    depths: tuple[int, ...]
    attachment_band: str
    attachment_limit: int | None
    attachment_direction: AttachmentDirection
    dependency_mode: DependencyMode
    dependency_measure: DependencyMeasure
    conventions: ConventionSettings
    attachment_row: str
    dependency_row: str
    metrics: tuple[str, ...]


class StructuralDiagnosticsSpec(EvaluatorSpec):
    """Declare model structure and the three fixed bracketing baselines."""

    kind: Literal["structural-diagnostics"] = "structural-diagnostics"
    readout: str
    astar: AStarSpec
    decode: DecodeSpec
    derivation_counts: bool
    dependency_interpretation: str
    labels: tuple[str, ...]
    lengths: tuple[int, ...]
    depths: tuple[int, ...]
    attachment_band: str
    attachment_limit: int | None
    attachment_direction: AttachmentDirection
    dependency_mode: DependencyMode
    dependency_measure: DependencyMeasure
    conventions: ConventionSpec

    @dx.validates("readout", "dependency_interpretation", "attachment_band")
    @classmethod
    def _names_are_nonempty(cls, value: str) -> str:
        return ensure_name("a structural diagnostic name", value)

    @dx.validates("labels")
    @classmethod
    def _labels_are_distinct(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        held = tuple(ensure_name("a bracketing label", value) for value in values)
        if len(set(held)) != len(held):
            msg = f"bracketing labels repeat: {held!r}"
            raise ValueError(msg)
        return held

    @dx.validates("lengths", "depths")
    @classmethod
    def _indices_are_distinct_nonnegative(
        cls, values: tuple[int, ...]
    ) -> tuple[int, ...]:
        held = tuple(
            ensure_non_negative("a bracketing bucket", value) for value in values
        )
        if len(set(held)) != len(held):
            msg = f"bracketing buckets repeat: {held!r}"
            raise ValueError(msg)
        return held

    @dx.validates("attachment_limit")
    @classmethod
    def _limit_is_nonnegative(cls, value: int | None) -> int | None:
        return (
            None
            if value is None
            else ensure_non_negative("an attachment length limit", value)
        )

    @dx.model_validator()
    def _checked_rules_have_a_proposer(self) -> StructuralDiagnosticsSpec:
        """Refuse structural parsing without an explicit checked-rule proposer."""
        if self.astar.proposer is None:
            msg = "structural diagnostics require an explicit A* proposer"
            raise ValueError(msg)
        return self

    def parse_refusal(self, parse: ParseSpec, /) -> DeclarationError | None:
        """Return why the run parse cannot serve the declared A* decode."""
        decode = cast("DecodeMethodDeclaration", self.decode)
        return astar_parse_refusal(
            self.astar, parse, marginals=decode.requires_marginals
        )

    @override
    def dependencies(self) -> tuple[Dependency, ...]:
        return (Dependency(self.readout, "terms-readout"),)

    def attachment_row(self) -> str:
        """Return the name of the attachment row the declared band reports."""
        return f"attachment:{self.attachment_band}:{self.attachment_direction}"

    def dependency_row(self) -> str:
        """Return the name of the dependency row the declared measure reports."""
        aggregation = "micro" if self.conventions.aggregation == "corpus" else "macro"
        suffix = (
            self.dependency_measure
            if self.dependency_measure in ("coverage", "sentence-exact")
            else f"{aggregation}-{self.dependency_measure}"
        )
        return f"dependency:{self.dependency_mode}:{self.conventions.coverage}:{suffix}"

    @override
    def metrics(self, labels: tuple[str, ...] = (), /) -> tuple[str, ...]:
        """Return the bracketing, attachment and derivation count rows."""
        del labels
        structural = (
            "unlabeled-bracketing-f1",
            *(f"bracketing-recall-label:{label}" for label in self.labels),
            *(f"bracketing-recall-length:{length}" for length in self.lengths),
            *(f"bracketing-recall-depth:{depth}" for depth in self.depths),
            "bracketing-baseline:left",
            "bracketing-baseline:right",
            "bracketing-baseline:balanced",
            self.attachment_row(),
            self.dependency_row(),
        )
        if not self.derivation_counts:
            return structural
        return (
            *structural,
            "normal-form-derivation-count",
            "unconstrained-derivation-count",
            "normal-form-retention",
        )

    def settings(self) -> StructuralDiagnosticsSettings:
        """Return immutable plugin settings."""
        return StructuralDiagnosticsSettings(
            self.readout,
            self.astar.build(),
            decode_decision(self.decode),
            self.derivation_counts,
            self.dependency_interpretation,
            self.labels,
            self.lengths,
            self.depths,
            self.attachment_band,
            self.attachment_limit,
            self.attachment_direction,
            self.dependency_mode,
            self.dependency_measure,
            self.conventions.settings(),
            self.attachment_row(),
            self.dependency_row(),
            self.metrics(),
        )
