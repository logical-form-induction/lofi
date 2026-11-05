"""Category-side evaluation for logical-form induction runs."""

from typing import TYPE_CHECKING, Literal, NamedTuple, cast, override

import didactic.api as dx

from lofi._checks import ensure_finite, ensure_name
from lofi.spec._ports import Dependency
from lofi.spec.components.run import EvaluationData, EvaluatorSpec, ParseSpec
from lofi.spec.evaluators.alignment import (
    AlignmentDeclaration,
    AlignmentSettings,
    AlignmentSpec,
)
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

__all__ = ["CategorySideEvaluatorSettings", "CategorySideEvaluatorSpec"]

type LabeledDependencyMode = Literal["labeled-directed", "labeled-undirected"]
"""A dependency matching relation that retains labels and argument slots."""


class CategorySideEvaluatorSettings(NamedTuple):
    """Resolved category-side evaluator settings."""

    readout: str
    supertag_beta: float
    astar: AStarOptions
    decode: DerivationDecision
    dependency_interpretation: str
    dependency_mode: LabeledDependencyMode
    conventions: ConventionSettings
    alignment: AlignmentSettings
    metrics: tuple[str, ...]


class CategorySideEvaluatorSpec(EvaluatorSpec):
    """Declare category recovery beside logical-form and inference evaluation.

    One decoded category readout supplies token beams, the lexical source for
    symbolic A* dependency decoding, and complete lexical types. Label
    correspondence is fitted on development and frozen before the test rows
    are scored.
    """

    kind: Literal["category-side"] = "category-side"
    readout: str
    supertag_beta: float
    astar: AStarSpec
    decode: DecodeSpec
    dependency_interpretation: str
    dependency_mode: LabeledDependencyMode
    conventions: ConventionSpec
    alignment: AlignmentSpec

    @dx.validates("readout", "dependency_interpretation")
    @classmethod
    def _names_are_nonempty(cls, value: str) -> str:
        """Return a nonempty category readout or interpretation name."""
        return ensure_name("a category-side evaluator name", value)

    @dx.validates("supertag_beta")
    @classmethod
    def _beta_is_a_threshold(cls, value: float) -> float:
        """Return a finite C&C-style beam threshold."""
        held = ensure_finite("a category-side supertag beta", value)
        if not 0.0 <= held < 1.0:
            msg = f"a category-side supertag beta is outside [0, 1): {value!r}"
            raise ValueError(msg)
        return held

    @dx.model_validator()
    def _checked_rules_have_a_proposer(self) -> CategorySideEvaluatorSpec:
        """Refuse category-side parsing without an explicit A* proposer."""
        if self.astar.proposer is None:
            msg = "category-side evaluation requires an explicit A* proposer"
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
        """Return the one decoded category readout dependency."""
        return (Dependency(self.readout, "terms-readout"),)

    @override
    def fit_split(self) -> str:
        """Return the exact retained partition used to fit category alignment."""
        return cast("AlignmentDeclaration", self.alignment).settings().fit_on

    @override
    def score_split(self) -> str:
        """Return the exact retained partition used for category scoring."""
        return cast("AlignmentDeclaration", self.alignment).settings().score_on

    @override
    def evaluation_data(self) -> EvaluationData:
        """Return that the evaluator scores each pair side's gold derivation."""
        return EvaluationData.PAIR_SIDE_DERIVATIONS

    @override
    def metrics(self, labels: tuple[str, ...] = (), /) -> tuple[str, ...]:
        """Return supertagging, dependency, and aligned lexical-type rows."""
        del labels
        aggregation = "micro" if self.conventions.aggregation == "corpus" else "macro"
        alignment = cast("AlignmentDeclaration", self.alignment).settings().kind
        return (
            "supertag-accuracy",
            (
                f"dependency:{self.dependency_mode}:{self.conventions.coverage}:"
                f"{aggregation}-f-score"
            ),
            "lexical-type-f1",
            f"primitive-{alignment}-accuracy",
            "primitive-random-relabeling-floor",
            f"constructor-{alignment}-accuracy",
            "constructor-random-relabeling-floor",
        )

    def settings(self) -> CategorySideEvaluatorSettings:
        """Return the immutable settings consumed by the evaluator plugin."""
        return CategorySideEvaluatorSettings(
            readout=self.readout,
            supertag_beta=self.supertag_beta,
            astar=self.astar.build(),
            decode=decode_decision(self.decode),
            dependency_interpretation=self.dependency_interpretation,
            dependency_mode=self.dependency_mode,
            conventions=self.conventions.settings(),
            alignment=cast("AlignmentDeclaration", self.alignment).settings(),
            metrics=self.metrics(),
        )
