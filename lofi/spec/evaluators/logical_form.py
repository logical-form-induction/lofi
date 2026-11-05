"""Held-out logical-form evaluator declaration."""

from typing import Literal, NamedTuple, cast, override

import didactic.api as dx

from lofi._checks import ensure_name, ensure_non_negative, ensure_positive
from lofi.spec._ports import Dependency
from lofi.spec.components.run import EvaluatorSpec
from lofi.spec.evaluators.alignment import (
    AlignmentDeclaration,
    AlignmentSettings,
    AlignmentSpec,
)
from lofi.spec.evaluators.conventions import ConventionSettings, ConventionSpec

__all__ = [
    "LogicalFormEvaluatorSettings",
    "LogicalFormEvaluatorSpec",
    "LogicalFormMeasure",
]

type LogicalFormMeasure = Literal["exact", "alpha", "normal-form", "smatch", "nted"]
"""A headline score a logical-form evaluator reports."""


class LogicalFormEvaluatorSettings(NamedTuple):
    """Settings resolved from a logical-form evaluator declaration."""

    readout: str
    measure: LogicalFormMeasure
    restarts: int
    permutation_limit: int
    conventions: ConventionSettings
    alignment: AlignmentSettings


class LogicalFormEvaluatorSpec(EvaluatorSpec):
    """Declare logical-form matching over decoded held-out forms.

    Parameters
    ----------
    readout : str
        Complete-term readout role that decodes logical forms.
    measure : LogicalFormMeasure
        Headline member of the logical-form match family.
    restarts : int
        Random Smatch search restarts after the deterministic starts.
    permutation_limit : int
        Largest argument-permutation set the fiber scorer may enumerate.
    conventions : ConventionSpec
        Conventions the resulting rows are reported under.
    alignment : AlignmentSpec
        Symbol alignment fitted on development and frozen for test.
    """

    kind: Literal["logical-form"] = "logical-form"
    readout: str
    measure: LogicalFormMeasure
    restarts: int = 4
    permutation_limit: int = 5_040
    conventions: ConventionSpec
    alignment: AlignmentSpec

    @dx.validates("readout")
    @classmethod
    def _readout_is_named(cls, value: str) -> str:
        """Return a nonempty complete-term readout role."""
        return ensure_name("a logical-form term-readout role", value)

    @dx.validates("restarts")
    @classmethod
    def _restarts_are_non_negative(cls, value: int) -> int:
        """Return a nonnegative Smatch restart count."""
        return ensure_non_negative("a Smatch restart count", value)

    @dx.validates("permutation_limit")
    @classmethod
    def _permutation_limit_is_positive(cls, value: int) -> int:
        """Return a positive argument-permutation enumeration limit."""
        return ensure_positive("an argument-permutation limit", value)

    @override
    def dependencies(self) -> tuple[Dependency, ...]:
        """Return the declared complete-term readout dependency."""
        return (Dependency(self.readout, "terms-readout"),)

    @override
    def metrics(self, labels: tuple[str, ...] = (), /) -> tuple[str, ...]:
        """Return the selected recovery metric and its alignment controls."""
        del labels
        kind = cast("AlignmentDeclaration", self.alignment).settings().kind
        return (
            f"logical-form-{self.measure}",
            f"{kind}-accuracy",
            "random-relabeling-floor",
            "fiber-relative-recovery",
            "fixed-vocabulary-accuracy",
            "fiber-floor",
            "fiber-size",
        )

    @override
    def fit_split(self) -> str:
        """Return the exact retained partition used to fit alignment."""
        return cast("AlignmentDeclaration", self.alignment).settings().fit_on

    @override
    def score_split(self) -> str:
        """Return the exact retained partition used to score alignment."""
        return cast("AlignmentDeclaration", self.alignment).settings().score_on

    def settings(self) -> LogicalFormEvaluatorSettings:
        """Return the settings the evaluator plugin consumes."""
        return LogicalFormEvaluatorSettings(
            self.readout,
            self.measure,
            self.restarts,
            self.permutation_limit,
            self.conventions.settings(),
            cast("AlignmentDeclaration", self.alignment).settings(),
        )
