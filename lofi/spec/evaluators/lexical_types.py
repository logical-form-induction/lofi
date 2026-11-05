"""Held-out lexical-type evaluator declaration."""

from typing import Literal, NamedTuple, cast, override

import didactic.api as dx

from lofi._checks import ensure_name
from lofi.spec._ports import Dependency
from lofi.spec.components.run import EvaluatorSpec
from lofi.spec.evaluators.alignment import (
    AlignmentDeclaration,
    AlignmentSettings,
    AlignmentSpec,
)
from lofi.spec.evaluators.conventions import ConventionSettings, ConventionSpec

__all__ = ["LexicalTypeEvaluatorSettings", "LexicalTypeEvaluatorSpec"]


class LexicalTypeEvaluatorSettings(NamedTuple):
    """The settings a lexical-type evaluator is built from.

    Attributes
    ----------
    readout : str
        The decoded term-readout role whose single-word frontiers predict each
        token's type.
    conventions : ConventionSettings
        The conventions tokens are scored and reported under.
    alignment : AlignmentSettings
        Development-fitted label alignment and its report settings.
    """

    readout: str
    conventions: ConventionSettings
    alignment: AlignmentSettings


class LexicalTypeEvaluatorSpec(EvaluatorSpec):
    """Declare lexical-type F1 over the held-out split.

    Parameters
    ----------
    readout : str
        The decoded term-readout role whose single-word frontiers predict each
        token's type.
    conventions : ConventionSpec
        The conventions tokens are scored and reported under.
    alignment : AlignmentSpec
        Label correspondence fitted on development and frozen for test.
    """

    kind: Literal["lexical-types"] = "lexical-types"
    readout: str
    conventions: ConventionSpec
    alignment: AlignmentSpec

    @dx.validates("readout")
    @classmethod
    def _readout_is_named(cls, value: str) -> str:
        """Return a nonempty term-readout role."""
        return ensure_name("a lexical-type term-readout role", value)

    @override
    def dependencies(self) -> tuple[Dependency, ...]:
        """Return the declared term-readout dependency."""
        return (Dependency(self.readout, "terms-readout"),)

    @override
    def metrics(self, labels: tuple[str, ...] = (), /) -> tuple[str, ...]:
        """Return recovery, aligned accuracy, and random-floor rows."""
        del labels
        kind = cast("AlignmentDeclaration", self.alignment).settings().kind
        return (
            "lexical-type-f1",
            f"primitive-{kind}-accuracy",
            "primitive-random-relabeling-floor",
            f"constructor-{kind}-accuracy",
            "constructor-random-relabeling-floor",
        )

    @override
    def fit_split(self) -> str:
        """Return the exact retained partition used to fit alignment."""
        return cast("AlignmentDeclaration", self.alignment).settings().fit_on

    @override
    def score_split(self) -> str:
        """Return the exact retained partition used to score alignment."""
        return cast("AlignmentDeclaration", self.alignment).settings().score_on

    def settings(self) -> LexicalTypeEvaluatorSettings:
        """Return the settings the plugin builds this evaluator from."""
        return LexicalTypeEvaluatorSettings(
            readout=self.readout,
            conventions=self.conventions.settings(),
            alignment=cast("AlignmentDeclaration", self.alignment).settings(),
        )
