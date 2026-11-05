"""Held-out gold-derivation reachability evaluator declaration."""

from typing import Final, Literal, NamedTuple, override

import didactic.api as dx

from lofi._checks import ensure_name
from lofi.spec.components.run import EvaluatorSpec
from lofi.spec.evaluators.conventions import ConventionSettings, ConventionSpec

__all__ = [
    "REACHABILITY_ROWS",
    "ReachabilityEvaluatorSettings",
    "ReachabilityEvaluatorSpec",
]

REACHABILITY_ROWS: Final[tuple[str, ...]] = (
    "reachability-steps",
    "reachability-derivations",
    "reachability-blocked-label",
    "reachability-blocked-rule",
    "reachability-blocked-construction",
)
"""The rows a reachability evaluator reports, in report order."""


class ReachabilityEvaluatorSettings(NamedTuple):
    """Settings resolved from a reachability evaluator declaration."""

    inventory: str
    corpus: str
    language: str
    conventions: ConventionSettings


class ReachabilityEvaluatorSpec(EvaluatorSpec):
    """Declare gold-derivation reachability under the run grammar's inventory.

    Each held-out gold derivation is checked against the label and rule
    inventory of the grammar the parser was trained under. The rows report
    step and derivation reachability, and how many unreachable derivations a
    missing label, an unrecovered recorded rule, or an unrecovered unnamed
    construction blocks, with one component per blocking histogram cell.

    Parameters
    ----------
    inventory : str
        The name the rule inventory is reported under.
    corpus : str
        The name the evaluated corpus is reported under.
    language : str
        The language of the evaluated derivations.
    conventions : ConventionSpec
        Conventions the resulting rows are reported under.
    """

    kind: Literal["reachability"] = "reachability"
    inventory: str
    corpus: str
    language: str
    conventions: ConventionSpec

    @dx.validates("inventory", "corpus", "language")
    @classmethod
    def _names_are_nonempty(cls, value: str) -> str:
        """Return a nonempty report name."""
        return ensure_name("a reachability report name", value)

    @override
    def metrics(self, labels: tuple[str, ...] = (), /) -> tuple[str, ...]:
        """Return every row this declaration emits, in report order."""
        del labels
        return REACHABILITY_ROWS

    def settings(self) -> ReachabilityEvaluatorSettings:
        """Return the settings the evaluator plugin consumes."""
        return ReachabilityEvaluatorSettings(
            self.inventory, self.corpus, self.language, self.conventions.settings()
        )
