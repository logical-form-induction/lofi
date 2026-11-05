"""Rule-set validation errors."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi.judge._values import Diagnostic

__all__ = ["RuleSetValidationError"]


class RuleSetValidationError(ValueError):
    """Rule-set capability error."""

    __slots__ = ("_diagnostics", "_requirement", "_ruleset")

    def __init__(
        self,
        *,
        ruleset: str,
        requirement: str,
        diagnostics: Sequence[Diagnostic],
    ) -> None:
        self._ruleset = ruleset
        self._requirement = requirement
        self._diagnostics = tuple(diagnostics)
        details = "; ".join(diagnostic.message for diagnostic in self._diagnostics)

        super().__init__(
            f"rule set {ruleset!r} does not meet {requirement!r}: {details}"
        )

    @property
    def ruleset(self) -> str:
        """Return the rejected rule-set name."""
        return self._ruleset

    @property
    def requirement(self) -> str:
        """Return the name of the requirement the rule set does not meet."""
        return self._requirement

    @property
    def diagnostics(self) -> tuple[Diagnostic, ...]:
        """Return every capability diagnostic."""
        return self._diagnostics
