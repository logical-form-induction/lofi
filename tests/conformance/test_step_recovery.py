"""Concrete conformance cases for observed-step recovery."""

from typing import TYPE_CHECKING

import pytest
from tests.conformance.step_recovery import (
    StepRecoveryConformance,
    StepRecoverySuite,
)
from tests.support.rules import declared_rules

from lofi.spec import (
    ConstantPatternSpec,
    PatternRuleSpec,
    RuleSetSpec,
    TermPatternSpec,
)
from lofi.spec.grammar import SignatureSpec

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi.judge import Rule
    from lofi.terms import Term


class FixedRecovery:
    """A small external step-recovery implementation."""

    def recover(
        self,
        inputs: Sequence[int],
        output: int,
        /,
    ) -> tuple[str, ...]:
        """Return two labels for the declared arithmetic step."""
        if tuple(inputs) == (1, 2) and output == 3:
            return ("first", "second")

        return ()


def _fixed_conformance() -> StepRecoveryConformance[int, str]:
    """Build conformance inputs for an external recovery relation."""
    return StepRecoveryConformance(
        FixedRecovery(),
        (1, 2),
        3,
        ("first", "second"),
        (2, 1),
        4,
    )


def _ruleset_conformance() -> StepRecoveryConformance[
    Term,
    Rule[Term],
]:
    """Build conformance inputs for the concrete rule-set surface."""
    adopted = declared_rules(
        SignatureSpec(
            name="step-recovery-conformance",
            primitives=("input", "output", "other"),
            syntax="s-expression",
        ),
        RuleSetSpec(
            name="step-recovery",
            rules=(
                PatternRuleSpec(
                    name="first",
                    premises=(ConstantPatternSpec(name="input"),),
                    conclusion=ConstantPatternSpec(name="output"),
                ),
                PatternRuleSpec(
                    name="second",
                    premises=(TermPatternSpec(name="X", sort="Cat"),),
                    conclusion=ConstantPatternSpec(name="output"),
                ),
            ),
        ),
    )
    rules = adopted.rule_set
    first, second = rules
    resolve = adopted.resolve_category
    source, target, other = (
        resolve(f'("#const[Cat]" "{name}")') for name in ("input", "output", "other")
    )

    return StepRecoveryConformance(
        rules,
        (source,),
        target,
        (first, second),
        (other,),
        source,
    )


class TestExternalStepRecovery(StepRecoverySuite):
    """Conformance cases for an external structural implementation."""

    @pytest.fixture
    def step_recovery(self) -> StepRecoveryConformance[int, str]:
        """Build the external fixture."""
        return _fixed_conformance()


class TestRuleSetStepRecovery(StepRecoverySuite):
    """Conformance cases for the concrete rule-set implementation."""

    @pytest.fixture
    def step_recovery(
        self,
    ) -> StepRecoveryConformance[Term, Rule[Term]]:
        """Build the rule-set fixture."""
        return _ruleset_conformance()
