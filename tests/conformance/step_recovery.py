"""Conformance tests for observed-step recovery."""

from typing import NamedTuple, NoReturn

import pytest

from lofi._checks import LawViolationError
from lofi.data import StepRecovery

__all__ = ["StepRecoveryConformance", "StepRecoverySuite"]


class StepRecoveryConformance[T, Recovered](NamedTuple):
    """Inputs shared by step-recovery implementations."""

    subject: StepRecovery[T, Recovered]
    recovered_inputs: tuple[T, ...]
    recovered_output: T
    expected: tuple[Recovered, ...]
    unrecovered_inputs: tuple[T, ...]
    unrecovered_output: T


def _fail(detail: str, *, law: str) -> NoReturn:
    """Raise a named conformance failure.

    Parameters
    ----------
    detail : str
    law : str
    """
    message = f"{law}: {detail}"

    raise LawViolationError(message)


class StepRecoverySuite:
    """Conformance laws for observed-step recovery relations."""

    pytestmark = pytest.mark.conformance

    def test_the_subject_satisfies_the_step_recovery_protocol[T, Recovered](
        self,
        step_recovery: StepRecoveryConformance[T, Recovered],
    ) -> None:
        if not isinstance(step_recovery.subject, StepRecovery):
            _fail(
                "the subject does not publish recover",
                law="the_subject_satisfies_the_step_recovery_protocol",
            )

    def test_a_recovered_step_returns_every_expected_value[T, Recovered](
        self,
        step_recovery: StepRecoveryConformance[T, Recovered],
    ) -> None:
        found = step_recovery.subject.recover(
            step_recovery.recovered_inputs,
            step_recovery.recovered_output,
        )

        if found != step_recovery.expected or not found:
            _fail(
                f"recovered {found!r}, expected {step_recovery.expected!r}",
                law="a_recovered_step_returns_every_expected_value",
            )

    def test_an_unrecovered_step_returns_no_value[T, Recovered](
        self,
        step_recovery: StepRecoveryConformance[T, Recovered],
    ) -> None:
        found = step_recovery.subject.recover(
            step_recovery.unrecovered_inputs,
            step_recovery.unrecovered_output,
        )

        if found:
            _fail(
                f"an unrecovered step returned {found!r}",
                law="an_unrecovered_step_returns_no_value",
            )

    def test_recovered_values_have_stable_order[T, Recovered](
        self,
        step_recovery: StepRecoveryConformance[T, Recovered],
    ) -> None:
        if len(step_recovery.expected) < 2:
            _fail(
                "the fixture supplies fewer than two recovered values",
                law="recovered_values_have_stable_order",
            )

        first = step_recovery.subject.recover(
            step_recovery.recovered_inputs,
            step_recovery.recovered_output,
        )
        second = step_recovery.subject.recover(
            step_recovery.recovered_inputs,
            step_recovery.recovered_output,
        )

        if first != step_recovery.expected or second != first:
            _fail(
                f"successive recoveries returned {first!r} and {second!r}",
                law="recovered_values_have_stable_order",
            )

    def test_recovery_does_not_mutate_its_inputs[T, Recovered](
        self,
        step_recovery: StepRecoveryConformance[T, Recovered],
    ) -> None:
        inputs = list(step_recovery.recovered_inputs)
        before = tuple(inputs)
        step_recovery.subject.recover(inputs, step_recovery.recovered_output)

        if tuple(inputs) != before:
            _fail(
                f"recovery changed inputs from {before!r} to {tuple(inputs)!r}",
                law="recovery_does_not_mutate_its_inputs",
            )
