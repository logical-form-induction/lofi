"""Conformance tests for logical-form typing systems."""

from typing import TYPE_CHECKING, NamedTuple

import pytest

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi.terms import OpName, Term
    from lofi.typecheck import TypingResult, TypingRule, TypingSystem

__all__ = ["TypingSystemConformance", "TypingSystemSuite"]


class TypingSystemConformance[S](NamedTuple):
    """Inputs shared by typing-system implementations."""

    subject: TypingSystem[S]
    logical_form: Term
    signature: S
    rules: Mapping[OpName, TypingRule]
    expected: TypingResult
    general: Term
    specific: Term
    unrelated: Term


class TypingSystemSuite:
    """Conformance laws for typing-system implementations."""

    pytestmark = pytest.mark.conformance

    def test_check_returns_the_full_typing_judgment[S](
        self,
        conformance: TypingSystemConformance[S],
    ) -> None:
        result = conformance.subject.check(
            conformance.logical_form,
            conformance.signature,
            conformance.rules,
        )

        assert result == conformance.expected

    def test_subsumption_binds_only_the_general_type[S](
        self,
        conformance: TypingSystemConformance[S],
    ) -> None:
        subject = conformance.subject

        assert subject.subsumes(conformance.general, conformance.specific) is not None
        assert subject.subsumes(conformance.specific, conformance.general) is None
        assert subject.subsumes(conformance.general, conformance.unrelated) is None

    def test_unification_is_symmetric[S](
        self,
        conformance: TypingSystemConformance[S],
    ) -> None:
        subject = conformance.subject

        assert subject.unifies(conformance.general, conformance.specific) is not None
        assert subject.unifies(conformance.specific, conformance.general) is not None
