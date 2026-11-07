"""Reach lexical-entry type constraints through a configuration document."""

from typing import TYPE_CHECKING

import pytest
from tests.acceptance.driver import composed

from lofi.run import plan_training
from lofi.spec import DeclarationError
from lofi.typecheck import TypeChecker

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi.terms import OpName, Path, Term
    from lofi.typecheck import LexicalSignature, TypingResult, TypingRule


def test_a_document_refuses_a_meaning_conflicting_with_its_lexical_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = TypeChecker.check
    injected: list[Mapping[Path, Term] | None] = []

    def record(
        checker: TypeChecker,
        term: Term,
        signature: LexicalSignature,
        rules: Mapping[OpName, TypingRule],
        /,
        *,
        annotations: Mapping[Path, Term] | None = None,
        constraints: Mapping[Path, Term] | None = None,
    ) -> TypingResult:
        """Record the external constraints passed across the boundary."""
        injected.append(constraints)

        return original(
            checker,
            term,
            signature,
            rules,
            annotations=annotations,
            constraints=constraints,
        )

    monkeypatch.setattr(TypeChecker, "check", record)

    with pytest.raises(
        DeclarationError,
        match=r"meaning of 'john' at 'NP' does not type as the erasure",
    ):
        plan_training(composed("conflicting-lexical-meaning.yaml"))

    assert injected[-1] is not None
    assert set(injected[-1]) == {()}
