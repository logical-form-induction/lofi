from types import MappingProxyType
from typing import TYPE_CHECKING

import pytest
from tests.conformance.typing_system import (
    TypingSystemConformance,
    TypingSystemSuite,
)

from lofi.terms import Operator, Path, Signature, TermStore
from lofi.typecheck import Scheme, Substitution, TypingResult, TypingRule, TypingSystem

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi.terms import OpName, Term


class SurfaceTypingSystem:
    """Small implementation used to check the protocol shape."""

    def __init__(self, types: TermStore) -> None:
        self._types = types

    def check(
        self,
        term: Term,
        signature: str,
        rules: Mapping[OpName, TypingRule],
        /,
        *,
        annotations: Mapping[Path, Term] | None = None,
        constraints: Mapping[Path, Term] | None = None,
    ) -> TypingResult:
        """Return a fixed judgment after observing each input."""
        del term, signature, rules, annotations, constraints
        truth = self._types.op(self._types.signature.by_name("t"))

        return TypingResult({(): truth}, (), Scheme(truth))

    def subsumes(
        self,
        general: Term,
        specific: Term,
        /,
    ) -> Substitution | None:
        """Bind one free atom to the truth type."""
        truth = self._types.op(self._types.signature.by_name("t"))

        if general.op.payload_sort != "free" or specific is not truth:
            return None

        atom = general.payload

        if atom is None:
            return None

        return MappingProxyType({atom: truth})

    def unifies(
        self,
        left: Term,
        right: Term,
        /,
    ) -> Substitution | None:
        """Bind either free atom to the truth type."""
        truth = self._types.op(self._types.signature.by_name("t"))

        if left is right:
            return MappingProxyType({})
        if left.op.payload_sort == "free" and right is truth:
            atom = left.payload
        elif right.op.payload_sort == "free" and left is truth:
            atom = right.payload
        else:
            return None

        return None if atom is None else MappingProxyType({atom: truth})


def test_a_structural_implementation_satisfies_the_typing_protocol() -> None:
    types = TermStore(
        Signature(
            "types",
            ("Ty",),
            (Operator("t", "Ty"),),
            open_sorts=("Ty",),
        )
    )
    logical_forms = TermStore(
        Signature("logical-forms", ("Lf",), (Operator("form", "Lf"),))
    )
    subject = SurfaceTypingSystem(types)
    logical_form = logical_forms.op(logical_forms.signature.by_name("form"))
    judgment = subject.check(logical_form, "lexicon", {})
    truth = types.op(types.signature.by_name("t"))

    assert isinstance(subject, TypingSystem)
    assert judgment.scheme == Scheme(truth)
    variable = types.fvar("Ty", "A")

    assert subject.subsumes(variable, truth) == {variable.payload: truth}
    assert subject.unifies(truth, variable) == {variable.payload: truth}


@pytest.fixture
def conformance() -> TypingSystemConformance[str]:
    """Return a conformance fixture for the structural implementation."""
    types = TermStore(
        Signature(
            "types",
            ("Ty",),
            (Operator("t", "Ty"),),
            open_sorts=("Ty",),
        )
    )
    logical_forms = TermStore(
        Signature(
            "logical-forms",
            ("Lf",),
            (Operator("form", "Lf"),),
            open_sorts=("Lf",),
        )
    )
    truth = types.op(types.signature.by_name("t"))
    variable = types.fvar("Ty", "A")
    unrelated = logical_forms.fvar("Lf", "other")
    logical_form = logical_forms.op(logical_forms.signature.by_name("form"))

    return TypingSystemConformance(
        SurfaceTypingSystem(types),
        logical_form,
        "lexicon",
        {},
        TypingResult({(): truth}, (), Scheme(truth)),
        variable,
        truth,
        unrelated,
    )


class TestSurfaceTypingSystem(TypingSystemSuite):
    """Typing-system contract for the structural fixture."""
