from typing import Final

import pytest

from lofi.grammars.ccg._values import marked_refinement
from lofi.judge import Conjunctions
from lofi.terms import (
    FeatureErasure,
    Identity,
    Operator,
    RepresentativeKernel,
    Signature,
    SignatureStableMatching,
    Term,
    TermStore,
)

SIGNATURE: Final = Signature(
    "matching-refinement",
    ("Cat",),
    (
        Operator("s", "Cat"),
        Operator("s-dcl", "Cat", erases_to="s"),
        Operator("n", "Cat"),
        Operator("np", "Cat"),
        Operator("conj", "Cat"),
        Operator("s-conj", "Cat", erases_to="s"),
        Operator("s-dcl-conj", "Cat", erases_to="s"),
    ),
)
STORE: Final = TermStore(SIGNATURE)
S: Final = STORE.op(SIGNATURE.by_name("s"))
S_DCL: Final = STORE.op(SIGNATURE.by_name("s-dcl"))
N: Final = STORE.op(SIGNATURE.by_name("n"))
NP: Final = STORE.op(SIGNATURE.by_name("np"))
CONJ: Final = STORE.op(SIGNATURE.by_name("conj"))
S_CONJ: Final = STORE.op(SIGNATURE.by_name("s-conj"))
S_DCL_CONJ: Final = STORE.op(SIGNATURE.by_name("s-dcl-conj"))
TERMS: Final = (S, S_DCL, N, NP, CONJ, S_CONJ, S_DCL_CONJ)
CONJUNCTIONS: Final = Conjunctions(
    "matching-refinement",
    (CONJ,),
    ((S, S_CONJ), (S_DCL, S_DCL_CONJ)),
)


def test_marks_are_kept_apart_only_where_the_base_would_equate_them() -> None:
    identity = Identity[Term]()
    erased = marked_refinement(FeatureErasure(), CONJUNCTIONS)

    assert marked_refinement(identity, CONJUNCTIONS) is identity
    assert marked_refinement(identity, None) is identity
    assert erased.name.startswith("ccg-feature-erasure:")
    assert isinstance(erased, SignatureStableMatching)
    assert erased.signature_extension_stable
    assert erased.equivalent(S, S_DCL)
    assert erased.equivalent(S_CONJ, S_DCL_CONJ)
    assert not erased.equivalent(S, S_CONJ)
    assert not erased.equivalent(S_CONJ, S)
    assert erased.key(S_CONJ) is erased.key(S_DCL_CONJ)


class _Collapse:
    """A congruence that equates every category and keys on one term."""

    __slots__ = ("_has_key", "_representative")

    def __init__(self, representative: Term, *, has_key: bool) -> None:
        self._representative = representative
        self._has_key = has_key

    @property
    def name(self) -> str:
        """Return the relation name."""
        return "collapse"

    @property
    def has_key(self) -> bool:
        """Return whether representative keys are available."""
        return self._has_key

    @property
    def representatives(self) -> RepresentativeKernel:
        """Compile the fixture's finite universal relation."""
        return RepresentativeKernel.identity().refine(
            tuple((term, self._representative) for term in TERMS)
        )

    def key(self, term: Term, /) -> Term:
        """Return the one representative."""
        del term
        return self._representative

    def equivalent(self, left: Term, right: Term, /) -> bool:
        """Return that every two categories are equivalent."""
        del left, right
        return True


@pytest.mark.parametrize("has_key", [True, False])
def test_a_refinement_keeps_its_base_keys_outside_the_marks(*, has_key: bool) -> None:
    refined = marked_refinement(_Collapse(NP, has_key=has_key), CONJUNCTIONS)

    assert refined.name.startswith("ccg-collapse:")
    assert isinstance(refined, SignatureStableMatching)
    assert not refined.signature_extension_stable
    assert refined.has_key is has_key
    assert refined.key(S) is NP
    assert refined.key(S_DCL) is NP
    assert refined.equivalent(S, N)
    assert not refined.equivalent(S, S_CONJ)
    assert refined.equivalent(S_CONJ, S_DCL_CONJ)
    assert refined.key(S_CONJ) is refined.key(S_DCL_CONJ)
