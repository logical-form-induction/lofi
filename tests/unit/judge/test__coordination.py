from typing import TYPE_CHECKING, Final

import pytest

from lofi.judge import (
    Conjunctions,
)
from lofi.terms import (
    Operator,
    Signature,
    Term,
    TermStore,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


SIGNATURE: Final = Signature(
    "coordination",
    ("Cat",),
    (
        Operator("s", "Cat"),
        Operator("np", "Cat"),
        Operator("conj", "Cat"),
        Operator("comma", "Cat"),
        Operator("conj-mark", "Cat", arg_sorts=("Cat",)),
    ),
    features={"s": ("dcl", "b", "conj")},
)
STORE: Final = TermStore(SIGNATURE)
S: Final = STORE.op(SIGNATURE.by_name("s"))
S_DCL: Final = STORE.op(SIGNATURE.featured("s", "dcl"))
S_B: Final = STORE.op(SIGNATURE.featured("s", "b"))
NP: Final = STORE.op(SIGNATURE.by_name("np"))
CONJ: Final = STORE.op(SIGNATURE.by_name("conj"))
COMMA: Final = STORE.op(SIGNATURE.by_name("comma"))
MARK: Final = SIGNATURE.by_name("conj-mark")
MARKED_S: Final = STORE.op(MARK, S)
MARKED_DCL: Final = STORE.op(MARK, S_DCL)
MARKED_B: Final = STORE.op(MARK, S_B)
MARKED_NP: Final = STORE.op(MARK, NP)


CONJUNCTIONS: Final = Conjunctions(
    "ccg",
    (CONJ, COMMA),
    (
        (S_DCL, MARKED_DCL),
        (S_B, MARKED_B),
        (S, MARKED_S),
        (NP, MARKED_NP),
    ),
)


class TestConjunctions:
    """Finite-conjunction-view tests."""

    def test_the_view_copies_its_sequences(self) -> None:
        """Copy marker and category-pair sequences."""
        markers = [CONJ]
        pairs = [(S, MARKED_S)]
        view = Conjunctions("copy", markers, pairs)
        markers.append(COMMA)
        pairs.append((NP, MARKED_NP))

        assert view.name == "copy"
        assert view.markers == (CONJ,)
        assert view.pairs == ((S, MARKED_S),)

    @pytest.mark.parametrize(
        ("name", "markers", "pairs", "message"),
        [
            ("", (CONJ,), ((S, MARKED_S),), "name is empty"),
            ("empty-markers", (), ((S, MARKED_S),), "at least one marker"),
            ("empty-pairs", (CONJ,), (), "at least one category pair"),
        ],
    )
    def test_the_view_refuses_incomplete_declarations(
        self,
        name: str,
        markers: Sequence[Term],
        pairs: Sequence[tuple[Term, Term]],
        message: str,
    ) -> None:
        """Reject missing conjunction-view fields."""
        with pytest.raises(ValueError, match=message):
            Conjunctions(name, markers, pairs)

    @pytest.mark.parametrize("field", ["name", "markers", "pairs"])
    def test_the_view_is_frozen(self, field: str) -> None:
        """Reject field reassignment and deletion."""
        with pytest.raises(AttributeError, match="immutable"):
            setattr(CONJUNCTIONS, field, ())

        with pytest.raises(AttributeError, match="immutable"):
            delattr(CONJUNCTIONS, field)
