from copy import copy, deepcopy
from typing import cast

from lofi.grammars.tlg import ModalMatching
from lofi.terms import RootMatching, Signature, Term, TermStore, Vocabulary


def _term() -> Term:
    signature = Signature(
        "modal",
        ("Cat",),
        (),
        vocabularies=(Vocabulary("Cat", ("S",)),),
    )
    return TermStore(signature).const("Cat", "S")


def test_modal_matching_uses_exact_native_representatives() -> None:
    matching = ModalMatching(
        ("linear", "structural"),
        (("linear", "linear"), ("linear", "structural")),
    )
    term = _term()

    assert matching.has_key
    assert matching.key(term) is term
    assert matching.equivalent(term, term)
    assert matching.representatives.key(term) is term
    assert matching.name.startswith("modal-exact:")


def test_modal_matching_is_an_immutable_value() -> None:
    matching = ModalMatching(("linear",), (("linear", "linear"),))

    assert copy(matching) is matching
    assert deepcopy(matching) is matching
    assert matching == ModalMatching(("linear",), (("linear", "linear"),))
    assert hash(matching) == hash(ModalMatching(("linear",), (("linear", "linear"),)))


def test_modal_matching_identifies_a_root_only_with_itself() -> None:
    matching = ModalMatching(("a",), ())

    assert isinstance(matching, RootMatching)
    assert matching.identified_roots("\\[a]", cast("Signature", object())) == {"\\[a]"}
