from typing import TYPE_CHECKING, Final, cast

import pytest
from tests.language import language

from lofi.lf import ArgumentRenaming

if TYPE_CHECKING:
    from lofi.terms import Term

LANGUAGE: Final = language(
    ("a", "b", "h0", "h1", "likes", "loves", "rain", "x", "y"),
    types={
        "a": "e",
        "b": "e",
        "h0": "e",
        "h1": "e",
        "likes": "e -> e -> t",
        "loves": "e -> e -> t",
        "rain": "t",
        "x": "e",
        "y": "e",
    },
)
STORE: Final = LANGUAGE.store


def _op(name: str, *arguments: Term) -> Term:
    return LANGUAGE.apply(LANGUAGE.constant(name), *arguments)


def _likes(subject: str, object_: str) -> Term:
    return LANGUAGE.apply(
        LANGUAGE.constant("likes"),
        LANGUAGE.constant(subject),
        LANGUAGE.constant(object_),
    )


def test_the_identity_renames_nothing_and_returns_the_very_node() -> None:
    form = _likes("h0", "h1")

    assert ArgumentRenaming.identity().apply(STORE, form) is form
    assert len(ArgumentRenaming.identity()) == 0
    assert ArgumentRenaming.identity() == ArgumentRenaming({})


def test_a_renaming_replaces_the_named_leaf_constants_and_interns_the_result() -> None:
    renaming = ArgumentRenaming({"h0": "a", "h1": "b"})

    assert renaming.apply(STORE, _likes("h0", "h1")) is _likes("a", "b")
    assert renaming.apply(STORE, _likes("h1", "h0")) is _likes("b", "a")
    assert len(renaming) == 2
    assert dict(renaming.correspondences) == {"h0": "a", "h1": "b"}


def test_two_hypothesis_constants_may_share_one_premise_constant() -> None:
    renaming = ArgumentRenaming({"h0": "a", "h1": "a"})

    assert renaming.apply(STORE, _likes("h0", "h1")) is _likes("a", "a")


def test_renaming_reaches_under_connectives_and_binders() -> None:
    renaming = ArgumentRenaming({"h0": "a"})
    body = _op("and", _op("not", _likes("h0", "h1")), LANGUAGE.constant("rain"))
    bound = LANGUAGE.apply(
        LANGUAGE.constant("exists"),
        STORE.bind(
            LANGUAGE.calculus.lam,
            "x",
            LANGUAGE.apply(
                LANGUAGE.constant("likes"),
                LANGUAGE.constant("h0"),
                STORE.fvar(LANGUAGE.term_sort, "x"),
            ),
        ),
    )
    expected = LANGUAGE.apply(
        LANGUAGE.constant("exists"),
        STORE.bind(
            LANGUAGE.calculus.lam,
            "x",
            LANGUAGE.apply(
                LANGUAGE.constant("likes"),
                LANGUAGE.constant("a"),
                STORE.fvar(LANGUAGE.term_sort, "x"),
            ),
        ),
    )

    assert renaming.apply(STORE, body) is _op(
        "and", _op("not", _likes("a", "h1")), LANGUAGE.constant("rain")
    )
    assert renaming.apply(STORE, bound) is expected


def test_a_form_naming_none_of_the_constants_is_returned_as_the_very_node() -> None:
    renaming = ArgumentRenaming({"h0": "a"})
    form = _op("not", _likes("x", "y"))

    assert renaming.apply(STORE, form) is form


def test_a_predicate_constant_is_renamed_like_any_other_leaf() -> None:
    renaming = ArgumentRenaming({"likes": "loves"})

    assert renaming.apply(STORE, _likes("a", "b")) is LANGUAGE.apply(
        LANGUAGE.constant("loves"),
        LANGUAGE.constant("a"),
        LANGUAGE.constant("b"),
    )


def test_renamings_compare_and_hash_by_their_correspondences() -> None:
    first = ArgumentRenaming({"h0": "a", "h1": "b"})
    second = ArgumentRenaming({"h1": "b", "h0": "a"})
    third = ArgumentRenaming({"h0": "b", "h1": "a"})

    assert first == second
    assert first != third
    assert first != "h0"
    assert len({first, second, third}) == 2
    assert repr(first) == "ArgumentRenaming({'h0': 'a', 'h1': 'b'})"


def test_a_renaming_is_immutable() -> None:
    renaming = ArgumentRenaming({"h0": "a"})

    with pytest.raises(AttributeError, match="immutable"):
        renaming.__setattr__("_correspondences", {})

    with pytest.raises(AttributeError, match="immutable"):
        renaming.__delattr__("_correspondences")

    with pytest.raises(TypeError):
        cast("dict[str, str]", renaming.correspondences)["h1"] = "b"


def test_every_name_is_a_nonempty_string() -> None:
    with pytest.raises(ValueError, match="hypothesis constant"):
        ArgumentRenaming({"": "a"})

    with pytest.raises(ValueError, match="premise constant"):
        ArgumentRenaming({"h0": ""})

    with pytest.raises(TypeError, match="premise constant"):
        ArgumentRenaming(cast("dict[str, str]", {"h0": 3}))
