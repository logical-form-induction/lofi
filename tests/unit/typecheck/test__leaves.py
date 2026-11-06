"""Interior values for declared leaf constants."""

from copy import copy, deepcopy

import pytest

from lofi.typecheck import LeafConstant, LeafLexicon


def test_a_leaf_lexicon_is_an_ordered_frozen_constant_declaration() -> None:
    lexicon = LeafLexicon(
        (
            LeafConstant("jo", "e", ("individual",)),
            LeafConstant("sleep", "e -> t", ("predicate",)),
            LeafConstant("think", "infer", ("predicate",)),
        )
    )

    assert tuple(entry.name for entry in lexicon) == ("jo", "sleep", "think")
    assert lexicon[1].type == "e -> t"
    assert lexicon[2].type == "infer"
    assert copy(lexicon) is lexicon
    assert deepcopy(lexicon) is lexicon

    with pytest.raises(AttributeError, match="immutable"):
        lexicon.__setattr__("_entries", ())


def test_leaf_constants_require_unique_names_and_nonempty_declarations() -> None:
    with pytest.raises(ValueError, match="leaf constant needs a name"):
        LeafConstant("", "e", ())

    with pytest.raises(ValueError, match="needs a type or 'infer'"):
        LeafConstant("jo", "", ())

    with pytest.raises(ValueError, match="class needs a name"):
        LeafConstant("jo", "e", ("",))

    with pytest.raises(ValueError, match="declared twice"):
        LeafLexicon((LeafConstant("jo", "e", ()), LeafConstant("jo", "t", ())))
