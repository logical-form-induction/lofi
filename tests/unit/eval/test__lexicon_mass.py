from typing import Final, cast

import pytest
from tests.support.term_reference import categories, infix_syntax

from lofi.eval._lexicon_mass import (
    LexiconMassRow,
    SupertagConfusionTable,
    lexicon_mass_report,
    supertag_confusion_table,
)
from lofi.eval._supertagging import CategoryProbability, TokenSupertags
from lofi.terms import ForeignTermError, Term, TermStore, TermSyntax

_SIGNATURE: Final = categories()
_SYNTAX: Final = infix_syntax(_SIGNATURE, ("/", "fwd"), ("\\", "bwd"))

type _Observation = tuple[str, TokenSupertags, Term]


def _store() -> TermStore:
    return TermStore(_SIGNATURE)


def _atoms(store: TermStore) -> tuple[Term, ...]:
    return tuple(store.op(_SIGNATURE.by_name(name)) for name in ("np", "n", "s"))


def _token(*entries: tuple[Term, float]) -> TokenSupertags:
    return TokenSupertags(
        tuple(CategoryProbability(category, mass) for category, mass in entries)
    )


def test_lexicon_mass_report_matches_independent_hand_arithmetic() -> None:
    store = _store()
    np, n, s = _atoms(store)
    np_over_n = store.op(_SIGNATURE.by_name("fwd"), np, n)
    observations: tuple[_Observation, ...] = (
        ("noun", _token((np, 0.5), (n, 0.3), (np_over_n, 0.2)), np),
        ("noun", _token((np, 0.2), (n, 0.3), (np_over_n, 0.5)), np_over_n),
        ("verb", _token((n, 0.25), (s, 0.75)), n),
    )

    report = lexicon_mass_report(store, _SYNTAX, observations, k=2)

    assert report == (
        LexiconMassRow("noun", (("np", 0.35), ("np/n", 0.35)), 0.7),
        LexiconMassRow("verb", (("s", 0.75), ("n", 0.25)), 0.25),
    )
    assert lexicon_mass_report(store, _SYNTAX, reversed(observations), k=2) == report


def test_gold_point_mass_and_unattested_mass_separate() -> None:
    store = _store()
    np, n, _ = _atoms(store)

    point = lexicon_mass_report(
        store,
        _SYNTAX,
        (("noun", _token((np, 1.0)), np),),
        k=1,
    )
    spread = lexicon_mass_report(
        store,
        _SYNTAX,
        (("noun", _token((np, 0.75), (n, 0.25)), np),),
        k=2,
    )

    assert point[0].gold_attested_mass == 1.0
    assert spread[0].gold_attested_mass == 0.75
    assert spread[0].gold_attested_mass < 1.0


def test_top_mass_ties_use_canonical_text() -> None:
    store = _store()
    np, n, _ = _atoms(store)

    row = lexicon_mass_report(
        store,
        _SYNTAX,
        (("tie", _token((np, 0.5), (n, 0.5)), np),),
        k=2,
    )[0]

    assert row.top_categories == (("n", 0.5), ("np", 0.5))


def test_mass_aggregation_uses_structural_keys_across_distinct_nodes() -> None:
    parent = _store()

    with parent.region("report", max_nodes=20) as child:
        child_np = child.op(_SIGNATURE.by_name("np"))
        parent_np = parent.op(_SIGNATURE.by_name("np"))

        assert child_np is not parent_np
        assert child_np.key() == parent_np.key()

        row = lexicon_mass_report(
            child,
            _SYNTAX,
            (
                ("noun", _token((child_np, 1.0)), child_np),
                ("noun", _token((parent_np, 1.0)), parent_np),
            ),
            k=2,
        )[0]

        assert row.top_categories == (("np", 1.0),)
        assert row.gold_attested_mass == 1.0


def test_confusion_axes_use_depth_then_canonical_text() -> None:
    store = _store()
    np, n, s = _atoms(store)
    fwd = _SIGNATURE.by_name("fwd")
    bwd = _SIGNATURE.by_name("bwd")
    s_over_np = store.op(fwd, s, np)
    s_under_np = store.op(bwd, s, np)
    depth_three = store.op(fwd, s_over_np, n)
    observations: tuple[_Observation, ...] = (
        ("atom", _token((n, 0.6), (np, 0.3), (s, 0.1)), np),
        ("functor", _token((s_under_np, 0.7), (s_over_np, 0.3)), s_over_np),
        ("nested", _token((depth_three, 1.0)), depth_three),
    )

    table = supertag_confusion_table(store, _SYNTAX, observations)

    assert table.categories == ("n", "np", "s/np", "s\\np", "s/np/n")
    assert table.counts == (
        (0, 0, 0, 0, 0),
        (1, 0, 0, 0, 0),
        (0, 0, 0, 1, 0),
        (0, 0, 0, 0, 0),
        (0, 0, 0, 0, 1),
    )


def test_confusion_argmax_preserves_candidate_declaration_order() -> None:
    store = _store()
    np, n, _ = _atoms(store)
    first: tuple[_Observation, ...] = (("tie", _token((np, 0.5), (n, 0.5)), n),)
    second: tuple[_Observation, ...] = (("tie", _token((n, 0.5), (np, 0.5)), n),)

    first_table = supertag_confusion_table(store, _SYNTAX, first)
    second_table = supertag_confusion_table(store, _SYNTAX, second)

    assert first_table.categories == ("n", "np")
    assert first_table.counts == ((0, 1), (0, 0))
    assert second_table.categories == ("n",)
    assert second_table.counts == ((1,),)


def test_empty_observations_produce_empty_values() -> None:
    store = _store()

    assert lexicon_mass_report(store, _SYNTAX, (), k=2) == ()
    assert supertag_confusion_table(store, _SYNTAX, ()) == SupertagConfusionTable(
        (), ()
    )


def test_inputs_require_positive_k_names_and_one_shared_lineage() -> None:
    store = _store()
    np, _, _ = _atoms(store)

    for bad in (0, -1):
        with pytest.raises(ValueError, match=r"positive|negative"):
            lexicon_mass_report(store, _SYNTAX, (), k=bad)

    boolean = True

    with pytest.raises(TypeError, match="integer"):
        lexicon_mass_report(store, _SYNTAX, (), k=cast("int", boolean))

    with pytest.raises(ValueError, match=r"word class.*empty"):
        lexicon_mass_report(
            store,
            _SYNTAX,
            (("", _token((np, 1.0)), np),),
            k=1,
        )

    foreign = _store()
    foreign_np, _, _ = _atoms(foreign)

    with pytest.raises(ForeignTermError):
        lexicon_mass_report(
            store,
            _SYNTAX,
            (("noun", _token((foreign_np, 1.0)), np),),
            k=1,
        )

    with pytest.raises(ForeignTermError):
        lexicon_mass_report(
            store,
            _SYNTAX,
            (("noun", _token((np, 1.0)), foreign_np),),
            k=1,
        )

    with store.region("report", max_nodes=20) as child:
        child_n = child.op(_SIGNATURE.by_name("n"))
        row = lexicon_mass_report(
            child,
            _SYNTAX,
            (("noun", _token((np, 0.5), (child_n, 0.5)), np),),
            k=2,
        )[0]

        assert row.top_categories == (("n", 0.5), ("np", 0.5))


def test_inputs_require_a_store_syntax_pair_and_typed_observations() -> None:
    store = _store()
    np, _, _ = _atoms(store)
    observations: tuple[_Observation, ...] = (("noun", _token((np, 1.0)), np),)
    other_signature = categories()
    other_syntax = infix_syntax(
        other_signature,
        ("/", "fwd"),
        ("\\", "bwd"),
    )

    with pytest.raises(TypeError, match="not a TermStore"):
        lexicon_mass_report(
            cast("TermStore", "store"),
            _SYNTAX,
            observations,
            k=1,
        )

    with pytest.raises(TypeError, match="does not implement TermSyntax"):
        supertag_confusion_table(
            store,
            cast("TermSyntax", "syntax"),
            observations,
        )

    with pytest.raises(ValueError, match="cannot report terms"):
        lexicon_mass_report(store, other_syntax, observations, k=1)

    wrong_posterior = cast("_Observation", ("noun", "posterior", np))

    with pytest.raises(TypeError, match="posterior is not TokenSupertags"):
        lexicon_mass_report(store, _SYNTAX, (wrong_posterior,), k=1)

    wrong_gold = cast("_Observation", ("noun", _token((np, 1.0)), "gold"))

    with pytest.raises(TypeError, match="gold category is not a term"):
        supertag_confusion_table(store, _SYNTAX, (wrong_gold,))


def test_return_values_are_frozen_hashable_and_validated() -> None:
    row = LexiconMassRow("noun", (("np", 1.0),), 1.0)
    table = SupertagConfusionTable(("n", "np"), ((1, 0), (0, 2)))

    assert hash(row) == hash(LexiconMassRow("noun", (("np", 1.0),), 1.0))
    assert hash(table) == hash(SupertagConfusionTable(("n", "np"), ((1, 0), (0, 2))))
    assert row != "noun"
    assert table != "table"

    row_field = "_word_class"

    with pytest.raises(AttributeError, match="immutable"):
        setattr(row, row_field, "verb")

    with pytest.raises(AttributeError, match="immutable"):
        delattr(table, "_counts")

    with pytest.raises(ValueError, match="repeats"):
        LexiconMassRow("noun", (("np", 0.5), ("np", 0.5)), 1.0)

    with pytest.raises(ValueError, match="not a 2 by 2 matrix"):
        SupertagConfusionTable(("n", "np"), ((1, 0),))

    with pytest.raises(ValueError, match="outside"):
        LexiconMassRow("noun", (("np", 1.1),), 1.0)
