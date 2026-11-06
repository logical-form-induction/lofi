from typing import Final

import pytest

from lofi.eval import Alignment, Contingency, Split
from lofi.eval._reconstruction import (
    constructor_set_recovery,
    lexical_type_recovery,
    primitive_set_recovery,
    type_system_reconstruction,
)
from lofi.terms import Operator, Signature, Term, TermStore, Vocabulary

PREDICTED_SIGNATURE: Final = Signature(
    "predicted-reconstruction",
    ("Ty",),
    (
        Operator("u0", "Ty"),
        Operator("u1", "Ty"),
        Operator("c0", "Ty", arg_sorts=("Ty", "Ty")),
        Operator("c1", "Ty", arg_sorts=("Ty", "Ty")),
    ),
)
GOLD_SIGNATURE: Final = Signature(
    "gold-reconstruction",
    ("Ty",),
    (
        Operator("g0", "Ty"),
        Operator("g1", "Ty"),
        Operator("/", "Ty", arg_sorts=("Ty", "Ty")),
        Operator("\\", "Ty", arg_sorts=("Ty", "Ty")),
    ),
)
PREDICTED_STORE: Final = TermStore(PREDICTED_SIGNATURE)
GOLD_STORE: Final = TermStore(GOLD_SIGNATURE)
U0: Final = PREDICTED_STORE.op(PREDICTED_SIGNATURE.by_name("u0"))
U1: Final = PREDICTED_STORE.op(PREDICTED_SIGNATURE.by_name("u1"))
G0: Final = GOLD_STORE.op(GOLD_SIGNATURE.by_name("g0"))
G1: Final = GOLD_STORE.op(GOLD_SIGNATURE.by_name("g1"))


def _table(
    induced: tuple[str, ...],
    gold: tuple[str, ...],
    counts: dict[tuple[str, str], int] | None = None,
    *,
    split: Split,
) -> Contingency:
    """Build a complete fixture contingency.

    Parameters
    ----------
    induced : tuple[str, ...]
        Induced labels.
    gold : tuple[str, ...]
        Gold labels.
    counts : dict[tuple[str, str], int] | None
        Nonzero counts to place in the table.
    split : Split
        Split assigned to the table.
    """
    complete = {(left, right): 0 for left in induced for right in gold}
    complete.update({} if counts is None else counts)

    return Contingency(complete, split=split)


def _primitive_alignment(mapping: dict[str, str] | None = None) -> Alignment:
    """Build the fixture primitive alignment.

    Parameters
    ----------
    mapping : dict[str, str] | None
        Alignment mapping, or None for the default permutation.
    """
    table = _table(
        ("u0", "u1"),
        ("g0", "g1"),
        {("u0", "g1"): 4, ("u1", "g0"): 3},
        split="dev",
    )

    return Alignment(
        {"u0": "g1", "u1": "g0"} if mapping is None else mapping,
        table,
    )


def _primitive_counts(
    counts: dict[tuple[str, str], int] | None = None,
) -> Contingency:
    """Build held-out primitive counts.

    Parameters
    ----------
    counts : dict[tuple[str, str], int] | None
        Nonzero counts, or None for the default table.
    """
    return _table(
        ("u0", "u1"),
        ("g0", "g1"),
        {("u0", "g1"): 6, ("u1", "g0"): 4} if counts is None else counts,
        split="test",
    )


def _constructor_alignment() -> Alignment:
    """Build the fixture constructor alignment."""
    return Alignment(
        {"c0": "/", "c1": "\\"},
        _table(
            ("c0", "c1"),
            ("/", "\\"),
            {("c0", "/"): 5, ("c1", "\\"): 5},
            split="dev",
        ),
    )


def _constructor_counts() -> Contingency:
    """Build held-out constructor counts."""
    return _table(
        ("c0", "c1"),
        ("/", "\\"),
        {("c0", "/"): 4, ("c1", "\\"): 6},
        split="test",
    )


def _constructor(store: TermStore, name: str, left: Term, right: Term) -> Term:
    """Build one binary fixture type.

    Parameters
    ----------
    store : TermStore
        Store owning the type.
    name : str
        Constructor name.
    left : Term
        Left argument.
    right : Term
        Right argument.
    """
    return store.op(store.signature.by_name(name), left, right)


def test_known_primitive_relabeling_has_perfect_inventory_recovery() -> None:
    recovered = primitive_set_recovery(
        _primitive_alignment(),
        _primitive_counts(),
    )

    assert recovered.induced_size == 2
    assert recovered.gold_size == 2
    assert recovered.exact_set_match
    assert dict(recovered.per_primitive_f1) == {"g0": 1.0, "g1": 1.0}


def test_zero_count_primitives_have_zero_occurrence_f1() -> None:
    recovered = primitive_set_recovery(
        _primitive_alignment({"u0": "g0", "u1": "g0"}),
        _primitive_counts(counts={}),
    )

    assert recovered.exact_set_match is False
    assert dict(recovered.per_primitive_f1) == {
        "g0": 0.0,
        "g1": 0.0,
    }


def test_exact_inventory_recovery_requires_a_bijection() -> None:
    fitted = _table(
        ("u0", "u1"),
        ("g0",),
        {("u0", "g0"): 1, ("u1", "g0"): 1},
        split="dev",
    )
    scored = _table(
        ("u0", "u1"),
        ("g0",),
        {("u0", "g0"): 1, ("u1", "g0"): 1},
        split="test",
    )

    recovered = primitive_set_recovery(
        Alignment({"u0": "g0", "u1": "g0"}, fitted),
        scored,
    )

    assert not recovered.exact_set_match


def test_exact_inventory_recovery_requires_the_scored_gold_inventory() -> None:
    fitted = _table(
        ("u0", "u1"),
        ("h0", "h1"),
        {("u0", "h0"): 1, ("u1", "h1"): 1},
        split="dev",
    )

    recovered = primitive_set_recovery(
        Alignment({"u0": "h0", "u1": "h1"}, fitted),
        _primitive_counts(),
    )

    assert not recovered.exact_set_match
    assert dict(recovered.per_primitive_f1) == {"g0": 0.0, "g1": 0.0}


def test_per_primitive_f1_uses_held_out_occurrence_counts() -> None:
    alignment = Alignment(
        {"u0": "g0", "u1": "g1"},
        _table(
            ("u0", "u1"),
            ("g0", "g1"),
            {("u0", "g0"): 50, ("u1", "g1"): 50},
            split="dev",
        ),
    )
    scored = _primitive_counts(
        {
            ("u0", "g0"): 3,
            ("u0", "g1"): 1,
            ("u1", "g0"): 2,
            ("u1", "g1"): 4,
        }
    )

    recovered = primitive_set_recovery(alignment, scored)

    assert dict(recovered.per_primitive_f1) == {
        "g0": pytest.approx(6 / 9),
        "g1": pytest.approx(8 / 11),
    }


def test_role_mutual_information_uses_declared_constructor_roles() -> None:
    observations = (
        ("c0", "result"),
        ("c0", "result"),
        ("c1", "argument"),
        ("c1", "argument"),
    )
    role_specific = constructor_set_recovery(
        _constructor_alignment(),
        _constructor_counts(),
        observations,
        roles=("result", "argument", "left-factor", "right-factor"),
    )
    undirected_alignment = Alignment(
        {"only": "/"},
        _table(
            ("only",),
            ("/", "\\"),
            {("only", "/"): 2, ("only", "\\"): 2},
            split="dev",
        ),
    )
    undirected_counts = _table(
        ("only",),
        ("/", "\\"),
        {("only", "/"): 2, ("only", "\\"): 2},
        split="test",
    )
    undirected = constructor_set_recovery(
        undirected_alignment,
        undirected_counts,
        (
            ("only", "result"),
            ("only", "result"),
            ("only", "argument"),
            ("only", "argument"),
        ),
        roles=("result", "argument"),
    )

    assert role_specific.exact_set_match
    assert role_specific.role_mutual_information == pytest.approx(0.69314718056)
    assert undirected.exact_set_match is False
    assert undirected.role_mutual_information == pytest.approx(0.0)


def test_role_mutual_information_uses_the_induced_choice() -> None:
    fitted = _table(
        ("c0", "c1"),
        ("/", "\\"),
        {("c0", "/"): 2, ("c1", "/"): 2},
        split="dev",
    )
    scored = _table(
        ("c0", "c1"),
        ("/", "\\"),
        {("c0", "/"): 2, ("c1", "/"): 2},
        split="test",
    )

    recovered = constructor_set_recovery(
        Alignment({"c0": "/", "c1": "/"}, fitted),
        scored,
        (
            ("c0", "result"),
            ("c0", "result"),
            ("c1", "argument"),
            ("c1", "argument"),
        ),
        roles=("result", "argument"),
    )

    assert not recovered.exact_set_match
    assert recovered.role_mutual_information == pytest.approx(0.69314718056)


def test_role_mutual_information_includes_unaligned_induced_choices() -> None:
    fitted = _table(
        ("c0", "c1"),
        ("/", "\\"),
        {("c0", "/"): 2, ("c1", "\\"): 2},
        split="dev",
    )

    recovered = constructor_set_recovery(
        Alignment({"c0": "/"}, fitted),
        _constructor_counts(),
        (
            ("c0", "result"),
            ("c0", "result"),
            ("c1", "argument"),
            ("c1", "argument"),
        ),
        roles=("result", "argument"),
    )

    assert not recovered.exact_set_match
    assert recovered.role_mutual_information == pytest.approx(0.69314718056)


def test_lexical_recovery_scores_ambiguous_words_and_token_occurrences() -> None:
    predicted = {
        "common": (U0, U1),
        "rare": (U0, U0),
    }
    gold = {
        "common": (G1, G0),
        "rare": (G1, G0),
    }
    recovered = lexical_type_recovery(
        predicted,
        gold,
        (U0, U1, U0, U0),
        (G1, G0, G1, G0),
        (PREDICTED_STORE, GOLD_STORE),
        primitive_alignment=_primitive_alignment(),
        constructor_alignment=_constructor_alignment(),
    )

    assert dict(recovered.per_word_f1) == {"common": 1.0, "rare": 0.5}
    assert recovered.type_f1 == 0.75
    assert recovered.token_accuracy == 0.75
    assert recovered.words == 2
    assert recovered.tokens == 4


def test_lexical_recovery_applies_a_primitive_permutation_to_constant_payloads() -> (
    None
):
    vocabulary = Vocabulary("Cat", ("NP", "S"))
    signature = Signature(
        "payload-reconstruction",
        ("Cat",),
        (),
        vocabularies=(vocabulary,),
    )
    store = TermStore(signature)
    constant = signature.by_name("#const[Cat]")
    noun = store.op(constant, payload=vocabulary.id("NP"))
    sentence = store.op(constant, payload=vocabulary.id("S"))
    alignment = Alignment(
        {"NP": "S", "S": "NP"},
        _table(
            ("NP", "S"),
            ("NP", "S"),
            {("NP", "S"): 2, ("S", "NP"): 2},
            split="dev",
        ),
    )

    recovered = lexical_type_recovery(
        {"left": (noun,), "right": (sentence,)},
        {"left": (sentence,), "right": (noun,)},
        (noun, sentence),
        (sentence, noun),
        (store, store),
        primitive_alignment=alignment,
        constructor_alignment=None,
    )

    assert recovered.type_f1 == 1.0
    assert recovered.token_accuracy == 1.0


def test_complete_reconstruction_is_perfect_under_known_relabeling() -> None:
    predicted_type = _constructor(PREDICTED_STORE, "c0", U0, U1)
    gold_type = _constructor(GOLD_STORE, "/", G1, G0)
    report = type_system_reconstruction(
        {"word": (predicted_type,)},
        {"word": (gold_type,)},
        (predicted_type,),
        (gold_type,),
        (PREDICTED_STORE, GOLD_STORE),
        primitive_alignment=_primitive_alignment(),
        primitive_contingency=_primitive_counts(),
        constructor_alignment=_constructor_alignment(),
        constructor_contingency=_constructor_counts(),
        role_pairs=(("c0", "result"), ("c1", "argument")),
        roles=("result", "argument"),
    )

    assert report.primitives.exact_set_match
    assert report.constructors.exact_set_match
    assert report.lexicon.type_f1 == 1.0
    assert report.lexicon.token_accuracy == 1.0


def test_lexical_recovery_requires_matching_vocabularies() -> None:
    with pytest.raises(ValueError, match="identical predicted and gold vocabularies"):
        lexical_type_recovery(
            {"word": (U0,)},
            {"other": (G1,)},
            (),
            (),
            (PREDICTED_STORE, GOLD_STORE),
            primitive_alignment=_primitive_alignment(),
            constructor_alignment=_constructor_alignment(),
        )


def test_lexical_recovery_counts_unpaired_token_occurrences_as_errors() -> None:
    recovered = lexical_type_recovery(
        {"word": (U0,)},
        {"word": (G1,)},
        (U0,),
        (),
        (PREDICTED_STORE, GOLD_STORE),
        primitive_alignment=_primitive_alignment(),
        constructor_alignment=_constructor_alignment(),
    )

    assert recovered.tokens == 1
    assert recovered.token_accuracy == 0.0


def test_inventory_recovery_refuses_the_alignment_fitting_split() -> None:
    alignment = _primitive_alignment()

    with pytest.raises(ValueError, match="cannot score the 'dev' fitting split"):
        primitive_set_recovery(alignment, alignment.contingency)


def test_complete_reconstruction_requires_one_scoring_split() -> None:
    predicted_type = _constructor(PREDICTED_STORE, "c0", U0, U1)
    gold_type = _constructor(GOLD_STORE, "/", G1, G0)
    constructor_counts = _table(
        ("c0", "c1"),
        ("/", "\\"),
        {("c0", "/"): 4, ("c1", "\\"): 6},
        split="train",
    )

    with pytest.raises(ValueError, match="must score the same split"):
        type_system_reconstruction(
            {"word": (predicted_type,)},
            {"word": (gold_type,)},
            (predicted_type,),
            (gold_type,),
            (PREDICTED_STORE, GOLD_STORE),
            primitive_alignment=_primitive_alignment(),
            primitive_contingency=_primitive_counts(),
            constructor_alignment=_constructor_alignment(),
            constructor_contingency=constructor_counts,
            role_pairs=(("c0", "result"), ("c1", "argument")),
            roles=("result", "argument"),
        )
