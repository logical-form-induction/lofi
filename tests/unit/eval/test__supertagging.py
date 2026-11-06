from typing import TYPE_CHECKING, Final, cast

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from tests.support.term_reference import categories, infix_syntax

from lofi._types import ItemId
from lofi.eval._ports import Metric
from lofi.eval._supertagging import (
    SUPERTAG_ROWS,
    CategoryProbability,
    SupertagAccuracy,
    SupertagConfusionRow,
    SupertagLengthError,
    SupertagPrediction,
    TokenSupertags,
)
from lofi.eval.laws import MetricFixture, check_metric_laws
from lofi.terms import ForeignTermError, RegionStateError, Term, TermStore

if TYPE_CHECKING:
    from collections.abc import Sequence


_SIGNATURE: Final = categories()
_SYNTAX: Final = infix_syntax(_SIGNATURE, ("/", "fwd"), ("\\", "bwd"))


def _store() -> TermStore:
    return TermStore(_SIGNATURE)


def _atoms(store: TermStore) -> tuple[Term, ...]:
    return tuple(store.op(_SIGNATURE.by_name(name)) for name in ("np", "n", "s"))


def _token(
    *entries: tuple[Term, float],
) -> TokenSupertags:
    return TokenSupertags(
        tuple(
            CategoryProbability(category, probability)
            for category, probability in entries
        )
    )


def _prediction(
    tokens: Sequence[TokenSupertags],
    parser: Sequence[Term] | None,
) -> SupertagPrediction:
    return SupertagPrediction(tokens, parser)


def test_probabilities_are_finite_linear_values() -> None:
    store = _store()
    np, _, _ = _atoms(store)

    assert CategoryProbability(np, 1).probability == 1.0

    boolean = True
    not_a_probability = cast("float", boolean)

    with pytest.raises(TypeError, match="not a number"):
        CategoryProbability(np, not_a_probability)

    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="not finite"):
            CategoryProbability(np, bad)

    for bad in (-0.01, 1.01):
        with pytest.raises(ValueError, match="outside"):
            CategoryProbability(np, bad)


def test_token_distributions_are_ordered_normalized_and_structurally_unique() -> None:
    store = _store()
    np, n, _ = _atoms(store)
    entries = [CategoryProbability(np, 0.7), CategoryProbability(n, 0.3)]
    distribution = TokenSupertags(entries)
    entries.reverse()

    assert tuple(entry.category for entry in distribution.candidates) == (np, n)

    with pytest.raises(ValueError, match="no candidates"):
        TokenSupertags(())

    with pytest.raises(ValueError, match="sums"):
        _token((np, 0.6), (n, 0.3))

    foreign = _store()
    foreign_np, _, _ = _atoms(foreign)

    with pytest.raises(ValueError, match="repeats a structural category"):
        _token((np, 0.5), (foreign_np, 0.5))

    with pytest.raises(TypeError, match="not a category probability"):
        TokenSupertags(cast("Sequence[CategoryProbability]", ("np",)))


def test_public_input_values_are_immutable_and_hashable() -> None:
    store = _store()
    np, n, _ = _atoms(store)
    probability = CategoryProbability(np, 0.6)
    distribution = _token((np, 0.6), (n, 0.4))
    prediction = _prediction((distribution,), (np,))
    row = SupertagConfusionRow("tagger", "np", "n", 2)

    assert probability == CategoryProbability(np, 0.6)
    assert hash(probability) == hash(CategoryProbability(np, 0.6))
    assert distribution == _token((np, 0.6), (n, 0.4))
    assert prediction == _prediction((distribution,), (np,))
    assert row == SupertagConfusionRow("tagger", "np", "n", 2)
    assert len({probability, distribution, prediction, row}) == 4
    assert probability != "np"
    assert distribution != "np"
    assert prediction != "np"
    assert row != "np"

    for value, field in (
        (probability, "_probability"),
        (distribution, "_candidates"),
        (prediction, "_tokens"),
        (row, "_count"),
    ):
        with pytest.raises(AttributeError, match="immutable"):
            setattr(value, field, 0)

        with pytest.raises(AttributeError, match="immutable"):
            delattr(value, field)


def test_the_two_accuracy_variants_diverge_on_the_hand_fixture() -> None:
    store = _store()
    np, n, s = _atoms(store)
    first = _prediction(
        (_token((np, 0.9), (n, 0.1)), _token((n, 0.7), (s, 0.3))),
        (np, s),
    )
    second = _prediction(
        (_token((s, 0.6), (np, 0.4)), _token((np, 0.8), (n, 0.2))),
        (s, np),
    )
    metric = SupertagAccuracy(store, _SYNTAX, beta=0.5)
    metric.update(first, (np, n), key=ItemId("first"))
    metric.update(second, (s, np), key=ItemId("second"))
    rows = metric.rows()

    assert tuple(rows) == SUPERTAG_ROWS
    assert rows["tagger-accuracy"].value == 1.0
    assert rows["parser-category-accuracy"].value == 0.75
    assert rows["coverage"].value == 1.0
    assert rows["multi-tag-accuracy"].value == 1.0
    assert rows["ambiguity"].value == 1.25
    assert rows["tagger-accuracy"].n == 4
    assert rows["coverage"].n == 2
    assert metric.compute() == rows["tagger-accuracy"]


def test_dropping_a_parse_only_lowers_parser_accuracy_and_coverage() -> None:
    store = _store()
    np, n, s = _atoms(store)
    first = _prediction(
        (_token((np, 0.9), (n, 0.1)), _token((n, 0.7), (s, 0.3))),
        (np, s),
    )
    second_tokens = (
        _token((s, 0.6), (np, 0.4)),
        _token((np, 0.8), (n, 0.2)),
    )
    complete = SupertagAccuracy(store, _SYNTAX, beta=0.5)
    dropped = SupertagAccuracy(store, _SYNTAX, beta=0.5)

    for metric in (complete, dropped):
        metric.update(first, (np, n), key=ItemId("first"))

    complete.update(_prediction(second_tokens, (s, np)), (s, np), key=ItemId("second"))
    dropped.update(_prediction(second_tokens, None), (s, np), key=ItemId("second"))
    full_rows = complete.rows()
    dropped_rows = dropped.rows()

    assert dropped_rows["tagger-accuracy"] == full_rows["tagger-accuracy"]
    assert dropped_rows["multi-tag-accuracy"] == full_rows["multi-tag-accuracy"]
    assert dropped_rows["ambiguity"] == full_rows["ambiguity"]
    assert dropped_rows["parser-category-accuracy"].value == 0.25
    assert dropped_rows["parser-category-accuracy"].n == 4
    assert dropped_rows["coverage"].value == 0.5


def test_an_unparsed_sentence_counts_every_parser_token_as_wrong() -> None:
    store = _store()
    np, n, _ = _atoms(store)
    metric = SupertagAccuracy(store, _SYNTAX, beta=0.5)
    prediction = _prediction(
        (
            _token((np, 0.8), (n, 0.2)),
            _token((n, 0.8), (np, 0.2)),
            _token((np, 0.7), (n, 0.3)),
        ),
        None,
    )
    metric.update(prediction, (np, n, np), key=ItemId("unparsed"))
    parser = metric.rows()["parser-category-accuracy"]
    parser_confusions = tuple(
        row for row in metric.confusion() if row.variant == "parser"
    )

    assert parser.value == 0.0
    assert parser.n == 3
    assert parser.components["unparsed_sentences"] == 1.0
    assert sum(row.count for row in parser_confusions) == 3
    assert all(row.predicted is None for row in parser_confusions)


def test_the_c_and_c_threshold_is_strict_at_the_boundary() -> None:
    store = _store()
    np, n, s = _atoms(store)
    metric = SupertagAccuracy(store, _SYNTAX, beta=0.5)
    metric.update(
        _prediction((_token((np, 0.6), (n, 0.3), (s, 0.1)),), (np,)),
        (n,),
        key=ItemId("boundary"),
    )
    rows = metric.rows()

    assert rows["tagger-accuracy"].value == 0.0
    assert rows["multi-tag-accuracy"].value == 0.0
    assert rows["ambiguity"].value == 1.0


def test_declared_order_breaks_probability_ties_stably() -> None:
    store = _store()
    np, n, _ = _atoms(store)
    first = SupertagAccuracy(store, _SYNTAX, beta=0.5)
    second = SupertagAccuracy(store, _SYNTAX, beta=0.5)
    first.update(
        _prediction((_token((np, 0.5), (n, 0.5)),), (n,)),
        (n,),
        key=ItemId("tie"),
    )
    second.update(
        _prediction((_token((n, 0.5), (np, 0.5)),), (n,)),
        (n,),
        key=ItemId("tie"),
    )

    assert first.compute().value == 0.0
    assert second.compute().value == 1.0
    assert first.rows()["multi-tag-accuracy"].value == 1.0
    assert second.rows()["multi-tag-accuracy"].value == 1.0


def test_length_mismatches_name_the_item_and_field() -> None:
    store = _store()
    np, n, _ = _atoms(store)
    metric = SupertagAccuracy(store, _SYNTAX, beta=0.5)

    with pytest.raises(SupertagLengthError) as tagger_error:
        metric.update(
            _prediction((_token((np, 1.0)),), None),
            (np, n),
            key=ItemId("short-tagger"),
        )

    assert tagger_error.value.item_id == "short-tagger"
    assert tagger_error.value.field == "tagger"
    assert tagger_error.value.expected == 2
    assert tagger_error.value.received == 1

    with pytest.raises(SupertagLengthError) as parser_error:
        metric.update(
            _prediction((_token((np, 1.0)), _token((n, 1.0))), (np,)),
            (np, n),
            key=ItemId("short-parser"),
        )

    assert parser_error.value.item_id == "short-parser"
    assert parser_error.value.field == "parser"


def test_confusions_render_through_the_injected_ccg_syntax() -> None:
    store = _store()
    np, n, s = _atoms(store)
    functor = store.op(_SIGNATURE.by_name("fwd"), s, np)
    syntax = infix_syntax(_SIGNATURE, (">", "fwd"), ("<", "bwd"))
    metric = SupertagAccuracy(store, syntax, beta=0.5)
    metric.update(
        _prediction((_token((n, 0.6), (functor, 0.4)),), (functor,)),
        (functor,),
        key=ItemId("syntax"),
    )
    rows = metric.confusion()

    assert tuple((row.variant, row.gold, row.predicted, row.count) for row in rows) == (
        ("tagger", "s>np", "n", 1),
        ("parser", "s>np", "s>np", 1),
    )


def test_structural_equality_spans_distinct_terms_in_one_lineage() -> None:
    root = _store()
    np, _, s = _atoms(root)

    with root.region("sentence", max_nodes=20) as child:
        local = child.op(_SIGNATURE.by_name("fwd"), s, np)
        promoted = child.promote(local)
        metric = SupertagAccuracy(child, _SYNTAX, beta=0.5)

        assert local is not promoted
        assert local.key() == promoted.key()
        metric.update(
            _prediction((_token((promoted, 1.0)),), (promoted,)),
            (local,),
            key=ItemId("lineage"),
        )
        assert metric.compute().value == 1.0
        assert metric.rows()["parser-category-accuracy"].value == 1.0


def test_foreign_terms_and_released_stores_are_refused() -> None:
    root = _store()
    foreign = _store()
    np, _, _ = _atoms(root)
    foreign_np, _, _ = _atoms(foreign)
    metric = SupertagAccuracy(root, _SYNTAX, beta=0.5)

    with pytest.raises(ForeignTermError):
        metric.update(
            _prediction((_token((np, 1.0)),), (np,)),
            (foreign_np,),
            key=ItemId("foreign"),
        )

    def released_metric() -> SupertagAccuracy:
        held: SupertagAccuracy | None = None

        with root.region("released", max_nodes=10) as child:
            held = SupertagAccuracy(child, _SYNTAX, beta=0.5)

        if held is None:
            msg = "the term-store region did not open"
            raise AssertionError(msg)

        return held

    child_metric = released_metric()
    with pytest.raises(RegionStateError):
        child_metric.update(
            _prediction((_token((np, 1.0)),), None),
            (np,),
            key=ItemId("released"),
        )


def test_constructor_and_update_boundaries_reject_malformed_values() -> None:
    store = _store()
    np, _, _ = _atoms(store)
    other_signature = categories()

    with pytest.raises(ValueError, match="cannot score"):
        SupertagAccuracy(
            store,
            infix_syntax(other_signature, ("/", "fwd"), ("\\", "bwd")),
            beta=0.5,
        )

    for bad in (-0.1, 1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match=r"outside|not finite"):
            SupertagAccuracy(store, _SYNTAX, beta=bad)

    with pytest.raises(TypeError, match="not one"):
        SupertagAccuracy(store, _SYNTAX, beta=0.5).update(
            cast("SupertagPrediction", "prediction"),
            (np,),
            key=ItemId("malformed"),
        )

    with pytest.raises(TypeError, match="not a term"):
        SupertagPrediction((), cast("Sequence[Term]", ("np",)))


def test_rows_and_confusions_are_immutable_and_stably_ordered() -> None:
    store = _store()
    np, n, _ = _atoms(store)
    metric = SupertagAccuracy(store, _SYNTAX, beta=0.5)
    metric.update(
        _prediction((_token((n, 0.6), (np, 0.4)),), None),
        (np,),
        key=ItemId("row"),
    )
    rows = metric.rows()

    assert tuple(rows) == SUPERTAG_ROWS
    assert metric.name == "supertag-accuracy"
    assert metric.conventions is None
    assert metric.order_invariant
    assert metric.beta == 0.5
    assert tuple(row.variant for row in metric.confusion()) == ("tagger", "parser")

    with pytest.raises(TypeError):
        cast("dict[str, int]", rows)["new-row"] = 1


_PROPERTY_STORE: Final = _store()
_PROPERTY_NP, _PROPERTY_N, _PROPERTY_S = _atoms(_PROPERTY_STORE)

type _Example = tuple[str, SupertagPrediction, tuple[Term, ...]]

_LAW_POOL: Final[tuple[_Example, ...]] = (
    (
        "first",
        _prediction(
            (_token((_PROPERTY_NP, 0.7), (_PROPERTY_N, 0.3)),), (_PROPERTY_NP,)
        ),
        (_PROPERTY_NP,),
    ),
    (
        "second",
        _prediction((_token((_PROPERTY_N, 0.6), (_PROPERTY_S, 0.4)),), None),
        (_PROPERTY_S,),
    ),
    (
        "third",
        _prediction((_token((_PROPERTY_S, 1.0)),), (_PROPERTY_N,)),
        (_PROPERTY_S,),
    ),
)


def _property_metric() -> SupertagAccuracy:
    return SupertagAccuracy(_PROPERTY_STORE, _SYNTAX, beta=0.5)


def test_the_metric_satisfies_the_metric_protocol_and_laws() -> None:
    metric: Metric[SupertagPrediction, tuple[Term, ...]] = _property_metric()

    assert isinstance(metric, Metric)
    check_metric_laws(
        MetricFixture(
            _property_metric,
            st.sampled_from(_LAW_POOL),
            max_examples=200,
        )
    )


_TOKEN_SPEC = st.tuples(st.booleans(), st.booleans(), st.booleans(), st.booleans())
_SENTENCE_SPEC = st.tuples(st.lists(_TOKEN_SPEC, min_size=1, max_size=5), st.booleans())


@settings(max_examples=200, deadline=None)
@given(st.lists(_SENTENCE_SPEC, min_size=1, max_size=6))
def test_all_rows_use_their_fixed_corpus_micro_denominators(
    sentences: list[tuple[list[tuple[bool, bool, bool, bool]], bool]],
) -> None:
    metric = _property_metric()
    tagger_correct = 0
    parser_correct = 0
    multi_correct = 0
    survivors = 0
    tokens = 0
    parsed = 0

    for sentence_index, (specs, was_parsed) in enumerate(sentences):
        gold: list[Term] = []
        distributions: list[TokenSupertags] = []
        parser: list[Term] = []

        for gold_is_np, best_is_np, broad, parser_is_np in specs:
            gold_category = _PROPERTY_NP if gold_is_np else _PROPERTY_N
            best = _PROPERTY_NP if best_is_np else _PROPERTY_N
            other = _PROPERTY_N if best_is_np else _PROPERTY_NP
            parser_category = _PROPERTY_NP if parser_is_np else _PROPERTY_N
            high, low = (0.6, 0.4) if broad else (0.8, 0.2)
            gold.append(gold_category)
            distributions.append(_token((best, high), (other, low)))
            parser.append(parser_category)
            tagger_hit = best_is_np == gold_is_np
            tagger_correct += tagger_hit
            parser_correct += was_parsed and parser_is_np == gold_is_np
            multi_correct += broad or tagger_hit
            survivors += 2 if broad else 1

        metric.update(
            _prediction(distributions, parser if was_parsed else None),
            gold,
            key=ItemId(f"sentence-{sentence_index}"),
        )
        tokens += len(specs)
        parsed += was_parsed

    rows = metric.rows()

    assert rows["tagger-accuracy"].value == tagger_correct / tokens
    assert rows["tagger-accuracy"].n == tokens
    assert rows["parser-category-accuracy"].value == parser_correct / tokens
    assert rows["parser-category-accuracy"].n == tokens
    assert rows["coverage"].value == parsed / len(sentences)
    assert rows["coverage"].n == len(sentences)
    assert rows["multi-tag-accuracy"].value == multi_correct / tokens
    assert rows["multi-tag-accuracy"].n == tokens
    assert rows["ambiguity"].value == survivors / tokens
    assert rows["ambiguity"].n == tokens


@settings(max_examples=200, deadline=None)
@given(
    st.tuples(
        st.integers(min_value=1, max_value=100),
        st.integers(min_value=1, max_value=100),
        st.integers(min_value=1, max_value=100),
    ),
    st.integers(min_value=0, max_value=2),
    st.floats(min_value=0.0, max_value=0.99, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.0, max_value=0.99, allow_nan=False, allow_infinity=False),
)
def test_raising_beta_only_removes_survivors_and_gold_membership(
    weights: tuple[int, int, int],
    gold_index: int,
    first_beta: float,
    second_beta: float,
) -> None:
    total = sum(weights)
    terms = (_PROPERTY_NP, _PROPERTY_N, _PROPERTY_S)
    distribution = _token(
        *((term, weight / total) for term, weight in zip(terms, weights, strict=True))
    )
    prediction = _prediction((distribution,), None)
    low = SupertagAccuracy(_PROPERTY_STORE, _SYNTAX, beta=min(first_beta, second_beta))
    high = SupertagAccuracy(_PROPERTY_STORE, _SYNTAX, beta=max(first_beta, second_beta))

    for metric in (low, high):
        metric.update(prediction, (terms[gold_index],), key=ItemId("beta"))

    low_rows = low.rows()
    high_rows = high.rows()
    low_ambiguity = low_rows["ambiguity"].value
    high_ambiguity = high_rows["ambiguity"].value
    low_multi = low_rows["multi-tag-accuracy"].value
    high_multi = high_rows["multi-tag-accuracy"].value

    assert low_ambiguity is not None
    assert high_ambiguity is not None
    assert low_multi is not None
    assert high_multi is not None
    assert high_ambiguity <= low_ambiguity
    assert high_multi <= low_multi
