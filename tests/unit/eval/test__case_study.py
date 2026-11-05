import ast
from copy import copy, deepcopy
from pathlib import Path
from pickle import dumps, loads
from re import finditer
from typing import Final

import pytest
from tests.support.term_reference import categories

import lofi.eval._case_study as case_study
from lofi._types import ItemId
from lofi.data import ByteSpan, Item, Token, TokenSpan
from lofi.eval._case_study import (
    COMPLEMENT_CONTRAST,
    FINITENESS_CONTRAST,
    MissingByteSpanError,
    MissingPosteriorError,
    MissingTextError,
    SpanContrast,
    SpanGroup,
    SpanGroupReport,
    SpanHit,
    TokenSurfaceMismatchError,
    induced_type_report,
    locate_span_groups,
)
from lofi.eval._supertagging import CategoryProbability, TokenSupertags
from lofi.terms import Term, TermStore

_SIGNATURE: Final = categories()


def _item(name: str, text: str) -> Item:
    """Build an item with UTF-8 byte spans."""
    tokens = tuple(
        Token(
            index,
            match.group(),
            ByteSpan(
                len(text[: match.start()].encode()),
                len(text[: match.end()].encode()),
            ),
        )
        for index, match in enumerate(finditer(r"\S+", text))
    )

    return Item(ItemId(name), tokens, text=text)


def _store() -> TermStore:
    """Build a store for test categories."""
    return TermStore(_SIGNATURE)


def _atoms(store: TermStore) -> tuple[Term, ...]:
    """Build the three atomic reference categories."""
    return tuple(store.op(_SIGNATURE.by_name(name)) for name in ("np", "n", "s"))


def _posterior(
    *entries: tuple[Term, float],
) -> TokenSupertags:
    """Build one test posterior."""
    return TokenSupertags(
        tuple(
            CategoryProbability(category, probability)
            for category, probability in entries
        )
    )


def _contrast(*substrings: str) -> SpanContrast:
    """Build a contrast with one matching group."""
    return SpanContrast(
        "test",
        (
            SpanGroup("target", substrings),
            SpanGroup("control", ("a control form absent from every fixture",)),
        ),
    )


def test_shipped_contrasts_declare_the_prespecified_surface_forms() -> None:
    assert COMPLEMENT_CONTRAST.name == ("declarative-versus-interrogative-complements")
    assert tuple(group.name for group in COMPLEMENT_CONTRAST.groups) == (
        "declarative",
        "interrogative",
    )
    assert tuple(group.substrings for group in COMPLEMENT_CONTRAST.groups) == (
        ("that something happened",),
        ("whether something happened", "which thing happened"),
    )
    assert FINITENESS_CONTRAST.name == "finite-versus-infinitival-complements"
    assert tuple(group.name for group in FINITENESS_CONTRAST.groups) == (
        "finite",
        "infinitival",
    )
    assert tuple(group.substrings for group in FINITENESS_CONTRAST.groups) == (
        (
            "that something happened",
            "whether something happened",
            "which thing happened",
        ),
        (
            "to do something",
            "whether to do something",
            "which thing to do",
            "for something to happen",
        ),
    )


def test_location_is_case_sensitive_and_rejects_embedded_token_substrings() -> None:
    item = _item("multibyte", "Niño the he niño")
    hits = locate_span_groups((item,), _contrast("he", "niño"))

    assert tuple(hit.substring for hit in hits) == ("he", "niño")
    assert tuple(hit.token_span for hit in hits) == (
        TokenSpan(2, 3),
        TokenSpan(3, 4),
    )
    assert tuple(hit.byte_span.substring(item.text or "") for hit in hits) == (
        "he",
        "niño",
    )
    assert hits[1].byte_span == ByteSpan(13, 18)


def test_location_preserves_declared_group_source_and_token_order() -> None:
    contrast = SpanContrast(
        "order",
        (SpanGroup("alpha", ("alpha",)), SpanGroup("beta", ("beta",))),
    )
    second = _item("second", "beta alpha alpha")
    first = _item("first", "alpha beta")
    hits = locate_span_groups((second, first), contrast)

    assert tuple((hit.group.name, hit.item_id, hit.token_span) for hit in hits) == (
        ("alpha", ItemId("second"), TokenSpan(1, 2)),
        ("alpha", ItemId("second"), TokenSpan(2, 3)),
        ("alpha", ItemId("first"), TokenSpan(0, 1)),
        ("beta", ItemId("second"), TokenSpan(0, 1)),
        ("beta", ItemId("first"), TokenSpan(1, 2)),
    )


def test_report_aggregates_every_occurrence_by_structural_term_key() -> None:
    item = _item(
        "first",
        "whether niño left and whether niño left",
    )
    contrast = _contrast("whether niño left")
    store = _store()
    np, n, s = _atoms(store)
    posteriors = {
        (ItemId("first"), TokenSpan(0, 3)): _posterior((np, 0.8), (n, 0.2)),
        (ItemId("first"), TokenSpan(4, 7)): _posterior((np, 0.2), (s, 0.8)),
    }
    target, control = induced_type_report((item,), contrast, posteriors)

    assert target.occurrence_count == 2
    assert target.sentence_count == 1
    assert tuple(hit.token_span for hit in target.hits) == (
        TokenSpan(0, 3),
        TokenSpan(4, 7),
    )
    assert tuple(key for key, _ in target.distribution) == (
        np.key(),
        n.key(),
        s.key(),
    )
    assert tuple(probability for _, probability in target.distribution) == (
        pytest.approx(0.5),
        pytest.approx(0.1),
        pytest.approx(0.4),
    )
    assert control.group == contrast.groups[1]
    assert control.hits == ()
    assert control.distribution == ()
    assert control.sentence_count == 0


def test_report_aggregates_equal_categories_from_distinct_stores() -> None:
    first_item = _item("one", "that happened")
    second_item = _item("two", "that happened")
    contrast = _contrast("that happened")
    first = _store()
    second = _store()
    first_np, _, _ = _atoms(first)
    second_np, second_n, _ = _atoms(second)
    report = induced_type_report(
        (first_item, second_item),
        contrast,
        {
            (ItemId("one"), TokenSpan(0, 2)): _posterior((first_np, 1.0)),
            (ItemId("two"), TokenSpan(0, 2)): _posterior(
                (second_np, 0.7), (second_n, 0.3)
            ),
        },
    )[0]

    assert report.distribution == ((first_np.key(), 0.85), (second_n.key(), 0.15))
    assert report.sentence_count == 2


def test_missing_inputs_raise_errors_that_name_the_offending_item_and_span() -> None:
    no_text = Item(
        ItemId("no-text"),
        (Token(0, "that", ByteSpan(0, 4)),),
    )
    no_span = Item(ItemId("no-span"), (Token(0, "that"),), text="that")
    mismatch = Item(
        ItemId("mismatch"),
        (Token(0, "other", ByteSpan(0, 4)),),
        text="that",
    )
    matched = _item("matched", "that")
    contrast = _contrast("that")

    with pytest.raises(MissingTextError) as text_error:
        locate_span_groups((no_text,), contrast)

    assert text_error.value.item_id == ItemId("no-text")

    with pytest.raises(MissingByteSpanError) as span_error:
        locate_span_groups((no_span,), contrast)

    assert span_error.value.item_id == ItemId("no-span")
    assert span_error.value.position == 0

    with pytest.raises(TokenSurfaceMismatchError) as surface_error:
        locate_span_groups((mismatch,), contrast)

    assert surface_error.value.item_id == ItemId("mismatch")
    assert surface_error.value.position == 0

    with pytest.raises(MissingPosteriorError) as posterior_error:
        induced_type_report((matched,), contrast, {})

    assert posterior_error.value.item_id == ItemId("matched")
    assert posterior_error.value.span == TokenSpan(0, 1)


def test_token_span_cannot_split_a_utf8_code_point() -> None:
    item = Item(
        ItemId("split-code-point"),
        (Token(0, "é", ByteSpan(0, 1)),),
        text="é",
    )

    with pytest.raises(TokenSurfaceMismatchError) as error:
        locate_span_groups((item,), _contrast("é"))

    assert error.value.item_id == ItemId("split-code-point")
    assert error.value.position == 0


def test_declarations_reject_duplicate_surface_forms() -> None:
    with pytest.raises(ValueError, match="repeats a substring"):
        SpanGroup("repeated", ("same", "same"))

    with pytest.raises(ValueError, match="multiple groups"):
        SpanContrast(
            "repeated",
            (SpanGroup("first", ("same",)), SpanGroup("second", ("same",))),
        )


def test_public_values_are_frozen_hashable_copyable_and_pickleable() -> None:
    substrings = ["that happened"]
    group = SpanGroup("target", substrings)
    control = SpanGroup("control", ("whether happened",))
    groups = [group, control]
    contrast = SpanContrast("test", groups)
    hit = SpanHit(
        group,
        ItemId("one"),
        "that happened",
        TokenSpan(0, 2),
        ByteSpan(0, 13),
    )
    store = _store()
    np, _, _ = _atoms(store)
    report = SpanGroupReport(group, (hit,), ((np.key(), 1.0),))

    substrings.append("later mutation")
    groups.reverse()

    assert group.substrings == ("that happened",)
    assert contrast.groups == (group, control)

    for value, field in (
        (group, "_name"),
        (contrast, "_name"),
        (hit, "_substring"),
        (report, "_distribution"),
    ):
        assert hash(value) == hash(copy(value))
        assert deepcopy(value) == value
        assert loads(dumps(value)) == value

        with pytest.raises(AttributeError, match="immutable"):
            setattr(value, field, "changed")

        with pytest.raises(AttributeError, match="immutable"):
            delattr(value, field)


def test_case_study_code_has_no_forbidden_search_or_analysis_dependencies() -> None:
    tree = ast.parse(Path(case_study.__file__).read_text(encoding="utf-8"))
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imports.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    identifiers = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}

    assert {"lower", "casefold"}.isdisjoint(calls)
    assert not any(
        module.startswith(("lofi.deduce", "lofi.nn", "matplotlib", "plotly", "seaborn"))
        for module in imports
    )
    assert not any(
        identifier.casefold().startswith("gold") for identifier in identifiers
    )
