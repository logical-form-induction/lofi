import pickle

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from lofi._types import ItemId
from lofi.data._values import (
    ByteSpan,
    Item,
    LicenseRef,
    Licensing,
    Provenance,
    Reproducibility,
    Token,
    TokenSpan,
)

type CorpusValue = (
    ByteSpan
    | Item
    | LicenseRef
    | Licensing
    | Provenance
    | Reproducibility
    | Token
    | TokenSpan
)

MAX_EXAMPLES = 50


@st.composite
def byte_spans(draw: st.DrawFn) -> ByteSpan:
    start = draw(st.integers(min_value=0, max_value=64))
    end = draw(st.integers(min_value=start, max_value=start + 64))

    return ByteSpan(start, end)


@st.composite
def spans(draw: st.DrawFn) -> TokenSpan:
    start = draw(st.integers(min_value=0, max_value=64))
    end = draw(st.integers(min_value=start, max_value=start + 64))

    return TokenSpan(start, end)


@st.composite
def tokens(draw: st.DrawFn) -> Token:
    return Token(
        draw(st.integers(min_value=0, max_value=64)),
        draw(st.text(max_size=6)),
        draw(st.none() | byte_spans()),
    )


def license_refs() -> st.SearchStrategy[LicenseRef]:
    return st.builds(
        LicenseRef,
        st.text(min_size=1, max_size=6),
        st.none() | st.text(max_size=6),
    )


def licensings() -> st.SearchStrategy[Licensing]:
    return st.builds(
        Licensing,
        st.text(min_size=1, max_size=6),
        st.lists(license_refs(), max_size=3).map(tuple),
    )


def reproducibilities() -> st.SearchStrategy[Reproducibility]:
    optional_text = st.none() | st.text(max_size=6)

    return st.builds(
        Reproducibility,
        optional_text,
        optional_text,
        optional_text,
        optional_text,
        st.none() | st.integers(min_value=0, max_value=1024),
    )


def provenances() -> st.SearchStrategy[Provenance]:
    return st.builds(
        Provenance,
        st.text(min_size=1, max_size=6),
        licensings(),
        st.lists(st.text(max_size=6), max_size=3).map(tuple),
        st.none() | reproducibilities(),
    )


@st.composite
def items(draw: st.DrawFn) -> Item:
    surfaces = draw(st.lists(st.text(min_size=1, max_size=6), max_size=5))
    text = " ".join(surfaces)
    drawn: list[Token] = []
    offset = 0

    for index, surface in enumerate(surfaces):
        width = len(surface.encode())
        drawn.append(Token(index, surface, ByteSpan(offset, offset + width)))
        offset += width + 1

    return Item(
        ItemId(draw(st.text(min_size=1, max_size=8))),
        tuple(drawn),
        text=text,
        languages=tuple(
            draw(st.lists(st.sampled_from(("en", "es", "zh")), max_size=2))
        ),
        split=draw(st.none() | st.sampled_from(("train", "dev", "test"))),
        provenance=draw(st.none() | provenances()),
    )


VALUE_NAMES = (
    "ByteSpan",
    "TokenSpan",
    "Token",
    "LicenseRef",
    "Licensing",
    "Reproducibility",
    "Provenance",
    "Item",
)

VALUE_STRATEGIES: tuple[st.SearchStrategy[CorpusValue], ...] = (
    byte_spans(),
    spans(),
    tokens(),
    license_refs(),
    licensings(),
    reproducibilities(),
    provenances(),
    items(),
)

WRITTEN_FIELDS = (
    "start",
    "end",
    "index",
    "identifier",
    "expression",
    "code_uri",
    "source_method",
    "item_id",
)


@pytest.mark.parametrize("strategy", VALUE_STRATEGIES, ids=VALUE_NAMES)
@given(data=st.data())
@settings(max_examples=MAX_EXAMPLES)
def test_a_value_survives_pickling_equal_and_equally_hashed(
    strategy: st.SearchStrategy[CorpusValue],
    data: st.DataObject,
) -> None:
    value = data.draw(strategy)
    restored = pickle.loads(pickle.dumps(value))

    assert restored is not value
    assert restored == value
    assert hash(restored) == hash(value)
    assert {value, restored} == {value}


@pytest.mark.parametrize(
    ("strategy", "field"),
    tuple(zip(VALUE_STRATEGIES, WRITTEN_FIELDS, strict=True)),
    ids=VALUE_NAMES,
)
@given(data=st.data())
@settings(max_examples=MAX_EXAMPLES)
def test_a_value_refuses_an_attribute_write(
    strategy: st.SearchStrategy[CorpusValue],
    field: str,
    data: st.DataObject,
) -> None:
    value = data.draw(strategy)

    with pytest.raises(AttributeError):
        setattr(value, field, None)


def test_a_span_covers_its_positions() -> None:
    assert TokenSpan(0, 3).length == 3
    assert TokenSpan(0, 0).length == 0
    assert TokenSpan(4, 4).length == 0


def test_adjacent_spans_concatenate() -> None:
    assert TokenSpan(0, 2).concat(TokenSpan(2, 5)) == TokenSpan(0, 5)


@given(start=st.integers(0, 32), middle=st.integers(0, 32), width=st.integers(0, 32))
@settings(max_examples=MAX_EXAMPLES)
def test_concatenating_adjacent_spans_adds_their_lengths(
    start: int,
    middle: int,
    width: int,
) -> None:
    left = TokenSpan(start, start + middle)
    right = TokenSpan(start + middle, start + middle + width)

    assert left.concat(right).length == left.length + right.length


def test_non_adjacent_spans_refuse_to_concatenate() -> None:
    with pytest.raises(ValueError, match="not adjacent"):
        TokenSpan(0, 2).concat(TokenSpan(3, 5))

    with pytest.raises(ValueError, match="not adjacent"):
        TokenSpan(0, 4).concat(TokenSpan(3, 5))


@pytest.mark.parametrize("constructor", [TokenSpan, ByteSpan])
def test_a_span_runs_forward_from_a_non_negative_start(
    constructor: type[TokenSpan | ByteSpan],
) -> None:
    with pytest.raises(ValueError, match="starts before"):
        constructor(-1, 3)

    with pytest.raises(ValueError, match="ends before it starts"):
        constructor(3, 1)


@given(item=items())
@settings(max_examples=MAX_EXAMPLES)
def test_a_byte_span_recovers_the_surface_it_was_taken_from(item: Item) -> None:
    text = item.text
    assert text is not None

    for token in item.tokens:
        span = token.span
        assert span is not None
        assert span.substring(text) == token.surface
        assert span.length == len(token.surface.encode())


def test_a_byte_span_refuses_to_read_past_the_text() -> None:
    with pytest.raises(ValueError, match="runs past the text"):
        ByteSpan(0, 9).substring("short")


def test_a_byte_span_counts_bytes_and_not_characters() -> None:
    text = "el niño duerme"
    span = ByteSpan(3, 8)

    assert span.substring(text) == "niño"
    assert span.length == 5


def test_a_token_position_is_non_negative() -> None:
    with pytest.raises(ValueError, match="position is negative"):
        Token(-1, "cat")


def test_an_item_refuses_a_gap_in_its_token_positions() -> None:
    with pytest.raises(ValueError, match="whose index is 2"):
        Item(ItemId("a"), (Token(0, "the"), Token(2, "cat")))


def test_an_item_refuses_a_byte_span_past_its_text() -> None:
    with pytest.raises(ValueError, match="past the text's"):
        Item(ItemId("a"), (Token(0, "the", ByteSpan(0, 40)),), text="the cat")


def test_an_item_refuses_an_empty_id() -> None:
    with pytest.raises(ValueError, match="has no id"):
        Item(ItemId(""), ())


def test_the_same_sentence_twice_is_two_items() -> None:
    tokens_of = (Token(0, "boats"), Token(1, "boats"), Token(2, "boats"))
    first = Item(ItemId("judgment:1"), tokens_of)
    second = Item(ItemId("judgment:2"), tokens_of)

    assert first != second
    assert len({first, second}) == 2

    ratings = {first.item_id: 3.0, second.item_id: 6.0}

    assert ratings[second.item_id] == 6.0


def test_provenance_stays_outside_an_item_identity() -> None:
    tokens_of = (Token(0, "the"), Token(1, "cat"))
    gold = Provenance("manual-native", Licensing("CC-BY-4.0"))
    silver = Provenance("automatic", Licensing("CC-BY-4.0"))
    annotated = Item(ItemId("x"), tokens_of, provenance=gold)
    induced = Item(ItemId("x"), tokens_of, provenance=silver)

    assert annotated == induced
    assert hash(annotated) == hash(induced)
    assert annotated.provenance != induced.provenance


def test_provenance_survives_a_pickling_that_equality_would_not_notice() -> None:
    provenance = Provenance(
        "converted",
        Licensing("CC-BY-4.0", (LicenseRef("MIT", "annotation"),)),
        ("arXiv:2401.00001",),
        Reproducibility("https://example.invalid/code", "abc123", "lofi run", "uv", 7),
    )
    item = Item(ItemId("x"), (Token(0, "the"),), provenance=provenance)
    restored = pickle.loads(pickle.dumps(item))

    assert restored.provenance == provenance


def test_a_token_span_is_not_a_byte_span() -> None:
    assert TokenSpan(0, 3) != ByteSpan(0, 3)
    assert ByteSpan(0, 3) != TokenSpan(0, 3)
    assert len({TokenSpan(0, 3), ByteSpan(0, 3)}) == 2


def test_a_value_is_not_equal_to_an_unrelated_object() -> None:
    assert TokenSpan(0, 3) != (0, 3)
    assert Token(0, "cat") != "cat"


def test_an_unlicensed_release_is_not_constructible() -> None:
    with pytest.raises(ValueError, match="no SPDX expression"):
        Licensing("")

    with pytest.raises(ValueError, match="no identifier"):
        LicenseRef("")


def test_provenance_names_a_source_method() -> None:
    with pytest.raises(ValueError, match="no source method"):
        Provenance("", Licensing("CC-BY-4.0"))
