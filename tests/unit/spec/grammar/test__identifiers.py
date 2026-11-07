import keyword

import pytest
from hypothesis import given
from hypothesis import strategies as st

from lofi.spec import IdentifierError
from lofi.spec.grammar._identifiers import decode_identifier, encode_identifier

SPELLINGS = (
    ("S", "S"),
    ("S[dcl]", "S_5b_dcl_5d_"),
    ("/", "_2f_"),
    ("\\", "_5c_"),
    ("->", "_2d__3e_"),
    ("_", "_5f_"),
    ("a_b", "a_5f_b"),
    ("for", "_66_or"),
    ("1S", "_31_S"),
    ("λx", "_3bb_x"),
)


@pytest.mark.parametrize(("name", "identifier"), SPELLINGS)
def test_a_name_encodes_to_its_documented_spelling(name: str, identifier: str) -> None:
    assert encode_identifier(name) == identifier
    assert decode_identifier(identifier) == name


@given(st.text(min_size=1))
def test_decoding_undoes_encoding_on_any_nonempty_name(name: str) -> None:
    identifier = encode_identifier(name)

    assert identifier.isidentifier()
    assert not keyword.iskeyword(identifier)
    assert decode_identifier(identifier) == name


@given(st.text(min_size=1), st.text(min_size=1))
def test_distinct_names_encode_to_distinct_identifiers(left: str, right: str) -> None:
    assert (encode_identifier(left) == encode_identifier(right)) == (left == right)


@pytest.mark.parametrize("word", sorted(keyword.kwlist))
def test_a_keyword_name_is_escaped_rather_than_spelled(word: str) -> None:
    identifier = encode_identifier(word)

    assert identifier != word
    assert identifier.startswith("_")
    assert decode_identifier(identifier) == word


@pytest.mark.parametrize(
    "identifier",
    [
        "_53_",
        "_05b_",
        "_5B_",
        "_+5b_",
        "_5_b_",
        "S_",
        "_",
        "for",
        "1S",
        "_110000_",
        "_x_",
    ],
)
def test_a_spelling_encoding_produces_is_the_only_one_decoding_accepts(
    identifier: str,
) -> None:
    with pytest.raises(IdentifierError):
        decode_identifier(identifier)


def test_the_empty_name_has_no_identifier_and_the_empty_identifier_no_name() -> None:
    with pytest.raises(IdentifierError, match="empty"):
        encode_identifier("")

    with pytest.raises(IdentifierError):
        decode_identifier("")
