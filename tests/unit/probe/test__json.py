from typing import TYPE_CHECKING

import pytest

from lofi.probe._json import a_count, a_name, a_number, an_object, field

if TYPE_CHECKING:
    from lofi._types import JsonValue

FIELDS: dict[str, JsonValue] = {
    "method": "eap-ig",
    "effect": 0.25,
    "replicates": 999,
    "blank": "",
    "flag": True,
    "nothing": None,
}


def test_an_object_passes_a_json_object_through() -> None:
    assert an_object(FIELDS, "provenance") is FIELDS


def test_an_object_refuses_a_list_and_names_what_was_expected() -> None:
    with pytest.raises(TypeError, match="serialized provenance is not a JSON object"):
        an_object([1, 2], "provenance")


def test_a_missing_field_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="no 'seed' field"):
        field(FIELDS, "seed")


def test_a_field_holding_null_is_present_and_reads_back_as_none() -> None:
    assert field(FIELDS, "nothing") is None


def test_a_name_comes_back_as_a_string() -> None:
    assert a_name(FIELDS, "method") == "eap-ig"


def test_a_name_that_is_empty_is_refused() -> None:
    with pytest.raises(ValueError, match="'blank' field is empty"):
        a_name(FIELDS, "blank")


def test_a_name_that_is_a_number_is_refused_rather_than_stringified() -> None:
    with pytest.raises(TypeError, match="'effect' field is not a string"):
        a_name(FIELDS, "effect")


def test_a_number_comes_back_widened_to_float() -> None:
    read = a_number(FIELDS, "replicates")

    assert read == 999.0
    assert isinstance(read, float)


def test_a_number_that_is_a_boolean_is_refused() -> None:
    with pytest.raises(TypeError, match="'flag' field is not a number"):
        a_number(FIELDS, "flag")


def test_a_count_comes_back_as_an_integer() -> None:
    assert a_count(FIELDS, "replicates") == 999


def test_a_count_that_is_a_float_is_refused() -> None:
    with pytest.raises(TypeError, match="'effect' field is not an integer"):
        a_count(FIELDS, "effect")


def test_a_count_that_is_negative_is_refused() -> None:
    with pytest.raises(ValueError, match="'seed' field is negative"):
        a_count({"seed": -1}, "seed")
