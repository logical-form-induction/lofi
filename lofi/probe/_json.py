"""JSON serialization for probing results."""

from typing import TYPE_CHECKING

from lofi._checks import ensure_name, ensure_non_negative

if TYPE_CHECKING:
    from lofi._types import JsonValue

__all__ = ["a_count", "a_name", "a_number", "an_object", "field"]


def an_object(value: JsonValue, what: str) -> dict[str, JsonValue]:
    """Return a JSON value as an object.

    Parameters
    ----------
    value : JsonValue
        The value to narrow.
    what : str
        What the object was meant to describe, for the message.

    Returns
    -------
    dict[str, JsonValue]
        The value.

    Raises
    ------
    TypeError
        If the value is not a JSON object.
    """
    if not isinstance(value, dict):
        msg = f"a serialized {what} is not a JSON object: {value!r}"
        raise TypeError(msg)

    return value


def field(fields: dict[str, JsonValue], name: str) -> JsonValue:
    """Return one field of a JSON object.

    Parameters
    ----------
    fields : dict[str, JsonValue]
        The object to read.
    name : str
        The field to read from it.

    Returns
    -------
    JsonValue
        The value stored under that name.

    Raises
    ------
    ValueError
        If the object has no such field.
    """
    if name not in fields:
        msg = f"a serialized record has no {name!r} field"
        raise ValueError(msg)

    return fields[name]


def a_name(fields: dict[str, JsonValue], name: str) -> str:
    """Return one field of a JSON object as a non-empty string.

    Parameters
    ----------
    fields : dict[str, JsonValue]
        The object to read.
    name : str
        The field to read from it.

    Returns
    -------
    str
        The string stored under that name.

    Raises
    ------
    TypeError
        If the field does not hold a string.
    ValueError
        If the object omits the field, or the string is empty.
    """
    value = field(fields, name)

    if not isinstance(value, str):
        msg = f"the {name!r} field is not a string: {value!r}"
        raise TypeError(msg)

    return ensure_name(f"the {name!r} field", value)


def a_number(fields: dict[str, JsonValue], name: str) -> float:
    """Return one field of a JSON object as a float.

    Parameters
    ----------
    fields : dict[str, JsonValue]
        The object to read.
    name : str
        The field to read from it.

    Returns
    -------
    float
        The number stored under that name.

    Raises
    ------
    TypeError
        If the field does not hold a number.
    ValueError
        If the object omits the field.
    """
    value = field(fields, name)

    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"the {name!r} field is not a number: {value!r}"
        raise TypeError(msg)

    return float(value)


def a_count(fields: dict[str, JsonValue], name: str) -> int:
    """Return one field of a JSON object as a non-negative integer.

    Parameters
    ----------
    fields : dict[str, JsonValue]
        The object to read.
    name : str
        The field to read from it.

    Returns
    -------
    int
        The integer stored under that name.

    Raises
    ------
    TypeError
        If the field does not hold an integer.
    ValueError
        If the object omits the field, or the integer is negative.
    """
    value = field(fields, name)

    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"the {name!r} field is not an integer: {value!r}"
        raise TypeError(msg)

    return ensure_non_negative(f"the {name!r} field", value)
