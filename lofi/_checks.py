"""Validates shared values and report law violations."""

from __future__ import annotations

import os
import re
from math import isfinite
from typing import Final, Never, NoReturn

__all__ = [
    "CHECKS",
    "LawViolationError",
    "NumericalFailureError",
    "article",
    "ensure_commit",
    "ensure_finite",
    "ensure_fraction",
    "ensure_name",
    "ensure_non_negative",
    "ensure_positive",
    "fail_law",
    "refuse_reassignment",
]


CHECKS: Final[bool] = os.environ.get("LOFI_CHECK", "0") == "1"
"""Indicates whether coarse-grained invariant checks run."""

_COMMIT: Final = re.compile(r"[0-9a-f]{40}")
"""A full lowercase hexadecimal commit hash."""


class LawViolationError(Exception):
    """Reports a failed algebraic law."""


class NumericalFailureError(FloatingPointError):
    """Reports a training or validation quantity that is not finite.

    Parameters
    ----------
    quantity : str
        What was not finite, as the step or metric that reported it names it.
    value : float | None
        The value reported, or None where only its finiteness was reported.

    Attributes
    ----------
    quantity : str
        What was not finite.
    value : float | None
        The value reported, or None where only its finiteness was reported.
    """

    def __init__(self, quantity: str, value: float | None) -> None:
        self.quantity = quantity
        self.value = value
        reported = "" if value is None else f": {value!r}"
        super().__init__(f"{quantity} is not finite{reported}")


def ensure_finite(field: str, value: float) -> float:
    """Return a finite float.

    Parameters
    ----------
    field : str
        Field name used in error messages.
    value : float
        Number to validate. Integers are accepted, but booleans are not.

    Returns
    -------
    float
        The validated value converted to ``float``.

    Raises
    ------
    TypeError
        If ``value`` is not numeric or is a boolean.
    ValueError
        If ``value`` is infinite or NaN.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"{field} is not a number: {value!r}"
        raise TypeError(msg)

    number = float(value)

    if not isfinite(number):
        msg = f"{field} is not finite: {value!r}"
        raise ValueError(msg)

    return number


def ensure_fraction(field: str, value: float) -> float:
    """Return a finite float in the closed unit interval.

    Parameters
    ----------
    field : str
        Field name used in error messages.
    value : float
        Number to validate. Integers are accepted, but booleans are not.

    Returns
    -------
    float
        The validated value converted to ``float``.

    Raises
    ------
    TypeError
        If ``value`` is not numeric or is a boolean.
    ValueError
        If ``value`` is not finite or lies outside $[0, 1]$.
    """
    share = ensure_finite(field, value)

    if not 0.0 <= share <= 1.0:
        msg = f"{field} is outside [0, 1]: {value!r}"
        raise ValueError(msg)

    return share


def ensure_non_negative(field: str, value: int) -> int:
    """Return a nonnegative integer.

    Parameters
    ----------
    field : str
        Field name used in error messages.
    value : int
        Integer to validate. Booleans are not accepted.

    Returns
    -------
    int
        The validated value.

    Raises
    ------
    TypeError
        If ``value`` is not an integer or is a boolean.
    ValueError
        If ``value`` is negative.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{field} is not an integer: {value!r}"
        raise TypeError(msg)

    if value < 0:
        msg = f"{field} is negative: {value!r}"
        raise ValueError(msg)

    return value


def ensure_name(field: str, value: str) -> str:
    """Return a nonempty name.

    Parameters
    ----------
    field : str
        Field name used in error messages.
    value : str
        String to validate.

    Returns
    -------
    str
        The validated string.

    Raises
    ------
    TypeError
        If ``value`` is not a string.
    ValueError
        If ``value`` is empty.
    """
    if not isinstance(value, str):
        msg = f"{field} is not a string: {value!r}"
        raise TypeError(msg)

    if not value:
        msg = f"{field} is empty"
        raise ValueError(msg)

    return value


def ensure_commit(field: str, value: str) -> str:
    """Return a full lowercase hexadecimal commit hash.

    Parameters
    ----------
    field : str
        Field name used in error messages.
    value : str
        String to validate.

    Returns
    -------
    str
        The validated commit hash.

    Raises
    ------
    TypeError
        If ``value`` is not a string.
    ValueError
        If ``value`` is not forty lowercase hexadecimal digits.
    """
    if _COMMIT.fullmatch(ensure_name(field, value)) is None:
        msg = f"{field} is a full lowercase commit hash, not {value!r}"
        raise ValueError(msg)

    return value


def ensure_positive(field: str, value: int) -> int:
    """Return a positive integer.

    Parameters
    ----------
    field : str
        The field name used in error messages.
    value : int
        The value to validate.

    Returns
    -------
    int
        The value, unchanged.

    Raises
    ------
    ValueError
        If the value is zero or negative.
    """
    held = ensure_non_negative(field, value)

    if held == 0:
        msg = f"{field} is not positive: {value!r}"
        raise ValueError(msg)

    return held


def fail_law(detail: str, *, law: str) -> NoReturn:
    """Raise a law violation, naming the law and what went wrong.

    Parameters
    ----------
    detail : str
        What the law found.
    law : str
        The law's name.

    Raises
    ------
    LawViolationError
        Always.
    """
    message = f"{law}: {detail}"

    raise LawViolationError(message)


def refuse_reassignment(what: str) -> Never:
    """Raise the error every frozen value reports on field reassignment.

    Parameters
    ----------
    what : str
        The value kind, for the message.

    Raises
    ------
    AttributeError
        Always.
    """
    msg = f"{article(what)} {what} is immutable"

    raise AttributeError(msg)


def article(noun: str) -> str:
    """Return the indefinite article a noun takes.

    Parameters
    ----------
    noun : str
        The noun the article precedes.

    Returns
    -------
    str
        ``"an"`` before a vowel letter and ``"a"`` otherwise.
    """
    return "an" if noun[:1].lower() in "aeiou" else "a"
